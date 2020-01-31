using Google.Protobuf;
using Onnx;
using UnityEngine;
using UnityEngine.Profiling;
using UnityEditor;
using UnityEditor.Experimental.AssetImporters;
using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

[assembly: InternalsVisibleToAttribute("Barracuda.EditorTests")]

namespace Barracuda
{
    /// <summary>
    /// Asset Importer for Open Neural Network Exchange (ONNX) files.
    /// For more information about ONNX file format see: https://github.com/onnx/onnx
    /// </summary>
    [ScriptedImporter(3, new[] { "onnx" })]
    public class ONNXModelImporter : ScriptedImporter
    {
        // Configuration
        public bool patchReshapeToSupportBatchSize = true;
        public bool patchRemoveTrailingTFExportCharacters = false;
        private const string iconName = "ONNXModelIcon";

        private readonly Dictionary<string, ONNXTensor> m_OverrideGlobalInputs = new Dictionary<string, ONNXTensor>()
        {
            { "sequence_length:0", new ONNXTensor(new Tensor(1, 1, new[] { 1f }), new [] { 1L }) },
            { "sequence_length",   new ONNXTensor(new Tensor(1, 1, new[] { 1f }), new [] { 1L }) }
        };
        private readonly HashSet<string> m_ShouldNotBeBaked = new HashSet<string>()
        {
            // the following nodes handle constant inputs in a custom manner and should not be baked:
            "Constant", "Reshape", "Shape", "Slice", "Gather", "Transpose", "Squeeze", "Unsqueeze",

            // the following nodes are dynamic in nature and can not be baked even when all inputs are constant:
            "RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike"
        };

        // Shortcuts
        private Dictionary<string, ONNXTensor> constantTensors { get { return m_ModelTensors.constants; } }
        private Dictionary<string, VariableTensor> variableTensors { get { return m_ModelTensors.variables; } }
        private void Add(string opType, Action<ModelBuilder, ONNXNodeWrapper> opImportAction)
        {
            m_NodeImporters.Add(opType, opImportAction);
        }

        // ONNX parser declaration
        // Add operator handlers here
        public ONNXModelImporter()
        {
            // TODO: setup m_NodeImporters via initializer list
            // TODO: simplify code to avoid passing node.Name over and over again
            Add("Constant", (net, node) => {
                node.UnsupportedAttribute("sparse_value");
                Const(node, node.ValueAsTensor);
            });
            Add("Reshape", (net, node)  => {
                long[] onnxShape;
                if (node.InputCount > 1) // Reshape-5
                    onnxShape = node.Input1Constant(onnxLayout:"C", name:"shape").AsLongs();
                else // Reshape-1
                    onnxShape = node.Shape;

                if (node.IsInput0Const)
                { // reshape constant source tensor and store it as the new constant
                    var reshapedTensor = constantTensors[node.Input0].Reshape(onnxShape);
                    Const(node, reshapedTensor);
                }
                else
                {
                    var symbolicShape = ONNXLayout.ConvertSymbolicShapeToBarracuda(onnxShape, "NCHW");
                    if (patchReshapeToSupportBatchSize)
                        symbolicShape[0] = 0; // force keep batch size
                    net.Reshape(node.Name, node.Input0, symbolicShape);
                    Output(node, rank:symbolicShape.Length);
                }
            });
            Add("Shape", (net, node)    => {
                // @TODO: dynamic implementation that would return real shape during execution of the model
                if (!node.IsInput0Const)
                    throw new OnnxLayerImportException(
                        $"Currently only constant inputs for node of type {node.OperatorType} are supported. Instead input of {node.Name} is pointing to non-constant node {node.Input0}.");

                var shapeTensor = new ONNXTensor(
                    data:new Tensor(4, 1, new[] { -1f, (float)node.Input0Rank, -1f, -1f }),
                    onnxShape:new [] { 4L });
                Const(node, shapeTensor);
                Output(node, rank:1);
            });
            Add("Unsqueeze", (net, node) => {
                if (node.IsInput0Const)
                {
                    var unsqueezed = constantTensors[node.Input0].Unsqueeze(node.Axes);
                    Const(node, unsqueezed);
                }
                else
                {
                    // NOTE: axis=0 or 1 will require Transpose between channels and other spatial dimensions when converted to Barracuda layout.
                    // As we have different layouts between ONNX and Barracuda, Unsqueeze might require actual Transpose not just Reshape!

                    // ONNX pseudocode here:
                    // a = Tensor [2, 10]             # NC   -> barracuda N11C
                    // b = Unsqueeze(a, axis=0)
                    // # b is now Tensor [1, 2, 10]   # NCHW -> barrada NHWC
                    // Because ONNX is NCHW, but generally hell knows what goes where and Barracuda is strict NHWC. We end up with:
                    // `a` would be [2, 1, 1, 10], but `b` would have to be [1, 10, 1, 2]. Note the actual data swap in channels!

                    bool mightNeedTranspose = node.Axes.Any(axis => axis <= 1);
                    if (mightNeedTranspose)
                        Warn(net, node, $"Unsqeeze on axis next to batch dimension for non-constant tensors might lead to unexpected results.");

                    net.Identity(node.Name, node.Input0);
                }
            });
            Add("Squeeze", (net, node) => {
                if (node.IsInput0Const)
                {
                    var squeezed = constantTensors[node.Input0].Squeeze(node.Axes);
                    Const(node, squeezed);
                }
                else
                {
                    // See Unsqueeze above for explanation
                    bool mightNeedTranspose = node.Axes.Any(axis => axis <= 1);
                    if (mightNeedTranspose)
                        Warn(net, node, $"Sqeeze on any axis next to batch dimension for non-constant tensors might lead to unexpected results.");

                    net.Identity(node.Name, node.Input0);
                }
            });
            Add("Flatten", (net, node)  => {
                node.UnsupportedAttribute("axis", 1);
                net.Flatten(node.Name, node.Input0);
                Output(node, rank:2);
            });
            Add("Concat", (net, node) => {
                int axis = node.AxisOptional(0);

                // TODO: write dedicated ONNXTensor.Concat() so that output shape is exact to ONNX
                // if (node.AreAllInputsConst) Const(node, ONNXTensor.Concat(node.Inputs.Select(i => constantTensors[i]), axis));

                axis = ONNXLayout.ConvertAxisToBarracuda(axis, onnxRank:node.Input0Rank, onnxLayout:"NCHW");
                net.Concat(node.Name, node.Inputs, axis);

                bool lastAxis = (axis == -1 || axis == node.Input0Rank - 1); // last axis in Barracuda is feature axis
                if (lastAxis)
                {
                    var featuresConcatenated = node.Inputs.Sum(i => variableTensors[i].features);
                    Output(node, features:featuresConcatenated);
                }
            });
            Add("Slice", (net, node) => {
                int[] starts, ends, axes, steps;
                if (node.InputCount > 1) // Slice-10
                {
                    var constStarts      = node.Input1Constant(onnxLayout:"C", name:"starts");
                    var constEnds        = node.Input2Constant(onnxLayout:"C", name:"ends");
                    var defaultAxes = new Tensor(constStarts.shape, Enumerable.Range(0, constStarts.length).Select(v => (float)v).ToArray());
                    var constAxes        = node.Input3ConstantOptional(defaultAxes, onnxLayout:"C", name:"axes");
                    var constSteps       = node.Input4ConstantOptional(constStarts.shape, 1.0f, onnxLayout:"C", name:"steps");

                    starts  = constStarts.AsInts();
                    ends    = constEnds.AsInts();
                    axes    = constAxes.AsInts();
                    steps   = constSteps.AsInts();
                }
                else // Slice-1
                {
                    starts      = node.Starts;
                    ends        = node.Ends;
                    axes        = node.AxesOptional(Enumerable.Range(0, starts.Length).ToArray());
                    steps       = Enumerable.Repeat(1, starts.Length).ToArray();
                }

                Debug.Assert(starts.Length == ends.Length);
                var onnxRank    = node.Input0Rank;
                var onnxLast    = (long)int.MaxValue;
                var onnxStarts  = Enumerable.Repeat(0L, onnxRank).ToArray();
                var onnxEnds    = Enumerable.Repeat(onnxLast, onnxRank).ToArray(); // by default copy the whole axis till the end
                var onnxSteps   = Enumerable.Repeat(1L, onnxRank).ToArray();

                // NOTE: begin=0, end=0, stride=1  <=  full range from existing axis
                //       begin=0, end=inf,stride=1 <=  full range from existing axis
                //       begin=0, end=X, stride=1  <=  full range from existing axis, if X==last element on this axis
                //       begin=0, end=0, stride=0  <=  new axis OR shrink axis to single 1st element
                //       begin=N, end=N, stride=0  <=              shrink axis to single Nth element
                // These notes are copied from TensorExtensions.ApplyStridedSlice(...)

                for (int i = 0; i < axes.Length; ++i)
                {
                    var axis = axes[i];
                    if (axis < 0)
                        axis += onnxRank;
                    axis = Math.Min(Math.Max(axis, 0), onnxRank);
                    onnxStarts[axis] = starts[i];
                    onnxEnds[axis]   = ends[i];
                    onnxSteps[axis]  = steps[i];
                }

                if (node.IsInput0Const)
                {
                    var slicedTensor = constantTensors[node.Input0].Slice(
                        starts:onnxStarts.Select(x => (int)x).ToArray(),
                        ends:onnxEnds.Select(x => (int)x).ToArray(),
                        steps:onnxSteps.Select(x => (int)x).ToArray());
                    Const(node, slicedTensor);
                }
                else
                {
                    net.StridedSlice(node.Name, node.Input0,
                        starts:ONNXLayout.ConvertSymbolicShapeToBarracuda(onnxStarts, onnxLayout:"NCHW"),
                        ends:ONNXLayout.ConvertSymbolicShapeToBarracuda(onnxEnds, onnxLayout:"NCHW"),
                        strides:ONNXLayout.ConvertSymbolicShapeToBarracuda(onnxSteps, onnxLayout:"NCHW"));
                }
            });
            Add("Gather", (net, node) =>
            {
                int axis = node.AxisOptional(0);
                if (node.IsInput0Const)
                {
                    var indices = node.Input1Constant(onnxLayout:"C", name:"indices").AsInts();
                    var gatheredTensor = constantTensors[node.Input0].Gather(axis, indices);
                    Const(node, gatheredTensor);
                }
                else
                {
                    axis = ONNXLayout.ConvertAxisToBarracuda(axis, onnxRank:node.Input0Rank, onnxLayout:"NCHW");
                    net.Gather(node.Name, node.Input0, node.Input1, axis);
                }
            });
            Add("OneHot", (net, node) => {
                node.UnsupportedAttribute("axis", -1);

                var defaultOffOn = new Tensor(1,1,1,2, new float[] {0, 1});

                var depth = (int)node.Input1Constant(onnxLayout:"C", name:"depth")[0];
                var offon = node.Input2ConstantOptional(defaultOffOn, onnxLayout:"C", name:"values");
                net.OneHot(node.Name, node.Input0, depth, (int)offon[1], (int)offon[0]);
            });

            // Activation ops
            Add("Relu", (net, node)     => { net.Relu(node.Name, node.Input0); });
            Add("Softmax", (net, node)  => { net.Softmax(node.Name, node.Input0); node.UnsupportedAttribute("axis", 1); });
            Add("Tanh", (net, node)     => { net.Tanh(node.Name, node.Input0); });
            Add("Sqrt", (net, node)     => { net.Sqrt(node.Name, node.Input0); });
            Add("Sigmoid", (net, node)  => { net.Sigmoid(node.Name, node.Input0); });
            Add("Elu", (net, node)      => { net.Elu(node.Name, node.Input0, node.AlphaOptional(1f)); });
            Add("LeakyRelu",(net, node) => { net.LeakyRelu(node.Name, node.Input0, node.AlphaOptional(0.01f)); });
            Add("Selu", (net, node)     => { net.Selu(node.Name, node.Input0, node.AlphaOptional(1.67326f), node.GammaOptional(1.0507f)); });
            Add("Swish", (net, node)    => { net.Swish(node.Name, node.Input0); });
            Add("PRelu", (net, node)    => { net.PRelu(node.Name, node.Input0, node.Input1); });
            Add("LogSoftmax", (net, node)   => { net.LogSoftmax(node.Name, node.Input0); node.UnsupportedAttribute("axis", 1); });
            // TODO: Add("Hardmax", (net, node)      => { net.Hardmax(node.Name, node.Input0); node.UnsupportedAttribute("axis", 1); });
            // TODO: Add("Softplus", (net, node)     => { net.Softplus(node.Name, node.Input0); });
            // TODO: Add("Softsign", (net, node)     => { net.Softsign(node.Name, node.Input0); });
            // TODO: Add("HardSigmoid", (net, node)  => { net.HardSigmoid(node.Name, node.Input0, node.AlphaOptional(0.2f), node.BetaOptional(0.5f)); });
            Add("Exp", (net, node)      => { net.Exp(node.Name, node.Input0); });
            Add("Log", (net, node)      => { net.Log(node.Name, node.Input0); });
            Add("Reciprocal", (net, node) => { net.Reciprocal(node.Name, node.Input0); });
            Add("Abs", (net, node)      => { net.Abs(node.Name, node.Input0); });
            Add("Neg", (net, node)      => { net.Neg(node.Name, node.Input0); });
            Add("Ceil", (net, node)     => { net.Ceil(node.Name, node.Input0); });
            Add("Floor", (net, node)    => { net.Floor(node.Name, node.Input0); });
            Add("Round", (net, node)    => { net.Round(node.Name, node.Input0); });
            Add("Clip", (net, node)     => { net.Clip(node.Name, node.Input0, node.MinOptional(float.MinValue),  node.MaxOptional(float.MaxValue)); });

            // Broadcast ops
            Add("Add", (net, node)     => { net.Add(node.Name, node.Inputs); });
            Add("Sum", (net, node)     => { net.Add(node.Name, node.Inputs); }); // Sum is implemented via Add
            Add("Sub", (net, node)     => { net.Sub(node.Name, node.Inputs); });
            Add("Mul", (net, node)     => { net.Mul(node.Name, node.Inputs); });
            Add("Div", (net, node)     => { net.Div(node.Name, node.Inputs); });
            Add("Pow", (net, node)     => { net.Pow(node.Name, node.Inputs); });
            Add("Min", (net, node)     => { net.Min(node.Name, node.Inputs); });
            Add("Max", (net, node)     => { net.Max(node.Name, node.Inputs); });
            Add("Mean", (net, node)    => { net.Mean(node.Name, node.Inputs); });

            // Logical ops
            Add("Greater", (net, node) => { net.Greater(node.Name, node.Input0, node.Input1); });
            Add("Less", (net, node)    => { net.Less(node.Name, node.Input0, node.Input1); });
            Add("Equal", (net, node)   => { net.Equal(node.Name, node.Input0, node.Input1); });
            Add("Or", (net, node)      => { net.LogicalOr(node.Name, node.Input0, node.Input1); });
            Add("And", (net, node)     => { net.LogicalAnd(node.Name, node.Input0, node.Input1); });
            Add("Not", (net, node)     => { net.LogicalNot(node.Name, node.Input0); });
            Add("Xor", (net, node)     => { net.LogicalXor(node.Name, node.Input0, node.Input1); });

            // Padding ops
            Add("Pad", (net, node) =>
            {
                var mode = node.GetOptionalString("mode", "constant");
                switch (mode)
                {
                    case "constant": net.Border2D(node.Name, node.Input0, node.Pads, node.GetOptionalFloat("value", 0.0f)); break;
                    case "reflect": net.Pad2DReflect(node.Name, node.Input0, node.Pads); break;
                    case "edge": net.Pad2DEdge(node.Name, node.Input0, node.Pads); break;
                }
            });

            // Pooling ops
            Add("AveragePool", (net, node) => {
                node.UnsupportedAttribute("ceil_mode", 0);
                node.UnsupportedAttribute("count_include_pad", 0);
                net.AvgPool2D(node.Name, node.Input0, node.KernelShape, node.Strides, node.Pads);
            });
            Add("MaxPool", (net, node) => {
                node.UnsupportedAttribute("ceil_mode", 0);
                node.UnsupportedAttribute("dilations", new[] {1, 1});
                node.UnsupportedAttribute("storage_order", 0);
                net.MaxPool2D(node.Name, node.Input0, node.KernelShape, node.Strides, node.Pads);
            });
            Add("GlobalAveragePool", (net, node) => { net.GlobalAvgPool2D(node.Name, node.Input0); });
            Add("GlobalMaxPool",     (net, node) => { net.GlobalMaxPool2D(node.Name, node.Input0); });
            Add("Upsample", (net, node) => {
                node.UnsupportedAttribute("mode", "nearest");

                float[] scales;
                if (node.InputCount > 1)
                    scales = node.Input1Constant(onnxLayout:"C", name:"scales").AsFloats();
                else
                    scales = node.Scales;

                if (scales.Length < 2 || scales.Length > 5)
                    throw new OnnxLayerImportException(
                        $"Input scales of unsupported length {scales.Length} in {node.Name} ot fype {node.OperatorType}.");

                if ((scales[0] != 1) || (scales[1] != 1))
                    Warn(net, node, $"Unsupported scaling, only H and W scaling are supported. Value will be ignored and defaulted to 1.");

                // skip NC from onnx NCHW layout
                scales = scales.Skip(2).ToArray();

                if (scales.Length == 1) // append default H, if 1D
                    scales = new[] {1f, scales[0]};

                Resize2D(net, node, scales);
            });
            Add("Resize", (net, node) => {
                node.UnsupportedAttribute("mode", "nearest");

                float[] scales;
                if (node.InputCount > 2) // Resize-11
                {
                    node.UnsupportedAttribute("coordinate_transformation_mode", "half_pixel");
                    node.UnsupportedAttribute("cubic_coeff_a", -0.75f);
                    node.UnsupportedAttribute("exclude_outside", 0);
                    node.UnsupportedAttribute("extrapolation_value", 0f);
                    node.UnsupportedAttribute("nearest_mode", "round_prefer_floor");

                    // Inputs (3 - 4)
                    // X : T1
                    // roi : T2, It only takes effect when coordinate_transformation_mode is "tf_crop_and_resize"
                    // scales : tensor(float)
                    // sizes (optional) : tensor(int64)

                    // TODO: cropping via roi input
                    // TODO: support sizes
                    scales = node.Input2Constant(onnxLayout:"C", name:"scales").AsFloats();
                }
                else // Resize-10
                {
                    scales = node.Input1Constant(onnxLayout:"C", name:"scales").AsFloats();
                }

                if (scales.Length < 2 || scales.Length > 5)
                    throw new OnnxLayerImportException(
                        $"Input scales of unsupported length {scales.Length} in {node.Name} ot fype {node.OperatorType}.");

                if ((scales[0] != 1) || (scales[1] != 1))
                    Warn(net, node, $"Unsupported scaling, only H and W scaling are supported. Value will be ignored and defaulted to 1.");

                // skip NC from onnx NCHW layout
                scales = scales.Skip(2).ToArray();

                Resize2D(net, node, scales);
            });
            Add("Transpose", (net, node) =>
            {
                if (!node.IsInput0Const)
                    throw new OnnxLayerImportException(
                        $"Currently only constant inputs for node of type {node.OperatorType} are supported. Instead input of {node.Name} is pointing to non-constant node {node.Input0}.");

                var permutations = node.GetOptionalIntArray("perm", new []{3, 2, 1, 0});
                var transposedTensor = constantTensors[node.Input0].Permute(permutations);
                Const(node, transposedTensor);
            });

            // Tensor ops
            Add("Gemm", (net, node)     => {
                node.UnsupportedAttribute("alpha", 1.0f);
                node.UnsupportedAttribute("beta", 1.0f);
                node.UnsupportedAttribute("transA", 0);
                var onnxLayout = node.TransBOptional() ? "KC" : "CK";
                var weights = node.Input1Constant(onnxLayout, name:"B");
                var biases  = node.Input2ConstantOptional(Bias(weights.shape), 0.0f, onnxLayout:"C", name:"C");
                // Change data layout from "channels first" to "channels last"
                weights = SwapSpatialDimensionsAndFeaturesInMatMulWeights(weights, node.Input0Features);
                net.Dense(node.Name, node.Input0, weights, biases);
                Output(node, features:weights.channels, rank:2); // Gemm forces flatten of the input to rank 2
            });
            Add("MatMul", (net, node)   => {
                var weights = node.Input1Constant(onnxLayout:"CK", name:"B");
                var biases  = node.DefaultTensor(Bias(weights.shape), 0.0f);
                // Change data layout from "channels first" to "channels last"
                weights = SwapSpatialDimensionsAndFeaturesInMatMulWeights(weights, node.Input0Features);
                net.Dense(node.Name, node.Input0, weights, biases);
                Output(node, features:weights.channels, rank:2); // MatMul forces flatten of the input to rank 2
            });
            Add("Conv", (net, node)     => {
                node.UnsupportedAttribute("dilations", new[] {1, 1});
                node.IgnoredAttribute("kernel_shape", "Kernel shape is derived from K tensor weights instead");
                var kernels = node.Input1Constant(onnxLayout:"KCHW", name:"W");
                var biases  = node.Input2ConstantOptional(Bias(kernels.shape), 0.0f, onnxLayout:"C", name:"B");

                if (node.GroupOptional() > 1)
                    net.DepthwiseConv2D(node.Name, node.Input0, node.Strides, node.Pads, kernels, biases);
                else
                    net.Conv2D(node.Name, node.Input0, node.Strides, node.Pads, kernels, biases);
                Output(node, features:kernels.channels);
            });
            Add("ConvTranspose", (net, node)     => {
                node.UnsupportedAttribute("dilations", new[] {1, 1});
                node.UnsupportedAttribute("group", 1);
                node.UnsupportedAttribute("output_shape", new int[0]);
                node.IgnoredAttribute("kernel_shape", "Kernel shape is derived from K tensor weights instead");
                var kernels = node.Input1Constant(onnxLayout:"CKHW", name:"W");
                var biases  = node.Input2ConstantOptional(Bias(kernels.shape), 0.0f, onnxLayout:"C", name:"B");
                net.Conv2DTrans(node.Name, node.Input0, node.Strides, node.Pads, node.OutputPadding, kernels, biases);
                Output(node, features:kernels.channels);
            });
            Add("BatchNormalization", (net, node) => {
                var variance  = node.Input4Constant(onnxLayout:"C", name:"var");
                var scale     = node.Input1ConstantOptional(variance.shape, 1.0f, onnxLayout:"C", name:"scale");
                var bias      = node.Input2ConstantOptional(variance.shape, 0.0f, onnxLayout:"C", name:"B");
                var mean      = node.Input3ConstantOptional(variance.shape, 0.0f, onnxLayout:"C", name:"mean");
                var fusedData = FuseBatchNormWeights(scale, bias, mean, variance, node.EpsilonOptional());
                net.ScaleBias(node.Name, node.Input0, fusedData.Item1, fusedData.Item2);
            });
            Add("InstanceNormalization", (net, node) => {
                var scale     = node.Input1Constant(onnxLayout:"C", name:"scale");
                var bias      = node.Input2ConstantOptional(scale.shape, 0.0f, onnxLayout:"C", name:"B");
                net.Normalization(node.Name, node.Input0, scale, bias, node.EpsilonOptional());
            });
            // random ops
            Add("RandomNormal", (net, node) => {
                float mean     = node.GetOptionalFloat("mean",  1.0f);
                float scale    = node.GetOptionalFloat("scale", 0.0f);
                float seed     = node.GetOptionalFloat("seed",  0.0f);
                int[] shape    = node.GetRequiredIntArray("shape");
                net.RandomNormal(node.Name, mean, scale, seed, shape);
            });
            Add("RandomNormalLike", (net, node) => {
                float mean     = node.GetOptionalFloat("mean",  1.0f);
                float scale    = node.GetOptionalFloat("scale", 0.0f);
                float seed     = node.GetOptionalFloat("seed",  0.0f);
                net.RandomNormal(node.Name, mean, scale, seed, node.Input0);
            });
            Add("RandomUniform", (net, node) => {
                float high     = node.GetOptionalFloat("high",  1.0f);
                float low      = node.GetOptionalFloat("low",   0.0f);
                float seed     = node.GetOptionalFloat("seed",  0.0f);
                int[] shape    = node.GetRequiredIntArray("shape");
                net.RandomUniform(node.Name, low, high, seed, shape);
            });
            Add("RandomUniformLike", (net, node) => {
                float high     = node.GetOptionalFloat("high",  1.0f);
                float low      = node.GetOptionalFloat("low",   0.0f);
                float seed     = node.GetOptionalFloat("seed",  0.0f);
                net.RandomUniform(node.Name, low, high, seed, node.Input0);
            });
            Add("ReduceMax", (net, node)  => {
                Reduce(net, node, Layer.Type.ReduceMax);
            });
            Add("ReduceMean", (net, node) => {
                Reduce(net, node, Layer.Type.ReduceMean);
            });
            Add("ReduceMin", (net, node)  => {
                Reduce(net, node, Layer.Type.ReduceMin);
            });
            Add("ReduceProd", (net, node) => {
                Reduce(net, node, Layer.Type.ReduceProd);
            });
            Add("ReduceSum", (net, node)  => {
                Reduce(net, node, Layer.Type.ReduceSum);
            });


            // Ignore, noop during inference
            Add("Identity", (net, node)     => { net.Identity(node.Name, node.Input0); });
            Add("Cast", (net, node)         => { net.Identity(node.Name, node.Input0); });
            Add("Dropout", (net, node)      => { net.Identity(node.Name, node.Input0); });
        }

        // Fuse training time BatchNorm tensors into Scale & Bias
        internal static Tuple<Tensor, Tensor> FuseBatchNormWeights(Tensor gamma, Tensor beta, Tensor mean, Tensor variance, float epsilon)
        {
            // https://github.com/Tencent/ncnn/blob/master/src/layer/batchnorm.cpp
            // float sqrt_var = sqrt(var_data[i]);
            // a_data[i] = bias_data[i] - slope_data[i] * mean_data[i] / sqrt_var;
            // b_data[i] = slope_data[i] / sqrt_var;
            // ...
            // ptr[i] = b * ptr[i] + a;
            Debug.Assert(gamma.shape == beta.shape);
            Debug.Assert(gamma.shape == mean.shape);
            Debug.Assert(gamma.shape == variance.shape);
            Tensor scale = new Tensor(gamma.shape);
            Tensor bias = new Tensor(gamma.shape);
            for (int i = 0; i < gamma.length; ++i)
            {
                scale[i] = gamma[i] / Mathf.Sqrt(variance[i] + epsilon);
                bias[i] = beta[i] - gamma[i] * mean[i] / Mathf.Sqrt(variance[i] + epsilon);
            }
            return Tuple.Create(scale, bias);
        }

        // Transpose channels first to channels last data in MatMul/GEMM weight tensor
        internal static Tensor SwapSpatialDimensionsAndFeaturesInMatMulWeights(Tensor weights, int featureCount)
        {
            Debug.Assert(featureCount <= weights.flatHeight);
            if (featureCount != weights.flatHeight)
            {
                var shape = weights.shape;
                var implicitSpatialDimensionsInWeights = shape.flatHeight / featureCount;
                Debug.Assert(shape.flatHeight % featureCount == 0);
                // reshape: C__K -> CHWK
                weights = weights.Reshape(
                    new TensorShape(featureCount, implicitSpatialDimensionsInWeights, 1, shape.channels));
                // permute: CHWK -> HWCK
                weights = ONNXTensor.Permute(weights, new int[] {1,0,2,3}); // @TODO: use Permute(, onnxLayout:CHWK)
                // reshape: HWCK -> C__K
                weights = weights.Reshape(shape);
            }
            return weights;
        }

        internal static TensorShape Bias(TensorShape shape)
        {
            return new TensorShape(1, 1, 1, shape.channels);
        }

        internal static void Resize2D(ModelBuilder net, ONNXNodeWrapper node, float[] scales)
        {
            if (!scales.All(x => x > 0.0f))
                Warn(net, node, $"Only positive scale values are supported.");

            if (scales.All(x => x < 1.0f))
            {
                if (!scales.All(x => Mathf.Approximately(1f/x, Mathf.Round(1f/x))))
                    Warn(net, node, $"Only inverse of scale values which produce integer are currently supported. Inverse of scale value will be rounded to closest integer.");

                var noStride = new[] {1, 1};
                var noPad = new[] {0, 0};
                var inverseScalesRoundedToInt = scales.Select(x => (int)Mathf.Round(1f/x)).ToArray();
                // @TODO: nearest, actually this is bilinear downsampling
                net.AvgPool2D(node.Name, node.Input0, inverseScalesRoundedToInt, noStride, noPad);
            }
            else
            {
                if (!scales.All(x => Mathf.Approximately(x, Mathf.Round(x))))
                    Warn(net, node, $"Only integer scale values are currently supported. Scale value will be rounded to closest integer value.");

                var scalesRoundedToInt = scales.Select(x => (int)Mathf.Round(x)).ToArray();
                net.Upsample2D(node.Name, node.Input0, scalesRoundedToInt);
            }
        }

        internal void Reduce(ModelBuilder net, ONNXNodeWrapper node, Layer.Type reduceType)
        {
            node.UnsupportedAttribute("keepdims", 1);

            // @TODO: extract into separate function and reuse for other Reduce ops
            var features = node.Input0Features;
            var rank = node.Input0Rank;
            object input = node.Input0;

            foreach (var onnxAxis in node.Axes)
            {
                var axis = ONNXLayout.ConvertAxisToBarracuda(onnxAxis, onnxRank: rank, onnxLayout: "NCHW");
                input = net.Reduce(reduceType, $"{node.Name}__axis{axis}", input, axis);

                bool lastAxis = (axis == -1 || axis == node.Input0Rank - 1); // last axis in Barracuda is feature axis
                if (lastAxis)
                    features = 1; // if reducing over the last feature axis, then operation will collapse all features to 1
                rank--; // rank will be reduced after this operation
                Output(name, features: features, rank: rank);
            }

            net.Identity(node.Name, input);
        }
        
        // ---------------------------------------------------------------------------------
        // Implementation
        private Texture2D m_IconTexture;
        private ONNXModelTensors m_ModelTensors = new ONNXModelTensors();
        private readonly Dictionary<string, Action<ModelBuilder, ONNXNodeWrapper>> m_NodeImporters =
            new Dictionary<string, Action<ModelBuilder, ONNXNodeWrapper>>();

        private Model ConvertOnnxModel(ModelProto onnxModel)
        {
            var model = new Model();
            var modelBuilder = new ModelBuilder(model);

            // Convert graph inputs & outputs
            var initializersByName = onnxModel.Graph.Initializer.ToDictionary(i => i.Name, i => true);
            foreach (ValueInfoProto i in onnxModel.Graph.Input)
            {
                // skip input tensors that have initializer data, they are constant tensors not global inputs
                if (initializersByName.ContainsKey(i.Name))
                    continue;

                if (m_OverrideGlobalInputs.ContainsKey(i.Name))
                {
                    Const(i.Name, m_OverrideGlobalInputs[i.Name]);
                    continue;
                }

                modelBuilder.Input(i.Name, ONNXLayout.ConvertSymbolicShapeToBarracuda(i.Type.TensorType.Shape, onnxLayout:"NCHW"));
                Output(i.Name, onnxShape:i.Type.TensorType.Shape.Dim.Select(d => d.DimValue).ToArray(), onnxLayout:"NCHW");
            }
            foreach (ValueInfoProto o in onnxModel.Graph.Output)
                modelBuilder.Output(o.Name);

            // TODO: process model (recurrent nodes) memories

            // Read constants from initializer list
            foreach (TensorProto initializer in onnxModel.Graph.Initializer)
                Const(initializer.Name, new ONNXTensor(initializer));

            // Convert graph nodes
            foreach (NodeProto onnxNode in onnxModel.Graph.Node)
            {
                var node = new ONNXNodeWrapper(onnxNode, m_ModelTensors, model.Warnings);
                var nodeId = node.Name;
                var opType = node.OperatorType;

                Output(node);

                bool injectDummy = false;
                if (m_NodeImporters.ContainsKey(opType))
                {
                    try
                    {
                        if (node.AreAllInputsConst && !m_ShouldNotBeBaked.Contains(opType))
                        {
                            Profiler.BeginSample($"Bake {opType} {node.Name}");
                            var bakedTensor = BakeNodeIntoConstant(m_NodeImporters[opType], node);
                            Const(node.Name, bakedTensor);
                            var printTensor = bakedTensor.ToBarracuda("NCHW");
                            D.Log($"Baked node {nodeId} into constant of shape {printTensor.shape} and values: {printTensor.DataToString()}");
                            Profiler.EndSample();
                        }
                        else
                        {
                            Profiler.BeginSample($"Import {opType} {node.Name}");
                            m_NodeImporters[opType](modelBuilder, node);
                            Profiler.EndSample();
                        }
                    }
                    catch (Exception e)
                    {
                        // We support the layer but something went wrong while importing it
                        // We log the problem and insert an identity layer
                        string message = $"Unexpected error while parsing layer {nodeId} of type {opType}.\n{e.Message}\n\nJson: {onnxNode}\n{e.StackTrace}\n";
                        Warn(model, nodeId, message);
                        injectDummy = true;
                    }
                }
                else
                {
                    //We don't support this type of layer
                    //We log the problem and insert an identity layer
                    string message = $"Unknown type encountered while parsing layer {nodeId} of type {opType}. We replace by an identity layer.";
                    Warn(model, nodeId, message);
                    injectDummy = true;
                }

                if (injectDummy)
                {
                    var originalLayerHadInputs = (node.InputCount > 0);
                    if (originalLayerHadInputs)
                        modelBuilder.Identity(nodeId, node.Input0);
                    else // if errorneous layer had no inputs, inject dummy constant which does not require any inputs
                        modelBuilder.Const(nodeId, new Tensor());
                }

                m_ModelTensors.CompleteUninitializedFields(node);
            }

            // Convert constant tensors
            int insertionIndex = 0;
            foreach(var entry in constantTensors)
                modelBuilder.Const(entry.Key, entry.Value.ToBarracuda(onnxLayout: "CONST"),
                    insertionIndex++);

            // Model should not contain any broken links in the end
            var unconnectedInputs = ModelAnalyzer.FindBrokenLinks(model);
            Debug.Assert(unconnectedInputs.Length == 0);
            if (unconnectedInputs.Length > 0)
            {
                var message = $"Broken links: {string.Join(", ", unconnectedInputs)}";
                Warn(model, "", message);
            }

            // Parse meta data
            var irVersion = onnxModel.IrVersion; // legacy
            if (onnxModel.OpsetImport?.Count > 0)
                irVersion = onnxModel.OpsetImport[0].Version;
            model.ProducerName = $"{onnxModel.ProducerName} v{onnxModel.ProducerVersion}";
            model.IrSource = "ONNX";
            model.IrVersion = $"{irVersion}";

            // strip :0 at the end of string name for TF import
            if (patchRemoveTrailingTFExportCharacters)
            {
                model.inputs   = model.inputs.Select(i   => { i.name = i.name.EndsWith(":0") ? i.name.Remove(i.name.Length-2) : i.name;
                                                              return i; }).ToList();
                model.outputs  = model.outputs.Select(o  => { o = o.EndsWith(":0") ? o.Remove(o.Length-2) : o;
                                                             return o; }).ToList();
                model.memories = model.memories.Select(m => { m.input  = m.input.EndsWith(":0")  ? m.input.Remove(m.input.Length-2)   : m.input;
                                                              m.output = m.output.EndsWith(":0") ? m.output.Remove(m.output.Length-2) : m.output;
                                                              return m; }).ToList();
                model.layers   = model.layers.Select(l   => { l.name = l.name.EndsWith(":0") ? l.name.Remove(l.name.Length-2) : l.name;
                                                              for(int i = 0; i < l.datasets.Length; i++)
                                                              {
                                                                l.datasets[i].name = l.datasets[i].name.EndsWith(":0") ? l.datasets[i].name.Remove(l.datasets[i].name.Length-2) : l.datasets[i].name;
                                                              }
                                                              for(int i = 0; i < l.inputs.Length; i++)
                                                              {
                                                                l.inputs[i] = l.inputs[i].EndsWith(":0") ? l.inputs[i].Remove(l.inputs[i].Length-2) : l.inputs[i];
                                                              }
                                                              return l; }).ToList();
            }

            return model;
        }

        private ONNXTensor BakeNodeIntoConstant(Action<ModelBuilder, ONNXNodeWrapper> opImportAction, ONNXNodeWrapper node)
        {
            var model = new Model();
            var net = new ModelBuilder(model);

            // add all inputs as constants
            Debug.Assert(node.AreAllInputsConst);
            for (var i = 0; i < node.InputCount; ++i)
            {
                var assumeOnnxLayout = i == 0 ? "NCHW" : "CONST";
                var input = node.Inputs[i];
                net.Const(input,
                    constantTensors[input].ToBarracuda(assumeOnnxLayout));
            }

            // add node that we are going to bake into the constant
            opImportAction(net, node);

            // bake
            var noInputs = new Dictionary<string, Tensor>();

            var useCPUforBaking = WorkerFactory.Device.CPU;
            var worker = WorkerFactory.CreateWorker(model, useCPUforBaking);
            var result = worker.ExecuteAndWaitForCompletion(noInputs);

            // convert from Barracuda back into ONNX layout
            var onnxData = ONNXTensor.Permute(result, new int[] { 0, 3, 1, 2 }); // NHWC -> NCHW
            var onnxShape = onnxData.shape.ToArray().Select(x => (long)x).ToArray();
            return new ONNXTensor(onnxData, onnxShape).SqueezeAll();
        }

        // Asset importer callback
        public override void OnImportAsset(AssetImportContext ctx)
        {
            var onnxModel = new ModelProto();
            using (var readStream = new FileStream(ctx.assetPath, FileMode.Open))
            using (var inputStream = new CodedInputStream(readStream))
                onnxModel.MergeFrom(inputStream);

            var model = ConvertOnnxModel(onnxModel);
            D.Log($"ONNX v{model.IrVersion}. Producer: {model.ProducerName}. Asset path: {ctx.assetPath}");
            D.Log($"Barracuda model: {model}");

            NNModelData assetData = ScriptableObject.CreateInstance<NNModelData>();
            using (var memoryStream = new MemoryStream())
            using (var writer = new BinaryWriter(memoryStream))
            {
                ModelWriter.Save(writer, model);
                assetData.Value = memoryStream.ToArray();
            }
            assetData.name = "Data";
            assetData.hideFlags = HideFlags.HideInHierarchy;

            NNModel asset = ScriptableObject.CreateInstance<NNModel>();
            asset.modelData = assetData;

            ctx.AddObjectToAsset("main obj", asset, LoadIconTexture());
            ctx.AddObjectToAsset("model data", assetData);

            ctx.SetMainObject(asset);
        }

        private Texture2D LoadIconTexture()
        {
            if (m_IconTexture == null)
            {
                string[] allCandidates = AssetDatabase.FindAssets(iconName);

                if (allCandidates.Length > 0)
                {
                    m_IconTexture = AssetDatabase.LoadAssetAtPath(AssetDatabase.GUIDToAssetPath(allCandidates[0]), typeof(Texture2D)) as Texture2D;
                }
            }
            return m_IconTexture;
        }

        // Helpers to keep track of model tensors
        private void Const(ONNXNodeWrapper node, ONNXTensor onnxTensor)
        {
            m_ModelTensors.AddConstant(node.Name, onnxTensor);
        }
        private void Const(string name, ONNXTensor onnxTensor)
        {
            m_ModelTensors.AddConstant(name, onnxTensor);
        }

        private void Output(ONNXNodeWrapper node, int features = -1, int rank = -1)
        {
            Output(node.Name, features, rank);
        }
        private void Output(string name, int features = -1, int rank = -1)
        {
            m_ModelTensors.AddVariable(name, features, rank);
        }
        private void Output(string name, ONNXTensor onnxTensor)
        {
            m_ModelTensors.AddVariable(name, onnxTensor);
        }
        private void Output(string name, long[] onnxShape, string onnxLayout)
        {
            m_ModelTensors.AddVariable(name, onnxShape, onnxLayout);
        }

        // Logging helpers
        private static void Warn(ModelBuilder builder, ONNXNodeWrapper node, string message)
        {
            Warn(builder.model, node.Name, message);
        }

        private static void Warn(Model model, string layerName, string message)
        {
            model.Warnings.Add(new Model.ImporterWarning(layerName,message));
            Debug.LogWarning(message);
        }
    }

    public class OnnxLayerImportException : Exception
    {
        public OnnxLayerImportException(string message) : base(message) { }
    }
}
