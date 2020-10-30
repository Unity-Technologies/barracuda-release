using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Google.Protobuf;
using Onnx;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Profiling;

namespace Unity.Barracuda.ONNX
{
    /// <summary>
    /// ONNX model converter to Barracuda format.
    /// </summary>
    public class ONNXModelConverter
    {
        // Configuration
        private bool treatErrorsAsWarnings;
        private bool optimizeModel = true;

        // TF2ONNX known issue: (as of 1.5.4)
        // - Conv are framed with Transposes as long as the NCHW flag is not set
        //      (note this seems that it's going to be fixed https://github.com/onnx/tensorflow-onnx/pull/796)
        // - Tensorflow appends :0 to all node names
        bool fixTF2ONNXExportIssues = false;

        private readonly Dictionary<string, ONNXTensor> m_OverrideGlobalInputs = new Dictionary<string, ONNXTensor>()
        {
            { "sequence_length:0", new ONNXTensor(new Tensor(1, 1, new[] { 1f }), new [] { 1 }) },
            { "sequence_length",   new ONNXTensor(new Tensor(1, 1, new[] { 1f }), new [] { 1 }) }
        };
        private readonly HashSet<string> m_ShouldNotBeBaked = new HashSet<string>()
        {
            // the following nodes handle constant inputs in a custom manner and should not be baked:
            "Constant", "Reshape", "Shape", "Slice", "Gather", "Transpose", "Squeeze", "Unsqueeze", "NonZero", "ConstantOfShape",

            // the following nodes are dynamic in nature and can not be baked even when all inputs are constant:
            "RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike"
        };
        private readonly HashSet<string> m_AllInputsChannelFirst = new HashSet<string>()
        {
            // the following onnx nodes have all of there inputs as channel first layout
            "Concat", "Add", "Sum", "Sub", "Mul", "Div", "Pow", "Min", "Max", "Mean", "Greater", "Less", "Equal", "Or", "And", "Xor", "Where"
        };

        // Shortcuts
        private Dictionary<string, ONNXTensor> constantTensors { get { return m_ModelTensors.constants; } }
        private Dictionary<string, VariableTensor> variableTensors { get { return m_ModelTensors.variables; } }
        private Dictionary<string, string> lstmInputs = new Dictionary<string, string>();
        private Dictionary<string, string> lstmOutputs = new Dictionary<string, string>();
        private List<string> layerRequiringUpstreamPatch = new List<string>();
        private void Add(string opType, Action<ModelBuilder, ONNXNodeWrapper> opImportAction)
        {
            m_NodeImporters.Add(opType, opImportAction);
        }

        /// <summary>
        /// Convert ONNX model and return Barracuda Model object.
        /// </summary>
        /// <param name="filePath">Location of the input ONNX model.</param>
        /// <returns>Barracuda Model object.</returns>
        public Model Convert(string filePath)
        {
            using (var readStream = new FileStream(filePath, FileMode.Open, FileAccess.Read))
            using (var inputStream = new CodedInputStream(readStream))
                return Convert(inputStream);
        }

        /// <summary>
        /// Convert ONNX model and return Barracuda Model object.
        /// </summary>
        /// <param name="buffer">Memory buffer containing ONNX model.</param>
        /// <returns>Barracuda Model object.</returns>
        public Model Convert(byte[] buffer)
        {
            using (var inputStream = new CodedInputStream(buffer))
                return Convert(inputStream);
        }

        internal Model Convert(CodedInputStream inputStream)
        {
            var onnxModel = new ModelProto();

            onnxModel.MergeFrom(inputStream);

            fixTF2ONNXExportIssues = (onnxModel.ProducerName == "tf2onnx");

            return ConvertOnnxModel(onnxModel);
        }


        // ONNX parser declaration
        // Add operator handlers here

        /// <summary>
        /// Constructs ONNX model converter
        /// </summary>
        /// <param name="optimizeModel">Enable/disable various model optimizations while importing model from ONNX format.</param>
        /// <param name="treatErrorsAsWarnings">Treat import errors as warnings.</param>
        /// <param name="forceArbitraryBatchSize">Repair model input batch size. Sometimes needed for ONNX models coming from PyTorch.</param>
        public ONNXModelConverter(bool optimizeModel, bool treatErrorsAsWarnings = false,
                                        bool forceArbitraryBatchSize = true)
        {
            this.optimizeModel = optimizeModel;
            this.treatErrorsAsWarnings = treatErrorsAsWarnings;

            var defaultZeroTensor = new ONNXTensor(new Tensor(1, 1, new[] { 0f }), new[] { 1 });
            var defaultOneTensor = new ONNXTensor(new Tensor(1, 1, new[] { 1f }), new[] { 1 });
            var toNCHW = new [] { 0, 3, 1, 2 };
            var toNHWC = new [] { 0, 2, 3, 1 };

            // TODO: setup m_NodeImporters via initializer list
            // TODO: simplify code to avoid passing node.Name over and over again
            Add("Constant", (net, node) => {
                node.UnsupportedAttribute("sparse_value");
                Const(node, node.ValueAsTensor);
            });
            Add("ConstantOfShape", (net, node) => {
                Assert.IsTrue(node.InputCount > 0);
                var valueTensor = node.GetOptionalTensor("value", defaultZeroTensor);
                var onnxShape = node.Input0ConstantONNXShape(name: "input");
                var dataShape = ONNXLayout.ConvertShapeToBarracuda(onnxShape, onnxLayout:"?");
                var tensorData = new Tensor(dataShape);
                tensorData.Fill(valueTensor[0]);
                var constantOfShape = new ONNXTensor(tensorData, onnxShape);
                Const(node, constantOfShape);
            });
            Add("Reshape", (net, node)  => {
                int[] onnxShape;
                if (node.InputCount > 1) // Reshape-5
                {
                    if (node.IsInput1Const)
                    {
                        onnxShape = node.Input1Constant(onnxLayout: "C", name: "shape").AsInts();
                    }
                    else
                    {
                        if (node.Input0Rank <= 4 && variableTensors.TryGetValue(node.Input0, out VariableTensor previousOutput)
                            && previousOutput.layout != VariableTensor.Layout.ChannelsLast)
                        {
                            // For handling all reshapes correctly with dynamic shapes (e.g. rank 3) perform in NCHW layout
                            Layer nchwTranspose = net.Transpose($"Transpose_{node.Input0}_For_{node.Name}", node.Input0, toNCHW);
                            Layer reshape = net.Reshape($"{node.Name}_NCHW", nchwTranspose, node.Input1);
                            net.Transpose(node.Name, reshape, toNHWC);
                            Output(node, rank:4);
                        }
                        else
                        {
                            net.Reshape(node.Name, node.Input0, node.Input1);
                        }
                        return;
                    }
                }
                else // Reshape-1
                    onnxShape = node.Shape;

                if (node.IsInput0Const)
                {
                    // reshape constant source tensor and store it as the new constant
                    var reshapedTensor = constantTensors[node.Input0].Reshape(onnxShape);
                    Const(node, reshapedTensor);
                }
                else
                {
                    Layer reshapeLayer = null;

                    int numDimensionContainingChannelsInformationAfterReshape = 1;
                    var symbolicShape = ONNXLayout.ConvertReshapeToBarracuda(onnxShape, node.Input0Rank, out numDimensionContainingChannelsInformationAfterReshape);
                    int variableDimension = Array.IndexOf(symbolicShape, -1);
                    bool containsNoVariableDimensions = variableDimension == -1;

                    if (containsNoVariableDimensions)
                    {
                        if (forceArbitraryBatchSize)
                            symbolicShape[0] = -1; // force arbitrary batch size
                    }
                    else if (onnxShape.Length <= 4 && node.Input0Rank <= 4
                        && variableDimension != TensorShape.C
                        && onnxShape[onnxShape.Length - 1] == -1
                        && variableTensors.TryGetValue(node.Input0, out VariableTensor previousOutput)
                        && previousOutput.layout != VariableTensor.Layout.ChannelsLast)
                    {
                        // Collapsing any of the spatial dimensions requires a reshape in NCHW layout
                        int[] onnxPaddedShape;
                        if (onnxShape.Length == 3) // correct NCH to NCW
                            onnxPaddedShape = new[] { onnxShape[0], onnxShape[1], 1, onnxShape[2] };
                        else
                            onnxPaddedShape = onnxShape.Concat(Enumerable.Repeat(1, 4 - onnxShape.Length)).ToArray();

                        Layer nchwTranspose = net.Transpose($"Transpose_{node.Input0}_For_{node.Name}", node.Input0, toNCHW);
                        reshapeLayer = net.Reshape($"{node.Name}_NCHW", nchwTranspose, onnxPaddedShape);
                        reshapeLayer = net.Transpose(node.Name, reshapeLayer, toNHWC);
                    }

                    if (reshapeLayer == null)
                        reshapeLayer = net.Reshape(node.Name, node.Input0, symbolicShape);

                    reshapeLayer.axis = numDimensionContainingChannelsInformationAfterReshape;
                    Output(node, rank:onnxShape.Length);
                }
            });
            Add("Expand", (net, node) => {
                var onnxShape = node.Input1Constant(onnxLayout: "C", name: "shape").AsInts();
                var symbolicShape = ONNXLayout.ConvertSymbolicShapeToBarracuda(onnxShape, "NCHW");
                bool containsNoVariableDimensions = Array.IndexOf(symbolicShape, -1) == -1;
                if (containsNoVariableDimensions && forceArbitraryBatchSize)
                    symbolicShape[0] = -1; // force arbitrary batch size

                net.Expand(node.Name, node.Input0, symbolicShape);
                Output(node, rank:symbolicShape.Length);
            });
            Add("Shape", (net, node)    =>
            {
                float[] shapeValuesAsFloats;
                if (node.IsInput0Const)
                {
                    shapeValuesAsFloats = constantTensors[node.Input0].shape.Select(x => (float)x).ToArray();
                }
                else
                {
                    switch (node.Input0Rank)
                    {
                        default:
                        case 4: // NCHW
                        case 3: // NCW
                        case 2: // NC
                            // @TODO: dynamic implementation that would return real shape during execution of the model
                            //
                            // meanwhile at import time we assume 0 (taken from input tensor) for the spatial dimensions
                            // NOTE: this assumption works for common Upsample opset=9 case:
                            //     Upsample.scales = (shape.hw * constant) / shape.hw
                            // however this would not work for potential (opset=10) cases like:
                            //     Resize.size = shape.hw + constant

                            // stored in ONNX layout
                            var shapeWithChannelsFirst = new[] { 0f, node.Input0Features }; // NC
                            var fillSpatialDimensionsWithUnknown = 0f;
                            var numberOfSpatialDimensions = node.Input0Rank - 2;
                            var shapeFollowedWithSpatialDimensions = Enumerable.Repeat(fillSpatialDimensionsWithUnknown, numberOfSpatialDimensions);
                            shapeValuesAsFloats = shapeWithChannelsFirst.Concat(shapeFollowedWithSpatialDimensions).ToArray();

                            break;
                        case 1: // C
                            shapeValuesAsFloats = new[] {(float)node.Input0Features};
                            break;
                        case 0: // scalar
                            shapeValuesAsFloats = new[] {0f};
                            break;
                    }
                }

                var shapeLength = shapeValuesAsFloats.Length;
                Assert.IsTrue(shapeLength == node.Input0Rank);

                var shape = new int[8];
                shape[0] = shapeLength;
                var shapeTensor = new ONNXTensor(
                    // NOTE: stored in single rank ONNX layout
                    // with data in the 1st dimension
                    // thus `shapeLength` specifies the length of the 1st dimension
                    data:new Tensor(shape, shapeValuesAsFloats),
                    onnxShape:new [] { shapeLength });

                Const(node, shapeTensor);
                Output(node, features:shapeLength, productOfShape:node.Input0);
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

                    var features = node.Input0Features;
                    var inputRank = node.Input0Rank;
                    var outputRank = inputRank + 1;
                    Output(node.Name, features: features, rank: outputRank);

                    // ONNX pseudocode here:
                    // a = Tensor [2, 10]             # NC   -> barracuda N11C
                    // b = Unsqueeze(a, axis=0)
                    // # b is now Tensor [1, 2, 10]   # NCHW -> barrada NHWC
                    // Because ONNX is NCHW, but generally hell knows what goes where and Barracuda is strict NHWC. We end up with:
                    // `a` would be [2, 1, 1, 10], but `b` would have to be [1, 10, 1, 2]. Note the actual data swap in channels!
                    int axis = node.Axes[0];
                    if (axis < 0)
                        axis = node.Input0Rank+1 - axis;

                    var transpose = ONNXLayout.UnSqueezeAxisPermutationForMappingONNXLayoutToBarracuda(inputRank, axis, "NCHW");
                    net.Transpose(node.Name, node.Input0, transpose);
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
                    var features = node.Input0Features;
                    var inputRank = node.Input0Rank;
                    var outputRank = inputRank - 1;
                    Output(node.Name, features: features, rank: outputRank);

                    // See Unsqueeze above for explanation
                    int axis = node.Axes[0];
                    if (axis < 0)
                        axis = node.Input0Rank + 1 - axis;

                    var transpose = ONNXLayout.SqueezeAxisPermutationForMappingONNXLayoutToBarracuda(inputRank, axis, "NCHW");
                    net.Transpose(node.Name, node.Input0, transpose);
                }
            });
            Add("Flatten", (net, node)  => {
                node.UnsupportedAttribute("axis", 1);
                net.Flatten(node.Name, node.Input0);
                Output(node, rank:2);
            });
            Add("Concat", (net, node) => {
                int axis = node.AxisOptional(0);

                if (node.Inputs.Length == 1)
                    net.Identity(node.Name, node.Input0);
                else
                {
                    // TODO: write dedicated ONNXTensor.Concat() so that output shape is exact to ONNX
                    // if (node.AreAllInputsConst)
                    //     Const(node, ONNXTensor.Concat(node.Inputs.Select(i => constantTensors[i]).ToArray(), axis));

                    axis = ONNXLayout.ConvertAxisToBarracuda(axis, onnxRank: node.Input0Rank, onnxLayout: "NCHW");
                    net.Concat(node.Name, node.Inputs, axis, true);

                    bool lastAxis = (axis == -1 || axis == TensorShape.C || axis == node.Input0Rank - 1); // last axis in Barracuda is feature axis
                    if (lastAxis)
                    {
                        var featuresConcatenated = node.Inputs.Sum(i => variableTensors[i].features);
                        Output(node, features: featuresConcatenated);
                    }
                }
            });
            Add("Split", (net, node) => {

                int axis = node.AxisOptional(0);
                int[] splits;
                try {
                    splits = node.GetRequiredIntArray("split");
                } catch (OnnxLayerImportException) {
                    throw new OnnxLayerImportException($"Unsupported default attribute `split` for node {node.Name} of type Split. Value is required.");
                }

                Assert.IsTrue(splits.Length == node.Outputs.Length);
                axis = ONNXLayout.ConvertAxisToBarracuda(axis, onnxRank:node.Input0Rank, onnxLayout:"NCHW");
                int currentSliceStartIndex = 0;

                //Convert `Split` into multiple `StridedSlice` operations.
                for (int i = 0; i < splits.Length; ++i)
                {
                    var starts  = new int[] {0, 0, 0, 0, 0, 0, 0, 0};
                    var ends    = new int[] {0, 0, 0, 0, 0, 0, 0, 0};
                    var strides = new int[] {1, 1, 1, 1, 1, 1, 1, 1};
                    starts[axis] = currentSliceStartIndex;
                    ends[axis] = starts[axis] + splits[i];
                    net.StridedSlice(node.Outputs[i], node.Input0,starts,ends,strides);
                    currentSliceStartIndex += splits[i];
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

                Assert.IsTrue(starts.Length == ends.Length);
                var onnxRank    = node.Input0Rank;
                var onnxStarts  = Enumerable.Repeat(0, onnxRank).ToArray();
                var onnxEnds    = Enumerable.Repeat(int.MaxValue, onnxRank).ToArray(); // by default copy the whole axis till the end
                var onnxSteps   = Enumerable.Repeat(1, onnxRank).ToArray();

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
                    var slicedTensor = constantTensors[node.Input0].Slice(starts:onnxStarts, ends:onnxEnds, steps:onnxSteps);
                    Const(node, slicedTensor);
                }
                else
                {
                    net.StridedSlice(node.Name, node.Input0,
                        starts:ONNXLayout.PermuteToBarracuda(onnxStarts, onnxLayout:"NCHW",0),
                        ends:ONNXLayout.PermuteToBarracuda(onnxEnds, onnxLayout:"NCHW",int.MaxValue),
                        strides:ONNXLayout.PermuteToBarracuda(onnxSteps, onnxLayout:"NCHW",1));
                }
            });
            Add("Gather", (net, node) =>
            {
                int axis = node.AxisOptional(0);

                if (node.IsInput0Const && node.IsInput1Const)
                {
                    var indices = node.Input1Constant(onnxLayout:"C", name:"indices").AsInts();

                    // If the previous node was a shape and we're gathering an inferred value, then don't treat the shape as a constant
                    if (node.Input0.IndexOf("shape", StringComparison.OrdinalIgnoreCase) >= 0
                        && indices.Length == 1 && indices[0] > 0
                        && constantTensors[node.Input0].ToBarracuda("C")[indices[0]] == 0 // Must resolve at runtime
                        && variableTensors.TryGetValue(node.Input0, out VariableTensor input0Tensor)
                        && variableTensors.TryGetValue(input0Tensor.productOfShape, out VariableTensor shapeInputTensor))
                    {
                        axis = ONNXLayout.ConvertAxisToBarracuda(indices[0], shapeInputTensor.rank, "NCHW");
                        net.Shape(node.Name, input0Tensor.productOfShape, axis);
                        D.Log($"Re-writing {node.Name} to a Shape+Axis layer (results in a scalar)");
                    }
                    else
                    {
                        ONNXTensor gatheredTensor = constantTensors[node.Input0].Gather(axis, indices);
                        Const(node, gatheredTensor);
                    }
                }
                else
                {
                    int input1Rank = node.Input1Rank;
                    if (node.IsInput1Const)
                    {
                        // The original rank was cached above since our constant tensor requires a shape of rank 1 and original may have been a scalar
                        var indices = node.Input1Constant(onnxLayout: "C", name: "indices").AsFloats();
                        var constTensor = new ONNXTensor(new Tensor(new [] { indices.Length, 1, 1, 1, 1, 1, 1, 1 }, indices), new [] { indices.Length });
                        Const(node.Input1, constTensor);
                    }

                    axis = ONNXLayout.ConvertAxisToBarracuda(axis, onnxRank:node.Input0Rank, onnxLayout:"NCHW");
                    net.Gather(node.Name, node.Input0, node.Input1, axis, true);
                    Output(node.Name, rank: input1Rank + node.Input0Rank - 1);
                }
            });
            Add("NonMaxSuppression", (net, node) =>
            {
                int centerPointBox = node.GetOptionalInt("center_point_box", 0);

                var boxes = node.GetRequiredInput(0);
                var scores = node.GetRequiredInput(1);
                object maxOutputBoxesPerClass = 0f;
                object iouThreshold = 0f;
                object scoreThreshold = 0f;

                if (node.InputCount > 4 && node.IsInput2Const && node.IsInput3Const && node.IsInput4Const
                    || node.InputCount > 3 && node.IsInput2Const && node.IsInput3Const
                    || node.InputCount > 2 && node.IsInput2Const)
                {
                    // Use constant version (possibly with defaults)
                    maxOutputBoxesPerClass = node.Input2ConstantOptional((float)maxOutputBoxesPerClass, "C", nameof(maxOutputBoxesPerClass))[0];
                    iouThreshold = node.Input3ConstantOptional((float)iouThreshold, "C", nameof(iouThreshold))[0];
                    scoreThreshold = node.Input4ConstantOptional((float)scoreThreshold, "C", nameof(scoreThreshold))[0];
                }
                else
                {
                    // Use dynamic tensor version
                    maxOutputBoxesPerClass = node.Input2Optional;
                    iouThreshold = node.Input3Optional;
                    scoreThreshold = node.Input4Optional;
                }

                net.NonMaxSuppression(node.Name, boxes, scores, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, centerPointBox);
                Output(node, rank: 2);
            });
            Add("OneHot", (net, node) => {
                node.UnsupportedAttribute("axis", -1);

                var defaultOffOn = new Tensor(2, 0, new float[] {0, 1});

                var depth = (int)node.Input1Constant(onnxLayout:"C", name:"depth")[0];
                var offon = node.Input2ConstantOptional(defaultOffOn, onnxLayout:"C", name:"values");
                net.OneHot(node.Name, node.Input0, depth, (int)offon[1], (int)offon[0]);
            });
            Add("TopK", (net, node) => {
                int axis = node.AxisOptional(-1);
                axis = ONNXLayout.ConvertAxisToBarracuda(axis, onnxRank:node.Input0Rank, onnxLayout:"NCHW");

                // TopK-11 introduced these options
                bool largest = node.GetOptionalInt("largest", 1) == 1;
                // If sorted = false, then the output is undefined
                bool sorted = node.GetOptionalInt("sorted", 1) == 1;

                string kName;
                if (node.InputCount > 1) // TopK-10 introduced K as an input tensor
                {
                    kName = node.Input1;
                }
                else
                {
                    // TopK-1
                    int k = node.GetRequiredInt("k");
                    kName = "Const_TopK";
                    var kTensor = new ONNXTensor(
                        data:new Tensor(new[] { 1, 1, 1, 1 }, new[] { (float)k }, kName),
                        onnxShape:new [] { 1 });

                    Const(node, kTensor);
                }

                Layer indices = net.TopKIndices(node.Outputs[1], node.Input0, kName, axis, largest, sorted);
                Output(node.Outputs[1], rank: node.Input0Rank);
                net.TopKValues(node.Outputs[0], node.Input0, indices, axis);
                Output(node.Outputs[0], rank: node.Input0Rank);
            });

            Add("NonZero", (net, node) => {

                if (node.IsInput0Const)
                {
                    var nonZeroTensor = constantTensors[node.Input0].NonZero();
                    Const(node, nonZeroTensor);
                }
                else
                {
                    net.NonZero(node.Name, node.Input0);
                    Output(node.Outputs[0], rank: 2);
                }
            });

            // LSTM

            //    - it = f(Xt*Wi + Ht_1*Ri + Wbi + Rbi)
            //    - ft = f(Xt*Wf + Ht_1*Rf + Wbf + Rbf)
            //    - ct = g(Xt*Wc + Ht_1*Rc + Wbc + Rbc), c means j in our formula
            //    - Ct =   ft . Ct_  + it . ct
            //    - ot = f(Xt*Wo + Ht_1*Ro + Wbo + Rbo)
            //    - Ht =   ot . h(Ct)

            Add("LSTM", (net, node) =>
            {
                var W = node.Input1Constant(onnxLayout: "RKC", name: "W");
                var R = node.Input2Constant(onnxLayout: "RKC", name: "R");
                var B = node.Input3Constant(onnxLayout: "RC", name: "B");

                // gate order [iofj]

                var ops = new ReferenceCPUOps();
                var w_i = ops.StridedSlice(W, new[] {0,0,0,0}, new[] {W.batch,1,1,W.channels/4 }, new[] {1, 1, 1, 1});
                var w_o = ops.StridedSlice(W, new[] {0,0,0,W.channels/4}, new[] {W.batch,1,1,2*W.channels/4 }, new[] {1, 1, 1, 1});
                var w_f = ops.StridedSlice(W, new[] {0,0,0,2*W.channels/4}, new[] {W.batch,1,1,3*W.channels/4 }, new[] {1, 1, 1, 1});
                var w_j = ops.StridedSlice(W, new[] {0,0,0,3*W.channels/4}, new[] {W.batch,1,1,4*W.channels/4 }, new[] {1, 1, 1, 1});

                var r_i = ops.StridedSlice(R, new[] {0,0,0,0}, new[] {R.batch,1,1,R.channels/4 }, new[] {1, 1, 1, 1});
                var r_o = ops.StridedSlice(R, new[] {0,0,0,R.channels/4}, new[] {R.batch,1,1,2*R.channels/4 }, new[] {1, 1, 1, 1});
                var r_f = ops.StridedSlice(R, new[] {0,0,0,2*R.channels/4}, new[] {R.batch,1,1,3*R.channels/4 }, new[] {1, 1, 1, 1});
                var r_j = ops.StridedSlice(R, new[] {0,0,0,3*R.channels/4}, new[] {R.batch,1,1,4*R.channels/4 }, new[] {1, 1, 1, 1});

                var wb_i = ops.StridedSlice(B, new[] {0,0,0,0}, new[] {1,1,1,B.channels/8 }, new[] {1, 1, 1, 1});
                var wb_o = ops.StridedSlice(B, new[] {0,0,0,B.channels/8}, new[] {1,1,1,2*B.channels/8 }, new[] {1, 1, 1, 1});
                var wb_f = ops.StridedSlice(B, new[] {0,0,0,2*B.channels/8}, new[] {1,1,1,3*B.channels/8 }, new[] {1, 1, 1, 1});
                var wb_j = ops.StridedSlice(B, new[] {0,0,0,3*B.channels/8}, new[] {1,1,1,4*B.channels/8 }, new[] {1, 1, 1, 1});

                var rb_i = ops.StridedSlice(B, new[] {0,0,0,4*B.channels/8}, new[] {1,1,1,5*B.channels/8 }, new[] {1, 1, 1, 1});
                var rb_o = ops.StridedSlice(B, new[] {0,0,0,5*B.channels/8}, new[] {1,1,1,6*B.channels/8 }, new[] {1, 1, 1, 1});
                var rb_f = ops.StridedSlice(B, new[] {0,0,0,6*B.channels/8}, new[] {1,1,1,7*B.channels/8 }, new[] {1, 1, 1, 1});
                var rb_j = ops.StridedSlice(B, new[] {0,0,0,7*B.channels/8}, new[] {1,1,1,8*B.channels/8 }, new[] {1, 1, 1, 1});


                var memSize = r_i.flatHeight;

                var baseLSTMName = ResolveLstmInputName(node);
                var initial_h = $"{baseLSTMName}_h";
                var initial_c = $"{baseLSTMName}_c";

                var baseLSTMOutputName = ResolveLstmOutputName(node);
                var output_h = $"{baseLSTMOutputName}_h";
                var output_c = $"{baseLSTMOutputName}_c";


                var i_mad_w = net.Dense($"{node.Name}_bc_i_mad_w", node.Input0, w_i, wb_i);
                var i_mad_r = net.Dense($"{node.Name}_bc_i_mad_r", initial_h, r_i, rb_i);
                var i_mad = net.Add($"{node.Name}_bc_i_mad", new [] {i_mad_w, i_mad_r});

                var j_mad_w = net.Dense($"{node.Name}_bc_j_mad_w", node.Input0, w_j, wb_j);
                var j_mad_r = net.Dense($"{node.Name}_bc_j_mad_r", initial_h, r_j, rb_j);
                var j_mad = net.Add($"{node.Name}_bc_j_mad", new [] {j_mad_w, j_mad_r});

                var f_mad_w = net.Dense($"{node.Name}_bc_f_mad_w", node.Input0, w_f, wb_f);
                var f_mad_r = net.Dense($"{node.Name}_bc_f_mad_r", initial_h, r_f, rb_f);
                var f_mad = net.Add($"{node.Name}_bc_f_mad", new [] {f_mad_w, f_mad_r});

                var o_mad_w = net.Dense($"{node.Name}_bc_o_mad_w", node.Input0, w_o, wb_o);
                var o_mad_r = net.Dense($"{node.Name}_bc_o_mad_r", initial_h, r_o, rb_o);
                var o_mad = net.Add($"{node.Name}_bc_o_mad", new [] {o_mad_w, o_mad_r});

                var i = net.Sigmoid($"{node.Name}_bc_i_sigmoid", i_mad);
                var j = net.Tanh($"{node.Name}_bc_j_tanh", j_mad);
                var f = net.Sigmoid($"{node.Name}_bc_f_sigmoid", f_mad);
                var o = net.Sigmoid($"{node.Name}_bc_o_sigmoid", o_mad);

                var state_c_mul = net.Mul($"{node.Name}_bc_state_c_mul", new[] {initial_c, f.name});
                var i_j_mul = net.Mul($"{node.Name}_bc_i_j_mul", new[] {i, j});
                var state_c = net.Add(output_c, new[] {state_c_mul, i_j_mul});
                var state_c_tanh = net.Tanh($"{node.Name}_bc_state_c_tanh", state_c);
                var state_h = net.Mul(output_h, new[] {o, state_c_tanh});

                net.Identity(node.Outputs[0], state_h);
                net.Identity(node.Outputs[1], state_h);
                net.Identity(node.Outputs[2], state_c);

                net.Memory(initial_c, state_c, new TensorShape(-1,1,1,memSize));
                net.Memory(initial_h, state_h, new TensorShape(-1,1,1,memSize));

                Output(node.Outputs[0], features:wb_o.channels, rank:2);
                Output(node.Outputs[1], features:wb_o.channels, rank:2);
                Output(node.Outputs[2], features:wb_o.channels, rank:2);

            });

            // Activation ops
            Add("Relu", (net, node)     => { net.Relu(node.Name, node.Input0); });
            Add("Softmax", (net, node) =>
            {
                const int defaultAxis = 1;
                int axis = node.AxisOptional(defaultAxis); // Leave in NCHW form and transpose instead

                string input = node.Input0;
                string output = node.Name;
                if (axis != 1)
                {
                    Layer transposeLayer = net.Transpose($"Transpose_For_{node.Name}", input, toNCHW);
                    input = transposeLayer.name;
                    output = $"{node.Name}_NCHW"; // Use an intermediate node name since the original name will now be final transpose
                }

                Layer layer = net.Softmax(output, input, axis);

                if (axis != 1)
                    net.Transpose(node.Name, layer.name, toNHWC);
            });
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
            Add("Clip", (net, node)     => {
                float minValue = float.MinValue;
                float maxValue = float.MaxValue;

                if (node.InputCount > 1) // Clip-11
                {
                    minValue = node.Input1ConstantOptional(minValue, onnxLayout:"C", name:"min")[0];
                    maxValue = node.Input2ConstantOptional(maxValue, onnxLayout:"C", name:"max")[0];
                }
                else
                {
                    minValue = node.MinOptional(minValue);
                    maxValue = node.MaxOptional(maxValue);
                }
                net.Clip(node.Name, node.Input0, minValue, maxValue);
            });
            Add("Acos", (net, node) => { net.Acos(node.Name, node.Input0); });
            Add("Acosh", (net, node) => { net.Acosh(node.Name, node.Input0); });
            Add("Asin", (net, node) => { net.Asin(node.Name, node.Input0); });
            Add("Asinh", (net, node) => { net.Asinh(node.Name, node.Input0); });
            Add("Atan", (net, node) => { net.Atan(node.Name, node.Input0); });
            Add("Atanh", (net, node) => { net.Atanh(node.Name, node.Input0); });
            Add("Cos", (net, node) => { net.Cos(node.Name, node.Input0); });
            Add("Cosh", (net, node) => { net.Cosh(node.Name, node.Input0); });
            Add("Sin", (net, node) => { net.Sin(node.Name, node.Input0); });
            Add("Sinh", (net, node) => { net.Sinh(node.Name, node.Input0); });
            Add("Tan", (net, node) => { net.Tan(node.Name, node.Input0); });

            string[] GetCorrectedConstants(ONNXNodeWrapper node, ModelBuilder net)
            {
                string[] inputs = new string[node.Inputs.Length];
                Array.Copy(node.Inputs, inputs, inputs.Length);

                if (node.IsInput1Const)
                {
                    string onnxLayout;
                    switch (node.Input1Rank)
                    {
                        case 1:
                            onnxLayout = "C";
                            break;

                        default:
                            onnxLayout = "NCHW";
                            break;
                    }

                    string constName = $"Const_{node.Input1}";
                    if (!constantTensors.ContainsKey(constName))
                    {
                        Tensor tensorData = node.Input1Constant(onnxLayout, node.Input1);
                        Layer layer = net.Const(constName, tensorData);
                        inputs[1] = layer.name;
                        Const(constName, new ONNXTensor(tensorData, tensorData.shape.ToArray()));
                    }
                }

                return inputs;
            }

            // Broadcast ops
            Add("Add", (net, node)     => { net.Add(node.Name, GetCorrectedConstants(node, net)); });
            Add("Sum", (net, node)     => { net.Add(node.Name, GetCorrectedConstants(node, net)); }); // Sum is implemented via Add
            Add("Sub", (net, node)     => { net.Sub(node.Name, GetCorrectedConstants(node, net)); });
            Add("Mul", (net, node)     => { net.Mul(node.Name, GetCorrectedConstants(node, net)); });
            Add("Div", (net, node)     => { net.Div(node.Name, GetCorrectedConstants(node, net)); });
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
            Add("Where", (net, node)   => { net.Where(node.Name, node.Input0, node.Input1, node.Input2); });

            // Padding ops
            Add("Pad", (net, node) =>
            {
                var mode = node.ModeOptional("constant");
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

                int[] strides = node.Strides;
                int[] pads = node.Pads;

                if (strides.Length == 1)
                    strides = new[] { 1, strides[0] };
                Assert.IsTrue(strides.Length == 2);

                int[] kernenShape = node.KernelShape;
                if (kernenShape.Length == 1)
                    kernenShape = new[] { 1, kernenShape[0] };

                net.MaxPool2D(node.Name, node.Input0, kernenShape, strides, pads);
            });
            Add("GlobalAveragePool", (net, node) => { net.GlobalAvgPool2D(node.Name, node.Input0); });
            Add("GlobalMaxPool",     (net, node) => { net.GlobalMaxPool2D(node.Name, node.Input0); });
            Add("Upsample", (net, node) => {
                // @TODO: the same for Resize node
                string mode = node.ModeOptional("nearest");
                if (node.InputCount == 2 && !node.IsInput1Const)
                    net.Upsample2D(node.Name, node.Inputs[0], node.Inputs[1], IsModeBilinear(net, node, mode));
                else
                    Resize2D(net, node, node.Scales, mode);
            });
            Add("Resize", (net, node) => {
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
                }

                if (node.InputCount > 3)
                {
                    var mode = node.ModeOptional("nearest");
                    var bilinear = IsModeBilinear(net, node, mode);
                    net.Resample2D(node.Name, node.Input0, node.Sizes, bilinear);
                }
                else
                {
                    Resize2D(net, node, node.Scales, node.ModeOptional("nearest"));
                }
            });
            Add("Transpose", (net, node) =>
            {
                // From https://github.com/onnx/onnx/blob/master/docs/Operators.md#transpose
                // By default, reverse the dimensions, otherwise permute the axes according to the values given.

                if (node.IsInput0Const)
                {
                    int inputTensorRank = constantTensors[node.Input0].rank;
                    var defaultPermutations = new int[inputTensorRank];
                    for (int i = 0; i < inputTensorRank; ++i)
                        defaultPermutations[i] = inputTensorRank - 1 - i;
                    var permutations = node.GetOptionalIntArray("perm", defaultPermutations);

                    var transposedTensor = constantTensors[node.Input0].Permute(permutations);
                    Const(node, transposedTensor);
                }
                else
                {
                    var defaultPermutations = new[] {5, 4, 3, 2, 1, 0};
                    var permutations = node.GetOptionalIntArray("perm", defaultPermutations);
                    if (permutations.Length > 6)
                        throw new OnnxLayerImportException($"Transpose support up to 6 dimensions but got a permutations of rank {permutations}.");

                    bool fromTF = net.model.ProducerName.Contains("tf2onnx");

                    if (Enumerable.SequenceEqual(permutations, new[] { 0, 3, 1, 2 }) || // NHWC -> NCHW
                        Enumerable.SequenceEqual(permutations, new[] { 0, 2, 3, 1 }))   // NCHW -> NHWC
                    {
                        // @TODO: reorder uptream nodes and global input dimensions accordingly from NHWC -> NCHW
                        net.Identity(node.Name, node.Input0);

                        if (permutations[1] == 3)       // NHWC -> NCHW
                            Output(node, layout: VariableTensor.Layout.ChannelsFirst);
                        else if (permutations[1] == 2)  // NCHW -> NHWC
                        {
                            Output(node, layout: VariableTensor.Layout.ChannelsLast);
                            layerRequiringUpstreamPatch.Add(node.Name);
                        }
                        else
                            Assert.IsTrue("Reached unexpected branch" == "");
                    }
                    else if (Enumerable.SequenceEqual(permutations, new[] { 0, 2, 1 }))       // NWC <-> NCW
                    {
                        // @TODO: reorder uptream nodes and global input dimensions accordingly from NHWC -> NCHW
                        if (fromTF)
                        {
                            Warn(net, node, $"Use '--inputs-as-nchw' flag when exporting model from Tensorflow with tf2onnx");
                            net.Identity(node.Name, node.Input0);

                            // flip layout
                            if (node.Input0Layout == VariableTensor.Layout.ChannelsLast)
                                Output(node, layout: VariableTensor.Layout.ChannelsFirst);
                            else
                            {
	                            Output(node, layout: VariableTensor.Layout.ChannelsLast);
    	                        layerRequiringUpstreamPatch.Add(node.Name);
                            }
                        }
                        else
                        {
                            int[] barracudaPermutation = { 0, 1, 3, 2 };
                            net.Transpose(node.Name, node.Input0, barracudaPermutation);
                        }
                    }
                    else if (Enumerable.SequenceEqual(permutations, new[] { 1, 0, 2 })) // batch <-> seq_length
                    {
                        // LSTM layout is problematic as it's usually flanked by a few Transposed if exported from TF
                        // @TODO investigate if better solution
                        net.Identity(node.Name, node.Input0);
                    }
                    else
                    {
                        //Here we assume `Channels` are represented by only one dimensions and it that it is the 2nd one.
                        //however in some case (example: shufflenet, sub-pixel-cnn) reshape-transpose-reshape pattern lead
                        //to channels being represented by two dimenssion this is handled in
                        //FixReshapeTransposePatternWhenChannelsAreSplitIntoMultipleDimensions()

                        //Expand received permutation to 6D adding padding between Channels and other feature dimensions.
                        int numDimensionDimensionsThatWerePaddedAtCenterOfTranspose = 0;
                        var permutationsNCTDHW = ONNXLayout.ExpandONNXPermutationToNCTDHW(permutations, out numDimensionDimensionsThatWerePaddedAtCenterOfTranspose);

                        //From channel first to channel last.
                        var permutationsNTDHWC = ONNXLayout.ConvertPermutationToLayout(permutationsNCTDHW, "NCTDHW", "NTDHWC");

                        //6d to 8d
                        int[] permuteSRNTDHWC = new int[TensorShape.MaxRank];
                        permuteSRNTDHWC[0] = 0;
                        permuteSRNTDHWC[1] = 1;
                        for (int i = 0; i < 6; ++i)
                            permuteSRNTDHWC[i+2] = 2+permutationsNTDHWC[i];

                        var layer = net.Transpose(node.Name, node.Input0, permuteSRNTDHWC);
                        layer.axis = numDimensionDimensionsThatWerePaddedAtCenterOfTranspose;
                    }
                }
            });

            Add("DepthToSpace", (net, node) => {
                net.DepthToSpace(node.Name, node.Input0, node.BlockSize, node.ModeOptional("DCR"));
            });

            Add("SpaceToDepth", (net, node) => {
                net.SpaceToDepth(node.Name, node.Input0, node.BlockSize);
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
                weights = SwapSpatialDimensionsAndFeaturesInMatMulWeights(weights, node.Input0Features, node.Input0Layout);
                net.Dense(node.Name, node.Input0, weights, biases);
                Output(node, features:weights.channels, rank:2); // Gemm forces flatten of the input to rank 2
            });
            Add("MatMul", (net, node)   => {
                if (node.InputCount == 2 && !node.IsInput1Const || node.Input0Rank != 2)
                {
                    net.MatMul(node.Name, node.Input0, node.Input1);
                    Output(node, features: node.Input0Features, rank: node.Input0Rank);
                }
                else
                {
                    var weights = node.Input1Constant(onnxLayout: "CK", name: "B");
                    var biases = node.DefaultTensor(Bias(weights.shape), 0.0f);
                    // Change data layout from "channels first" to "channels last"
                    weights = SwapSpatialDimensionsAndFeaturesInMatMulWeights(weights, node.Input0Features, node.Input0Layout);
                    net.Dense(node.Name, node.Input0, weights, biases);
                    Output(node, features: weights.channels, rank: 2); // MatMul forces flatten of the input to rank 2
                }
            });
            Add("Conv", (net, node)     => {
                int[] dilations = new[] { 1, 1 }; // @TODO trap on wrong values
                int[] strides = node.Strides;
                int[] pads = node.Pads;

                node.IgnoredAttribute("kernel_shape", "Kernel shape is derived from K tensor weights instead");
                var kernels = node.Input1Constant(onnxLayout: "KCHW", name: "W");

                var kernelRank = node.Input1Rank;
                if (kernelRank == 3) // Conv1D
                {
                    dilations = node.DilatationsOptional(new[] { 1 }); // @TODO trap on wrong values
                    Assert.IsTrue(dilations.Length == 1);
                    dilations = new[] { dilations[0], 1 };

                    if (strides.Length == 1)
                        strides = new[] { strides[0], 1 };

                    if (pads.Length == 2)
                        pads = new[] { pads[0], 0, pads[1], 0 };
                }
                else if (kernelRank == 4) // Conv2D
                {
                    dilations = node.DilatationsOptional(new[] { 1, 1 });
                    Assert.IsTrue(dilations.Length == 2);
                }
                else
                {
                    Warn(net, node, $"Unsuported Conv kernel rank. Conv2D assumes rank 4, Conv1D assumes rank 3, but got {kernelRank}.");
                }

                Assert.IsTrue(dilations.Length == 2);
                if (dilations[0] != 1 || dilations[1] != 1)
                    kernels = DilateKernel(kernels, dilations); // @TODO inefficient method. Support dilatation in kernel code properly

                var biases = node.Input2ConstantOptional(Bias(kernels.shape), 0.0f, onnxLayout: "C", name: "B");

                if (node.GroupOptional() > 1)
                    net.DepthwiseConv2D(node.Name, node.Input0, strides, pads, kernels, biases);
                else
                    net.Conv2D(node.Name, node.Input0, strides, pads, kernels, biases);

                Output(node, features: kernels.channels);
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
                if (variance.length != scale.length || scale.length != bias.length || bias.length != mean.length)
                    Warn(net, node, $"Number of elements in all parameters for BatchNorm must be the same." +
                        $"Parameter shapes are: {scale.shape}, {bias.shape}, {mean.shape}, {variance.shape}");
                if (variance.channels != node.Input0Features && node.Input0Features > 0)
                    Warn(net, node, $"Number of elements in BatchNorm must match features from the previous layer. Was expecting {node.Input0Features}, but got {variance.channels}.");
                var fusedData = FuseBatchNormWeights(scale, bias, mean, variance, node.EpsilonOptional(), node.Input0Features);
                net.ScaleBias(node.Name, node.Input0, fusedData.Item1, fusedData.Item2);
            });
            Add("ImageScaler", (net, node) =>
            {
                var attrBias = node.Bias;
                var attrScale = node.ScaleOptional();
                int maxElements = attrBias.Length;

                Tensor scale = new Tensor(1, maxElements);
                Tensor bias = new Tensor(1, maxElements);
                for (int i = 0; i < maxElements; ++i)
                {
                    scale[i] = attrScale;
                    bias[i] = attrBias[i];
                }
                net.ScaleBias(node.Name, node.Input0, scale, bias);
            });
            Add("InstanceNormalization", (net, node) => {
                var scale     = node.Input1Constant(onnxLayout:"C", name:"scale");
                var bias      = node.Input2ConstantOptional(scale.shape, 0.0f, onnxLayout:"C", name:"B");
                if (scale.length != bias.length)
                    Warn(net, node, $"Number of elements in all parameters for InstanceNorm must be the same." +
                        $"Parameter shapes are: {scale.shape}, {bias.shape}");
                if (scale.channels != node.Input0Features && node.Input0Features > 0)
                {
                    Warn(net, node, $"Number of elements in InstanceNorm must match features from the previous layer. Was expecting {node.Input0Features}, but got {scale.channels}.");
                    var scaleArray = scale.ToReadOnlyArray();
                    Array.Resize(ref scaleArray, node.Input0Features);
                    var biasArray = bias.ToReadOnlyArray();
                    Array.Resize(ref biasArray, node.Input0Features);
                    scale = new Tensor(1, node.Input0Features, scaleArray);
                    bias = new Tensor(1, node.Input0Features, biasArray);
                }
                net.Normalization(node.Name, node.Input0, scale, bias, node.EpsilonOptional());
            });
            Add("LRN", (net, node) => {
                float bias = node.GetOptionalFloat("bias", 1.0f);
                int size = node.GetRequiredInt("size");
                net.LRN(node.Name, node.Input0, node.AlphaOptional(0.0001f), node.BetaOptional(0.75f), bias, size);
            });
            // random ops
            Add("RandomNormal", (net, node) => {
                var shape = ONNXLayout.ConvertShapeToBarracuda(onnxShape:node.Shape, onnxLayout:"NCHW");
                net.RandomNormal(node.Name, shape, node.MeanOptional(), node.ScaleOptional(), node.Seed);
                Output(node, rank:node.Shape.Length);
            });
            Add("RandomNormalLike", (net, node) => {
                net.RandomNormal(node.Name, node.Input0, node.MeanOptional(), node.ScaleOptional(), node.Seed);
            });
            Add("RandomUniform", (net, node) => {
                float high     = node.GetOptionalFloat("high",  1.0f);
                float low      = node.GetOptionalFloat("low",   0.0f);
                var shape = ONNXLayout.ConvertShapeToBarracuda(onnxShape:node.Shape, onnxLayout:"NCHW");
                net.RandomUniform(node.Name, shape, low, high, node.Seed);
                Output(node, rank:node.Shape.Length);
            });
            Add("RandomUniformLike", (net, node) => {
                float high     = node.GetOptionalFloat("high",  1.0f);
                float low      = node.GetOptionalFloat("low",   0.0f);
                net.RandomUniform(node.Name, node.Input0, low, high, node.Seed);
            });
            Add("Multinomial", (net, node) => {
                int samples    = node.GetOptionalInt("sample_size", 1);
                net.Multinomial(node.Name, node.Input0, samples, node.Seed);
            });

            // Reduce ops
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

        private string ResolveLstmOutputName(ONNXNodeWrapper node)
        {
            var baseLSTMOutputName = "recurrent_out";
            if (lstmOutputs.ContainsKey(node.Name))
            {
                var actualName = lstmOutputs[node.Name];
                if (actualName.EndsWith(":0"))
                    actualName = actualName.Substring(0, actualName.Length - 2);

                if (actualName.EndsWith("_h") || actualName.EndsWith("_c"))
                    baseLSTMOutputName = actualName.Substring(0, actualName.Length - 2);
                else
                    baseLSTMOutputName = actualName;
            }

            return baseLSTMOutputName;
        }

        private string ResolveLstmInputName(ONNXNodeWrapper node)
        {
            var baseLSTMName = "recurrent_in";
            if (lstmInputs.ContainsKey(node.Name))
            {
                var actualName = lstmInputs[node.Name];
                if (actualName.EndsWith(":0"))
                    actualName = actualName.Substring(0, actualName.Length - 2);

                if (actualName.EndsWith("_h") || actualName.EndsWith("_c"))
                    baseLSTMName = actualName.Substring(0, actualName.Length - 2);
                else
                    baseLSTMName = actualName;
            }

            return baseLSTMName;
        }

        // TODO: move to commonly used utility funcs
        internal static bool IsEmpty(object o)
        {
            if (o == null)
                return true;

            if (o is string)
                return o as string == "";

            return false;
        }

        // Fuse training time BatchNorm tensors into Scale & Bias
        internal static Tuple<Tensor, Tensor> FuseBatchNormWeights(Tensor gamma, Tensor beta, Tensor mean, Tensor variance, float epsilon, int maxElements = -1)
        {
            // https://github.com/Tencent/ncnn/blob/master/src/layer/batchnorm.cpp
            // float sqrt_var = sqrt(var_data[i]);
            // a_data[i] = bias_data[i] - slope_data[i] * mean_data[i] / sqrt_var;
            // b_data[i] = slope_data[i] / sqrt_var;
            // ...
            // ptr[i] = b * ptr[i] + a;
            Assert.IsTrue(gamma.channels == gamma.length); // assert 1d tensor
            Assert.IsTrue(gamma.shape == beta.shape);
            Assert.IsTrue(gamma.shape == mean.shape);
            Assert.IsTrue(gamma.shape == variance.shape);
            if (maxElements <= 0 || gamma.length < maxElements) // clip to the smallest valid number of channels
                maxElements = gamma.length;
            Tensor scale = new Tensor(1, maxElements);
            Tensor bias = new Tensor(1, maxElements);
            for (int i = 0; i < maxElements; ++i)
            {
                scale[i] = gamma[i] / Mathf.Sqrt(variance[i] + epsilon);
                bias[i] = beta[i] - gamma[i] * mean[i] / Mathf.Sqrt(variance[i] + epsilon);
            }
            return Tuple.Create(scale, bias);
        }

        // Transpose channels first to channels last data in MatMul/GEMM weight tensor
        internal static Tensor SwapSpatialDimensionsAndFeaturesInMatMulWeights(Tensor weights, int featureCount, VariableTensor.Layout layout)
        {
            Assert.IsTrue(featureCount <= weights.flatHeight);

            var weightsAssumeChannelsFirstLayout = (layout != VariableTensor.Layout.ChannelsLast);
            if (featureCount != weights.flatHeight && weightsAssumeChannelsFirstLayout)
            {
                var shape = weights.shape;
                var implicitSpatialDimensionsInWeights = shape.flatHeight / featureCount;
                Assert.IsTrue(shape.flatHeight % featureCount == 0);
                // reshape: __C____K -> __C__HWK
                weights = weights.Reshape(
                    new TensorShape(featureCount, implicitSpatialDimensionsInWeights, 1, shape.channels));
                // permute: __C__HWK -> __H__WCK
                var permutations =
                    TensorExtensions.Get8DPermutationsForNHWCPermutationsAndShape(weights.shape, new int[] {1, 0, 2, 3});
                weights = ONNXTensor.Permute(weights, permutations);
                // reshape: __H__WCK -> __C____K
                weights = weights.Reshape(shape);
            }
            return weights;
        }

        internal static Model PatchFromIncorrectlyAssumedChannelsFirstToChannelsLastLayoutUpstream(Model model, List<string> layerRequiringUpstreamPatch)
        {
            HashSet<int> patchedInputIndices = new HashSet<int>();
            HashSet<string> patchedLayerAxis = new HashSet<string>();

            var inputIndexByName = new Dictionary<string, int>();
            for (var i = 0; i < model.inputs.Count; ++i)
                inputIndexByName.Add(model.inputs[i].name, i);

            // NOTE: although original input had NHWC layout
            // (most probably exported from Tensorflow without '--inputs-as-nchw' flag)
            // earlier when parsing input and axis we made incorrect assumption that they were NCHW
            // now we need to revert that assumption!
            foreach (var rootNodeForPatch in layerRequiringUpstreamPatch)
            {
                int inputIndex = -1;
                var upstream = ModelAnalyzer.FindUpstreamLayers(model, new[] {rootNodeForPatch});
                foreach (var layer in upstream)
                {
                    // patch axis
                    if (!patchedLayerAxis.Contains(layer.name) && (
                        layer.type == Layer.Type.Concat ||
                        layer.type == Layer.Type.Gather ||
                        layer.type == Layer.Type.TopKValues))//TODO handle ReduceXX and StridedSlice
                    {
                        patchedLayerAxis.Add(layer.name);
                        if (layer.axis == 6) layer.axis = TensorShape.C;
                        else if (layer.axis == TensorShape.C) layer.axis = 6;
                    }
                    //patch inputs
                    foreach (var inputName in layer.inputs)
                    {
                        if (inputIndexByName.TryGetValue(inputName, out inputIndex) &&
                            !patchedInputIndices.Contains(inputIndex))
                        {
                            // example (NCHW): -1,2,2,16 -> (incorrect) -1,2,16,2 -> (fix) -1,2,2,16
                            // example  (NCW): -1,2,16   -> (incorrect) -1,1,16,2 -> (fix) -1,1,2,16
                            patchedInputIndices.Add(inputIndex);
                            var inputDesc = model.inputs[inputIndex];
                            inputDesc.shape = ONNXLayout.Permute(inputDesc.shape, new[] {-1, -1, 2, -1, -1, 7, 5, 6});
                            model.inputs[inputIndex] = inputDesc;
                        }
                    }
                    // @TODO: figure out, if there is any case where we would have to propagate fixed layout assumption downstream?
                }
            }

            return model;
        }

        internal static Tensor DilateKernel(Tensor kernel, int[] dilations)
        {
            Assert.IsTrue(dilations.Length == 2);
            Assert.IsTrue(dilations[0] > 0);
            Assert.IsTrue(dilations[1] > 0);

            // https://arxiv.org/pdf/1603.07285.pdf
            Tensor dilatedKernel = new Tensor(new TensorShape(kernel.shape.kernelHeight + (kernel.shape.kernelHeight - 1) * (dilations[0] - 1),
                                                              kernel.shape.kernelWidth  + (kernel.shape.kernelWidth - 1)  * (dilations[1] - 1),
                                                              kernel.shape.kernelDepth, kernel.shape.kernelCount));

            for (int c = 0; c < dilatedKernel.kernelDepth; ++c)
                for (int k = 0; k < dilatedKernel.kernelCount; ++k)
                    {
                        for (int y = 0; y < kernel.shape.kernelHeight; ++y)
                            for (int x = 0; x < kernel.shape.kernelWidth; ++x)
                            {
                                int ox = x * dilations[0];
                                int oy = y * dilations[1];

                                int strideX = x == (kernel.shape.kernelWidth - 1)  ? 1 : dilations[0];
                                int strideY = y == (kernel.shape.kernelHeight - 1) ? 1 : dilations[1];

                                for (int dx = 0; dx < strideX; dx++)
                                    for (int dy = 0; dy < strideY; dy++)
                                    {
                                        dilatedKernel[oy + dy, ox + dx, c, k] = 0.0f;
                                    }

                                float v = kernel[y, x, c, k];
                                dilatedKernel[oy, ox, c, k] = v;
                        }
                }

            return dilatedKernel;
        }

        internal static TensorShape Bias(TensorShape shape)
        {
            return new TensorShape(1, 1, 1, shape.channels);
        }

        internal static bool IsModeBilinear(ModelBuilder net, ONNXNodeWrapper node, string mode)
        {
            bool bilinear = false;
            if (mode == "linear" || mode == "bilinear")
                bilinear = true;
            else if (mode != "nearest")
                Warn(net, node, $"Mode `{mode}` is not supported for type {node.OperatorType}.");

            return bilinear;
        }

        internal static void Resize2D(ModelBuilder net, ONNXNodeWrapper node, float[] scales, string mode)
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
                net.Upsample2D(node.Name, node.Input0, scalesRoundedToInt, IsModeBilinear(net, node, mode));
            }
        }

        private static int[] GetPermutationToMatchReduceWithDroppedDimensionsFromONNX(int[] droppedONNXAxis, int rank)
        {
            Assert.IsTrue(droppedONNXAxis.Length>0);

            //Barracuda always have all dimensions, however in ONNX it is not the case one can drop dimensions,
            //Here we handle the case of ReduceXXX ops when they do so.
            //An example:
            //ONNX -> NCHW
            //Reduce on C with keepDims=False.
            //ONNX -> NHW
            //However ONNX tensor semantic are deducted by position to be mapped to Barracuda in the following way:
            //ONNX 1D -> N    -> Barracuda N,1,1,1
            //ONNX 2D -> NC   -> Barracuda N,1,1,C
            //ONNX 3D -> NCW  -> Barracuda N,1,W,C
            //ONNX 4D -> NCHW -> Barracuda N,H,W,C
            //Thus the output tensor above (NHW) will be mapped to N,1,W,C in Barracuda
            //while Reduce in Barracuda would rather output N,H,W,1 if keepDim would be true.
            //Here we find the transpose needed in Barracuda to match the ONNX behavior as seen by Barracuda.
            //ie the transpose from N,H,W,1 to N,1,W,C in this case aka 0,3,2,1.

            //ONNX input Layout from rank
            string onnxLayout;
            switch (rank)
            {
                case 1: onnxLayout = "N";
                    break;
                case 2: onnxLayout = "NC";
                    break;
                case 3: onnxLayout = "NCW";
                    break;
                case 4: onnxLayout = "NCHW";
                    break;
                default:
                    //TODO support 8D
                    throw new OnnxLayerImportException($"Reduce ops support up to 4D at the moment, however received an input of rank {rank}.");
            }

            //ONNX Layout once dimensions are dropped (example: NHW if C was dropped)
            string onnxLayoutDimensionsDropped = onnxLayout;
            foreach (var axis in droppedONNXAxis)
            {
                var onnxAxis = axis;
                if (onnxAxis < 0)
                    onnxAxis = rank + axis;
                string semanticToRemove = onnxLayout[onnxAxis].ToString();
                onnxLayoutDimensionsDropped = onnxLayoutDimensionsDropped.Replace(semanticToRemove, string.Empty);
            }
            Assert.IsTrue(onnxLayoutDimensionsDropped.Length>0);

            //Find all missing dimensions that will be unitary in Barracuda
            var missingDimensions = new List<char>();
            foreach (var dim in "NHWC")
            {
                if (!onnxLayoutDimensionsDropped.Contains(dim))
                    missingDimensions.Add(dim);
            }

            //Find semantic of onnx layout with dropped dimension in Barracuda
            var barracudaSemanticLayoutFromONNXReduce = new char[4];
            switch (onnxLayoutDimensionsDropped.Length)
            {
                case 1:
                    //ONNX 1D -> N -> Barracuda N,1,1,1
                    barracudaSemanticLayoutFromONNXReduce[0] = onnxLayoutDimensionsDropped[0];
                    barracudaSemanticLayoutFromONNXReduce[1] = missingDimensions[0];
                    barracudaSemanticLayoutFromONNXReduce[2] = missingDimensions[1];
                    barracudaSemanticLayoutFromONNXReduce[3] = missingDimensions[2];
                    break;
                case 2:
                    //ONNX 2D -> NC -> Barracuda N,1,1,C
                    barracudaSemanticLayoutFromONNXReduce[0] = onnxLayoutDimensionsDropped[0];
                    barracudaSemanticLayoutFromONNXReduce[1] = missingDimensions[0];
                    barracudaSemanticLayoutFromONNXReduce[2] = missingDimensions[1];
                    barracudaSemanticLayoutFromONNXReduce[3] = onnxLayoutDimensionsDropped[1];
                    break;
                case 3:
                    //3D -> NCW -> Barracuda N,1,W,C
                    barracudaSemanticLayoutFromONNXReduce[0] = onnxLayoutDimensionsDropped[0];
                    barracudaSemanticLayoutFromONNXReduce[1] = missingDimensions[0];
                    barracudaSemanticLayoutFromONNXReduce[2] = onnxLayoutDimensionsDropped[2];
                    barracudaSemanticLayoutFromONNXReduce[3] = onnxLayoutDimensionsDropped[1];
                    break;
            }

            //Find permutation from NHWC Barracuda layout when mapped from ONNX with dropped dimensions.
            var permutation = new int[4];
            for(int idTarget = 0; idTarget<permutation.Length; ++idTarget)
            {
                char semantic = barracudaSemanticLayoutFromONNXReduce[idTarget];
                permutation[idTarget] = "NHWC".IndexOf(semantic);;
            }
            return permutation;
        }

        internal void Reduce(ModelBuilder net, ONNXNodeWrapper node, Layer.Type reduceType)
        {
            var keepdims = node.GetOptionalInt("keepdims", 1);

            var features = node.Input0Features;
            var rank = node.Input0Rank;
            object input = node.Input0;

            var axes = node.AxesOptional(new[] { 0 });
            foreach (var onnxAxis in axes)
            {
                //TODO support 8D inputs
                var axis = ONNXLayout.ConvertAxisToBarracuda(onnxAxis, onnxRank: rank, onnxLayout: "NCHW");
                var nameR = $"{node.Name}__axis{axis}";
                input = net.Reduce(reduceType, nameR, input, axis, true);
                if (axis == TensorShape.C)
                    features = 1; // this operation collapse all features to 1
                Output(nameR, features: features, rank: rank);
            }

            if (keepdims != 1 && rank > 1) // keepdims removes dimensions in the context of onnx thus we need to repack/transpose to match behavior.
            {
                var nameT = $"{node.Name}__transpose";
                var transpose = GetPermutationToMatchReduceWithDroppedDimensionsFromONNX(axes, rank);
                input = net.Transpose(nameT, input, transpose);

                Output(nameT, features: -1, rank: rank - node.Axes.Length);
            }

            net.Identity(node.Name, input);
        }

        private ONNXModelTensors m_ModelTensors = new ONNXModelTensors();
        private readonly Dictionary<string, Action<ModelBuilder, ONNXNodeWrapper>> m_NodeImporters =
            new Dictionary<string, Action<ModelBuilder, ONNXNodeWrapper>>();

        private Model ConvertOnnxModel(ModelProto onnxModel)
        {
            var model = new Model();
            var modelBuilder = new ModelBuilder(model);

            // Builds list of nodes that should not be included into the final Barracuda Model, mostly for LSTMs
            var nodesToSkip = BuildNodeSkipList(onnxModel.Graph);

            // Convert graph inputs & outputs
            var initializersByName = onnxModel.Graph.Initializer.ToDictionary(i => i.Name, i => true);
            foreach (ValueInfoProto i in onnxModel.Graph.Input)
            {
                // skip input tensors that have initializer data, they are constant tensors not global inputs
                // also skip nodes that should be trimmed
                if (initializersByName.ContainsKey(i.Name) || nodesToSkip.Contains(i.Name))
                    continue;

                if (m_OverrideGlobalInputs.ContainsKey(i.Name))
                {
                    Const(i.Name, m_OverrideGlobalInputs[i.Name]);
                    continue;
                }

                modelBuilder.Input(i.Name, ONNXLayout.ConvertSymbolicShapeToBarracuda(i.Type.TensorType.Shape, onnxLayout:"NCHW"));
                var shapeValues = i.Type.TensorType.Shape.Dim.Select(d => d.DimValue).ToArray();
                Output(i.Name, onnxShape: shapeValues, onnxLayout:"NCHW");
            }
            foreach (ValueInfoProto o in onnxModel.Graph.Output)
                modelBuilder.Output(o.Name);

            // Read constants from initializer list
            foreach (TensorProto initializer in onnxModel.Graph.Initializer)
                Const(initializer.Name, new ONNXTensor(initializer));

            // Convert graph nodes
            foreach (NodeProto onnxNode in onnxModel.Graph.Node)
            {
                if (nodesToSkip.Contains(ONNXNodeWrapper.GetName(onnxNode)))
                    continue;

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
                            var bakedTensor = BakeNodeIntoConstant(opType, node);
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
                        string message = $"Unexpected error while parsing layer {nodeId} of type {opType}.";
                        Err(model, nodeId, message,
                            extendedMessage:"Will replace it by an Identity layer.",
                            debugMessage:$"{e.Message}\n\nJson: {onnxNode}\n{e.StackTrace}\n");
                        injectDummy = true;
                    }
                }
                else
                {
                    // We don't support this type of layer
                    // We log the problem and insert an identity layer
                    string message = $"Unknown type {opType} encountered while parsing layer {nodeId}.";
                    Err(model, nodeId, message, extendedMessage:"Will replace it by an Identity layer.");
                    injectDummy = true;
                }

                if (injectDummy)
                {
                    var originalLayerHadInputs = (node.InputCount > 0);
                    if (originalLayerHadInputs)
                    {
                        var originalLayerHadConstantInput = node.IsInput0Const;
                        if (originalLayerHadConstantInput)
                            Const(nodeId, constantTensors[node.Input0]); // copy constant
                        else
                            modelBuilder.Identity(nodeId, node.Input0);
                    }
                    else // if errorneous layer had no inputs, inject dummy constant which does not require any inputs
                        modelBuilder.Const(nodeId, new Tensor());
                }

                m_ModelTensors.CompleteUninitializedFields(node);
            }

            // Convert constant tensors
            var requiredConstants = new HashSet<string>(ModelAnalyzer.FindBrokenLinks(model));
            // ML-Agents metadata is stored in otherwise unreferenced constants
            var unreferencedConstantsContainMLAgentsMetadata = UnreferencedNodes(onnxModel.Graph);
            requiredConstants.UnionWith(unreferencedConstantsContainMLAgentsMetadata); // keep ML-Agents metadata
            int insertionIndex = 0; // insert constants at the beginning of the model
            foreach(var entry in constantTensors)
                if (requiredConstants.Contains(entry.Key)) // skip if constant is unused
                    modelBuilder.Const(entry.Key, entry.Value.ToBarracuda(onnxLayout:GetONNXLayoutForConstant(model, entry.Key)),
                        insertionIndex++);

            model = ModelOptimizer.Optimize(model, allowFusing: optimizeModel, keepLayers:requiredConstants); // keep ML-Agents metadata

            model = FixReshapeTransposePatternWhenChannelsAreSplitIntoMultipleDimensions(model);

            model = PatchFromIncorrectlyAssumedChannelsFirstToChannelsLastLayoutUpstream(model, layerRequiringUpstreamPatch);

            // strip :0 at the end of string name for TF import
            if (fixTF2ONNXExportIssues)
                model = TrimTensorflowNames(model);

            Validate(model);

            // Parse meta data
            var irVersion = onnxModel.IrVersion; // legacy
            if (onnxModel.OpsetImport?.Count > 0)
                irVersion = onnxModel.OpsetImport[0].Version;
            model.ProducerName = $"{onnxModel.ProducerName} v{onnxModel.ProducerVersion}";
            model.IrSource = "ONNX";
            model.IrVersion = $"{irVersion}";

            return model;
        }

        private bool IsLayerInputChannelDependant(Layer.Type opType, int index)
        {
            return index == 0 ||               //First input is usually channel order dependants
                   opType == Layer.Type.Add || //however some operator have all input channel dependants
                   opType == Layer.Type.Sub ||
                   opType == Layer.Type.Mul ||
                   opType == Layer.Type.Div ||
                   opType == Layer.Type.Pow ||
                   opType == Layer.Type.Min ||
                   opType == Layer.Type.Max ||
                   opType == Layer.Type.Mean ||
                   opType == Layer.Type.Greater ||
                   opType == Layer.Type.GreaterEqual ||
                   opType == Layer.Type.Less ||
                   opType == Layer.Type.LessEqual ||
                   opType == Layer.Type.Equal ||
                   opType == Layer.Type.LogicalOr ||
                   opType == Layer.Type.LogicalAnd ||
                   opType == Layer.Type.LogicalXor ||
                   opType == Layer.Type.Where ||
                   opType == Layer.Type.Concat;
        }

        private string GetONNXLayoutForConstant(Model model, string nodeName)
        {
            int constLayoutRequestCount = 0;
            int nctdhwRequestCount = 0;

            //find all layer using that constant as an input.
            foreach (var l in model.layers)
            {
                for (int i = 0; i < l.inputs.Length; ++i)
                {
                    if (l.inputs[i] == nodeName)
                    {
                        if (IsLayerInputChannelDependant(l.type, i))
                            ++nctdhwRequestCount;
                        else
                            ++constLayoutRequestCount;
                    }
                }
            }

            if (nctdhwRequestCount != 0 && constLayoutRequestCount != 0)
            {
                Err(model, nodeName, $"{nodeName} is both used as channel order dependant constant and a plain constant, this is not supported at the moment.");
            }

            return nctdhwRequestCount>constLayoutRequestCount?"NCTDHW":"CONST";
        }

        private ONNXTensor BakeNodeIntoConstant(string opType, ONNXNodeWrapper node)
        {
            var model = new Model();
            var net = new ModelBuilder(model);

            // add all inputs as constants
            Assert.IsTrue(node.AreAllInputsConst);
            for (var i = 0; i < node.InputCount; ++i)
            {
                var assumeOnnxLayout = (m_AllInputsChannelFirst.Contains(opType) || i == 0) ? "NCTDHW" : "CONST";
                var input = node.Inputs[i];
                net.Const(input,
                    constantTensors[input].ToBarracuda(assumeOnnxLayout));
            }

            // add node that we are going to bake into the constant
            m_NodeImporters[opType](net, node);

            // bake
            var useCPUforBaking = WorkerFactory.Device.CPU;
            using (var worker = WorkerFactory.CreateWorker(model, useCPUforBaking))
            {
                var bakedConstant = worker.Execute().PeekOutput();

                // convert from Barracuda back into ONNX layout
                var onnxData = ONNXTensor.Permute(bakedConstant, new int[] {0,1,2,7,3,4,5,6}); // S,R,N,T,D,H,W,C (channelLast)-> S,R,N,C,H,W (channelFirst)
                var onnxShape = onnxData.shape.ToArray();

                return new ONNXTensor(onnxData, onnxShape).SqueezeAll();
            }
        }

        static private void Validate(Model model)
        {
            // Model should not contain any broken links in the end
            var unconnectedInputs = ModelAnalyzer.FindBrokenLinks(model);
            Assert.IsTrue(unconnectedInputs.Length == 0);
            if (unconnectedInputs.Length > 0)
            {
                var message = $"Broken links: {string.Join(", ", unconnectedInputs)}";
                Warn(model, "", message);
            }
        }

        private HashSet<string> UnreferencedNodes(GraphProto graph)
        {
            var allNodes = new HashSet<string>();
            var allInputs = new HashSet<string>();
            foreach (var node in graph.Node)
            {
                allNodes.Add(ONNXNodeWrapper.GetName(node));
                foreach (var input in node.Input)
                    allInputs.Add(input);
            }

            // Remove all global output nodes
            foreach (ValueInfoProto o in graph.Output)
                allNodes.Remove(o.Name);

            // Remove all nodes that are referenced by Inputs to get the set of unreferenced ones
            var unreferencedNodes = allNodes;
            unreferencedNodes.ExceptWith(allInputs);
            return unreferencedNodes;
        }

        private void BacktraceNodeInputs(Dictionary<string, NodeProto> nameToNode,
            NodeProto[] startingNodes,
            Action<NodeProto> regularNodeCallback,
            Action<NodeProto> inputNodeCallback)
        {
            HashSet<NodeProto> nodesToCheck = new HashSet<NodeProto>(startingNodes);

            while (nodesToCheck.Count > 0)
            {
                var el = nodesToCheck.First();
                regularNodeCallback(el);
                nodesToCheck.Remove(el);

                if (el.Input.Count > 0)
                {
                    if (nameToNode.ContainsKey(el.Input[0]))
                        nodesToCheck.Add(nameToNode[el.Input[0]]); // regular node
                    else
                        inputNodeCallback(el);
                }
            }
        }

        private HashSet<string> BuildNodeSkipList(GraphProto graph)
        {
            var res = new HashSet<string>();
            var nameToNode = graph.Node.ToDictionary(i => ONNXNodeWrapper.GetName(i), i => i);

            var outputToLSTMNode = new Dictionary<string, string>();

            // Skip all LSTM _h & _c inputs as they will be accessible directly via Model.memories
            foreach (NodeProto onnxNode in graph.Node)
            {
                if (onnxNode.OpType == "LSTM")
                {
                    var lstmNodeName = ONNXNodeWrapper.GetName(onnxNode);
                    BacktraceNodeInputs(
                        nameToNode,
                        new[] {nameToNode[onnxNode.Input[5]], nameToNode[onnxNode.Input[6]]},
                        el => { res.Add(ONNXNodeWrapper.GetName(el)); },
                        el => { lstmInputs[lstmNodeName] = el.Input[0]; res.Add(el.Input[0]);}
                        );

                    outputToLSTMNode[onnxNode.Output[1]] = lstmNodeName; // _h
                    outputToLSTMNode[onnxNode.Output[2]] = lstmNodeName; // _c
                }
            }

            // Also trace from outputs to LSTM nodes to figure out names of the output _h and _c nodes
            foreach (var output in graph.Output)
            {
                if (!nameToNode.ContainsKey(output.Name))
                    continue;

                // As LSTM has 3 outputs and backtracing is done only via output[0]
                // then output[1] and output[2] will be treated as leaf input nodes
                BacktraceNodeInputs(
                    nameToNode,
                    new[] {nameToNode[output.Name]},
                    el => {  },
                    el =>
                    {
                        var inputName = el.Input[0];
                        if (outputToLSTMNode.ContainsKey(inputName))
                        {
                            lstmOutputs[outputToLSTMNode[inputName]] = output.Name;
                        }
                    }
                );
            }

            return res;
        }

        static private Model FixReshapeTransposePatternWhenChannelsAreSplitIntoMultipleDimensions(Model model)
        {
            var transposes = model.layers.Where(l => l.type == Layer.Type.Transpose).ToList();
            foreach (var transposeLayer in transposes)
            {
                var previousLayer = model.layers.Find(l => l.name == transposeLayer.inputs[0]);
                if (previousLayer == null)
                    continue;

                if (previousLayer.type != Layer.Type.Reshape)
                    continue;

                var numChannelsDimension = previousLayer.axis;
                if (numChannelsDimension <= 1)
                    continue;

                //NOTE: See also ConvertReshapeToBarracuda() for mode detail on the problem.
                //In some network like shufflenet and superresolution_cnn and yolov3 a reshape is used
                //before a transpose to split the channels resulting in a tensor with
                //multiple dimension used for channels, this is a problem when importing to
                //barracuda as the semantic of the dimensions are changed and this change the
                //way channel first to channel last conversion should happen. The code below
                //is a limited to support for that.
                Assert.IsTrue(numChannelsDimension == 2);

                var permutationSRNTDHWC = transposeLayer.pool;
                if (permutationSRNTDHWC.Length != 8)
                {
                    Warn(model, transposeLayer.name,
                        $"Expecting a permutation of rank 8 after Reshape '{previousLayer.name}' itself outputting more than one channel dimension. Permutation can't be patched to account for the extra channel dimensions.");
                    continue;
                }

                //Revert channel first to last conversion that was incorrectly done in the context of channels being one dimensional.
                var permute8DChannelFirst = ONNXLayout.ConvertPermutationToLayout(permutationSRNTDHWC, "SRNTDHWC","SRNCTDHW");

                //Find source layout format
                int centerPaddingThatWasAddedInPermutation = transposeLayer.axis;
                Assert.IsTrue(centerPaddingThatWasAddedInPermutation <= 1);
                Assert.IsTrue(centerPaddingThatWasAddedInPermutation >= 0);
                string sourceLayout = (centerPaddingThatWasAddedInPermutation == 0) ? "SRNT12HW" : "SRN1T2HW";

                //TODO/HEURISTIC: We only use semantic when channel and other features are not interleaved during permutations.
                //This is a work around to create the right permutation for the shufflenet(true)/superresolution(false)/yolov3(false),
                //it probably does not generalise well however. In next version of the importer we might need to introduce transposes
                //in channel last mode to generalise fully.
                bool useSemanticToLookupInSourceLayout = !IsPermutationMixingChannelsAndOtherFeatures(sourceLayout, permute8DChannelFirst);

                //Apply channel first to last conversion, considering channels are two dimensional (C1C2).
                int[] permuteSRNTHWC1C2  = ONNXLayout.ConvertPermutationToLayout(permute8DChannelFirst, sourceLayout, "SRNTHW12", useSemanticToLookupInSourceLayout);
                transposeLayer.pool = permuteSRNTHWC1C2;
            }

            return model;
        }

        static private bool IsPermutationMixingChannelsAndOtherFeatures(string layout, int[] permutation)
        {
            //Convention here is that channels are described as numbers, while other features by letters.
            Assert.IsTrue(layout.Length == permutation.Length);
            for (int i = 0; i < permutation.Length; ++i)
            {
                bool sourceIsAChannel = Char.IsNumber(layout[i]);
                bool targetIsAChannel = Char.IsNumber(layout[permutation[i]]);
                if (sourceIsAChannel != targetIsAChannel)
                    return true;
            }
            return false;
        }

        static private Model TrimTensorflowNames(Model model)
        {
            model.inputs   = model.inputs.Select(i   => {
                i.name = TrimTensorflowName(i.name);
                return i;
            }).ToList();

            model.outputs  = model.outputs.Select(o  => {
                return TrimTensorflowName(o);
            }).ToList();

            model.memories = model.memories.Select(m => {
                m.input  = TrimTensorflowName(m.input);
                m.output = TrimTensorflowName(m.output);
                return m;
            }).ToList();

            model.layers   = model.layers.Select(l   => {
                l.name = TrimTensorflowName(l.name);
                for(int i = 0; i < l.datasets.Length; i++)
                    l.datasets[i].name = TrimTensorflowName(l.datasets[i].name);
                for(int i = 0; i < l.inputs.Length; i++)
                    l.inputs[i] = TrimTensorflowName(l.inputs[i]);
                return l;
            }).ToList();

            return model;
        }

        static private string TrimTensorflowName(string name)
        {
            if (name.EndsWith(":0"))
                return name.Remove(name.Length-2);
            return name;
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

        private void Output(ONNXNodeWrapper node, int features = -1, int rank = -1,
            VariableTensor.Layout layout = VariableTensor.Layout.Unknown)
        {
            Output(node.Name, features, rank, layout);
        }
        private void Output(string name, int features = -1, int rank = -1,
            VariableTensor.Layout layout = VariableTensor.Layout.Unknown)
        {
            m_ModelTensors.AddVariable(name, features, rank, layout);
        }
        private void Output(string name, ONNXTensor onnxTensor)
        {
            m_ModelTensors.AddVariable(name, onnxTensor);
        }
        private void Output(string name, long[] onnxShape, string onnxLayout)
        {
            m_ModelTensors.AddVariable(name, onnxShape, onnxLayout);
        }

        private void Output(ONNXNodeWrapper node, int features, string productOfShape)
        {
            m_ModelTensors.AddVariable(node.Name, features, productOfShape);
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

        private void Err(Model model, string layerName, string message, string extendedMessage = "", string debugMessage = "")
        {
            if (treatErrorsAsWarnings)
            {
                model.Warnings.Add(new Model.ImporterWarning(layerName,$"{message} {extendedMessage}"));
                Debug.LogWarning($"{message} {extendedMessage}\n{debugMessage}");
            }
            else
                throw new OnnxImportException($"{message}\n{debugMessage}");
        }
    }

    /// <summary>
    /// ONNX import exception
    /// </summary>
    public class OnnxImportException : Exception
    {
        /// <summary>
        /// Create `OnnxImportException`
        /// </summary>
        /// <param name="message">message</param>
        public OnnxImportException(string message) : base(message) { }
    }

    /// <summary>
    /// ONNX layer import exception
    /// </summary>
    public class OnnxLayerImportException : Exception
    {
        /// <summary>
        /// Create `OnnxLayerImportException`
        /// </summary>
        /// <param name="message">message</param>
        public OnnxLayerImportException(string message) : base(message) { }
    }
}
