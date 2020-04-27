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

namespace Unity.Barracuda
{
    /// <summary>
    /// Asset Importer for Open Neural Network Exchange (ONNX) files.
    /// For more information about ONNX file format see: https://github.com/onnx/onnx
    /// </summary>
    [ScriptedImporter(7, new[] { "onnx" })]
    public class ONNXModelImporter : ScriptedImporter
    {
        // Configuration
        public bool optimizeModel = true;
        public bool forceArbitraryBatchSize = true;
        public bool treatErrorsAsWarnings = false;

        // TF2ONNX known issue: (as of 1.5.4)
        // - Conv are framed with Transposes as long as the NCHW flag is not set
        //      (note this seems that it's going to be fixed https://github.com/onnx/tensorflow-onnx/pull/796)
        // - Tensorflow appends :0 to all node names
        bool fixTF2ONNXExportIssues = false;

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
        private Dictionary<string, string> lstmInputs = new Dictionary<string, string>();
        private Dictionary<string, string> lstmOutputs = new Dictionary<string, string>();
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
                {
                    onnxShape = node.Input1Constant(onnxLayout:"C", name:"shape").AsLongs();
                    var onnxRank = onnxShape.Length;

                    var shapeLike = variableTensors[node.Input1].productOfShape;
                    if (!IsEmpty(shapeLike) &&
                        variableTensors[shapeLike].rank == onnxRank &&
                        ONNXLayout.CanSymbolicShapeBeUsedWithReshapeLike(onnxShape, node.Input0Features))
                    {
                        // special case of Shape followed by Reshape
                        net.Reshape(node.Name, node.Input0, shapeLike);
                        Output(node, rank:onnxShape.Length); // stop propagating productOfShape further
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
                    var symbolicShape = ONNXLayout.ConvertSymbolicShapeToBarracuda(onnxShape, "NCHW");
                    bool containsNoVariableDimensions = Array.IndexOf(symbolicShape, -1) == -1;
                    if (containsNoVariableDimensions && forceArbitraryBatchSize)
                        symbolicShape[0] = -1; // force arbitrary batch size
                    net.Reshape(node.Name, node.Input0, symbolicShape);

                    // Change temporary reverted
                    //Output(node, symbolicShape[3], rank:symbolicShape.Length);
                    //if (symbolicShape[3] == -1)
                    //    Warn(net, node, $"Reshape with unknown feature count is not supported at the moment.");
                    Output(node, rank:symbolicShape.Length);
                }
            });
            Add("Shape", (net, node)    => {
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
                            // meanwhile at import time we assume -1 for the spatial dimensions
                            // NOTE: this assumption works for common Upsample opset=9 case:
                            //     Upsample.scales = (shape.hw * constant) / shape.hw
                            // however this would not work for potential (opset=10) cases like:
                            //     Resize.size = shape.hw + constant

                            // stored in ONNX layout
                            var shapeWithChannelsFirst = new[] { -1f, (float)node.Input0Features }; // NC
                            var fillSpatialDimensionsWithUnknown = -1f;
                            var numberOfSpatialDimensions = node.Input0Rank - 2;
                            var shapeFollowedWithSpatialDimensions = Enumerable.Repeat(fillSpatialDimensionsWithUnknown, numberOfSpatialDimensions); // fill with -1
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
                Debug.Assert(shapeLength == node.Input0Rank);

                var shapeTensor = new ONNXTensor(
                    // NOTE: stored in single rank ONNX layout
                    // with data in the 1st dimension
                    // thus `shapeLength` specifies the length of the 1st dimension
                    data:new Tensor(shapeLength, 0, shapeValuesAsFloats),
                    onnxShape:new [] { (long)shapeLength });

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
                    // pad slicing indices to Barracuda format, 4 dimensions are always expected
                    // since values are indices will pad with 0 (in case of TensorShape padding would be 1)
                    Array.Resize(ref onnxStarts, 4);
                    Array.Resize(ref onnxEnds, 4);
                    Array.Resize(ref onnxSteps, 4);

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

                var defaultOffOn = new Tensor(2, 0, new float[] {0, 1});

                var depth = (int)node.Input1Constant(onnxLayout:"C", name:"depth")[0];
                var offon = node.Input2ConstantOptional(defaultOffOn, onnxLayout:"C", name:"values");
                net.OneHot(node.Name, node.Input0, depth, (int)offon[1], (int)offon[0]);
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
                net.MaxPool2D(node.Name, node.Input0, node.KernelShape, node.Strides, node.Pads);
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

                Resize2D(net, node, node.Scales, node.ModeOptional("nearest"));
            });
            Add("Transpose", (net, node) =>
            {
                // From https://github.com/onnx/onnx/blob/master/docs/Operators.md#transpose
                // By default, reverse the dimensions, otherwise permute the axes according to the values given.
                var defaultPermutations = new[] {3, 2, 1, 0};
                var permutations = node.GetOptionalIntArray("perm", defaultPermutations);

                if (node.IsInput0Const)
                {
                    var transposedTensor = constantTensors[node.Input0].Permute(permutations);
                    Const(node, transposedTensor);
                }
                else
                {
                    if (Enumerable.SequenceEqual(permutations, new[] {0, 3, 1, 2}) || // NHWC -> NCHW
                        Enumerable.SequenceEqual(permutations, new[] {0, 2, 3, 1}) && // NCHW -> NHWC
                        fixTF2ONNXExportIssues
                        )
                    {
                        // @TODO: reorder uptream nodes and global input dimensions accordingly from NHWC -> NCHW
                        net.Identity(node.Name, node.Input0);

                        if (permutations[1] == 3)       // NHWC -> NCHW
                            Output(node, layout:VariableTensor.Layout.ChannelsFirst);
                        else if (permutations[1] == 2)  // NCHW -> NHWC
                        {
                            Output(node, layout:VariableTensor.Layout.ChannelsLast);
                            PatchUpstreamInputLayoutFromIncorrectlyAssumedChannelsFirstToChannelsLast(net, node);
                        }
                        else
                            Debug.Assert("Reached unexpected branch" == "");
                    }
                    else if (Enumerable.SequenceEqual(permutations, new[] {0, 2, 1}))       // NWC <-> NCW
                    {
                        // @TODO: reorder uptream nodes and global input dimensions accordingly from NHWC -> NCHW
                        Warn(net, node, $"Use '--inputs-as-nchw' flag when exporting model from Tensorflow with tf2onnx");
                        net.Identity(node.Name, node.Input0);

                        // flip layout
                        if (node.Input0Layout == VariableTensor.Layout.ChannelsLast)
                            Output(node, layout:VariableTensor.Layout.ChannelsFirst);
                        else
                        {
                            Output(node, layout:VariableTensor.Layout.ChannelsLast);
                            PatchUpstreamInputLayoutFromIncorrectlyAssumedChannelsFirstToChannelsLast(net, node);
                        }
                    }
                    else if (Enumerable.SequenceEqual(permutations, new[] {1, 0, 2})) // batch <-> seq_length
                    {
                        net.Identity(node.Name, node.Input0);
                    }
                    else
                        throw new OnnxLayerImportException(
                            $"Currently only constant inputs for node of type {node.OperatorType} are supported. Instead input of {node.Name} is pointing to non-constant node {node.Input0}.");
                }
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
                var weights = node.Input1Constant(onnxLayout:"CK", name:"B");
                var biases  = node.DefaultTensor(Bias(weights.shape), 0.0f);
                // Change data layout from "channels first" to "channels last"
                weights = SwapSpatialDimensionsAndFeaturesInMatMulWeights(weights, node.Input0Features, node.Input0Layout);
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
                    scale = new Tensor(1, node.Input0Features, scale.readonlyArray);
                    bias = new Tensor(1, node.Input0Features, bias.readonlyArray);
                }
                net.Normalization(node.Name, node.Input0, scale, bias, node.EpsilonOptional());
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
            Debug.Assert(gamma.channels == gamma.length); // assert 1d tensor
            Debug.Assert(gamma.shape == beta.shape);
            Debug.Assert(gamma.shape == mean.shape);
            Debug.Assert(gamma.shape == variance.shape);
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
            Debug.Assert(featureCount <= weights.flatHeight);

            var weightsAssumeChannelsFirstLayout = (layout != VariableTensor.Layout.ChannelsLast);
            if (featureCount != weights.flatHeight && weightsAssumeChannelsFirstLayout)
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

        internal static void PatchUpstreamInputLayoutFromIncorrectlyAssumedChannelsFirstToChannelsLast(ModelBuilder net, ONNXNodeWrapper node)
        {
            var model = net.model;
            var inputIndexByName = new Dictionary<string, int>();
            for (var i = 0; i < model.inputs.Count; ++i)
                inputIndexByName.Add(model.inputs[i].name, i);

            int inputIndex = -1;
            var upstream = ModelAnalyzer.FindUpstreamLayers(model, new[] {node.Name});
            foreach (var layer in upstream)
                foreach (var inputName in layer.inputs)
                    if (inputIndexByName.TryGetValue(inputName, out inputIndex))
                    {
                        var inputDesc = model.inputs[inputIndex];
                        if (inputDesc.shape.Length == 3 || inputDesc.shape.Length == 4)
                        {
                            // NOTE: although original input had NHWC layout
                            // (most probably exported from Tensorflow without '--inputs-as-nchw' flag)
                            // earlier when parsing inputs we made incorrect assumption that this input is NCHW
                            // now we need to revert that assumption!

                            // example (NCHW): -1,2,2,16 -> (incorrect) -1,2,16,2 -> (fix) -1,2,2,16
                            // example  (NCW): -1,2,16   -> (incorrect) -1,1,16,2 -> (fix) -1,1,2,16
                            inputDesc.shape = ONNXLayout.Permute(inputDesc.shape, new[] {0,1,3,2});
                            model.inputs[inputIndex] = inputDesc;

                            // @TODO: figure out, if there is any case where we would have to propogate fixed layout assumption downstream?
                        }
                        else if (inputDesc.shape.Length > 4)
                            throw new OnnxLayerImportException(
                                $"Currently only global inputs of rank 4 are supported, instead {inputName} has rank of {inputDesc.shape.Length}");
                    }
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

        internal void Reduce(ModelBuilder net, ONNXNodeWrapper node, Layer.Type reduceType)
        {
            node.UnsupportedAttribute("keepdims", 1);

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

        public override void OnImportAsset(AssetImportContext ctx)
        {
            var onnxModel = new ModelProto();
            using (var readStream = new FileStream(ctx.assetPath, FileMode.Open, FileAccess.Read))
            using (var inputStream = new CodedInputStream(readStream))
                onnxModel.MergeFrom(inputStream);

            fixTF2ONNXExportIssues = (onnxModel.ProducerName == "tf2onnx");
            if (fixTF2ONNXExportIssues)
                D.Log("Hot fix for known tf2onnx issues");

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
                Output(i.Name, onnxShape:i.Type.TensorType.Shape.Dim.Select(d => d.DimValue).ToArray(), onnxLayout:"NCHW");
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
                    modelBuilder.Const(entry.Key, entry.Value.ToBarracuda(onnxLayout: "CONST"),
                        insertionIndex++);

            model = ModelOptimizer.Optimize(model, allowFusing: optimizeModel, keepLayers:requiredConstants); // keep ML-Agents metadata

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

        static private void Validate(Model model)
        {
            // Model should not contain any broken links in the end
            var unconnectedInputs = ModelAnalyzer.FindBrokenLinks(model);
            Debug.Assert(unconnectedInputs.Length == 0);
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

        // Icon helper
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

    public class OnnxImportException : Exception
    {
        public OnnxImportException(string message) : base(message) { }
    }

    public class OnnxLayerImportException : Exception
    {
        public OnnxLayerImportException(string message) : base(message) { }
    }
}
