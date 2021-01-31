using System;
using System.Collections.Generic;
using System.Linq;

namespace Unity.Barracuda.Compiler.Passes
{
    class IntermediateToRunnableNCHWPass : IModelPass
    {
        readonly BurstCPUOps m_Ops = new BurstCPUOps();

        readonly int[] k_ToNCHW = { 0, 3, 1, 2 };
        readonly int[] k_ToNHWC = { 0, 2, 3, 1 };
        readonly int[] k_FromNCHtoN1WC = { 0, 3, 2, 1 };
        readonly int[] k_FromN1WCtoNCH = { 0, 3, 2, 1 };

        public void Run(ref Model model)
        {
            if (model.layout != "iNCHW")
                return;

            var nchw = model.ShallowCopy();
            nchw.layers.Clear();
            nchw.layout = "NCHW";

            var modelBuilder = new ModelBuilder(nchw);

            var rewriters = new Dictionary<Layer.Type, Func<Layer, ModelBuilder, bool>>();
            var layerRenames = new Dictionary<string, string>();
            var inputRemaps = new Dictionary<string, string>();

            // return true if layer should be included in rewritten model, false if it was replaced
            rewriters.Add(Layer.Type.NonMaxSuppression, (layer, net) =>
            {
                string boxes = layer.inputs[0];
                string scores = layer.inputs[1];

                Layer boxesTransposed = net.Transpose($"Transpose_For_{boxes}", boxes, k_FromNCHtoN1WC);
                Layer scoresTransposed = net.Transpose($"Transpose_For_{scores}", scores, k_FromNCHtoN1WC);

                // Most of the layer stays intact
                string originalLayerName = layer.name;
                layer.name = $"{layer.name}_NHWC";
                layer.inputs[0] = boxesTransposed.name;
                layer.inputs[1] = scoresTransposed.name;
                net.model.layers.Add(layer);

                net.Transpose(originalLayerName, layer.name, k_ToNCHW);

                return false;
            });
            rewriters.Add(Layer.Type.Activation, (layer, net) =>
            {
                if (layer.activation == Layer.Activation.LogSoftmax)
                    return TransposeInput0(layer, net);

                return true;
            });
            // Pad
            rewriters.Add(Layer.Type.Border2D, TransposeInput0);
            rewriters.Add(Layer.Type.Pad2DReflect, TransposeInput0);
            rewriters.Add(Layer.Type.Pad2DEdge, TransposeInput0);

            rewriters.Add(Layer.Type.GlobalAvgPool2D, TransposeInput0);
            rewriters.Add(Layer.Type.GlobalMaxPool2D, TransposeInput0);

            // Upsample
            rewriters.Add(Layer.Type.Upsample2D, (layer, net) =>
            {
                if (layer.inputs.Length > 1)
                    return TransposeInput01(layer, net); // Upsample usage
                else
                    return TransposeInput0(layer, net); // Resize usage
            });
            rewriters.Add(Layer.Type.Upsample3D, TransposeInput01); // Upsample usage
            rewriters.Add(Layer.Type.AvgPool2D, TransposeInput0); // ModelBuilder: Resize2D

            // Resize: could be Resample2D, AvgPool2D, or Upsample2D
            rewriters.Add(Layer.Type.Resample2D, TransposeInput0);

            // Gemm
            rewriters.Add(Layer.Type.Dense, TransposeInput0);

            // Conv
            rewriters.Add(Layer.Type.DepthwiseConv2D, Transpose0UsingRank);
            rewriters.Add(Layer.Type.Conv2D, Transpose0UsingRank);
            rewriters.Add(Layer.Type.Conv3D, Transpose0UsingRank);
            rewriters.Add(Layer.Type.Conv2DTrans, Transpose0UsingRank);

            // BatchNormalization
            rewriters.Add(Layer.Type.ScaleBias, Transpose0UsingRank);

            // InstanceNormalization
            rewriters.Add(Layer.Type.Normalization, Transpose0UsingRank);

            rewriters.Add(Layer.Type.StridedSlice, StridedSlice);
            rewriters.Add(Layer.Type.Gather, AxisToBarracuda);
            rewriters.Add(Layer.Type.Concat, AxisToBarracuda);


            foreach (var l in model.layers)
            {
                if (l.flags.HasFlag(Layer.Flags.Preserve)
                    || !rewriters.TryGetValue(l.type, out Func<Layer, ModelBuilder, bool> rw)
                    || rw(l, modelBuilder))
                {
                    // Either no re-write was needed or the layer was not replaced
                    nchw.layers.Add(l);
                }
            }

            model = nchw;
        }

        bool AxisToBarracuda(Layer layer, ModelBuilder net)
        {
            string input0 = layer.inputs[0];
            Model.Input input0Info = net.model.inputs.First(i => i.name == layer.inputs[0]);

            var onnxRank = input0Info.rank;

            switch (onnxRank)
            {
                case 6:
                    layer.axis += 2;
                    break;
                case 5:
                    layer.axis = layer.axis + (layer.axis == 0 ? 2 : 3);
                    break;
                default:
                    layer.axis = layer.axis + (layer.axis == 0 ? 2 : 4);
                    break;
            }

            return true;
        }

        bool StridedSlice(Layer layer, ModelBuilder net)
        {
            string input0 = layer.inputs[0];
            Model.Input input0Info = net.model.inputs.First(i => i.name == layer.inputs[0]);

            var starts = layer.pad;
            var ends = layer.pool;
            var steps = layer.stride;
            var axes = layer.axes;

            var onnxRank = input0Info.rank;
            var onnxStarts = Enumerable.Repeat(0, onnxRank).ToArray();
            var onnxEnds = Enumerable.Repeat(int.MaxValue, onnxRank).ToArray(); // by default copy the whole axis till the end
            var onnxSteps = Enumerable.Repeat(1, onnxRank).ToArray();

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
                onnxEnds[axis] = ends[i];
                onnxSteps[axis] = steps[i];
            }

            layer.pad = new[] { 0, 0, 0, 0, 0, 0, 0, 0 };// ONNXLayout.PermuteToBarracuda(onnxStarts, onnxLayout: "NCHW", 0);
            layer.pool = new[] { int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue };// ONNXLayout.PermuteToBarracuda(onnxEnds, onnxLayout: "NCHW", int.MaxValue);
            layer.stride = new[] { 1, 1, 1, 1, 1, 1, 1, 1 }; // ONNXLayout.PermuteToBarracuda(onnxSteps, onnxLayout: "NCHW", 1);

            for (int i = 0; i < onnxRank; i++)
            {
                switch (onnxRank)
                {
                    case 6:
                        layer.pad[i + 2] = onnxStarts[i];
                        layer.pool[i + 2] = onnxEnds[i];
                        layer.stride[i + 2] = onnxSteps[i];
                        break;
                    case 5:
                        layer.pad[i + (i == 0 ? 2 : 3)] = onnxStarts[i];
                        layer.pool[i + (i == 0 ? 2 : 3)] = onnxEnds[i];
                        layer.stride[i + (i == 0 ? 2 : 3)] = onnxSteps[i];
                        break;
                    default :
                        layer.pad[i + (i == 0 ? 2 : 4)] = onnxStarts[i];
                        layer.pool[i + (i == 0 ? 2 : 4)] = onnxEnds[i];
                        layer.stride[i + (i == 0 ? 2 : 4)] = onnxSteps[i];
                        break;
                }
            }

            return true;
        }

        bool Transpose0UsingRank(Layer layer, ModelBuilder net)
        {
            string input0 = layer.inputs[0];
            Model.Input input0Info = net.model.inputs.First(i => i.name == layer.inputs[0]);

            Layer input0Transposed = net.Transpose($"Transpose_For_{input0}", input0, input0Info.rank == 3 ? k_FromNCHtoN1WC : k_ToNHWC);

            // Most of the layer stays intact
            string originalLayerName = layer.name;
            layer.name = $"{layer.name}_NHWC";
            layer.inputs[0] = input0Transposed.name;
            net.model.layers.Add(layer);

            net.Transpose(originalLayerName, layer.name, input0Info.rank == 3 ? k_FromN1WCtoNCH : k_ToNCHW);

            return false;
        }

        bool TransposeInput01(Layer layer, ModelBuilder net)
        {
            string input0 = layer.inputs[0];
            string input1 = layer.inputs[1];

            Layer input0Transposed = net.Transpose($"Transpose_For_{input0}", input0, k_ToNHWC);
            Layer input1Transposed = net.Transpose($"Transpose_For_{input1}", input1, k_ToNHWC);
            string originalLayerName = layer.name;
            layer.name = $"{layer.name}_NHWC";
            layer.inputs[0] = input0Transposed.name;
            layer.inputs[1] = input1Transposed.name;
            net.model.layers.Add(layer);

            net.Transpose(originalLayerName, layer.name, k_ToNCHW);

            return false;
        }

        bool TransposeInput0(Layer layer, ModelBuilder net)
        {
            string input0 = layer.inputs[0];

            Layer input0Transposed = net.Transpose($"Transpose_For_{input0}", input0, k_ToNHWC);
            string originalLayerName = layer.name;
            layer.name = $"{layer.name}_NHWC";
            layer.inputs[0] = input0Transposed.name;
            net.model.layers.Add(layer);

            net.Transpose(originalLayerName, layer.name, k_ToNCHW);

            return false;
        }
    }
}
