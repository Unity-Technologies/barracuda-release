using System;
using System.Collections.Generic;
using System.Linq;

namespace Unity.Barracuda.Compiler.Passes
{
    class IntermediateToRunnableNCHWPass : IModelPass
    {
        readonly int[] k_ToNCHW = { 0, 3, 1, 2 };
        readonly int[] k_ToNHWC = { 0, 2, 3, 1 };
        readonly int[] k_FromNCHtoN1WC = { 0, 3, 2, 1 };
        readonly int[] k_FromN1WCtoNCH = { 0, 3, 2, 1 };

        public void Run(ref Model model)
        {
            if (model.layout != "iNCHW")
                return;

            IDictionary<string, int?> ranksByName;
            IDictionary<string, TensorShape?> shapesByName;
            IRShapeInferenceHelper.RankInference.ListTemporaryTensorRanks(model, out ranksByName);
            var inputShapes = new Dictionary<string, TensorShape>();
            foreach (var i in model.inputs)
            {
                if (!ModelAnalyzer.IsInputShapeAcceptablyKnowForShapeInference(i))
                    continue;
                inputShapes[i.name] = new TensorShape(i.shape);
            }

            IRShapeInferenceHelper.ShapeInference.ListTemporaryTensorShapesNCHW(model, inputShapes, ref ranksByName, out shapesByName);

            var nchw = model.ShallowCopy();
            nchw.layers.Clear();
            nchw.layout = "NCHW";

            var modelBuilder = new ModelBuilder(nchw);

            var rewriters = new Dictionary<Layer.Type, Func<Layer, ModelBuilder, bool>>();
            var layerRenames = new Dictionary<string, string>();
            var inputRemaps = new Dictionary<string, string>();

            // return true if layer should be included in rewritten model, false if it was replaced
            rewriters.Add(Layer.Type.Unsqueeze, (layer, net) =>
            {
                if (layer.pool.Length > 1)
                    // Multiple axes unsupported; leave layer as-is
                    return true;

                string input0 = layer.inputs[0];

                if (!shapesByName.TryGetValue(input0, out TensorShape? input0Shape) || !input0Shape.HasValue)
                    throw new Exception($"Must have input shape for {input0} for Unsqueeze");

                if (!ranksByName.TryGetValue(input0, out int? input0Rank) || !input0Rank.HasValue)
                    throw new Exception($"Must have input rank for {input0} for Unsqueeze");

                int rank = input0Rank.Value;

                if (rank >= 4)
                    // Only 4D unsqueezes of rank 3 or less are supported
                    return true;

                int axis = layer.pool[0];
                if (axis < 0)
                    axis = rank + axis;

                int[] shape8D = input0Shape.Value.ToArray(); // 8D
                List<int> shape = new List<int>();
                shape.Add(shape8D[TensorShape.DataBatch]);
                if (rank > 1)
                    shape.Add(shape8D[TensorShape.H]); // C in NCHW
                if (rank > 2)
                    shape.Add(shape8D[TensorShape.W]); // H in NCHW
                shape.Insert(axis, 1);
                shape.AddRange(Enumerable.Repeat(1, 4 - shape.Count));

                net.Reshape(layer.name, input0, shape.ToArray());

                return false;
            });
            rewriters.Add(Layer.Type.Squeeze, (layer, net) =>
            {
                if (layer.pool.Length > 1)
                    // Multiple axes unsupported; leave layer as-is
                    return true;

                string input0 = layer.inputs[0];

                // Replace w/ a Transpose since Barracuda tensors are full rank
                if (!ranksByName.TryGetValue(input0, out int? input0Rank) || !input0Rank.HasValue)
                    throw new Exception($"Must have input rank for {input0} for Squeeze");

                int rank = input0Rank.Value;
                int axis = layer.pool[0];
                if (axis < 0)
                    axis = rank + axis;

                var transpose = SqueezeAxisPermutation(rank, axis);
                net.Transpose(layer.name, input0, transpose);

                return false;
            });
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
            rewriters.Add(Layer.Type.MatMul, TransposeInput01UsingRank);

            // Conv
            rewriters.Add(Layer.Type.DepthwiseConv2D, Transpose0UsingRank);
            rewriters.Add(Layer.Type.Conv2D, Transpose0UsingRank);
            rewriters.Add(Layer.Type.Conv3D, Transpose0UsingRank);
            rewriters.Add(Layer.Type.Conv2DTrans, Transpose0UsingRank);

            // BatchNormalization
            rewriters.Add(Layer.Type.ScaleBias, Transpose0UsingRank);

            // InstanceNormalization
            rewriters.Add(Layer.Type.Normalization, Transpose0UsingRank);

            // broadcastable ops
            rewriters.Add(Layer.Type.Add, TransposeForBroadcast);
            rewriters.Add(Layer.Type.Mul, TransposeForBroadcast);
            rewriters.Add(Layer.Type.Sub, TransposeForBroadcast);
            rewriters.Add(Layer.Type.Div, TransposeForBroadcast);


            rewriters.Add(Layer.Type.StridedSlice, SliceToBarracuda);
            rewriters.Add(Layer.Type.Gather, GatherToBarracuda);
            rewriters.Add(Layer.Type.Concat, AxisToBarracuda);
            rewriters.Add(Layer.Type.Tile, ShapeToBarracuda);
            rewriters.Add(Layer.Type.Reshape, ShapeToBarracuda);
            rewriters.Add(Layer.Type.Transpose, TransposeToBarracuda);
            rewriters.Add(Layer.Type.Expand, (layer, net) =>
            {
                string input0 = layer.inputs[0];
                Model.Input input0Info = net.model.inputs.First(i => i.name == layer.inputs[0]);

                var rank0 = input0Info.rank;
                var size = layer.pool.ToList();

                if (rank0 >= size.Count)
                {
                    for (int i = 0; i < rank0 - size.Count; i++)
                        size.Insert(0, 1);
                    layer.pool = size.ToArray();
                    return ShapeToBarracuda(layer, net);
                }

                // inputShape needs to be unsqueezed
                var transpose = RankChangePermutationBarracuda(rank0, size.Count);
                Layer nchwTranspose = net.Transpose($"Transpose_{input0}_For_{layer.name}", input0, transpose);

                ShapeToBarracuda(layer, net);

                net.Expand(layer.name, nchwTranspose, layer.pool);

                return false;
            });
            rewriters.Add(Layer.Type.OneHot, (layer, net) =>
            {
                string input0 = layer.inputs[0];
                Model.Input input0Info = net.model.inputs.First(i => i.name == layer.inputs[0]);

                Layer input0Transposed = net.Transpose($"Transpose_For_{input0}", input0, k_ToNHWC);

                // Most of the layer stays intact
                string originalLayerName = layer.name;
                layer.name = $"{layer.name}_NHWC";
                layer.inputs[0] = input0Transposed.name;
                layer.axis = input0Info.rank;
                net.model.layers.Add(layer);

                // OneHot outputRank = inputRank + 1
                net.Transpose(originalLayerName, layer.name, input0Info.rank == 2 ? k_FromN1WCtoNCH : k_ToNCHW);

                return false;
            });

            // Reduce
            rewriters.Add(Layer.Type.ReduceL1, AxisToBarracuda);
            rewriters.Add(Layer.Type.ReduceL2, AxisToBarracuda);
            rewriters.Add(Layer.Type.ReduceMax, AxisToBarracuda);
            rewriters.Add(Layer.Type.ReduceMean, AxisToBarracuda);
            rewriters.Add(Layer.Type.ReduceMin, AxisToBarracuda);
            rewriters.Add(Layer.Type.ReduceProd, AxisToBarracuda);
            rewriters.Add(Layer.Type.ReduceSum, AxisToBarracuda);
            rewriters.Add(Layer.Type.ReduceLogSum, AxisToBarracuda);
            rewriters.Add(Layer.Type.ReduceSumSquare, AxisToBarracuda);
            rewriters.Add(Layer.Type.ReduceLogSumExp, AxisToBarracuda);

            foreach (var l in model.layers)
            {
                if (!rewriters.TryGetValue(l.type, out Func<Layer, ModelBuilder, bool> rw) || rw(l, modelBuilder))
                {
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
            if (layer.axis < 0)
                layer.axis += onnxRank;

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

        bool GatherToBarracuda(Layer layer, ModelBuilder net)
        {
            string input0 = layer.inputs[0];
            Model.Input input0Info = net.model.inputs.First(i => i.name == layer.inputs[0]);

            string input1 = layer.inputs[1];
            Model.Input input1Info = net.model.inputs.First(i => i.name == layer.inputs[1]);

            layer.pool = new[] { input0Info.rank, input1Info.rank };

            return AxisToBarracuda(layer, net);
        }

        bool TransposeToBarracuda(Layer layer, ModelBuilder net)
        {
            string input0 = layer.inputs[0];
            Model.Input input0Info = net.model.inputs.First(i => i.name == layer.inputs[0]);

            var onnxTranspose = layer.pool;

            var rank = input0Info.rank;
            switch (rank)
            {
                case 2:
                {
                    // onnx : 5,7 => 5,7,1,1 / 7,5
                    layer.pool = new[] { layer.pool[0], layer.pool[1], 2, 3 };
                    return true;
                }
                case 3:
                {
                    // onnx : 5,7,3 => 5,7,3,1 / 7,5,3,1 / 7,3,5,1 ...
                    layer.pool = new[] { layer.pool[0], layer.pool[1], layer.pool[2], 3 };
                    return true;
                }
                case 4:
                {
                    return true;
                }
                default:
                    throw new ArgumentException($"Unsupported transpose");
            }
        }

        bool ShapeToBarracuda(Layer layer, ModelBuilder net)
        {
             var size = layer.pool;

            // Don't use Tensorshape as this can remove a wild card
            const int _ = 1;
            if (size.Length == 1)
                layer.pool = new[] { _, _, size[0], _, _, 1, 1, 1 }; // [1,1,N,1,1,1,1,1]
            else if (size.Length == 2)
                layer.pool = new[] { _, _, size[0], _, _, size[1], 1, 1 }; // [1,1,N,1,C,1,1,1]
            else if (size.Length == 3)
                layer.pool = new[] { _, _, size[0], _, _, size[1], size[2], 1 }; // [1,1,N,1,1,C,W,1]
            else if (size.Length == 4)
                layer.pool = new[] { _, _, size[0], _, _, size[1], size[2], size[3] }; // [1,1,N,1,1,C,H,W]
            else if (size.Length == 5)
                layer.pool = new[] { _, _, size[0], _, size[1], size[2], size[3], size[4] }; // [1,1,N,1,D,H,W,C]
            else if (size.Length == 6)
                layer.pool = new[] { _, _, size[0], size[1], size[2], size[3], size[4], size[5] }; // [1,1,N,T,D,H,W,C]
            else
                layer.pool = new[] { size[0], size[1], size[2], size[3], size[4], size[5], size[6], size[7] }; // [S,R,N,T,D,H,W,C]

            return true;
        }

        static int[] SqueezeAxisPermutation(int rank, int axis)
        {
            var identity = new[] { 0, 1, 2, 3 };

            if (rank == 5)
            {
                //            axis:    0        1        2        3        4
                // ONNX:      NCDHW    CDHW     NDHW     NCHW     NCDW     NCDH
                // { 0,1,2,3,4,5,6,7}
                //   _,_,N,_,C,D,H,W
                if (axis == 0)
                    return new[] { 0, 1, 4, 3, 5, 6, 7, 2 };
                if (axis == 1)
                    return new[] { 0, 1, 2, 3, 5, 6, 7, 4 };
                if (axis == 2)
                    return new[] { 0, 1, 2, 3, 4, 6, 7, 5 };
                if (axis == 3)
                    return new[] { 0, 1, 2, 3, 4, 5, 7, 6 };

                return new[] { 0, 1, 2, 3, 4, 5, 6, 7 };
            }
            if (rank == 4)
            {
                //            axis:   0       1      2      3
                // ONNX:      NCHW    CHW     NHW    NCW    NCH
                if (axis == 0)
                    return new[] { 1, 2, 3, 0 };
                if (axis == 1)
                    return new[] { 0, 2, 3, 1 };
                if (axis == 2)
                    return new[] { 0, 1, 3, 2 };

                return identity;
            }
            if (rank == 3)
            {
                //            axis:   0       1      2
                // ONNX:      NCH     CH      NH     NC
                if (axis == 0)
                    return new[] { 1, 2, 0, 3 };
                if (axis == 1)
                    return new[] { 0, 2, 1, 3 };

                return identity;
            }
            if (rank == 2)
            {
                //            axis:   0       1
                // ONNX:      NC      C       N
                if (axis == 0)
                    return new[] { 1, 0, 2, 3 };

                return identity;
            }
            if (rank == 1)
                return identity;

            throw new InvalidOperationException($"Not supported Squeeze operation with rank {rank}");
        }

        bool SliceToBarracuda(Layer layer, ModelBuilder net)
        {
            string input0 = layer.inputs[0];
            Model.Input input0Info = net.model.inputs.First(i => i.name == layer.inputs[0]);
            int rank = input0Info.rank;

            var starts = layer.pad;
            var ends = layer.pool;
            var steps = layer.stride;
            var axes = layer.axes;

            var onnxStarts = Enumerable.Repeat(0, rank).ToArray();
            var onnxEnds = Enumerable.Repeat(int.MaxValue, rank).ToArray(); // by default copy the whole axis till the end
            var onnxSteps = Enumerable.Repeat(1, rank).ToArray();

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
                    axis += rank;
                axis = Math.Min(Math.Max(axis, 0), rank);

                onnxStarts[axis] = starts[i];
                onnxEnds[axis] = ends[i];
                onnxSteps[axis] = steps[i];
            }

            switch (rank)
            {
                case 1:
                    layer.pad = new[] { 0, 0, onnxStarts[0], 0, 0, 0, 0, 0 };
                    layer.pool = new[] { int.MaxValue, int.MaxValue, onnxEnds[0], int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue };
                    layer.stride = new[] { 1, 1, onnxSteps[0], 1, 1, 1, 1, 1 };
                    break;
                case 2:
                    layer.pad = new[] { 0, 0, onnxStarts[0], 0, 0, onnxStarts[1], 0, 0 };
                    layer.pool = new[] { int.MaxValue, int.MaxValue, onnxEnds[0], int.MaxValue, int.MaxValue, onnxEnds[1], int.MaxValue, int.MaxValue };
                    layer.stride = new[] { 1, 1, onnxSteps[0], 1, 1, onnxSteps[1], 1, 1 };
                    break;
                case 3:
                    layer.pad = new[] { 0, 0, onnxStarts[0], 0, 0, onnxStarts[1], onnxStarts[2], 0 };
                    layer.pool = new[] { int.MaxValue, int.MaxValue, onnxEnds[0], int.MaxValue, int.MaxValue, onnxEnds[1], onnxEnds[2], int.MaxValue };
                    layer.stride = new[] { 1, 1, onnxSteps[0], 1, 1, onnxSteps[1], onnxSteps[2], 1 };
                    break;
                case 4:
                    layer.pad = new[] { 0, 0, onnxStarts[0], 0, 0, onnxStarts[1], onnxStarts[2], onnxStarts[3] };
                    layer.pool = new[] { int.MaxValue, int.MaxValue, onnxEnds[0], int.MaxValue, int.MaxValue, onnxEnds[1], onnxEnds[2], onnxEnds[3] };
                    layer.stride = new[] { 1, 1, onnxSteps[0], 1, 1, onnxSteps[1], onnxSteps[2], onnxSteps[3] };
                    break;
                default:
                    throw new ArgumentException($"Unsupported tensor rank {rank} for StridedSlice");
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
        bool TransposeInput01UsingRank(Layer layer, ModelBuilder net)
        {
            string input0 = layer.inputs[0];
            Model.Input input0Info = net.model.inputs.First(i => i.name == layer.inputs[0]);

            string input1 = layer.inputs[1];
            Model.Input input1Info = net.model.inputs.First(i => i.name == layer.inputs[1]);

            Layer input0Transposed = net.Transpose($"Transpose_For_{input0}", input0, input0Info.rank == 3 ? k_FromNCHtoN1WC : k_ToNHWC);
            Layer input1Transposed = net.Transpose($"Transpose_For_{input1}", input1, input1Info.rank == 3 ? k_FromNCHtoN1WC : k_ToNHWC);

            string originalLayerName = layer.name;
            layer.name = $"{layer.name}_NHWC";
            layer.inputs[0] = input0Transposed.name;
            layer.inputs[1] = input1Transposed.name;
            net.model.layers.Add(layer);

            net.Transpose(originalLayerName, layer.name, input0Info.rank == 3 ? k_FromN1WCtoNCH : k_ToNCHW);

            return false;
        }

        bool TransposeForBroadcast(Layer layer, ModelBuilder net)
        {
            int maxRankI = 0;
            for(int i = 0; i < layer.inputs.Length; i++)
            {
                Model.Input inputInfo = net.model.inputs.First(x => x.name == layer.inputs[i]);
                maxRankI = Math.Max(maxRankI, inputInfo.rank);
            }

            List<Layer> insertedTranspose = new List<Layer>();
            for (int i = 0; i < layer.inputs.Length; i++)
            {
                string input = layer.inputs[i];
                Model.Input inputInfo = net.model.inputs.First(x => x.name == layer.inputs[i]);
                int inputRank = inputInfo.rank;

                var transpose = GetTransposeForBroadCast(inputRank, maxRankI);
                Layer inputTransposed = net.Transpose($"Transpose_For_{input}", input, transpose);
                insertedTranspose.Add(inputTransposed);
            }

            string originalLayerName = layer.name;
            layer.name = $"{layer.name}_NHWC";
            for (int i = 0; i < layer.inputs.Length; i++)
            {
                layer.inputs[i] = insertedTranspose[i].name;

            }
            net.model.layers.Add(layer);

            net.Transpose(originalLayerName, layer.name, new [] { 0, 1, 2, 3 });

            return false;
        }

        int[] GetTransposeForBroadCast(int rank0, int rank1)
        {
            if (rank0 == rank1)
                return new[] { 0, 1, 2, 3 };

            if (rank1 == 0 || rank1 == 1)
                return new[] { 0, 1, 2, 3 };
            if (rank1 == 2)
            {
                // 3 + 53 => 1,3
                if (rank0 == 0 || rank0 == 1)
                    return new[] { 1, 0, 2, 3 };
                else
                    throw new ArgumentException($"Unsupported rank permutation change {rank0} to {rank1}");
            }
            else if (rank1 == 3)
            {
                // 3 + 753 => 1,1,3
                if (rank0 == 0 || rank0 == 1)
                    return new[] { 1, 2, 0, 3 };
                // 53 + 753 => 1,5,3
                else if (rank0 == 2)
                    return new[] { 2, 0, 1, 3 };
                else
                    throw new ArgumentException($"Unsupported rank permutation change {rank0} to {rank1}");
            }
            else if (rank1 == 4)
            {
                // 3 + 9753 => 1,1,1,3
                if (rank0 == 0 || rank0 == 1)
                    return new[] { 1, 2, 3, 0 };
                // 53 + 9753 => 1,1,5,3
                else if (rank0 == 2)
                    return new[] { 2, 3, 0, 1 };
                // 753 + 9753 => 1,1,5,3
                else if (rank0 == 3)
                    return new[] { 3, 0, 1, 2 };
                else
                    throw new ArgumentException($"Unsupported rank permutation change {rank0} to {rank1}");
            }
            else
                throw new ArgumentException($"Unsupported rank permutation change {rank0} to {rank1}");
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

        private static int[] RankChangePermutationBarracuda(int rank0, int rank1)
        {
            var identity = new[] { 0, 1, 2, 3 };
            if (rank0 == 0)
                return identity;
            else if (rank0 == 1)
            {
                // ONNX:
                // 8 -> 1,8
                // 8 -> 1,1,8
                // 8 -> 1,1,1,8
                if (rank1 == 0 || rank1 == 1)
                    return identity;
                else if (rank1 == 2)
                    return new[] { 1, 0, 2, 3 };
                else if (rank1 == 3)
                    return new[] { 1, 2, 0, 3 };
                else if (rank1 == 4)
                    return new[] { 1, 2, 3, 0 };
                else
                    throw new ArgumentException($"Unsupported rank permutation change {rank0} to {rank1}");
            }
            else if (rank0 == 2)
            {
                // ONNX:
                // 28 -> 1,2,8
                // 28 -> 1,1,2,8
                if (rank1 == 3)
                    return new[] { 2, 0, 1, 3 };
                else if (rank1 == 4)
                    return new[] { 2, 3, 0, 1 };
                else
                    throw new ArgumentException($"Unsupported rank permutation change {rank0} to {rank1}");
            }
            else if (rank0 == 3)
            {
                // ONNX:
                // 5,2,8 -> 1,5,2,8
                if (rank1 == 4)
                    return new[] { 3, 0, 1, 2 };
                else
                    throw new ArgumentException($"Unsupported rank permutation change {rank0} to {rank1}");
            }
            else
                throw new ArgumentException($"Unsupported rank permutation change {rank0} to {rank1}");
        }
    }
}
