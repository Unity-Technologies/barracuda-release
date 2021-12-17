using System;
using System.Collections.Generic;
using System.Linq;

namespace Unity.Barracuda.Compiler.Passes
{
    partial class NCHWToNHWCPass
    {
        Dictionary<Layer.Type, Func<Layer, ModelBuilder, bool>> InstantiateRewriterNCHWToNHWC()
        {
            var rewriters = new Dictionary<Layer.Type, Func<Layer, ModelBuilder, bool>>();

            // return true if layer should be included in rewritten model, false if it was replaced
            rewriters.Add(Layer.Type.Load, ConvertDatasets);
            rewriters.Add(Layer.Type.Reshape, (layer, net) =>
            {
                // TODO reshape with pool as constant
                string input0 = layer.inputs[0];
                if (!m_RanksByName.TryGetValue(input0, out int? input0Rank) || !input0Rank.HasValue)
                    throw new Exception($"Must have input rank for {input0} in order to convert Reshape to NHWC");

                int outputRank = 4;
                Layer nchwTranspose;
                // TODO cleanup?
                if (input0Rank.Value == 1)
                    nchwTranspose = net.Identity($"Transpose_{input0}_For_{layer.name}", input0);
                else if (input0Rank.Value == 2)
                    nchwTranspose = net.Transpose($"Transpose_{input0}_For_{layer.name}", input0, k_FromNHWCtoNCHW);
                else if (input0Rank.Value == 3)
                    nchwTranspose = net.Transpose($"Transpose_{input0}_For_{layer.name}", input0, k_FromN1WCtoNCH);
                else if (input0Rank.Value == 4)
                    nchwTranspose = net.Transpose($"Transpose_{input0}_For_{layer.name}", input0, k_FromNHWCtoNCHW);
                else if (input0Rank.Value == 5)
                    nchwTranspose = net.Transpose($"Transpose_{input0}_For_{layer.name}", input0, new[] { 0, 1, 2, 3, 7, 4, 5, 6 });
                else
                    // TODO 8D?
                    nchwTranspose = net.Transpose($"Transpose_{input0}_For_{layer.name}", input0, new[] { 0, 1, 2, 7, 3, 4, 5, 6 });

                Layer reshape = null;
                if (layer.inputs.Length > 1)
                {
                    string input1 = layer.inputs[1];
                    if (!m_RanksByName.TryGetValue(input1, out int? input1Rank) || !input1Rank.HasValue)
                        throw new Exception($"Must have input rank for {input1} in order to convert Reshape to NHWC");

                    if (input1Rank.Value == 1) // shape is in the tensor
                    {
                        if (!m_ShapesByName.TryGetValue(input1, out TensorShape? input1Shape) || !input1Shape.HasValue)
                            throw new Exception($"Must have input shape for {input1} in order to convert Reshape to NHWC");

                        outputRank = input1Shape.Value[TensorShape.DataBatch];
                    }

                    reshape = net.Reshape($"{layer.name}_NCHW", nchwTranspose, input1);
                }
                else if (layer.pool.Length > 0)
                {
                    outputRank = layer.pool.Length;

                    var shape = IRShapeInferenceHelper.ShapeInference.OnnxLayoutToTensorShapeLayout(layer.pool);

                    reshape = net.Reshape($"{layer.name}_NCHW", nchwTranspose, shape);
                }

                // TODO cleanup?
                if (outputRank == 1)
                    nchwTranspose = net.Identity(layer.name, reshape);
                else if (outputRank == 2)
                    nchwTranspose = net.Transpose(layer.name, reshape, k_FromNCHWtoNHWC);
                else if (outputRank == 3)
                    net.Transpose(layer.name, reshape, k_FromNCHtoN1WC);
                else if (outputRank == 4)
                    net.Transpose(layer.name, reshape, k_FromNCHWtoNHWC);
                else if (outputRank == 5)
                    net.Transpose(layer.name, reshape, new[] { 0, 1, 2, 3, 5, 6, 7, 4 });
                else
                    // TODO 8D?
                    net.Transpose(layer.name, reshape, new[] { 0, 1, 2, 4, 5, 6, 7, 3 });

                return false;
            });
            rewriters.Add(Layer.Type.Expand, (layer, net) =>
            {
                string input0 = layer.inputs[0];
                if (!m_RanksByName.TryGetValue(input0, out int? input0Rank) || !input0Rank.HasValue)
                    throw new Exception($"Must have input rank for {input0} in order to convert Reshape to NHWC");

                int rank0 = input0Rank.Value;
                var size = layer.pool.ToList();
          
                if (rank0 >= size.Count)
                {
                    for (int i = 0; i < rank0 - size.Count; i++)
                        size.Insert(0, 1);
                    layer.pool = size.ToArray();
                    return ConvertShape(layer, net);
                }

                // inputShape needs to be unsqueezed
                var transpose = RankChangePermutationBarracuda(rank0, size.Count);
                net.Transpose(layer.name, input0, transpose);

                ConvertShape(layer, net);

                return false;
            });
            rewriters.Add(Layer.Type.Shape, (layer, net) =>
            {
                if (layer.axis >= 0)
                    ConvertAxis(layer, net);

                return true;
            });
            rewriters.Add(Layer.Type.Transpose, (layer, net) =>
            {
                int rank = layer.pool.Length;
                int[] onnxTranspose = layer.pool;

                // TODO cleanup?
                switch (rank)
                {
                    case 2:
                        {
                            // onnx : 5,7 => 5,7 / 7,5
                            // barracuda : 5,_,_,7 => 5,_,_,7 / 7,_,_,5
                            layer.pool = new[] { 0, 1, 2, 3 };
                            layer.pool[0] = onnxTranspose[0] == 1 ? 3 : onnxTranspose[0];
                            layer.pool[3] = onnxTranspose[1] == 1 ? 3 : onnxTranspose[1];
                            return true;
                        }
                    case 3:
                        {
                            // onnx : 5,7,3 => 5,7,3 / 7,5,3 / 7,3,5 ...
                            // barracuda : 5,_,7,3 => 7,_,3,5 / 7,_,5,3 ...
                            layer.pool = new[] { 0, 1, 2, 3 };
                            layer.pool[0] = onnxTranspose[0] == 1 ? 3 : onnxTranspose[0] == 2 ? 2 : onnxTranspose[0];
                            layer.pool[3] = onnxTranspose[1] == 1 ? 3 : onnxTranspose[1] == 2 ? 2 : onnxTranspose[1];
                            layer.pool[2] = onnxTranspose[2] == 1 ? 3 : onnxTranspose[2] == 2 ? 2 : onnxTranspose[2];
                            return true;
                        }
                    case 4:
                        {
                            layer.pool = new[] { 0, 1, 2, 3 };
                            layer.pool[0] = onnxTranspose[0] == 1 ? 3 : onnxTranspose[0] == 2 ? 1 : onnxTranspose[0] == 3 ? 2 : onnxTranspose[0];
                            layer.pool[3] = onnxTranspose[1] == 1 ? 3 : onnxTranspose[1] == 2 ? 1 : onnxTranspose[1] == 3 ? 2 : onnxTranspose[1];
                            layer.pool[1] = onnxTranspose[2] == 1 ? 3 : onnxTranspose[2] == 2 ? 1 : onnxTranspose[2] == 3 ? 2 : onnxTranspose[2];
                            layer.pool[2] = onnxTranspose[3] == 1 ? 3 : onnxTranspose[3] == 2 ? 1 : onnxTranspose[3] == 3 ? 2 : onnxTranspose[3];
                            return true;
                        }
                    case 5:
                        {
                            // onnx : 5,7,3,4,9 => 5,9,4,7,3 / 3,9,4,7,5 ...
                            layer.pool = new[] { 0, 1, 2, 3, 4, 5, 6, 7 };
                            //  [1,1,N,1,D,H,W,C]

                            layer.pool[2] = onnxTranspose[0] == 0 ? 2 : onnxTranspose[0] == 1 ? 7 : onnxTranspose[0] + 2;
                            layer.pool[7] = onnxTranspose[1] == 0 ? 2 : onnxTranspose[1] == 1 ? 7 : onnxTranspose[1] + 2;
                            layer.pool[4] = onnxTranspose[2] == 0 ? 2 : onnxTranspose[2] == 1 ? 7 : onnxTranspose[2] + 2;
                            layer.pool[5] = onnxTranspose[3] == 0 ? 2 : onnxTranspose[3] == 1 ? 7 : onnxTranspose[3] + 2;
                            layer.pool[6] = onnxTranspose[4] == 0 ? 2 : onnxTranspose[4] == 1 ? 7 : onnxTranspose[4] + 2;

                            return true;
                        }
                    default:
                        {
                            // TODO 8D?
                            layer.pool = new[] { 0, 1, 2, 3, 4, 5, 6, 7 };
                            // NCTDHW

                            layer.pool[2] = onnxTranspose[0] == 0 ? 2 : onnxTranspose[0] == 1 ? 7 : onnxTranspose[0] + 1;
                            layer.pool[7] = onnxTranspose[1] == 0 ? 2 : onnxTranspose[1] == 1 ? 7 : onnxTranspose[1] + 1;
                            layer.pool[3] = onnxTranspose[2] == 0 ? 2 : onnxTranspose[2] == 1 ? 7 : onnxTranspose[2] + 1;
                            layer.pool[4] = onnxTranspose[3] == 0 ? 2 : onnxTranspose[3] == 1 ? 7 : onnxTranspose[3] + 1;
                            layer.pool[5] = onnxTranspose[4] == 0 ? 2 : onnxTranspose[4] == 1 ? 7 : onnxTranspose[4] + 1;
                            layer.pool[6] = onnxTranspose[5] == 0 ? 2 : onnxTranspose[5] == 1 ? 7 : onnxTranspose[5] + 1;

                            return true;
                        }
                }
            });
            rewriters.Add(Layer.Type.Unsqueeze, (layer, net) =>
            {
                // Replace w/ a Transpose since Barracuda tensors are full rank (i.e. grab an unused dimension)
                string input0 = layer.inputs[0];
                if (!m_RanksByName.TryGetValue(input0, out int? input0Rank) || !input0Rank.HasValue)
                    throw new Exception($"Must have input rank for {input0} in order to convert axis for Unsqueeze");

                int rank = input0Rank.Value;
                var combinePermutations = new[] { 0, 1, 2, 3 };
                for (int i = 0; i < layer.pool.Length; i++)
                {
                    int axis = layer.pool[i];
                    if (axis < 0)
                        axis = rank + 1 - axis;

                    var transpose = UnSqueezeAxisPermutationForMappingNCHWLayoutToBarracuda(rank, axis);

                    // there could be a 4 / 8D shape mismatch
                    if (transpose.Length == 8 && combinePermutations.Length == 4)
                        combinePermutations = Permutation4DTo8D(combinePermutations);

                    combinePermutations = TensorExtensions.Permute(transpose, combinePermutations);

                    rank++;
                }
                net.Transpose(layer.name, input0, combinePermutations);

                return false;
            });
            rewriters.Add(Layer.Type.Squeeze, (layer, net) =>
            {
                // Replace w/ a Transpose since Barracuda tensors are full rank
                string input0 = layer.inputs[0];
                if (!m_RanksByName.TryGetValue(input0, out int? input0Rank) || !input0Rank.HasValue)
                    throw new Exception($"Must have input rank for {input0} in order to convert axis for Squeeze");

                int rank = input0Rank.Value;
                var combinePermutations = new[] { 0, 1, 2, 3 };
                for (int i = 0; i < layer.pool.Length; i++)
                {
                    int axis = layer.pool[i];
                    if (axis < 0)
                        axis = rank + 1 - axis;

                    var transpose = SqueezeAxisPermutationForMappingNCHWLayoutToBarracuda(rank, axis);

                    // there could be a 4 / 8D shape mismatch
                    if (transpose.Length == 8 && combinePermutations.Length == 4)
                        combinePermutations = Permutation4DTo8D(combinePermutations);

                    combinePermutations = TensorExtensions.Permute(transpose, combinePermutations);

                    rank--;
                }
                net.Transpose(layer.name, input0, combinePermutations);

                return false;
            });
            rewriters.Add(Layer.Type.Flatten, (layer, net) =>
            {
                string input0 = layer.inputs[0];
                if (!m_RanksByName.TryGetValue(input0, out int? input0Rank) || !input0Rank.HasValue)
                    throw new Exception($"Must have input rank for {input0} in order to convert Flatten to NHWC");

                Layer nchwTranspose = net.Transpose($"Transpose_{input0}_For_{layer.name}", input0, input0Rank.Value == 3 ? k_FromN1WCtoNCH : k_FromNHWCtoNCHW);
                net.Flatten(layer.name, nchwTranspose);
                // No need to transpose back b/c final shape is always NC (rank 2)

                return false;
            });
            rewriters.Add(Layer.Type.Concat, ConvertAxis);
            rewriters.Add(Layer.Type.StridedSlice, (layer, net) =>
            {
                int rank = 4;
                if (m_RanksByName.ContainsKey(layer.name) && m_RanksByName[layer.name] != null)
                    rank = m_RanksByName[layer.name].Value;

                var name = layer.name;

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

                layer.pad = PermuteToBarracuda(onnxStarts, rank, 0);
                layer.pool = PermuteToBarracuda(onnxEnds, rank, int.MaxValue);
                layer.stride = PermuteToBarracuda(onnxSteps, rank, 1);

                return true;
            });
            rewriters.Add(Layer.Type.Tile, (layer, net) =>
            {
                if (layer.inputs.Length == 1)
                {
                    int rank = 4;
                    if (m_RanksByName.ContainsKey(layer.name) && m_RanksByName[layer.name] != null)
                        rank = m_RanksByName[layer.name].Value;
                    layer.pool = PermuteToBarracuda(layer.pool, rank, 1);// TensorExtensions.Permute(layer.pool, k_FromNCHWtoNHWC);
                }

                return true;
            });
            rewriters.Add(Layer.Type.Activation, ConvertActivation);
            rewriters.Add(Layer.Type.Gather, ConvertGather);
            rewriters.Add(Layer.Type.TopKIndices, ConvertAxis);
            rewriters.Add(Layer.Type.TopKValues, ConvertAxis);

            rewriters.Add(Layer.Type.RandomNormal, ConvertNormal);
            rewriters.Add(Layer.Type.RandomUniform, ConvertNormal);

            rewriters.Add(Layer.Type.ReduceMax, Reduce);
            rewriters.Add(Layer.Type.ReduceMean, Reduce);
            rewriters.Add(Layer.Type.ReduceMin, Reduce);
            rewriters.Add(Layer.Type.ReduceProd, Reduce);
            rewriters.Add(Layer.Type.ReduceSum, Reduce);

            rewriters.Add(Layer.Type.ArgMax, Reduce);
            rewriters.Add(Layer.Type.ArgMin, Reduce);

            rewriters.Add(Layer.Type.Upsample2D, Upsample);
            rewriters.Add(Layer.Type.Resample2D, Upsample);
            rewriters.Add(Layer.Type.Upsample3D, Upsample);

            rewriters.Add(Layer.Type.MatMul, (layer, net) =>
            {
                string input0 = layer.inputs[0];
                if (!m_RanksByName.TryGetValue(input0, out int? input0Rank) || !input0Rank.HasValue)
                    throw new Exception($"Must have input rank for {input0} in order to convert axis for NHWC op");

                string input1 = layer.inputs[1];
                if (!m_RanksByName.TryGetValue(input1, out int? input1Rank) || !input1Rank.HasValue)
                    throw new Exception($"Must have input rank for {input1} in order to convert axis for NHWC op");

                layer.pool = new[] { input0Rank.Value, input1Rank.Value };

                return true;
            });

            rewriters.Add(Layer.Type.OneHot, (layer, net) =>
            {
                string input0 = layer.inputs[0];
                if (!m_RanksByName.TryGetValue(input0, out int? input0Rank) || !input0Rank.HasValue)
                    throw new Exception($"Must have input rank for {input0} in order to convert axis for NHWC op");

                layer.axis = input0Rank.Value;

                return true;
            });

            rewriters.Add(Layer.Type.Pad, Pad);


            return rewriters;
        }

        int[] GetChannelsLastPermutationsFromRank(int rank)
        {
            int[] fromNtoC = { 3, 1, 2, 0 };
            int[] k_FromNCtoN11C = { 0, 2, 3, 1 };
            int[] k_FromNCDHWtoNDHWC = { 0, 1, 2, 3, 5, 6, 7, 4 };

            int[] permutations = k_FromNCHWtoNHWC;
            if (rank == 5)
                permutations = k_FromNCDHWtoNDHWC;
            else if (rank == 3)
                permutations = k_FromNCHtoN1WC;
            else if (rank == 2)
                permutations = k_FromNCtoN11C;
            // else if (rank == 1) // AE: are we keeping rank 1 in N now?
            //     permutations = fromNtoC;

            return permutations;
        }

        int GetApproximateRankFromTensorShape(TensorShape shape)
        {
            // dimensions misreports rank if a dimension is 1
            int rank = shape.dimensions;
            // NOTE: NCHW shape reinterpretation of barracuda layout: N == batch, C == height, H == width, W == height
            if (shape.batch == 1)
                rank++;
            if (shape.height == 1 && (shape.width > 1 || shape.height > 1))
                rank++;

            return rank;
        }

        bool ConvertDatasets(Layer layer, ModelBuilder net)
        {
            for (var i = 0; i < layer.datasets.Length; i++)
            {
                var X = layer.DataSetToTensor(i);

                // NCH is treated as NC1W in Barracuda
                TensorShape shape = X.shape;

                int rank = layer.axis; // rank that may have been shoved into the layer on import (e.g. Const)
                if (rank < 0)
                    rank = GetApproximateRankFromTensorShape(shape);

                int[] permutations = GetChannelsLastPermutationsFromRank(rank);
                var O = m_Ops.Transpose(X, permutations);
                layer.ApplyTensorToDataSet(O, i);

                O.Dispose();
                X.Dispose();
            }

            return true;
        }

        bool ConvertActivation(Layer layer, ModelBuilder net)
        {
            if (layer.activation != Layer.Activation.Softmax && layer.activation != Layer.Activation.LogSoftmax)
                return true;

            string input0 = layer.inputs[0];
            if (!m_RanksByName.TryGetValue(input0, out int? input0Rank) || !input0Rank.HasValue)
                throw new Exception($"Must have input rank for {input0} in order to convert axis for NHWC op");

            int axis = layer.axis;
            if (axis < 0)
                axis += input0Rank.Value;

            int[] permutations = AxisPermutationsForMappingNCHWLayoutToBarracuda(input0Rank.Value);
            layer.axis = Array.IndexOf(permutations, axis);

            return true;
        }

        bool ConvertNormal(Layer layer, ModelBuilder net)
        {
            if (layer.inputs.Length == 1)
                return true;

            var shape = new TensorShape(layer.pool);
            var permutations = shape.Get8DPermutationsForNCHWPermutationsAndShape(k_FromNCHWtoNHWC);

            // Preserve symbolic shape by operating on int array instead of TensorShape, which would resolve unknown dimensions
            layer.pool = TensorExtensions.Permute(layer.pool, permutations);

            return true;
        }

        bool ConvertShape(Layer layer, ModelBuilder net)
        {
            layer.pool = IRShapeInferenceHelper.ShapeInference.OnnxLayoutToBarracudaTensorShape(layer.pool).ToArray();
            return true;
        }

        bool ConvertAxis(Layer layer, ModelBuilder net)
        {
            string input0 = layer.inputs[0];
            if (!m_RanksByName.TryGetValue(input0, out int? input0Rank) || !input0Rank.HasValue)
                throw new Exception($"Must have input rank for {input0} in order to convert axis for NHWC op");

            int axis = layer.axis;
            if (axis < 0)
                axis += input0Rank.Value;

            int[] permutations = AxisPermutationsForMappingNCHWLayoutToBarracuda(input0Rank.Value);
            layer.axis = Array.IndexOf(permutations, axis);

            return true;
        }

        bool ConvertGather(Layer layer, ModelBuilder net)
        {
            string input0 = layer.inputs[0];
            if (!m_RanksByName.TryGetValue(input0, out int? input0Rank) || !input0Rank.HasValue)
                throw new Exception($"Must have input rank for {input0} in order to convert axis for NHWC op");

            string input1 = layer.inputs[1];
            if (!m_RanksByName.TryGetValue(input1, out int? input1Rank) || !input1Rank.HasValue)
                throw new Exception($"Must have input rank for {input1} in order to convert axis for NHWC op");

            layer.pool = new[] { input0Rank.Value, input1Rank.Value };

            return ConvertAxis(layer, net);
        }


        bool Upsample(Layer layer, ModelBuilder net)
        {
            string input0 = layer.inputs[0];
            if (!m_RanksByName.TryGetValue(input0, out int? input0Rank) || !input0Rank.HasValue)
                throw new Exception($"Must have input rank for {input0} in order to convert axis for NHWC op");

            if (layer.inputs.Length > 1) // dynamic case
                return true;

            int[] scales = layer.pool;
            scales = scales.Skip(2).ToArray();
            switch (scales.Length)
            {
                case 0:
                    layer.pool = new[] { 1, 1 };
                    break;
                case 1:
                    layer.pool = new[] { scales[0], 1 };                 // 1D W => W_
                    break;
                case 2:
                    layer.pool = new[] { scales[1], scales[0] };         // 2D HW => WH
                    break;
                case 3:
                    layer.pool = new[] { scales[2], scales[1], scales[0] };  // 3D DHW => WHD
                    break;
                default:
                    throw new Exception($"Attribute pads of unsupported length {scales.Length} in {layer.name} ot type {layer.type}.");
            }

            return true;
        }

        bool Pad(Layer layer, ModelBuilder net)
        {
            string input0 = layer.inputs[0];
            if (!m_RanksByName.TryGetValue(input0, out int? input0Rank) || !input0Rank.HasValue)
                throw new Exception($"Must have input rank for {input0} in order to convert pad for NHWC op");

            var autopadOption = (Layer.AutoPad)(layer.pool[0]);

            if (input0Rank <= 4)
            {
                if (autopadOption == Layer.AutoPad.NotSet)
                {
                    if (input0Rank == 4) // CHW => WHC
                        layer.pad = new[] { layer.pad[3], layer.pad[2], layer.pad[1], layer.pad[7], layer.pad[6], layer.pad[5] };
                    else if (input0Rank == 3) // CW => W_C
                        layer.pad = new[] { layer.pad[2], 0, layer.pad[1], layer.pad[5], layer.pad[4] };
                }
                else
                {
                    int autopad = -(int)(autopadOption);
                    layer.pad = new[] { autopad, autopad, autopad, autopad };
                }
                switch (layer.axis)
                {
                    case 0:
                        layer.type = Layer.Type.Border2D;
                        break;
                    case 1:
                        layer.type = Layer.Type.Pad2DReflect;
                        break;
                    case 2:
                        layer.type = Layer.Type.Pad2DEdge;
                        break;
                    case 3:
                        layer.type = Layer.Type.Pad2DSymmetric;
                        break;
                }
                layer.axis = -1;
                return true;
            }
            else if (input0Rank == 5)
            {
                // CDHW => WHDC
                layer.pad = new[] { layer.pad[4], layer.pad[3], layer.pad[2], layer.pad[1], layer.pad[9], layer.pad[8], layer.pad[7], layer.pad[6] };
                layer.type = Layer.Type.Border3D;
                return true;
            }

            throw new Exception($"Unsuported Pad layer, {layer.name}");
        }

        bool Reduce(Layer layer, ModelBuilder net)
        {
            string input0 = layer.inputs[0];
            if (!m_RanksByName.TryGetValue(input0, out int? input0Rank) || !input0Rank.HasValue)
                throw new Exception($"Must have input rank for {input0} in order to convert axis for NHWC op");

            int axis = layer.axis;
            if (axis < 0)
                axis += input0Rank.Value;

            int[] permutations = AxisPermutationsForMappingNCHWLayoutToBarracuda(input0Rank.Value);
            layer.axis = Array.IndexOf(permutations, axis);


            int keepdims = (int)layer.alpha;

            if (keepdims != 1 && input0Rank.Value > 1) // keepdims removes dimensions in the context of onnx thus we need to repack/transpose to match behavior.
            {
                string name = layer.name;
                layer.name = $"{layer.name}__reduce";

                net.Reduce(layer.type, layer.name, input0, layer.axis, true, -1);


                var nameT = $"{layer.name}__transpose";
                var transpose = GetPermutationToMatchReduceWithDroppedDimensionsFromONNX(new[] { axis }, input0Rank.Value);
                var transposeLayer = net.Transpose(nameT, layer, transpose);

                net.Identity(name, transposeLayer);
            }
            else
            {
                net.Reduce(layer.type, layer.name, input0, layer.axis, true, -1);
            }

            return false;
        }

        static int[] AxisPermutationsForMappingNCHWLayoutToBarracuda(int rank)
        {
            const int _ = -1;

            switch (rank)
            {
                case 6:
                    return new[] { _, _, 0, 2, 3, 4, 5, 1 };
                case 5:
                    return new[] { _, _, 0, _, 2, 3, 4, 1 };
                case 4:
                    return new[] { _, _, 0, _, _, 2, 3, 1 };
                case 3:
                    return new[] { _, _, 0, _, _, _, 2, 1 };
                case 2:
                    return new[] { _, _, 0, _, _, _, _, 1 };
                case 1:
                case 0:
                    return new[] { _, _, 0, _, _, _, _, _ };
            }

            throw new ArgumentException($"Unsupported tensor rank {rank}");
        }

        public static int[] PermuteToBarracuda(int[] shape, int rank = 4, int defaultValue = 1)
        {
            var permutations = AxisPermutationsForMappingNCHWLayoutToBarracuda(rank); // Originally was NCHW
            UnityEngine.Debug.Assert(shape.Length <= permutations.Length);
            UnityEngine.Debug.Assert(shape.Length >= permutations.Count(v => v >= 0));
            var output = new int[permutations.Length];
            for (var i = 0; i < permutations.Length; ++i)
            {
                output[i] = permutations[i] >= 0 ? shape[permutations[i]] : defaultValue;
            }

            return output;
        }

        static int[] UnSqueezeAxisPermutationForMappingNCHWLayoutToBarracuda(int onnxRank, int onnxAxis)
        {
            var identity = new[] { 0, 1, 2, 3 };


            if (onnxRank == 4)
            {
                //            axis:   0          1          2          3          4
                // ONNX:      NCHW    1NCHW      N1CHW      NC1HW      NCH1W      NCHW1
                // Barracuda: NHWC    1__CHWN    N__CHW1    N__1HWC    N__H1WC    N__HW1C
                if (onnxAxis == 0)
                    return new[] { 0, 1, 3, 4, 7, 5, 6, 2 };
                else if (onnxAxis == 1)
                    return new[] { 0, 1, 2, 3, 7, 5, 6, 4 };
                else if (onnxAxis == 2)
                    return new[] { 0, 1, 2, 3, 4, 5, 6, 7 };
                else if (onnxAxis == 3)
                    return new[] { 0, 1, 2, 3, 5, 4, 6, 7 };
                else
                    return new[] { 0, 1, 2, 3, 5, 6, 4, 7 };
            }
            else if (onnxRank == 3)
            {
                //            axis:   0       1      2      3
                // ONNX:      NCH     1NCH    N1CH   NC1H   NCH1
                // Barracuda: N_HC    1CHN    NCH1   N1HC   NH1C
                if (onnxAxis == 0)
                    return new[] { 1, 3, 2, 0 };
                else if (onnxAxis == 1)
                    return new[] { 0, 3, 2, 1 };
                else if (onnxAxis == 2)
                    return identity;
                else
                    return new[] { 0, 2, 1, 3 };
            }
            else if (onnxRank == 2)
            {
                //            axis:   0       1      2
                // ONNX:      NC      1NC     N1C    NC1
                // Barracuda: N__C    1_CN    N_C1   N_1C
                if (onnxAxis == 0)
                    return new[] { 1, 2, 3, 0 };
                else if (onnxAxis == 1)
                    return new[] { 0, 1, 3, 2 };
                else
                    return identity;
            }
            else if (onnxRank == 1)
            {
                //            axis:   0       1
                // ONNX:      N       1N      N1
                // Barracuda: N___    1__N    N__1
                if (onnxAxis == 0)
                    return new[] { 1, 2, 3, 0 };
                else
                    return identity;
            }
            else if (onnxRank == 0)
                return identity;
            else
                throw new InvalidOperationException($"Not supported UnSqueeze operation with rank {onnxRank}");
        }

        static int[] SqueezeAxisPermutationForMappingNCHWLayoutToBarracuda(int onnxRank, int onnxAxis)
        {
            var identity = new[] { 0, 1, 2, 3 };

            if (onnxRank == 5)
            {
                //            axis:    0        1        2        3        4
                // ONNX:      NCDHW    CDHW     NDHW     NCHW     NCDW     NCDH
                // Barracuda: N_DHWC   C__HWD   N__HWD   N__HWC   N__DWC   N__DHC
                // { 0,1,2,3,4,5,6,7}
                //   _,_,N,_,D,H,W,C
                if (onnxAxis == 0)
                    return new[] { 0, 1, 7, 3, 2, 5, 6, 4 };
                else if (onnxAxis == 1)
                    return new[] { 0, 1, 2, 3, 7, 5, 6, 4 };
                else if (onnxAxis == 2)
                    return new[] { 0, 1, 2, 3, 4, 5, 6, 7 };
                else if (onnxAxis == 3)
                    return new[] { 0, 1, 2, 3, 5, 4, 6, 7 };
                else
                    return new[] { 0, 1, 2, 3, 6, 4, 5, 7 };
            }
            else if (onnxRank == 4)
            {
                //            axis:   0       1      2      3
                // ONNX:      NCHW    CHW     NHW    NCW    NCH
                // Barracuda: NHWC    C_WH    N_WH   N_WC   N_HC
                if (onnxAxis == 0)
                    return new[] { 3, 0, 2, 1 };
                else if (onnxAxis == 1)
                    return new[] { 0, 3, 2, 1 };
                else if (onnxAxis == 2)
                    return identity;
                else
                    return new[] { 0, 2, 1, 3 };
            }
            else if (onnxRank == 3)
            {
                //            axis:   0       1      2
                // ONNX:      NCH     CH      NH     NC
                // Barracuda: N_HC    C__H    N__H   N__C
                if (onnxAxis == 0)
                    return new[] { 3, 0, 1, 2 };
                else if (onnxAxis == 1)
                    return new[] { 0, 1, 3, 2 };
                else
                    return identity;
            }
            else if (onnxRank == 2)
            {
                //            axis:   0       1
                // ONNX:      NC      C       N
                // Barracuda: N__C    C___    N___
                if (onnxAxis == 0)
                    return new[] { 3, 0, 1, 2 };
                else
                    return identity;
            }
            else if (onnxRank == 1)
                return identity;
            else
                throw new InvalidOperationException($"Not supported Squeeze operation with rank {onnxRank}");
        }

        private static int[] GetPermutationToMatchReduceWithDroppedDimensionsFromONNX(int[] droppedONNXAxis, int rank)
        {
            //Assert.IsTrue(droppedONNXAxis.Length > 0);

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
                case 1:
                    onnxLayout = "N";
                    break;
                case 2:
                    onnxLayout = "NC";
                    break;
                case 3:
                    onnxLayout = "NCW";
                    break;
                case 4:
                    onnxLayout = "NCHW";
                    break;
                default:
                    //TODO support 8D
                    throw new Exception($"Reduce ops support up to 4D at the moment, however received an input of rank {rank}.");
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
            // Assert.IsTrue(onnxLayoutDimensionsDropped.Length > 0);

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
            for (int idTarget = 0; idTarget < permutation.Length; ++idTarget)
            {
                char semantic = barracudaSemanticLayoutFromONNXReduce[idTarget];
                permutation[idTarget] = "NHWC".IndexOf(semantic); ;
            }
            return permutation;
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
                // barracuda
                // 8,_,_,_ => 1,_,_,8
                // 8,_,_,_ => 1,_8,1
                // 8,_,_,_ => 1,1,8,1
                if (rank1 == 0 || rank1 == 1)
                    return identity;
                else if (rank1 == 2)
                    return new[] { 1, 2, 3, 0 };
                else if (rank1 == 3)
                    return new[] { 1, 2, 0, 3 };
                else if (rank1 == 4)
                    return new[] { 1, 2, 0, 3 };
                else
                    throw new ArgumentException($"Unsupported rank permutation change {rank0} to {rank1}");
            }
            else if (rank0 == 2)
            {
                // ONNX:
                // 28 -> 1,2,8
                // 28 -> 1,1,2,8
                // barracuda
                // 2__8 => 1,_8,2
                // 2__8 => 1,2,8,1
                if (rank1 == 3)
                    return new[] { 1, 2, 3, 0 };
                else if (rank1 == 4)
                    return new[] { 1, 0, 3, 2 };
                else
                    throw new ArgumentException($"Unsupported rank permutation change {rank0} to {rank1}");
            }
            else if (rank0 == 3)
            {
                // ONNX:
                // 5,2,8 -> 1,5,2,8
                // barracuda
                // 5,_,8,2 => 1,2,8,5
                if (rank1 == 4)
                    return new[] { 1, 3, 2, 0 };
                else
                    throw new ArgumentException($"Unsupported rank permutation change {rank0} to {rank1}");
            }
            else
                throw new ArgumentException($"Unsupported rank permutation change {rank0} to {rank1}");
        }

        static public int[] Permutation4DTo8D(int[] permutations)
        {
            if (permutations.Length == TensorShape.MaxRank)
                return permutations;

            int batchOldAxis   = TensorExtensions.Convert4DTo8DAxis(permutations[0]);
            int heighOldAxis   = TensorExtensions.Convert4DTo8DAxis(permutations[1]);
            int widthOldIndex  = TensorExtensions.Convert4DTo8DAxis(permutations[2]);
            int channeOldIndex = TensorExtensions.Convert4DTo8DAxis(permutations[3]);
            return new int[] { 0, 1, batchOldAxis, 3, 4, heighOldAxis, widthOldIndex, channeOldIndex };
        }

    }
}
