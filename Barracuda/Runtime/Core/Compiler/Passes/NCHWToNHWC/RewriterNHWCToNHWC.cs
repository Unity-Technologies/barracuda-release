using System;
using System.Collections.Generic;
using System.Linq;

namespace Unity.Barracuda.Compiler.Passes
{
    partial class NCHWToNHWCPass
    {
        Dictionary<Layer.Type, Action<Layer, ModelBuilder>> InstantiateRewriterNHWCToNHWC()
        {
            var rewritersNHWC = new Dictionary<Layer.Type, Action<Layer, ModelBuilder>>();

            // TODO, upsample is sometimes in NHWC mode
            rewritersNHWC.Add(Layer.Type.Reshape, (layer, net) =>
            {
                if (layer.inputs.Length == 1)
                {
                    var size = layer.pool;

                    // Don't use Tensorshape as this can remove a wild card
                    const int _ = 1;
                    if (size.Length == 1)
                        layer.pool = new[] { _, _, size[0], _, _, 1, 1, 1 }; // [1,1,N,1,1,1,1,1]
                    else if (size.Length == 2)
                        layer.pool = new[] { _, _, size[0], _, _, 1, 1, size[1] }; // [1, 1, N, 1, 1, 1, 1, C]
                    else if (size.Length == 3)
                        layer.pool = new[] { _, _, size[0], _, _, _, size[1], size[2] }; // [1,1,N,1,1,1,W,C]
                    else if (size.Length == 4)
                        layer.pool = new[] { _, _, size[0], _, _, size[1], size[2], size[3] }; // [1,1,N,1,1,H,W,C]
                    else if (size.Length == 5)
                        layer.pool = new[] { _, _, size[0], _, size[1], size[2], size[3], size[4] }; // [1,1,N,1,D,H,W,C]
                    else if (size.Length == 6)
                        layer.pool = new[] { _, _, size[0], size[1], size[2], size[3], size[4], size[5] }; // [1,1,N,T,D,H,W,C]
                    else
                        layer.pool = new[] { size[0], size[1], size[2], size[3], size[4], size[5], size[6], size[7] }; // [S,R,N,T,D,H,W,C]
                }
            });
            rewritersNHWC.Add(Layer.Type.Transpose, (layer, net) =>
            {
                var size = layer.pool;
                if (size.Length == 1)
                {
                    layer.pool = new[] { 0, 1, 2, 3 }; // [N,_,_,_]
                    layer.pool[0] = size[0];
                }
                else if (size.Length == 2)
                {
                    layer.pool = new[] { 0, 1, 2, 3 }; // [N, _, _, C]
                    layer.pool[0] = size[0] == 0 ? 0 : size[0] + 2;
                    layer.pool[3] = size[1] == 0 ? 0 : size[1] + 2;
                }
                else if (size.Length == 3)
                {
                    layer.pool = new[] { 0, 1, 2, 3 }; // [N, _, W, C]
                    layer.pool[0] = size[0] == 0 ? 0 : size[0] + 1;
                    layer.pool[2] = size[1] == 0 ? 0 : size[1] + 1;
                    layer.pool[3] = size[2] == 0 ? 0 : size[2] + 1;
                }
                else if (size.Length == 4)
                    layer.pool = size; // [N,H,W,C]
                else if (size.Length == 5)
                {
                    layer.pool = new[] { 0, 1, 2, 3, 4, 5, 6, 7 }; // [_,_,N,_,D,H,W,C]
                    layer.pool[2] = size[0] == 0 ? 2 : size[0] + 3;
                    layer.pool[4] = size[1] == 0 ? 2 : size[1] + 3;
                    layer.pool[5] = size[2] == 0 ? 2 : size[2] + 3;
                    layer.pool[6] = size[3] == 0 ? 2 : size[3] + 3;
                    layer.pool[7] = size[4] == 0 ? 2 : size[4] + 3;
                }
                else if (size.Length == 6)
                {
                    layer.pool = new[] { 0, 1, 2, 3, 4, 5, 6, 7 }; // [1,1,N,T,D,H,W,C]
                    layer.pool[2] = size[0] + 2;
                    layer.pool[3] = size[1] + 2;
                    layer.pool[4] = size[2] + 2;
                    layer.pool[5] = size[3] + 2;
                    layer.pool[6] = size[4] + 2;
                    layer.pool[7] = size[5] + 2;
                }
                else
                    layer.pool = new[] { size[0], size[1], size[2], size[3], size[4], size[5], size[6], size[7] }; // [S,R,N,T,D,H,W,C]
            });
            rewritersNHWC.Add(Layer.Type.Gather, ConvertAxisNHWC);
            rewritersNHWC.Add(Layer.Type.Concat, ConvertAxisNHWC);
            rewritersNHWC.Add(Layer.Type.ReduceMax, ConvertAxisNHWC);
            rewritersNHWC.Add(Layer.Type.ReduceMean, ConvertAxisNHWC);
            rewritersNHWC.Add(Layer.Type.ReduceMin, ConvertAxisNHWC);
            rewritersNHWC.Add(Layer.Type.ReduceProd, ConvertAxisNHWC);
            rewritersNHWC.Add(Layer.Type.ReduceSum, ConvertAxisNHWC);
            rewritersNHWC.Add(Layer.Type.ArgMax, ConvertAxisNHWC);
            rewritersNHWC.Add(Layer.Type.ArgMin, ConvertAxisNHWC);
            rewritersNHWC.Add(Layer.Type.Activation, ConvertAxisNHWC);
            rewritersNHWC.Add(Layer.Type.StridedSlice, (layer, net) =>
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

                switch (rank)
                {
                    case 1:
                        layer.pad = new[] { 0, 0, onnxStarts[0], 0, 0, 0, 0, 0 };
                        layer.pool = new[] { int.MaxValue, int.MaxValue, onnxEnds[0], int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue };
                        layer.stride = new[] { 1, 1, onnxSteps[0], 1, 1, 1, 1, 1 };
                        break;
                    case 2:
                        layer.pad = new[] { 0, 0, onnxStarts[0], 0, 0, 0, 0, onnxStarts[1] };
                        layer.pool = new[] { int.MaxValue, int.MaxValue, onnxEnds[0], int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, onnxEnds[1] };
                        layer.stride = new[] { 1, 1, onnxSteps[0], 1, 1, 1, 1, onnxSteps[1] };
                        break;
                    case 3:
                        layer.pad = new[] { 0, 0, onnxStarts[0], 0, 0, 0, onnxStarts[1], onnxStarts[2] };
                        layer.pool = new[] { int.MaxValue, int.MaxValue, onnxEnds[0], int.MaxValue, int.MaxValue, int.MaxValue, onnxEnds[1], onnxEnds[2] };
                        layer.stride = new[] { 1, 1, onnxSteps[0], 1, 1, 1, onnxSteps[1], onnxSteps[2] };
                        break;
                    case 4:
                        layer.pad = new[] { 0, 0, onnxStarts[0], 0, 0, onnxStarts[1], onnxStarts[2], onnxStarts[3] };
                        layer.pool = new[] { int.MaxValue, int.MaxValue, onnxEnds[0], int.MaxValue, int.MaxValue, onnxEnds[1], onnxEnds[2], onnxEnds[3] };
                        layer.stride = new[] { 1, 1, onnxSteps[0], 1, 1, onnxSteps[1], onnxSteps[2], onnxSteps[3] };
                        break;
                    case 5:
                        layer.pad = new[] { 0, 0, onnxStarts[0], 0, onnxStarts[1], onnxStarts[2], onnxStarts[3], onnxStarts[4] };
                        layer.pool = new[] { int.MaxValue, int.MaxValue, onnxEnds[0], int.MaxValue, onnxEnds[1], onnxEnds[2], onnxEnds[3], onnxEnds[4] };
                        layer.stride = new[] { 1, 1, onnxSteps[0], 1, onnxSteps[1], onnxSteps[2], onnxSteps[3], onnxSteps[4] };
                        break;
                    default:
                        throw new ArgumentException($"Unsupported tensor rank {rank} for StridedSlice");
                }
            });
            rewritersNHWC.Add(Layer.Type.Flatten, (layer, net) =>
            {
                layer.type = Layer.Type.Nop;
            });
            rewritersNHWC.Add(Layer.Type.Squeeze, (layer, net) =>
            {
                layer.type = Layer.Type.Nop;
            });
            rewritersNHWC.Add(Layer.Type.Unsqueeze, (layer, net) =>
            {
                layer.type = Layer.Type.Nop;
            });
            rewritersNHWC.Add(Layer.Type.Load, (layer, net) =>
            {
                int rank = layer.axis;
                if (rank != 2 && rank != 3)
                    return;

                var constX = layer.DataSetToTensor(0);

                var shape = constX.shape;
                switch (rank)
                {
                    case 2:
                        // _,_,N,_,_,C,_,_ => _,_,N,_,_,_,_,C
                        shape = new TensorShape(shape.batch, shape.height);
                        break;
                    case 3:
                        // _,_,N,_,_,W,C,_ => _,_,N,_,_,_,W,C
                        shape = new TensorShape(shape.batch, shape.height, shape.width);
                        break;
                }

                var reshapedX = m_Ops.Reshape(constX, shape);
                layer.ApplyTensorToDataSet(reshapedX, 0);
                reshapedX.Dispose();
                constX.Dispose();
            });


            return rewritersNHWC;
        }


        void ConvertAxisNHWC(Layer layer, ModelBuilder net)
        {
            if (layer.type == Layer.Type.Activation && layer.activation != Layer.Activation.Softmax)
                return;

            string input0 = layer.inputs[0];
            if (!m_RanksByName.TryGetValue(input0, out int? input0Rank) || !input0Rank.HasValue)
                throw new Exception($"Must have input rank for {input0} in order to convert axis for NHWC op");

            var axis = layer.axis;
            if (input0Rank == 1 || input0Rank == 0)
                // N => _,_N,_,_,_,_,_
                // 0       2
                layer.axis = 2;
            else if (input0Rank == 2)
                // N,C => _,_,N,_,_,_,_,C
                // 0,1        2         7
                layer.axis = axis == 0 ? 2 : 7;
            else if (input0Rank == 3)
                // N,W,C => _,_N,_,_,_,W,C
                // 0,1,2       2       6,7
                layer.axis = axis == 0 ? 2 : axis + 5;
            else if (input0Rank == 4)
                // N,H,W,C => _,_N,_,_,H,W,C
                // 0,1,2,3       2     5,6,7
                layer.axis = axis == 0 ? 2 : axis + 4;
            else if (input0Rank == 5)
                // N,D,H,W,C => N,_,D,H,W,C
                // 0,1,2,3,4    2,  4,5,6,7
                layer.axis = axis == 0 ? 2 : axis + 3;
            else if (input0Rank == 6)
                // N,T,D,H,W,C => N,T,D,H,W,C
                // 0,1,2,3,4,5    2,3,4,5,6,7
                layer.axis = axis + 2;
            else
                throw new ArgumentException($"Unsupported tensor rank {input0Rank} for StridedSlice");
        }

    }
}
