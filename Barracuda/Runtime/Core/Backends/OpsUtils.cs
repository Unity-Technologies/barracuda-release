using System.Collections.Generic;

namespace Unity.Barracuda {

class OpsUtils
{
    // Split W, R, and B into [iofj] tensors w, r, wb, rb
    public static void SplitWRBForLSTM(IOps ops, Tensor W, Tensor R, Tensor B, out Tensor[] w, out Tensor[] r, out Tensor[] wb, out Tensor[] rb)
    {
        w = new[]
        {
            // w_i
            ops.StridedSlice(W, new[] { 0, 0, 0, 0 }, new[] { W.batch, 1, 1, W.channels / 4 }, new[] { 1, 1, 1, 1 }),
            // w_o
            ops.StridedSlice(W, new[] { 0, 0, 0, W.channels / 4 }, new[] { W.batch, 1, 1, 2 * W.channels / 4 }, new[] { 1, 1, 1, 1 }),
            // w_f
            ops.StridedSlice(W, new[] { 0, 0, 0, 2 * W.channels / 4 }, new[] { W.batch, 1, 1, 3 * W.channels / 4 }, new[] { 1, 1, 1, 1 }),
            // w_j
            ops.StridedSlice(W, new[] { 0, 0, 0, 3 * W.channels / 4 }, new[] { W.batch, 1, 1, 4 * W.channels / 4 }, new[] { 1, 1, 1, 1 }),
        };

        r = new[]
        {
            // r_i
            ops.StridedSlice(R, new[] { 0, 0, 0, 0 }, new[] { R.batch, 1, 1, R.channels / 4 }, new[] { 1, 1, 1, 1 }),
            // r_o
            ops.StridedSlice(R, new[] { 0, 0, 0, R.channels / 4 }, new[] { R.batch, 1, 1, 2 * R.channels / 4 }, new[] { 1, 1, 1, 1 }),
            // r_f
            ops.StridedSlice(R, new[] { 0, 0, 0, 2 * R.channels / 4 }, new[] { R.batch, 1, 1, 3 * R.channels / 4 }, new[] { 1, 1, 1, 1 }),
            // r_j
            ops.StridedSlice(R, new[] { 0, 0, 0, 3 * R.channels / 4 }, new[] { R.batch, 1, 1, 4 * R.channels / 4 }, new[] { 1, 1, 1, 1 })
        };

        wb = new[]
        {
            // wb_i
            ops.StridedSlice(B, new[] { 0, 0, 0, 0 }, new[] { 1, 1, 1, B.channels / 8 }, new[] { 1, 1, 1, 1 }),
            // wb_o
            ops.StridedSlice(B, new[] { 0, 0, 0, B.channels / 8 }, new[] { 1, 1, 1, 2 * B.channels / 8 }, new[] { 1, 1, 1, 1 }),
            // wb_f
            ops.StridedSlice(B, new[] { 0, 0, 0, 2 * B.channels / 8 }, new[] { 1, 1, 1, 3 * B.channels / 8 }, new[] { 1, 1, 1, 1 }),
            // wb_j
            ops.StridedSlice(B, new[] { 0, 0, 0, 3 * B.channels / 8 }, new[] { 1, 1, 1, 4 * B.channels / 8 }, new[] { 1, 1, 1, 1 })
        };

        rb = new []
        {
            // rb_i
            ops.StridedSlice(B, new[] { 0, 0, 0, 4 * B.channels / 8 }, new[] { 1, 1, 1, 5 * B.channels / 8 }, new[] { 1, 1, 1, 1 }),
            // rb_o
            ops.StridedSlice(B, new[] { 0, 0, 0, 5 * B.channels / 8 }, new[] { 1, 1, 1, 6 * B.channels / 8 }, new[] { 1, 1, 1, 1 }),
            // rb_f
            ops.StridedSlice(B, new[] { 0, 0, 0, 6 * B.channels / 8 }, new[] { 1, 1, 1, 7 * B.channels / 8 }, new[] { 1, 1, 1, 1 }),
            // rb_j
            ops.StridedSlice(B, new[] { 0, 0, 0, 7 * B.channels / 8 }, new[] { 1, 1, 1, 8 * B.channels / 8 }, new[] { 1, 1, 1, 1 })
        };
    }

    public static void BakeConstantWRBIntoLSTMLayer(Layer layer, Tensor W, Tensor R, Tensor B)
    {
        string name = layer.name;

        // Bake out constant tensors into layer
        void AddDataset(List<Layer.DataSet> datasets, BarracudaArray weights, string tensorName, Tensor t, ref int offset)
        {
            var dataset = new Layer.DataSet();
            dataset.name            = $"{name}/{tensorName}";
            dataset.shape           = t.shape;
            dataset.itemSizeInBytes = 4;
            dataset.length          = t.shape.length;
            dataset.offset          = offset;
            datasets.Add(dataset);

            t.ToReadOnlyArray().CopyToBarracudaArray(weights, offset);

            offset += t.shape.length;
        }

        var layerDatasets = new List<Layer.DataSet>();
        var layerWeights = new BarracudaArray(W.shape.length + R.shape.length + B.shape.length);
        int dataOffset = 0;

        var ops = new ReferenceCPUOps();
        using (var td = new TensorScope())
        {
            TensorScope.F _ = td._;

            Tensor[] w_iofj, r_iofj, wb_iofj, rb_iofj;
            SplitWRBForLSTM(ops, W, R, B, out w_iofj, out r_iofj, out wb_iofj, out rb_iofj);

            var indexName = new[] { "i", "o", "f", "j" };

            for (int i = 0; i < w_iofj.Length; i++)
            {
                AddDataset(layerDatasets, layerWeights, $"w_{indexName[i]}", _(w_iofj[i]), ref dataOffset);
            }

            for (int i = 0; i < w_iofj.Length; i++)
            {
                AddDataset(layerDatasets, layerWeights, $"r_{indexName[i]}", _(r_iofj[i]), ref dataOffset);
            }

            for (int i = 0; i < w_iofj.Length; i++)
            {
                AddDataset(layerDatasets, layerWeights, $"wb_{indexName[i]}", _(wb_iofj[i]), ref dataOffset);
            }

            for (int i = 0; i < w_iofj.Length; i++)
            {
                AddDataset(layerDatasets, layerWeights, $"rb_{indexName[i]}", _(rb_iofj[i]), ref dataOffset);
            }
        }

        layer.datasets = layerDatasets.ToArray();
        layer.weights = layerWeights;
    }
}


} // namespace Unity.Barracuda
