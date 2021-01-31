using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Assertions;

namespace Unity.Barracuda.Compiler.Passes
{
    /// <summary>
    /// Used to expand any ops that don't have native implementations (e.g. LSTM)
    /// </summary>
    class ExpandOpsPass : IModelPass
    {
        readonly BurstCPUOps m_Ops = new BurstCPUOps();

        public void Run(ref Model model)
        {
            var expandedModel = model.ShallowCopy();
            var modelBuilder = new ModelBuilder(expandedModel);

            var rewriters = new Dictionary<Layer.Type, Func<Layer, ModelBuilder, bool>>();
            rewriters.Add(Layer.Type.LSTM, ConvertLSTM);

            foreach (var l in model.layers)
            {
                // Some nodes output multiple layers (e.g. LSTM), so don't process or include those layers
                if (expandedModel.layers.Exists(alreadyOutputLayer => alreadyOutputLayer.name == l.name))
                    continue;

                if (!rewriters.TryGetValue(l.type, out Func<Layer, ModelBuilder, bool> rw) || rw(l, modelBuilder))
                {
                    // Either no re-write was needed or the layer was not replaced
                    expandedModel.layers.Add(l);
                }
            }

            model = expandedModel;
        }

        bool ConvertLSTM(Layer layer, ModelBuilder net)
        {
            return ConvertLSTM(layer, net, m_Ops);
        }

        public static bool ConvertLSTM(Layer layer, ModelBuilder net, IOps ops)
        {
            // LSTM
            // TODO need to transpose before when dealing with batches?
            var transposedInput = net.Transpose($"Transpose_for_{layer.name}", layer.inputs[0], new[] { 3, 1, 2, 0 });
            layer.inputs[0] = transposedInput.name;

            //    - it = f(Xt*Wi + Ht_1*Ri + Wbi + Rbi)
            //    - ft = f(Xt*Wf + Ht_1*Rf + Wbf + Rbf)
            //    - ct = g(Xt*Wc + Ht_1*Rc + Wbc + Rbc), c means j in our formula
            //    - Ct =   ft . Ct_  + it . ct
            //    - ot = f(Xt*Wo + Ht_1*Ro + Wbo + Rbo)
            //    - Ht =   ot . h(Ct)

            var W = layer.DataSetToTensor(0);
            var R = layer.DataSetToTensor(1);
            var B = layer.DataSetToTensor(2);

            // gate order [iofj]

            var w_i = ops.StridedSlice(W, new[] { 0, 0, 0, 0 }, new[] { W.batch, 1, 1, W.channels / 4 }, new[] { 1, 1, 1, 1 });
            var w_o = ops.StridedSlice(W, new[] { 0, 0, 0, W.channels / 4 }, new[] { W.batch, 1, 1, 2 * W.channels / 4 }, new[] { 1, 1, 1, 1 });
            var w_f = ops.StridedSlice(W, new[] { 0, 0, 0, 2 * W.channels / 4 }, new[] { W.batch, 1, 1, 3 * W.channels / 4 }, new[] { 1, 1, 1, 1 });
            var w_j = ops.StridedSlice(W, new[] { 0, 0, 0, 3 * W.channels / 4 }, new[] { W.batch, 1, 1, 4 * W.channels / 4 }, new[] { 1, 1, 1, 1 });

            var r_i = ops.StridedSlice(R, new[] { 0, 0, 0, 0 }, new[] { R.batch, 1, 1, R.channels / 4 }, new[] { 1, 1, 1, 1 });
            var r_o = ops.StridedSlice(R, new[] { 0, 0, 0, R.channels / 4 }, new[] { R.batch, 1, 1, 2 * R.channels / 4 }, new[] { 1, 1, 1, 1 });
            var r_f = ops.StridedSlice(R, new[] { 0, 0, 0, 2 * R.channels / 4 }, new[] { R.batch, 1, 1, 3 * R.channels / 4 }, new[] { 1, 1, 1, 1 });
            var r_j = ops.StridedSlice(R, new[] { 0, 0, 0, 3 * R.channels / 4 }, new[] { R.batch, 1, 1, 4 * R.channels / 4 }, new[] { 1, 1, 1, 1 });

            var wb_i = ops.StridedSlice(B, new[] { 0, 0, 0, 0 }, new[] { 1, 1, 1, B.channels / 8 }, new[] { 1, 1, 1, 1 });
            var wb_o = ops.StridedSlice(B, new[] { 0, 0, 0, B.channels / 8 }, new[] { 1, 1, 1, 2 * B.channels / 8 }, new[] { 1, 1, 1, 1 });
            var wb_f = ops.StridedSlice(B, new[] { 0, 0, 0, 2 * B.channels / 8 }, new[] { 1, 1, 1, 3 * B.channels / 8 }, new[] { 1, 1, 1, 1 });
            var wb_j = ops.StridedSlice(B, new[] { 0, 0, 0, 3 * B.channels / 8 }, new[] { 1, 1, 1, 4 * B.channels / 8 }, new[] { 1, 1, 1, 1 });

            var rb_i = ops.StridedSlice(B, new[] { 0, 0, 0, 4 * B.channels / 8 }, new[] { 1, 1, 1, 5 * B.channels / 8 }, new[] { 1, 1, 1, 1 });
            var rb_o = ops.StridedSlice(B, new[] { 0, 0, 0, 5 * B.channels / 8 }, new[] { 1, 1, 1, 6 * B.channels / 8 }, new[] { 1, 1, 1, 1 });
            var rb_f = ops.StridedSlice(B, new[] { 0, 0, 0, 6 * B.channels / 8 }, new[] { 1, 1, 1, 7 * B.channels / 8 }, new[] { 1, 1, 1, 1 });
            var rb_j = ops.StridedSlice(B, new[] { 0, 0, 0, 7 * B.channels / 8 }, new[] { 1, 1, 1, 8 * B.channels / 8 }, new[] { 1, 1, 1, 1 });


            var memSize = r_i.flatHeight;

            var baseLSTMName = layer.outputs[3];
            var initial_h = $"{baseLSTMName}_h";
            var initial_c = $"{baseLSTMName}_c";

            var baseLSTMOutputName = layer.outputs[4];
            var output_h = $"{baseLSTMOutputName}_h";
            var output_c = $"{baseLSTMOutputName}_c";

            var i_mad_w = net.Dense($"{layer.name}_bc_i_mad_w", layer.inputs[0], w_i, wb_i);
            var i_mad_r = net.Dense($"{layer.name}_bc_i_mad_r", initial_h, r_i, rb_i);
            var i_mad = net.Add($"{layer.name}_bc_i_mad", new[] { i_mad_w, i_mad_r });

            var j_mad_w = net.Dense($"{layer.name}_bc_j_mad_w", layer.inputs[0], w_j, wb_j);
            var j_mad_r = net.Dense($"{layer.name}_bc_j_mad_r", initial_h, r_j, rb_j);
            var j_mad = net.Add($"{layer.name}_bc_j_mad", new[] { j_mad_w, j_mad_r });

            var f_mad_w = net.Dense($"{layer.name}_bc_f_mad_w", layer.inputs[0], w_f, wb_f);
            var f_mad_r = net.Dense($"{layer.name}_bc_f_mad_r", initial_h, r_f, rb_f);
            var f_mad = net.Add($"{layer.name}_bc_f_mad", new[] { f_mad_w, f_mad_r });

            var o_mad_w = net.Dense($"{layer.name}_bc_o_mad_w", layer.inputs[0], w_o, wb_o);
            var o_mad_r = net.Dense($"{layer.name}_bc_o_mad_r", initial_h, r_o, rb_o);
            var o_mad = net.Add($"{layer.name}_bc_o_mad", new[] { o_mad_w, o_mad_r });

            var i = net.Sigmoid($"{layer.name}_bc_i_sigmoid", i_mad);
            var j = net.Tanh($"{layer.name}_bc_j_tanh", j_mad);
            var f = net.Sigmoid($"{layer.name}_bc_f_sigmoid", f_mad);
            var o = net.Sigmoid($"{layer.name}_bc_o_sigmoid", o_mad);

            var state_c_mul = net.Mul($"{layer.name}_bc_state_c_mul", new[] { initial_c, f.name });
            var i_j_mul = net.Mul($"{layer.name}_bc_i_j_mul", new[] { i, j });
            var state_c = net.Add(output_c, new[] { state_c_mul, i_j_mul });
            var state_c_tanh = net.Tanh($"{layer.name}_bc_state_c_tanh", state_c);
            var state_h = net.Mul(output_h, new[] { o, state_c_tanh });

            net.Identity(layer.outputs[0], state_h);
            net.Identity(layer.outputs[1], state_h);
            net.Identity(layer.outputs[2], state_c);

            net.Memory(initial_c, state_c, new TensorShape(-1, 1, 1, memSize));
            net.Memory(initial_h, state_h, new TensorShape(-1, 1, 1, memSize));

            return false;
        }
    }
}
