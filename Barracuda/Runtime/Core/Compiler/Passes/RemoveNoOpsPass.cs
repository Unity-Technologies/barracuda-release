using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Assertions;

namespace Unity.Barracuda.Compiler.Passes.Cleanup
{
    // TODO remove useless patterns:
    // Reduce keepdim 0 -> * -> Reshape
    class RemoveNoOpsPass : IModelPass
    {
        public void Run(ref Model model)
        {
            var noopLayers = new List<Layer>();
            var remap = new Dictionary<string, string>();

            // algorithm:
            // - if input is pointing to a noop, we need to remap it to upstream layer
            // - if layer is a noop, store its link to upstream layer
            // layers are in order of appearance, so if layer_N has layer_M as input, we'd have treated layer_M before
            for (int l = 0; l < model.layers.Count; ++l)
            {
                var layer = model.layers[l];

                // replace removed layers with their upstream inputs
                for (int i = 0; i < layer.inputs.Length; ++i)
                {
                    var input = layer.inputs[i];
                    if (remap.ContainsKey(input))
                    {
                        Assert.IsTrue(noopLayers.Any(x => input == x.name));
                        model.layers[l].inputs[i] = remap[input];
                    }
                    else
                    {
                        Assert.IsFalse(noopLayers.Any(x => input == x.name));
                    }
                }

                if (layer.flags.HasFlag(Layer.Flags.Preserve))
                    continue;

                if (layer.inputs.Length == 0) // const
                    continue;

                // if layer is noop = nop, identity or flatten
                if (IsLayerNoop(layer))
                {
                    Assert.IsTrue(layer.inputs.Length == 1); // noop layers have only 1 input
                    remap[layer.name] = layer.inputs[0];
                    noopLayers.Add(layer);
                }
            }

            foreach (var l in noopLayers)
            {
                model.layers.Remove(l);
            }
        }

        public static bool IsPermutationNoop(int[] permutations)
        {
            for (int i = 0; i < permutations.Length; ++i)
                if (permutations[i] != i)
                    return false;
            return true;
        }

        public static bool IsLayerNoop(Layer layer)
        {
            // LSTM outputs, TODO remove?
            // TODO: move this in IsLayerLSTMRelated
            if (layer.activation == Layer.Activation.None && layer.pad.Length > 0
                && layer.name.IndexOf("lstm", StringComparison.OrdinalIgnoreCase) >= 0)
            {
                return false;
            }

            return layer.type == Layer.Type.Nop ||
                   (layer.type == Layer.Type.Activation && layer.activation == Layer.Activation.None) ||
                   (layer.type == Layer.Type.Transpose && IsPermutationNoop(layer.pool) ||
                   (layer.type == Layer.Type.StridedSlice
                        // Nothing is actually being done in this case since it is the full range with single stepping, so skip it
                        && layer.pad.All(s => s == 0)
                        && layer.pool.All(e => e == int.MaxValue)
                        && layer.stride.All(s => s == 1))) ||
                   (layer.type == Layer.Type.Transpose && Enumerable.SequenceEqual(layer.pool, new [] { 0, 1, 2, 3 })) ||
                   (layer.type == Layer.Type.Expand && layer.inputs.Length == 1 && layer.pool.Length >= 1 && layer.pool.All(x => x == 1));
        }
    }
}
