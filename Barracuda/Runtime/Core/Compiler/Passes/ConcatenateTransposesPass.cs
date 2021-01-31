using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Assertions;

namespace Unity.Barracuda.Compiler.Passes.Optimization
{
    class ConcatenateTransposesPass : IModelPass
    {
        public void Run(ref Model model)
        {
            int previousLayerCount;
            do
            {
                // Continue to reduce until no more reductions can happen
                previousLayerCount = model.layers.Count;
                ConcatenateTransposes(ref model);
            } while (model.layers.Count < previousLayerCount);
        }

        void ConcatenateTransposes(ref Model model)
        {
            var remap = new Dictionary<string, string>();

            var layerReferences = new Dictionary<string, int>();
            for (int l = 0; l < model.layers.Count - 1; ++l)
            {
                Layer layer = model.layers[l];
                string[] layerInputs = layer.inputs;

                for (int i = 0; i < layerInputs.Length; i++)
                {
                    if (layerReferences.TryGetValue(layerInputs[i], out int count))
                        count++;
                    else
                        count = 0;

                    layerReferences[layerInputs[i]] = count;
                }
            }

            for (int l = 0; l < model.layers.Count - 1; ++l)
            {
                var layer = model.layers[l];
                var nextLayer = model.layers[l + 1];

                if (layer.flags.HasFlag(Layer.Flags.Preserve))
                    continue;

                if (remap.ContainsKey(layer.name)) // This layer will get removed
                    continue;

                string[] layerInputs = layer.inputs;
                for (int i = 0; i < layerInputs.Length; i++)
                {
                    if (remap.TryGetValue(layerInputs[i], out string replacement))
                        layerInputs[i] = replacement;
                }

                // Only concatenate serial transpose layers
                if (layer.type == Layer.Type.Transpose
                    && nextLayer.type == Layer.Type.Transpose
                    && nextLayer.inputs.Contains(layer.name)
                    && layerReferences.TryGetValue(layer.name, out int references)
                    && references <= 1
                    && layerReferences.TryGetValue(nextLayer.name, out references)
                    && references <= 1)
                {
                    int[] permutations = layer.pool;
                    int[] combinePermutations = nextLayer.pool;

                    permutations = TensorExtensions.Permute(permutations, combinePermutations);
                    layer.pool = permutations;

                    remap[nextLayer.name] = layer.name;
                }
            }

            var removeLayers = remap.Keys;
            model.layers.RemoveAll(l => removeLayers.Contains(l.name));
        }
    }
}
