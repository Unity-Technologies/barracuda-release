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

                if (remap.ContainsKey(layer.name)) // This layer will get removed
                    continue;

                string[] layerInputs = layer.inputs;
                for (int i = 0; i < layerInputs.Length; i++)
                {
                    if (remap.TryGetValue(layerInputs[i], out string replacement))
                        layerInputs[i] = replacement;
                }

                if (layer.flags.HasFlag(Layer.Flags.Preserve) || nextLayer.flags.HasFlag(Layer.Flags.Preserve))
                    continue;

                // Only concatenate serial transpose layers
                if (layer.type == Layer.Type.Transpose
                    && nextLayer.type == Layer.Type.Transpose
                    && nextLayer.inputs.Contains(layer.name)
                    && layerReferences.TryGetValue(layer.name, out int references)
                    && references <= 1
                    && layerReferences.TryGetValue(nextLayer.name, out references)
                    && references <= 1)
                {
                    int[] permutations = new int[] { 0, 1, 2, 3, 4, 5, 6, 7 };
                    if (layer.pool.Length == 4)
                    {
                        permutations[2] = TensorExtensions.Convert4DTo8DAxis(layer.pool[0]);
                        permutations[5] = TensorExtensions.Convert4DTo8DAxis(layer.pool[1]);
                        permutations[6] = TensorExtensions.Convert4DTo8DAxis(layer.pool[2]);
                        permutations[7] = TensorExtensions.Convert4DTo8DAxis(layer.pool[3]);
                    }
                    else
                    {
                        permutations[0] = layer.pool[0];
                        permutations[1] = layer.pool[1];
                        permutations[2] = layer.pool[2];
                        permutations[3] = layer.pool[3];
                        permutations[4] = layer.pool[4];
                        permutations[5] = layer.pool[5];
                        permutations[6] = layer.pool[6];
                        permutations[7] = layer.pool[7];
                    }

                    int[] combinePermutations = new int[] { 0, 1, 2, 3, 4, 5, 6, 7 };
                    if (nextLayer.pool.Length == 4)
                    {
                        combinePermutations[2] = TensorExtensions.Convert4DTo8DAxis(nextLayer.pool[0]);
                        combinePermutations[5] = TensorExtensions.Convert4DTo8DAxis(nextLayer.pool[1]);
                        combinePermutations[6] = TensorExtensions.Convert4DTo8DAxis(nextLayer.pool[2]);
                        combinePermutations[7] = TensorExtensions.Convert4DTo8DAxis(nextLayer.pool[3]);
                    }
                    else
                    {
                        combinePermutations[0] = nextLayer.pool[0];
                        combinePermutations[1] = nextLayer.pool[1];
                        combinePermutations[2] = nextLayer.pool[2];
                        combinePermutations[3] = nextLayer.pool[3];
                        combinePermutations[4] = nextLayer.pool[4];
                        combinePermutations[5] = nextLayer.pool[5];
                        combinePermutations[6] = nextLayer.pool[6];
                        combinePermutations[7] = nextLayer.pool[7];
                    }


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
