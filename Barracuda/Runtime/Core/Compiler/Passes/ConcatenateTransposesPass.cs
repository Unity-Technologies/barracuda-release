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

            var transposeReferences = new Dictionary<string, int>();
            var layerDownstreamCounts = new Dictionary<string, int>();
            for (int l = 0; l < model.layers.Count; ++l)
            {
                Layer layer = model.layers[l];

                layerDownstreamCounts[layer.name] = 0;

                foreach (var input in layer.inputs)
                {
                    if (layerDownstreamCounts.ContainsKey(input))
                        layerDownstreamCounts[input] += 1;
                }

                if (layer.type != Layer.Type.Transpose)
                    continue;

                transposeReferences[layer.name] = l;
            }

            var remap = new Dictionary<string, string>();

            for (int l = 0; l < model.layers.Count; ++l)
            {
                Layer layer = model.layers[l];
                if (layer.type != Layer.Type.Transpose)
                    continue;

                string input = layer.inputs[0];

                if (!transposeReferences.ContainsKey(input))
                    continue;

                if (layerDownstreamCounts[input] != 1)
                    continue;

                Layer previousLayer = model.layers[transposeReferences[input]];

                if (previousLayer.flags.HasFlag(Layer.Flags.Preserve) && layer.flags.HasFlag(Layer.Flags.Preserve))
                    continue;

                // previous layer is a transpose and current layer is the only downstream layer
                var permutations = MergeTranspose(previousLayer.pool, layer.pool);

                bool reverseMerge = previousLayer.flags.HasFlag(Layer.Flags.Preserve);

                // merge previous into current unless previous cannot be removed, else reverse
                if (reverseMerge)
                {
                    remap[layer.name] = previousLayer.name;
                    previousLayer.pool = permutations;
                }
                else
                {
                    remap[previousLayer.name] = layer.name;
                    layer.pool = permutations;
                    layer.inputs = previousLayer.inputs.ToArray();
                }
            }

            for (int l = 0; l < model.layers.Count; ++l)
            {
                Layer layer = model.layers[l];
                for (int i = 0; i < layer.inputs.Length; i++)
                {
                    var input = layer.inputs[i];
                    if (remap.ContainsKey(input))
                        model.layers[l].inputs[i] = remap[input];
                }
            }

            model.layers.RemoveAll(l => remap.ContainsKey(l.name));
        }

        int[] MergeTranspose(int[] transpose0, int[] tranpose1)
        {
            int[] permutations = new int[] { 0, 1, 2, 3, 4, 5, 6, 7 };
            if (transpose0.Length == 4)
            {
                permutations[2] = TensorExtensions.Convert4DTo8DAxis(transpose0[0]);
                permutations[5] = TensorExtensions.Convert4DTo8DAxis(transpose0[1]);
                permutations[6] = TensorExtensions.Convert4DTo8DAxis(transpose0[2]);
                permutations[7] = TensorExtensions.Convert4DTo8DAxis(transpose0[3]);
            }
            else
            {
                permutations[0] = transpose0[0];
                permutations[1] = transpose0[1];
                permutations[2] = transpose0[2];
                permutations[3] = transpose0[3];
                permutations[4] = transpose0[4];
                permutations[5] = transpose0[5];
                permutations[6] = transpose0[6];
                permutations[7] = transpose0[7];
            }

            int[] combinePermutations = new int[] { 0, 1, 2, 3, 4, 5, 6, 7 };
            if (tranpose1.Length == 4)
            {
                combinePermutations[2] = TensorExtensions.Convert4DTo8DAxis(tranpose1[0]);
                combinePermutations[5] = TensorExtensions.Convert4DTo8DAxis(tranpose1[1]);
                combinePermutations[6] = TensorExtensions.Convert4DTo8DAxis(tranpose1[2]);
                combinePermutations[7] = TensorExtensions.Convert4DTo8DAxis(tranpose1[3]);
            }
            else
            {
                combinePermutations[0] = tranpose1[0];
                combinePermutations[1] = tranpose1[1];
                combinePermutations[2] = tranpose1[2];
                combinePermutations[3] = tranpose1[3];
                combinePermutations[4] = tranpose1[4];
                combinePermutations[5] = tranpose1[5];
                combinePermutations[6] = tranpose1[6];
                combinePermutations[7] = tranpose1[7];
            }

            permutations = TensorExtensions.Permute(permutations, combinePermutations);

            return permutations;
        }
    }
}
