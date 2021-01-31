using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Assertions;

namespace Unity.Barracuda.Compiler.Passes.Optimization
{
    class FuseLinearLayersPass : IModelPass
    {
        public void Run(ref Model model)
        {
            var constantLayers = new Dictionary<string, Layer>();
            foreach (var l in model.layers)
            {
                if (IsLayerConstant(l))
                    constantLayers[l.name] = l;
            }

            // pack mathops const inputs into layer database
            PackConstantsForMathOps(model, constantLayers);

            var remap = new Dictionary<string, string>();
            var mergedLayers = new HashSet<Layer>();

            for (int l = 0; l < model.layers.Count; ++l)
            {
                var layer = model.layers[l];

                bool isLayerLinear = LinearLayerFusing.IsLayerLinear(layer, constantLayers);
                bool isLayerPreserved = layer.flags.HasFlag(Layer.Flags.Preserve);
                bool layerHasActivation = IsLayerFusedActivation(layer);

                if (!isLayerLinear)
                    continue;

                // if layer has an activation, we fuse it, but treat it as non linear for future children
                if (!layerHasActivation)
                {
                    remap[layer.name] = layer.name;
                }

                // Multi input nodes can only fuse constants and same inputs
                // only merge constants. @TODO: fuse equal input nodes
                var nonLinearInputs = layer.inputs.Where(x => !remap.ContainsKey(x) && !constantLayers.ContainsKey(x)).ToList();
                var linearInputs = layer.inputs.Where(x => remap.ContainsKey(x)).ToList();

                // merge layer with one linearInput and eventual constants
                if (nonLinearInputs.Count > 0 || linearInputs.Count > 1)
                    continue;

                var input = linearInputs[0];

                // input is a linear layer, fuse it
                int inputLayerIndex = model.layers.FindIndex(x => x.name == remap[input]);
                Layer inputLayer = model.layers[inputLayerIndex];

                if (!AreLayersFusable(inputLayer, layer))
                    continue;

                // convention: layer will be fused into inputLayer
                // => fused layer will have the same inputs as inputLayer
                Layer fusedLayer = FuseConsecutiveLayers(inputLayer, layer);

                if (LayerComplextity(fusedLayer) > LayerComplextity(inputLayer) + LayerComplextity(layer))
                    continue;

                if (layerHasActivation)
                {
                    fusedLayer.activation = layer.activation;
                }

                bool hasNoSkipConnection = (model.GetDownStreamLayersCount(input) == 1);
                //  if input has more than 1 child, we can't override input with fused result
                //  same if input is preserved
                if (!hasNoSkipConnection || model.layers.Any(p => p.flags.HasFlag(Layer.Flags.Preserve) && p.name == input))
                {
                    fusedLayer.name = layer.name;
                    model.layers[l] = fusedLayer;
                    continue;
                }

                // preserve layer if output/memory
                if (isLayerPreserved)
                {
                    // cannot merge layer into input:
                    // remove input, no need to remap as inputs == input.inputs
                    fusedLayer.name = layer.name;
                    mergedLayers.Add(inputLayer);
                    model.layers[l] = fusedLayer;
                }
                else
                {
                    // merge layer into input
                    // remove current and remap input names
                    mergedLayers.Add(layer);
                    remap[layer.name] = fusedLayer.name;
                    model.layers[inputLayerIndex] = fusedLayer;
                }
            }

            // remove merged layers
            model.layers.RemoveAll(x => mergedLayers.Contains(x));

            // update remapped inputs
            for (int l = 0; l < model.layers.Count; ++l)
            {
                Layer layer = model.layers[l];
                for (int i = 0; i < layer.inputs.Length; ++i)
                {
                    var input = layer.inputs[i];
                    if (remap.ContainsKey(input))
                        model.layers[l].inputs[i] = remap[input];
                }
            }

            // unpack maths ops const inputs into new const layer
            UnpackConstantsForMathOps(model);

            // remove unused constants
            foreach (var l in model.layers)
                foreach (var i in l.inputs)
                {
                    if (constantLayers.ContainsKey(i))
                        constantLayers.Remove(i);
                }
            model.layers.RemoveAll(x => constantLayers.ContainsKey(x.name) &&
                                        !x.flags.HasFlag(Layer.Flags.Preserve));
        }

        public static bool IsLayerConstant(Layer layer)
        {
            return layer.type == Layer.Type.Load;
        }
        static bool IsLayerFusedActivation(Layer layer)
        {
            return layer.activation != Layer.Activation.None;
        }

        static StaticLayerOppComplexity m_LayerComplexity = new StaticLayerOppComplexity();
        static long LayerComplextity(Layer l) { return m_LayerComplexity.LayerComplextity(l); }

        static LinearLayerFusing linearLayerFuser = new LinearLayerFusing();
        static Layer FuseConsecutiveLayers(Layer previous, Layer current)
        {
            return linearLayerFuser.FuseLayers(previous, current);
        }
        static bool AreLayersFusable(Layer l0, Layer l1)
        {
            // can't fuse if input has a fused activation or if fusing code not implemented
            return !IsLayerFusedActivation(l0) && linearLayerFuser.AreLayersFusable(l0, l1);
        }

        private static void PackConstantsForMathOps(Model model, Dictionary<string, Layer> constantLayers)
        {
            for (int l = 0; l < model.layers.Count; ++l)
            {
                var layer = model.layers[l];

                if (!LinearLayerFusing.IsLayerLinearMathOp(layer))
                    continue;
                var constInputs = layer.inputs.Count(x => constantLayers.ContainsKey(x));
                // @TODO fuse multi const inputs here
                if (!(layer.inputs.Length == 2 && constInputs == 1))
                    continue;

                var constInput = layer.inputs.ToList().Find(x => constantLayers.ContainsKey(x));

                layer.datasets = new Layer.DataSet[constantLayers[constInput].datasets.Length];
                Array.Copy(constantLayers[constInput].datasets, layer.datasets, constantLayers[constInput].datasets.Length);
                layer.weights = new float[constantLayers[constInput].weights.Length];
                Array.Copy(constantLayers[constInput].weights, layer.weights, constantLayers[constInput].weights.Length);

                layer.axis = constantLayers[constInput].axis; // rank TODO name correctly

                model.layers[l].inputs = layer.inputs.Where(x => x != constInput).ToArray();
            }
        }

        private static void UnpackConstantsForMathOps(Model model)
        {
            List<Layer> newConstants = new List<Layer>();
            for (int l = 0; l < model.layers.Count; ++l)
            {
                var layer = model.layers[l];
                if (!LinearLayerFusing.IsLayerLinearMathOp(layer))
                    continue;

                if (layer.datasets == null || layer.datasets.Length != 1)
                    continue;

                var name = "c" + layer.name;
                Layer constInput = new Layer(name, Layer.Type.Load);

                constInput.datasets = new Layer.DataSet[layer.datasets.Length];
                Array.Copy(layer.datasets, constInput.datasets, layer.datasets.Length);
                for (int d = 0; d < constInput.datasets.Length; ++d)
                    constInput.datasets[d].name = name;

                constInput.weights = new float[layer.weights.Length];
                Array.Copy(layer.weights, constInput.weights, layer.weights.Length);

                constInput.axis = layer.axis; // rank TODO rename

                Array.Resize(ref layer.inputs, layer.inputs.Length + 1);
                layer.inputs[layer.inputs.Length - 1] = constInput.name;

                newConstants.Add(constInput);

                layer.datasets = new Layer.DataSet[0];
                layer.weights = new float[0];
            }
            newConstants.AddRange(model.layers);
            model.layers = newConstants;
        }
    }
}
