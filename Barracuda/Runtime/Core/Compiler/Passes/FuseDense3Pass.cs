using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Assertions;

namespace Unity.Barracuda.Compiler.Passes.Optimization
{
    class FuseDense3Pass : IModelPass
    {
        public void Run(ref Model model)
        {
            // MatMul (rank3) + known input -> Add/Sub => Dense3
            var constLayers = new Dictionary<string, Layer>();
            foreach (var l in model.layers)
            {
                if (l.type == Layer.Type.Load)
                    constLayers[l.name] = l;
            }
            var preserve = new HashSet<string>(
                model.memories.Select(mem => mem.input).Concat(
                model.memories.Select(mem => mem.output)).Concat(
                model.outputs));


            var removeLayers = new HashSet<string>();
            var remap = new Dictionary<string, string>();

            for (int l = 0; l < model.layers.Count - 1; ++l)
            {
                Layer layer = model.layers[l];

                List<Layer> downStreamLayers = GetDownStreamLayers(model, layer.name);

                if (!IsLayerDense3(layer, downStreamLayers, constLayers))
                    continue;

                if (preserve.Contains(layer.name) || preserve.Contains(downStreamLayers[0].name))
                    continue;

                string weights = (layer.inputs.Where(x => constLayers.ContainsKey(x)).ToList())[0];
                Layer constWeights = constLayers[weights];
                var weightArray = constWeights.weights;
                var weightShape = constWeights.datasets[0].shape;

                Layer downStreamLayer = downStreamLayers[0];
                string bias = (downStreamLayer.inputs.Where(x => x != layer.name).ToList())[0];
                Layer constBias = constLayers[bias];
                TensorShape biasShape = new TensorShape(1, 1, 1, Mathf.Max(weightShape.channels, constBias.datasets[0].shape.length));
                var biasArray = constBias.weights;

                var inputs = layer.inputs.Where(x => x != weights).ToArray();

                Layer mergedLayer = new Layer(layer.name, Layer.Type.Dense3);

                mergedLayer.inputs = inputs;

                mergedLayer.datasets = new Layer.DataSet[2];
                mergedLayer.datasets[0].name = $"{mergedLayer.name}/W";
                mergedLayer.datasets[0].shape = weightShape;
                mergedLayer.datasets[0].itemSizeInBytes = 4;
                mergedLayer.datasets[0].length = weightShape.length;
                mergedLayer.datasets[0].offset = 0;
                mergedLayer.datasets[1].name = $"{mergedLayer.name}/B";
                mergedLayer.datasets[1].shape = biasShape;
                mergedLayer.datasets[1].itemSizeInBytes = 4;
                mergedLayer.datasets[1].length = biasShape.length;
                mergedLayer.datasets[1].offset = weightShape.length;
                mergedLayer.weights = new float[weightShape.length + biasShape.length];

                weightArray.CopyTo(mergedLayer.weights, 0);
                if (constBias.datasets[0].shape.length == 1)
                {
                    for (int i = 0; i < biasShape.length; i++)
                        mergedLayer.weights[mergedLayer.datasets[1].offset + i] = biasArray[0];
                }
                else
                    biasArray.CopyTo(mergedLayer.weights, mergedLayer.datasets[1].offset);


                model.layers[l] = mergedLayer;

                if (!preserve.Contains(constWeights.name))
                    removeLayers.Add(constWeights.name);
                removeLayers.Add(downStreamLayer.name);
                if (!preserve.Contains(constBias.name))
                    removeLayers.Add(constBias.name);
                remap[downStreamLayer.name] = mergedLayer.name;
            }

            model.layers.RemoveAll(l => removeLayers.Contains(l.name));
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
        }

        List<Layer> GetDownStreamLayers(Model model, string name)
        {
            return model.layers.Where(x => x.inputs.Contains(name)).ToList();
        }

        bool IsLayerDense3(Layer layer, List<Layer> downStreamLayers, Dictionary<string, Layer> constLayers)
        {
            if (layer.type != Layer.Type.MatMul)
                return false;
            if (!(layer.pool.Length == 2 && (layer.pool[0] == 3 && layer.pool[1] < 3)))
                return false;
            if (!(constLayers.ContainsKey(layer.inputs[0]) || constLayers.ContainsKey(layer.inputs[1])))
                return false;
            if (downStreamLayers.Count != 1)
                return false;
            Layer downstreamLayer = downStreamLayers[0];
            if (!(downstreamLayer.type == Layer.Type.Add || downstreamLayer.type == Layer.Type.Sub))
                return false;
            string input = (downstreamLayer.inputs.Where(x => x != layer.name).ToList())[0];
            if (!constLayers.ContainsKey(input))
                return false;
            Layer constAdd = constLayers[input];
            if (constAdd.axis > 1)
                return false;
            return true;
        }
    }
}
