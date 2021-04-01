using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Assertions;

namespace Unity.Barracuda.Compiler.Passes
{
    class ShapeContractionPass : IModelPass
    {
        public void Run(ref Model model)
        {
            if (!model.layout.Contains("NCHW"))
                return;

            var remap = new Dictionary<string, string>();

            for (int l = 1; l < model.layers.Count; ++l)
            {
                var previousLayer = model.layers[l - 1];
                var layer = model.layers[l];

                if (layer.flags.HasFlag(Layer.Flags.Preserve))
                    continue;

                string[] layerInputs = layer.inputs;
                for (int i = 0; i < layerInputs.Length; i++)
                {
                    if (remap.TryGetValue(layerInputs[i], out string replacement))
                        layerInputs[i] = replacement;
                }

                if (previousLayer.type == Layer.Type.Shape
                    && layer.type == Layer.Type.Gather)
                {
                    string indicesInput = layer.inputs[1];
                    var indicesConstant = model.layers.FirstOrDefault(c => c.type == Layer.Type.Load && c.name == indicesInput);
                    if (indicesConstant != null)
                    {
                        Tensor indices = indicesConstant.DataSetToTensor(0);
                        if (indices.length == 1) // Shape only supports selecting one axis in place of the full shape
                        {
                            // Update the axis on the shape layer
                            previousLayer.axis = (int)indices[0];
                            remap[layer.name] = previousLayer.name;
                        }
                    }
                }
                else if (previousLayer.type == Layer.Type.Shape
                    && layer.type == Layer.Type.ConstantOfShape)
                {
                    layer.axis = 1;
                    layer.type = Layer.Type.ConstantOfShape;
                    layer.inputs[0] = previousLayer.inputs[0];
                    remap[previousLayer.name] = layer.name;
                }
            }

            var removeLayers = remap.Keys;
            model.layers.RemoveAll(l => removeLayers.Contains(l.name));
        }
    }
}
