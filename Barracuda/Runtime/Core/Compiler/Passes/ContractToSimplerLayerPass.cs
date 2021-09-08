using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Assertions;

namespace Unity.Barracuda.Compiler.Passes.Optimization
{
    class ContractToSimplerLayerPass : IModelPass
    {
        public void Run(ref Model model)
        {
            for (int l = 0; l < model.layers.Count; ++l)
            {
                Layer layer = model.layers[l];

                if (layer.type == Layer.Type.Concat)
                {
                    model.layers[l] = ContractConcat(layer);
                }
            }
        }

        private Layer ContractConcat(Layer layer)
        {
            if (layer.inputs.Any(o => o != layer.inputs[0]))
                return layer;

            Layer newLayer = new Layer(layer.name, Layer.Type.Tile);

            newLayer.type = Layer.Type.Tile;

            newLayer.pool = new[] { 1, 1, 1, 1, 1, 1, 1, 1 };
            newLayer.pool[layer.axis] = layer.inputs.Length;
            newLayer.inputs = new[] { layer.inputs[0] };

            return newLayer;
        }
    }
}
