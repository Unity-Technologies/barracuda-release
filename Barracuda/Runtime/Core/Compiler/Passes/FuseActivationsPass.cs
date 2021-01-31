using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Assertions;

namespace Unity.Barracuda.Compiler.Passes.Optimization
{
    class FuseActivationPass : IModelPass
    {
        public void Run(ref Model model)
        {
            //Fused activation
            var fusableActivations = model.layers.Where(l => l.type == Layer.Type.Activation && IsActivationFusable(l.activation)).ToList();
            foreach (var activationLayer in fusableActivations)
            {
                if (activationLayer.inputs.Length != 1)
                    continue;

                var mainLayer = model.layers.Find(l => l.name == activationLayer.inputs[0]);
                if (mainLayer == null)
                    continue;

                if (!IsLayerSupportingActivationFusing(mainLayer.type))
                    continue;

                if (mainLayer.activation != Layer.Activation.None)
                    continue;

                if (model.outputs.Contains(mainLayer.name))
                    continue;

                if (model.memories.Exists(m => m.output == mainLayer.name))
                    continue;

                //Need to check that no other layers uses mainLayer directly.
                //Activation in the graph below can not be fused because (concat) layer needs raw output of (conv) layer
                //conv -> relu -----.
                //    \             v
                //     `---------> concat
                if (model.layers.Exists(l => l != activationLayer && l.inputs.Contains(mainLayer.name)))
                    continue;

                if (activationLayer.flags.HasFlag(Layer.Flags.Preserve))
                    continue;

                FuseActivation(ref model, mainLayer, activationLayer);
            }
        }

        public static bool IsLayerSupportingActivationFusing(Layer.Type layerType)
        {
            return layerType == Layer.Type.Dense ||
                   layerType == Layer.Type.Conv2D ||
                   layerType == Layer.Type.Conv3D ||
                   layerType == Layer.Type.DepthwiseConv2D ||
                   layerType == Layer.Type.Conv2DTrans ||
                   layerType == Layer.Type.Normalization;
        }

        public static bool IsActivationFusable(Layer.Activation activationType)
        {
            var fusedActivationType = (Layer.FusedActivation)activationType;
            return Enum.IsDefined(typeof(Layer.FusedActivation), fusedActivationType);
        }

        static private void FuseActivation(ref Model model, Layer mainLayer, Layer activationToFuse)
        {
            //patch `mainLayer`
            mainLayer.activation = activationToFuse.activation;

            //patch all layers depending on `activationToFuse`
            foreach (var l in model.layers)
            {
                for (int i = 0; i < l.inputs.Length; ++i)
                {
                    if (l.inputs[i] == activationToFuse.name)
                        l.inputs[i] = mainLayer.name;
                }
            }

            //remove `activationToFuse` if not an output, if an output make it an identity layer instead.
            if (model.outputs.Contains(activationToFuse.name) || model.memories.Exists(m => m.output == activationToFuse.name))
            {
                activationToFuse.type = Layer.Type.Nop;
                activationToFuse.activation = Layer.Activation.None;
            }
            else
                model.layers.Remove(activationToFuse);
        }
    }
}
