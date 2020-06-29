using System;
using System.Collections.Generic;
using System.Linq; // ToArray(), ToDictionary()
using UnityEngine.Assertions;

namespace Unity.Barracuda
{

public class ModelOptimizer
{
    static public Model Optimize(Model model, bool allowFusing, HashSet<string> keepLayers = null)
    {
        // Strip unused layers
        var unusedLayers = new HashSet<string>(ModelAnalyzer.FindUnusedLayers(model));
        if (keepLayers != null) // Except explicitly specified for keeping
            unusedLayers.ExceptWith(keepLayers);
        model.layers = model.layers.Where(l => !unusedLayers.Contains(l.name)).ToList();

        if (allowFusing)
        {
            FuseActivations(model);
        }

        return model;
    }

    public static bool IsLayerSupportingActivationFusing(Layer.Type layerType)
    {
        return layerType == Layer.Type.Dense ||
               layerType == Layer.Type.Conv2D ||
               layerType == Layer.Type.DepthwiseConv2D ||
               layerType == Layer.Type.Conv2DTrans ||
               layerType == Layer.Type.Normalization;
    }

    public static bool IsActivationFusable(Layer.Activation activationType)
    {
        var fusedActivationType = (Layer.FusedActivation) activationType;
        return Enum.IsDefined(typeof(Layer.FusedActivation), fusedActivationType);
    }

    static private void FuseActivation(Model model, Layer mainLayer, Layer activationToFuse)
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

    static public void FuseActivations(Model model)
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

            FuseActivation(model, mainLayer, activationLayer);
        }
    }

    public static Model RemoveNoop(Model model)
    {
        var noopLayers = new List<Layer>();
        var remap = new Dictionary<string, string>();

        // outputs and memories can be queried by the user, make sure they are not removed
        var preserve = new HashSet<string>(
            model.memories.Select(mem => mem.input).Concat(
            model.memories.Select(mem => mem.output)).Concat(
            model.outputs));

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

            if (preserve.Contains(layer.name))
                continue;

            if (layer.inputs.Length == 0) // const
                continue;

            // if layer is noop = nop, identity or flatten
            if (layer.type == Layer.Type.Nop ||
                layer.type == Layer.Type.Flatten ||
                (layer.type == Layer.Type.Activation && layer.activation == Layer.Activation.None))
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

        return model;
    }
}

} // namespace Unity.Barracuda
