using System;
using System.Collections.Generic;
using System.Linq; // ToArray(), ToDictionary()
using UnityEngine.Assertions;

namespace Barracuda 
{

public class ModelOptimizer
{
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

} // namespace Barracuda
