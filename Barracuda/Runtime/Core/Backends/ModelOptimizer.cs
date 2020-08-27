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
        RemoveUnused(model, keepLayers);

        if (allowFusing)
        {
            FuseLinear(model, keepLayers);
            FuseActivations(model);
        }

        return model;
    }

    public static void RemoveUnused(Model model, HashSet<string> keepLayers)
    {
        // TODO: strip layers not useful to compute output
        var preserve = new HashSet<string>(
            model.memories.Select(mem => mem.input).Concat(
            model.memories.Select(mem => mem.output)).Concat(
            model.outputs));

        // Strip unused layers
        var unusedLayers = new HashSet<string>(ModelAnalyzer.FindUnusedLayers(model));
        if (keepLayers != null) // Except explicitly specified for keeping
            unusedLayers.ExceptWith(keepLayers);
        model.layers = model.layers.Where(l => !unusedLayers.Contains(l.name) || preserve.Contains(l.name)).ToList();
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

    static bool IsLayerNoop(Layer layer)
    {
        return layer.type == Layer.Type.Nop ||
               layer.type == Layer.Type.Flatten ||
               layer.type == Layer.Type.Activation && layer.activation == Layer.Activation.None ||
               layer.type == Layer.Type.Transpose && (layer.pool[0] == 0 && layer.pool[1] == 1 && layer.pool[2] == 2 && layer.pool[3] == 3);
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
            if (IsLayerNoop(layer))
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


    static bool IsLayerConstant(Layer layer)
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

    private static void PackConstants(Model model, Dictionary<string, Layer> constantLayers)
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

             model.layers[l].inputs = layer.inputs.Where(x => x != constInput).ToArray();
        }
    }

    private static void UnpackConstants(Model model)
    {
        List<Layer> newConstants = new List<Layer>();
        for (int l = 0; l < model.layers.Count; ++l)
        {
            var layer = model.layers[l];
            if(!LinearLayerFusing.IsLayerLinearMathOp(layer))
                continue;

            if (layer.datasets == null || layer.datasets.Length != 1)
                continue;

            Layer constInput = new Layer("c" + layer.name,Layer.Type.Load);

            constInput.datasets = new Layer.DataSet[layer.datasets.Length];
            Array.Copy(layer.datasets, constInput.datasets, layer.datasets.Length);
            for(int d = 0; d < constInput.datasets.Length; ++d)
                constInput.datasets[d].name = "";

            constInput.weights = new float[layer.weights.Length];
            Array.Copy(layer.weights, constInput.weights, layer.weights.Length);

            Array.Resize(ref layer.inputs, layer.inputs.Length + 1);
            layer.inputs[layer.inputs.Length-1] = constInput.name;

            newConstants.Add(constInput);

            layer.datasets = new Layer.DataSet[0];
            layer.weights = new float[0];
        }
        newConstants.AddRange(model.layers);
        model.layers = newConstants;
    }

    public static void FuseLinear(Model model, HashSet<string> keepLayers = null)
    {
        // outputs and memories can be queried by the user, make sure they are not removed
        var preserve = new HashSet<string>(
            model.memories.Select(mem => mem.input).Concat(
            model.memories.Select(mem => mem.output)).Concat(
            model.outputs));

        var constantLayers = new Dictionary<string, Layer>();
        foreach (var l in model.layers)
        {
            if (IsLayerConstant(l))
                constantLayers[l.name] = l;
        }

        // pack constants into layer database
        PackConstants(model, constantLayers);

        var remap = new Dictionary<string, string>();
        var mergedLayers = new HashSet<Layer>();

        for (int l = 0; l < model.layers.Count; ++l)
        {
            var layer = model.layers[l];

            bool isLayerLinear = LinearLayerFusing.IsLayerLinear(layer, constantLayers);
            bool isLayerPreserved = preserve.Contains(layer.name);
            bool layerHasActivation = IsLayerFusedActivation(layer);

            if(!isLayerLinear)
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

            if(!AreLayersFusable(inputLayer, layer))
                continue;

            // convention: layer will be fused into inputLayer
            // => fused layer will have the same inputs as inputLayer
            Layer fusedLayer = FuseConsecutiveLayers(inputLayer, layer);

            if(LayerComplextity(fusedLayer) > LayerComplextity(inputLayer) + LayerComplextity(layer))
                continue;

            if (layerHasActivation)
            {
                fusedLayer.activation = layer.activation;
            }

            bool hasNoSkipConnection = (model.GetDownStreamLayersCount(input) == 1);
            //  if input has more than 1 child, we can't override input with fused result
            //  same if input is preserved
            if (!hasNoSkipConnection || preserve.Contains(input)) 
            {
                fusedLayer.name = layer.name;
                model.layers[l] = fusedLayer;
                continue;
            }

            // preserve layer if output/memory
            if(isLayerPreserved)
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
                if(remap.ContainsKey(input))
                    model.layers[l].inputs[i] = remap[input];
            }
        }

        // unpack constants
        UnpackConstants(model);

        // remove unused constants
        foreach (var l in model.layers)
            foreach (var i in l.inputs)
            {
                if (constantLayers.ContainsKey(i))
                    constantLayers.Remove(i);
            }
        model.layers.RemoveAll(x => constantLayers.ContainsKey(x.name) &&
                                    !preserve.Contains(x.name) &&
                                    (keepLayers == null ? true : !keepLayers.Contains(x.name)));
    }
}

} // namespace Unity.Barracuda
