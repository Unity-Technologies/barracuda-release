using System;
using System.Collections.Generic;
using System.Linq;

namespace Unity.Barracuda.Compiler.Passes.Cleanup
{
    class RemoveUnusedLayersPass : IModelPass
    {
        public void Run(ref Model model)
        {
            // TODO: strip layers not useful to compute output
            // Strip unused layers
            var unusedLayers = new HashSet<string>(ModelAnalyzer.FindUnusedLayers(model));
            model.layers = model.layers.Where(l => !unusedLayers.Contains(l.name) || l.flags.HasFlag(Layer.Flags.Preserve)).ToList();
        }
    }
}
