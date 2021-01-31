using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace Unity.Barracuda.Compiler.Passes
{
    class PreserveLayersPass : IModelPass
    {
        public void Run(ref Model model)
        {
            // outputs and memories can be queried by the user, make sure they are not removed
            IEnumerable<string> preserve = model.memories.Select(mem => mem.input).Concat(
                model.memories.Select(mem => mem.output)).Concat(
                model.outputs);

            foreach (Layer l in model.layers)
            {
                if (preserve.Contains(l.name))
                    l.flags |= Layer.Flags.Preserve;
            }
        }
    }
}
