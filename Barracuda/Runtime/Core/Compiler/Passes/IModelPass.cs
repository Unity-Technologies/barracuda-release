using System.Collections.Generic;

namespace Unity.Barracuda.Compiler.Passes
{
    interface IModelPass
    {
        /// <summary>
        /// Run a pass over the whole model modifying in-place
        /// </summary>
        /// <param name="model">Model to modify</param>
        void Run(ref Model model);
    }
}
