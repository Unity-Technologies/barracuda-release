using System.Collections.Generic;

namespace Unity.Barracuda.Compiler.Passes
{
    interface IValidateModelPass
    {
        /// <summary>
        /// Run a pass over the whole model
        /// </summary>
        /// <param name="model">Model to validate</param>
        void Run(Model model, ref List<Model.ImporterWarning> warnings);
    }
}
