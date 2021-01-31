using System;
using System.Collections.Generic;
using System.Linq;

namespace Unity.Barracuda.Compiler.Passes
{
    class IntermediateToRunnableNHWCPass : IModelPass
    {
        public void Run(ref Model model)
        {
            var shapeInferencePass = new IRShapeInferenceAndConstantFusing();
            shapeInferencePass.Run(ref model);

            // Optimization
            var linearLayerFusingPass = new Optimization.FuseLinearLayersPass();
            linearLayerFusingPass.Run(ref model);
            var activationFusingPass = new Optimization.FuseActivationPass();
            activationFusingPass.Run(ref model);

            // Cleanup
            var removeUnusedPass = new Cleanup.RemoveUnusedLayersPass();
            removeUnusedPass.Run(ref model);
            var removeNoOpPass = new Cleanup.RemoveNoOpsPass();
            removeNoOpPass.Run(ref model);

            // TODO, put asserts in ImporterWarning?
            var warnings = new List<Model.ImporterWarning>();
            var validateNCHWPass = new ValidateNCHWPass();
            validateNCHWPass.Run(model, ref warnings);

            // to runnable NHWC
            var nhwcPass = new NCHWToNHWCPass();
            nhwcPass.Run(ref model);

            var validateNHWCPass = new ValidateNHWCPass();
            validateNHWCPass.Run(model, ref warnings);

            model.Warnings.AddRange(warnings);
        }
    }
}
