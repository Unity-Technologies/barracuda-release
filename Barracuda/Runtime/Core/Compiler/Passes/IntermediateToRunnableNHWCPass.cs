using System;
using System.Collections.Generic;
using System.Linq;

namespace Unity.Barracuda.Compiler.Passes
{
    class IntermediateToRunnableNHWCPass : IModelPass
    {
        public bool Optimize { get; set; } = false;

        public void Run(ref Model model)
        {
            var warnings = new List<Model.ImporterWarning>();
            var shapeInferencePass = new IRShapeInferenceAndConstantFusing();
            shapeInferencePass.Run(ref model, warnings);

            if (Optimize)
            {
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
            }

            // TODO, put asserts in ImporterWarning?
            var validateNCHWPass = new ValidateNCHWPass();
            validateNCHWPass.Run(model, ref warnings);

            // to runnable NHWC
            var nhwcPass = new NCHWToNHWCPass();
            nhwcPass.Run(ref model);

            // optimizations
            if (Optimize)
            {
                var contractToSimplerLayerPass = new Optimization.ContractToSimplerLayerPass();
                contractToSimplerLayerPass.Run(ref model);

                var concatenateTransposesPass = new Optimization.ConcatenateTransposesPass();
                concatenateTransposesPass.Run(ref model);

                var dense3FusingPass = new Optimization.FuseDense3Pass();
                dense3FusingPass.Run(ref model);
            }

            var validateNHWCPass = new ValidateNHWCPass();
            validateNHWCPass.Run(model, ref warnings);

            model.Warnings.AddRange(warnings);
        }
    }
}
