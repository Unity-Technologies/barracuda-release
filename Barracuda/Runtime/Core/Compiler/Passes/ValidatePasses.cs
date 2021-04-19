using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEditor;

namespace Unity.Barracuda.Compiler.Passes
{
    internal enum MessageType
    {
        None = 0,
        Info = 1,
        Warning = 2,
        Error = 3
    }

    class ValidationHelper
    {
        public static void AppendWarning(bool condition, string layer, string message, ref List<Model.ImporterWarning> warnings, MessageType level = MessageType.Info)
        {
            if (!condition)
                warnings.Add(new Model.ImporterWarning(layer, $"MessageType.{(int)level}" + message));
        }
    }

    class ValidateNCHWShapesPass : IValidateModelPass
    {
        public void Run(Model model, ref List<Model.ImporterWarning> warnings)
        {
            var modelTemp = model.ShallowCopy();
            IDictionary<string, TensorShape> inputShapes = new Dictionary<string, TensorShape>();
            // force batch to 1
            for (int i = 0; i < modelTemp.inputs.Count; i++)
            {
                var input = modelTemp.inputs[i];
                var shape = input.shape.ToArray();
                if (shape[TensorShape.DataBatch] <= 0)
                    shape[TensorShape.DataBatch] = 1;
                input.shape = shape;
                modelTemp.inputs[i] = input;

                if (!ModelAnalyzer.IsInputShapeAcceptablyKnowForShapeInference(input))
                    continue;

                inputShapes[input.name] = new TensorShape(input.shape);
            }

            ValidationHelper.AppendWarning(inputShapes.Count == modelTemp.inputs.Count, "model", "Input Shape: unkown non batch dimension", ref warnings);

            IRShapeInferenceAndConstantFusing shapeInferencePass = new IRShapeInferenceAndConstantFusing();
            shapeInferencePass.Run(ref modelTemp);

            IDictionary<string, int?> ranksByName;
            IRShapeInferenceHelper.RankInference.ListTemporaryTensorRanks(modelTemp, out ranksByName);
            IDictionary<string, TensorShape?> shapesByName;
            IRShapeInferenceHelper.ShapeInference.ListTemporaryTensorShapesNCHW(modelTemp, inputShapes, ranksByName, out shapesByName);

            int negativeRanks = ranksByName.Values.Count(x => x < 0);
            ValidationHelper.AppendWarning(negativeRanks == 0, "model", $"StaticRankInference: {negativeRanks} negative rank(s) found!", ref warnings, MessageType.Warning);

            int knowRanks = ranksByName.Count(x => x.Value != null);
            int knowShapes = shapesByName.Count(x => x.Value != null);

            ValidationHelper.AppendWarning(knowRanks == knowShapes, "model", "StaticShape/RankInference: known ranks # != known shape #", ref warnings);

            foreach (var i in modelTemp.inputs)
            {
                var name = i.name;
                ValidationHelper.AppendWarning(ranksByName.ContainsKey(name), name, "StaticRankInference: did not find input", ref warnings);
                if (ranksByName.ContainsKey(name))
                    ValidationHelper.AppendWarning(ranksByName[name] != null, name, "StaticRankInference: unknown input rank at compile time", ref warnings);

                ValidationHelper.AppendWarning(shapesByName.ContainsKey(name), name, "StaticShapeInference: did not find input", ref warnings);
                if (shapesByName.ContainsKey(name))
                    ValidationHelper.AppendWarning(shapesByName[name] != null, name, "StaticShapeInference: unknown input shape for at compile time", ref warnings);
            }
            foreach (var l in modelTemp.layers)
            {
                var name = l.name;
                ValidationHelper.AppendWarning(ranksByName.ContainsKey(name), name, "StaticRankInference: did not find layer", ref warnings);
                if (ranksByName.ContainsKey(name))
                    ValidationHelper.AppendWarning(ranksByName[name] != null, name, "StaticRankInference: unknown layer rank at compile time", ref warnings);

                ValidationHelper.AppendWarning(shapesByName.ContainsKey(name), name, "StaticShapeInference: did not find layer", ref warnings);
                if (shapesByName.ContainsKey(name))
                    ValidationHelper.AppendWarning(shapesByName[name] != null, name, "StaticShapeInference: unknown layer shape at compile time", ref warnings);
            }
        }
    }

    class ValidateIntermediateNCHWModelLayers : IValidateModelPass
    {
        public void Run(Model model, ref List<Model.ImporterWarning> warnings)
        {
            foreach (var l in model.layers)
            {
                var name = l.name;
                var type = l.type;
                if(type == Layer.Type.Upsample2D)
                {
                    if (l.inputs.Length == 2)
                        continue; // dynamic Upsample

                    var sizes = l.pool;
                    if (sizes != null)
                        ValidationHelper.AppendWarning((sizes[0] == 1) && (sizes[1] == 1), name, "ValidateIntermediateNCHWModelLayers:Upsample2D Only spatial(H and W) resizing is currently supported." +
                                                                                                 " Non spatial sizes (N and C) will be ignored and default to identity.", ref warnings);
                }
                else if (type == Layer.Type.Upsample3D)
                {
                    if (l.inputs.Length == 2)
                        continue; // dynamic Upsample

                    var sizes = l.pool;
                    if (sizes != null)
                        ValidationHelper.AppendWarning((sizes[0] == 1) && (sizes[1] == 1), name, "ValidateIntermediateNCHWModelLayers:Upsample3D Only spatial(H and W) resizing is currently supported." +
                                                                                                 " Non spatial sizes (N and C) will be ignored and default to identity.", ref warnings);
                }
                else if (type == Layer.Type.Range)
                {
                    ValidationHelper.AppendWarning(true, name, "ValidateIntermediateNCHWModelLayers::Range only const inputs supported", ref warnings, MessageType.Error);
                }
            }
        }
    }

    class ValidateBrokenLinksPass : IValidateModelPass
    {
        private static string[] FindBrokenLinks(Model model, HashSet<string> links)
        {
            var allVariables = new HashSet<string>(model.layers.Select(i => i.name));
            var globalInputs = new HashSet<string>(model.inputs.Select(i => i.name));
            var memoryInputs = new HashSet<string>(model.memories.Select(i => i.input));
            allVariables.UnionWith(globalInputs);
            allVariables.UnionWith(memoryInputs);

            var brokenLinks = links;
            brokenLinks.ExceptWith(allVariables);
            return brokenLinks.ToArray();
        }

        private static string[] FindBrokenLinks(Model model, string[] links)
        {
            return FindBrokenLinks(model, new HashSet<string>(links));
        }

        public static string[] FindBrokenLinks(Model model)
        {
            // check global outputs
            var linksToInspect = new HashSet<string>(model.outputs);

            // and all layers
            foreach (var layer in model.layers)
                foreach (var i in layer.inputs)
                    linksToInspect.Add(i);

            return FindBrokenLinks(model, linksToInspect);
        }

        public void Run(Model model, ref List<Model.ImporterWarning> warnings)
        {
            // Model should not contain any broken links in the end
            var unconnectedInputs = FindBrokenLinks(model);
            if (unconnectedInputs.Length > 0)
            {
                foreach (var x in unconnectedInputs)
                    ValidationHelper.AppendWarning(false, x, "ValidateBrokenLinks: broken Links : ", ref warnings, MessageType.Warning);
            }
        }
    }

    class ValidateUniqueOutputsPass : IValidateModelPass
    {
        public void Run(Model model, ref List<Model.ImporterWarning> warnings)
        {
            // validate, all model outputs are unique
            // https://stackoverflow.com/questions/18547354/c-sharp-linq-find-duplicates-in-list
            var duplicateOutputs = model.outputs.GroupBy(x => x)
                .Where(g => g.Count() > 1)
                .Select(y => y.Key);
            foreach (var o in duplicateOutputs)
                ValidationHelper.AppendWarning(false, o, "ValidateUniqueOutputs: Output is specified more than once in the model", ref warnings, MessageType.Warning);
        }
    }

    class ValidateUnconectedLayersPass : IValidateModelPass
    {
        public void Run(Model model, ref List<Model.ImporterWarning> warnings)
        {
            // validate, model contains no unconnected layers
            var unconnectedOutputs = ModelAnalyzer.FindUnconnectedOutputs(model);
            foreach (var o in unconnectedOutputs)
                ValidationHelper.AppendWarning(false, o, "ValidateUnconnectedLayers: Layer is specified as output, but is missing in the model", ref warnings, MessageType.Warning);
        }
    }

    class ValidateNCHWPass : IValidateModelPass
    {
        public void Run(Model model, ref List<Model.ImporterWarning> warnings)
        {
            var validatePasses = new List<IValidateModelPass>
            {
                new ValidateNCHWShapesPass(),
                new ValidateIntermediateNCHWModelLayers(),
                new ValidateUniqueOutputsPass(),
                new ValidateUnconectedLayersPass()
            };

            foreach (var validate in validatePasses)
                validate.Run(model, ref warnings);
        }
    }

    class ValidateNHWCPass : IValidateModelPass
    {
        public void Run(Model model, ref List<Model.ImporterWarning> warnings)
        {
            var validatePasses = new List<IValidateModelPass>
            {
                new ValidateUniqueOutputsPass(),
                new ValidateUnconectedLayersPass()
            };

            foreach (var validate in validatePasses)
                validate.Run(model, ref warnings);
        }
    }
}
