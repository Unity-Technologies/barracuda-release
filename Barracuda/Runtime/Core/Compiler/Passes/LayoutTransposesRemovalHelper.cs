using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Assertions;

namespace Unity.Barracuda.Compiler.Passes
{
    class LayoutTransposeRemovalHelper
    {
        List<string> nhwcImporters = new List<string> { "tf2onnx", "keras2onnx" };
        public bool IsImporterLikelyNHWCLayout(string importer) { return nhwcImporters.Exists(x => importer.Contains(x)); }

        public static bool IsLayerTransposeCommutative(Layer layer)
        {
            return layer.type == Layer.Type.Load ||
                   layer.type == Layer.Type.Activation ||
                   layer.type == Layer.Type.DepthToSpace ||
                   layer.type == Layer.Type.SpaceToDepth ||
                   layer.type == Layer.Type.ScaleBias ||
                   layer.type == Layer.Type.LRN ||
                   layer.type == Layer.Type.Multinomial ||
                   layer.type == Layer.Type.Nop ||
                   layer.type == Layer.Type.Dropout ||
                   // ok with Tensors as is
                   layer.type == Layer.Type.Expand ||
                   layer.type == Layer.Type.Unsqueeze ||
                   layer.type == Layer.Type.Squeeze ||
                   layer.type == Layer.Type.StridedSlice ||
                   layer.type == Layer.Type.Gather ||
                   layer.type == Layer.Type.OneHot ||
                   layer.type == Layer.Type.TopKIndices ||
                   layer.type == Layer.Type.TopKValues ||
                   layer.type == Layer.Type.Add ||
                   layer.type == Layer.Type.Sub ||
                   layer.type == Layer.Type.Mul ||
                   layer.type == Layer.Type.Div ||
                   layer.type == Layer.Type.Pow ||
                   layer.type == Layer.Type.Min ||
                   layer.type == Layer.Type.Max ||
                   layer.type == Layer.Type.Mean ||
                   layer.type == Layer.Type.Greater ||
                   layer.type == Layer.Type.Less ||
                   layer.type == Layer.Type.Equal ||
                   layer.type == Layer.Type.LogicalOr ||
                   layer.type == Layer.Type.LogicalAnd ||
                   layer.type == Layer.Type.LogicalNot ||
                   layer.type == Layer.Type.LogicalXor ||
                   layer.type == Layer.Type.Where ||
                   layer.type == Layer.Type.Transpose ||
                   layer.type == Layer.Type.MatMul ||
                   layer.type == Layer.Type.RandomNormal ||
                   layer.type == Layer.Type.RandomUniform ||
                   layer.type == Layer.Type.ReduceMax ||
                   layer.type == Layer.Type.ReduceMean ||
                   layer.type == Layer.Type.ReduceMin ||
                   layer.type == Layer.Type.ReduceProd ||
                   layer.type == Layer.Type.ReduceSum;
        }

        public static int[] PermuteShape(int[] shape, int[] permutations)
        {
            Assert.IsTrue(shape.Length <= permutations.Length);
            Assert.IsTrue(shape.Count(v => v > 1) <= permutations.Count(v => v >= 0));
            var output = new int[permutations.Length];
            for (var i = 0; i < permutations.Length; ++i)
                output[i] = permutations[i] >= 0 ? shape[permutations[i]] : 1;
            return output;
        }

        private static bool IsLayerTranpose(Layer layer) { return layer.type == Layer.Type.Transpose; }
        private static bool IsLayerConv(Layer layer)
        {
            return layer.type == Layer.Type.Conv2D ||
                   layer.type == Layer.Type.DepthwiseConv2D ||
                   layer.type == Layer.Type.Conv2DTrans; // TODO 3D conv has probably the same issue
        }
        private static bool IsLayerReshape(Layer layer) { return layer.type == Layer.Type.Reshape; }
        private static bool IsLayerSqueeze(Layer layer) { return layer.type == Layer.Type.Squeeze; }
        private static bool IsLayerFlatten(Layer layer) { return layer.type == Layer.Type.Flatten; }
        private static bool IsLayerConst(Layer layer) { return layer.type == Layer.Type.Load; }
        private static bool IsLayerRandom(Layer layer) { return layer.type == Layer.Type.RandomNormal || layer.type == Layer.Type.RandomUniform; }
        private static bool IsReshapeTransposeToNCHW(Layer layer, TensorShape inputShape)
        {
            if (layer.inputs.Length > 1)
                return false;
            var newShape = layer.pool;
            if (newShape.Length != 4)
                return false;
            if ((newShape[0] != inputShape.batch) && (newShape[0] != -1) && (newShape[0] != 0))
                return false;
            if (newShape[1] == inputShape.channels && newShape[2] == inputShape.height && newShape[3] == inputShape.width)
                return true;
            return false;
        }
        private static bool IsReshapeTransposeToNHWC(Layer layer, TensorShape inputShape)
        {
            // TODO take onnx shape
            if (layer.inputs.Length > 1)
                return false;
            var newShape = layer.pool;
            if (newShape.Length != 4)
                return false;
            if ((newShape[0] != inputShape.batch) && (newShape[0] != -1) && (newShape[0] != 0))
                return false;
            if (newShape[3] == inputShape.height && newShape[1] == inputShape.width && newShape[2] == inputShape.channels)
                return true;
            return false;
        }
        private bool IsSqueezeTransposeToNHWC(Layer layer, int inputRank)
        {
            var squeezedRank = IRShapeInferenceHelper.RankInference.InferOutputRank(layer, new[] { inputRank });

            return (inputRank == 4) && (squeezedRank <= 2);
        }

        private bool IsFlattenTransposeToNHWC(Layer layer, int inputRank)
        {
            var flattenedRank = IRShapeInferenceHelper.RankInference.InferOutputRank(layer, new[] { inputRank });

            return (inputRank == 4) && (flattenedRank <= 2);
        }

        private static bool IsLayerInput(Model model, string name) { return model.inputs.Exists(x => x.name == name); }

        public enum ChannelsOrder
        {
            NHWC,
            NCHW,
            TransposeToNHWC,
            TransposeToNCHW
        }

        // works on IRModel
        public bool InferAllLayersChannelOrder(Model model, out Dictionary<string, ChannelsOrder> layerChannelOrder)
        {
            // TF2Onnx : pattern T (.* Conv .*) T-1
            // * being transpose commutative layer
            layerChannelOrder = new Dictionary<string, ChannelsOrder>();

            IDictionary<string, TensorShape?> shapesByName = new Dictionary<string, TensorShape?>();
            IDictionary<string, int?> ranksByName = new Dictionary<string, int?>();
            foreach (var i in model.inputs)
            {
                ranksByName[i.name] = i.rank;
                if (!ModelAnalyzer.IsInputShapeAcceptablyKnowForShapeInference(i))
                    continue;
                shapesByName[i.name] = new TensorShape(i.shape);
            }

            IRShapeInferenceAndConstantFusing shapeInferencePass = new IRShapeInferenceAndConstantFusing();
            shapeInferencePass.InferAllShapes(model, ref shapesByName, ref ranksByName);

            bool inputsNHWC = false;
            bool inputsNHWCExportedInputsAsNCHW = false;

            bool patternMatchStart = false;
            bool patternMatchConv = false;
            // tf to onnx does not swizzle axis, need to match * Conv * T-1 ...
            bool patternMatchStartInputsAsNCHWConv = false;
            for (int l = 0; l < model.layers.Count; l++)
            {
                var layer = model.layers[l];
                if (!patternMatchStart &&
                    IsLayerTranpose(layer) && Enumerable.SequenceEqual(layer.pool, new[] { 0, 3, 1, 2 }) ||
                    IsLayerReshape(layer) && (shapesByName[layer.inputs[0]] != null) && IsReshapeTransposeToNCHW(layer, shapesByName[layer.inputs[0]].Value) )
                {
                    patternMatchStart = true;
                }
                else if (patternMatchStart && patternMatchConv &&
                    ((IsLayerTranpose(layer) && Enumerable.SequenceEqual(layer.pool, new[] { 0, 2, 3, 1 })) ||
                     (IsLayerReshape(layer) && (shapesByName[layer.inputs[0]] != null) && IsReshapeTransposeToNHWC(layer, shapesByName[layer.inputs[0]].Value)) ||
                     (IsLayerSqueeze(layer) && (ranksByName[layer.inputs[0]] != null) && IsSqueezeTransposeToNHWC(layer, ranksByName[layer.inputs[0]].Value)) ||
                     (IsLayerFlatten(layer) && (ranksByName[layer.inputs[0]] != null) && IsFlattenTransposeToNHWC(layer, ranksByName[layer.inputs[0]].Value))))
                {
                    inputsNHWC = true;
                }
                else if (patternMatchStart && IsLayerConv(layer))
                {
                    patternMatchConv = true;
                }

                if (!inputsNHWCExportedInputsAsNCHW && patternMatchStartInputsAsNCHWConv &&
                    ((IsLayerTranpose(layer) && Enumerable.SequenceEqual(layer.pool, new[] { 0, 2, 3, 1 })) ||
                     (IsLayerReshape(layer) && (shapesByName[layer.inputs[0]] != null) && IsReshapeTransposeToNHWC(layer, shapesByName[layer.inputs[0]].Value))))
                {
                    inputsNHWCExportedInputsAsNCHW = true;
                }
                else if (!patternMatchStartInputsAsNCHWConv && !patternMatchStart && IsLayerConv(layer))
                {
                    patternMatchStartInputsAsNCHWConv = true;
                }
            }

            // flag each layer as being NHWC or NCHW
            for (int i = 0; i < model.inputs.Count; i++)
            {
                Model.Input input = model.inputs[i];
                if (!inputsNHWCExportedInputsAsNCHW)
                    layerChannelOrder[input.name] = inputsNHWC ? ChannelsOrder.NHWC : ChannelsOrder.NCHW;
                else
                    layerChannelOrder[input.name] = ChannelsOrder.NCHW;
            }

            for (int l = 0; l < model.layers.Count; l++)
            {
                var layer = model.layers[l];

                if (IsLayerTranpose(layer) && Enumerable.SequenceEqual(layer.pool, new[] { 0, 3, 1, 2 }) ||
                    IsLayerReshape(layer) && (shapesByName[layer.inputs[0]] != null) && IsReshapeTransposeToNCHW(layer, shapesByName[layer.inputs[0]].Value) &&
                    layerChannelOrder[layer.inputs[0]] == ChannelsOrder.NHWC)
                {
                    layerChannelOrder[layer.name] = ChannelsOrder.TransposeToNCHW;
                }
                else if (IsLayerTranpose(layer) && Enumerable.SequenceEqual(layer.pool, new[] { 0, 2, 3, 1 }) ||
                         IsLayerReshape(layer) && (shapesByName[layer.inputs[0]] != null) && IsReshapeTransposeToNHWC(layer, shapesByName[layer.inputs[0]].Value) &&
                         layerChannelOrder[layer.inputs[0]] == ChannelsOrder.NCHW)
                {
                    layerChannelOrder[layer.name] = ChannelsOrder.TransposeToNHWC;
                }
                else
                {
                    string inputWithKnownOrder = null;
                    for (int i = 0; i < layer.inputs.Length; i++)
                    {
                        var input = layer.inputs[i];
                        if (layerChannelOrder.ContainsKey(input))
                        {
                            inputWithKnownOrder = input;
                            break;
                        }
                    }

                    if (inputWithKnownOrder == null)
                        continue;
                    Assert.IsNotNull(inputWithKnownOrder);
                    ChannelsOrder inputOrder = layerChannelOrder[inputWithKnownOrder];

                    if (inputOrder == ChannelsOrder.TransposeToNCHW)
                        inputOrder = ChannelsOrder.NCHW;
                    else if (inputOrder == ChannelsOrder.TransposeToNHWC)
                        inputOrder = ChannelsOrder.NHWC;

                    // all layers with unknown layout are const
                    for (int i = 0; i < layer.inputs.Length; i++)
                    {
                        var input = layer.inputs[i];
                        if (!layerChannelOrder.ContainsKey(input))
                        {
                            layerChannelOrder[input] = inputOrder;
                        }
                    }

                    layerChannelOrder[layer.name] = inputOrder;
                }
            }

            // TODO Assert that all layers have a channel order
            // Assert that all layers are NHWC if inputsNHWC
            return inputsNHWC || inputsNHWCExportedInputsAsNCHW;
        }

        public void RemoveAllChannelLayoutTransposes(ref Model model, Dictionary<string, ChannelsOrder> layerChannelOrder)
        {
            // TODO transpose inputs? here
            Dictionary<string, Layer> transposesToRemove = new Dictionary<string, Layer>();

            for (int l = 0; l < model.layers.Count; l++)
            {
                var layer = model.layers[l];

                if (!layerChannelOrder.ContainsKey(layer.name))
                    continue;

                if ((layerChannelOrder[layer.name] == ChannelsOrder.TransposeToNCHW) || (layerChannelOrder[layer.name] == ChannelsOrder.TransposeToNHWC))
                {
                    // find all layers that have layer has input
                    // if transpose is output, replace it with a noop
                    if (model.outputs.Contains(layer.name))
                    {
                        string[] inputs = layer.inputs;
                        layer = new Layer(layer.name, Layer.Activation.None);
                        layer.inputs = inputs;
                        model.layers[l] = layer;

                        continue;
                    }
                    // add it
                    transposesToRemove[layer.name] = layer;
                }
            }

            for (int l = 0; l < model.layers.Count; l++)
            {
                var layer = model.layers[l];
                for(int i = 0; i < layer.inputs.Length; i++)
                {
                    var input = layer.inputs[i];
                    if (transposesToRemove.TryGetValue(input, out Layer transpose))
                    {
                        layer.inputs[i] = transpose.inputs[0];
                    }
                }
            }

            model.layers = model.layers.Except(transposesToRemove.Values).ToList();
        }
    }
}
