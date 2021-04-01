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
        private bool IsLayerNecessarilyNCHWOnnx(Layer layer)
        {
            return layer.type == Layer.Type.Conv2D ||
                   layer.type == Layer.Type.Conv3D ||
                   layer.type == Layer.Type.Conv2DTrans ||
                   layer.type == Layer.Type.Conv3DTrans ||
                   layer.type == Layer.Type.DepthwiseConv2D ||
                   layer.type == Layer.Type.DepthToSpace ||
                   layer.type == Layer.Type.SpaceToDepth;
        }

        private static bool IsLayerTranpose(Layer layer) { return layer.type == Layer.Type.Transpose; }
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

        private bool IsLayerChangingLayoutToNHWC(Layer layer, IDictionary<string, TensorShape?> shapesByName, IDictionary<string, int?> ranksByName)
        {
            return (IsLayerTranpose(layer) && Enumerable.SequenceEqual(layer.pool, new[] { 0, 2, 3, 1 })) ||
                   (IsLayerReshape(layer) && (shapesByName[layer.inputs[0]] != null) && IsReshapeTransposeToNHWC(layer, shapesByName[layer.inputs[0]].Value)) ||
                   (IsLayerSqueeze(layer) && (ranksByName[layer.inputs[0]] != null)  && IsSqueezeTransposeToNHWC(layer, ranksByName[layer.inputs[0]].Value)) ||
                   (IsLayerFlatten(layer) && (ranksByName[layer.inputs[0]] != null)  && IsFlattenTransposeToNHWC(layer, ranksByName[layer.inputs[0]].Value));
        }

        private bool IsLayerChangingLayoutToNCHW(Layer layer, IDictionary<string, TensorShape?> shapesByName, IDictionary<string, int?> ranksByName)
        {
            return (IsLayerTranpose(layer) && Enumerable.SequenceEqual(layer.pool, new[] { 0, 3, 1, 2 })) ||
                   (IsLayerReshape(layer) && (shapesByName[layer.inputs[0]] != null) && IsReshapeTransposeToNCHW(layer, shapesByName[layer.inputs[0]].Value));
        }

        public enum ChannelsOrder
        {
            NHWC,
            NCHW,
            TransposeToNHWC,
            TransposeToNCHW,
            // used only in InferAllLayersChannelOrder
            NativeNCHW
        }

        private enum FlowDirection
        {
            Seed,
            Downstream,
            Upstream
        }

        // works on IRModel
        public bool InferAllLayersChannelOrder(Model model, out Dictionary<string, ChannelsOrder> layerChannelOrder)
        {
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

            // flood-fill approach: NCHW layout is propagated from NCHW ops
            //  * onnx-nchw ops are flagged as being native nchw
            //  * nchw layout is propagated to upstream and downstream nodes
            //  foreach node:
            //    take layout being propagated to
            //    if T or T-1 flip layout depending on upstream/downstream direction
            //    - stop if layout is the same as previously propagated
            //    - native nchw layout has priority
            Queue<(string, ChannelsOrder, FlowDirection)> layersToInferLayout = new Queue<(string, ChannelsOrder, FlowDirection)>();
            for (int l = 0; l < model.layers.Count; l++)
            {
                var layer = model.layers[l];
                if (!IsLayerNecessarilyNCHWOnnx(layer))
                    continue;

                layersToInferLayout.Enqueue((layer.name, ChannelsOrder.NativeNCHW, FlowDirection.Seed));
            }

            while (layersToInferLayout.Any())
            {
                (string, ChannelsOrder, FlowDirection) layerData = layersToInferLayout.Dequeue();
                string name = layerData.Item1;
                ChannelsOrder deducedChannelOrder = layerData.Item2;
                // 0: in-place native
                // 1: downstream
                // 2: upstream
                FlowDirection flowDirection = layerData.Item3;


                if (!layerChannelOrder.ContainsKey(name))
                    layerChannelOrder[name] = deducedChannelOrder;
                else if (deducedChannelOrder == layerChannelOrder[name])
                    continue;
                else if (layerChannelOrder[name] == ChannelsOrder.NativeNCHW)
                    continue;
                // heuristic to stop ping-pong loop, prioritize NHWC over NCHW as it implies less transposes
                // TODO: count # of transpose swaps
                else if (layerChannelOrder[name] == ChannelsOrder.NHWC)
                    continue;

                Layer layer;
                bool found = ModelAnalyzer.FindLayerByName(model, name, out layer);
                if (IsLayerChangingLayoutToNHWC(layer, shapesByName, ranksByName))
                {
                    // NCHW -> T -> NHWC
                    if (((deducedChannelOrder == ChannelsOrder.NCHW) || (deducedChannelOrder == ChannelsOrder.NativeNCHW)) && (flowDirection == FlowDirection.Downstream))
                        deducedChannelOrder = ChannelsOrder.TransposeToNHWC;
                    // NCHW <- T <- NHWC
                    else if ((deducedChannelOrder == ChannelsOrder.NHWC) && (flowDirection == FlowDirection.Upstream))
                        deducedChannelOrder = ChannelsOrder.TransposeToNHWC;
                }
                else if (IsLayerChangingLayoutToNCHW(layer, shapesByName, ranksByName))
                {
                    // NHWC -> T-1 -> NCHW
                    if ((deducedChannelOrder == ChannelsOrder.NHWC) && (flowDirection == FlowDirection.Downstream))
                        deducedChannelOrder = ChannelsOrder.TransposeToNCHW;
                    // NHWC <- T-1 <- NCHW
                    else if (((deducedChannelOrder == ChannelsOrder.NCHW) || (deducedChannelOrder == ChannelsOrder.NativeNCHW)) && (flowDirection == FlowDirection.Upstream))
                        deducedChannelOrder = ChannelsOrder.TransposeToNCHW;
                }

                if ((deducedChannelOrder == ChannelsOrder.TransposeToNCHW || deducedChannelOrder == ChannelsOrder.TransposeToNHWC) && (deducedChannelOrder == layerChannelOrder[name]))
                    continue;

                layerChannelOrder[name] = deducedChannelOrder;

                foreach (var input in layer.inputs)
                {
                    if(deducedChannelOrder == ChannelsOrder.TransposeToNCHW)
                        layersToInferLayout.Enqueue((input, ChannelsOrder.NHWC, FlowDirection.Upstream));
                    else if(deducedChannelOrder == ChannelsOrder.TransposeToNHWC)
                        layersToInferLayout.Enqueue((input, ChannelsOrder.NCHW, FlowDirection.Upstream));
                    else
                        layersToInferLayout.Enqueue((input, deducedChannelOrder, FlowDirection.Upstream));
                }

                var outputs = ModelAnalyzer.FindLayerOutputs(model, layer.name);
                foreach (var output in outputs)
                {
                    if (deducedChannelOrder == ChannelsOrder.TransposeToNCHW)
                        layersToInferLayout.Enqueue((output, ChannelsOrder.NCHW, FlowDirection.Downstream));
                    else if (deducedChannelOrder == ChannelsOrder.TransposeToNHWC)
                        layersToInferLayout.Enqueue((output, ChannelsOrder.NHWC, FlowDirection.Downstream));
                    else
                        layersToInferLayout.Enqueue((output, deducedChannelOrder, FlowDirection.Downstream));
                }
            }

            bool modelExportedASNHWC = false;
            foreach (string key in layerChannelOrder.Keys.ToList())
            {
                var value = layerChannelOrder[key];
                if (value == ChannelsOrder.NativeNCHW)
                    layerChannelOrder[key] = ChannelsOrder.NCHW;

                if (value == ChannelsOrder.NHWC)
                    modelExportedASNHWC = true;
            }

            return modelExportedASNHWC;
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

                if (!((layerChannelOrder[layer.name] == ChannelsOrder.TransposeToNCHW) || (layerChannelOrder[layer.name] == ChannelsOrder.TransposeToNHWC)))
                    continue;

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

            for (int l = 0; l < model.layers.Count; l++)
            {
                var layer = model.layers[l];
                for(int i = 0; i < layer.inputs.Length; i++)
                {
                    var input = layer.inputs[i];
                    if (transposesToRemove.TryGetValue(input, out Layer transpose))
                        layer.inputs[i] = transpose.inputs[0];
                }
            }

            model.layers = model.layers.Except(transposesToRemove.Values).ToList();
        }
    }
}
