
using System;
using System.Collections.Generic;
using System.Linq;
using Onnx;
using UnityEditor;
using UnityEngine.Analytics;

namespace Unity.Barracuda.Editor
{
    internal class BarracudaAnalytics
    {
        static bool s_EventRegistered = false;
        const int k_MaxEventsPerHour = 1000;
        const int k_MaxNumberOfElements = 1000;
        const string k_VendorKey = "unity.barracuda";
        const string k_ImportEventName = "uBarracudaImport";

        static bool EnableAnalytics()
        {
            AnalyticsResult result = EditorAnalytics.RegisterEventWithLimit(k_ImportEventName, k_MaxEventsPerHour, k_MaxNumberOfElements, k_VendorKey);
            if (result == AnalyticsResult.Ok)
                s_EventRegistered = true;

            return s_EventRegistered;
        }

        struct BarracudaImportAnalyticsData
        {
            public string model_type;
            public string original_layers;
            public string imported_layers;
            public string import_warnings;
        }

        public static void SendBarracudaImportEvent(object originalModel, Model importedModel)
        {
            //The event shouldn't be able to report if this is disabled but if we know we're not going to report
            //Lets early out and not waste time gathering all the data
            if (!EditorAnalytics.enabled)
                return;

            if (!EnableAnalytics())
                return;


            var data = new BarracudaImportAnalyticsData();

            try
            {
                data.original_layers = AnalyzeONNXModel(originalModel);
                data.imported_layers = AnalyzeNNModel(importedModel);
                data.model_type = string.IsNullOrEmpty(data.original_layers) ? "NN" : "ONNX";
                data.import_warnings = AnalyzeWarnings(importedModel);
            }
            catch (Exception e)
            {
                D.LogError($"Failed collecting Barracuda analytics: {e}");
            }

            EditorAnalytics.SendEventWithLimit(k_ImportEventName, data);
        }

        static string AnalyzeONNXModel(object originalModel)
        {
            if (!(originalModel is ModelProto))
                return "";

            var layers = new Dictionary<string, int>();

            var onnxModel = originalModel as ModelProto;
            foreach (var node in onnxModel.Graph.Node)
            {
                var layerDescription = node.OpType;

                if (!layers.ContainsKey(layerDescription))
                    layers[layerDescription] = 1;
                else
                    layers[layerDescription] += 1;
            }

            return DictionaryToJson(layers);
        }

        static string AnalyzeNNModel(Model importedModel)
        {
            var layers = new Dictionary<string, int>();

            foreach (Layer layer in importedModel.layers)
            {
                var layerDescription = LayerToString(layer);

                if (!layers.ContainsKey(layerDescription))
                    layers[layerDescription] = 1;
                else
                    layers[layerDescription] += 1;
            }

            return DictionaryToJson(layers);
        }

        static string LayerToString(Layer layer)
        {
            var layerDescription = layer.type.ToString();

            if (layer.type == Layer.Type.Conv2D || layer.type == Layer.Type.Conv2DTrans ||
                layer.type == Layer.Type.Conv3D || layer.type == Layer.Type.Conv3DTrans ||
                layer.type == Layer.Type.DepthwiseConv2D)
            {
                layerDescription += "_" + ConvShapeToString(layer);
            }

            if (layer.activation != Layer.Activation.None)
                layerDescription += "_" + layer.activation.ToString();

            return layerDescription;
        }

        static string ConvShapeToString(Layer layer)
        {
            if (layer.type == Layer.Type.Conv2D ||
                layer.type == Layer.Type.DepthwiseConv2D ||
                layer.type == Layer.Type.Conv2DTrans)
                return string.Join("_",
                    layer.datasets.Where(d => d.name.EndsWith("/K")).Select(it =>
                        $"{it.shape.kernelHeight}x{it.shape.kernelWidth}x{it.shape.kernelDepth}x{it.shape.kernelCount}"));

            if (layer.type == Layer.Type.Conv3D ||
                layer.type == Layer.Type.Conv3DTrans)
                return string.Join("_",
                    layer.datasets.Where(d => d.name.EndsWith("/K")).Select(it =>
                        $"{it.shape.kernelSpatialDepth}x{it.shape.kernelHeight}x{it.shape.kernelWidth}x{it.shape.kernelDepth}x{it.shape.kernelCount}"));

            return "";
        }

        static string AnalyzeWarnings(Model importedModel)
        {
            return "[" + string.Join(",",importedModel.Warnings.Select(item => $"'{item.LayerName}:{item.Message}'")) + "]";
        }

        static string DictionaryToJson(Dictionary<string, int> dict)
        {
            var entries = dict.Select(d => $"\"{d.Key}\":{string.Join(",", d.Value)}");
            return "{" + string.Join(",", entries) + "}";
        }
    }
}
