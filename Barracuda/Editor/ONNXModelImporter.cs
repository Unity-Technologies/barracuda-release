using Google.Protobuf;
using Onnx;
using UnityEngine;
using UnityEngine.Profiling;
using UnityEditor;
#if UNITY_2020_2_OR_NEWER
using UnityEditor.AssetImporters;
using UnityEditor.Experimental.AssetImporters;
#else
using UnityEditor.Experimental.AssetImporters;
#endif
using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using Unity.Barracuda.ONNX;

[assembly: InternalsVisibleToAttribute("Barracuda.EditorTests")]

namespace Unity.Barracuda
{
    /// <summary>
    /// Asset Importer for Open Neural Network Exchange (ONNX) files.
    /// For more information about ONNX file format see: https://github.com/onnx/onnx
    /// </summary>
    [ScriptedImporter(10, new[] { "onnx" })]
    public class ONNXModelImporter : ScriptedImporter
    {
        // Configuration
        public bool optimizeModel = true;
        public bool forceArbitraryBatchSize = true;
        public bool treatErrorsAsWarnings = false;

        public const string iconName = "ONNXModelIcon";


        private Texture2D m_IconTexture;

        public override void OnImportAsset(AssetImportContext ctx)
        {
            var converter = new ONNXModelConverter(optimizeModel);

            var model = converter.Convert(ctx.assetPath);

            NNModelData assetData = ScriptableObject.CreateInstance<NNModelData>();
            using (var memoryStream = new MemoryStream())
            using (var writer = new BinaryWriter(memoryStream))
            {
                ModelWriter.Save(writer, model);
                assetData.Value = memoryStream.ToArray();
            }
            assetData.name = "Data";
            assetData.hideFlags = HideFlags.HideInHierarchy;

            NNModel asset = ScriptableObject.CreateInstance<NNModel>();
            asset.modelData = assetData;

            ctx.AddObjectToAsset("main obj", asset, LoadIconTexture());
            ctx.AddObjectToAsset("model data", assetData);

            ctx.SetMainObject(asset);
        }

        // Icon helper
        private Texture2D LoadIconTexture()
        {
            if (m_IconTexture == null)
            {
                string[] allCandidates = AssetDatabase.FindAssets(iconName);

                if (allCandidates.Length > 0)
                {
                    m_IconTexture = AssetDatabase.LoadAssetAtPath(AssetDatabase.GUIDToAssetPath(allCandidates[0]), typeof(Texture2D)) as Texture2D;
                }
            }
            return m_IconTexture;
        }



    }


}
