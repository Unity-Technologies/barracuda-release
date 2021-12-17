using UnityEngine;
using UnityEditor;
#if UNITY_2020_2_OR_NEWER
using UnityEditor.AssetImporters;
using UnityEditor.Experimental.AssetImporters;
#else
using UnityEditor.Experimental.AssetImporters;
#endif
using System;
using System.IO;
using System.Runtime.CompilerServices;
using Unity.Barracuda.Editor;
using Unity.Barracuda.ONNX;

[assembly: InternalsVisibleToAttribute("Barracuda.EditorTests")]
[assembly: InternalsVisibleToAttribute("Unity.Barracuda.Tests")]

namespace Unity.Barracuda
{
    /// <summary>
    /// Asset Importer for Open Neural Network Exchange (ONNX) files.
    /// For more information about ONNX file format see: https://github.com/onnx/onnx
    /// </summary>
    [ScriptedImporter(34, new[] { "onnx" })]
    public class ONNXModelImporter : ScriptedImporter
    {
        // Configuration
        /// <summary>
        /// Enable ONNX model optimization during import. Set via importer UI
        /// </summary>
        public bool optimizeModel = true;

        /// <summary>
        /// Fix batch size for ONNX models. Set via importer UI
        /// </summary>
        public bool forceArbitraryBatchSize = true;

        /// <summary>
        /// Treat errors as warnings. Set via importer UI
        /// </summary>
        public bool treatErrorsAsWarnings = false;

        [SerializeField, HideInInspector]
        internal ONNXModelConverter.ImportMode importMode = ONNXModelConverter.ImportMode.Standard;

        [SerializeField, HideInInspector]
        internal ONNXModelConverter.DataTypeMode weightsTypeMode = ONNXModelConverter.DataTypeMode.Default;
        [SerializeField, HideInInspector]
        internal ONNXModelConverter.DataTypeMode activationTypeMode = ONNXModelConverter.DataTypeMode.Default;

        internal const string iconName = "ONNXModelIcon";


        private Texture2D m_IconTexture;

        /// <summary>
        /// Scripted importer callback
        /// </summary>
        /// <param name="ctx">Asset import context</param>
        public override void OnImportAsset(AssetImportContext ctx)
        {
            ONNXModelConverter.ModelImported += BarracudaAnalytics.SendBarracudaImportEvent;
            var converter = new ONNXModelConverter(optimizeModel, treatErrorsAsWarnings, forceArbitraryBatchSize, importMode);

            var model = converter.Convert(ctx.assetPath);

            if (weightsTypeMode == ONNXModelConverter.DataTypeMode.ForceHalf)
                model.ConvertWeights(DataType.Half);
            else if (weightsTypeMode == ONNXModelConverter.DataTypeMode.ForceFloat)
                model.ConvertWeights(DataType.Float);

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
