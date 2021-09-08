using System.IO;
using Unity.Barracuda.Editor;
using UnityEditor;
using UnityEngine;
#if UNITY_2020_2_OR_NEWER
using UnityEditor.AssetImporters;
using UnityEditor.Experimental.AssetImporters;
#else
using UnityEditor.Experimental.AssetImporters;
#endif

namespace Unity.Barracuda
{
    /// <summary>
    /// Asset Importer of barracuda models.
    /// </summary>
    [ScriptedImporter(3, new[] {"nn"})]
    public class NNModelImporter : ScriptedImporter {
        private const string iconName = "NNModelIcon";

        private Texture2D iconTexture;

        /// <summary>
        /// Scripted importer callback
        /// </summary>
        /// <param name="ctx">Asset import context</param>
        public override void OnImportAsset(AssetImportContext ctx)
        {
            var model = File.ReadAllBytes(ctx.assetPath);

            // Analyze model and send analytics if enabled
            var nnModel = ModelLoader.Load(ctx.assetPath, skipWeights:true);
            BarracudaAnalytics.SendBarracudaImportEvent(null, nnModel);

            var assetData = ScriptableObject.CreateInstance<NNModelData>();
            assetData.Value = model;
            assetData.name = "Data";
            assetData.hideFlags = HideFlags.HideInHierarchy;

            var asset = ScriptableObject.CreateInstance<NNModel>();
            asset.modelData = assetData;
            ctx.AddObjectToAsset("main obj", asset, LoadIconTexture());
            ctx.AddObjectToAsset("model data", assetData);

            ctx.SetMainObject(asset);
        }

        private Texture2D LoadIconTexture()
        {
            if (iconTexture == null)
            {
                string[] allCandidates = AssetDatabase.FindAssets(iconName);

                if (allCandidates.Length > 0)
                {
                    iconTexture = AssetDatabase.LoadAssetAtPath(AssetDatabase.GUIDToAssetPath(allCandidates[0]), typeof(Texture2D)) as Texture2D;
                }
            }
            return iconTexture;
        }

    }
}
