using System;
using System.Collections.Generic;
using System.Linq;

namespace Unity.Barracuda.Compiler.Passes
{
    partial class NCHWToNHWCPass : IModelPass
    {
        IDictionary<string, int?> m_RanksByName;
        IDictionary<string, TensorShape?> m_ShapesByName;

        // NHWC models, layout re-ordering
        bool m_isModelExportedFromNHWC;
        Dictionary<string, LayoutTransposeRemovalHelper.ChannelsOrder> m_layersChannelOrder;

        readonly BurstCPUOps m_Ops = new BurstCPUOps();

        static readonly int[] k_FromNHWCtoNCHW = { 0, 3, 1, 2 };
        static readonly int[] k_FromNCHWtoNHWC = { 0, 2, 3, 1 };
        static readonly int[] k_FromNCHtoN1WC = { 0, 3, 2, 1 };
        static readonly int[] k_FromN1WCtoNCH = { 0, 3, 2, 1 };
        readonly int[] k_ToNCHW = { 0, 3, 1, 2 };
        readonly int[] k_ToNHWC = { 0, 2, 3, 1 };

        public void Run(ref Model model)
        {
            if (!model.layout.Contains("NCHW"))
                return;

            // This is a necessary pass for NCHW models that have the layout built into the model itself (e.g. SSD)
            // It's necessary to contract this into a single layer, so that the Gather pass doesn't get converted
            var shapeContractionPass = new ShapeContractionPass();
            shapeContractionPass.Run(ref model);

            // Remove shape-gather-reshape pattern when they map a transpose to NHWC operation
            var shapeGatherReshapeToNHWCRemovePass = new ShapeGatherReshapeToNHWCRemovePass();
            shapeGatherReshapeToNHWCRemovePass.Run(ref model);

            Rewrite(ref model);

            // Preserve any new layers that must be preserved (e.g. new LSTM outputs)
            // TODO: outputs are preserved, adjust optimization passes to properly merge outputs by renaming layers
            var preserveLayersPass = new PreserveLayersPass();
            preserveLayersPass.Run(ref model);

            // cleanup
            var removeUnusedPass = new Cleanup.RemoveUnusedLayersPass();
            removeUnusedPass.Run(ref model);
            var removeNoOpPass = new Cleanup.RemoveNoOpsPass();
            removeNoOpPass.Run(ref model);
        }

        void Rewrite(ref Model model)
        {
            IRShapeInferenceHelper.RankInference.ListTemporaryTensorRanks(model, out m_RanksByName);
            var inputShapes = new Dictionary<string, TensorShape>();
            foreach (var i in model.inputs)
            {
                if (!ModelAnalyzer.IsInputShapeAcceptablyKnowForShapeInference(i))
                    continue;
                inputShapes.Add(i.name, new TensorShape(i.shape));
            }

            IRShapeInferenceHelper.ShapeInference.ListTemporaryTensorShapesNCHW(model, inputShapes, ref m_RanksByName, out m_ShapesByName);

            var nhwc = model.ShallowCopy();
            nhwc.layers.Clear();
            nhwc.layout = "NHWC";

            // TF2ONNX transpose pattern -> part of the model are in NHWC and not NCHW
            // * identify those
            // * transpose inputs to NCHW
            // * remove layout transposes
            // * convert axis/constants accordingly
            LayoutTransposeRemovalHelper transposeRemoval = new LayoutTransposeRemovalHelper();
            m_isModelExportedFromNHWC = transposeRemoval.InferAllLayersChannelOrder(model, out m_layersChannelOrder);

            if (m_isModelExportedFromNHWC && !transposeRemoval.IsImporterLikelyNHWCLayout(model.ProducerName))
                nhwc.Warnings.Add(new Model.ImporterWarning("model", "model detected as NCHW, but not natively in this layout, behavior might be erroneous"));

            // remove layout change transposes
            if (m_isModelExportedFromNHWC)
                transposeRemoval.RemoveAllChannelLayoutTransposes(ref model, m_layersChannelOrder);

            var modelBuilder = new ModelBuilder(nhwc);

            for (int i = 0; i < nhwc.inputs.Count; i++)
            {
                Model.Input input = nhwc.inputs[i];

                int[] shape = input.shape;
                var tensorShape = new TensorShape(shape);
                int[] rankPermutations = GetChannelsLastPermutationsFromRank(input.rank);
                int[] permutations = tensorShape.Get8DPermutationsForNCHWPermutationsAndShape(rankPermutations);

                // Preserve symbolic shape by operating on int array instead of TensorShape, which would resolve unknown dimensions
                if (m_isModelExportedFromNHWC) // transpose input shape if importer preserved NHWC layout
                {
                    if (m_layersChannelOrder[input.name] == LayoutTransposeRemovalHelper.ChannelsOrder.NCHW)
                        input.shape = TensorExtensions.Permute(shape, permutations);
                    else
                    {
                        var onnxShape = new List<int> { shape[2], shape[5], shape[6], shape[7] };
                        onnxShape.RemoveRange(input.rank, 4 - input.rank);
                        input.shape = IRShapeInferenceHelper.ShapeInference.BarracudaLayoutToTensorShapeLayout(onnxShape.ToArray());
                    }
                }
                else
                {
                    input.shape = TensorExtensions.Permute(shape, permutations);
                }
                nhwc.inputs[i] = input;
            }

            // NCHW -> Barracuda NHWC rewriter (some layer need to insert aditional layers to be Barracuda compatible)
            var rewriters = InstantiateRewriterNCHWToNHWC();
            // NHWC -> Barracuda NHWC rewriter (axis and constant padding padding)
            var rewritersNHWC = InstantiateRewriterNHWCToNHWC();


            foreach (var l in model.layers)
            {
                // Some nodes output multiple layers (e.g. LSTM), so don't process or include those layers
                if (nhwc.layers.Exists(alreadyOutputLayer => alreadyOutputLayer.name == l.name))
                    continue;

                if (m_layersChannelOrder.TryGetValue(l.name, out LayoutTransposeRemovalHelper.ChannelsOrder layerChannelOrder))
                {
                    if (m_isModelExportedFromNHWC && (layerChannelOrder == LayoutTransposeRemovalHelper.ChannelsOrder.NHWC))
                    {
                        if (!rewritersNHWC.TryGetValue(l.type, out Func<Layer, ModelBuilder, bool> rwNCHW) || rwNCHW(l, modelBuilder))
                        {
                            nhwc.layers.Add(l);
                        }
                        continue;
                    }
                }

                if (!rewriters.TryGetValue(l.type, out Func<Layer, ModelBuilder, bool> rw) || rw(l, modelBuilder))
                {
                    // Either no re-write was needed or the layer was not replaced
                    nhwc.layers.Add(l);
                }
            }

            // We need to correct constants to have broadcast work correctly
            // ONNX: 1,64,32 + c:32
            // Barracuda: 1,_32,64 + c:_,_,32,64 and not c:32,_,_,_
            // X:5,7 + c: 6,9,5,7 = 6,9,5,7
            // X: 5,_,_,7 + c: 6,5,7,9 = ???
            CorrectConstantsForBroadCast(ref nhwc);
            CorrectDynamicInputsForBroadCast(ref nhwc);

            // for NHWC importers, perform slightly more aggressive output shape check
            // => add transposes to match onnx layout
            if (transposeRemoval.IsImporterLikelyNHWCLayout(model.ProducerName))
                CorrectOutputLayoutToMatchNHWCLayout(ref nhwc);

            model = nhwc;
        }
    }
}
