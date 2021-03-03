using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Assertions;

namespace Unity.Barracuda.Compiler.Passes
{
    class ShapeGatherReshapeToNHWCRemovePass : IModelPass
    {
        public void Run(ref Model model)
        {
            if (!model.layout.Contains("NCHW"))
                return;

            var layersToRemove = new List<string>();

            for (int l = 2; l < model.layers.Count; ++l)
            {
                if (model.layers[l - 2].type != Layer.Type.Shape ||
                    model.layers[l - 1].type != Layer.Type.Gather ||
                    model.layers[l - 0].type != Layer.Type.Reshape)
                    continue;

                var shapeLayer = model.layers[l - 2];
                var gatherLayer = model.layers[l - 1];
                var reshapeLayer = model.layers[l - 0];

                if (shapeLayer.flags.HasFlag(Layer.Flags.Preserve) ||
                    gatherLayer.flags.HasFlag(Layer.Flags.Preserve))
                    continue;

                //Is reshape using gather as input?
                if (reshapeLayer.inputs[1] != gatherLayer.name)
                    continue;

                //Is gather using shape as input?
                if (gatherLayer.inputs[0] != shapeLayer.name)
                    continue;

                //Are those layer used by other node of the model?
                if (!CanLayerBeRemoved(shapeLayer, gatherLayer, model) ||
                    !CanLayerBeRemoved(gatherLayer, reshapeLayer, model))
                    continue;

                //Is gather converting that shape to channel last?
                if (!IsGather1DAndConvertingToChannelLast(gatherLayer, model))
                    continue;

                //Then those three layer are equivalent to a transpose to channel last.
                //this transpose is itself not needed as we are converting to channel last.
                //so we can just replace those three layers by a single identity.
                reshapeLayer.type = Layer.Type.Activation;
                reshapeLayer.activation = Layer.Activation.None;
                reshapeLayer.pool = new int[0];
                reshapeLayer.axis = -1;
                reshapeLayer.inputs = shapeLayer.inputs;

                layersToRemove.Add(shapeLayer.name);
                layersToRemove.Add(gatherLayer.name);
            }

            model.layers.RemoveAll(l => layersToRemove.Contains(l.name));
        }

        bool IsGather1DAndConvertingToChannelLast(Layer gatherLayer, Model model)
        {
            Assert.AreEqual(Layer.Type.Gather,gatherLayer.type);
            if (gatherLayer.axis > 0)
                return false;

            var indicesAsConstants = model.layers.FirstOrDefault(c => c.type == Layer.Type.Load && c.name == gatherLayer.inputs[1]);
            if (indicesAsConstants == null)
                return false;

            var indices = indicesAsConstants.DataSetToTensor(0).ToReadOnlyArray();
            if (Enumerable.SequenceEqual(indices, new float[] { 0, 2, 3, 1 }) ||
                Enumerable.SequenceEqual(indices, new float[] { 0, 1, 2, 4, 5, 6, 7, 3 }))
                return true;

            return false;
        }

        bool CanLayerBeRemoved(Layer layerToRemove, Layer acceptedChildLayer, Model model)
        {
            if (model.outputs.Contains(layerToRemove.name))
                return false;

            if (model.memories.Exists(m => m.output == layerToRemove.name))
                return false;

            //Need to check that no other layers use layerToRemove but the one accepted child that we will process.
            if (model.layers.Exists(l => l != acceptedChildLayer && l.inputs.Contains(layerToRemove.name)))
                return false;

            return true;
        }
    }
}
