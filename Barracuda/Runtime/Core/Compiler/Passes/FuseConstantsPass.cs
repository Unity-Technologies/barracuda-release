using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace Unity.Barracuda.Compiler.Passes.Optimization
{
    class FuseConstantsPass : IModelPass
    {
        public void Run(ref Model model)
        {
            FuseConstants(ref model);
        }

        public static void FuseConstants(ref Model model)
        {
            var knownLayersValue = new Dictionary<string, Tensor>();
            var newKnownLayers = new HashSet<string>();
            var keepLayers = new HashSet<string>();

            for (int l = 0; l < model.layers.Count; ++l)
            {
                var layer = model.layers[l];
                if (layer.flags == Layer.Flags.Preserve)
                    keepLayers.Add(layer.name);

                // NN is a directed graph, if we just fused constants + shapes, update following nodes
                // TODO optimization, pass in index, or add shape

                if (ModelOptimizer.IsLayerConstant(layer))
                    knownLayersValue[layer.name] = new Tensor(layer.datasets[0].shape.ToArray(), layer.weights);

                bool allInputsAreKnown = layer.inputs.Length > 0 ? knownLayersValue.ContainsKey(layer.inputs[0]) : false;
                for (int i = 1; i < layer.inputs.Length; i++)
                    allInputsAreKnown &= knownLayersValue.ContainsKey(layer.inputs[i]);

                // if all inputs are known, execute layer
                if (!allInputsAreKnown)
                    continue;

                var layerInputs = new Dictionary<string, Tensor>();
                var opsModel = new Model();
                for (int i = 0; i < layer.inputs.Length; i++)
                {
                    Model.Input input;
                    input.name = layer.inputs[i];
                    input.shape = knownLayersValue[input.name].shape.ToArray();
                    input.rank = knownLayersValue[input.name].shape.dimensions;

                    opsModel.inputs.Add(input);
                    layerInputs[input.name] = knownLayersValue[input.name];
                }
                opsModel.layers.Add(layer);
                opsModel.outputs.Add(layer.name);

                // bake
                var useCPUforBaking = WorkerFactory.Device.CPU;
                using (var worker = WorkerFactory.CreateWorker(opsModel, useCPUforBaking))
                {
                    // TODO use ModelIR2RunnableNCHWPass
                    var bakedConstant = worker.Execute(layerInputs).PeekOutput();
                    bakedConstant.TakeOwnership();
                    knownLayersValue[layer.name] = bakedConstant;
                    newKnownLayers.Add(layer.name);
                }
            }

            // remove new baked layers since we will insert constants for those
            model.layers.RemoveAll(x => newKnownLayers.Contains(x.name) && !keepLayers.Contains(x.name));

            // TODO use ModelBuilder?
            foreach (var l in newKnownLayers)
            {
                if (keepLayers.Contains(l))
                    continue;

                var name = l;
                var tensor = knownLayersValue[name];
                Layer c = new Layer(name, Layer.Type.Load);

                c.datasets = new Layer.DataSet[1];
                c.datasets[0].name = name;
                c.datasets[0].shape = tensor.shape;
                c.datasets[0].itemSizeInBytes = 4;
                c.datasets[0].length = tensor.shape.length;
                c.datasets[0].offset = 0;

                c.axis = tensor.shape.dimensions;

                c.weights = new float[tensor.length];
                Array.Copy(tensor.ToReadOnlyArray(), c.weights, tensor.length);
                model.layers.Insert(0, c);
            }

            // clear allocated tensors
            foreach (var l in knownLayersValue)
                l.Value.Dispose();

            // remove unused constants
            var removeUnusedLayersPass = new Cleanup.RemoveUnusedLayersPass();
            removeUnusedLayersPass.Run(ref model);
        }
    }
}
