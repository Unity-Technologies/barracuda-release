using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Assertions;

namespace Unity.Barracuda.Compiler.Passes
{
    class IRShapeInferenceAndConstantFusing : IModelPass
    {
        public void Run(ref Model model)
        {
            IDictionary<string, TensorShape?> inputShapes = new Dictionary<string, TensorShape?>();
            IDictionary<string, int?> inputRanks = new Dictionary<string, int?>();
            List<Model.Input> inputs = model.inputs;
            foreach (var i in inputs)
            {
                inputRanks[i.name] = i.rank;
                if (!ModelAnalyzer.IsInputShapeAcceptablyKnowForShapeInference(i))
                    continue;
                inputShapes[i.name] = new TensorShape(i.shape);
            }
            FuseShapesIntoConstants(ref model, inputShapes, inputRanks);
        }

        private static Tensor ShapeToNCHWTensor(TensorShape shape, int rank)
        {
            switch (rank)
            {
                case 0:
                    return new Tensor(new TensorShape(1), new float[] { 0 });
                case 1:
                    return new Tensor(new TensorShape(1), new float[] { shape.batch });
                case 2:
                    return new Tensor(new TensorShape(2), new float[] { shape.batch, shape.height });
                case 3:
                    return new Tensor(new TensorShape(3), new float[] { shape.batch, shape.height, shape.width });
                case 4:
                    return new Tensor(new TensorShape(4), new float[] { shape.batch, shape.height, shape.width, shape.channels });
                case 5:
                    return new Tensor(new TensorShape(5), new float[] { shape.batch, shape.depth, shape.height, shape.width, shape.channels });
                default:
                    return new Tensor(new TensorShape(8), new float[] { shape.sequenceLength, shape.numberOfDirections, shape.batch, shape.extraDimension, shape.depth, shape.height, shape.width, shape.channels });
            }
        }

        public void FuseShapesIntoConstants(ref Model model, IDictionary<string, TensorShape?> shapesByName, IDictionary<string, int?> ranksByName)
        {
            var toRunnableNCHW = new IntermediateToRunnableNCHWPass();

            var knownLayersValue = new Dictionary<string, Tensor>();
            var newKnownLayers = new HashSet<string>();
            var keepLayers = new HashSet<string>();

            for (int l = 0; l < model.layers.Count; ++l)
            {
                var layer = model.layers[l];
                if (layer.flags == Layer.Flags.Preserve)
                    keepLayers.Add(layer.name);

                // NN is a directed graph, if we just fused constants + shapes, update following nodes
                // re-evaluate shapes
                FuseInputsIntoLayer(ref layer, knownLayersValue);
                // TODO optimization, pass in index, or add shape
                IRShapeInferenceHelper.RankInference.UpdateKnownTensorRanks(model, ranksByName);
                IRShapeInferenceHelper.ShapeInference.UpdateKnownTensorShapesNCHW(model, ranksByName, ref shapesByName);

                if (ModelOptimizer.IsLayerConstant(layer))
                    knownLayersValue[layer.name] = new Tensor(layer.datasets[0].shape.ToArray(), layer.weights);
                else if (layer.type == Layer.Type.Shape)
                {
                    // assert inputs.Lenght == 1
                    var input = layer.inputs[0];
                    if (shapesByName.ContainsKey(input) && shapesByName[input] != null &&
                        ranksByName.ContainsKey(input)  && ranksByName[input]  != null
                        )
                    {
                        var shape = shapesByName[input].Value;
                        var rank = ranksByName[input].Value;
                        knownLayersValue[layer.name] = ShapeToNCHWTensor(shape, rank);
                        newKnownLayers.Add(layer.name);
                        continue;
                    }
                }

                bool allInputsAreKnown = layer.inputs.Length > 0 ? knownLayersValue.ContainsKey(layer.inputs[0]) : false;
                for (int i = 1; i < layer.inputs.Length; i++)
                    allInputsAreKnown &= knownLayersValue.ContainsKey(layer.inputs[i]);

                // if all inputs are known, execute layer
                if (!allInputsAreKnown)
                    continue;

                var layerInputs = new Dictionary<string, Tensor>();
                var opsModel = new Model();
                opsModel.layout = "iNCHW";
                for (int i = 0; i < layer.inputs.Length; i++)
                {
                    Model.Input input;
                    input.name = layer.inputs[i];
                    input.shape = shapesByName[input.name].Value.ToArray();
                    input.rank = ranksByName[input.name].Value;

                    opsModel.inputs.Add(input);
                    layerInputs[input.name] = knownLayersValue[input.name];
                }
                opsModel.layers.Add(layer);
                opsModel.outputs.Add(layer.name);

                toRunnableNCHW.Run(ref opsModel);

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

                c.axis = ranksByName[c.name].Value;

                c.weights = new float[tensor.length];
                Array.Copy(tensor.ToReadOnlyArray(), c.weights, tensor.length);
                model.layers.Insert(0,c);
            }

            foreach (var l in knownLayersValue)
                l.Value.Dispose();

            // TODO remove?
            // remove unused constants
            var removeUnusedLayersPass = new Cleanup.RemoveUnusedLayersPass();
            removeUnusedLayersPass.Run(ref model);
        }

        public void InferAllShapes(Model model, ref IDictionary<string, TensorShape?> shapesByName, ref IDictionary<string, int?> ranksByName)
        {
            var knownLayersValue = new Dictionary<string, Tensor>();
            var newKnownLayers = new HashSet<string>();

            for (int l = 0; l < model.layers.Count; ++l)
            {
                var layer = model.layers[l];

                // NN is a directed graph, if we just fused constants + shapes, update following nodes
                // re-evaluate shapes
                FuseInputsIntoLayer(ref layer, knownLayersValue);
                // TODO optimization, pass in index, or add shape
                IRShapeInferenceHelper.RankInference.UpdateKnownTensorRanks(model, ranksByName);
                IRShapeInferenceHelper.ShapeInference.UpdateKnownTensorShapesNCHW(model, ranksByName, ref shapesByName);

                if (ModelOptimizer.IsLayerConstant(layer))
                    knownLayersValue[layer.name] = new Tensor(layer.datasets[0].shape.ToArray(), layer.weights);
                else if (layer.type == Layer.Type.Shape)
                {
                    // assert inputs.Lenght == 1
                    var input = layer.inputs[0];
                    if (shapesByName.ContainsKey(input) && shapesByName[input] != null &&
                        ranksByName.ContainsKey(input) && ranksByName[input] != null
                        )
                    {
                        var shape = shapesByName[input].Value;
                        var rank = ranksByName[input].Value;
                        knownLayersValue[layer.name] = ShapeToNCHWTensor(shape, rank);
                        newKnownLayers.Add(layer.name);
                    }
                }

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
                    input.shape = shapesByName[input.name].Value.ToArray();
                    input.rank = ranksByName[input.name].Value;

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

            // clear allocated tensors
            foreach (var l in knownLayersValue)
                l.Value.Dispose();

            // remove unused constants
            var removeUnusedLayersPass = new Cleanup.RemoveUnusedLayersPass();
            removeUnusedLayersPass.Run(ref model);
        }

        private bool IsLayerKnown(string name, Dictionary<string, Tensor> knownLayersValue)
        {
            return knownLayersValue.ContainsKey(name) && (name != null);
        }

        public void FuseInputsIntoLayer(ref Layer layer, Dictionary<string, Tensor> knownLayersValue)
        {
            switch (layer.type)
            {
                case Layer.Type.Upsample2D:
                {
                    if (layer.inputs.Length <= 1 || !IsLayerKnown(layer.inputs[1], knownLayersValue))
                        return;

                    float[] scales = knownLayersValue[layer.inputs[1]].ToReadOnlyArray();

                    if (scales[0] == 1 && scales[1] == 1 && scales[2] < 1.0f && scales[3] < 1.0f)
                    {
                        scales = new[] { scales[2], scales[3] };
                        layer.type = Layer.Type.AvgPool2D;
                        layer.pad = new[] { 0, 0 };
                        var inverseScalesRoundedToInt = scales.Select(x => (int)Mathf.Round(1f / x)).ToArray();
                        layer.stride = new[] { 1, 1 };
                        layer.pool = inverseScalesRoundedToInt;
                    }
                    else
                    {
                        layer.inputs = new[] { layer.inputs[0] };
                        layer.pool = Array.ConvertAll(scales, x => (int)x);
                    }
                    return;
                }
                case Layer.Type.Resample2D:
                {
                    if (layer.inputs.Length <= 1 || !IsLayerKnown(layer.inputs[1], knownLayersValue))
                        return;

                    int[] sizes = Array.ConvertAll(knownLayersValue[layer.inputs[1]].ToReadOnlyArray(), x => (int)x);

                    layer.inputs = new[] { layer.inputs[0] };
                    layer.pool = sizes;
                    return;
                }
                case Layer.Type.Expand:
                {
                    if (layer.inputs.Length <= 1 || !IsLayerKnown(layer.inputs[1], knownLayersValue))
                        return;

                    float[] shapeValue = knownLayersValue[layer.inputs[1]].ToReadOnlyArray();
                    var shape = new int[shapeValue.Length];
                    for (int i = 0; i < shapeValue.Length; i++)
                        shape[i] = (int)shapeValue[i];

                    layer.pool = shape;
                    layer.inputs = new[] { layer.inputs[0] };
                    return;
                }
                case Layer.Type.Tile:
                {
                    if (layer.inputs.Length <= 1 || !IsLayerKnown(layer.inputs[1], knownLayersValue))
                        return;

                    float[] repeats = knownLayersValue[layer.inputs[1]].ToReadOnlyArray();
                    layer.inputs = new[] { layer.inputs[0] };
                    var shape = new int[4]; // Must be rank 4
                    for (int i = 0; i < shape.Length; i++)
                        shape[i] = i < repeats.Length ? (int)repeats[i] : 1;

                    layer.pool = shape;
                    return;
                }
                case Layer.Type.Reshape:
                {
                    if (layer.inputs.Length <= 1 || !IsLayerKnown(layer.inputs[1], knownLayersValue))
                        return;

                    float[] shapeValue = knownLayersValue[layer.inputs[1]].ToReadOnlyArray();
                    var shape = new int[shapeValue.Length];
                    for (int i = 0; i < shapeValue.Length; i++)
                        shape[i] = (int)shapeValue[i];

                    layer.pool = shape;
                    layer.inputs = new[] { layer.inputs[0] };
                    return;
                }
                case Layer.Type.ConstantOfShape:
                {
                    if (layer.inputs.Length < 1 || !IsLayerKnown(layer.inputs[0], knownLayersValue))
                        return;

                    Tensor input = knownLayersValue[layer.inputs[0]];
                    var shape = Array.ConvertAll(input.ToReadOnlyArray(), x => (int)x);
                    var tensorShape = IRShapeInferenceHelper.ShapeInference.OnnxLayoutToTensorShape(shape);

                    layer.type = Layer.Type.Load;

                    layer.axis = input.dimensions; // TODO real rank
                    layer.datasets = new Layer.DataSet[1];
                    layer.datasets[0].name = layer.name;
                    layer.datasets[0].shape = tensorShape;
                    layer.datasets[0].itemSizeInBytes = 4;
                    layer.datasets[0].length = tensorShape.length;
                    layer.datasets[0].offset = 0;
                    layer.weights = new float[tensorShape.length];

                    var tensor = new Tensor(tensorShape);
                    tensor.Fill(layer.alpha);
                    tensor.ToReadOnlyArray().CopyTo(layer.weights, 0);

                    layer.inputs = new string[0];
                    return;
                }
                case Layer.Type.Activation:
                {
                    if (layer.activation == Layer.Activation.None)
                    {
                        if (layer.inputs.Length < 1 || !IsLayerKnown(layer.inputs[0], knownLayersValue))
                            return;

                        Tensor input = knownLayersValue[layer.inputs[0]];
                        var tensorShape = input.shape;

                        layer.type = Layer.Type.Load;

                        layer.axis = input.dimensions; // TODO real rank
                        layer.datasets = new Layer.DataSet[1];
                        layer.datasets[0].name = layer.name;
                        layer.datasets[0].shape = tensorShape;
                        layer.datasets[0].itemSizeInBytes = 4;
                        layer.datasets[0].length = tensorShape.length;
                        layer.datasets[0].offset = 0;
                        layer.weights = new float[tensorShape.length];

                        input.ToReadOnlyArray().CopyTo(layer.weights, 0);

                        layer.inputs = new string[0];
                    }

                    return;
                }
                default:
                    return;
            }
        }
    }
}
