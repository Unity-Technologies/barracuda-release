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
            Run(ref model, null);
        }

        //TODO this pass is handling data transformation in a destructive way and thus loss validation information.
        //find a cleaner way to report import warnings.
        public void Run(ref Model model, List<Model.ImporterWarning> warnings)
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
            FuseShapesIntoConstants(ref model, inputShapes, inputRanks, ref warnings);
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

        private void FuseShapesIntoConstants(ref Model model, IDictionary<string, TensorShape?> shapesByName, IDictionary<string, int?> ranksByName, ref List<Model.ImporterWarning> warnings)
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
                FuseInputsIntoLayer(ref layer, knownLayersValue, ranksByName, warnings);
                // TODO optimization, pass in index, or add shape
                IRShapeInferenceHelper.RankInference.UpdateKnownTensorRanks(model, ranksByName);
                IRShapeInferenceHelper.ShapeInference.UpdateKnownTensorShapesNCHW(model, ref ranksByName, ref shapesByName);

                if (ModelOptimizer.IsLayerConstant(layer))
                    knownLayersValue[layer.name] = new Tensor(layer.datasets[0].shape, layer.weights);
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
                Layer newLayer = new Layer(layer.name.ToString(), layer.activation);
                newLayer.type = layer.type;
                newLayer.activation = layer.activation;
                newLayer.pad = layer.pad.ToArray();
                newLayer.stride = layer.stride.ToArray();
                newLayer.pool = layer.pool.ToArray();
                newLayer.axis = layer.axis;
                newLayer.alpha = layer.alpha;
                newLayer.beta = layer.beta;
                newLayer.inputs = layer.inputs.ToArray();
                newLayer.datasets = layer.datasets;
                newLayer.weights = layer.weights;
                if(layer.outputs != null)
                    newLayer.outputs = layer.outputs.ToArray();
                if (layer.axes != null)
                    newLayer.axes = layer.axes.ToArray();


                opsModel.layers.Add(newLayer);
                opsModel.outputs.Add(newLayer.name);

                toRunnableNCHW.Run(ref opsModel);

                // bake
                var useCPUforBaking = WorkerFactory.Device.CPU;
                using (var worker = WorkerFactory.CreateWorker(opsModel, useCPUforBaking))
                {
                    var bakedConstant = worker.Execute(layerInputs).CopyOutput();
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

                c.weights = new BarracudaArray(tensor.length);
                BarracudaArray.Copy(tensor.ToReadOnlyArray(), c.weights, tensor.length);
                model.layers.Insert(0,c);
            }

            foreach (var l in knownLayersValue)
                l.Value.Dispose();

            // TODO remove?
            // remove unused constants
            var removeUnusedLayersPass = new Cleanup.RemoveUnusedLayersPass();
            removeUnusedLayersPass.Run(ref model);
        }

        // TODO: refactor with FuseShapesIntoConstants
        public void InferAllShapes(Model model, ref IDictionary<string, TensorShape?> shapesByName, ref IDictionary<string, int?> ranksByName)
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
                FuseInputsIntoLayer(ref layer, knownLayersValue, ranksByName, null);//TODO handle potential folding errors/warnings
                // TODO optimization, pass in index, or add shape
                IRShapeInferenceHelper.ShapeInference.UpdateKnownTensorShapesNCHW(model, ref ranksByName, ref shapesByName);
                IRShapeInferenceHelper.RankInference.UpdateKnownTensorRanks(model, ranksByName);

                if (ModelOptimizer.IsLayerConstant(layer))
                    knownLayersValue[layer.name] = new Tensor(layer.datasets[0].shape, layer.weights);
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
                Layer newLayer = new Layer(layer.name.ToString(), layer.activation);
                newLayer.type = layer.type;
                newLayer.activation = layer.activation;
                newLayer.pad = layer.pad.ToArray();
                newLayer.stride = layer.stride.ToArray();
                newLayer.pool = layer.pool.ToArray();
                newLayer.axis = layer.axis;
                newLayer.alpha = layer.alpha;
                newLayer.beta = layer.beta;
                newLayer.inputs = layer.inputs.ToArray();
                newLayer.datasets = layer.datasets;
                newLayer.weights = layer.weights;
                if (layer.outputs != null)
                    newLayer.outputs = layer.outputs.ToArray();
                if (layer.axes != null)
                    newLayer.axes = layer.axes.ToArray();


                opsModel.layers.Add(newLayer);
                opsModel.outputs.Add(newLayer.name);

                toRunnableNCHW.Run(ref opsModel);

                toRunnableNCHW.Run(ref opsModel);

                // bake
                var useCPUforBaking = WorkerFactory.Device.CPU;
                using (var worker = WorkerFactory.CreateWorker(opsModel, useCPUforBaking))
                {
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

        public void FuseInputsIntoLayer(ref Layer layer, Dictionary<string, Tensor> knownLayersValue, IDictionary<string, int?> ranksByName, List<Model.ImporterWarning> warnings)
        {
            switch (layer.type)
            {
                case Layer.Type.Border2D:
                case Layer.Type.Border3D:
                case Layer.Type.Pad2DEdge:
                case Layer.Type.Pad2DReflect:
                case Layer.Type.Pad2DSymmetric:
                {
                    if (layer.inputs.Length <= 1 || !IsLayerKnown(layer.inputs[1], knownLayersValue))
                        return;

                    float[] padsFloat = knownLayersValue[layer.inputs[1]].ToReadOnlyArray();
                    layer.inputs = new[] { layer.inputs[0] };
                    var pads = Array.ConvertAll(padsFloat, x => (int)x);

                    var starts = pads.Take(pads.Length / 2).ToArray();
                    var ends = pads.Skip(pads.Length / 2).ToArray();
                    bool[] dimHavePadding = new bool[starts.Length];
                    for (int i = 0; i < starts.Length; ++i)  {
                        dimHavePadding[i] = starts[i] != 0 && ends[i] != 0;
                    }

                    if (dimHavePadding.SequenceEqual(new bool []{ false, true, true, false }))
                    {
                        // Look like this padding operator is defined over NHWC layout
                        // We skip first and last dimension thus
                        starts = starts.Skip(1).Take(2).ToArray();
                        ends = ends.Skip(1).Take(2).ToArray();
                        layer.axes = new int[] { -1 };// Mark the layer padding as being imported as NHWC layout
                    }
                    else
                    {
                        // Skip non-spatial dimensions N, C (NCHW layout)
                        starts = starts.Skip(2).ToArray();
                        ends = ends.Skip(2).ToArray();
                    }

                    switch (starts.Length)
                    {
                        case 1: layer.pad = new [] { starts[0], 0, ends[0], 0 }; break; // 1D W => W_
                        case 2: layer.pad = new [] { starts[1], starts[0], ends[1],   ends[0] }; break; // 2D HW => WH
                        default: layer.pad = new [] { starts[2], starts[1], starts[0], ends[2],   ends[1],   ends[0] }; break; // 3D DHW => WHD
                    }

                    float value = 0.0f;
                    if (layer.inputs.Length >= 3 && IsLayerKnown(layer.inputs[2], knownLayersValue))
                        value = knownLayersValue[layer.inputs[2]].ToReadOnlyArray()[0];

                    layer.beta = value;
                    return;
                }
                case Layer.Type.Upsample2D:
                {
                    if (layer.inputs.Length <= 1 || !IsLayerKnown(layer.inputs[1], knownLayersValue))
                        return;

                    float[] scales = knownLayersValue[layer.inputs[1]].ToReadOnlyArray();

                    if (scales[0] == 1 && scales[1] == 1 && scales[2] < 1.0f && scales[3] < 1.0f && layer.axis >= 0.0f)
                    {
                        ValidationHelper.AppendWarning(scales.All(x => Mathf.Approximately(1f / x, Mathf.Round(1f / x))),
                            layer.name, $"Only inverse of scale values which produce integer are currently supported. Inverse of scale value will be rounded to closest integer.", ref warnings, MessageType.Warning);

                        scales = new[] { scales[2], scales[3] };
                        layer.type = Layer.Type.AvgPool2D;
                        layer.pad = new[] { 0, 0, 0, 0 };
                        var inverseScalesRoundedToInt = scales.Select(x => (int)Mathf.Round(1f / x)).ToArray();
                        layer.stride = inverseScalesRoundedToInt;
                        layer.pool = inverseScalesRoundedToInt;
                    }
                    else
                    {
                        ValidationHelper.AppendWarning(scales.All(x => Mathf.Approximately(x, Mathf.Round(x))),
                            layer.name, $"Only integer scale values are currently supported. Scale value will be rounded to closest integer value.", ref warnings, MessageType.Warning);

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
                case Layer.Type.MatMul:
                {
                    var input0 = layer.inputs[0]; var input1 = layer.inputs[1];
                    if (!ranksByName.ContainsKey(input0) || !ranksByName[input0].HasValue)
                        return;
                    if (!ranksByName.ContainsKey(input1) || !ranksByName[input1].HasValue)
                        return;
                    int rank0 = ranksByName[input0].Value;
                    int rank1 = ranksByName[input1].Value;

                    if(rank0 > 2 || rank1 > 2)
                        return;

                    if (!IsLayerKnown(input1, knownLayersValue))
                        return;

                    layer.type = Layer.Type.Dense;

                    var weight = knownLayersValue[input1];
                        weight = weight.Reshape(new TensorShape(weight.batch, weight.height));
                    var biasShape = new TensorShape(1, 1, 1, weight.shape.channels);

                    layer.inputs = new [] { input0 };
                    layer.datasets = new Layer.DataSet[2];
                    layer.datasets[0].name            = $"{layer.name}/W";
                    layer.datasets[0].shape           = weight.shape;
                    layer.datasets[0].itemSizeInBytes = 4;
                    layer.datasets[0].length          = weight.shape.length;
                    layer.datasets[0].offset          = 0;
                    layer.datasets[1].name            = $"{layer.name}/B";
                    layer.datasets[1].shape           = biasShape;
                    layer.datasets[1].itemSizeInBytes = 4;
                    layer.datasets[1].length          = biasShape.length;
                    layer.datasets[1].offset          = weight.shape.length;
                    layer.weights                     = new BarracudaArray(weight.shape.length + biasShape.length);

                    weight.ToReadOnlyArray().CopyToBarracudaArray(layer.weights, 0);
                    var zeroBias = new float[biasShape.length];
                    zeroBias.CopyToBarracudaArray(layer.weights, weight.shape.length);
                    return;
                }
                case Layer.Type.Tile:
                {
                    if (layer.inputs.Length <= 1 || !IsLayerKnown(layer.inputs[1], knownLayersValue))
                        return;

                    var shape = Array.ConvertAll(knownLayersValue[layer.inputs[1]].ToReadOnlyArray(), x => (int)x);
                    layer.pool = shape;

                    layer.inputs = new[] { layer.inputs[0] };
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


                    layer.axis = shape.Length;
                    layer.datasets = new Layer.DataSet[1];
                    layer.datasets[0].name = layer.name;
                    layer.datasets[0].shape = tensorShape;
                    layer.datasets[0].itemSizeInBytes = 4;
                    layer.datasets[0].length = tensorShape.length;
                    layer.datasets[0].offset = 0;
                    layer.weights = new BarracudaArray(tensorShape.length);

                    var tensor = new Tensor(tensorShape);
                    tensor.Fill(layer.alpha);
                    tensor.ToReadOnlyArray().CopyToBarracudaArray(layer.weights, 0);

                    layer.inputs = new string[0];
                    return;
                }
                case Layer.Type.LSTM:
                {
                    if (layer.inputs.Length <= 3 || !knownLayersValue.TryGetValue(layer.inputs[1], out Tensor W)
                        || !knownLayersValue.TryGetValue(layer.inputs[2], out Tensor R)
                        || !knownLayersValue.TryGetValue(layer.inputs[3], out Tensor B))
                        return;

                    var ops = new ReferenceCPUOps();
                    using (var td = new TensorScope())
                    {
                        TensorScope.F _ = td._;

                        W = _(ops.Transpose(W, new[] { 2, 0, 3, 1 }));
                        R = _(ops.Transpose(R, new[] { 2, 0, 3, 1 }));
                        B = _(ops.Transpose(B, new[] { 0, 2, 3, 1 }));

                        OpsUtils.BakeConstantWRBIntoLSTMLayer(layer, W, R, B);
                    }

                    layer.inputs = new[] { layer.inputs[0], layer.inputs[4], layer.inputs[5] };

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

                        int rank = input.dimensions;
                        if (ranksByName[layer.name] != null)
                            rank = ranksByName[layer.name].Value;

                        layer.axis = rank;
                        layer.datasets = new Layer.DataSet[1];
                        layer.datasets[0].name = layer.name;
                        layer.datasets[0].shape = tensorShape;
                        layer.datasets[0].itemSizeInBytes = 4;
                        layer.datasets[0].length = tensorShape.length;
                        layer.datasets[0].offset = 0;
                        layer.weights = new BarracudaArray(tensorShape.length);

                        input.ToReadOnlyArray().CopyToBarracudaArray(layer.weights, 0);

                        layer.inputs = new string[0];
                    }

                    return;
                }
                case Layer.Type.Range:
                {
                    if (layer.inputs.Length < 3 || !IsLayerKnown(layer.inputs[0], knownLayersValue) || !IsLayerKnown(layer.inputs[1], knownLayersValue) || !IsLayerKnown(layer.inputs[2], knownLayersValue))
                        return;

                    Tensor input0 = knownLayersValue[layer.inputs[0]];
                    Tensor input1 = knownLayersValue[layer.inputs[1]];
                    Tensor input2 = knownLayersValue[layer.inputs[2]];

                    var start = input0[0];
                    var limit = input1[0];
                    var delta = input2[0];

                    int nbOfElements = Mathf.Max((int)Mathf.Ceil((limit - start) / delta), 0);

                    layer.type = Layer.Type.Load;

                    layer.axis = 1;
                    layer.datasets = new Layer.DataSet[1];
                    layer.datasets[0].name = layer.name;
                    layer.datasets[0].shape = new TensorShape(nbOfElements, 1);
                    layer.datasets[0].itemSizeInBytes = 4;
                    layer.datasets[0].length = nbOfElements;
                    layer.datasets[0].offset = 0;
                    layer.weights = new BarracudaArray(nbOfElements);

                    for(int i=0; i < nbOfElements; ++i)
                    {
                        layer.weights[i] = start + (i * delta);
                    }

                    layer.inputs = new string[0];
                    return;
                }
                case Layer.Type.StridedSlice:
                {
                    if (layer.inputs.Length <= 1 ||
                        !IsLayerKnown(layer.inputs[1], knownLayersValue) || !IsLayerKnown(layer.inputs[2], knownLayersValue) || !IsLayerKnown(layer.inputs[3], knownLayersValue) || !IsLayerKnown(layer.inputs[4], knownLayersValue))
                            return;

                    var starts = Array.ConvertAll(knownLayersValue[layer.inputs[1]].ToReadOnlyArray(), x => x <= (float)int.MinValue ? int.MinValue : x >= (float)int.MaxValue ? int.MaxValue : (int)x);
                    var ends = Array.ConvertAll(knownLayersValue[layer.inputs[2]].ToReadOnlyArray(), x => x <= (float)int.MinValue ? int.MinValue : x >= (float)int.MaxValue ? int.MaxValue : (int)x);

                    var strides = Enumerable.Repeat(1, starts.Length).Select(v => (int)v).ToArray();
                    if (layer.inputs.Length >= 4)
                        strides = Array.ConvertAll(knownLayersValue[layer.inputs[3]].ToReadOnlyArray(), x => (int)x);
                    var axes = Enumerable.Range(0, starts.Length).Select(v => (int)v).ToArray();
                    if (layer.inputs.Length == 5)
                        axes = Array.ConvertAll(knownLayersValue[layer.inputs[4]].ToReadOnlyArray(), x => (int)x);

                    layer.pad = starts;
                    layer.pool = ends;
                    layer.stride = strides;
                    layer.axes = axes;

                    layer.inputs = new[] { layer.inputs[0] };

                    return;
                }
                case Layer.Type.Squeeze:
                {
                    if (layer.inputs.Length <= 1 || !IsLayerKnown(layer.inputs[1], knownLayersValue))
                        return;

                    int[] axes = Array.ConvertAll(knownLayersValue[layer.inputs[1]].ToReadOnlyArray(), x => (int)x);

                    layer.pool = axes;
                    layer.inputs = new[] { layer.inputs[0] };
                    return;
                }
                case Layer.Type.Unsqueeze:
                {
                    if (layer.inputs.Length <= 1 || !IsLayerKnown(layer.inputs[1], knownLayersValue))
                        return;

                    int[] axes = Array.ConvertAll(knownLayersValue[layer.inputs[1]].ToReadOnlyArray(), x => (int)x);

                    layer.pool = axes;
                    layer.inputs = new[] { layer.inputs[0] };
                    return;
                }
                case Layer.Type.Pad:
                {
                    if (layer.inputs.Length <= 1)
                        return;
                    if (layer.inputs.Length == 2 && !IsLayerKnown(layer.inputs[1], knownLayersValue))
                        return;
                    if (layer.inputs.Length == 3 && !IsLayerKnown(layer.inputs[1], knownLayersValue) && !IsLayerKnown(layer.inputs[2], knownLayersValue))
                        return;

                    float value = (layer.inputs.Length == 2) ? layer.beta : knownLayersValue[layer.inputs[2]].ToReadOnlyArray()[0];
                    int[] pads = Array.ConvertAll(knownLayersValue[layer.inputs[1]].ToReadOnlyArray(), x => (int)x);

                    layer.beta = value;
                    layer.pad = pads;
                    layer.inputs = (layer.inputs.Length == 2) ? new [] { layer.inputs[0] } : new [] { layer.inputs[0], layer.inputs[1] };
                    return;
                }
                default:
                    return;
            }
        }
    }
}
