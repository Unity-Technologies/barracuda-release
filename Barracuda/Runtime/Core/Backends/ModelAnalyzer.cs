using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;

using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Profiling;

[assembly: InternalsVisibleTo("Unity.Barracuda.ONNX")]
[assembly: InternalsVisibleTo("Unity.Barracuda.Editor")]

namespace Unity.Barracuda {


internal class ModelAnalyzer
{
    public static string GetDefaultInputName(Model model)
    {
        bool modelHasOnlyOneInput = model.inputs.Count == 1;
        if (modelHasOnlyOneInput)
            return model.inputs[0].name;

        var memories = new HashSet<string>();
        foreach (var m in model.memories)
            memories.Add(m.input);

        // find the first unconnected input as a default model input
        var previousLayerNames = new HashSet<string>();
        foreach (var l in model.layers)
        {
            previousLayerNames.Add(l.name);

            bool layerDoesNotNeedInput = (l.type == Layer.Type.Load);

            if (layerDoesNotNeedInput)
                continue;

            foreach (var inputName in l.inputs)
            {
                bool inputIsUnconnected = !previousLayerNames.Contains(inputName);
                bool inputIsNotPartOfMemory = !memories.Contains(inputName);

                if (inputIsUnconnected && inputIsNotPartOfMemory)
                    return inputName;
            }
        }

        return "";
    }

    static public string GetDefaultOutputName(Model model)
    {
        if (model.outputs.Count == 1)
            return model.outputs[0];

        if (model.layers.Count > 0)
        {
            var lastLayer = model.layers[model.layers.Count - 1];
            return lastLayer.name;
        }

        return "";
    }

    public static TensorShape?[] ListTemporaryTensorShapes(Model model, IDictionary<string, TensorShape> inputShapes)
    {
        IDictionary<string, TensorShape?> shapesByName;
        return ListTemporaryTensorShapes(model, inputShapes, out shapesByName);
    }

    public static TensorShape?[] ListTemporaryTensorShapes(Model model, IDictionary<string, TensorShape> inputShapes,
        out IDictionary<string, TensorShape?> shapesByName)
    {
        Profiler.BeginSample ("Barracuda.ListTemporaryTensorShapes");
        var shapes = new List<TensorShape?>();
        shapesByName = new Dictionary<string, TensorShape?>();
        foreach (var entry in inputShapes)
            shapesByName.Add(entry.Key, entry.Value);

        TensorShape? Xn;
        shapesByName.TryGetValue(GetDefaultInputName(model), out Xn); // default input
        TensorShape? O = Xn;

        foreach (var l in model.layers)
        {
            if (l.inputs.Length > 0 && shapesByName.TryGetValue(l.inputs[0], out TensorShape? xShape))
                Xn = xShape;
            else
                Xn = O; // previous output is used, if-and-only-if layer has no explicit inputs

            if (Xn == null)
            {
                shapes.Add(Xn);
                shapesByName.Add(l.name, Xn);
                continue;
            }

            TensorShape X = Xn.Value;

            if (l.type == Layer.Type.Dense)
            {
                Assert.IsNotNull(l.datasets);
                var W = l.datasets[0].shape;
                O = new TensorShape(X.flatHeight, W.flatWidth);
            }
            else if (l.type == Layer.Type.MatMul)
            {
                if (!shapesByName.ContainsKey(l.inputs[1]) || shapesByName[l.inputs[1]] == null)
                {
                    O = null;
                    break;
                }

                var Y = shapesByName[l.inputs[1]].Value;

                int rankX;
                int rankY;
                List<int> onnxXshape;
                List<int> onnxYshape;

                if (l.pool == null || l.pool.Length == 0)
                {
                    LegacyGetXYRanks(X, Y, out rankX, out rankY);
                }
                else
                {
                    rankX = l.pool[0];
                    rankY = l.pool[1];
                }

                onnxXshape = Compiler.IRShapeInferenceHelper.ShapeInference.BarracudaShapeToOnnxLayout(X, rankX);
                onnxYshape = Compiler.IRShapeInferenceHelper.ShapeInference.BarracudaShapeToOnnxLayout(Y, rankY);

                int rankO = Math.Max(rankX, rankY);

                // pad 1 on front of shape to both be rankO shape
                for (int i = 0; i < (rankX - rankY); i++)
                    onnxYshape.Insert(0, 1);

                for (int i = 0; i < (rankY - rankX); i++)
                    onnxXshape.Insert(0, 1);

                if (rankO == 2)
                    O = new TensorShape(onnxXshape[0], 1, 1, onnxYshape[1]);
                else if (rankO == 3)
                    O = new TensorShape(Math.Max(onnxXshape[0], onnxYshape[0]), 1, onnxYshape[2], onnxXshape[1]);
                else
                    O = new TensorShape(Math.Max(onnxXshape[0], onnxYshape[0]), onnxXshape[2], onnxYshape[3], Math.Max(onnxXshape[1], onnxYshape[1]));
            }
            else if (
                l.type == Layer.Type.Conv2D ||
                l.type == Layer.Type.Conv3D ||
                l.type == Layer.Type.DepthwiseConv2D)
            {
                var K = l.datasets[0].shape;

                Assert.IsNotNull(l.stride);
                Assert.IsNotNull(l.pad);
                var pad = X.AdjustPadToKernel(K, l.stride, l.pad);

                O = X.ApplyKernel(K, l.stride, pad);
            }
            else if (
                l.type == Layer.Type.Conv2DTrans)
            {
                var K = l.datasets[0].shape;
                Assert.IsNotNull(l.stride);
                Assert.IsNotNull(l.pad);
                // pool size is treated as output_adjustment aka output_padding here
                var outputAdjustment = l.pool;
                var pad = X.AdjustPadToKernel(K, l.stride, l.pad);
                O = X.ApplyKernelInverse(K, l.stride, pad, outputAdjustment);
            }
            else if (
                l.type == Layer.Type.Upsample2D)
            {
                if(inputShapes.Count > 1)
                {
                    O = null;
                }
                else
                {
                    // pool size is treated as upsample coefficient here
                    Assert.IsNotNull(l.pool);
                    Assert.AreEqual(l.pool.Length, 2);
                    O = new TensorShape(X.batch, X.height * l.pool[1], X.width * l.pool[0], X.channels);
                }
            }
            else if (
                l.type == Layer.Type.Upsample3D)
            {
                if(inputShapes.Count > 1)
                {
                    O = null;
                }
                else
                {
                    // pool size is treated as upsample coefficient here
                    Assert.IsNotNull(l.pool);
                    Assert.AreEqual(l.pool.Length, 3);
                    O = new TensorShape(1,1,X.batch, 1, X.depth * l.pool[2], X.height * l.pool[1], X.width * l.pool[0], X.channels);
                }
            }
            else if (
                l.type == Layer.Type.Resample2D)
            {
                if(inputShapes.Count > 1)
                {
                    O = null;
                }
                else
                {
                    // pool is treated as resample size here
                    var size = l.pool;
                    Assert.IsNotNull(size);
                    Assert.AreEqual(size.Length, 2);
                    O = new TensorShape(X.batch, size[1], size[0], X.channels);
                }
            }
            else if (
                l.type == Layer.Type.DepthToSpace)
            {
                    // pool size is treated as blocksize here
                    Assert.IsNotNull(l.pool);
                    Assert.AreEqual(l.pool.Length, 2);
                    Assert.AreEqual(X.channels % (l.pool[0] * l.pool[1]), 0);
                    O = new TensorShape(X.batch, X.height * l.pool[1], X.width * l.pool[0], X.channels / (l.pool[0] * l.pool[1]));
            }
            else if (
                l.type == Layer.Type.SpaceToDepth)
            {
                // pool size is treated as blocksize here
                Assert.IsNotNull(l.pool);
                Assert.AreEqual(l.pool.Length, 2);
                O = new TensorShape(X.batch, X.height / l.pool[1], X.width / l.pool[0], X.channels * (l.pool[0] * l.pool[1]));
            }
            else if (
                l.type == Layer.Type.MaxPool2D ||
                l.type == Layer.Type.AvgPool2D)
            {
                Assert.IsNotNull(l.pool);
                Assert.IsNotNull(l.stride);
                Assert.IsNotNull(l.pad);
                var pad = X.AdjustPadToPool(l.pool, l.stride, l.pad);
                O = X.ApplyPool(l.pool, l.stride, pad);
            }
            else if (
                l.type == Layer.Type.GlobalMaxPool2D ||
                l.type == Layer.Type.GlobalAvgPool2D)
            {
                O = new TensorShape(X.batch, 1, 1, X.channels);
            }
            else if (
                l.type == Layer.Type.Border2D ||
                l.type == Layer.Type.Border3D ||
                l.type == Layer.Type.Pad2DReflect ||
                l.type == Layer.Type.Pad2DSymmetric ||
                l.type == Layer.Type.Pad2DEdge)
            {
                Assert.IsNotNull(l.pad);
                O = X.ApplyBorder(l.pad);
            }
            else if (
                l.type == Layer.Type.Conv3D ||
                l.type == Layer.Type.Conv3DTrans ||
                l.type == Layer.Type.Upsample3D ||
                l.type == Layer.Type.MaxPool3D ||
                l.type == Layer.Type.AvgPool3D ||
                l.type == Layer.Type.GlobalMaxPool3D ||
                l.type == Layer.Type.GlobalAvgPool3D ||
                l.type == Layer.Type.Border3D)
            {
                throw new NotImplementedException();
            }
            else if (
                l.type == Layer.Type.RandomNormal ||
                l.type == Layer.Type.RandomUniform)
            {
                Assert.IsNotNull(l.pool);
                // pool size is treated as shape constant, if not empty
                // otherwise shape of the previous tensor is used
                if (l.pool.Length > 0)
                    O = new TensorShape(l.pool);
                else
                    O = X;
            }
            else if (
                l.type == Layer.Type.Multinomial)
            {
                Assert.IsNotNull(l.pool);
                Assert.AreEqual(l.pool.Length, 1);
                O = new TensorShape(X.batch, l.pool[0]);
            }
            else if (
                l.type == Layer.Type.OneHot)
            {
                Assert.IsNotNull(l.pool);
                Assert.AreEqual(l.pool.Length, 1);
                int features = X.flatWidth;
                int depth = l.pool[0];

                if (X.flatWidth == 1) // 1D input
                    O = new TensorShape(X.batch, depth);
                else
                    O = new TensorShape(X.batch, 1, depth, features);
            }
            else if (
                l.type == Layer.Type.Add ||
                l.type == Layer.Type.Sub ||
                l.type == Layer.Type.Mul ||
                l.type == Layer.Type.Div ||
                l.type == Layer.Type.Pow ||
                l.type == Layer.Type.Min ||
                l.type == Layer.Type.Max ||
                l.type == Layer.Type.Mean||
                l.type == Layer.Type.Greater ||
                l.type == Layer.Type.GreaterEqual ||
                l.type == Layer.Type.Less ||
                l.type == Layer.Type.LessEqual ||
                l.type == Layer.Type.Equal ||
                l.type == Layer.Type.LogicalOr ||
                l.type == Layer.Type.LogicalAnd ||
                l.type == Layer.Type.LogicalXor)
            {
                // gather shapes by names
                var list = new List<TensorShape>(l.inputs.Length);
                bool allShapesKnown = true;
                foreach (var i in l.inputs)
                {
                    if (shapesByName.TryGetValue(i, out TensorShape? shape) && shape != null)
                        list.Add(shape.Value);
                    else
                        allShapesKnown = false;
                }

                O = allShapesKnown ? TensorExtensions.Max(list.ToArray()) : default(TensorShape?);
            }
            else if (
                l.type == Layer.Type.ReduceL1 ||
                l.type == Layer.Type.ReduceL2 ||
                l.type == Layer.Type.ReduceLogSum ||
                l.type == Layer.Type.ReduceLogSumExp ||
                l.type == Layer.Type.ReduceMax ||
                l.type == Layer.Type.ReduceMean ||
                l.type == Layer.Type.ReduceMin ||
                l.type == Layer.Type.ReduceProd ||
                l.type == Layer.Type.ReduceSum ||
                l.type == Layer.Type.ReduceSumSquare ||
                l.type == Layer.Type.ArgMax ||
                l.type == Layer.Type.ArgMin)
            {
                O = X.Reduce(l.axis);
            }
            else if (
                l.type == Layer.Type.Flatten)
            {
                O = X.Flatten();
            }
            else if (
                l.type == Layer.Type.Reshape)
            {
                // pool size is treated as the shape, if not empty
                var size = l.pool;

                Assert.IsNotNull(size);

                if (size.Length == 0 && l.inputs.Length > 1)
                {
                    switch (l.axis)
                    {
                        // Legacy - use the shape of the input tensor as the shape
                        case -1:
                            if (shapesByName.TryGetValue(l.inputs[1], out TensorShape? shape))
                                size = shape.Value.ToArray();
                            break;

                        // Use the tensor values as the shape; Calculated at runtime
                        case 1:
                            O = null;
                            break;
                    }

                    if (O == null)
                        break;
                }

                Assert.IsTrue( (size.Length == 4) || (size.Length == 8));
                O = X.Reshape(size);
            }
            else if (
                l.type == Layer.Type.Expand)
            {
                // pool size is treated as new shape
                var newShape = l.pool;

                Assert.IsNotNull(newShape);
                Assert.IsTrue(newShape.Length == 8 || newShape.Length == 4);

                O = new TensorShape(newShape);
            }
            else if (
                l.type == Layer.Type.Transpose)
            {
                var permutations = l.pool;
                if (permutations == null)
                    O = new TensorShape(X.flatWidth, X.flatHeight);
                else
                {
                    Assert.IsTrue(permutations.Length == 8 || permutations.Length == 4);
                    O = X.Permute(permutations);
                }
            }
            else if (
                l.type == Layer.Type.Gather)
            {
                if (!shapesByName.TryGetValue(l.inputs[0], out TensorShape? input0Shape) || input0Shape == null
                    || !shapesByName.TryGetValue(l.inputs[1], out TensorShape? input1Shape) || input1Shape == null)
                {
                    O = null;
                    break;
                }

                int[] shape = input0Shape.Value.ToArray();
                shape[l.axis] = input1Shape.Value.length;

                O = new TensorShape(shape);
            }
            else if (
                l.type == Layer.Type.Squeeze ||
                l.type == Layer.Type.Unsqueeze)
            {
                O = X;
            }
            else if (
                l.type == Layer.Type.Concat)
            {
                // gather shapes by names
                var list = new List<TensorShape>(l.inputs.Length);
                bool allShapesKnown = true;
                foreach (var i in l.inputs)
                {
                    if (!shapesByName.TryGetValue(i, out var shape) || shape == null)
                    {
                        allShapesKnown = false;
                        continue;
                    }
                    list.Add(shape.Value);
                }

                O = allShapesKnown ? TensorExtensions.Concat(list.ToArray(), l.axis) : default(TensorShape?);
            }
            else if (
                l.type == Layer.Type.StridedSlice)
            {
                Assert.IsNotNull(l.pad);
                Assert.IsNotNull(l.pool);
                Assert.IsNotNull(l.stride);
                O = X.ApplyStridedSlice(l.pad, l.pool, l.stride);
            }
            else if (
                l.type == Layer.Type.Tile)
            {
                // pool size is treated as tiling coefficient here
                Assert.IsNotNull(l.pool);
                var scale = l.pool;
                O = X.Scale(scale);
            }
            else if (
                l.type == Layer.Type.Load)
            {
                O = l.datasets[0].shape;
            }
            else if (// elementwise operations
                l.type == Layer.Type.Nop ||
                l.type == Layer.Type.Activation ||
                l.type == Layer.Type.ScaleBias ||
                l.type == Layer.Type.Normalization ||
                l.type == Layer.Type.LRN ||
                l.type == Layer.Type.Dropout ||
                l.type == Layer.Type.LogicalNot ||
                l.type == Layer.Type.Where)
            {
                // works in place, keeps the same shape size
                O = X;
            }
            else if (
                l.type == Layer.Type.TopKIndices ||
                l.type == Layer.Type.TopKValues ||
                l.type == Layer.Type.NonMaxSuppression ||
                l.type == Layer.Type.NonZero)
            {
                // Calculated at runtime
                O = null;
            }
            else if (l.type == Layer.Type.Shape)
            {
                int shapeRank = l.axis > 0 ? 1 : X.length;
                O = new TensorShape(shapeRank, 1, 1, 1);
            }
            else if (
                l.type == Layer.Type.Conv3D ||
                l.type == Layer.Type.Conv3DTrans ||
                l.type == Layer.Type.Upsample3D ||
                l.type == Layer.Type.MaxPool3D ||
                l.type == Layer.Type.AvgPool3D ||
                l.type == Layer.Type.GlobalMaxPool3D ||
                l.type == Layer.Type.GlobalAvgPool3D ||
                l.type == Layer.Type.Border3D)
            {
                throw new NotImplementedException("3D operations are not implemented yet!");
            }
            else
            {
                throw new NotImplementedException($"Layer type {l.type} needs to be explicitly handled");
            }

            shapes.Add(O);
            shapesByName.Add(l.name, O);
        }

        Profiler.EndSample();
        return shapes.ToArray();
    }

    // TODO: Remove when the legacy importer / code path is no longer needed (i.e. when pool is always set)
    public static void LegacyGetXYRanks(TensorShape X, TensorShape Y, out int rankX, out int rankY)
    {
        // ONNX rank 2 : N,C => N,1,1,C
        //      rank 3 : one must be N C W, (batches = N) => N, 1, W, C
        //      rank 4 : one must be N C H W, (batches = N * C) => N H W C
        // X and Y can be different ranks
        var onnxXshape = new List<int> { X.batch, X.channels, X.height, X.width };
        if (X.height == 1) onnxXshape = new List<int> { X.batch, X.channels, X.width, 1 };
        var onnxYshape = new List<int> { Y.batch, Y.channels, Y.height, Y.width };
        if (Y.height == 1) onnxYshape = new List<int> { Y.batch, Y.channels, Y.width, 1 };

        rankX = 0;
        for (int i = 3; i >= 0; i--)
        {
            if (onnxXshape[i] != 1)
            {
                rankX = i + 1;
                break;
            }
        }

        rankY = 0;
        for (int i = 3; i >= 0; i--)
        {
            if (onnxYshape[i] != 1)
            {
                rankY = i + 1;
                break;
            }
        }
    }

    public static bool TryGetOutputTensorShape(Model model, IDictionary<string, TensorShape> inputShapes, string output, out TensorShape shape)
    {
        shape = new TensorShape();
        IDictionary<string, TensorShape?> shapesByName;
        ListTemporaryTensorShapes(model, inputShapes, out shapesByName);

        TensorShape? dynamicShape;
        bool found = shapesByName.TryGetValue(output, out dynamicShape);
        if (found && (dynamicShape != null))
            shape = dynamicShape.Value;
        return found;
    }

    public static bool TryGetOutputTensorShape(Model model, string output, out TensorShape shape)
    {
        var inputShapes = new Dictionary<string, TensorShape>();
        foreach (var i in model.inputs)
            inputShapes.Add(i.name, new TensorShape(i.shape));
        return TryGetOutputTensorShape(model, inputShapes, output, out shape);
    }

    public static bool FindLayerByName(Model model, string name, out Layer layer)
    {
        layer = new Layer("",Layer.Type.Nop);
        foreach (var l in model.layers)
        {
            if (l.name == name)
            {
                layer = l;
                return true;
            }
        }
        return false;
    }

    public static HashSet<Layer> FindLayersThatRequireStorage(Model model)
    {
        var allInputsExceptFromPreviousLayer = new HashSet<string>();
        Layer prevLayer = null;
        foreach (var layer in model.layers)
        {
            foreach (var input in layer.inputs)
                if (prevLayer != null && input != prevLayer.name)
                    allInputsExceptFromPreviousLayer.Add(input);
            prevLayer = layer;
        }

        var allOutputs = new HashSet<string>();
        foreach (var output in model.outputs)
            allOutputs.Add(output);
        foreach (var memory in model.memories)
            allOutputs.Add(memory.output);
        allOutputs.Add(GetDefaultOutputName(model));

        var requireStorage = new HashSet<Layer>();
        foreach (var layer in model.layers)
        {
            // loading constant tensor requires storage
            if (layer.type == Layer.Type.Load)
                requireStorage.Add(layer);

            // @TBD: implement safety check that ensures Nop never has input
            // otherwise it has to be treated as Load operation
            if (layer.type == Layer.Type.Nop)
                requireStorage.Add(layer);

            if (allInputsExceptFromPreviousLayer.Contains(layer.name) ||
                allOutputs.Contains(layer.name))
                requireStorage.Add(layer);
        }

        return requireStorage;
    }

    public static HashSet<Layer> FindUpstreamLayers(Model model, string[] outputs)
    {
        // TODO: replace with var layersByName = model.layers.ToDictionary(i => i.name, i => i);
        var layersByName = new Dictionary<string, Layer>();
        foreach (var l in model.layers)
            layersByName.Add(l.name, l);

        var connected = new HashSet<Layer>();
        var layersToVisit = new HashSet<Layer>();
        foreach (var o in outputs)
            if (layersByName.ContainsKey(o))
            {
                layersToVisit.Add(layersByName[o]);
                connected.Add(layersByName[o]);
            }

        while (layersToVisit.Count > 0)
        {
            var visitNext = new HashSet<Layer>();
            foreach (var l in layersToVisit)
                foreach (var i in l.inputs)
                    if (layersByName.ContainsKey(i))
                    {
                        visitNext.Add(layersByName[i]);
                        connected.Add(layersByName[i]);
                    }

            layersToVisit = visitNext;
        }
        return connected;
    }

    public static TensorShape FindLargestNecessaryTensorShape(Model model, IDictionary<string, TensorShape> inputShapes)
    {
        Profiler.BeginSample ("Barracuda.FindLargestNecessaryTensorShape");

        var shapes = ListTemporaryTensorShapes(model, inputShapes);

        var maxTensorShape = new TensorShape(1,1,1,1);
        foreach (var X in shapes)
            if (X?.length > maxTensorShape.length)
                maxTensorShape = X.Value;

        Profiler.EndSample ();

        return maxTensorShape;
    }

    public static TensorShape FindLargestArgumentTensorShape(Model model)
    {
        TensorShape maxTensorShape = new TensorShape(1,1,1,1);
        foreach (var layer in model.layers)
            foreach (var arg in layer.datasets)
                if (arg.shape.length > maxTensorShape.length)
                    maxTensorShape = arg.shape;

        return maxTensorShape;
    }

    public static string[] FindUnusedLayers(Model model)
    {
        var layerUsageByName = model.layers.ToDictionary(i => i.name, i => false);
        foreach (var layer in model.layers)
        {
            if (layer.flags.HasFlag(Layer.Flags.Preserve))
                layerUsageByName[layer.name] = true;

            foreach (var i in layer.inputs)
            {
                layerUsageByName[i] = true;
            }
        }

        foreach (var o in model.outputs)
        {
            layerUsageByName[o] = true;
        }

        foreach (var mem in model.memories)
        {
            layerUsageByName[mem.output] = true;
        }

        return layerUsageByName.Where(keyValue => !keyValue.Value).Select(keyValue => keyValue.Key).ToArray();
    }

    private static string[] FindBrokenLinks(Model model, HashSet<string> links)
    {
        var allVariables = new HashSet<string>(model.layers.Select(i => i.name));
        var globalInputs = new HashSet<string>(model.inputs.Select(i => i.name));
        var memoryInputs = new HashSet<string>(model.memories.Select(i => i.input));
        allVariables.UnionWith(globalInputs);
        allVariables.UnionWith(memoryInputs);

        var brokenLinks = links;
        brokenLinks.ExceptWith(allVariables);
        return brokenLinks.ToArray();
    }

    private static string[] FindBrokenLinks(Model model, string[] links)
    {
        return FindBrokenLinks(model, new HashSet<string>(links));
    }

    public static string[] FindBrokenLinks(Model model)
    {
        // check global outputs
        var linksToInspect = new HashSet<string>(model.outputs);

        // and all layers
        foreach (var layer in model.layers)
            foreach (var i in layer.inputs)
                linksToInspect.Add(i);

        return FindBrokenLinks(model, linksToInspect);
    }

    public static string[] FindUnconnectedInputs(Model model)
    {
        var unconnected = model.inputs.ToDictionary(i => i.name, i => true);

        // check global outputs
        foreach (var o in model.outputs)
            unconnected.Remove(o);

        // and all layers
        foreach (var layer in model.layers)
            foreach (var i in layer.inputs)
                unconnected.Remove(i);

        return unconnected.Keys.ToArray();
    }

    static public string[] FindUnconnectedOutputs(Model model)
    {
        return FindBrokenLinks(model, model.outputs.ToArray());
    }

    public static bool IsLayerBroacastable(Layer layer)
    {
        return layer.type == Layer.Type.Add ||
                layer.type == Layer.Type.Sub ||
                layer.type == Layer.Type.Mul ||
                layer.type == Layer.Type.Div ||
                layer.type == Layer.Type.Pow ||
                layer.type == Layer.Type.Min ||
                layer.type == Layer.Type.Max ||
                layer.type == Layer.Type.Mean ||
                layer.type == Layer.Type.Greater ||
                layer.type == Layer.Type.GreaterEqual ||
                layer.type == Layer.Type.Less ||
                layer.type == Layer.Type.LessEqual ||
                layer.type == Layer.Type.Equal ||
                layer.type == Layer.Type.LogicalOr ||
                layer.type == Layer.Type.LogicalAnd ||
                layer.type == Layer.Type.LogicalXor ||
                layer.type == Layer.Type.Where ||
                layer.type == Layer.Type.Concat;
    }

    // Allow some unknown input dimension for shape inference pass
    // for now batch does not yield problematic shape inference, so allow for unkown batch
    public static bool IsInputShapeAcceptablyKnowForShapeInference(Model.Input input) // acceptable unknown shape : N
    {
        for (int i = 0; i < input.shape.Length; i++)
        {
            var x = input.shape[i];
            if (x <= 0 && i != TensorShape.DataBatch)
                return false;
        }
        return true;
    }
}


} // namespace Unity.Barracuda
