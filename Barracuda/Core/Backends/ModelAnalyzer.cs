using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq; // ToArray(), ToDictionary()

using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Profiling;

namespace Barracuda {


public class ModelAnalyzer
{
    static public string GetDefaultInputName(Model model)
    {
        if (model.inputs.Count == 1)
            return model.inputs[0].name;

        var previousLayerNames = new HashSet<string>();

        // find first unconnected layer
        foreach (var l in model.layers)
        {
            previousLayerNames.Add(l.name);

            bool layerDoesNotNeedInput = (l.type == Layer.Type.Load);

            if (layerDoesNotNeedInput)
                continue;

            if (l.inputs.Length != 1)
                continue;

            // treat layer as default input layer
            // if-and-only-if layer has only 1 input AND is not connected to any previous layer
            var inputName = l.inputs[0];
            if (!previousLayerNames.Contains(inputName))
                return inputName;
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

    static public TensorShape[] ListTemporaryTensorShapes(Model model, IDictionary<string, TensorShape> inputShapes)
    {
        IDictionary<string, TensorShape> shapesByName;
        return ListTemporaryTensorShapes(model, inputShapes, out shapesByName);
    }

    static public TensorShape[] ListTemporaryTensorShapes(Model model, IDictionary<string, TensorShape> inputShapes,
        out IDictionary<string, TensorShape> shapesByName)
    {
        Profiler.BeginSample ("Barracuda.ListTemporaryTensorShapes");
        var shapes = new List<TensorShape>();
        shapesByName = new Dictionary<string, TensorShape>();
        foreach (var entry in inputShapes)
            shapesByName.Add(entry.Key, entry.Value);

        TensorShape X;
        shapesByName.TryGetValue(GetDefaultInputName(model), out X); // default input
        var O = X;

        foreach (var l in model.layers)
        {
            if (l.inputs.Length > 0 && shapesByName.ContainsKey(l.inputs[0]))
                X = shapesByName[l.inputs[0]];
            else
                X = O; // previous output is used, if-and-only-if layer has no explicit inputs

            if (l.type == Layer.Type.Dense)
            {
                Assert.IsNotNull(l.datasets);
                var W = l.datasets[0].shape;
                O = new TensorShape(X.flatHeight, W.flatWidth);
            }
            else if (
                l.type == Layer.Type.Conv2D ||
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
                // pool size is treated as upsample coefficient here
                Assert.IsNotNull(l.pool);
                Assert.AreEqual(l.pool.Length, 2);
                O = new TensorShape(X.batch, X.height * l.pool[1], X.width * l.pool[0], X.channels);
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
                O = new TensorShape(X.batch, 1, features, depth);
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
                foreach (var i in l.inputs)
                {
                    if (shapesByName.ContainsKey(i))
                        list.Add(shapesByName[i]);
                }

                O = TensorExtensions.Max(list.ToArray());
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
                l.type == Layer.Type.ReduceSumSquare)
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
                // pool size is treated as reshape coefficient, if not empty
                // otherwise shape of the 2nd input tensor is used
                var size = l.pool;

                Assert.IsNotNull(size);
                if (size.Length == 0 && l.inputs.Length > 1)
                    size = shapesByName[l.inputs[1]].ToArray();

                Assert.AreEqual(size.Length, 4);
                // pool size is treated as reshape coefficient here
                O = X.Reshape(size);
            }
            else if (
                l.type == Layer.Type.Transpose)
            {
                O = new TensorShape(X.flatWidth, X.flatHeight);
            }
            else if (
                l.type == Layer.Type.Squeeze ||
                l.type == Layer.Type.Unsqueeze)
            {
                throw new NotImplementedException();
            }
            else if (
                l.type == Layer.Type.Concat)
            {
                // gather shapes by names
                var list = new List<TensorShape>(l.inputs.Length);
                foreach (var i in l.inputs)
                {
                    if (shapesByName.ContainsKey(i))
                        list.Add(shapesByName[i]);
                }

                O = TensorExtensions.Concat(list.ToArray(), l.axis);
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
                Assert.AreEqual(l.pool.Length, 4);
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
                l.activation == Layer.Activation.PRelu)
            {
                // works in place, keeps the same shape size
                O = X;
            }
            else
            {
                throw new NotImplementedException();
            }

            shapes.Add(O);
            shapesByName.Add(l.name, O);
        }

        Profiler.EndSample();
        return shapes.ToArray();
    }

    static public bool TryGetOutputTensorShape(Model model, IDictionary<string, TensorShape> inputShapes, string output, out TensorShape shape)
    {
        IDictionary<string, TensorShape> shapesByName;
        ListTemporaryTensorShapes(model, inputShapes, out shapesByName);
        return shapesByName.TryGetValue(output, out shape);
    }

    static public bool TryGetOutputTensorShape(Model model, string output, out TensorShape shape)
    {
        var inputShapes = new Dictionary<string, TensorShape>();
        foreach (var i in model.inputs)
            inputShapes.Add(i.name, new TensorShape(i.shape));
        return TryGetOutputTensorShape(model, inputShapes, output, out shape);
    }

    static public HashSet<Layer> FindLayersThatRequireStorage(Model model)
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

    /*static public HashSet<Layer> FindUpstreamLayers(Model model, string[] outputs)
    {
        var layersByName = new Dictionary<string, Layer>();
        foreach (var l in model.layers)
            layersByName.Add(l.name, l);

        var connected = new HashSet<Layer>();
        Func<string[], HashSet<Layer>(), HashSet<Layer>()> visitor = (layerNames, visitNext) =>
        {
            foreach (var i in layerNames)
                if (layersByName.ContainsKey(i))
                {
                    visitNext.Add(layersByName[i]);
                    connected.Add(layersByName[i]);
                }
            return visitNext;
        };

        var layersToVisit = visitor(outputs, new HashSet<Layer>());
        while (layersToVisit.Count > 0)
        {
            var visitNext = new HashSet<Layer>();
            foreach (var l in layersToVisit)
                visitor(l.inputs, visitNext);
            layersToVisit = visitNext;
        }
        return connected;
    }*/

    static public HashSet<Layer> FindUpstreamLayers(Model model, string[] outputs)
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

    static public TensorShape FindLargestNecessaryTensorShape(Model model, IDictionary<string, TensorShape> inputShapes)
    {
        Profiler.BeginSample ("Barracuda.FindLargestNecessaryTensorShape");

        var shapes = ListTemporaryTensorShapes(model, inputShapes);

        var maxTensorShape = new TensorShape(1,1,1,1);
        foreach (var X in shapes)
            if (X.length > maxTensorShape.length)
                maxTensorShape = X;

        Profiler.EndSample ();

        return maxTensorShape;
    }

    static public TensorShape FindLargestArgumentTensorShape(Model model)
    {
        TensorShape maxTensorShape = new TensorShape(1,1,1,1);
        foreach (var layer in model.layers)
            foreach (var arg in layer.datasets)
                if (arg.shape.length > maxTensorShape.length)
                    maxTensorShape = arg.shape;

        return maxTensorShape;
    }

    static public string[] FindBrokenLinks(Model model)
    {
        var globalInputsByName = model.inputs.ToDictionary(i => i.name, i => true);
        var layersByName = model.layers.ToDictionary(i => i.name, i => i);
        var brokenLinks = new HashSet<string>();

        foreach (var layer in model.layers)
            foreach (var i in layer.inputs)
                if (!layersByName.ContainsKey(i) && !globalInputsByName.ContainsKey(i))
                    brokenLinks.Add(i);
        return brokenLinks.ToArray();
    }

    static public string[] FindUnconnectedInputs(Model model)
    {
        var unconnected = model.inputs.ToDictionary(i => i.name, i => true);
        foreach (var layer in model.layers)
            foreach (var i in layer.inputs)
                unconnected.Remove(i);
        return unconnected.Keys.ToArray();
    }

    static public string[] FindUnconnectedOutputs(Model model, List<string> outputs)
    {
        var unconnected = outputs.ToDictionary(i => i, i => true);
        foreach (var layer in model.layers)
            unconnected.Remove(layer.name);
        return unconnected.Keys.ToArray();
    }

    static public string[] FindUnconnectedOutputs(Model model)
    {
        return FindUnconnectedOutputs(model, model.outputs);
    }
}


} // namespace Barracuda
