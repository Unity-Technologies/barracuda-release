using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Profiling;

[assembly: InternalsVisibleTo("Unity.Barracuda.Tests")]

namespace Unity.Barracuda {

public static class ModelLoader
{
    /// <summary>
    /// Return an object oriented representation (aka: `Model`) of a neural network from a binary representation of type `NNModel`.
    /// By default details are not logged to the console, set `verbose` to true to see loading details.
    /// </summary>
    public static Model Load(NNModel model, bool verbose = false, bool skipWeights = false)
    {
        return Load(model.modelData.Value, verbose, skipWeights);
    }

    /// <summary>
    /// Return an object oriented representation (aka: `Model`) of a neural network from a `.bc` file from the the streaming asset folder.
    /// By default details are not logged to the console, set `verbose` to true to see loading details.
    /// </summary>
    public static Model LoadFromStreamingAssets(string filename, bool verbose = false, bool skipWeights = false)
    {
        return Load(Path.Combine(Application.streamingAssetsPath, filename), verbose, skipWeights);
    }

    /// <summary>
    /// Return an object oriented representation (aka: `Model`) of a neural network from a `.bc` file.
    /// By default details are not logged to the console, set `verbose` to true to see loading details.
    /// </summary>
    public static Model Load(string filepath, bool verbose = false, bool skipWeights = false)
    {
        return Load(Open(filepath), verbose, true, skipWeights);
    }

    /// <summary>
    /// Return an object oriented representation (aka: `Model`) of a neural network from a byte[] array.
    /// By default details are not logged to the console, set `verbose` to true to see loading details.
    /// </summary>
    public static Model Load(byte[] stream, bool verbose = false, bool skipWeights = false)
    {
        return Load(Open(stream), verbose, true, skipWeights);
    }

    #region Private and internal

    internal static Model Load(byte[] stream, bool verbose = true, bool applyPatching = true, bool skipWeights = false)
    {
        return Load(Open(stream), verbose, applyPatching, skipWeights);
    }

    private static Model Load(BinaryReader fileReader, bool verbose = true, bool applyPatching = true, bool skipWeights = false)
    {
        using (BinaryReader file = fileReader)
        {
            Profiler.BeginSample("Barracuda.LoadLayers");

            Model model = new Model();
            List<Layer> layers = new List<Layer>();

            long version = file.ReadInt64() % 0xff; // magic
            if (version != Model.Version)
                throw new NotSupportedException($"Format version not supported: {version}");

            var count = file.ReadInt32();
            model.inputs = new List<Model.Input>(count);
            for (var i = 0; i < count; ++i)
            {
                model.inputs.Add(new Model.Input {name = ReadString(file), shape = ReadInt32Array(file)});
            }

            model.outputs = ReadStringArray(file).ToList();

            count = file.ReadInt32();
            model.memories  = new List<Model.Memory>(count);
            for (var m = 0; m < count; ++m)
            {
                model.memories.Add(new Model.Memory
                {
                    shape = new TensorShape(ReadInt32Array(file)),
                    input = ReadString(file),
                    output = ReadString(file)
                });
            }

            int numberOfLayers = file.ReadInt32();
            for (var l = 0; l < numberOfLayers; ++l)
            {
                var name            = ReadString(file);
                var type            = (Layer.Type)file.ReadInt32();
                var activation      = (Layer.Activation)file.ReadInt32();
                Layer layer         = new Layer(name, type, activation);
                                      ReadInt32Array(file); // dummy
                                      ReadInt32Array(file); // dummy
                layer.pad           = ReadInt32Array(file);
                layer.stride        = ReadInt32Array(file);
                layer.pool          = ReadInt32Array(file);
                layer.axis          = file.ReadInt32();
                layer.alpha         = file.ReadSingle();
                layer.beta          = file.ReadSingle();
                                      ReadInt32Array(file); // dummy

                layer.inputs        = ReadStringArray(file);

                layer.datasets      = new Layer.DataSet[file.ReadInt32()];
                for (var i = 0; i < layer.datasets.Length; ++i)
                {
                    layer.datasets[i].name            = ReadString(file);
                    layer.datasets[i].shape           = new TensorShape(ReadInt32Array(file));
                    layer.datasets[i].offset          = file.ReadInt64();
                    layer.datasets[i].itemSizeInBytes = file.ReadInt32();
                    layer.datasets[i].length          = file.ReadInt32();
                }

                layers.Add(layer);

                if (verbose)
                    D.Log(
                        $"layer {l}, {layer.name} type: {layer.type} " +
                                 $"{((layer.activation != Layer.Activation.None) ? $"activation {layer.activation} " : "")}" +
                                 $"tensors: {layer.datasets.Length} inputs: {String.Join(",", layer.inputs)}");

                if (verbose)
                    foreach (var t in layer.datasets)
                        D.Log($"        Tensor: {t.shape} offset: {t.offset} len: {t.length}");

                if(applyPatching)
                    PatchLayer(layers, layer);
            }
            model.layers = layers;

            Int64 dataLength = 0;
            for (var l = 0; l < model.layers.Count; ++l)
                for (var d = 0; d < model.layers[l].datasets.Length; ++d)
                    dataLength += model.layers[l].datasets[d].length;

            Profiler.EndSample();

            if (!skipWeights)
            {
                Profiler.BeginSample("Barracuda.AllocModel");
                var sharedWeights = new float[dataLength];
                Profiler.EndSample();

                Profiler.BeginSample("Barracuda.LoadModel");
                byte[] bytes = file.ReadBytes(Convert.ToInt32(dataLength * sizeof(float))); // @TODO: support larger than MaxInt32 data blocks
                Buffer.BlockCopy(bytes, 0, sharedWeights, 0, bytes.Length);
                Profiler.EndSample();

                for (var l = 0; l < model.layers.Count; ++l)
                    model.layers[l].weights = sharedWeights;
                }
            else
            {
                file.BaseStream.Seek(Convert.ToInt32(dataLength * sizeof(float)), SeekOrigin.Current);
            }

            //Importer Reporting
            try
            {
                model.IrSource = ReadString(file);
                model.IrVersion = ReadString(file);
                model.ProducerName = ReadString(file);
                var numWarnings = file.ReadInt32();
                for (var i = 0; i < numWarnings; ++i)
                {
                    model.Warnings.Add(new Model.ImporterWarning(ReadString(file), ReadString(file)));
                }
            }
            catch (EndOfStreamException)
            {
                //Do nothing Importer Reporting data might not be present for backward compatibility reasons
            }

            return model;
        }
    }

    private static void PatchLayer(List<Layer> layers, Layer layer)
    {
        // Split Load so that each constant tensor gets its own layer
        // for the sake of simplicity of the execution code
        if (layer.type == Layer.Type.Load &&
            layer.datasets.Length > 1)
        {
            foreach (var t in layer.datasets)
            {
                Layer layerC        = new Layer(t.name, Layer.Type.Load);// load using tensor name
                layerC.inputs       = layer.inputs;
                layerC.datasets     = new[] { t };

                layers.Add(layerC);
            }

            // patch original layer
            layer.name              = layer.name + "_nop";
            layer.type              = Layer.Type.Nop;
        }

        // Split activation part into separate layer when activation fusing is not supported.
        // NOTE: Keras specific. Only Keras exporter packs both Dense/Conv and Activation into the same layer.
        // @TODO: move layer split directly into Keras exporter
        if (layer.type          != Layer.Type.Activation &&
            layer.activation    != Layer.Activation.None &&
            (!ModelOptimizer.IsLayerSupportingActivationFusing(layer.type) || !ModelOptimizer.IsActivationFusable(layer.activation)))
        {
            var affineOutput    = layer.name + "_tmp";

            Layer layerA        = new Layer(layer.name, layer.activation);// take the original layer name
            layerA.inputs       = new[] { affineOutput };

            // patch original layer
            layer.name           = affineOutput;
            layer.activation     = Layer.Activation.None;
            Assert.AreEqual(layers[layers.Count-1].name, layer.name);
            Assert.AreEqual(layers[layers.Count-1].activation, layer.activation);

            layers.Add(layerA);
        }

        // @TODO: Enable Dropout
        // @TEMP: disabled runtime Dropout noise to get more predictable results for auto testing
        if (layer.type == Layer.Type.Dropout)
        {
            layer.type          = Layer.Type.Activation;
            layer.activation    = Layer.Activation.None;
        }
    }

    private static Int32[] ReadInt32Array(BinaryReader file)
    {
        var arr = new Int32[file.ReadInt32()];
        byte[] bytes = file.ReadBytes(Convert.ToInt32(arr.Length * sizeof(Int32)));
        Buffer.BlockCopy(bytes, 0, arr, 0, bytes.Length);
        return arr;
    }

    private static string ReadString(BinaryReader file)
    {
        var chars = file.ReadChars(file.ReadInt32());
        return new string(chars);
    }

    private static string[] ReadStringArray(BinaryReader file)
    {
        var arr = new string[file.ReadInt32()];
        for (var i = 0; i < arr.Length; ++i)
            arr[i] = ReadString(file);
        return arr;
    }

    private static BinaryReader Open(string filename)
    {
        return new BinaryReader(new FileStream(filename, FileMode.Open, FileAccess.Read));
    }

    private static BinaryReader Open(byte[] bytes)
    {
        return new BinaryReader(new MemoryStream(bytes, false));
    }
    #endregion
}


} // namespace Unity.Barracuda
