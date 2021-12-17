// #define DEBUG_TIMING
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Profiling;

[assembly: InternalsVisibleTo("Unity.Barracuda.Tests")]

namespace Unity.Barracuda {

/// <summary>
/// Barracuda `Model` loader
/// </summary>
public static class ModelLoader
{
    /// <summary>
    /// Return an object oriented representation (aka: `Model`) of a neural network from a binary representation of type `NNModel`.
    /// By default details are not logged to the console, set `verbose` to true to see loading details.
    /// </summary>
    /// <param name="model">model</param>
    /// <param name="verbose">verbose</param>
    /// <param name="skipWeights">skip loading weights (fast loading, metadata only)</param>
    /// <returns>loaded Model</returns>
    public static Model Load(NNModel model, bool verbose = false, bool skipWeights = false)
    {
        return Load(model.modelData.Value, verbose, skipWeights);
    }

    /// <summary>
    /// Return an object oriented representation (aka: `Model`) of a neural network from a binary representation of type `NNModel`.
    /// By default details are not logged to the console, set `verbose` to true to see loading details.
    /// </summary>
    /// <param name="nnModel">binary representation of model</param>
    /// <param name="model">object-oriented representation of model (must initialize before calling method)</param>
    /// <param name="verbose">verbose</param>
    /// <param name="skipWeights">skip loading weights (fast loading, metadata only)</param>
    /// <param name="maxTimePerYield">the maximum amount of time to spend between in computation before yielding</param>
    /// <returns>IEnumerator (use with StartCoroutine)</returns>
    public static IEnumerator LoadAsync(NNModel nnModel, Model model, bool verbose = false, bool skipWeights = false, float maxTimePerYield = 0.01f)
    {
        Assert.IsNotNull(model);
        var enumerator = LoadAsync(Open(nnModel.modelData.Value), model, verbose, true, skipWeights, maxTimePerYield);

        while (enumerator.MoveNext())
        {
            model = (Model)enumerator.Current;
            if (model != null)
                yield return null;
        }
    }

    /// <summary>
    /// Return an object oriented representation (aka: `Model`) of a neural network from a `.bc` file from the the streaming asset folder.
    /// By default details are not logged to the console, set `verbose` to true to see loading details.
    /// </summary>
    /// <param name="filename">file name</param>
    /// <param name="verbose">verbose</param>
    /// <param name="skipWeights">skip loading weights (fast loading, metadata only)</param>
    /// <returns>loaded Model</returns>
    public static Model LoadFromStreamingAssets(string filename, bool verbose = false, bool skipWeights = false)
    {
        return Load(Path.Combine(Application.streamingAssetsPath, filename), verbose, skipWeights);
    }

    /// <summary>
    /// Return an object oriented representation (aka: `Model`) of a neural network from a `.bc` file from the the streaming asset folder.
    /// By default details are not logged to the console, set `verbose` to true to see loading details.
    /// </summary>
    /// <param name="filename">file name</param>
    /// <param name="model">object-oriented representation of model (must initialize before calling method)</param>
    /// <param name="verbose">verbose</param>
    /// <param name="skipWeights">skip loading weights (fast loading, metadata only)</param>
    /// <param name="maxTimePerYield">the maximum amount of time to spend between in computation before yielding</param>
    /// <returns>IEnumerator (use with StartCoroutine)</returns>
    public static IEnumerator LoadAsyncFromStreamingAssets(string filename, Model model, bool verbose = false, bool skipWeights = false, float maxTimePerYield = 0.01f)
    {
        Assert.IsNotNull(model);
        var enumerator = LoadAsync(Open(Path.Combine(Application.streamingAssetsPath, filename)), model, verbose, true, skipWeights, maxTimePerYield);

        do
        {
            model = (Model)enumerator.Current;
            if (model != null)
                yield return null;
        } while (enumerator.MoveNext());
    }

    /// <summary>
    /// Return an object oriented representation (aka: `Model`) of a neural network from a `.bc` file.
    /// By default details are not logged to the console, set `verbose` to true to see loading details.
    /// </summary>
    /// <param name="filepath">file name</param>
    /// <param name="verbose">verbose</param>
    /// <param name="skipWeights">skip loading weights (fast loading, metadata only)</param>
    /// <returns>loaded Model</returns>
    public static Model Load(string filepath, bool verbose = false, bool skipWeights = false)
    {
        return Load(Open(filepath), verbose, true, skipWeights);
    }

    /// <summary>
    /// Return an object oriented representation (aka: `Model`) of a neural network from a `.bc` file.
    /// By default details are not logged to the console, set `verbose` to true to see loading details.
    /// </summary>
    /// <param name="filepath">file name</param>
    /// <param name="model">object-oriented representation of model (must initialize before calling method)</param>
    /// <param name="verbose">verbose</param>
    /// <param name="skipWeights">skip loading weights (fast loading, metadata only)</param>
    /// <param name="maxTimePerYield">the maximum amount of time to spend between in computation before yielding</param>
    /// <returns>IEnumerator (use with StartCoroutine)</returns>
    public static IEnumerator LoadAsync(string filepath, Model model, bool verbose = false, bool skipWeights = false, float maxTimePerYield = 0.01f)
    {
        Assert.IsNotNull(model);
        var enumerator = LoadAsync(Open(filepath), model, verbose, true, skipWeights, maxTimePerYield);

        while (enumerator.MoveNext())
        {
            model = (Model)enumerator.Current;
            if (model != null)
                yield return null;
        }
    }


    /// <summary>
    /// Return an object oriented representation (aka: `Model`) of a neural network from a byte[] array.
    /// By default details are not logged to the console, set `verbose` to true to see loading details.
    /// </summary>
    /// <param name="stream">binary representation of model as a byte array</param>
    /// <param name="verbose">verbose</param>
    /// <param name="skipWeights">skip loading weights (fast loading, metadata only)</param>
    /// <returns>loaded Model</returns>
    public static Model Load(byte[] stream, bool verbose = false, bool skipWeights = false)
    {
        return Load(Open(stream), verbose, true, skipWeights);
    }

    /// <summary>
    /// Return an object oriented representation (aka: `Model`) of a neural network from a byte[] array.
    /// By default details are not logged to the console, set `verbose` to true to see loading details.
    /// </summary>
    /// <param name="stream">binary representation of model as a byte array</param>
    /// <param name="model">object-oriented representation of model (must initialize before calling method)</param>
    /// <param name="verbose">verbose</param>
    /// <param name="skipWeights">skip loading weights (fast loading, metadata only)</param>
    /// <param name="maxTimePerYield">the maximum amount of time to spend between in computation before yielding</param>
    /// <returns>IEnumerator (use with StartCoroutine)</returns>
    public static IEnumerator LoadAsync(byte[] stream, Model model, bool verbose = false, bool skipWeights = false, float maxTimePerYield = 0.01f)
    {
        Assert.IsNotNull(model);
        var enumerator = LoadAsync(Open(stream), model, verbose, true, skipWeights, maxTimePerYield);

        while (enumerator.MoveNext())
        {
            model = (Model)enumerator.Current;
            if (model != null)
                yield return null;
        }
    }

    #region Private and internal

    internal static Model Load(byte[] stream, bool verbose = true, bool applyPatching = true, bool skipWeights = false)
    {
        return Load(Open(stream), verbose, applyPatching, skipWeights);
    }

    private static int ConvertLayerAxisFor8DShapeSupportIfNeeded(int axis, long version, Layer.Type layerType)
    {
        if (version > Model.LastVersionWithout8DSupport)
            return axis;

        //Prior to version 17, 8D tensors were not supported thus axis was expressed in NCHW format for Gather, Concat and Reduce layers.
        if (layerType == Layer.Type.ReduceL2 ||
            layerType == Layer.Type.ReduceLogSum ||
            layerType == Layer.Type.ReduceLogSumExp ||
            layerType == Layer.Type.ReduceMax ||
            layerType == Layer.Type.ReduceMean ||
            layerType == Layer.Type.ReduceMin ||
            layerType == Layer.Type.ReduceProd ||
            layerType == Layer.Type.ReduceSum ||
            layerType == Layer.Type.ReduceSumSquare ||
            layerType == Layer.Type.Gather ||
            layerType == Layer.Type.Concat)
            axis = TensorExtensions.Convert4DTo8DAxis(axis);

        return axis;
    }

    static Model Load(BinaryReader fileReader, bool verbose = true, bool applyPatching = true, bool skipWeights = false)
    {
        Model model = null;
        var enumerator = LoadAsync(fileReader, null, verbose, applyPatching, skipWeights);

        while (enumerator.MoveNext())
        {
            model = (Model)enumerator.Current;
            if (model != null)
                break;
        }

        return model;
    }

    static IEnumerator LoadAsync(BinaryReader fileReader, Model model, bool verbose = true, bool applyPatching = true, bool skipWeights = false, float maxTimePerYield = 0f)
    {
        using (BinaryReader file = fileReader)
        {
            Profiler.BeginSample("Barracuda.LoadLayers");
            float timeStart = Time.realtimeSinceStartup;

            if (model == null)
                model = new Model();
            List<Layer> layers = new List<Layer>();

            long version = file.ReadInt64() % 0xff; // magic
            if (version != Model.Version && version != Model.LastVersionWithout8DSupport && version != Model.LastVersionWithoutWeightsAlignmentSupport)
                throw new NotSupportedException($"Format version not supported: {version}");

            var count = file.ReadInt32();
            model.inputs = new List<Model.Input>(count);
            for (var i = 0; i < count; ++i)
            {
                model.inputs.Add(new Model.Input {name = ReadString(file), shape = ReadInt32Array(file)});

                if (maxTimePerYield > 0 && Time.realtimeSinceStartup - timeStart > maxTimePerYield)
                {
#if DEBUG_TIMING
                    UnityEngine.Debug.Log(Time.realtimeSinceStartup - timeStart);
#endif
                    yield return null;
                    timeStart = Time.realtimeSinceStartup;
                }
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

                if (maxTimePerYield > 0 && Time.realtimeSinceStartup - timeStart > maxTimePerYield)
                {
#if DEBUG_TIMING
                    UnityEngine.Debug.Log(Time.realtimeSinceStartup - timeStart);
#endif
                    yield return null;
                    timeStart = Time.realtimeSinceStartup;
                }
            }

            int numberOfLayers = file.ReadInt32();
            for (var l = 0; l < numberOfLayers; ++l)
            {
                var name            = ReadString(file);
                var layerType       = (Layer.Type)file.ReadInt32();
                var activation      = (Layer.Activation)file.ReadInt32();
                Layer layer         = new Layer(name, layerType, activation);
                                      ReadInt32Array(file); // dummy
                                      ReadInt32Array(file); // dummy
                layer.pad           = ReadInt32Array(file);
                layer.stride        = ReadInt32Array(file);
                layer.pool          = ReadInt32Array(file);
                layer.axis          = ConvertLayerAxisFor8DShapeSupportIfNeeded(file.ReadInt32(), version, layerType);
                layer.alpha         = file.ReadSingle();
                layer.beta          = file.ReadSingle();
                                      ReadInt32Array(file); // dummy

                layer.inputs        = ReadStringArray(file);

                if (maxTimePerYield > 0 && Time.realtimeSinceStartup - timeStart > maxTimePerYield)
                {
#if DEBUG_TIMING
                    UnityEngine.Debug.Log(Time.realtimeSinceStartup - timeStart);
#endif
                    yield return null;
                    timeStart = Time.realtimeSinceStartup;
                }

                layer.datasets      = new Layer.DataSet[file.ReadInt32()];
                for (var i = 0; i < layer.datasets.Length; ++i)
                {
                    if (maxTimePerYield > 0 && Time.realtimeSinceStartup - timeStart > maxTimePerYield)
                    {
#if DEBUG_TIMING
                        UnityEngine.Debug.Log(Time.realtimeSinceStartup - timeStart);
#endif
                        yield return null;
                        timeStart = Time.realtimeSinceStartup;
                    }

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

                if (applyPatching)
                    PatchLayer(layers, layer);

                if (maxTimePerYield > 0 && Time.realtimeSinceStartup - timeStart > maxTimePerYield)
                {
#if DEBUG_TIMING
                    UnityEngine.Debug.Log(Time.realtimeSinceStartup - timeStart + ": " + l);
#endif
                    yield return null;
                    timeStart = Time.realtimeSinceStartup;
                }
            }
            model.layers = layers;

            Int64 numWeightsToRead = 0;
            for (var l = 0; l < model.layers.Count; ++l)
            {
                for (var d = 0; d < model.layers[l].datasets.Length; ++d)
                {
                    numWeightsToRead += model.layers[l].datasets[d].length;

                    if (maxTimePerYield > 0 && Time.realtimeSinceStartup - timeStart > maxTimePerYield)
                    {
#if DEBUG_TIMING
                        UnityEngine.Debug.Log(Time.realtimeSinceStartup - timeStart);
#endif
                        yield return null;
                        timeStart = Time.realtimeSinceStartup;
                    }
                }
            }

            Profiler.EndSample();

            DataType weightsDataType = DataType.Float;
            if (version >= 20)
            {
                //Version 20 introduce weights type but full model need to be in the same type. Per layer no supported yet.
                weightsDataType = (DataType)file.ReadInt32();
            }

            if (version >= 19)
            {
                //Padding so weights are aligned on Model.WeightsAlignment bytes
                long streamCurrentPosition = file.BaseStream.Position;
                long paddingForAlignment = Model.WeightsAlignment - (streamCurrentPosition % Model.WeightsAlignment);
                file.BaseStream.Seek(paddingForAlignment, SeekOrigin.Current);
            }

            if (skipWeights)
                SkipLargeByteArray(file, numWeightsToRead * BarracudaArray.DataItemSize(weightsDataType));
            else
            {
                if (maxTimePerYield > 0 && Time.realtimeSinceStartup - timeStart > maxTimePerYield)
                {
#if DEBUG_TIMING
                    UnityEngine.Debug.Log(Time.realtimeSinceStartup - timeStart);
#endif
                    yield return null;
                    timeStart = Time.realtimeSinceStartup;
                }

                var sharedWeightsArray = ReadLargeWeightArray(file, numWeightsToRead, weightsDataType);

                Assert.AreEqual(weightsDataType, sharedWeightsArray.Type);
                for (var l = 0; l < model.layers.Count; ++l)
                {
                    model.layers[l].weights = sharedWeightsArray;

                    if (maxTimePerYield > 0 && Time.realtimeSinceStartup - timeStart > maxTimePerYield)
                    {
#if DEBUG_TIMING
                        UnityEngine.Debug.Log(Time.realtimeSinceStartup - timeStart);
#endif
                        yield return null;
                        timeStart = Time.realtimeSinceStartup;
                    }
                }
            }

            // Importer Reporting
            try
            {
                model.IrSource = ReadString(file);
                model.IrVersion = ReadString(file);
                model.ProducerName = ReadString(file);
                int numWarnings = file.ReadInt32();
                for (var i = 0; i < numWarnings; ++i)
                {
                    model.Warnings.Add(new Model.ImporterWarning(ReadString(file), ReadString(file)));
                }

                if (version >= 18)
                {
                    int numMetadataProps = file.ReadInt32();
                    for (var i = 0; i < numMetadataProps; ++i)
                    {
                        model.Metadata.Add(ReadString(file), ReadString(file));
                    }
                }
            }
            catch (EndOfStreamException)
            {
                //Do nothing Importer Reporting data might not be present for backward compatibility reasons
            }

            yield return model;
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
                Layer layerC        = new Layer(t.name, Layer.Type.Load); // load using tensor name
                layerC.inputs       = layer.inputs;
                layerC.datasets     = new[] { t };

                layers.Add(layerC);
            }

            // patch original layer
            layer.name              = layer.name + "_nop";
            layer.type              = Layer.Type.Nop;
            layer.datasets          = new Layer.DataSet[] {};
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

    private static void SkipLargeByteArray(BinaryReader file, Int64 count)
    {
        file.BaseStream.Seek(count, SeekOrigin.Current);
    }

    private static BarracudaArray ReadLargeWeightArray(BinaryReader file, Int64 count, DataType dataType)
    {
        int bytesToRead;
        Int64 bytesToReadInt64 = count * BarracudaArray.DataItemSize(dataType);
        try
        {
            bytesToRead = Convert.ToInt32(bytesToReadInt64); // throws OverflowException
        }
        catch (OverflowException)
        {
            throw new OverflowException($"Files larger than 2GB currently are not supported. Attempt to read {bytesToReadInt64} bytes.");
        }

        //1-Try to remap byte[] stream to avoid allocation
        Profiler.BeginSample("Barracuda.RemapWeights");
        BarracudaArray remappedWeights = null;
        try
        {
            Stream stream = file.BaseStream;
            var memoryStream = stream as MemoryStream;
            var sourceBuffer = memoryStream?.GetBuffer();
            int currentPosition = (int)memoryStream?.Position;
            remappedWeights = new BarracudaArrayFromManagedArray(sourceBuffer, currentPosition, dataType, (int) count);
        }
        #if UNITY_EDITOR
        catch (InvalidOperationException e)
        {
            UnityEngine.Debug.Log("ModelLoader: Can't remap memory stream to underlying data type, allocation and copy will occurs. Exception: " + e);
        }
        #else
        catch (InvalidOperationException) {}
        #endif
        if (remappedWeights != null)
        {
            //We remapped memory. Need to advance stream position to be consistent with read behavior.
            file.BaseStream.Position += bytesToRead;
            Profiler.EndSample();
            return remappedWeights;
        }
        Profiler.EndSample();

        //2-Can't remap will copy from managed memory to native
        Profiler.BeginSample("Barracuda.AllocWeights");
        BarracudaArray loadedWeights = new BarracudaArray((int)count, dataType);
        Profiler.EndSample();

        Profiler.BeginSample("Barracuda.LoadWeights");
        try
        {
            var readBuffer = new byte[4096]; // 4Kb is close to optimal read size.
                                             // See for measurements: https://www.jacksondunstan.com/articles/3568
                                             // Read size vs relative read-time:
                                             // 64b: x10, 128b: x6, 256b: x4, 1Kb: x3, 4Kb: x3
            int writeOffset = 0;
            while (writeOffset < bytesToRead)
            {
                var bytesLeftToRead = bytesToRead - writeOffset;
                var readSizeInBytes = Math.Min(readBuffer.Length, bytesLeftToRead);

                Assert.IsTrue(readSizeInBytes > 0);
                Assert.IsTrue(readSizeInBytes <= readBuffer.Length);
                readSizeInBytes = file.BaseStream.Read(readBuffer, offset:0, count:readSizeInBytes);
                if (readSizeInBytes == 0)
                    throw new IOException($"Unexpected EOF reached. Read {writeOffset / sizeof(float)} out of expected {count} floats before reaching end of file.");

                BarracudaArray.BlockCopy(
                    sourceArray:readBuffer, sourceByteOffset:0,
                    destinationArray:loadedWeights, destinationByteOffset:writeOffset,
                    lengthInBytes:readSizeInBytes);
                writeOffset += readSizeInBytes;
            }
            Assert.AreEqual(writeOffset, bytesToRead);
        }
        finally
        {
            Profiler.EndSample();
        }

        return loadedWeights;
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
        return new BinaryReader(new MemoryStream(bytes, 0, bytes.Length, false, true));
    }
    #endregion
}


} // namespace Unity.Barracuda
