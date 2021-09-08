using System;
using System.Collections.Generic;
using System.IO;

using UnityEngine;
using UnityEngine.Assertions;
using System.IO.Compression;


namespace Unity.Barracuda {

/// <summary>
/// Test set loading utility
/// </summary>
public class TestSet
{
    private RawTestSet rawTestSet;
    private JSONTestSet jsonTestSet;

    /// <summary>
    /// Create with raw test set
    /// </summary>
    /// <param name="rawTestSet">raw test set</param>
    public TestSet(RawTestSet rawTestSet)
    {
        this.rawTestSet = rawTestSet;
    }

    /// <summary>
    /// Create with JSON test set
    /// </summary>
    /// <param name="jsonTestSet">JSON test set</param>
    public TestSet(JSONTestSet jsonTestSet)
    {
        this.jsonTestSet = jsonTestSet;
    }

    /// <summary>
    /// Create `TestSet`
    /// </summary>
    public TestSet()
    {
    }

    /// <summary>
    /// Check if test set supports named tensors
    /// </summary>
    /// <returns>`true` if named tensors are supported</returns>
    public bool SupportsNames()
    {
        if (rawTestSet != null)
            return false;

        return true;
    }

    /// <summary>
    /// Get output tensor count
    /// </summary>
    /// <returns></returns>
    public int GetOutputCount()
    {
        if (rawTestSet != null)
            return 1;

        return jsonTestSet.outputs.Length;
    }

    /// <summary>
    /// Get output tensor data
    /// </summary>
    /// <param name="idx">tensor index</param>
    /// <returns>tensor data</returns>
    public float[] GetOutputData(int idx = 0)
    {
        if (rawTestSet != null)
            return rawTestSet.labels;

        return jsonTestSet.outputs[idx].data;
    }

    /// <summary>
    /// Get output tensor name
    /// </summary>
    /// <param name="idx">tensor index</param>
    /// <returns>tensor name</returns>
    public string GetOutputName(int idx = 0)
    {
        if (rawTestSet != null)
            return null;

        string name = jsonTestSet.outputs[idx].name;
        return name.EndsWith(":0") ? name.Remove(name.Length - 2) : name;
    }

    /// <summary>
    /// Get input tensor count
    /// </summary>
    /// <returns></returns>
    public int GetInputCount()
    {
        if (rawTestSet != null)
            return 1;

        return jsonTestSet.inputs.Length;
    }

    /// <summary>
    /// Get input tensor name
    /// </summary>
    /// <param name="idx">input tensor index</param>
    /// <returns>tensor name</returns>
    public string GetInputName(int idx = 0)
    {
        if (rawTestSet != null)
            return "";

        string name = jsonTestSet.inputs[idx].name;
        return name.EndsWith(":0") ? name.Remove(name.Length - 2) : name;
    }

    /// <summary>
    /// Get input tensor data
    /// </summary>
    /// <param name="idx">input tensor index</param>
    /// <returns>tensor data</returns>
    public float[] GetInputData(int idx = 0)
    {
        if (rawTestSet != null)
            return rawTestSet.input;

        return jsonTestSet.inputs[idx].data;
    }

    /// <summary>
    /// Get input shape
    /// </summary>
    /// <param name="idx">input tensor index</param>
    /// <returns>input shape</returns>
    public TensorShape GetInputShape(int idx = 0)
    {
        if (rawTestSet != null)
            return new TensorShape(1,rawTestSet.input.Length);

        return new TensorShape(jsonTestSet.inputs[idx].shape.sequenceLength,
                               jsonTestSet.inputs[idx].shape.numberOfDirections,
                               jsonTestSet.inputs[idx].shape.batch,
                               jsonTestSet.inputs[idx].shape.extraDimension,
                               jsonTestSet.inputs[idx].shape.depth,
                               jsonTestSet.inputs[idx].shape.height,
                               jsonTestSet.inputs[idx].shape.width,
                               jsonTestSet.inputs[idx].shape.channels);
    }

    /// <summary>
    /// Get output tensor shape
    /// </summary>
    /// <param name="idx">output tensor index</param>
    /// <returns>tensor shape</returns>
    public TensorShape GetOutputShape(int idx = 0)
    {
        if (rawTestSet != null)
            return new TensorShape(1,rawTestSet.labels.Length);

        return new TensorShape(jsonTestSet.outputs[idx].shape.sequenceLength,
            jsonTestSet.outputs[idx].shape.numberOfDirections,
            jsonTestSet.outputs[idx].shape.batch,
            jsonTestSet.outputs[idx].shape.extraDimension,
            jsonTestSet.outputs[idx].shape.depth,
            jsonTestSet.outputs[idx].shape.height,
            jsonTestSet.outputs[idx].shape.width,
            jsonTestSet.outputs[idx].shape.channels);
    }

    /// <summary>
    /// Get inputs as `Tensor` dictionary
    /// </summary>
    /// <param name="inputs">dictionary to store results</param>
    /// <param name="batchCount">max batch count</param>
    /// <param name="fromBatch">start from batch</param>
    /// <returns>dictionary with input tensors</returns>
    /// <exception cref="Exception">thrown if called on raw test set (only JSON test set is supported)</exception>
    public Dictionary<string, Tensor> GetInputsAsTensorDictionary(Dictionary<string, Tensor> inputs = null, int batchCount = -1, int fromBatch = 0)
    {
        if (rawTestSet != null)
            throw new Exception("GetInputsAsTensorDictionary is not supported for RAW test suites");

        if (inputs == null)
            inputs = new Dictionary<string, Tensor>();

        for (var i = 0; i < GetInputCount(); i++)
            inputs[GetInputName(i)] = GetInputAsTensor(i, batchCount, fromBatch);

        return inputs;
    }

    /// <summary>
    /// Get outputs as `Tensor` dictionary
    /// </summary>
    /// <param name="outputs">dictionary to store results</param>
    /// <param name="batchCount">max batch count</param>
    /// <param name="fromBatch">start from batch</param>
    /// <returns>dictionary with input tensors</returns>
    /// <exception cref="Exception">thrown if called on raw test set (only JSON test set is supported)</exception>
    public Dictionary<string, Tensor> GetOutputsAsTensorDictionary(Dictionary<string, Tensor> outputs = null, int batchCount = -1, int fromBatch = 0)
    {
        if (rawTestSet != null)
            throw new Exception("GetOutputsAsTensorDictionary is not supported for RAW test suites");

        if (outputs == null)
            outputs = new Dictionary<string, Tensor>();

        for (var i = 0; i < GetOutputCount(); i++)
            outputs[GetOutputName(i)] = GetOutputAsTensor(i, batchCount, fromBatch);

        return outputs;
    }

    /// <summary>
    /// Get input as `Tensor`
    /// </summary>
    /// <param name="idx">input index</param>
    /// <param name="batchCount">max batch count</param>
    /// <param name="fromBatch">start from batch</param>
    /// <returns>`Tensor`</returns>
    /// <exception cref="Exception">thrown if called on raw test set (only JSON test set is supported)</exception>
    public Tensor GetInputAsTensor(int idx = 0, int batchCount = -1, int fromBatch = 0)
    {
        if (rawTestSet != null)
            throw new Exception("GetInputAsTensor is not supported for RAW test suites");

        TensorShape shape = GetInputShape(idx);
        Assert.IsTrue(shape.sequenceLength==1 && shape.numberOfDirections==1);
        var array = GetInputData(idx);
        var maxBatchCount = array.Length / shape.flatWidth;

        fromBatch = Math.Min(fromBatch, maxBatchCount - 1);
        if (batchCount < 0)
            batchCount = maxBatchCount - fromBatch;

        // pad data with 0s, if test-set doesn't have enough batches
        var shapeArray = shape.ToArray();
        shapeArray[TensorShape.DataBatch] = batchCount;
        var tensorShape = new TensorShape(shapeArray);
        var managedBufferStartIndex = fromBatch * tensorShape.flatWidth;
        var count = Math.Min(batchCount, maxBatchCount - fromBatch) * tensorShape.flatWidth;
        float[] dataToUpload = new float[tensorShape.length];
        Array.Copy(array, managedBufferStartIndex, dataToUpload, 0, count);

        var data = new ArrayTensorData(tensorShape.length);
        data.Upload(dataToUpload, tensorShape, 0);

        var res = new Tensor(tensorShape, data);
        res.name = GetInputName(idx);
        res.name = res.name.EndsWith(":0") ? res.name.Remove(res.name.Length - 2) : res.name;

        return res;
    }

    /// <summary>
    /// Get output as `Tensor`
    /// </summary>
    /// <param name="idx">output index</param>
    /// <param name="batchCount">max batch count</param>
    /// <param name="fromBatch">start from batch</param>
    /// <returns>`Tensor`</returns>
    /// <exception cref="Exception">thrown if called on raw test set (only JSON test set is supported)</exception>
    public Tensor GetOutputAsTensor(int idx = 0, int batchCount = -1, int fromBatch = 0)
    {
        if (rawTestSet != null)
            throw new Exception("GetOutputAsTensor is not supported for RAW test suites");

        TensorShape shape = GetOutputShape(idx);
        Assert.IsTrue(shape.sequenceLength==1 && shape.numberOfDirections==1);
        var barracudaArray = new BarracudaArrayFromManagedArray(GetOutputData(idx));

        var maxBatchCount = barracudaArray.Length / shape.flatWidth;

        fromBatch = Math.Min(fromBatch, maxBatchCount - 1);
        if (batchCount < 0)
            batchCount = maxBatchCount - fromBatch;
        batchCount = Math.Min(batchCount, maxBatchCount - fromBatch);

        var shapeArray = shape.ToArray();
        shapeArray[TensorShape.DataBatch] = batchCount;
        var tensorShape = new TensorShape(shapeArray);

        var offset = fromBatch * tensorShape.flatWidth;
        var res = new Tensor(tensorShape, new SharedArrayTensorData(barracudaArray, tensorShape, offset));
        res.name = GetOutputName(idx);
        res.name = res.name.EndsWith(":0") ? res.name.Remove(res.name.Length - 2) : res.name;

        return res;
    }
}

/// <summary>
/// Raw test structure
/// </summary>
public class RawTestSet
{
    /// <summary>
    /// Input data
    /// </summary>
    public float[] input;

    /// <summary>
    /// Output data
    /// </summary>
    public float[] labels;
}

/// <summary>
/// JSON test structure
/// </summary>
[Serializable]
public class JSONTestSet
{
    /// <summary>
    /// Inputs
    /// </summary>
    public JSONTensor[] inputs;

    /// <summary>
    /// Outputs
    /// </summary>
    public JSONTensor[] outputs;
}

/// <summary>
/// JSON tensor shape
/// </summary>
[Serializable]
public class JSONTensorShape
{
    /// <summary>
    /// Sequence length
    /// </summary>
    public int sequenceLength;

    /// <summary>
    /// Number of directions
    /// </summary>
    public int numberOfDirections;

    /// <summary>
    /// Batch
    /// </summary>
    public int batch;

    /// <summary>
    /// Extra dimension
    /// </summary>
    public int extraDimension;

    /// <summary>
    /// Depth
    /// </summary>
    public int depth;

    /// <summary>
    /// Height
    /// </summary>
    public int height;

    /// <summary>
    /// Width
    /// </summary>
    public int width;

    /// <summary>
    /// Channels
    /// </summary>
    public int channels;
}

/// <summary>
/// JSON tensor
/// </summary>
[Serializable]
public class JSONTensor
{
    /// <summary>
    /// Name
    /// </summary>
    public string name;

    /// <summary>
    /// Shape
    /// </summary>
    public JSONTensorShape shape;

    /// <summary>
    /// Tensor type
    /// </summary>
    public string type;

    /// <summary>
    /// Tensor data
    /// </summary>
    public float[] data;
}

/// <summary>
/// Test set loader
/// </summary>
public class TestSetLoader
{
    /// <summary>
    /// Load test set from file
    /// </summary>
    /// <param name="filename">file name</param>
    /// <returns>`TestSet`</returns>
    public static TestSet Load(string filename)
    {
        if (filename.ToLower().EndsWith(".raw"))
            return LoadRaw(filename);
        else if (filename.ToLower().EndsWith(".gz"))
            return LoadGZ(filename);

        return LoadJSON(filename);
    }

    /// <summary>
    /// Load GZ
    /// </summary>
    /// <param name="filename">file name</param>
    /// <returns>`TestSet`</returns>
    public static TestSet LoadGZ(string filename)
    {
        var jsonFileName = filename.Substring(0, filename.Length - 3);
        var sourceArchiveFileName = Path.Combine(Application.streamingAssetsPath, "TestSet", filename);
        var destinationDirectoryName = sourceArchiveFileName.Substring(0, sourceArchiveFileName.Length - 3);

        FileInfo fileToDecompress = new FileInfo(sourceArchiveFileName);
        using (FileStream originalFileStream = fileToDecompress.OpenRead())
        {
            using (FileStream decompressedFileStream = File.Create(destinationDirectoryName))
            {
                using (GZipStream decompressionStream = new GZipStream(originalFileStream, CompressionMode.Decompress))
                {
                    decompressionStream.CopyTo(decompressedFileStream);
                }
            }
        }

        return LoadJSON(jsonFileName);
    }

    /// <summary>
    /// Load JSON
    /// </summary>
    /// <param name="filename">file name</param>
    /// <returns>`TestSet`</returns>
    public static TestSet LoadJSON(string filename)
    {
        string json = "";

        if (filename.EndsWith(".json"))
            json = File.ReadAllText(Path.Combine(Application.streamingAssetsPath, "TestSet", filename));
        else
            json = Resources.Load<TextAsset>($"TestSet/{filename}").text;

        TestSet result = new TestSet(JsonUtility.FromJson<JSONTestSet>(json));
        return result;
    }

    /// <summary>
    /// Load raw test set
    /// </summary>
    /// <param name="filename">file name</param>
    /// <returns>`TestSet`</returns>
    public static TestSet LoadRaw(string filename)
    {
        string fullpath = Path.Combine(Application.streamingAssetsPath, "TestSet", filename);

        using(BinaryReader file = Open(fullpath))
        {

            var rawTestSet = new RawTestSet();
            rawTestSet.input = LoadFloatArray(file);
            rawTestSet.labels = LoadFloatArray(file);
            return new TestSet(rawTestSet);
        }
    }

    /// <summary>
    /// Load image
    /// </summary>
    /// <param name="filename">file name</param>
    /// <returns>`Texture`</returns>
    public static Texture LoadImage(string filename)
    {
        string fullpath = Path.Combine(Application.streamingAssetsPath, "TestSet", filename);

        var bytes = File.ReadAllBytes(fullpath);
        var tex = new Texture2D(2, 2);
        ImageConversion.LoadImage(tex, bytes, false); // LoadImage will auto-resize the texture dimensions
        tex.wrapMode = TextureWrapMode.Clamp;
        return tex;
    }

    /// <summary>
    /// Load float array
    /// </summary>
    /// <param name="file">binary file reader</param>
    /// <returns>float array</returns>
    public static float[] LoadFloatArray(BinaryReader file)
    {
        Int64 dataLength = file.ReadInt64();
        float[] array = new float[dataLength];
        byte[] bytes = file.ReadBytes(Convert.ToInt32(dataLength * sizeof(float))); // @TODO: support larger than MaxInt32 data blocks
        Buffer.BlockCopy(bytes, 0, array, 0, bytes.Length);

        return array;
    }

    /// <summary>
    /// Open file with binary reader
    /// </summary>
    /// <param name="filename">file name</param>
    /// <returns>`BinaryReader`</returns>
    static BinaryReader Open(string filename)
    {
        return new BinaryReader(new FileStream(filename, FileMode.Open, FileAccess.Read));
    }
}


} // namespace Unity.Barracuda
