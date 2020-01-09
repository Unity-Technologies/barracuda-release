using System;
using System.Collections.Generic;
using System.IO;

using UnityEngine;

namespace Barracuda {


public class TestSet
{
    private RawTestSet rawTestSet;
    private JSONTestSet jsonTestSet;

    public TestSet(RawTestSet rawTestSet)
    {
        this.rawTestSet = rawTestSet;
    }

    public TestSet(JSONTestSet jsonTestSet)
    {
        this.jsonTestSet = jsonTestSet;
    }

    public TestSet()
    {
    }

    public bool SupportsNames()
    {
        if (rawTestSet != null)
            return false;

        return true;
    }

    public int GetOutputCount()
    {
        if (rawTestSet != null)
            return 1;

        return jsonTestSet.outputs.Length;
    }

    public float[] GetOutputData(int idx = 0)
    {
        if (rawTestSet != null)
            return rawTestSet.labels;

        return jsonTestSet.outputs[idx].data;
    }

    public string GetOutputName(int idx = 0)
    {
        if (rawTestSet != null)
            return null;

        return jsonTestSet.outputs[idx].name;
    }

    public int GetInputCount()
    {
        if (rawTestSet != null)
            return 1;

        return jsonTestSet.inputs.Length;
    }

    public string GetInputName(int idx = 0)
    {
        if (rawTestSet != null)
            return "";

        return jsonTestSet.inputs[idx].name;
    }

    public float[] GetInputData(int idx = 0)
    {
        if (rawTestSet != null)
            return rawTestSet.input;

        return jsonTestSet.inputs[idx].data;
    }

    public int[] GetInputShape(int idx = 0)
    {
        if (rawTestSet != null)
            return new int[4] {1, 1, 1, rawTestSet.input.Length};

        return new int[4] {Math.Max(jsonTestSet.inputs[idx].shape.batch,   1),
                           Math.Max(jsonTestSet.inputs[idx].shape.height,  1),
                           Math.Max(jsonTestSet.inputs[idx].shape.width,   1),
                           Math.Max(jsonTestSet.inputs[idx].shape.channels,1)};
    }

    public int[] GetOutputShape(int idx = 0)
    {
        if (rawTestSet != null)
            return new int[4] {1, 1, 1, rawTestSet.labels.Length};

        return new int[4] {Math.Max(jsonTestSet.outputs[idx].shape.batch,   1),
                           Math.Max(jsonTestSet.outputs[idx].shape.height,  1),
                           Math.Max(jsonTestSet.outputs[idx].shape.width,   1),
                           Math.Max(jsonTestSet.outputs[idx].shape.channels,1)};
    }

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

    public Dictionary<string, Tensor> GetOutputsAsTensorDictionary(Dictionary<string, Tensor> outputs = null, int batchCount = -1, int fromBatch = 0)
    {
        if (rawTestSet != null)
            throw new Exception("GetOutputsAsTensorDictionary is not supported for RAW test suites");

        if (outputs == null)
            outputs = new Dictionary<string, Tensor>();

        for (var i = 0; i < GetInputCount(); i++)
            outputs[GetOutputName(i)] = GetOutputAsTensor(i, batchCount, fromBatch);

        return outputs;
    }

    public Tensor GetInputAsTensor(int idx = 0, int batchCount = -1, int fromBatch = 0)
    {
        if (rawTestSet != null)
            throw new Exception("GetInputAsTensor is not supported for RAW test suites");

        var shape = GetInputShape(idx);
        var array = GetInputData(idx);
        var maxBatchCount = array.Length / (shape[1] * shape[2] * shape[3]);

        fromBatch = Math.Min(fromBatch, maxBatchCount - 1);
        if (batchCount < 0)
            batchCount = maxBatchCount - fromBatch;

        // pad data with 0s, if test-set doesn't have enough batches:
        // 1) new ArrayTensorData() will initialize to 0
        // 2) Upload will copy as much data as test-set has into ArrayTensorData
        var tensorShape = new TensorShape(batchCount, shape[1], shape[2], shape[3]);
        var data = new ArrayTensorData(tensorShape.length);
        data.Upload(array, fromBatch * tensorShape.flatWidth, Math.Min(batchCount, maxBatchCount - fromBatch) * tensorShape.flatWidth);

        var res = new Tensor(tensorShape, data);
        res.name = GetInputName(idx);

        return res;
    }

    public Tensor GetOutputAsTensor(int idx = 0, int batchCount = -1, int fromBatch = 0)
    {
        if (rawTestSet != null)
            throw new Exception("GetOutputAsTensor is not supported for RAW test suites");

        var shape = GetOutputShape(idx);
        var array = GetOutputData(idx);
        var maxBatchCount = array.Length / (shape[1] * shape[2] * shape[3]);

        fromBatch = Math.Min(fromBatch, maxBatchCount - 1);
        if (batchCount < 0)
            batchCount = maxBatchCount - fromBatch;
        batchCount = Math.Min(batchCount, maxBatchCount - fromBatch);

        var res = new Tensor(batchCount, shape[1], shape[2], shape[3],
            new SharedArrayTensorData(array, fromBatch * shape[1] * shape[2] * shape[3]));
        res.name = GetOutputName(idx);

        return res;
    }
}

public class RawTestSet
{
    public float[] input;
    public float[] labels;
}

[Serializable]
public class JSONTestSet
{
    public JSONTensor[] inputs;
    public JSONTensor[] outputs;
}


[Serializable]
public class JSONTensorShape
{
    public int batch;
    public int height;
    public int width;
    public int channels;
}

[Serializable]
public class JSONTensor
{
    public string name;
    public JSONTensorShape shape;
    public string type;
    public float[] data;
}


public class TestSetLoader
{
    public static TestSet Load(string filename)
    {
        if (filename.ToLower().EndsWith(".raw"))
            return LoadRaw(filename);

        return LoadJSON(filename);
    }

    public static TestSet LoadJSON(string filename)
    {
        string fullpath = Path.Combine(Application.streamingAssetsPath, "TestSet", filename);

        var json = File.ReadAllText(fullpath);
        TestSet result = new TestSet(JsonUtility.FromJson<JSONTestSet>(json));

        return result;
    }

    public static TestSet LoadRaw(string filename)
    {
        string fullpath = Path.Combine(Application.streamingAssetsPath, "TestSet", filename);

        using(BinaryReader file = Open(fullpath))
        {

            var rawTestSet = new RawTestSet();
            rawTestSet.input = LoadFloatArray(file);
            rawTestSet.labels = LoadFloatArray(file);
            return new TestSet(rawTestSet);;
        }
    }

    public static Texture LoadImage(string filename)
    {
        string fullpath = Path.Combine(Application.streamingAssetsPath, "TestSet", filename);

        var bytes = File.ReadAllBytes(fullpath);
        var tex = new Texture2D(2, 2);
        ImageConversion.LoadImage(tex, bytes, false); // LoadImage will auto-resize the texture dimensions
        tex.wrapMode = TextureWrapMode.Clamp;
        return tex;
    }

    public static float[] LoadFloatArray(BinaryReader file)
    {
        Int64 dataLength = file.ReadInt64();
        float[] array = new float[dataLength];
        byte[] bytes = file.ReadBytes(Convert.ToInt32(dataLength * sizeof(float))); // @TODO: support larger than MaxInt32 data blocks
        Buffer.BlockCopy(bytes, 0, array, 0, bytes.Length);

        return array;
    }

    static BinaryReader Open(string filename)
    {
        return new BinaryReader(new FileStream(filename, FileMode.Open, FileAccess.Read));
    }
}


} // namespace Barracuda
