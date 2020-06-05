using System;
using System.Linq; // Select
using System.Collections.Generic;

namespace Unity.Barracuda {

public class Layer
{
    public enum Type
    {
        Nop = 0,
        Dense = 1,
        MatMul = 2,

        Conv2D = 20,
        DepthwiseConv2D = 21,
        Conv2DTrans = 22,
        Upsample2D = 23,
        MaxPool2D = 25,
        AvgPool2D = 26,
        GlobalMaxPool2D = 27,
        GlobalAvgPool2D = 28,
        Border2D = 29,

        Conv3D = 30,                // TODO: NOT IMPLEMENTED
        Conv3DTrans = 32,           // TODO: NOT IMPLEMENTED
        Upsample3D = 33,            // TODO: NOT IMPLEMENTED
        MaxPool3D = 35,             // TODO: NOT IMPLEMENTED
        AvgPool3D = 36,             // TODO: NOT IMPLEMENTED
        GlobalMaxPool3D = 37,       // TODO: NOT IMPLEMENTED
        GlobalAvgPool3D = 38,       // TODO: NOT IMPLEMENTED
        Border3D = 39,              // TODO: NOT IMPLEMENTED

        Activation = 50,
        ScaleBias = 51,
        Normalization = 52,
        LRN = 53,

        Dropout = 60,
        RandomNormal = 64,
        RandomUniform = 65,
        Multinomial = 66,
        OneHot = 67,

        Add = 100,
        Sub = 101,
        Mul = 102,
        Div = 103,
        Pow = 104,
        Min = 110,
        Max = 111,
        Mean = 112,

        ReduceL1 = 120,             // TODO: NOT IMPLEMENTED
        ReduceL2 = 121,             // TODO: NOT IMPLEMENTED
        ReduceLogSum = 122,         // TODO: NOT IMPLEMENTED
        ReduceLogSumExp = 123,      // TODO: NOT IMPLEMENTED
        ReduceMax = 124,
        ReduceMean = 125,
        ReduceMin = 126,
        ReduceProd = 127,
        ReduceSum = 128,
        ReduceSumSquare = 129,      // TODO: NOT IMPLEMENTED

        Greater = 140,
        GreaterEqual = 141,
        Less = 142,
        LessEqual = 143,
        Equal = 144,
        LogicalOr = 145,
        LogicalAnd = 146,
        LogicalNot = 147,
        LogicalXor = 148,

        Pad2DReflect = 160,
        Pad2DSymmetric = 161,
        Pad2DEdge = 162,

        Flatten = 200,
        Reshape = 201,
        Transpose = 202,
        Squeeze = 203,              // TODO: NOT IMPLEMENTED
        Unsqueeze = 204,            // TODO: NOT IMPLEMENTED
        Gather = 205,
        DepthToSpace = 206,
        SpaceToDepth = 207,
        Expand = 208,
        Resample2D = 209,

        Concat = 210,
        StridedSlice = 211,
        Tile = 212,

        Load = 255
    }

    //Keep in sync with Tensor.cginc ACTIVATION defines
    public enum FusedActivation
    {
        None = Activation.None,
        Relu = Activation.Relu,
        Tanh = Activation.Tanh,
        Sigmoid = Activation.Sigmoid,
        Relu6 = Activation.Relu6,
        Swish = Activation.Swish,
        Neg = Activation.Neg,
        Sqrt = Activation.Sqrt,
        Exp = Activation.Exp,
        Log = Activation.Log
    }

    public enum Activation
    {
        None = 0,
        Relu = 1,
        Softmax = 2,
        Tanh = 3,
        Sigmoid = 4,
        Elu = 5,
        Relu6 = 6,
        LeakyRelu = 7,
        Selu = 8,
        Swish = 9,

        LogSoftmax = 10,
        Softplus = 11,              // TODO: NOT IMPLEMENTED
        Softsign = 12,              // TODO: NOT IMPLEMENTED

        PRelu = 13,

        Hardmax = 20,               // TODO: NOT IMPLEMENTED
        HardSigmoid = 21,           // TODO: NOT IMPLEMENTED

        Abs = 100,
        Neg = 101,
        Ceil = 102,
        Clip = 103,
        Floor = 104,
        Round = 105,                // TODO: NOT IMPLEMENTED

        Reciprocal = 110,
        Sqrt = 111,
        Pow = 112,
        Exp = 113,
        Log = 114,

        Acos = 200,                 // TODO: NOT IMPLEMENTED
        Acosh = 201,                // TODO: NOT IMPLEMENTED
        Asin = 202,                 // TODO: NOT IMPLEMENTED
        Asinh = 203,                // TODO: NOT IMPLEMENTED
        Atan = 204,                 // TODO: NOT IMPLEMENTED
        Atanh = 205,                // TODO: NOT IMPLEMENTED
        Cos = 206,                  // TODO: NOT IMPLEMENTED
        Cosh = 207,                 // TODO: NOT IMPLEMENTED
        Sin = 208,                  // TODO: NOT IMPLEMENTED
        Sinh = 209,                 // TODO: NOT IMPLEMENTED
        Tan = 210                   // TODO: NOT IMPLEMENTED
    }

    public enum AutoPad
    {
        Valid = 0,
        SameUpper = -1,
        SameLower = -2,
    }

    public enum DepthToSpaceMode
    {
        DCR,
        CRD
    }

    public struct DataSet
    {
        public string      name;
        public TensorShape shape;
        public Int64       offset;
        public Int32       itemSizeInBytes;
        public Int32       length;
    }

    public string     name;
    public Type       type;
    public Activation activation;
    public Int32[]    pad;
    public Int32[]    stride;
    public Int32[]    pool;
    public Int32      axis;
    public float      alpha;
    public float      beta;
    public string[]   inputs;

    public DataSet[]  datasets;
    public float[]    weights;

    private Layer(string layerName)
    {
        name = layerName;
        type = Type.Nop;
        activation = Activation.None;
        pad = new int[0];
        stride = new int[0];
        pool = new int[0];
        axis = -1;
        alpha = 1.0f;
        beta = 0.0f;
        inputs = new string[0];
        datasets = new DataSet[0];
        weights = new float[0];
    }

    public Layer(string layerName, Type layerType, Activation activationType = Activation.None) : this(layerName)
    {
        type = layerType;
        activation = activationType;
    }

    public Layer(string layerName, Activation activationType) : this(layerName)
    {
        type = Type.Activation;
        activation = activationType;
    }

    public override string ToString()
    {
        return ($"name:{name}, activation:{activation}, inputs:[{string.Join(",", inputs)}], " +
            $"pad:[{string.Join(",", pad)}], stride:[{string.Join(",", stride)}], pool:[{string.Join(",", pool)}], " +
            $"alpha:{alpha}, beta:{beta}, axis:{axis}, " +
            $"weights:[{string.Join(", ", datasets.Select(x => $"{x.name} {x.shape}"))}]".Replace(name+"/","").Replace(name+" ","")).
            Replace("activation:None, ", "").Replace("inputs:[], ", "").Replace("pad:[], ", "").
            Replace("stride:[], ", "").Replace("stride:[1,1], ", "").Replace("pool:[], ", "").
            Replace("alpha:1, ", "").Replace("beta:0, ", "").Replace("axis:-1, ", "").
            Replace("weights:[]", "");
    }
}

public class Model
{
    public const int Version = 16;

    public struct Input
    {
        public string  name;
        public Int32[] shape; // input shape can contain -1 for unspecified dimensions

        public Input WithName(string name)
        {
            return new Input {name = name, shape = shape};
        }
    }

    public struct Memory
    {
        public TensorShape   shape;
        public string        input;
        public string        output;
    }

    public List<Input>   inputs = new List<Input>();
    public List<string>  outputs = new List<string>();
    public List<Memory>  memories = new List<Memory>();
    public List<Layer>   layers = new List<Layer>();

    #region Importer info
    public string IrSource = "Script";
    public string IrVersion = "NA";
    public string ProducerName = "Script";
    public List<ImporterWarning> Warnings { get; } = new List<ImporterWarning>();
    public class ImporterWarning
    {

        public string Message { get; }
        public string LayerName { get; }
        public ImporterWarning(string layer, string msg)
        {
            Message = msg;
            LayerName = layer;
        }
    }
    #endregion

    public Model ShallowCopy()
    {
        var model = new Model();
        model.inputs.AddRange(inputs);
        model.outputs.AddRange(outputs);
        model.memories.AddRange(memories);
        model.layers.AddRange(layers);

        model.IrSource = IrSource;
        model.IrVersion = IrVersion;
        model.ProducerName = ProducerName;
        model.Warnings.AddRange(Warnings);
        return model;
    }

    public override string ToString()
    {
        // weights are not loaded for UI, recompute size
        var totalUniqueWeights = 0;
        for (var l = 0; l < layers.Count; ++l)
            for (var d = 0; d < layers[l].datasets.Length; ++d)
                totalUniqueWeights += layers[l].datasets[d].length;

        return $"inputs: [{string.Join(", ", inputs.Select(i => $"{i.name} ({string.Join(",", i.shape)})"))}], " +
            $"memories: [{string.Join(", ", memories.Select(m => $"{m.input} {m.shape} {m.output}"))}], " +
            $"outputs: [{string.Join(", ", outputs)}] " +
            $"\n{layers.Count} layers, {totalUniqueWeights:n0} weights: \n{string.Join("\n", layers.Select(i => $"{i.type} ({i})"))}";
    }
}

public static class ModelMetadataExtensions
{
    static public Tensor GetTensorByName(this Model model, string name)
    {
        foreach (var l in model.layers)
            foreach (var ds in l.datasets)
                if (ds.name == name)
                    return new Tensor(ds.shape,
                        new SharedArrayTensorData(l.weights, (int)ds.offset, (int)ds.shape.length), ds.name);

        return null;
    }

    static public TensorShape? GetShapeByName(this Model model, string name)
    {
        foreach (var i in model.inputs)
            if (i.name == name)
                return new TensorShape(i.shape);

        TensorShape shape;
        if (ModelAnalyzer.TryGetOutputTensorShape(model, name, out shape))
            return shape;

        foreach (var l in model.layers)
            foreach (var ds in l.datasets)
                if (ds.name == name)
                    return ds.shape;

        foreach (var mem in model.memories)
        {
            if (mem.input == name || mem.output == name)
                return mem.shape;
        }

        throw new System.Collections.Generic.KeyNotFoundException("Shape " + name + " not found!");
    }
}

} // namespace Unity.Barracuda
