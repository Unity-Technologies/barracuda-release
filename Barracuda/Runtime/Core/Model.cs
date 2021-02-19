using System;
using System.Linq; // Select
using System.Collections.Generic;
using Unity.Barracuda.Compiler.Passes;
using UnityEngine.Assertions;
using UnityEditor;

namespace Unity.Barracuda {

/// <summary>
/// Barracuda Model Layer
/// </summary>
public class Layer
{
    /// <summary>
    /// Layer Type
    /// </summary>
    public enum Type
    {
        /// <summary>
        /// No operation / identity layer
        /// </summary>
        Nop = 0,

        /// <summary>
        /// Dense layer
        /// </summary>
        Dense = 1,

        /// <summary>
        /// Matrix multiplication layer
        /// </summary>
        MatMul = 2,

        /// <summary>
        /// 2D Convolution layer
        /// </summary>
        Conv2D = 20,

        /// <summary>
        /// Depthwise Convolution layer
        /// </summary>
        DepthwiseConv2D = 21,

        /// <summary>
        /// Transpose 2D Convolution layer
        /// </summary>
        Conv2DTrans = 22,

        /// <summary>
        /// Upsampling layer
        /// </summary>
        Upsample2D = 23,

        /// <summary>
        /// Max Pool layer
        /// </summary>
        MaxPool2D = 25,

        /// <summary>
        /// Average Pool layer
        /// </summary>
        AvgPool2D = 26,

        /// <summary>
        /// Global Max Pool layer
        /// </summary>
        GlobalMaxPool2D = 27,

        /// <summary>
        /// Global Average Pool layer
        /// </summary>
        GlobalAvgPool2D = 28,

        /// <summary>
        /// Border / Padding layer
        /// </summary>
        Border2D = 29,

        /// <summary>
        /// 3D Convolution layer
        /// </summary>
        Conv3D = 30,

        /// <summary>
        /// Transpose 3D Convolution layer (not yet implemented)
        /// </summary>
        Conv3DTrans = 32,           // TODO: NOT IMPLEMENTED

        /// <summary>
        /// 3D Upsampling layer
        /// </summary>
        Upsample3D = 33,

        /// <summary>
        /// 3D Max Pool layer (not yet implemented)
        /// </summary>
        MaxPool3D = 35,             // TODO: NOT IMPLEMENTED

        /// <summary>
        /// 3D Average Pool layer (not yet implemented)
        /// </summary>
        AvgPool3D = 36,             // TODO: NOT IMPLEMENTED

        /// <summary>
        /// 3D Global Max Pool layer (not yet implemented)
        /// </summary>
        GlobalMaxPool3D = 37,       // TODO: NOT IMPLEMENTED

        /// <summary>
        /// 3D Global Average Pool layer (not yet implemented)
        /// </summary>
        GlobalAvgPool3D = 38,       // TODO: NOT IMPLEMENTED

        /// <summary>
        /// 3D Border / Padding layer
        /// </summary>
        Border3D = 39,

        /// <summary>
        /// Activation layer, see `Activation` enum for activation types
        /// </summary>
        Activation = 50,

        /// <summary>
        /// Scale + Bias layer
        /// </summary>
        ScaleBias = 51,

        /// <summary>
        /// Normalization layer
        /// </summary>
        Normalization = 52,

        /// <summary>
        /// LRN (Local Response Normalization) layer
        /// </summary>
        LRN = 53,

        /// <summary>
        /// Dropout layer (does nothing in inference)
        /// </summary>
        Dropout = 60,

        /// <summary>
        /// Random sampling from normal distribution layer
        /// </summary>
        RandomNormal = 64,

        /// <summary>
        /// Random sampling from uniform distribution layer
        /// </summary>
        RandomUniform = 65,

        /// <summary>
        /// Random sampling from multinomial distribution layer
        /// </summary>
        Multinomial = 66,

        /// <summary>
        /// OneHot layer
        /// </summary>
        OneHot = 67,

        /// <summary>
        /// TopK indices layer
        /// </summary>
        TopKIndices = 68,

        /// <summary>
        /// TopK values layer
        /// </summary>
        TopKValues = 69,

        /// <summary>
        /// NonZero layer
        /// </summary>
        NonZero = 70,

        /// <summary>
        /// Addition layer
        /// </summary>
        Add = 100,

        /// <summary>
        /// Subtraction layer
        /// </summary>
        Sub = 101,

        /// <summary>
        /// Multiplication layer
        /// </summary>
        Mul = 102,

        /// <summary>
        /// Division layer
        /// </summary>
        Div = 103,

        /// <summary>
        /// Power layer
        /// </summary>
        Pow = 104,

        /// <summary>
        /// Min layer
        /// </summary>
        Min = 110,

        /// <summary>
        /// Max layer
        /// </summary>
        Max = 111,

        /// <summary>
        /// Mean layer
        /// </summary>
        Mean = 112,

        /// <summary>
        /// Reduce L1 layer (not yet implemented)
        /// </summary>
        ReduceL1 = 120,             // TODO: NOT IMPLEMENTED

        /// <summary>
        /// Reduce L2 layer (not yet implemented)
        /// </summary>
        ReduceL2 = 121,             // TODO: NOT IMPLEMENTED

        /// <summary>
        /// Reduce LogSum layer (not yet implemented)
        /// </summary>
        ReduceLogSum = 122,         // TODO: NOT IMPLEMENTED

        /// <summary>
        /// Reduce LogSumExp layer (not yet implemented)
        /// </summary>
        ReduceLogSumExp = 123,      // TODO: NOT IMPLEMENTED

        /// <summary>
        /// Reduce with Max layer
        /// </summary>
        ReduceMax = 124,

        /// <summary>
        /// Reduce with Mean layer
        /// </summary>
        ReduceMean = 125,

        /// <summary>
        /// Reduce with Min layer
        /// </summary>
        ReduceMin = 126,

        /// <summary>
        /// Reduce with Prod layer
        /// </summary>
        ReduceProd = 127,

        /// <summary>
        /// Reduce with Sum layer
        /// </summary>
        ReduceSum = 128,

        /// <summary>
        /// Reduce with SumSquare layer (not yet implemented)
        /// </summary>
        ReduceSumSquare = 129,      // TODO: NOT IMPLEMENTED

        /// <summary>
        /// Logic operation: Greater layer
        /// </summary>
        Greater = 140,

        /// <summary>
        /// Logic operation: GreaterEqual layer
        /// </summary>
        GreaterEqual = 141,

        /// <summary>
        /// Logic operation: Less layer
        /// </summary>
        Less = 142,

        /// <summary>
        /// Logic operation: LessEqual layer
        /// </summary>
        LessEqual = 143,

        /// <summary>
        /// Logic operation: Equal layer
        /// </summary>
        Equal = 144,

        /// <summary>
        /// Logic operation: LogicalOr layer
        /// </summary>
        LogicalOr = 145,

        /// <summary>
        /// Logic operation: LogicalAnd layer
        /// </summary>
        LogicalAnd = 146,

        /// <summary>
        /// Logic operation: LogicalNot layer
        /// </summary>
        LogicalNot = 147,

        /// <summary>
        /// Logic operation: LogicalXor layer
        /// </summary>
        LogicalXor = 148,

        /// <summary>
        /// Logic operation: Where layer
        /// </summary>
        Where = 149,

        /// <summary>
        /// Reflection padding layer
        /// </summary>
        Pad2DReflect = 160,

        /// <summary>
        /// Symmetric padding layer
        /// </summary>
        Pad2DSymmetric = 161,

        /// <summary>
        /// Edge padding layer
        /// </summary>
        Pad2DEdge = 162,

        /// <summary>
        /// ArgMax layer
        /// </summary>
        ArgMax = 163,

        /// <summary>
        /// ArgMin layer
        /// </summary>
        ArgMin = 164,

        /// <summary>
        /// ConstantOfShape layer
        /// </summary>
        ConstantOfShape = 199,

        /// <summary>
        /// Flatten layer
        /// </summary>
        Flatten = 200,

        /// <summary>
        /// Reshape layer
        /// </summary>
        Reshape = 201,

        /// <summary>
        /// Transpose layer
        /// </summary>
        Transpose = 202,

        /// <summary>
        /// Squeeze layer (not fully supported)
        /// </summary>
        Squeeze = 203,              // TODO: NOT IMPLEMENTED

        /// <summary>
        /// Unsqueeze layer (not fully supported)
        /// </summary>
        Unsqueeze = 204,            // TODO: NOT IMPLEMENTED

        /// <summary>
        /// Gather layer
        /// </summary>
        Gather = 205,

        /// <summary>
        /// Depth to space layer
        /// </summary>
        DepthToSpace = 206,

        /// <summary>
        /// Space to depth layer
        /// </summary>
        SpaceToDepth = 207,

        /// <summary>
        /// Expand layer
        /// </summary>
        Expand = 208,

        /// <summary>
        /// 2D Resample layer
        /// </summary>
        Resample2D = 209,

        /// <summary>
        /// Concat layer
        /// </summary>
        Concat = 210,

        /// <summary>
        /// Strided slice layer
        /// </summary>
        StridedSlice = 211,

        /// <summary>
        /// Tile layer
        /// </summary>
        Tile = 212,

        /// <summary>
        /// Shape layer
        /// </summary>
        Shape = 213,

        /// <summary>
        /// Non max suppression layer
        /// </summary>
        NonMaxSuppression = 214,

        /// <summary>
        /// LSTM
        /// </summary>
        LSTM = 215,                // TODO: NOT IMPLEMENTED - Expanded via ExpandOpsPass

        /// <summary>
        /// Constant load layer (for internal use)
        /// </summary>
        Load = 255
    }

    //Keep in sync with Tensor.cginc ACTIVATION defines and IsActivationFusable() methods in ModelBuilder.cs and FuseActivationsPass.cs
    /// <summary>
    /// Fused activations enum
    /// </summary>
    public enum FusedActivation
    {
        /// <summary>
        /// None
        /// </summary>
        None = Activation.None,

        /// <summary>
        /// Relu
        /// </summary>
        Relu = Activation.Relu,

        /// <summary>
        /// Tanh
        /// </summary>
        Tanh = Activation.Tanh,

        /// <summary>
        /// Softplus
        /// </summary>
        Softplus = Activation.Softplus,

        /// <summary>
        /// Sigmoid
        /// </summary>
        Sigmoid = Activation.Sigmoid,

        /// <summary>
        /// Relu6
        /// </summary>
        Relu6 = Activation.Relu6,

        /// <summary>
        /// Swish
        /// </summary>
        Swish = Activation.Swish,

        /// <summary>
        /// Neg
        /// </summary>
        Neg = Activation.Neg,

        /// <summary>
        /// Sqrt
        /// </summary>
        Sqrt = Activation.Sqrt,

        /// <summary>
        /// Exp
        /// </summary>
        Exp = Activation.Exp,

        /// <summary>
        /// Log
        /// </summary>
        Log = Activation.Log,

        /// <summary>
        /// Acos
        /// </summary>
        Acos = Activation.Acos,

        /// <summary>
        /// Acosh
        /// </summary>
        Acosh = Activation.Acosh,

        /// <summary>
        /// Asin
        /// </summary>
        Asin = Activation.Asin,

        /// <summary>
        /// Asinh
        /// </summary>
        Asinh = Activation.Asinh,

        /// <summary>
        /// Atan
        /// </summary>
        Atan = Activation.Atan,

        /// <summary>
        /// Atanh
        /// </summary>
        Atanh = Activation.Atanh,

        /// <summary>
        /// Cos
        /// </summary>
        Cos = Activation.Cos,

        /// <summary>
        /// Cosh
        /// </summary>
        Cosh = Activation.Cosh,

        /// <summary>
        /// Sin
        /// </summary>
        Sin = Activation.Sin,

        /// <summary>
        /// Sinh
        /// </summary>
        Sinh = Activation.Sinh,

        /// <summary>
        /// Tan
        /// </summary>
        Tan = Activation.Tan
    }

    /// <summary>
    /// Activation enum
    /// </summary>
    public enum Activation
    {
        /// <summary>
        /// None
        /// </summary>
        None = 0,

        /// <summary>
        /// Relu
        /// </summary>
        Relu = 1,

        /// <summary>
        /// Softmax
        /// </summary>
        Softmax = 2,

        /// <summary>
        /// Tanh
        /// </summary>
        Tanh = 3,

        /// <summary>
        /// Sigmoid
        /// </summary>
        Sigmoid = 4,

        /// <summary>
        /// Elu
        /// </summary>
        Elu = 5,

        /// <summary>
        /// Relu6
        /// </summary>
        Relu6 = 6,

        /// <summary>
        /// LeakyRelu
        /// </summary>
        LeakyRelu = 7,

        /// <summary>
        /// Selu
        /// </summary>
        Selu = 8,

        /// <summary>
        /// Swish
        /// </summary>
        Swish = 9,

        /// <summary>
        /// LogSoftmax
        /// </summary>
        LogSoftmax = 10,

        /// <summary>
        /// Softplus
        /// </summary>
        Softplus = 11,

        /// <summary>
        /// Softsign (not yet implemented)
        /// </summary>
        Softsign = 12,              // TODO: NOT IMPLEMENTED

        /// <summary>
        /// PRelu
        /// </summary>
        PRelu = 13,

        /// <summary>
        /// Hardmax (not yet implemented)
        /// </summary>
        Hardmax = 20,               // TODO: NOT IMPLEMENTED

        /// <summary>
        /// HardSigmoid (not yet implemented)
        /// </summary>
        HardSigmoid = 21,           // TODO: NOT IMPLEMENTED

        /// <summary>
        /// Abs
        /// </summary>
        Abs = 100,

        /// <summary>
        /// Neg
        /// </summary>
        Neg = 101,

        /// <summary>
        /// Ceil
        /// </summary>
        Ceil = 102,

        /// <summary>
        /// Clip
        /// </summary>
        Clip = 103,

        /// <summary>
        /// Floor
        /// </summary>
        Floor = 104,

        /// <summary>
        /// Round (not yet implemented)
        /// </summary>
        Round = 105,                // TODO: NOT IMPLEMENTED

        /// <summary>
        /// Reciprocal
        /// </summary>
        Reciprocal = 110,

        /// <summary>
        /// Sqrt
        /// </summary>
        Sqrt = 111,

        /// <summary>
        /// Pow
        /// </summary>
        Pow = 112,

        /// <summary>
        /// Exp
        /// </summary>
        Exp = 113,

        /// <summary>
        /// Log
        /// </summary>
        Log = 114,

        /// <summary>
        /// Acos
        /// </summary>
        Acos = 200,

        /// <summary>
        /// Acosh
        /// </summary>
        Acosh = 201,

        /// <summary>
        /// Asin
        /// </summary>
        Asin = 202,

        /// <summary>
        /// Asinh
        /// </summary>
        Asinh = 203,

        /// <summary>
        /// Atan
        /// </summary>
        Atan = 204,

        /// <summary>
        /// Atanh
        /// </summary>
        Atanh = 205,

        /// <summary>
        /// Cos
        /// </summary>
        Cos = 206,

        /// <summary>
        /// Cosh
        /// </summary>
        Cosh = 207,

        /// <summary>
        /// Sin
        /// </summary>
        Sin = 208,

        /// <summary>
        /// Sinh
        /// </summary>
        Sinh = 209,

        /// <summary>
        /// Tan
        /// </summary>
        Tan = 210
    }

    /// <summary>
    /// Auto padding enum
    /// </summary>
    public enum AutoPad
    {
        /// <summary>
        /// Valid
        /// </summary>
        Valid = 0,

        /// <summary>
        /// Same upper
        /// </summary>
        SameUpper = -1,

        /// <summary>
        /// Same lower
        /// </summary>
        SameLower = -2,
    }

    /// <summary>
    /// Depth to space mode enum
    /// </summary>
    public enum DepthToSpaceMode
    {
        /// <summary>
        /// DCR (Depth Column Row)
        /// </summary>
        DCR,

        /// <summary>
        /// CRD (Column Row Depth)
        /// </summary>
        CRD
    }

    /// <summary>
    /// Layer param data structure
    /// </summary>
    public struct DataSet
    {
        /// <summary>
        /// Name
        /// </summary>
        public string      name;

        /// <summary>
        /// Shape
        /// </summary>
        public TensorShape shape;

        /// <summary>
        /// Offset from start
        /// </summary>
        public Int64       offset;

        /// <summary>
        /// Item size in bytes
        /// </summary>
        public Int32       itemSizeInBytes;

        /// <summary>
        /// Dataset length
        /// </summary>
        public Int32       length;
    }

    /// <summary>
    /// Layer preservation flags
    /// </summary>
    [Flags]
    public enum Flags
    {
        /// <summary>
        /// No flags defined
        /// </summary>
        None     =      0,

        /// <summary>
        /// Preserve the layer (e.g. don't remove it in a model pass)
        /// </summary>
        Preserve = 1 << 1,
    }

    /// <summary>
    /// Layer name
    /// </summary>
    public string     name;

    /// <summary>
    /// Layer type
    /// </summary>
    public Type       type;

    /// <summary>
    /// Layer flags (not serialized) - used for conversion
    /// </summary>
    [NonSerialized]
    public Flags      flags;

    /// <summary>
    /// Layer activation type
    /// </summary>
    public Activation activation;

    /// <summary>
    /// Padding shape
    /// </summary>
    public Int32[]    pad;

    /// <summary>
    /// Stride
    /// </summary>
    public Int32[]    stride;

    /// <summary>
    /// Pooling
    /// </summary>
    public Int32[]    pool;

    /// <summary>
    /// Axis
    /// </summary>
    public Int32      axis;

    /// <summary>
    /// Alpha
    /// </summary>
    public float      alpha;

    /// <summary>
    /// Beta
    /// </summary>
    public float      beta;

    /// <summary>
    /// Input (layer) names
    /// </summary>
    public string[]   inputs;

    /// <summary>
    /// Output (layer) names (not serialized) - used for conversion
    /// </summary>
    [NonSerialized]
    public string[]   outputs;

    /// <summary>
    /// Axes (not serialized) - used for conversion
    /// </summary>
    [NonSerialized]
    public Int32[]    axes;

    /// <summary>
    /// Datasets bound to layer
    /// </summary>
    public DataSet[]  datasets;

    /// <summary>
    /// Flat weights array (for actual shape see `datasets`)
    /// </summary>
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

    /// <summary>
    /// Constructs Layer
    /// </summary>
    /// <param name="layerName">layer name</param>
    /// <param name="layerType">layer type</param>
    /// <param name="activationType">layer activation type</param>
    public Layer(string layerName, Type layerType, Activation activationType = Activation.None) : this(layerName)
    {
        type = layerType;
        activation = activationType;
    }

    /// <summary>
    /// Constructs Activation Layer
    /// </summary>
    /// <param name="layerName">layer name</param>
    /// <param name="activationType">layer activation type</param>
    public Layer(string layerName, Activation activationType) : this(layerName)
    {
        type = Type.Activation;
        activation = activationType;
    }

    /// <summary>
    /// Layer summary string
    /// </summary>
    /// <returns>layer summary string</returns>
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

    /// <summary>
    /// Converts DataSet to Tensor
    /// </summary>
    /// <param name="index">dataset index</param>
    /// <returns>Tensor</returns>
    public Tensor DataSetToTensor(int index)
    {
        Assert.IsTrue(index < datasets.Length);
        var ds = datasets[index];
        return new Tensor(ds.shape, new SharedArrayTensorData(weights, (int)ds.offset, (int)ds.shape.length), ds.name);
    }

    /// <summary>
    /// Converts Tensor to DataSet
    /// </summary>
    /// <param name="X">input `Tensor`</param>
    /// <param name="index">dataset index</param>
    public void ApplyTensorToDataSet(Tensor X, int index)
    {
        Assert.IsTrue(index < datasets.Length);
        var ds = datasets[index];
        ds.shape = X.shape;
        Array.Copy(X.ToReadOnlyArray(), 0, weights, ds.offset, ds.shape.length);
        datasets[index] = ds;
    }
}

/// <summary>
/// Neural Net Model data structure
/// </summary>
public class Model
{
    /// <summary>
    /// Model version, incremented with each data structure change
    /// </summary>
    public const int Version = 17;
    internal const int LastVersionWithout8DSupport = 16;

    /// <summary>
    /// Input data structure
    /// </summary>
    public struct Input
    {
        /// <summary>
        /// Name
        /// </summary>
        public string  name;

        /// <summary>
        /// Shape as `int` array
        /// </summary>
        public Int32[] shape; // input shape can contain -1 for unspecified dimensions

        /// <summary>
        /// Input rank
        /// </summary>
        public int rank;

        /// <summary>
        /// Creates input structure with specified name
        /// </summary>
        /// <param name="name">name</param>
        /// <returns>Input structure</returns>
        public Input WithName(string name)
        {
            return new Input {name = name, shape = shape};
        }
    }

    /// <summary>
    /// Memory data structure. Used by recurrent models to store information about recurrent inputs/outputs
    /// </summary>
    public struct Memory
    {
        /// <summary>
        /// Shape
        /// </summary>
        public TensorShape   shape;

        /// <summary>
        /// Input name
        /// </summary>
        public string        input;

        /// <summary>
        /// Output name
        /// </summary>
        public string        output;
    }

    /// <summary>
    /// Model layout
    /// </summary>
    public string        layout = String.Empty;

    /// <summary>
    /// All model inputs
    /// </summary>
    public List<Input>   inputs = new List<Input>();

    /// <summary>
    /// All model outputs
    /// </summary>
    public List<string>  outputs = new List<string>();

    /// <summary>
    /// All model memories
    /// </summary>
    public List<Memory>  memories = new List<Memory>();

    /// <summary>
    /// All model layers
    /// </summary>
    public List<Layer>   layers = new List<Layer>();

    #region Importer info
    /// <summary>
    /// Model source metadata string
    /// </summary>
    public string IrSource = "Script";

    /// <summary>
    /// Model ONNX version metadata string
    /// </summary>
    public string IrVersion = "NA";

    /// <summary>
    /// Model producer metadata string
    /// </summary>
    public string ProducerName = "Script";

    /// <summary>
    /// Model import warnings
    /// </summary>
    public List<ImporterWarning> Warnings { get; } = new List<ImporterWarning>();

    /// <summary>
    /// Importer warning data structure
    /// </summary>
    public class ImporterWarning
    {
        /// <summary>
        /// Message
        /// </summary>
        public string Message { get; }

        /// <summary>
        /// Layer name
        /// </summary>
        public string LayerName { get; }

        /// <summary>
        /// Constructs ImporterWarning
        /// </summary>
        /// <param name="layer">layer name</param>
        /// <param name="msg">message</param>
        public ImporterWarning(string layer, string msg)
        {
            Message = msg;
            LayerName = layer;
        }
    }
    #endregion

    [Flags]
    internal enum Flags
    {
        /// <summary>
        /// No flags defined
        /// </summary>
        None     =      0,

        /// <summary>
        /// Requires compilation (i.e. for ops that must be expanded)
        /// </summary>
        NeedsCompilation = 1 << 1,
    }

    internal Flags flags;

    /// <summary>
    /// Build shallow copy of the model
    /// </summary>
    /// <returns>shallow copy of the model</returns>
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

    /// <summary>
    /// Model summary string
    /// </summary>
    /// <returns>Model summary string</returns>
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

    internal void Compile()
    {
        var expandOpsPass = new ExpandOpsPass();
        var model = this;
        expandOpsPass.Run(ref model);

        var validatePass = new ValidateNHWCPass();
        var warnings = new List<Model.ImporterWarning>();
        validatePass.Run(this, ref warnings);

        foreach (var warning in warnings)
            Debug.LogWarning(warning);

        // Clear flag
        flags &= ~Flags.NeedsCompilation;
    }

}

/// <summary>
/// Model metadata extensions
/// </summary>
public static class ModelMetadataExtensions
{
    /// <summary>
    /// Get model tensor by name
    /// </summary>
    /// <param name="model">Model</param>
    /// <param name="name">Tensor name</param>
    /// <returns>Tensor</returns>
    static public Tensor GetTensorByName(this Model model, string name)
    {
        foreach (var l in model.layers)
            foreach (var ds in l.datasets)
                if (ds.name == name)
                    return new Tensor(ds.shape,
                        new SharedArrayTensorData(l.weights, (int)ds.offset, (int)ds.shape.length), ds.name);

        return null;
    }

    /// <summary>
    /// Get model tensor shape by name
    /// </summary>
    /// <param name="model">Model</param>
    /// <param name="name">Tensor name</param>
    /// <returns>Tensor shape</returns>
    /// <exception cref="KeyNotFoundException"></exception>
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

    /// <summary>
    /// Get count of layers that directly depend on specified input
    /// </summary>
    /// <param name="model">Model</param>
    /// <param name="name">input name</param>
    /// <returns>count of layers that directly depend on specified input</returns>
    static public int GetDownStreamLayersCount(this Model model, string name)
    {
        return model.layers.Count(x => x.inputs.Contains(name));
    }
}

} // namespace Unity.Barracuda
