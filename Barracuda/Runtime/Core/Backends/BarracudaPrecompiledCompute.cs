using UnityEngine;
using UnityEngine.Assertions;
using System;
using System.Linq;
using System.Collections.Generic;


namespace Unity.Barracuda {

/// <summary>
/// Precompiled GPU compute `IOps` implementation
/// </summary>
public class PrecompiledComputeOps : ComputeOps, IModelCompiler
{
    /// <summary>
    /// Create `PrecompiledComputeOps`
    /// </summary>
    /// <param name="allocator">allocator</param>
    /// <param name="verbose">verbose flag</param>
    public PrecompiledComputeOps(ITensorAllocator allocator = null, bool verbose = false)
    : base(allocator, verbose)
    {
    }

    // ---------------------------------------------------------------------------------

    static internal ComputeFunc.TensorDecl _DeclX = ComputeFunc.GetTensorDecl("X");
    static internal ComputeFunc.TensorDecl _DeclO = ComputeFunc.GetTensorDecl("O");
    static internal ComputeFunc.TensorDecl _DeclW = ComputeFunc.GetTensorDecl("W");
    static internal ComputeFunc.TensorDecl _DeclK = ComputeFunc.GetTensorDecl("K");
    static internal ComputeFunc.TensorDecl _DeclB = ComputeFunc.GetTensorDecl("B");
    static internal int _DataX = ComputeFunc.GetTensorData("X");
    static internal int _DataO = ComputeFunc.GetTensorData("O");
    static internal int _DataW = ComputeFunc.GetTensorData("W");
    static internal int _DataK = ComputeFunc.GetTensorData("K");
    static internal int _DataB = ComputeFunc.GetTensorData("B");
    static internal int _DataWBK = ComputeFunc.GetTensorData("WBK");
    static internal int _Stride = Shader.PropertyToID("_Stride");
    static internal int _Pad = Shader.PropertyToID("_Pad");
    static internal int _Pool = Shader.PropertyToID("_Pool");
    static internal int _Alpha = Shader.PropertyToID("_Alpha");
    static internal int _Beta  = Shader.PropertyToID("_Beta");

    private struct CompiledInstruction
    {
        public ComputeKernel kernel;
        public Tensor[] tensors;
        public TensorShape shape;
    }

    private struct CompiledLayer
    {
        // output shape might not match instruction output shape
        public TensorShape shape;
        public CompiledInstruction[] instructions;

        // most layers are made up of 1 instruction
        public ComputeKernel kernel { get { return (instructions == null) ? new ComputeKernel() : instructions[0].kernel; } }
    }

    private int m_CachedModelHash;
    private Dictionary<Layer, CompiledLayer> m_CompiledLayers = new Dictionary<Layer, CompiledLayer>();
    private CompiledLayer m_Compiled;

    private Dictionary<string, ComputeBuffer> m_CachedModelBuffers = new Dictionary<string, ComputeBuffer>();

    private ComputeBuffer NewComputeBuffer(string name, int count, int stride)
    {
        if(!m_CachedModelBuffers.ContainsKey(name))
            m_CachedModelBuffers[name] = new ComputeBuffer(count, stride);
        if(m_CachedModelBuffers[name].count != count || m_CachedModelBuffers[name].stride != stride)
        {
            m_CachedModelBuffers[name].Dispose();
            m_CachedModelBuffers[name] = new ComputeBuffer(count, stride);
        }

        return m_CachedModelBuffers[name];
    }

    /// <inheritdoc/>
    public override void ResetAllocator(bool keepCachedMemory = true)
    {
        if (!keepCachedMemory)
        {
            foreach (var buf in m_CachedModelBuffers)
                buf.Value.Dispose();
            m_CachedModelBuffers.Clear();

            foreach (var l in m_CompiledLayers)
                foreach (var i in l.Value.instructions)
                {
                    if (i.tensors == null)
                        continue;
                    foreach (var t in i.tensors)
                        t.Dispose();
                }
            m_CompiledLayers.Clear();
        }

        base.ResetAllocator(keepCachedMemory);
    }

    private int CalcModelWithInputsHashCode(Model model, IDictionary<string, TensorShape> inputShapes)
    {
        var hash = model.GetHashCode();
        foreach (var entry in inputShapes)
        {
            hash = (hash * 7) + entry.Key.GetHashCode();
            hash = (hash * 7) + entry.Value.GetHashCode();
        }
        return hash;
    }

    private Tensor[] PrepareConv2dWinograd(Model model, Layer l)
    {
        var K = l.datasets[0];
        var Kshape = new TensorShape(K.shape.batch + 1, K.shape.height + 1, K.shape.width, K.shape.channels);

        var B = l.datasets[1];
        var Bshape = B.shape;

        var weights = new float[Kshape.length + Bshape.length];

        for (int c = 0; c < Kshape.kernelDepth; ++c)
            for (int k = 0; k < Kshape.kernelCount; ++k)
            {
                float g00 = l.weights[K.offset + K.shape.Index(0, 0, c, k)];
                float g01 = l.weights[K.offset + K.shape.Index(0, 1, c, k)];
                float g02 = l.weights[K.offset + K.shape.Index(0, 2, c, k)];
                float g10 = l.weights[K.offset + K.shape.Index(1, 0, c, k)];
                float g11 = l.weights[K.offset + K.shape.Index(1, 1, c, k)];
                float g12 = l.weights[K.offset + K.shape.Index(1, 2, c, k)];
                float g20 = l.weights[K.offset + K.shape.Index(2, 0, c, k)];
                float g21 = l.weights[K.offset + K.shape.Index(2, 1, c, k)];
                float g22 = l.weights[K.offset + K.shape.Index(2, 2, c, k)];

                // float4x3 Winograd_G = float4x3(float3(1, 0, 0), float3(0.5, 0.5, 0.5), float3(0.5, -0.5, 0.5), float3(0, 0, 1));
                // float3x4 Winograd_GT = transpose(Winograd_G);
                // float4x4 v = mul(Winograd_G, mul(g, Winograd_GT));
                float w00 = g00;
                float w01 = 0.5f * g00 + 0.5f * g01 + 0.5f * g02;
                float w02 = 0.5f * g00 - 0.5f * g01 + 0.5f * g02;
                float w03 = g02;

                float w10 = g10;
                float w11 = 0.5f * g10 + 0.5f * g11 + 0.5f * g12;
                float w12 = 0.5f * g10 - 0.5f * g11 + 0.5f * g12;
                float w13 = g12;

                float w20 = g20;
                float w21 = 0.5f * g20 + 0.5f * g21 + 0.5f * g22;
                float w22 = 0.5f * g20 - 0.5f * g21 + 0.5f * g22;
                float w23 = g22;

                float v00 = w00;
                float v01 = w01;
                float v02 = w02;
                float v03 = w03;

                float v10 = 0.5f * w00 + 0.5f * w10 + 0.5f * w20;
                float v11 = 0.5f * w01 + 0.5f * w11 + 0.5f * w21;
                float v12 = 0.5f * w02 + 0.5f * w12 + 0.5f * w22;
                float v13 = 0.5f * w03 + 0.5f * w13 + 0.5f * w23;

                float v20 = 0.5f * w00 - 0.5f * w10 + 0.5f * w20;
                float v21 = 0.5f * w01 - 0.5f * w11 + 0.5f * w21;
                float v22 = 0.5f * w02 - 0.5f * w12 + 0.5f * w22;
                float v23 = 0.5f * w03 - 0.5f * w13 + 0.5f * w23;

                float v30 = w20;
                float v31 = w21;
                float v32 = w22;
                float v33 = w23;

                weights[Kshape.Index(0, 0, c, k)] = v00;
                weights[Kshape.Index(1, 0, c, k)] = v10;
                weights[Kshape.Index(2, 0, c, k)] = v20;
                weights[Kshape.Index(3, 0, c, k)] = v30;
                weights[Kshape.Index(0, 1, c, k)] = v01;
                weights[Kshape.Index(1, 1, c, k)] = v11;
                weights[Kshape.Index(2, 1, c, k)] = v21;
                weights[Kshape.Index(3, 1, c, k)] = v31;
                weights[Kshape.Index(0, 2, c, k)] = v02;
                weights[Kshape.Index(1, 2, c, k)] = v12;
                weights[Kshape.Index(2, 2, c, k)] = v22;
                weights[Kshape.Index(3, 2, c, k)] = v32;
                weights[Kshape.Index(0, 3, c, k)] = v03;
                weights[Kshape.Index(1, 3, c, k)] = v13;
                weights[Kshape.Index(2, 3, c, k)] = v23;
                weights[Kshape.Index(3, 3, c, k)] = v33;
            }

        Buffer.BlockCopy(l.weights, (int)B.offset * sizeof(float), weights, Kshape.length * sizeof(float), B.length * sizeof(float));

        ComputeBuffer buffer = NewComputeBuffer(l.name + "_precompiled", Kshape.length + Bshape.length, sizeof(float));
        buffer.SetData(weights);
        var Kw = new Tensor(Kshape, new SharedComputeTensorData(buffer, Kshape, 0));
        var Bw = new Tensor(Bshape, new SharedComputeTensorData(buffer, Bshape, Kshape.length));

        return new Tensor[] { Kw, Bw };
    }

    private Tensor[] PrepareConv2DTrans(Model model, Layer l)
    {
        var K = l.datasets[0];
        var B = l.datasets[1];

        var weights = new float[K.length + B.length];

        for (int y = 0; y < K.shape.kernelHeight; ++y)
            for (int x = 0; x < K.shape.kernelWidth; ++x)
                for (int c = 0; c < K.shape.kernelDepth; ++c)
                    for (int k = 0; k < K.shape.kernelCount; ++k)
                        {
                            float v = l.weights[K.offset + K.shape.Index(K.shape.kernelHeight - 1 - y, K.shape.kernelWidth - 1 - x, c, k)];
                            weights[K.shape.Index(y, x, c, k)] = v;
                        }

        Buffer.BlockCopy(l.weights, (int)B.offset * sizeof(float), weights, K.length * sizeof(float), B.length * sizeof(float));

        ComputeBuffer buffer = NewComputeBuffer(l.name + "_precompiled", K.length + B.length, sizeof(float));
        buffer.SetData(weights);
        var Kw = new Tensor(K.shape, new SharedComputeTensorData(buffer, K.shape, 0));
        var Bw = new Tensor(B.shape, new SharedComputeTensorData(buffer, B.shape, K.length));

        return new Tensor[] { Kw, Bw };
    }

    /// <inheritdoc/>
    public virtual void PrepareModel(Model model, IDictionary<string, TensorShape> inputShapes)
    {
        var modelHash = CalcModelWithInputsHashCode(model, inputShapes);
        if (modelHash == m_CachedModelHash)
            return;
        m_CachedModelHash = modelHash;

        ResetAllocator(false);

        IDictionary<string, TensorShape?> shapesByName;
        ModelAnalyzer.ListTemporaryTensorShapes(model, inputShapes, out shapesByName);

        foreach (var l in model.layers)
        {
            if (m_CompiledLayers.ContainsKey(l))
                continue; // already compiled

            if (l.inputs.Length == 0)
                continue;   // don't need to compile layers without inputs, so far all of them are CPU only

            if (!shapesByName.TryGetValue(l.inputs[0], out TensorShape? input0Shape)
                || input0Shape == null
                || !shapesByName.TryGetValue(l.name, out TensorShape? outputShape)
                || outputShape == null)
                continue;

            var X = shapesByName[l.inputs[0]].Value;
            var O = shapesByName[l.name].Value;

            ComputeKernel kernel = new ComputeKernel();
            if (l.type == Layer.Type.Dense)
            {
                var instructions = new List<CompiledInstruction>();
                var itemSize = 4; // @TODO: itemSizeInBytes == 2 | float16
                kernel = BestKernel(ComputeKernelLibrary.Dense(X, l.datasets[0].shape, O, itemSize >> 2));
                instructions.Add(new CompiledInstruction {kernel = kernel, shape = O});

                if (ShouldFlattenInputForDenseLayer(X))
                {
                    var flattenedShape = X.Flatten();
                    var flattenKernel = BestKernel(ComputeKernelLibrary.ReshapeFromNHWCModel(flattenedShape));
                    instructions.Add(new CompiledInstruction { kernel = flattenKernel, shape = flattenedShape});
                }

                // FusedActivation
                var fusedActivation = (Layer.FusedActivation) l.activation;
                if (!IsFusedActivationSupported(fusedActivation))
                {
                    var activationKernel = BestKernel(ComputeKernelLibrary.Activation(X, O, fusedActivation.ToString()));
                    instructions.Add(new CompiledInstruction { kernel = activationKernel, shape = O });
                }

                m_CompiledLayers.Add(l, new CompiledLayer { instructions = instructions.ToArray(), shape = O });
                continue;
            }
            else if (l.type == Layer.Type.Dense3)
            {
                var instructions = new List<CompiledInstruction>();
                kernel = BestKernel(ComputeKernelLibrary.Dense3(X, l.datasets[0].shape, O));
                instructions.Add(new CompiledInstruction {kernel = kernel, shape = O});

                m_CompiledLayers.Add(l, new CompiledLayer { instructions = instructions.ToArray(), shape = O });
                continue;
            }
            else if (
                l.type == Layer.Type.Conv2D)
            {
                Assert.IsNotNull(l.stride);
                Assert.IsNotNull(l.pad);
                var instructions = new List<CompiledInstruction>();

                // Conv2D
                var kernelConv = BestKernel(ComputeKernelLibrary.Conv2D(X, l.datasets[0].shape, O, l.stride, l.pad));
                bool isConvWinograd = (kernelConv.func.kernelName.StartsWith("Conv2DWinograd"));

                instructions.Add(new CompiledInstruction { kernel = kernelConv, shape = O, tensors = isConvWinograd ? PrepareConv2dWinograd(model, l) : null });

                // FusedActivation
                var fusedActivation = (Layer.FusedActivation) l.activation;
                if (!IsFusedActivationSupported(fusedActivation))
                {
                    var activationKernel = BestKernel(ComputeKernelLibrary.Activation(X, O, fusedActivation.ToString()));
                    instructions.Add(new CompiledInstruction {kernel = activationKernel, shape = O});
                }

                m_CompiledLayers.Add(l, new CompiledLayer { instructions = instructions.ToArray(), shape = O });
                continue;
            }
            else if (
                l.type == Layer.Type.DepthwiseConv2D)
            {
                var instructions = new List<CompiledInstruction>();

                // DepthwiseConv2D
                var kernelDepthwiseConv = BestKernel(
                    ComputeKernelLibrary.DepthwiseConv2D(X, l.datasets[0].shape, O));
                instructions.Add(new CompiledInstruction { kernel = kernelDepthwiseConv, shape = O });

                // FusedActivation
                var fusedActivation = (Layer.FusedActivation) l.activation;
                if (!IsFusedActivationSupported(fusedActivation))
                {
                    var activationKernel = BestKernel(ComputeKernelLibrary.Activation(X, O, fusedActivation.ToString()));
                    instructions.Add(new CompiledInstruction {kernel = activationKernel, shape = O});
                }

                m_CompiledLayers.Add(l, new CompiledLayer { instructions = instructions.ToArray(), shape = O });
                continue;
            }
            else if (
                l.type == Layer.Type.Conv2DTrans)
            {
                    var instructions = new List<CompiledInstruction>();

                    var outputAdjustment = l.pool;
                    var stride = l.stride;

                    var K = l.datasets[0].shape;
                    var B = l.datasets[1].shape;
                    var pad = new int[]
                    {
                            K.kernelWidth - l.pad[0] - 1, K.kernelHeight - l.pad[1] - 1,
                            K.kernelWidth - l.pad[2] - 1, K.kernelHeight - l.pad[3] - 1
                    };

                    var XpaddedShape = new TensorShape(X.batch, stride[1] * (X.height - 1) + 1 + outputAdjustment[1], stride[0] * (X.width - 1) + 1 + outputAdjustment[0], X.channels);

                    var kernelFill = CompileKernel(new ComputeKernelLibrary.Entry("Conv2DTransPadFill", (X.channels, X.width, X.height), 1.0f, 0));

                    var kernelConv = BestKernel(
                        ComputeKernelLibrary.Conv2D(XpaddedShape, K, O, new int[] { 1, 1 }, pad));
                    bool isConvWinograd = (kernelConv.func.kernelName.StartsWith("Conv2DWinograd"));

                    var KBTensors = PrepareConv2DTrans(model, l);

                    instructions.Add(new CompiledInstruction { kernel = kernelFill, shape = XpaddedShape });
                    instructions.Add(new CompiledInstruction { shape = K, tensors = KBTensors });

                    if (isConvWinograd)
                    {
                        var layer = new Layer(l.name, l.type, l.activation);
                        layer.pad = l.pad;
                        layer.stride = l.stride;

                        layer.pool = l.pool.ToArray();
                        layer.axis = l.axis;
                        layer.alpha = l.alpha;
                        layer.beta = l.beta;
                        layer.inputs = l.inputs.ToArray();

                        var Kd = KBTensors[0];
                        var Bd = KBTensors[1];

                        layer.datasets = new Layer.DataSet[2];
                        layer.datasets[0].name = Kd.name;
                        layer.datasets[0].shape = Kd.shape;
                        layer.datasets[0].itemSizeInBytes = 4;
                        layer.datasets[0].length = Kd.length;
                        layer.datasets[0].offset = 0;

                        layer.datasets[1].name = Bd.name;
                        layer.datasets[1].shape = Bd.shape;
                        layer.datasets[1].itemSizeInBytes = 4;
                        layer.datasets[1].length = Bd.length;
                        layer.datasets[1].offset = 0;
                        layer.datasets[1].offset = Bd.length;

                        layer.weights = new float[Kd.length + Bd.length];

                        Array.Copy(Kd.ToReadOnlyArray(), 0, layer.weights, 0, Kd.length);
                        Array.Copy(Bd.ToReadOnlyArray(), 0, layer.weights, Kd.length, Bd.length);

                        instructions.Add(new CompiledInstruction { kernel = kernelConv, shape = O, tensors = PrepareConv2dWinograd(model, layer) });
                    }
                    else
                        instructions.Add(new CompiledInstruction { kernel = kernelConv, shape = O, tensors = null });

                    // FusedActivation
                    var fusedActivation = (Layer.FusedActivation)l.activation;
                    if (!IsFusedActivationSupported(fusedActivation))
                    {
                        var activationKernel = BestKernel(ComputeKernelLibrary.Activation(X, O, fusedActivation.ToString()));
                        instructions.Add(new CompiledInstruction { kernel = activationKernel, shape = O });
                    }

                    m_CompiledLayers.Add(l, new CompiledLayer { instructions = instructions.ToArray(), shape = O });

                    continue;
            }
            else if (l.type == Layer.Type.Upsample2D)
            {
                // axis is treated as upsample point/bilinear flag
                var bilinear = l.axis > 0;
                kernel = BestKernel(
                    ComputeKernelLibrary.Upsample2D(X, O, l.pool, bilinear));
            }
            else if (
                l.type == Layer.Type.MaxPool2D ||
                l.type == Layer.Type.AvgPool2D)
            {
                var kernelName = l.type.ToString();

                Assert.IsNotNull(l.pool);
                Assert.IsNotNull(l.stride);
                Assert.IsNotNull(l.pad);
                kernel = BestKernel(
                    ComputeKernelLibrary.Pool2D(X, O, kernelName));
            }
            else if (
                l.type == Layer.Type.GlobalMaxPool2D ||
                l.type == Layer.Type.GlobalAvgPool2D)
            {
                var poolKernelName = l.type.ToString().Substring(6) + "Reduce";
                var globalKernelName = l.type.ToString();

                var instructions = new List<CompiledInstruction>();
                var Xr = X;
                while (Xr.height * Xr.width >= 8*8*2*2)
                {
                    var lastLength = Xr.length;
                    var pool = new[] { 8, 8 };
                    var stride = pool;
                    var pad = new[] { 0, 0, 0, 0 };

                    var Oshape = Xr.ApplyPool(pool, stride, pad, ceilMode: true);
                    var Or = new TensorShape(Oshape.batch, ComputeHelper.IDivC(Oshape.height, 2), ComputeHelper.IDivC(Oshape.width, 2), Oshape.channels);
                    var poolKernel = BestKernel(
                        ComputeKernelLibrary.Pool2DReduce(Xr, Or, poolKernelName));

                    instructions.Add(new CompiledInstruction { kernel = poolKernel, shape = Or });

                    Xr = Or;
                    Assert.IsTrue(Xr.length < lastLength);
                }

                var globalKernel = BestKernel(
                    ComputeKernelLibrary.GlobalPool2D(Xr, O, globalKernelName));

                instructions.Add(new CompiledInstruction { kernel = globalKernel, shape = O });

                m_CompiledLayers.Add(l, new CompiledLayer { instructions = instructions.ToArray(), shape = O });

                continue;
            }
            else if (
                l.type == Layer.Type.ScaleBias)
            {
                kernel = BestKernel(
                    ComputeKernelLibrary.ScaleBias(X, O));
            }
            else if (
                l.type == Layer.Type.Normalization)
            {
                // GlobalAvgVariancePool2D
                var poolKernelName = "AvgVariancePool2DReduce";
                var globalKernelName = "GlobalAvgVariancePool2D";

                var instructions = new List<CompiledInstruction>();
                var Xr = X;
                while (Xr.height * Xr.width >= 8*8*2)
                {
                    var lastLength = Xr.length;
                    var pool = new[] { 8, 8 };
                    var stride = pool;
                    var pad = new[] { 0, 0, 0, 0 };

                    var Oshape = Xr.ApplyPool(pool, stride, pad, ceilMode: true);
                    var Or = new TensorShape(Oshape.batch, ComputeHelper.IDivC(Oshape.height, 2), ComputeHelper.IDivC(Oshape.width, 2), Oshape.channels);
                    var poolKernel = BestKernel(
                        ComputeKernelLibrary.PoolAvgVar2D(Xr, Or, poolKernelName));

                    instructions.Add(new CompiledInstruction { kernel = poolKernel, shape = Or });

                    Xr = Or;
                    Assert.IsTrue(Xr.length < lastLength);
                }

                var meanVariance = new TensorShape(Xr.batch, 2, 1, Xr.channels);
                var globalKernel = BestKernel(
                    ComputeKernelLibrary.GlobalPool2D(Xr, meanVariance, globalKernelName));
                instructions.Add(new CompiledInstruction { kernel = globalKernel, shape = meanVariance });

                // ScaleBias
                var S = l.datasets[0].shape;
                var B = l.datasets[1].shape;
                Assert.AreEqual(X.channels, B.channels); Assert.AreEqual(X.channels, S.channels);
                Assert.AreEqual(B.length, B.channels); Assert.AreEqual(S.length, S.channels);
                var normlizationKernel = BestKernel(ComputeKernelLibrary.NormalizationTail(X, O));
                instructions.Add(new CompiledInstruction { kernel = normlizationKernel, shape = O });

                // FusedActivation
                var fusedActivation = (Layer.FusedActivation) l.activation;
                if (!IsFusedActivationSupported(fusedActivation))
                {
                    var activationKernel = BestKernel(ComputeKernelLibrary.Activation(X, O, fusedActivation.ToString()));
                    instructions.Add(new CompiledInstruction { kernel = activationKernel, shape = O });
                }
                else
                {
                    instructions.Add(new CompiledInstruction { shape = O });
                }

                m_CompiledLayers.Add(l, new CompiledLayer { instructions = instructions.ToArray(), shape = O });
                continue;
            }
            else if (
                l.type == Layer.Type.Add ||
                l.type == Layer.Type.Sub ||
                l.type == Layer.Type.Mul ||
                l.type == Layer.Type.Div ||
                l.type == Layer.Type.Pow ||
                l.type == Layer.Type.Min ||
                l.type == Layer.Type.Max ||
                l.type == Layer.Type.Mean
                )
            {
                if (X.Is4D() && O.Is4D())
                {
                    var kernelName = "Broadcast" + l.type;
                    kernel = BestKernel(
                        ComputeKernelLibrary.Broadcast(X, O, kernelName));
                }
            }
            else if (
                    l.type == Layer.Type.Concat)
            {
                var instructions = new List<CompiledInstruction>();

                foreach (var input in l.inputs)
                {
                    var I = shapesByName[input];

                    if (I == null)
                    {
                        instructions.Add(new CompiledInstruction {});
                        continue;
                    }
                    var kernelI = BestKernel(ComputeKernelLibrary.Copy(I.Value, O));

                    instructions.Add(new CompiledInstruction { kernel = kernelI, shape = I.Value });
                }

                m_CompiledLayers.Add(l, new CompiledLayer { instructions = instructions.ToArray(), shape = O });
                continue;
            }
            else if (l.type == Layer.Type.ReduceMax ||
                     l.type == Layer.Type.ReduceMean ||
                     l.type == Layer.Type.ReduceMin ||
                     l.type == Layer.Type.ReduceProd ||
                     l.type == Layer.Type.ReduceSum)
            {
                Layer.Type kernelName = l.type;

                int axis = l.axis;
                axis = X.Axis(axis);
                int baseReducedDim = X[axis];

                int flatHeight, reducedDim, flatWidth;
                int unrolledH, unrolledW;

                var instructions = new List<CompiledInstruction>();
                var Xr = X;
                while (Xr[axis] >= 64*4)
                {
                    var lastLength = Xr.length;

                    var Or = Xr;
                    Or[axis] = ComputeHelper.IDivC(ComputeHelper.IDivC(Xr[axis], 64), 4);

                    ComputeReduceDispatchDim(Xr, Or, axis, out flatHeight, out reducedDim, out flatWidth);

                    s_PartialReduceSumDimensions[0] = flatHeight;
                    s_PartialReduceSumDimensions[1] = flatWidth;
                    s_PartialReduceSumDimensions[2] = reducedDim;

                    unrolledH = flatHeight / ((int)ComputeFunc.SafeDispatchLimit) + 1;
                    unrolledW = flatWidth / ((int)ComputeFunc.SafeDispatchLimit) + 1;

                    var poolKernel = BestKernel(ComputeKernelLibrary.PartialReduce(kernelName, flatHeight, reducedDim, flatWidth));

                    instructions.Add(new CompiledInstruction { kernel = poolKernel, shape = Or });

                    Xr = Or;
                    Assert.IsTrue(Xr.length < lastLength);
                }

                ComputeReduceDispatchDim(Xr, O, axis, out flatHeight, out reducedDim, out flatWidth);


                s_GlobalReduceSumDimensions[0] = flatHeight;
                s_GlobalReduceSumDimensions[1] = flatWidth;
                s_GlobalReduceSumDimensions[2] = baseReducedDim;


                unrolledH = flatHeight / ((int)ComputeFunc.SafeDispatchLimit) + 1;
                unrolledW = flatWidth / ((int)ComputeFunc.SafeDispatchLimit) + 1;

                var globalKernel = BestKernel(
                    ComputeKernelLibrary.GlobalReduce(kernelName, flatHeight, reducedDim, flatWidth));

                instructions.Add(new CompiledInstruction { kernel = globalKernel, shape = O });

                m_CompiledLayers.Add(l, new CompiledLayer { instructions = instructions.ToArray(), shape = O });

                continue;
            }
            // Activations
            else if (l.type == Layer.Type.Activation)
            {
                if (!X.Is4D())
                    //8D activation are not supported on compute path atm, will fallback.
                    continue;

                if (l.activation == Layer.Activation.Softmax)
                {
                    kernel = BestKernel(
                        ComputeKernelLibrary.Softmax(X, O));
                } else if (l.activation == Layer.Activation.LogSoftmax)
                {
                    kernel = BestKernel(
                        ComputeKernelLibrary.LogSoftmax(X, O));
                }
                else if (l.activation == Layer.Activation.PRelu)
                {
                    kernel = BestKernel(
                        ComputeKernelLibrary.PRelu(X, O));
                }
                else if (l.activation != Layer.Activation.None)
                {
                    try
                    {
                        var kernelName = l.activation.ToString();
                        kernel = BestKernel(
                            ComputeKernelLibrary.Activation(X, O, kernelName));
                    }
                    catch (System.ArgumentException)
                    {
                        //Not all activation are supported on compute path, some will fallback.
                        continue;
                    }
                }
            }

            m_CompiledLayers.Add(l, new CompiledLayer { instructions = new CompiledInstruction[]
            {
                new CompiledInstruction { kernel = kernel, shape = O }
            }, shape = O });
        }
    }

    /// <inheritdoc/>
    public virtual void PreExecuteLayer(Layer layer, Tensor[] inputs)
    {
        m_Compiled = new CompiledLayer();
        m_CompiledLayers.TryGetValue(layer, out m_Compiled);
    }

    // ---------------------------------------------------------------------------------
    private Tensor ApplyUnsupportedFusedActivationIfNeeded(Layer.FusedActivation fusedActivation, Tensor O)
    {
        if (!IsFusedActivationSupported(fusedActivation))
        {
            CompiledInstruction instructionActivation = m_Compiled.instructions[m_Compiled.instructions.Length - 1];
            Assert.IsNotNull(instructionActivation.kernel.shader);

            var fnActivation = instructionActivation.kernel;
            var Oactivation = NewTensor(O.shape);

            fnActivation.SetTensor("X", O.shape, Pin(O).buffer);
            fnActivation.SetTensor("O", Oactivation.shape, Pin(Oactivation).buffer);

            fnActivation.shader.SetFloat(_Alpha, 0.0f);
            fnActivation.shader.SetFloat(_Beta, 0.0f);

            fnActivation.Dispatch();
            return Oactivation;
        }

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Dense(Tensor X, Tensor W, Tensor B, Layer.FusedActivation fusedActivation)
    {
        if (m_Compiled.kernel.shader == null)
            return base.Dense(X, W, B, fusedActivation);

        Assert.IsTrue(W.dimensions <= 2);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(X.flatWidth, W.flatHeight);

        if (ShouldFlattenInputForDenseLayer(X.shape))
        {
            Assert.IsNotNull(m_Compiled.instructions[1].kernel.shader);
            var flattenedX = NewTensor(m_Compiled.instructions[1].shape);
            var flattenFn = m_Compiled.instructions[1].kernel;

            flattenFn.SetTensor(_DeclX, _DataX, X.shape, Pin(X).buffer);
            flattenFn.SetTensor(_DeclO, _DataO, flattenedX.shape, Pin(flattenedX).buffer);
            flattenFn.Dispatch();

            X = flattenedX;
        }

        Assert.IsNotNull(m_Compiled.kernel.shader);
        var O = NewTensor(m_Compiled.shape);
        var fn = m_Compiled.kernel;

        fn.SetTensor(_DeclX, _DataX, X.shape, Pin(X).buffer);
        fn.SetTensor(_DeclO, _DataO, O.shape, Pin(O).buffer);
        fn.SetTensorDecl(_DeclW, W.shape, Pin(W).offset);
        fn.SetTensorDecl(_DeclB, B.shape, Pin(B).offset);
        Assert.AreEqual(Pin(W).buffer, Pin(B).buffer);
        fn.SetTensorBuffer(_DataWBK, Pin(W).buffer);
        fn.shader.SetInt("_ActivationMode", (int)fusedActivation);

        fn.Dispatch();

        return ApplyUnsupportedFusedActivationIfNeeded(fusedActivation, O);
    }

    /// <inheritdoc/>
    public override Tensor Dense3(Tensor X, Tensor W, Tensor B)
    {
        if (m_Compiled.kernel.shader == null)
            return base.Dense3(X, W, B);

        Assert.IsNotNull(m_Compiled.kernel.shader);
        var O = NewTensor(m_Compiled.shape);
        var fn = m_Compiled.kernel;

        fn.SetTensor(_DeclX, _DataX, X.shape, Pin(X).buffer);
        fn.SetTensor(_DeclO, _DataO, O.shape, Pin(O).buffer);
        fn.SetTensorDecl(_DeclW, W.shape, Pin(W).offset);
        fn.SetTensorDecl(_DeclB, B.shape, Pin(B).offset);
        Assert.AreEqual(Pin(W).buffer, Pin(B).buffer);
        fn.SetTensorBuffer(_DataWBK, Pin(W).buffer);

        fn.Dispatch();

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Conv2D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, Layer.FusedActivation fusedActivation)
    {
        if (m_Compiled.kernel.shader == null)
            return base.Conv2D(X, K, B, stride, pad, fusedActivation);

        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(X.channels, K.kernelDepth);
        Assert.AreEqual(K.kernelCount, B.flatWidth);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);

        Assert.IsNotNull(m_Compiled.kernel.shader);
        var O = NewTensor(m_Compiled.shape);
        var fn = m_Compiled.kernel;

        fn.SetTensor(_DeclX, _DataX, X.shape, Pin(X).buffer);
        fn.SetTensor(_DeclO, _DataO, O.shape, Pin(O).buffer);

        if (m_Compiled.instructions[0].tensors?.Length == 2)
        {
            K = m_Compiled.instructions[0].tensors[0];
            B = m_Compiled.instructions[0].tensors[1];
        }

        fn.SetTensorDecl(_DeclK, K.shape, Pin(K).offset);
        fn.SetTensorDecl(_DeclB, B.shape, Pin(B).offset);
        Assert.AreEqual(Pin(K).buffer, Pin(B).buffer);
        fn.SetTensorBuffer(_DataWBK, Pin(K).buffer);

        fn.shader.SetInts(_Pad, pad);
        fn.shader.SetInts(_Stride, stride);
        fn.shader.SetInt("_ActivationMode", (int)fusedActivation);

        fn.Dispatch();

        return ApplyUnsupportedFusedActivationIfNeeded(fusedActivation, O);
    }

    /// <inheritdoc/>
    public override Tensor DepthwiseConv2D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, Layer.FusedActivation fusedActivation)
    {
        if (K.kernelDepth != 1 || m_Compiled.kernel.shader == null)
            return base.DepthwiseConv2D(X, K, B, stride, pad, fusedActivation);

        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(K.kernelDepth, 1);
        Assert.AreEqual(K.kernelCount, X.channels);
        Assert.AreEqual(K.kernelCount, B.flatWidth);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);

        Assert.IsNotNull(m_Compiled.kernel.shader);
        var O = NewTensor(m_Compiled.shape);
        var fn = m_Compiled.kernel;

        fn.SetTensor(_DeclX, _DataX, X.shape, Pin(X).buffer);
        fn.SetTensor(_DeclO, _DataO, O.shape, Pin(O).buffer);
        fn.SetTensorDecl(_DeclK, K.shape, Pin(K).offset);
        fn.SetTensorDecl(_DeclB, B.shape, Pin(B).offset);
        Assert.AreEqual(Pin(K).buffer, Pin(B).buffer);
        fn.SetTensorBuffer(_DataWBK, Pin(K).buffer);

        fn.shader.SetInts(_Pad, pad);
        fn.shader.SetInts(_Stride, stride);
        fn.shader.SetInt("_ActivationMode", (int)fusedActivation);

        fn.Dispatch();

        return ApplyUnsupportedFusedActivationIfNeeded(fusedActivation, O);
    }

    /// <inheritdoc/>
    public override Tensor Conv2DTrans(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, int[] outputAdjustment, Layer.FusedActivation fusedActivation)
    {
        if (m_Compiled.instructions == null)
            return base.Conv2DTrans(X, K, B, stride, pad, outputAdjustment, fusedActivation);

        Assert.IsTrue(m_Compiled.instructions.Length >= 3); // pad, kernel flip, conv, ? fusedActivation

        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(X.channels, K.kernelDepth);
        Assert.AreEqual(K.kernelCount, B.flatWidth);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);

        // refer to BarracudaCompute.cs for details
        // 0-pad X
        CompiledInstruction instruction0PadX = m_Compiled.instructions[0];
        Assert.IsNotNull(instruction0PadX.kernel.shader);

        var XpaddedShape = instruction0PadX.shape;
        var Xpadded = NewTensor(XpaddedShape);
        var fn0PadX = instruction0PadX.kernel;

        fn0PadX.SetTensor("X", X.shape, Pin(X).buffer);
        fn0PadX.SetTensor("O", Xpadded.shape, Pin(Xpadded).buffer);
        fn0PadX.shader.SetInts("_Stride", stride);
        fn0PadX.shader.SetInts("_Pad", outputAdjustment);
        fn0PadX.Dispatch();

        // kernel flip
        CompiledInstruction instructionKernelFlip = m_Compiled.instructions[1];
        Assert.IsTrue(instructionKernelFlip.tensors.Length >= 2);
        var Kflipped = instructionKernelFlip.tensors[0];
        var Bpacked = instructionKernelFlip.tensors[1];

        // convolution
        CompiledInstruction instructionConv = m_Compiled.instructions[2];
        Assert.IsNotNull(instructionConv.kernel.shader);
        var fnConv = instructionConv.kernel;

        var padTrans = new int[]
        {
            K.kernelWidth - pad[0] - 1, K.kernelHeight - pad[1] - 1,
            K.kernelWidth - pad[2] - 1, K.kernelHeight - pad[3] - 1
        };
        var strideTrans = new int[] { 1, 1 };

        if (fnConv.shader == null)
        {
            return base.Conv2D(Xpadded, Kflipped, Bpacked, strideTrans, padTrans, fusedActivation);
        }

        Assert.IsNotNull(fnConv.shader);

        var O = NewTensor(instructionConv.shape);

        fnConv.SetTensor("X", Xpadded.shape, Pin(Xpadded).buffer);
        fnConv.SetTensor(_DeclO, _DataO, O.shape, Pin(O).buffer);

        if(instructionConv.tensors?.Length == 2)
        {
            Kflipped = instructionConv.tensors[0];
            Bpacked = instructionConv.tensors[1];
        }

        fnConv.SetTensorDecl(_DeclK, Kflipped.shape, Pin(Kflipped).offset);
        fnConv.SetTensorDecl(_DeclB, Bpacked.shape, Pin(Bpacked).offset);
        Assert.AreEqual(Pin(Kflipped).buffer, Pin(Bpacked).buffer);
        fnConv.SetTensorBuffer(_DataWBK, Pin(Kflipped).buffer);

        fnConv.shader.SetInt("_ActivationMode", (int)fusedActivation);
        fnConv.shader.SetInts(_Pad, padTrans);
        fnConv.shader.SetInts(_Stride, strideTrans);

        fnConv.Dispatch();

        Xpadded.Dispose();

        return ApplyUnsupportedFusedActivationIfNeeded(fusedActivation, O);
    }

    /// <inheritdoc/>
    public override Tensor Upsample2D(Tensor X, int[] scale, bool bilinear)
    {
        if (m_Compiled.kernel.shader == null)
            return base.Upsample2D(X, scale, bilinear);

        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(scale.Length, 2);

        Assert.IsNotNull(m_Compiled.kernel.shader);
        var O = NewTensor(m_Compiled.shape);
        var fn = m_Compiled.kernel;

        fn.SetTensor(_DeclX, _DataX, X.shape, Pin(X).buffer);
        fn.SetTensor(_DeclO, _DataO, O.shape, Pin(O).buffer);

        fn.shader.SetInts(_Pool, scale);

        fn.Dispatch();
        return O;
    }

    /// <inheritdoc/>
    protected override Tensor Pool2D(string kernelName, Tensor X, int[] pool, int[] stride, int[] pad)
    {
        if (m_Compiled.kernel.shader == null)
            return base.Pool2D(kernelName, X, pool, stride, pad);

        Assert.AreEqual(pool.Length, 2);
        Assert.AreEqual(stride.Length, 2);

        Assert.IsNotNull(m_Compiled.kernel.shader);
        var O = NewTensor(m_Compiled.shape);
        var fn = m_Compiled.kernel;

        fn.SetTensor(_DeclX, _DataX, X.shape, Pin(X).buffer);
        fn.SetTensor(_DeclO, _DataO, O.shape, Pin(O).buffer);

        fn.shader.SetInts(_Pool, pool);
        fn.shader.SetInts(_Stride, stride);
        fn.shader.SetInts(_Pad, pad);

        fn.Dispatch();
        return O;
    }

    /// <inheritdoc/>
    public override Tensor ScaleBias(Tensor X, Tensor S, Tensor B)
    {
        if (m_Compiled.kernel.shader == null || !X.shape.Is4D())
            return base.ScaleBias(X, S, B);

        Assert.AreEqual(X.channels, B.channels); Assert.AreEqual(X.channels, S.channels);
        Assert.AreEqual(B.length, B.channels); Assert.AreEqual(S.length, S.channels);

        Assert.IsNotNull(m_Compiled.kernel.shader);
        var O = NewTensor(m_Compiled.shape);
        var fn = m_Compiled.kernel;

        fn.SetTensor(_DeclX, _DataX, X.shape, Pin(X).buffer);
        fn.SetTensor(_DeclO, _DataO, O.shape, Pin(O).buffer);
        fn.SetTensorDecl(_DeclW, S.shape, Pin(S).offset);
        fn.SetTensorDecl(_DeclB, B.shape, Pin(B).offset);
        Assert.AreEqual(Pin(S).buffer, Pin(B).buffer);
        fn.SetTensorBuffer(_DataWBK, Pin(S).buffer);

        fn.Dispatch();
        return O;
    }


    private Tensor GlobalPool2D(Tensor X)
    {
        Assert.IsTrue(X.shape.Is4D());
        s_GlobalPool2DInputDim[0] = X.height;
        s_GlobalPool2DInputDim[1] = X.width;
        for (var i = 0; i < m_Compiled.instructions.Length-1; ++i)
        {
            var pool = new[] { 8, 8 };
            var stride = pool;
            var pad = new[] { 0, 0, 0, 0 };

            CompiledInstruction instructionPool = m_Compiled.instructions[i];
            Assert.IsNotNull(instructionPool.kernel.shader);

            var Or = NewTensor(instructionPool.shape);
            var fnPool = instructionPool.kernel;

            fnPool.SetTensor("X", X.shape, Pin(X).buffer);
            fnPool.SetTensor("O", Or.shape, Pin(Or).buffer);

            fnPool.shader.SetInts("_Pool", pool);
            fnPool.shader.SetInts("_Stride", stride);
            fnPool.shader.SetInts("_Pad", pad);

            fnPool.Dispatch();
            X = Or;
        }

        CompiledInstruction instructionGlobalPool = m_Compiled.instructions[m_Compiled.instructions.Length - 1];
        Assert.IsNotNull(instructionGlobalPool.kernel.shader);

        var O = NewTensor(instructionGlobalPool.shape);
        var fnGlobalPool = instructionGlobalPool.kernel;

        fnGlobalPool.SetTensor("X", X.shape, Pin(X).buffer);
        fnGlobalPool.SetTensor("O", O.shape, Pin(O).buffer);
        fnGlobalPool.shader.SetInts("_Pool", s_GlobalPool2DInputDim);

        fnGlobalPool.Dispatch();
        return O;
    }

    /// <inheritdoc/>
    public override Tensor GlobalMaxPool2D(Tensor X)
    {
        if (m_Compiled.instructions == null)
            return base.GlobalMaxPool2D(X);

        return GlobalPool2D(X);
    }

    /// <inheritdoc/>
    public override Tensor GlobalAvgPool2D(Tensor X)
    {
        if (m_Compiled.instructions == null)
            return base.GlobalAvgPool2D(X);

        return GlobalPool2D(X);
    }

    /// <inheritdoc/>
    public override Tensor Normalization(Tensor X, Tensor S, Tensor B, int pool, int axis, float epsilon, Layer.FusedActivation fusedActivation)
    {
        if (!X.shape.Is4D())
            throw new NotImplementedException();

        if (axis != TensorShape.C && axis != -1)
            throw new NotImplementedException();

        if (pool <= 0)
            pool = X.batch;

        if (pool > 1)
            throw new NotImplementedException(); // @TODO: support other types of Normalization at test time
                                                 // Currently supported only pool=1 (InstanceNormalization)

        // [0,N] : AvgVariancePool2DReduce
        // N+1 : GlobalAvgVariancePool2D
        // N+2: Normalize
        // N+3 Activation

        var inputDim = new[] { X.height, X.width };

        var Xr = X;
        var X2r = X;
        bool isFirstDispatch = true;
        for (var i = 0; i < m_Compiled.instructions.Length - 3; ++i)
        {
            var poolReduce = new[] { 8, 8 };
            var stride = poolReduce;
            var pad = new[] { 0, 0, 0, 0 };

            CompiledInstruction instructionPool = m_Compiled.instructions[i];
            Assert.IsNotNull(instructionPool.kernel.shader);

            var Or = NewTensor(instructionPool.shape);
            var O2r = NewTensor(instructionPool.shape);
            var fnPool = instructionPool.kernel;

            fnPool.SetTensor("X", Xr.shape, Pin(Xr).buffer);
            fnPool.SetTensor("X2", X2r.shape, Pin(X2r).buffer);
            fnPool.SetTensor("O", Or.shape, Pin(Or).buffer);
            fnPool.SetTensor("O2", O2r.shape, Pin(O2r).buffer);

            fnPool.shader.SetInts("_Pool", poolReduce);
            fnPool.shader.SetInts("_Stride", stride);
            fnPool.shader.SetInts("_Pad", pad);
            fnPool.shader.SetInt("_IsFirstDispatch", isFirstDispatch ? 1 : 0);

            fnPool.Dispatch();

            Xr = Or;
            X2r = O2r;
            isFirstDispatch = false;
        }

        CompiledInstruction instructionGlobalPool = m_Compiled.instructions[m_Compiled.instructions.Length - 3];
        Assert.IsNotNull(instructionGlobalPool.kernel.shader);

        var meanVariance = NewTensor(instructionGlobalPool.shape);
        var fnGlobalPool = instructionGlobalPool.kernel;

        fnGlobalPool.SetTensor("X", Xr.shape, Pin(Xr).buffer);
        fnGlobalPool.SetTensor("X2", X2r.shape, Pin(X2r).buffer);
        fnGlobalPool.SetTensor("O", meanVariance.shape, Pin(meanVariance).buffer);
        fnGlobalPool.shader.SetInts("_Pool", inputDim);
        fnGlobalPool.shader.SetInt("_IsFirstDispatch", isFirstDispatch ? 1 : 0);

        fnGlobalPool.Dispatch();

        CompiledInstruction instructionNormalize = m_Compiled.instructions[m_Compiled.instructions.Length - 2];
        Assert.IsNotNull(instructionNormalize.kernel.shader);
        Assert.AreEqual(X.channels, B.channels); Assert.AreEqual(X.channels, S.channels);
        Assert.AreEqual(B.length, B.channels); Assert.AreEqual(S.length, S.channels);

        var O = NewTensor(X.shape);
        var fnNormalize = instructionNormalize.kernel;
        fnNormalize.SetTensor("X", X.shape, Pin(X).buffer);
        fnNormalize.SetTensor("O", O.shape, Pin(O).buffer);
        fnNormalize.SetTensor("W", meanVariance.shape, Pin(meanVariance).buffer);
        fnNormalize.SetTensorDecl("S", S.shape, Pin(S).offset);
        fnNormalize.SetTensorDecl("B", B.shape, Pin(B).offset);
        Assert.AreEqual(Pin(S).buffer, Pin(B).buffer);
        fnNormalize.SetTensorBuffer("WBK", Pin(S).buffer);
        fnNormalize.shader.SetFloat("_Epsilon", epsilon);
        fnNormalize.shader.SetInt("_ActivationMode", (int)fusedActivation);

        fnNormalize.Dispatch();

        return ApplyUnsupportedFusedActivationIfNeeded(fusedActivation, O);
    }

    internal override Tensor Reduce(Layer.Type kernelName, Tensor X, int axis)
    {
        if (m_Compiled.instructions == null)
            return base.Reduce(kernelName, X, axis);

        axis = X.shape.Axis(axis);
        int baseReducedDim = X.shape[axis];

        int flatHeight, reducedDim, flatWidth;
        int unrolledH, unrolledW;

        for (var i = 0; i < m_Compiled.instructions.Length-1; ++i)
        {
            CompiledInstruction instructionPool = m_Compiled.instructions[i];
            Assert.IsNotNull(instructionPool.kernel.shader);

            ComputeReduceDispatchDim(X.shape, instructionPool.shape, axis, out flatHeight, out reducedDim, out flatWidth);

            s_PartialReduceSumDimensions[0] = flatHeight;
            s_PartialReduceSumDimensions[1] = flatWidth;
            s_PartialReduceSumDimensions[2] = reducedDim;

            unrolledH = flatHeight / ((int)ComputeFunc.SafeDispatchLimit) + 1;
            unrolledW = flatWidth / ((int)ComputeFunc.SafeDispatchLimit) + 1;

            var Or = NewTensor(instructionPool.shape);
            var fnPool = instructionPool.kernel;

            fnPool.SetTensor("X", X.shape, Pin(X).buffer);
            fnPool.SetTensor("O", Or.shape, Pin(Or).buffer);
            fnPool.shader.SetInt("_UnrolledH", unrolledH);
            fnPool.shader.SetInt("_UnrolledW", unrolledW);
            fnPool.shader.SetInt("_ReducedDim", instructionPool.shape[axis]);
            fnPool.shader.SetInts("_Pool", s_PartialReduceSumDimensions);

            fnPool.Dispatch();
            X = Or;
        }

        CompiledInstruction instructionGlobalPool = m_Compiled.instructions[m_Compiled.instructions.Length - 1];
        Assert.IsNotNull(instructionGlobalPool.kernel.shader);

        ComputeReduceDispatchDim(X.shape, instructionGlobalPool.shape, axis, out flatHeight, out reducedDim, out flatWidth);


        s_GlobalReduceSumDimensions[0] = flatHeight;
        s_GlobalReduceSumDimensions[1] = flatWidth;
        s_GlobalReduceSumDimensions[2] = baseReducedDim;


        unrolledH = flatHeight / ((int)ComputeFunc.SafeDispatchLimit) + 1;
        unrolledW = flatWidth / ((int)ComputeFunc.SafeDispatchLimit) + 1;

        var O = NewTensor(instructionGlobalPool.shape);
        var fnGlobalPool = instructionGlobalPool.kernel;

        fnGlobalPool.SetTensor("X", X.shape, Pin(X).buffer);
        fnGlobalPool.SetTensor("O", O.shape, Pin(O).buffer);
        fnGlobalPool.shader.SetInt("_UnrolledH", unrolledH);
        fnGlobalPool.shader.SetInt("_UnrolledW", unrolledW);
        fnGlobalPool.shader.SetInt("_ReducedDim", reducedDim);
        fnGlobalPool.shader.SetInts("_Pool", s_GlobalReduceSumDimensions);

        fnGlobalPool.Dispatch();
        return O;
    }


    /// <inheritdoc/>
    protected override Tensor Activation(string kernelName, Tensor X, float alpha = 0f, float beta = 0f)
    {
        if (m_Compiled.kernel.shader == null)
            return base.Activation(kernelName, X, alpha, beta);

        Assert.IsNotNull(m_Compiled.kernel.shader);
        var O = NewTensor(m_Compiled.shape);
        var fn = m_Compiled.kernel;

        fn.SetTensor(_DeclX, _DataX, X.shape, Pin(X).buffer);
        fn.SetTensor(_DeclO, _DataO, O.shape, Pin(O).buffer);

        fn.shader.SetFloat(_Alpha, alpha);
        fn.shader.SetFloat(_Beta,  beta);

        fn.Dispatch();
        return O;
    }

    /// <inheritdoc/>
    public override Tensor PRelu(Tensor X, Tensor S)
    {
        if (m_Compiled.kernel.shader == null)
            return base.PRelu(X, S);

        Assert.IsTrue((X.flatWidth == S.flatWidth) || (S.flatWidth == 1));

        Assert.IsNotNull(m_Compiled.kernel.shader);
        var O = NewTensor(m_Compiled.shape);
        var fn = m_Compiled.kernel;

        fn.SetTensor(_DeclX, _DataX, X.shape, Pin(X).buffer);
        fn.SetTensor(_DeclO, _DataO, O.shape, Pin(O).buffer);
        fn.SetTensor(_DeclW, _DataW, S.shape, Pin(S).buffer);

        fn.Dispatch();
        return O;
    }

    /// <inheritdoc/>
    public override Tensor Softmax(Tensor X, int axis)
    {
        if (m_Compiled.kernel.shader == null || X.shape.sequenceLength != 1 || X.shape.numberOfDirections != 1 || axis > X.shape.FirstNotIdentityFeatureDimensionIndex())
            return base.Softmax(X, axis);

        Assert.IsNotNull(m_Compiled.kernel.shader);
        var O = NewTensor(m_Compiled.shape);
        var fn = m_Compiled.kernel;

        fn.SetTensor(_DeclX, _DataX, X.shape, Pin(X).buffer);
        fn.SetTensor(_DeclO, _DataO, O.shape, Pin(O).buffer);

        fn.Dispatch();
        return O;
    }

    /// <inheritdoc/>
    public override Tensor LogSoftmax(Tensor X)
    {
        if (m_Compiled.kernel.shader == null)
            return base.LogSoftmax(X);

        Assert.IsNotNull(m_Compiled.kernel.shader);
        var O = NewTensor(m_Compiled.shape);
        var fn = m_Compiled.kernel;

        fn.SetTensor(_DeclX, _DataX, X.shape, Pin(X).buffer);
        fn.SetTensor(_DeclO, _DataO, O.shape, Pin(O).buffer);

        fn.Dispatch();
        return O;
    }

    /// <inheritdoc/>
    protected override Tensor ElementwiseWithBroadcast(string kernelName, Tensor[] tensors)
    {
        if (m_Compiled.kernel.shader == null)
            return base.ElementwiseWithBroadcast(kernelName, tensors);

        Assert.IsNotNull(m_Compiled.kernel.shader);
        var fn = m_Compiled.kernel;

        Assert.IsTrue(tensors.Length > 0);
        var X = tensors[0];

        Tensor outputTensor1 = NewTensor(TensorExtensions.MaxShape(tensors));
        Tensor outputTensor2 = null;
        if (tensors.Length > 2)
            outputTensor2 = NewTensor(TensorExtensions.MaxShape(tensors));

        Tensor O = null;
        bool isFirstDispatch = true;
        for (int t = 1; t < tensors.Length; ++t)
        {
            var B = tensors[t];
            O = (t % 2 == 1) ? outputTensor1 : outputTensor2;

            fn.SetTensor(_DeclX, _DataX, X.shape, Pin(X).buffer);
            fn.SetTensor(_DeclO, _DataO, O.shape, Pin(O).buffer);
            fn.SetTensor(_DeclB, _DataB, B.shape, Pin(B).buffer, Pin(B).offset);
            fn.shader.SetFloat("_Alpha", 1.0f/(float)tensors.Length);
            fn.shader.SetInt("_IsFirstDispatch", isFirstDispatch ? 1 : 0);
            fn.shader.SetInts("_XStrides", GetInputTensorStridesOnDevice(X.shape, Pin(X).channelsOrder, s_XStrides));
            fn.shader.SetInts("_BStrides", GetInputTensorStridesOnDevice(B.shape, Pin(B).channelsOrder, s_BStrides));

            fn.Dispatch();

            X = O;
            isFirstDispatch = false;
        }

        if (O != outputTensor1) outputTensor1.Dispose();
        if (O != outputTensor2) outputTensor2?.Dispose();
        return O;
    }

    /// <inheritdoc/>
    public override Tensor Concat(Tensor[] tensors, int axis)
    {
        if (!TensorExtensions.AreAllTensorsConvertibleTo4D(tensors) || !TensorExtensions.Is8DAxisConvertibleTo4D(axis))
            return base.Concat(tensors, axis);

        if (m_Compiled.instructions == null)
            return base.Concat(tensors, axis);

        bool canUsePrecompiledBackend = true;
        foreach (var i in m_Compiled.instructions)
        {
            canUsePrecompiledBackend &= (i.kernel.shader != null);
        }
        foreach (var inputTensor  in tensors)
        {
            //input tensor is not in current memory layout, we need an extra transpose/dispatch
            if (ComputeInfo.channelsOrder == ComputeInfo.ChannelsOrder.NCHW && Pin(inputTensor).channelsOrder == ComputeInfo.ChannelsOrder.NHWC)
                canUsePrecompiledBackend = false;
        }
        if (!canUsePrecompiledBackend)
            return base.Concat(tensors, axis);

        var O = NewTensor(m_Compiled.shape);

        var offsets = s_ConcatOffsets;
        Array.Clear(offsets, 0, offsets.Length);
        axis = O.shape.Axis(axis);
        var axisNCHW = TensorExtensions.Convert8DAxisTo4D(axis);

        Assert.AreEqual(tensors.Length, m_Compiled.instructions.Length);
        for (int i = 0; i < tensors.Length; ++i)
        {
            var X = tensors[i];
            var instruction = m_Compiled.instructions[i];
            var fn = instruction.kernel;

            fn.SetTensor("X", X.shape, Pin(X).buffer);
            fn.SetTensor("O", O.shape, Pin(O).buffer);

            fn.shader.SetInts("_Pad", offsets);

            fn.Dispatch();

            offsets[axisNCHW] += X.shape[axis];
        }

        return O;
    }
}

} // namespace Unity.Barracuda
