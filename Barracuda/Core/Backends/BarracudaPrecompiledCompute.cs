using UnityEngine;
using UnityEngine.Assertions;
using System;
using System.Linq;
using System.Collections.Generic;


namespace Barracuda {


public class PrecompiledComputeOps : ComputeOps, IModelCompiler
{
    public PrecompiledComputeOps(ComputeShader[] kernels, ComputeShader referenceKernel,  ITensorAllocator allocator = null, bool verbose = false)
    : base(kernels, referenceKernel, allocator, verbose)
    {
    }

    // ---------------------------------------------------------------------------------

    static public ComputeFunc.TensorDecl _DeclX = ComputeFunc.GetTensorDecl("X");
    static public ComputeFunc.TensorDecl _DeclO = ComputeFunc.GetTensorDecl("O");
    static public ComputeFunc.TensorDecl _DeclW = ComputeFunc.GetTensorDecl("W");
    static public ComputeFunc.TensorDecl _DeclK = ComputeFunc.GetTensorDecl("K");
    static public ComputeFunc.TensorDecl _DeclB = ComputeFunc.GetTensorDecl("B");
    static public int _DataX = ComputeFunc.GetTensorData("X");
    static public int _DataO = ComputeFunc.GetTensorData("O");
    static public int _DataW = ComputeFunc.GetTensorData("W");
    static public int _DataK = ComputeFunc.GetTensorData("K");
    static public int _DataB = ComputeFunc.GetTensorData("B");
    static public int _DataWBK = ComputeFunc.GetTensorData("WBK");
    static public int _Stride = Shader.PropertyToID("_Stride");
    static public int _Pad = Shader.PropertyToID("_Pad");
    static public int _Pool = Shader.PropertyToID("_Pool");
    static public int _Alpha = Shader.PropertyToID("_Alpha");
    static public int _Beta  = Shader.PropertyToID("_Beta");

    struct CompiledLayer
    {
        public ComputeKernel kernel;
        public TensorShape shape;
    }

    private int m_CachedModelHash;
    private Dictionary<Layer, CompiledLayer> m_CompiledLayers = new Dictionary<Layer, CompiledLayer>();
    private CompiledLayer m_Compiled;

    protected int CalcModelWithInputsHashCode(Model model, IDictionary<string, TensorShape> inputShapes)
    {
        var hash = model.GetHashCode();
        foreach (var entry in inputShapes)
        {
            hash = (hash * 7) + entry.Key.GetHashCode();
            hash = (hash * 7) + entry.Value.GetHashCode();
        }
        return hash;
    }

    public virtual void PrepareModel(Model model, IDictionary<string, TensorShape> inputShapes)
    {
        var modelHash = CalcModelWithInputsHashCode(model, inputShapes);
        if (modelHash == m_CachedModelHash)
            return;

        m_CachedModelHash = modelHash;
        m_CompiledLayers.Clear();

        IDictionary<string, TensorShape> shapesByName;
        ModelAnalyzer.ListTemporaryTensorShapes(model, inputShapes, out shapesByName);

        foreach (var l in model.layers)
        {
            if (m_CompiledLayers.ContainsKey(l))
                continue; // already compiled

            if (l.inputs.Length == 0)
                continue;   // don't need to compile layers without inputs, so far all of them are CPU only

            var X = shapesByName[l.inputs[0]];
            var O = shapesByName[l.name];

            ComputeKernel kernel = new ComputeKernel();
            if (l.type == Layer.Type.Dense)
            {
                var itemSize = 4; // @TODO: itemSizeInBytes == 2 | float16
                kernel = BestKernel(
                    ComputeKernelLibrary.Dense(X, l.datasets[0].shape, O, itemSize >> 2));
            }
            else if (
                l.type == Layer.Type.Conv2D)
            {
                Assert.IsNotNull(l.stride);
                Assert.IsNotNull(l.pad);
                kernel = BestKernel(
                    ComputeKernelLibrary.Conv2D(X, l.datasets[0].shape, O, l.stride, l.pad));
            }
            else if (
                l.type == Layer.Type.DepthwiseConv2D)
            {
                kernel = BestKernel(
                    ComputeKernelLibrary.DepthwiseConv2D(X, l.datasets[0].shape, O));
            }
            else if (
                l.type == Layer.Type.Conv2DTrans)
            {
                kernel = BestKernel(
                    ComputeKernelLibrary.Conv2DTrans(X, l.datasets[0].shape, O));
            }
            else if (
                l.type == Layer.Type.Upsample2D)
            {
                kernel = BestKernel(
                    ComputeKernelLibrary.Upsample2D(X, O));
            }
            else if (
                l.type == Layer.Type.MaxPool2D ||
                l.type == Layer.Type.AvgPool2D)
            {
                var kernelName = l.type.ToString();

                Assert.IsNotNull(l.pool);
                Assert.IsNotNull(l.stride);
                Assert.IsNotNull(l.pad);
                var pad = X.AdjustPadToPool(l.pool, l.stride, l.pad);
                if (pad[0] == 0 && pad[1] == 0 && pad[2] == 0 && pad[3] == 0)
                    kernelName += "_NoPads";

                kernel = BestKernel(
                    ComputeKernelLibrary.Pool2D(X, O, kernelName));
            }
            // @TODO: reimplement GlobalPools, currently require different kernels for each pyramid step
            //else if (
            //    l.type == Layer.Type.GlobalMaxPool2D ||
            //    l.type == Layer.Type.GlobalAvgPool2D)
            //{
            //    var kernelName = l.type.ToString();
            //    kernel = BestKernel(
            //        ComputeKernelLibrary.GlobalPool2D(X, O, kernelName));
            //}
            else if (
                l.type == Layer.Type.ScaleBias)
            {
                kernel = BestKernel(
                    ComputeKernelLibrary.ScaleBias(X, O));
            }
            // @TODO: reimplement Normalization, which became a multi-kernel operation after optimizations
            //else if (
            //    l.type == Layer.Type.Normalization)
            //{
            //    kernel = BestKernel(
            //        ComputeKernelLibrary.Normalization(X, O));
            //}
            else if (
                l.type == Layer.Type.Add ||
                l.type == Layer.Type.Sub ||
                l.type == Layer.Type.Mul ||
                l.type == Layer.Type.Div ||
                l.type == Layer.Type.Pow ||
                l.type == Layer.Type.Min ||
                l.type == Layer.Type.Max
                // || l.type == Layer.Type.Mean @TODO: implement BroadcastMean
                )
            {
                var kernelName = "Broadcast" + l.type;
                kernel = BestKernel(
                    ComputeKernelLibrary.Broadcast(X, O, kernelName));
            }
            // @TODO: implement Concat, currently might require different kernel for each tensor
            //else if (
            //    l.type == Layer.Type.Concat) {}
            // Activations
            else if (l.type == Layer.Type.Activation)
            {
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
                    var kernelName = l.activation.ToString();
                    kernel = BestKernel(
                        ComputeKernelLibrary.Activation(X, O, kernelName));
                }
            }
            
            m_CompiledLayers.Add(l, new CompiledLayer { kernel = kernel, shape = O });
        }
    }

    public virtual void PreExecuteLayer(Layer layer, Tensor[] inputs)
    {
        m_Compiled = new CompiledLayer();
        m_CompiledLayers.TryGetValue(layer, out m_Compiled);
    }

    // ---------------------------------------------------------------------------------

    public override Tensor Dense(Tensor X, Tensor W, Tensor B)
    {
        if (m_Compiled.kernel.shader == null)
            return base.Dense(X, W, B);

        Assert.IsTrue(W.dimensions <= 2);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(X.flatWidth, W.flatHeight);

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

    public override Tensor Conv2D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad)
    {
        if (m_Compiled.kernel.shader == null ||
            m_Compiled.kernel.func.kernelName == "Conv2DWinograd_2x2_3x3") // currently Winograd requires 2 dispatches and can not be supported by Precompiled path
            return base.Conv2D(X, K, B, stride, pad);

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
        fn.SetTensorDecl(_DeclK, K.shape, Pin(K).offset);
        fn.SetTensorDecl(_DeclB, B.shape, Pin(B).offset);
        Assert.AreEqual(Pin(K).buffer, Pin(B).buffer);
        fn.SetTensorBuffer(_DataWBK, Pin(K).buffer);

        fn.shader.SetInts(_Pad, pad);
        fn.shader.SetInts(_Stride, stride);

        fn.Dispatch();
        return O;
    }

    public override Tensor DepthwiseConv2D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad)
    {
        if (K.kernelDepth != 1 || m_Compiled.kernel.shader == null)
            return base.DepthwiseConv2D(X, K, B, stride, pad);

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

        fn.Dispatch();
        return O;
    }

    public override Tensor Conv2DTrans(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, int[] outputAdjustment)
    {
        if (m_Compiled.kernel.shader == null)
            return base.Conv2DTrans(X, K, B, stride, pad, outputAdjustment);

        Assert.AreEqual(X.channels, K.kernelDepth);
        Assert.AreEqual(K.kernelCount, B.flatWidth);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);

        Assert.IsNotNull(m_Compiled.kernel.shader);
        var O = NewTensor(m_Compiled.shape);
        var fn = m_Compiled.kernel;

        pad = new int[]
        {
            K.kernelWidth - pad[0] - 1, K.kernelHeight - pad[1] - 1,
            K.kernelWidth - pad[2] - 1, K.kernelHeight - pad[3] - 1
        };

        fn.SetTensor(_DeclX, _DataX, X.shape, Pin(X).buffer);
        fn.SetTensor(_DeclO, _DataO, O.shape, Pin(O).buffer);
        fn.SetTensorDecl(_DeclK, K.shape, Pin(K).offset);
        fn.SetTensorDecl(_DeclB, B.shape, Pin(B).offset);
        Assert.AreEqual(Pin(K).buffer, Pin(B).buffer);
        fn.SetTensorBuffer(_DataWBK, Pin(K).buffer);

        fn.shader.SetInts(_Pad, pad);
        fn.shader.SetInts(_Stride, stride);

        fn.Dispatch();

        return O;
    }

    public override Tensor Upsample2D(Tensor X, int[] size)
    {
        if (m_Compiled.kernel.shader == null)
            return base.Upsample2D(X, size);

        Assert.AreEqual(size.Length, 2);

        Assert.IsNotNull(m_Compiled.kernel.shader);
        var O = NewTensor(m_Compiled.shape);
        var fn = m_Compiled.kernel;

        fn.SetTensor(_DeclX, _DataX, X.shape, Pin(X).buffer);
        fn.SetTensor(_DeclO, _DataO, O.shape, Pin(O).buffer);

        fn.shader.SetInts(_Pool, size);

        fn.Dispatch();
        return O;
    }

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

    public override Tensor ScaleBias(Tensor X, Tensor S, Tensor B)
    {
        if (m_Compiled.kernel.shader == null)
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

    // @TODO: reimplement Normalization, which became a multi-kernel operation after optimizations
    /*
    public override Tensor Normalization(Tensor X, Tensor S, Tensor B, int pool, int axis)
    {
        if (axis != 3 && axis != -1)
            throw new NotImplementedException();

        if (pool == 1 && X.batch != 1)
            throw new NotImplementedException(); // @TODO: Instance Normalization with batch > 1

        if (pool <= 0)
            pool = X.batch;

        if (pool > 1)
            throw new NotImplementedException(); // @TODO: support other types of Normalization at test time
                                                 // Currently supported only pool=1 (InstanceNormalization)


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
    */

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

    public override Tensor PRelu(Tensor X, Tensor S)
    {
        if (m_Compiled.kernel.shader == null)
            return base.PRelu(X, S);

        Assert.AreEqual(X.channels, S.channels);
        Assert.AreEqual(S.length, S.channels);

        Assert.IsNotNull(m_Compiled.kernel.shader);
        var O = NewTensor(m_Compiled.shape);
        var fn = m_Compiled.kernel;

        fn.SetTensor(_DeclX, _DataX, X.shape, Pin(X).buffer);
        fn.SetTensor(_DeclO, _DataO, O.shape, Pin(O).buffer);
        fn.SetTensor(_DeclW, _DataW, S.shape, Pin(S).buffer);

        fn.Dispatch();
        return O;
    }

    public override Tensor Softmax(Tensor X)
    {
        if (m_Compiled.kernel.shader == null)
            return base.Softmax(X);

        Assert.IsNotNull(m_Compiled.kernel.shader);
        var O = NewTensor(m_Compiled.shape);
        var fn = m_Compiled.kernel;

        fn.SetTensor(_DeclX, _DataX, X.shape, Pin(X).buffer);
        fn.SetTensor(_DeclO, _DataO, O.shape, Pin(O).buffer);

        fn.Dispatch();
        return O;
    }

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

    public override Tensor ElementwiseWithBroadcast(string kernelName, Tensor[] tensors)
    {
        if (m_Compiled.kernel.shader == null)
            return base.ElementwiseWithBroadcast(kernelName, tensors);

        Assert.IsNotNull(m_Compiled.kernel.shader);
        var O = NewTensor(m_Compiled.shape);
        var fn = m_Compiled.kernel;

        Assert.IsTrue(tensors.Length > 0);
        var X = tensors[0];

        for (int t = 1; t < tensors.Length; ++t)
        {
            var B = tensors[t];

            fn.SetTensor(_DeclX, _DataX, X.shape, Pin(X).buffer);
            fn.SetTensor(_DeclO, _DataO, O.shape, Pin(O).buffer);
            fn.SetTensor(_DeclB, _DataB, B.shape, Pin(B).buffer, Pin(B).offset);

            fn.Dispatch();
        }

        return O;
    }
}

} // namespace Barracuda
