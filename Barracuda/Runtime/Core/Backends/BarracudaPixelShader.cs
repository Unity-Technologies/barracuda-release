//#define DEBUG_TRACK_ALLOCATIONS

using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Experimental.Rendering; // AsyncGPUReadback
using UnityEngine.Assertions;
using UnityEngine.Profiling;
using System;
using System.Linq;
using System.Runtime.CompilerServices;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Collections;

[assembly: InternalsVisibleTo("Barracuda.EditorTests")]

namespace Unity.Barracuda {
public class TextureTensorData : UniqueResourceId, ITensorData
{
    private bool m_DisposeBufferAfterUse;
    private TensorShape m_Shape;
    private RenderTexture m_BufferAsTexture;
    private bool m_tensorBatchTilled = false;
    private bool m_tensorChannelTilled = false;

    public RenderTexture bufferAsTexture { get { return m_BufferAsTexture; } }
    public bool tensorBatchTilled { get { return m_tensorBatchTilled; } }
    public bool tensorChannelTilled { get { return m_tensorChannelTilled; } }

    public string name;

    public static int MaxTextureSize = 16384;

    public TextureTensorData(TensorShape shape, string buffername, bool clearOnInit = true)
    {
        name = buffername;

        int c4 = ComputeHelper.IDivC(shape.channels, 4);
        int c4w = c4;
        int c4h = 1;

        if (c4w * shape.width > MaxTextureSize)
        {
            c4w = Mathf.FloorToInt(MaxTextureSize / ((float)shape.width));
            c4h = ComputeHelper.IDivC(c4, c4w);
            m_tensorChannelTilled = true;
        }

        int bh = shape.batch;
        int bw = 1;

        if (bh * c4h * shape.height > MaxTextureSize)
        {
            bh = Mathf.FloorToInt(MaxTextureSize / ((float)(c4h * shape.height)));
            bw = ComputeHelper.IDivC(shape.batch, bh);
            m_tensorBatchTilled = true;
        }

        int h = bh * c4h * shape.height;
        int w = bw * c4w * shape.width;

        m_BufferAsTexture = new RenderTexture(w, h, 0, RenderTextureFormat.ARGBFloat);
        m_BufferAsTexture.Create();

        if (clearOnInit)
        {
            var previousActiveRT = RenderTexture.active;
            RenderTexture.active = m_BufferAsTexture;
            GL.Clear(true, true, Color.clear);
            RenderTexture.active = previousActiveRT;
        }

        m_Shape = shape;
        m_DisposeBufferAfterUse = true;
    }
    internal TextureTensorData(RenderTexture bufferAsTexture, TensorShape shape, string buffername)
    {
        name = buffername;
        m_BufferAsTexture = bufferAsTexture;
        m_Shape = shape;

        m_DisposeBufferAfterUse = false;
    }

    ~TextureTensorData()
    {
        if (m_BufferAsTexture == null)
            return;
        if (!m_DisposeBufferAfterUse)
            return;

        D.LogWarning($"Found unreferenced, but undisposed Tensor data which might lead to GPU resource leak: {ToString()}");

        Dispose();
    }

    public virtual void Dispose()
    {
        if (m_DisposeBufferAfterUse)
        {
            m_BufferAsTexture.Release();
            m_BufferAsTexture = null;
        }
        m_DisposeBufferAfterUse = false;
    }

    public virtual void Reserve(int count)
    {
        if (count > maxCapacity)
            throw new ArgumentException("TextureTensorData buffer is too small to reserve " + count + " elements.");
    }

    public virtual void Upload(float[] data, TensorShape shape, int managedBufferStartIndex = 0)
    {
        var numItemToCopy = shape.length;
        var numItemAvailableInData = data.Length - managedBufferStartIndex;

        Assert.IsTrue(managedBufferStartIndex >= 0);
        Assert.IsTrue(numItemToCopy <= numItemAvailableInData);

        int w = Mathf.Min(shape.length, MaxTextureSize);
        int h = Mathf.Max(1, ComputeHelper.IDivC(shape.length, w));

        Texture2D texture = new Texture2D(w, h, TextureFormat.RFloat, false);
        var textureData = texture.GetRawTextureData<float>();
        unsafe
        {
            UnsafeUtility.MemSet(textureData.GetUnsafePtr(), 0, sizeof(float) * (textureData.Length));
        }
        NativeArray<float>.Copy(data, managedBufferStartIndex, textureData, 0, shape.length);

        texture.Apply();

        Material material = new Material(PixelShaderSingleton.Instance.FindShader("Barracuda/BufferToTensor"));

        material.SetTexture("Xtex2D", texture);

        material.SetInt("_InputWidth", w);
        material.SetInt("_InputHeight", h);

        material.SetVector("OdeclShape", new Vector4(shape.batch, shape.height, shape.width, shape.channels));

        Graphics.Blit(null, m_BufferAsTexture, material);

        m_AsyncDownloadSchedulingFrame = -1;
    }

    public virtual bool ScheduleAsyncDownload(int count)
    {
        return WaitFor3Frames();
    }

    private int m_AsyncDownloadSchedulingFrame = -1;
    private bool WaitFor3Frames()
    {
        if (m_AsyncDownloadSchedulingFrame < 0)
            m_AsyncDownloadSchedulingFrame = Time.frameCount;
        var framesPassed = Time.frameCount - m_AsyncDownloadSchedulingFrame;
        return framesPassed > 3;
    }

    public virtual float[] Download(TensorShape shape)
    {
        Assert.IsTrue(shape.Is4D());

        var count = shape.length;

        Profiler.BeginSample("Barracuda.DownloadDataFromGPU");
        Assert.IsTrue(maxCapacity >= count);
        count = Math.Min(maxCapacity, count);

        m_AsyncDownloadSchedulingFrame = -1;

        int w = Mathf.Min(shape.length, MaxTextureSize);
        int h = Mathf.Max(1, ComputeHelper.IDivC(shape.length, w));

        Texture2D texture = new Texture2D(w, h, TextureFormat.RFloat, false);
        RenderTexture rttexture = new RenderTexture(w, h, 0, RenderTextureFormat.RFloat);


        Material material = new Material(PixelShaderSingleton.Instance.FindShader("Barracuda/TensorToBuffer"));


        material.SetVector("XdeclShape", new Vector4(shape.batch, shape.height, shape.width, shape.channels));
        material.SetTexture("Xdata", bufferAsTexture);
        material.SetInt("_OutputWidth", w);
        material.SetInt("_OutputHeight", h);

        Graphics.Blit(null, rttexture, material);


        var previousActiveRT = RenderTexture.active;
        RenderTexture.active = rttexture;
        Rect rectReadPicture = new Rect(0, 0, w, h);
        texture.ReadPixels(rectReadPicture, 0, 0);
        texture.Apply();

        var data = new float[count];
        Buffer.BlockCopy(texture.GetRawTextureData(), 0, data, 0, count * sizeof(float));

        RenderTexture.active = previousActiveRT;

        return data;
    }

    public virtual BarracudaArray SharedAccess(out int offset)
    {
        offset = 0;
        return new BarracudaArrayFromManagedArray(Download(new TensorShape(0, 0, 0, maxCapacity)));//fp16?
    }

    public virtual int maxCapacity { get
    {
        return m_Shape.length;
    } }

    public virtual bool inUse { get
    {
        return true;
    } }

    public virtual bool isGPUMem { get
    {
        return true;
    } }

    public override string ToString()
    {
        string allocationSource = "";

        return string.Format("(GPU:{0}#{1} {2} bufferAsTexture: {3} created at: {4})",
            name, GetHashCode(), m_Shape, m_BufferAsTexture, allocationSource);
    }
}

public class PixelShaderOps : ReferenceCPUOps
{
    public PixelShaderOps(ITensorAllocator allocator = null)
    : base(allocator)
    {
    }

    static private StringCache m_StringCache = new StringCache();

    public TextureTensorData Pin(Tensor X, bool uploadCache = true)
    {
        X.FlushCache(uploadCache);

        var onDevice = X.tensorOnDevice as TextureTensorData;
        if (onDevice == null)
        {
            var asTexture = X.tensorOnDevice as TextureAsTensorData;
            if (asTexture != null)
                X.AttachToDevice(TextureToTensorData(asTexture, X.name));
            else
            {
                if (uploadCache)
                    X.UploadToDevice(new TextureTensorData(X.shape, X.name)); // device is not compatible, create new array and upload
                else
                    X.AllocateOnDevice(new TextureTensorData(X.shape, X.name)); // device is not compatible, create new array but do not upload nor 0-fill
            }
        }
       
        Assert.IsNotNull(X.tensorOnDevice as TextureTensorData);
        Assert.IsNotNull((X.tensorOnDevice as TextureTensorData).bufferAsTexture);

        return X.tensorOnDevice as TextureTensorData;
    }

    internal void SetTensor(Material material, string name, Tensor X)
    {
        var XonDevice = Pin(X);
        // need to hide batch tilling due to perf regression on mobile
        if (XonDevice.tensorBatchTilled)
            material.EnableKeyword("BATCHTILLING_ON");

        material.SetVector(m_StringCache.Lookup(name, "declShape"), new Vector4(X.batch, X.height, X.width, X.channels));
        material.SetTexture(m_StringCache.Lookup(name, "data"), XonDevice.bufferAsTexture);
    } 

    internal Tensor Dispatch(Material material, TensorShape Oshape)
    {
        var O = NewTensor(Oshape, AllocScope.LayerOutput, "O");

        var pinO = Pin(O);
        material.SetVector("OdeclShape", new Vector4(Oshape.batch, O.height, O.width, O.channels));
        material.SetTexture("Odata", pinO.bufferAsTexture);
        // need to hide batch tilling due to perf regression on mobile
        if (pinO.tensorBatchTilled)
            material.EnableKeyword("BATCHTILLING_ON");

        Graphics.Blit(null, pinO.bufferAsTexture, material);

        return O;
    }


    // ---------------------------------------------------------------------------------

    internal ITensorData TextureToTensorData(TextureAsTensorData texData, string name)
    {        
        var tensorData = new TextureTensorData(texData.shape, name, false);

        Material material = new Material(PixelShaderSingleton.Instance.FindShader("Barracuda/TextureToTensor"));

        material.SetVector("OdeclShape", new Vector4(texData.shape.batch, texData.shape.height, texData.shape.width, texData.shape.channels));

        material.SetInt("_FlipY", texData.flip == TextureAsTensorData.Flip.Y ? 1 : 0);
        material.SetVector("_Scale", texData.scale);
        material.SetVector("_Bias", texData.bias);

        Vector4 offsets = Vector4.zero;
        foreach (var tex in texData.textures)
        {
            var texArr = tex as Texture2DArray;
            var rt = tex as RenderTexture;

            var texDepth = 1;
            if (texArr)
                texDepth = texArr.depth;
            else if (rt)
                texDepth = rt.volumeDepth;

            material.SetTexture("Xtex2D", tex);
            material.SetVector("_Pool", new Vector2(tex.width, tex.height));
            material.SetVector("_Pad", offsets);

            var channelWriteMask = TextureFormatUtils.FormatToChannelMask(tex, texData.interpretPixelAsChannels);
            var channelReadMap = TextureFormatUtils.FormatToChannelReadMap(tex, texData.interpretPixelAsChannels);
            material.SetVector("_ChannelWriteMask", new Vector4(channelWriteMask[0], channelWriteMask[1], channelWriteMask[2], channelWriteMask[3]));
            material.SetVector("_ChannelReadMap", new Vector4(channelReadMap[0], channelReadMap[1], channelReadMap[2], channelReadMap[3]));

            Graphics.Blit(null, tensorData.bufferAsTexture, material);

            if (texData.interpretDepthAs == TextureAsTensorData.InterpretDepthAs.Batch)
                offsets[0] += texDepth;
            else if (texData.interpretDepthAs == TextureAsTensorData.InterpretDepthAs.Channels)
                offsets[3] += texDepth * texData.interpretPixelAsChannels;
        }

        return tensorData;
    }

    /// <summary>
    /// Check if `fusedActivation` is supported in-place
    /// </summary>
    /// <param name="fusedActivation">fused activation type</param>
    /// <returns>`true` if supported in-place</returns>
    protected override bool IsFusedActivationSupported(Layer.FusedActivation fusedActivation)
    {
        switch (fusedActivation)
        {
            case Layer.FusedActivation.Relu:
                return true;
            case Layer.FusedActivation.None:
                return true;
            default:
                return false;
        }
    }

    /// <summary>
    /// Copy `Tensor` data to `RenderTexture`
    /// </summary>
    /// <param name="X">source `Tensor`</param>
    /// <param name="target">target `RenderTexture`</param>
    /// <param name="batch">batch</param>
    /// <param name="fromChannel">from channel</param>
    /// <param name="scale">scale</param>
    /// <param name="bias">bias</param>
    /// <param name="lut">LUT table</param>
    public void TensorToRenderTexture(Tensor X, RenderTexture target, int batch, int fromChannel, Vector4 scale, Vector4 bias, Texture3D lut)
    {
        if (!target.IsCreated())
        {
            target.Release();
            target.Create();
        }

        Material material = new Material(PixelShaderSingleton.Instance.FindShader("Barracuda/TensorToTexture")); 

        SetTensor(material, "X", X);
        material.SetVector("_Scale", scale);                           
        material.SetVector("_Bias", bias);
        material.SetVector("_Pad", new Vector4(batch, 0, 0, fromChannel));
        material.SetInt("_FlipY", 1);
        material.SetInt("_OutputHeight", target.height);
        material.SetInt("_OutputWidth", target.width);

        Graphics.Blit(null, target, material);
    }

    /// <inheritdoc/>
    public override Tensor Conv2D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, Layer.FusedActivation fusedActivation)
    {
        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(X.channels, K.kernelDepth);
        Assert.AreEqual(K.kernelCount, B.flatWidth);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(stride.Length, 2);//WH
        Assert.AreEqual(pad.Length, 4);

        var Oshape = X.shape.ApplyKernel(K.shape, stride, pad);
        Material material = new Material(PixelShaderSingleton.Instance.FindShader("Barracuda/Conv2D"));

        SetTensor(material, "X", X);
        SetTensor(material, "K", K);
        SetTensor(material, "B", B);

        material.SetVector("_Stride", new Vector4(stride[0], stride[1], 0, 0));
        material.SetVector("_Pad", new Vector4(pad[0], pad[1], pad[2], pad[3]));
        material.SetInt("_ActivationMode", (int)(fusedActivation));

        var O = Dispatch(material, Oshape);

        if (!IsFusedActivationSupported(fusedActivation))
            O = Activation(m_StringCache.Lookup("Barracuda/", fusedActivation.ToString()), O);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Conv2DTrans(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, int[] outputAdjustment, Layer.FusedActivation fusedActivation)
    {
        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(X.channels, K.kernelDepth);
        Assert.AreEqual(K.kernelCount, B.flatWidth);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);

        var Oshape = X.shape.ApplyKernelInverse(K.shape, stride, pad, outputAdjustment);
        Material material = new Material(PixelShaderSingleton.Instance.FindShader("Barracuda/Conv2DTrans"));

        // one pass version
        pad = new int[]
        {
            K.kernelWidth - pad[0] - 1, K.kernelHeight - pad[1] - 1,
            K.kernelWidth - pad[2] - 1, K.kernelHeight - pad[3] - 1
        };

        SetTensor(material, "X", X);
        SetTensor(material, "K", K);
        SetTensor(material, "B", B);

        material.SetVector("_Stride", new Vector4(stride[0], stride[1], 0, 0));
        material.SetVector("_Pad", new Vector4(pad[0], pad[1], 0, 0));
        material.SetInt("_ActivationMode", (int)(fusedActivation));

        var O = Dispatch(material, Oshape);

        if (!IsFusedActivationSupported(fusedActivation))
            O = Activation(m_StringCache.Lookup("Barracuda/", fusedActivation.ToString()), O);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor DepthwiseConv2D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, Layer.FusedActivation fusedActivation)
    {
        if (K.kernelDepth != 1)
            return base.DepthwiseConv2D(X, K, B, stride, pad, fusedActivation);

        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(K.kernelDepth, 1);
        Assert.AreEqual(K.kernelCount, X.channels);
        Assert.AreEqual(K.kernelCount, B.flatWidth);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);

        var Oshape = X.shape.ApplyKernel(K.shape, stride, pad);
        Material material = new Material(PixelShaderSingleton.Instance.FindShader("Barracuda/DepthwiseConv2D"));

        SetTensor(material, "X", X);
        SetTensor(material, "K", K);
        SetTensor(material, "B", B);

        material.SetVector("_Stride", new Vector4(stride[0], stride[1], 0, 0));
        material.SetVector("_Pad", new Vector4(pad[0], pad[1], pad[2], pad[3]));
        material.SetInt("_ActivationMode", (int)(fusedActivation));

        var O = Dispatch(material, Oshape);

        if (!IsFusedActivationSupported(fusedActivation))
            O = Activation(m_StringCache.Lookup("Barracuda/", fusedActivation.ToString()), O);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Dense(Tensor X, Tensor W, Tensor B, Layer.FusedActivation fusedActivation)
    {
        Assert.IsTrue(W.dimensions <= 2);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(X.flatWidth, W.flatHeight);

        var Oshape = new TensorShape(X.flatHeight, W.flatWidth);

        Material material = new Material(PixelShaderSingleton.Instance.FindShader("Barracuda/Dense"));

        SetTensor(material, "X", X);
        SetTensor(material, "W", W);
        SetTensor(material, "B", B);
        material.SetInt("_ActivationMode", (int)fusedActivation);

        var O = Dispatch(material, Oshape);

        if (!IsFusedActivationSupported(fusedActivation))
            O = Activation(m_StringCache.Lookup("Barracuda/", fusedActivation.ToString()), O);

        return O;
    }

    /// <summary>
    /// Elementwise broadcast for specified kernel
    /// </summary>
    /// <param name="kernelName">kernel name</param>
    /// <param name="tensors">input tensors</param>
    /// <returns>output `Tensor`</returns>
    /// <exception cref="NotImplementedException">thrown if input `Tensor` is not compatible with 4D shape</exception>
    protected virtual Tensor ElementwiseWithBroadcast(string kernelName, Tensor[] tensors)
    {
        var Oshape = TensorExtensions.MaxShape(tensors);
        var O = NewTensor(Oshape, AllocScope.LayerOutput, "O");

        Assert.IsTrue(tensors.Length > 0);
        var X = tensors[0];

        Material material = new Material(PixelShaderSingleton.Instance.FindShader(kernelName));

        for (int t = 1; t < tensors.Length; ++t)
        {
            var B = tensors[t];
            Assert.IsTrue(B.shape.Is4D());

            SetTensor(material, "X", X);
            SetTensor(material, "B", B);

            var pinO = Pin(O);
            material.SetVector("OdeclShape", new Vector4(O.batch, O.height, O.width, O.channels));

            Graphics.Blit(null, pinO.bufferAsTexture, material);

            X = O;
        }

        return X;
    }

    /// <inheritdoc/>
    public override Tensor Add(Tensor[] tensors)
    {
        if (tensors.Any(x => !x.shape.Is4D()))
            return base.Add(tensors);

        return ElementwiseWithBroadcast("Barracuda/BroadcastAdd", tensors);
    }

    /// <inheritdoc/>

    public override Tensor Sub(Tensor[] tensors)
    {
        if (tensors.Any(x => !x.shape.Is4D()))
            return base.Add(tensors);

        return ElementwiseWithBroadcast("Barracuda/BroadcastSub", tensors);
    }

    /// <inheritdoc/>
    public override Tensor Mul(Tensor[] tensors)
    {
        if (tensors.Any(x => !x.shape.Is4D()))
            return base.Add(tensors);

        return ElementwiseWithBroadcast("Barracuda/BroadcastMul", tensors);
    }

    /// <inheritdoc/>
    public override Tensor Min(Tensor[] tensors)
    {
        if (tensors.Any(x => !x.shape.Is4D()))
            return base.Add(tensors);

        return ElementwiseWithBroadcast("Barracuda/BroadcastMin", tensors);
    }

    /// <summary>
    /// Generic pooling 2D
    /// </summary>
    /// <param name="kernelName">kernel name</param>
    /// <param name="X">input</param>
    /// <returns>output `Tensor`</returns>
    protected virtual Tensor GlobalPool2D(string kernelName, Tensor X)
    {
        Assert.IsTrue(X.shape.Is4D());
        var Oshape = new TensorShape(X.batch, 1, 1, X.channels);

        Material material = new Material(PixelShaderSingleton.Instance.FindShader(kernelName));

        SetTensor(material, "X", X);

        return Dispatch(material, Oshape);
    }

    /// <inheritdoc/>
    public override Tensor GlobalAvgPool2D(Tensor X)
    {
        return GlobalPool2D("Barracuda/GlobalAvgPool2D", X);
    }

    /// <inheritdoc/>
    protected virtual Tensor Pool2D(string kernelName, Tensor X, int[] pool, int[] stride, int[] pad)
    {
        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(pool.Length, 2);
        Assert.AreEqual(stride.Length, 2);

        var Oshape = X.shape.ApplyPool(pool, stride, pad);

        Material material = new Material(PixelShaderSingleton.Instance.FindShader(kernelName));

        SetTensor(material, "X", X);

        material.SetVector("_Pool", new Vector4(pool[0], pool[1], 0, 0));
        material.SetVector("_Stride", new Vector4(stride[0], stride[1], 0, 0));
        material.SetVector("_Pad", new Vector4(pad[0], pad[1], pad[2], pad[3]));

        return Dispatch(material, Oshape);
    }

    /// <inheritdoc/>
    public override Tensor MaxPool2D(Tensor X, int[] pool, int[] stride, int[] pad)
    {
        return Pool2D("Barracuda/MaxPool2D", X, pool, stride, pad);
    }

    /// <inheritdoc/>
    public override Tensor AvgPool2D(Tensor X, int[] pool, int[] stride, int[] pad)
    {
        return Pool2D("Barracuda/AvgPool2D", X, pool, stride, pad);
    }

    /// <inheritdoc/>
    public override Tensor Normalization(Tensor X, Tensor S, Tensor B, int pool, int axis, float epsilon, Layer.FusedActivation fusedActivation)
    {
        if (!X.shape.Is4D())
            throw new NotImplementedException();

        if (axis != TensorShape.C && axis != -1)
            return base.Normalization(X, S, B, pool, axis, epsilon, fusedActivation);

        if (pool == 1 && X.batch != 1)
            return base.Normalization(X, S, B, pool, axis, epsilon, fusedActivation); // @TODO: Instance Normalization with batch > 1

        if (pool <= 0)
            pool = X.batch;

        Material material = new Material(PixelShaderSingleton.Instance.FindShader("Barracuda/InstanceNorm"));

        material.SetFloat("_Epsilon", epsilon);
        material.SetInt("_ActivationMode", (int)fusedActivation);

        SetTensor(material, "X", X);
        SetTensor(material, "W", S);
        SetTensor(material, "B", B);

        var O = Dispatch(material, X.shape);

        if (!IsFusedActivationSupported(fusedActivation))
            O = Activation(m_StringCache.Lookup("Barracuda/", fusedActivation.ToString()), O);

        return O;
    }

    /// <summary>
    /// Apply padding
    /// </summary>
    /// <param name="X">input</param>
    /// <param name="pad">padding</param>
    /// <param name="kernelName">kernel name</param>
    /// <param name="constant">constant</param>
    /// <returns>output `Tensor`</returns>
    protected virtual Tensor ApplyPadding(Tensor X, int[] pad, string kernelName, float constant = 0.0f)
    {
        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(pad.Length, 6);

        Assert.AreEqual(pad[2], 0, "PixelShader.ApplyPadding: unsupported channel-padding");
        Assert.AreEqual(pad[5], 0, "PixelShader.ApplyPadding: unsupported channel-padding");


        var Oshape = X.shape.ApplyBorder(pad);

        Material material = new Material(PixelShaderSingleton.Instance.FindShader(kernelName));

        SetTensor(material, "X", X);

        // TODO support C-padding
        material.SetVector("_Pad", new Vector4(pad[0], pad[1], pad[3], pad[4]));


        if (kernelName.Contains("Border2D"))
        {
            // NOTE: negative "pad" variable will crop X tensor
            int croppedWidth = X.width - Math.Max(0, -pad[3]);
            int croppedHeight = X.height - Math.Max(0, -pad[4]);
            var croppedSize = new int[] { 0, 0 };
            croppedSize[0] = croppedWidth;
            croppedSize[1] = croppedHeight;

            material.SetVector("_Pool", new Vector4(croppedSize[0], croppedSize[1], 0, 0));
            material.SetFloat("_Beta", constant);
        }

        return Dispatch(material, Oshape);
    }

    /// <inheritdoc/>
    public override Tensor Border2D(Tensor X, int[] pad, float constant)
    {
        return ApplyPadding(X, pad, "Barracuda/Border2D", constant);
    }

    /// <inheritdoc/>

    public override Tensor Pad2DReflect(Tensor X, int[] pad)
    {
        return ApplyPadding(X, pad, "Barracuda/Pad2DReflect");
    }

    /// <inheritdoc/>
    public override Tensor Pad2DSymmetric(Tensor X, int[] pad)
    {
        return ApplyPadding(X, pad, "Barracuda/Pad2DSymmetric");
    }

    /// <inheritdoc/>
    public override Tensor Pad2DEdge(Tensor X, int[] pad)
    {
        return ApplyPadding(X, pad, "Barracuda/Pad2DEdge");
    }

    /// <summary>
    /// Generic activation function
    /// </summary>
    /// <param name="kernelName">kernel name</param>
    /// <param name="X">input</param>
    /// <param name="alpha">alpha</param>
    /// <param name="beta">beta</param>
    /// <returns>output Tensor</returns>
    protected virtual Tensor Activation(string kernelName, Tensor X, float alpha = 0f, float beta = 0f)
    {
        Assert.IsTrue(X.shape.Is4D());

        var Oshape = X.shape;

        Material material = new Material(PixelShaderSingleton.Instance.FindShader(kernelName));

        SetTensor(material, "X", X);
        material.SetFloat("_Alpha", alpha);
        material.SetFloat("_Beta",  beta);

        return Dispatch(material, Oshape);
    }

    /// <inheritdoc/>
    public override Tensor Clip(Tensor X, float alpha, float beta)
    {
        if (!X.shape.Is4D())
            return base.Clip(X, alpha, beta);

        return Activation("Barracuda/Clip", X, alpha, beta);
    }

    /// <inheritdoc/>

    public override Tensor Relu(Tensor X)
    {
        if (!X.shape.Is4D())
            return base.Relu(X);

        return Activation("Barracuda/Relu", X);
    }

    /// <inheritdoc/>
    public override Tensor Selu(Tensor X, float alpha, float beta)
    {
        if (!X.shape.Is4D())
            return base.Selu(X, alpha, beta);

        return Activation("Barracuda/Selu", X, alpha, beta);
    }

    /// <inheritdoc/>
    public override Tensor LeakyRelu(Tensor X, float alpha)
    {
        if (!X.shape.Is4D())
            return base.LeakyRelu(X, alpha);

        return Activation("Barracuda/LeakyRelu", X, alpha);
    }

    /// <inheritdoc/>
    public override Tensor Tanh(Tensor X)
    {
        if (!X.shape.Is4D())
            return base.Tanh(X);

        return Activation("Barracuda/Tanh", X);
    }

    /// <inheritdoc/>
    public override Tensor Sqrt(Tensor X)
    {
        if (!X.shape.Is4D())
            return base.Sqrt(X);

        return Activation("Barracuda/Sqrt", X);
    }

    /// <inheritdoc/>
    public override Tensor Reciprocal(Tensor X)
    {
        if (!X.shape.Is4D())
            return base.Reciprocal(X);

        return Activation("Barracuda/Reciprocal", X);
    }

    /// <inheritdoc/>
    public override Tensor Upsample2D(Tensor X, int[] scale, bool bilinear)
    {
        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(scale.Length, 2);

        var Oshape = new TensorShape(X.batch, X.height*scale[1], X.width*scale[0], X.channels);

        Material material = new Material(PixelShaderSingleton.Instance.FindShader(bilinear ? "Barracuda/UpsampleBilinear2D" : "Barracuda/Upsample2D"));

        SetTensor(material, "X", X);
    
        material.SetVector("_Pool", new Vector4(scale[0], scale[1], 0,0));

        return Dispatch(material, Oshape);
    }

    /// <inheritdoc/>
    public override Tensor Resample2D(Tensor X, int[] size, bool bilinear)
    {
        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(size.Length, 2);

        var Oshape = new TensorShape(X.batch, size[1], size[0], X.channels);

        Material material = new Material(PixelShaderSingleton.Instance.FindShader(bilinear ? "Barracuda/ResampleBilinear2D" : "Barracuda/Resample2D"));


        SetTensor(material, "X", X);

        return Dispatch(material, Oshape);
    }

    /// <inheritdoc/>
    public override Tensor Concat(Tensor[] tensors, int axis)
    {
        if (tensors.Any(x => !x.shape.Is4D()))
            return base.Concat(tensors, axis);

        var Oshape = TensorExtensions.Concat(tensors, axis);
        axis = Oshape.Axis(axis);
        var axisNCHW = TensorExtensions.Convert8DAxisTo4D(axis);
        Vector4 offsets = Vector4.zero;

        Material material = new Material(PixelShaderSingleton.Instance.FindShader("Barracuda/Copy"));

        var O = NewTensor(Oshape, AllocScope.LayerOutput, "O");
        var Opred = NewTensor(Oshape, AllocScope.LayerOutput, "O");

        bool pingPong = true;
        bool isFirstPass = true;
        foreach (var inputTensor in tensors)
        {
            Assert.IsTrue(inputTensor.shape.Is4D());

            SetTensor(material, "X", inputTensor);
            SetTensor(material, "OPred", pingPong ? O : Opred);
            
            material.SetVector("_Pad", offsets);
            
            material.SetInt("_IsFirstPass", isFirstPass ? 1 : 0);
            
            var pinO = pingPong ? Pin(Opred) : Pin(O);
            material.SetVector("OdeclShape", new Vector4(O.batch, O.height, O.width, O.channels));
            
            Graphics.Blit(null, pinO.bufferAsTexture, material);

            offsets[axisNCHW] += inputTensor.shape[axis];

            isFirstPass = false;
            pingPong = !pingPong;
        }

        return pingPong ? O : Opred;
    }

    /// <inheritdoc/>
    public override Tensor StridedSlice(Tensor X, int[] starts, int[] ends, int[] strides)
    {
        if (X.shape.Is4D())
            return base.StridedSlice(X, starts, ends, strides);

        var Oshape = X.shape.ApplyStridedSlice(starts, ends, strides);

        Vector4 starts4d = new Vector4();
        starts4d[0] = Math.Min(TensorExtensions.WrapIndex(starts[TensorShape.DataBatch], X.batch), X.batch - 1);
        starts4d[1] = Math.Min(TensorExtensions.WrapIndex(starts[TensorShape.H], X.height), X.height - 1);
        starts4d[2] = Math.Min(TensorExtensions.WrapIndex(starts[TensorShape.W], X.width), X.width - 1);
        starts4d[3] = Math.Min(TensorExtensions.WrapIndex(starts[TensorShape.C], X.channels), X.channels - 1);

        Vector4 strides4d = new Vector4();
        strides4d[0] = strides[TensorShape.DataBatch];
        strides4d[1] = strides[TensorShape.H];
        strides4d[2] = strides[TensorShape.W];
        strides4d[3] = strides[TensorShape.C];

        Material material = new Material(PixelShaderSingleton.Instance.FindShader("Barracuda/StridedSlice"));
        
        SetTensor(material, "X", X);
        material.SetVector("_Stride", new Vector4(strides4d[0], strides4d[1], strides4d[2], strides4d[3]));
        material.SetVector("_Starts", new Vector4(starts4d[0], starts4d[1], starts4d[2], starts4d[3]));
        
        return Dispatch(material, Oshape);
    }

    /// <inheritdoc/>
    public override Tensor ScaleBias(Tensor X, Tensor S, Tensor B)
    {
        Assert.AreEqual(X.channels, B.channels); Assert.AreEqual(X.channels, S.channels);
        Assert.AreEqual(B.length, B.channels); Assert.AreEqual(S.length, S.channels);

        Material material = new Material(PixelShaderSingleton.Instance.FindShader("Barracuda/ScaleBias"));

        SetTensor(material, "X", X);
        SetTensor(material, "W", S);
        SetTensor(material, "B", B);


        return Dispatch(material, X.shape);
    }

    /// <inheritdoc/>
    public override Tensor Transpose(Tensor X, int[] permutations)
    {
        if (X.shape.Is4D())
            return base.Transpose(X, permutations);

        Material material = new Material(PixelShaderSingleton.Instance.FindShader("Barracuda/Transpose"));

        SetTensor(material, "X", X);


        material.SetVector("_Pool", new Vector4(Array.IndexOf(permutations, 0), Array.IndexOf(permutations, 1), Array.IndexOf(permutations, 2), Array.IndexOf(permutations, 3)));

        return Dispatch(material, X.shape.Permute(permutations));
    }

    /// <inheritdoc/>
    public override Tensor Reshape(Tensor X, TensorShape newShape)
    {
        if (X.shape == newShape)
            return Copy(X);

        var O = NewTensor(newShape, AllocScope.LayerOutput, "O");
        Graphics.Blit(Pin(X).bufferAsTexture, Pin(O).bufferAsTexture);
        
        return O;
    }

    /// <inheritdoc/>
    public override Tensor Copy(Tensor X)
    {
        var O = NewTensor(X.shape, AllocScope.LayerOutput, "O");
        Graphics.Blit(Pin(X).bufferAsTexture, Pin(O).bufferAsTexture);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Prepare(Tensor X)
    {
        Pin(X);
        return X;
    }

    /// <inheritdoc/>
    public override Tensor PrepareNoAlloc(Tensor X)
    {
        Pin(X, uploadCache: false);
        return X;
    }
}

} // namespace Unity.Barracuda
