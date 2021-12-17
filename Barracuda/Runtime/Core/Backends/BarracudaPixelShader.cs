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
using Object = UnityEngine.Object;

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

    /// <inheritdoc/>
    public virtual DataType dataType { get
    {
        return DataType.Float;//todo fp16
    } }

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
            // In emergency shutdown situations active RenderTexture might be the one we are trying to release
            if (RenderTexture.active == m_BufferAsTexture)
                RenderTexture.active = null;

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

        Object.DestroyImmediate(texture);

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
        return new BarracudaArrayFromManagedArray(Download(new TensorShape(0, 0, 0, maxCapacity)));//TODO fp16
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
        try
        {
            // m_BufferAsTexture.ToString() might throw exception if called from non-main thread
            return $"(GPU:{name}#{GetHashCode()} {m_Shape}) bufferAsTexture: {m_BufferAsTexture}";
        }
        catch (Exception)
        {
            return $"(GPU:{name}#{GetHashCode()} {m_Shape})";
        }

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

    internal Tensor Dispatch(Material material, DataType dataType, TensorShape Oshape)
    {
        var O = NewTensor(dataType, Oshape, AllocScope.LayerOutput, "O");

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
            var channelWriteMap = Vector4.zero;
            int c = 0;
            for(int i = 0; i < 4; i++)
            {
                channelWriteMap[i] = c;
                if (channelWriteMask[i] == 1)
                    c++;
            }
            material.SetVector("_ChannelWriteMask", new Vector4(channelWriteMask[0], channelWriteMask[1], channelWriteMask[2], channelWriteMask[3]));
            material.SetVector("_ChannelWriteMap", new Vector4(channelWriteMap[0], channelWriteMap[1], channelWriteMap[2], channelWriteMap[3]));
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
    /// <param name="flipY">flips the texture along the Y dimension (optional, default: true)</param>
    public void TensorToRenderTexture(Tensor X, RenderTexture target, int batch, int fromChannel, Vector4 scale, Vector4 bias, Texture3D lut, bool flipY = true)
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
        material.SetInt("_FlipY", flipY ? 1 : 0);
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

        var O = Dispatch(material, X.dataType, Oshape);

        if (!IsFusedActivationSupported(fusedActivation))
            O = Activation(fusedActivation.ToString(), O);

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

        var O = Dispatch(material, X.dataType, Oshape);

        if (!IsFusedActivationSupported(fusedActivation))
            O = Activation(fusedActivation.ToString(), O);

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

        var O = Dispatch(material, X.dataType, Oshape);

        if (!IsFusedActivationSupported(fusedActivation))
            O = Activation(fusedActivation.ToString(), O);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor MatMul(Tensor X, bool xTranspose, Tensor Y, bool yTranspose)
    {
        var O = new TensorShape(X.flatHeight, Y.flatWidth);
        if (xTranspose)
            O = new TensorShape(X.flatWidth, O.flatWidth);
        if (yTranspose)
            O = new TensorShape(O.flatHeight, Y.flatHeight);
   
        Material material = new Material(PixelShaderSingleton.Instance.FindShader("Barracuda/MatMul"));
        if (xTranspose)
            material.EnableKeyword("xTranspose_ON");
        if (yTranspose)
            material.EnableKeyword("yTranspose_ON");
   
        SetTensor(material, "X", X);
        SetTensor(material, "Y", Y);
   
        return Dispatch(material, X.dataType, O);
    }

    /// <summary>
    /// Check if `Flatten` is needed for `Dense` layer input
    /// </summary>
    /// <param name="X">input shape</param>
    /// <returns>`true` if `Flatten` is needed</returns>
    protected bool ShouldFlattenInputForDenseLayer(TensorShape X)
    {
        //In CHW flatten is return a tensor with items linearized in memory in regards to HWC layout.
        int flattenDimensions = (X.height > 1 ? 1 : 0) +
                                (X.width > 1 ? 1 : 0) +
                                (X.channels > 1 ? 1 : 0);
        return flattenDimensions > 1;
    }

    /// <inheritdoc/>
    public override Tensor Dense(Tensor X, Tensor W, Tensor B, Layer.FusedActivation fusedActivation)
    {
        Assert.IsTrue(W.dimensions <= 2);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(X.flatWidth, W.flatHeight);

        if (ShouldFlattenInputForDenseLayer(X.shape))
            X = Flatten(X);
   
        var Oshape = new TensorShape(X.flatHeight, W.flatWidth);

        Material material = new Material(PixelShaderSingleton.Instance.FindShader("Barracuda/Dense"));

        SetTensor(material, "X", X);
        SetTensor(material, "W", W);
        SetTensor(material, "B", B);
        material.SetInt("_ActivationMode", (int)fusedActivation);

        var O = Dispatch(material, X.dataType, Oshape);

        if (!IsFusedActivationSupported(fusedActivation))
            O = Activation(fusedActivation.ToString(), O);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Dense3(Tensor X, Tensor W, Tensor B)
    {
        var Oshape = new TensorShape(X.batch, 1, W.channels, X.channels);

        Material material = new Material(PixelShaderSingleton.Instance.FindShader("Barracuda/Dense3"));

        SetTensor(material, "X", X);
        SetTensor(material, "W", W);
        SetTensor(material, "B", B);

        return Dispatch(material, X.dataType, Oshape);
    }

    private Tensor ReduceHelper(string kernelName, Tensor X, int axis)
    {
        axis = X.shape.Axis(axis);
    
        var O = X.shape.Reduce(axis);
    
        Material material = new Material(PixelShaderSingleton.Instance.FindShader("Barracuda/Reduce"));
        material.EnableKeyword(kernelName);
    
        if(axis == TensorShape.DataBatch)
            material.EnableKeyword("ReduceN");
        if (axis == TensorShape.H)
            material.EnableKeyword("ReduceH");
        if (axis == TensorShape.W)
            material.EnableKeyword("ReduceW");
        if (axis == TensorShape.C)
            material.EnableKeyword("ReduceC");
    
        SetTensor(material, "X", X);
    
        return Dispatch(material, X.dataType, O);
    }
    
    /// <inheritdoc/>
    public override Tensor ArgMax(Tensor X, int axis)
    {
        return ReduceHelper("ArgMax", X, axis);
    }
    
    /// <inheritdoc/>
    public override Tensor ArgMin(Tensor X, int axis)
    {
        return ReduceHelper("ArgMin", X, axis);
    }
    
    /// <inheritdoc/>
    public override Tensor ReduceMin(Tensor X, int axis)
    {
        return ReduceHelper("ReduceMin", X, axis);
    }
    
    /// <inheritdoc/>
    public override Tensor ReduceMax(Tensor X, int axis)
    {
        return ReduceHelper("ReduceMax", X, axis);
    }
    
    /// <inheritdoc/>
    public override Tensor ReduceSum(Tensor X, int axis)
    {
        return ReduceHelper("ReduceSum", X, axis);
    }
    
    /// <inheritdoc/>
    public override Tensor ReduceMean(Tensor X, int axis)
    {
        return ReduceHelper("ReduceMean", X, axis);
    }
    
    /// <inheritdoc/>
    public override Tensor ReduceProd(Tensor X, int axis)
    {
        return ReduceHelper("ReduceProd", X, axis);
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
        var O = TensorExtensions.MaxShape(tensors);

        Assert.IsTrue(tensors.Length > 0);
        var X = tensors[0];

        Material material = new Material(PixelShaderSingleton.Instance.FindShader("Barracuda/Broadcast"));
        material.EnableKeyword(kernelName);

        bool isFirstDispatch = true;
        for (int t = 1; t < tensors.Length; ++t)
        {
            var B = tensors[t];
            Assert.IsTrue(B.shape.Is4D());

            SetTensor(material, "X", X);
            SetTensor(material, "B", B);

            material.SetFloat("_Alpha", 1.0f/(float)tensors.Length);
            material.SetInt("_IsFirstDispatch", isFirstDispatch ? 1 : 0);

            X = Dispatch(material, X.dataType, O);
            isFirstDispatch = false;
        }

        return X;
    }

    /// <inheritdoc/>
    public override Tensor Add(Tensor[] tensors)
    {
        if (tensors.Any(x => !x.shape.Is4D()))
            return base.Add(tensors);

        return ElementwiseWithBroadcast("Add", tensors);
    }

    /// <inheritdoc/>

    public override Tensor Sub(Tensor[] tensors)
    {
        if (tensors.Any(x => !x.shape.Is4D()))
            return base.Add(tensors);

        return ElementwiseWithBroadcast("Sub", tensors);
    }

    /// <inheritdoc/>
    public override Tensor Mul(Tensor[] tensors)
    {
        if (tensors.Any(x => !x.shape.Is4D()))
            return base.Add(tensors);

        return ElementwiseWithBroadcast("Mul", tensors);
    }

    /// <inheritdoc/>
    public override Tensor Div(Tensor[] tensors)
    {
        if (tensors.Any(x => !x.shape.Is4D()))
            return base.Div(tensors);

        return ElementwiseWithBroadcast("Div", tensors);
    }

    /// <inheritdoc/>
    public override Tensor Pow(Tensor[] tensors)
    {
        if (tensors.Any(x => !x.shape.Is4D()))
            return base.Pow(tensors);

        return ElementwiseWithBroadcast("Pow", tensors);
    }

    /// <inheritdoc/>
    public override Tensor Min(Tensor[] tensors)
    {
        if (tensors.Any(x => !x.shape.Is4D()))
            return base.Add(tensors);

        return ElementwiseWithBroadcast("Min", tensors);
    }
    
    /// <inheritdoc/>
    public override Tensor Max(Tensor[] tensors)
    {
        if (tensors.Any(x => !x.shape.Is4D()))
            return base.Max(tensors);

        return ElementwiseWithBroadcast("Max", tensors);
    }

    /// <inheritdoc/>
    public override Tensor Mean(Tensor[] tensors)
    {
        if (tensors.Any(x => !x.shape.Is4D()))
            return base.Mean(tensors);

        return ElementwiseWithBroadcast("Mean", tensors);
    }

    internal static Tensor[] s_ElementwiseBroadcastTensors = new Tensor[2];

    /// <inheritdoc/>
    public override Tensor Greater(Tensor A, Tensor B)
    {
        s_ElementwiseBroadcastTensors[0] = A;
        s_ElementwiseBroadcastTensors[1] = B;
        return ElementwiseWithBroadcast("Greater", s_ElementwiseBroadcastTensors);
    }

    /// <inheritdoc/>
    public override Tensor GreaterEqual(Tensor A, Tensor B)
    {
        s_ElementwiseBroadcastTensors[0] = A;
        s_ElementwiseBroadcastTensors[1] = B;
        return ElementwiseWithBroadcast("GreaterEqual", s_ElementwiseBroadcastTensors);
    }

    /// <inheritdoc/>
    public override Tensor Less(Tensor A, Tensor B)
    {
        s_ElementwiseBroadcastTensors[0] = A;
        s_ElementwiseBroadcastTensors[1] = B;
        return ElementwiseWithBroadcast("Less", s_ElementwiseBroadcastTensors);
    }

    /// <inheritdoc/>
    public override Tensor LessEqual(Tensor A, Tensor B)
    {
        s_ElementwiseBroadcastTensors[0] = A;
        s_ElementwiseBroadcastTensors[1] = B;
        return ElementwiseWithBroadcast("LessEqual", s_ElementwiseBroadcastTensors);
    }

    /// <inheritdoc/>
    public override Tensor Equal(Tensor A, Tensor B)
    {
        s_ElementwiseBroadcastTensors[0] = A;
        s_ElementwiseBroadcastTensors[1] = B;
        return ElementwiseWithBroadcast("Equal", s_ElementwiseBroadcastTensors);
    }

    /// <inheritdoc/>
    public override Tensor LogicalOr(Tensor A, Tensor B)
    {
        s_ElementwiseBroadcastTensors[0] = A;
        s_ElementwiseBroadcastTensors[1] = B;
        return ElementwiseWithBroadcast("LogicalOr", s_ElementwiseBroadcastTensors);
    }

    /// <inheritdoc/>
    public override Tensor LogicalAnd(Tensor A, Tensor B)
    {
        s_ElementwiseBroadcastTensors[0] = A;
        s_ElementwiseBroadcastTensors[1] = B;
        return ElementwiseWithBroadcast("LogicalAnd", s_ElementwiseBroadcastTensors);
    }

    /// <inheritdoc/>
    public override Tensor LogicalXor(Tensor A, Tensor B)
    {
        s_ElementwiseBroadcastTensors[0] = A;
        s_ElementwiseBroadcastTensors[1] = B;
        return ElementwiseWithBroadcast("LogicalXor", s_ElementwiseBroadcastTensors);
    }

    /// <inheritdoc/>
    public override Tensor LogicalNot(Tensor X)
    {
        return Activation("LogicalNot", X);
    }

    /// <inheritdoc/>
    public override Tensor Sign(Tensor X)
    {
        return Activation("Sign", X);
    }

    /// <inheritdoc/>
    public override Tensor Where(Tensor C, Tensor A, Tensor B)
    {
        Material material = new Material(PixelShaderSingleton.Instance.FindShader("Barracuda/BroadcastWhere"));

        var O = TensorExtensions.MaxShape(new[] { C, A, B });

        SetTensor(material, "X", C);
        SetTensor(material, "W", A);
        SetTensor(material, "K", B);

        return Dispatch(material, C.dataType, O);
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

        return Dispatch(material, X.dataType, Oshape);
    }

    /// <inheritdoc/>
    public override Tensor GlobalMaxPool2D(Tensor X)
    {
        return GlobalPool2D("Barracuda/GlobalMaxPool2D", X);
    }

    /// <inheritdoc/>
    public override Tensor GlobalAvgPool2D(Tensor X)
    {
        return GlobalPool2D("Barracuda/GlobalAvgPool2D", X);
    }

    /// <inheritdoc/>
    public override Tensor GlobalAvgVariancePool2D(Tensor X)
    {
        Assert.IsTrue(X.shape.Is4D());
        var O = new TensorShape(X.batch, 2, 1, X.channels);

        Material material = new Material(PixelShaderSingleton.Instance.FindShader("GlobalAvgVariancePool2D"));

        SetTensor(material, "X", X);

        return Dispatch(material, X.dataType, O);
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

        return Dispatch(material, X.dataType, Oshape);
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

        var O = Dispatch(material, X.dataType, X.shape);

        if (!IsFusedActivationSupported(fusedActivation))
            O = Activation(fusedActivation.ToString(), O);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor OneHot(Tensor X, int depth, float onValue, float offValue, int inputRank=-1)
    {
        if (inputRank == -1)
            inputRank = X.dimensions;

        if (inputRank >= 4)
            throw new NotImplementedException();

        TensorShape O;
        if (inputRank == 1)
            O = new TensorShape(X.flatHeight, depth);
        else if (inputRank == 2)
            O = new TensorShape(X.flatHeight, 1, depth, X.channels);
        else
            O = new TensorShape(X.batch, X.width, depth, X.channels);
        
        Material material = new Material(PixelShaderSingleton.Instance.FindShader("Barracuda/OneHot"));
        if (inputRank == 1)
            material.EnableKeyword("Input1D");
        else if (inputRank == 2)
            material.EnableKeyword("Input2D");
        else
            material.EnableKeyword("Input3D");

        SetTensor(material, "X", X);
        material.SetFloat("_Alpha", onValue);
        material.SetFloat("_Beta", offValue);

        return Dispatch(material, X.dataType, O);
    }

    /// <inheritdoc/>
    public override Tensor LRN(Tensor X, float alpha, float beta, float bias, int size)
    {
        var O = X.shape;

        Material material = new Material(PixelShaderSingleton.Instance.FindShader("Barracuda/LRN"));

        SetTensor(material, "X", X);
        material.SetFloat("_Alpha", alpha);
        material.SetFloat("_Beta",  beta);
        material.SetFloat("_Epsilon",  bias);
        material.SetInt("_Axis", size);

        return Dispatch(material, X.dataType, O);
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

        return Dispatch(material, X.dataType, Oshape);
    }

    /// <inheritdoc/>
    public override Tensor Border2D(Tensor X, int[] pad, float constant)
    {
        if (pad[2] != 0 || pad[5] != 0)
            return base.Border2D(X, pad, constant);

        return ApplyPadding(X, pad, "Barracuda/Border2D", constant);
    }

    /// <inheritdoc/>
    public override Tensor Pad2DReflect(Tensor X, int[] pad)
    {
        if (pad[2] != 0 || pad[5] != 0)
            return base.Pad2DReflect(X, pad);

        return ApplyPadding(X, pad, "Barracuda/Pad2DReflect");
    }

    /// <inheritdoc/>
    public override Tensor Pad2DSymmetric(Tensor X, int[] pad)
    {
        if (pad[2] != 0 || pad[5] != 0)
            return base.Pad2DSymmetric(X, pad);

        return ApplyPadding(X, pad, "Barracuda/Pad2DSymmetric");
    }

    /// <inheritdoc/>
    public override Tensor Pad2DEdge(Tensor X, int[] pad)
    {
        if (pad[2] != 0 || pad[5] != 0)
            return base.Pad2DEdge(X, pad);

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

        Material material = new Material(PixelShaderSingleton.Instance.FindShader("Barracuda/Activation"));
        material.EnableKeyword(kernelName);

        SetTensor(material, "X", X);
        material.SetFloat("_Alpha", alpha);
        material.SetFloat("_Beta",  beta);

        return Dispatch(material, X.dataType, Oshape);
    }

    /// <inheritdoc/>

    public override Tensor Relu(Tensor X)
    {
        if (!X.shape.Is4D())
            return base.Relu(X);
        return Activation("Relu", X);
    }

    /// <inheritdoc/>
    public override Tensor PRelu(Tensor X, Tensor S)
    {
        if (!X.shape.Is4D() && !S.shape.Is4D())
            return base.PRelu(X, S);

        Assert.IsTrue((X.flatWidth == S.flatWidth) || (S.flatWidth == 1));

        var O = X.shape;

        Material material = new Material(PixelShaderSingleton.Instance.FindShader("Barracuda/PRelu"));

        SetTensor(material, "X", X);
        SetTensor(material, "W", S);

        return Dispatch(material, X.dataType, O);
    }

        /// <inheritdoc/>
    public override Tensor Tanh(Tensor X)
    {
        if(!X.shape.Is4D())
            return base.Tanh(X);
        return Activation("Tanh", X);
    }

    /// <inheritdoc/>
    public override Tensor Softplus(Tensor X)
    {
        if(!X.shape.Is4D())
            return base.Softplus(X);
        return Activation("Softplus", X);
    }

    /// <inheritdoc/>
    public override Tensor Sigmoid(Tensor X)
    {
        if(!X.shape.Is4D())
            return base.Sigmoid(X);
        return Activation("Sigmoid", X);
    }

    /// <inheritdoc/>
    public override Tensor HardSigmoid(Tensor X, float alpha, float beta)
    {
        if(!X.shape.Is4D())
            return base.HardSigmoid(X, alpha, beta);
        return Activation("HardSigmoid", X, alpha, beta);
    }

    /// <inheritdoc/>
    public override Tensor Relu6(Tensor X)
    {
        if(!X.shape.Is4D())
            return base.Relu6(X);
        return Activation("Relu6", X);
    }

    /// <inheritdoc/>
    public override Tensor Elu(Tensor X, float alpha)
    {
        if(!X.shape.Is4D())
            return base.Elu(X, alpha);
        return Activation("Elu", X, alpha);
    }

    /// <inheritdoc/>
    public override Tensor LeakyRelu(Tensor X, float alpha)
    {
        if(!X.shape.Is4D())
            return base.LeakyRelu(X, alpha);
        return Activation("LeakyRelu", X, alpha);
    }

    /// <inheritdoc/>
    public override Tensor Selu(Tensor X, float alpha, float gamma)
    {
        if(!X.shape.Is4D())
            return base.Selu(X, alpha, gamma);
        return Activation("Selu", X, alpha, gamma);
    }

    /// <inheritdoc/>
    public override Tensor Swish(Tensor X)
    {
        if(!X.shape.Is4D())
            return base.Swish(X);
        return Activation("Swish", X);
    }

    /// <inheritdoc/>
    public override Tensor Abs(Tensor X)
    {
        if(!X.shape.Is4D())
            return base.Abs(X);
        return Activation("Abs", X);
    }

    /// <inheritdoc/>
    public override Tensor Neg(Tensor X)
    {
        if(!X.shape.Is4D())
            return base.Neg(X);
        return Activation("Neg", X);
    }

    /// <inheritdoc/>
    public override Tensor Ceil(Tensor X)
    {
        if(!X.shape.Is4D())
            return base.Ceil(X);
        return Activation("Ceil", X);
    }

    /// <inheritdoc/>
    public override Tensor Clip(Tensor X, float min, float max)
    {
        if(!X.shape.Is4D())
            return base.Clip(X, min, max);
        return Activation("Clip", X, min, max);
    }

    /// <inheritdoc/>
    public override Tensor Floor(Tensor X)
    {
        if(!X.shape.Is4D())
            return base.Floor(X);
        return Activation("Floor", X);
    }

    /// <inheritdoc/>
    public override Tensor Round(Tensor X)
    {
        if(!X.shape.Is4D())
            return base.Round(X);
        return Activation("Round", X);
    }

    /// <inheritdoc/>
    public override Tensor Reciprocal(Tensor X)
    {
        if(!X.shape.Is4D())
            return base.Reciprocal(X);
        return Activation("Reciprocal", X);
    }

    /// <inheritdoc/>
    public override Tensor Pow(Tensor X, float alpha)
    {
        if(!X.shape.Is4D())
            return base.Pow(X, alpha);
        return Activation("Pow", X, alpha);
    }

    /// <inheritdoc/>
    public override Tensor Exp(Tensor X)
    {
        if(!X.shape.Is4D())
            return base.Exp(X);
        return Activation("Exp", X);
    }

    /// <inheritdoc/>
    public override Tensor Log(Tensor X)
    {
        if(!X.shape.Is4D())
            return base.Log(X);
        return Activation("Log", X);
    }

    /// <inheritdoc/>
    public override Tensor Sqrt(Tensor X)
    {
        if(!X.shape.Is4D())
            return base.Sqrt(X);
        return Activation("Sqrt", X);
    }

    /// <inheritdoc/>
    public override Tensor Acos(Tensor X)
    {
        if(!X.shape.Is4D())
            return base.Acos(X);
        return Activation("Acos", X);
    }

    /// <inheritdoc/>
    public override Tensor Acosh(Tensor X)
    {
        if(!X.shape.Is4D())
            return base.Acosh(X);
        return Activation("Acosh", X);
    }

    /// <inheritdoc/>
    public override Tensor Asin(Tensor X)
    {
        if(!X.shape.Is4D())
            return base.Asin(X);
        return Activation("Asin", X);
    }

    /// <inheritdoc/>
    public override Tensor Asinh(Tensor X)
    {
        if(!X.shape.Is4D())
            return base.Asin(X);
        return Activation("Asinh", X);
    }

    /// <inheritdoc/>
    public override Tensor Atan(Tensor X)
    {
        if(!X.shape.Is4D())
            return base.Atan(X);
        return Activation("Atan", X);
    }

    /// <inheritdoc/>
    public override Tensor Atanh(Tensor X)
    {
        if(!X.shape.Is4D())
            return base.Atanh(X);
        return Activation("Atanh", X);
    }

    /// <inheritdoc/>
    public override Tensor Cos(Tensor X)
    {
        if(!X.shape.Is4D())
            return base.Cos(X);
        return Activation("Cos", X);
    }

    /// <inheritdoc/>
    public override Tensor Cosh(Tensor X)
    {
        if(!X.shape.Is4D())
            return base.Cosh(X);
        return Activation("Cosh", X);
    }

    /// <inheritdoc/>
    public override Tensor Sin(Tensor X)
    {
        if(!X.shape.Is4D())
            return base.Sin(X);
        return Activation("Sin", X);
    }

    /// <inheritdoc/>
    public override Tensor Sinh(Tensor X)
    {
        if(!X.shape.Is4D())
            return base.Sinh(X);
        return Activation("Sinh", X);
    }

    /// <inheritdoc/>
    public override Tensor Tan(Tensor X)
    {
        if(!X.shape.Is4D())
            return base.Tan(X);
        return Activation("Tan", X);
    }

    /// <inheritdoc/>
    public override Tensor Erf(Tensor X)
    {
        if(!X.shape.Is4D())
            return base.Erf(X);
        return Activation("Erf", X);
    }

    /// <inheritdoc/>
    public override Tensor Softmax(Tensor X, int axis)
    {
        if(!X.shape.Is4D())
            return base.Softmax(X, axis);

        axis = X.shape.Axis(axis);
       
        Material material = new Material(PixelShaderSingleton.Instance.FindShader("Barracuda/Softmax"));
    
        if(axis == TensorShape.DataBatch)
            material.EnableKeyword("ReduceN");
        if (axis == TensorShape.H)
            material.EnableKeyword("ReduceH");
        if (axis == TensorShape.W)
            material.EnableKeyword("ReduceW");
        if (axis == TensorShape.C)
            material.EnableKeyword("ReduceC");
    
        SetTensor(material, "X", X);
    
        return Dispatch(material, X.dataType, X.shape);
    }

    /// <inheritdoc/>
    public override Tensor LogSoftmax(Tensor X, int axis)
    {
        if(!X.shape.Is4D())
            return base.LogSoftmax(X, axis);

        axis = X.shape.Axis(axis);
       
        Material material = new Material(PixelShaderSingleton.Instance.FindShader("Barracuda/LogSoftmax"));
    
        if(axis == TensorShape.DataBatch)
            material.EnableKeyword("ReduceN");
        if (axis == TensorShape.H)
            material.EnableKeyword("ReduceH");
        if (axis == TensorShape.W)
            material.EnableKeyword("ReduceW");
        if (axis == TensorShape.C)
            material.EnableKeyword("ReduceC");
    
        SetTensor(material, "X", X);
    
        return Dispatch(material, X.dataType, X.shape);
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

        return Dispatch(material, X.dataType, Oshape);
    }

    /// <inheritdoc/>
    public override Tensor Resample2D(Tensor X, int[] size, bool bilinear)
    {
        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(size.Length, 2);

        var Oshape = new TensorShape(X.batch, size[1], size[0], X.channels);

        Material material = new Material(PixelShaderSingleton.Instance.FindShader(bilinear ? "Barracuda/ResampleBilinear2D" : "Barracuda/Resample2D"));


        SetTensor(material, "X", X);

        return Dispatch(material, X.dataType, Oshape);
    }

    /// <inheritdoc/>
    public override Tensor DepthToSpace(Tensor X, int[] blocksize, Layer.DepthToSpaceMode mode)
    {
        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(blocksize.Length, 2);

        var O = new TensorShape(X.batch, X.height * blocksize[1], X.width * blocksize[0], X.channels / (blocksize[0] * blocksize[1]));


        Material material = new Material(PixelShaderSingleton.Instance.FindShader(m_StringCache.Lookup("Barracuda/DepthToSpace_", mode.ToString())));

        SetTensor(material, "X", X);

        material.SetVector("_Pool", new Vector4(blocksize[0], blocksize[1], 0, 0));

        return Dispatch(material, X.dataType, O);
    }

    /// <inheritdoc/>
    public override Tensor SpaceToDepth(Tensor X, int[] blocksize)
    {
        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(blocksize.Length, 2);

        var O = new TensorShape(X.batch, X.height / blocksize[1], X.width / blocksize[0], X.channels * (blocksize[0] * blocksize[1]));


        Material material = new Material(PixelShaderSingleton.Instance.FindShader("Barracuda/SpaceToDepth"));

        SetTensor(material, "X", X);

        material.SetVector("_Pool", new Vector4(blocksize[0], blocksize[1], 0, 0));

        return Dispatch(material, X.dataType, O);
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

        Material material = new Material(PixelShaderSingleton.Instance.FindShader("Barracuda/Concat"));

        var dataType = tensors.Length > 0 ? tensors[0].dataType : DataType.Float;
        var O = NewTensor(dataType, Oshape, AllocScope.LayerOutput, "O");
        var Opred = NewTensor(dataType, Oshape, AllocScope.LayerOutput, "O");

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
        
        return Dispatch(material, X.dataType, Oshape);
    }

    /// <inheritdoc/>
    public override Tensor Tile(Tensor X, int[] repeats)
    {
        var O = X.shape.Scale(repeats);

        Material material = new Material(PixelShaderSingleton.Instance.FindShader("Barracuda/Tile"));

        SetTensor(material, "X", X);

        return Dispatch(material, X.dataType, O);
    }

    /// <inheritdoc/>
    public override Tensor Gather(Tensor[] tensors, int axis)
    {
        Tensor X = tensors[0];
        Tensor indices = tensors[1];

        var O = X.shape;
        O[axis] = indices.length;

        Material material = new Material(PixelShaderSingleton.Instance.FindShader("Barracuda/Gather"));
        SetTensor(material, "X", X);
        SetTensor(material, "K", indices);
        material.SetInt("_Axis", axis == TensorShape.DataBatch ? 0 : axis - 4);

        return Dispatch(material, X.dataType, O);
    }

    /// <inheritdoc/>
    public override Tensor ScatterND(Tensor X, Tensor indices, Tensor updates, Layer.ScatterNDReductionMode reduction)
    {
        // only support for scattering on C for now
        Assert.IsTrue(indices.batch == X.batch);
        Assert.IsTrue(updates.width == X.width && updates.height == X.height);
        var O = X.shape;
           
        Material material = new Material(PixelShaderSingleton.Instance.FindShader("Barracuda/ScatterND"));
        SetTensor(material, "X", X);
        SetTensor(material, "K", indices);
        SetTensor(material, "W", updates);

        if (reduction == Layer.ScatterNDReductionMode.None)
            material.EnableKeyword("ReduceNone");
        else if (reduction == Layer.ScatterNDReductionMode.Add)
            material.EnableKeyword("ReduceAdd");
        else if (reduction == Layer.ScatterNDReductionMode.Mul)
            material.EnableKeyword("ReduceMul");

        return Dispatch(material, X.dataType, O);
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


        return Dispatch(material, X.dataType, X.shape);
    }

    /// <inheritdoc/>
    public override Tensor Transpose(Tensor X, int[] permutations)
    {
        if (X.shape.Is4D())
            return base.Transpose(X, permutations);

        Material material = new Material(Shader.Find("Barracuda/Transpose"));

        SetTensor(material, "X", X);
    
    
        material.SetVector("_Pool", new Vector4(Array.IndexOf(permutations, 0), Array.IndexOf(permutations, 1), Array.IndexOf(permutations, 2), Array.IndexOf(permutations, 3)));
    
        return Dispatch(material, X.dataType, X.shape.Permute(permutations));
    }
    
    /// <inheritdoc/>
    public override Tensor Reshape(Tensor X, TensorShape newShape)
    {
        if (X.shape == newShape)
            return Copy(X);
    
        Material material = new Material(PixelShaderSingleton.Instance.FindShader("Barracuda/Copy"));
    
        SetTensor(material, "X", X);
    
        return Dispatch(material, X.dataType, newShape);
    }

    /// <inheritdoc/>
    public override Tensor Flatten(Tensor X)
    {
        var newShape = X.shape.Flatten();
        if (X.shape == newShape)
            return base.Flatten(X);

        return Reshape(X, newShape);
    }

    /// <inheritdoc/>
    public override Tensor Copy(Tensor X)
    {
        var O = NewTensor(X.dataType, X.shape, AllocScope.LayerOutput, "O");
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
