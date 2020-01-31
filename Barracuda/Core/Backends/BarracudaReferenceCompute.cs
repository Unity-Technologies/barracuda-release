//#define DEBUG_TRACK_ALLOCATIONS

using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Experimental.Rendering; // AsyncGPUReadback
using UnityEngine.Assertions;
using UnityEngine.Profiling;
using System;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;


namespace Barracuda {

public class ComputeTensorData : ITensorData
{
    private bool m_DisposeBufferAfterUse;
    private ComputeBuffer m_Buffer;
    private TensorShape m_Shape;
    private int m_Offset;

    public ComputeBuffer buffer { get { return m_Buffer; } }
    public TensorShape shape { get { return m_Shape; } }
    public int offset { get { return m_Offset; } }
    public string name;

#if DEBUG_TRACK_ALLOCATIONS
    protected StackTrace m_AllocationTrace;
#endif

    public ComputeTensorData(TensorShape shape, string buffername, bool clearOnInit = true)
    {
        name = buffername;
        m_Buffer = new ComputeBuffer(shape.length, sizeof(float));

        // @TODO: consider zero initialization only for "debug" mode
        if (clearOnInit)
        {
            float[] zeros = new float[shape.length];
            m_Buffer.SetData(zeros);
        }

        m_Shape = shape;
        m_Offset = 0;

        m_DisposeBufferAfterUse = true;

#if DEBUG_TRACK_ALLOCATIONS
        m_AllocationTrace = new System.Diagnostics.StackTrace();
#endif
    }

    public ComputeTensorData(ComputeBuffer buffer, TensorShape shape, int offset, string buffername)
    {
        name = buffername;
        m_Buffer = buffer;
        m_Shape = shape;
        m_Offset = offset;

        m_DisposeBufferAfterUse = false;
    }

    ~ComputeTensorData()
    {
        if (m_Buffer == null)
            return;
        if (!m_DisposeBufferAfterUse)
            return;

        D.LogWarning("Found undisposed " + ToString() + ". Disposing!");

        Dispose();
    }

    public virtual void Dispose()
    {
        if (m_DisposeBufferAfterUse)
        {
            m_Buffer.Dispose();
            m_Buffer = null;
        }
        m_DisposeBufferAfterUse = false;
    }

    public virtual void Reserve(int count)
    {
        if (m_Offset + count > GetMaxCount())
            throw new ArgumentException("ComputeTensorData buffer is too small to reserve " + count + " elements.");
    }

    public virtual void Upload(float[] data, int offset = 0, int count = -1)
    {
        Assert.IsTrue(offset >= 0);
        if (count < 0)
            count = Math.Min(GetMaxCount(), data.Length) - offset;
        Assert.IsTrue(offset + count <= data.Length);

        m_Buffer.SetData(data, offset, m_Offset, count);
        #if UNITY_2018
        m_AsyncDownloadRequested = false;
        #endif
    }

    #if UNITY_2018
    private bool m_AsyncDownloadRequested = false;
    private AsyncGPUReadbackRequest m_AsyncDownloadRequest;
    public virtual bool ScheduleAsyncDownload(int count)
    {
        if (!SystemInfo.supportsAsyncGPUReadback)
            return true;

        if (!m_AsyncDownloadRequested)
        {
            m_AsyncDownloadRequest = AsyncGPUReadback.Request(m_Buffer);
            m_AsyncDownloadRequested = true;
        }
        else
            m_AsyncDownloadRequest.Update();
        return m_AsyncDownloadRequest.done;
    }
    #else
    public virtual bool ScheduleAsyncDownload(int count)
    {
        return true;
    }
    #endif

    public virtual float[] Download(int count)
    {
        //;;D.logStackTraceEnabled = true;
        //;;Debug.Log("Download ComputeTensorData " + name + + GetMaxCount() + " " + count);
        //;;D.logStackTraceEnabled = false;

        Profiler.BeginSample("Barracuda.DownloadDataFromGPU");
        Assert.IsTrue(GetMaxCount() >= count);
        count = Math.Min(GetMaxCount(), count);

        #if UNITY_2018
        if (m_AsyncDownloadRequested)
        {
            m_AsyncDownloadRequested = false;
            m_AsyncDownloadRequest.WaitForCompletion();
            Profiler.EndSample();

            if (!m_AsyncDownloadRequest.hasError)
                return m_AsyncDownloadRequest.GetData<float>().ToArray();
        }
        #endif

        var data = new float[count];
        m_Buffer.GetData(data, 0, m_Offset, count);
        Profiler.EndSample();

        return data;
    }

    public virtual float[] SharedAccess(out int offset)
    {
        offset = m_Offset;
        return Download(GetMaxCount());
    }

    public virtual int GetMaxCount()
    {
        return m_Buffer.count;
    }

    public override string ToString()
    {
        string allocationSource = "";

#if DEBUG_TRACK_ALLOCATIONS
        allocationSource += "\nSource:\n" + m_AllocationTrace;
#endif

        return string.Format("(GPU:{0}#{1} {2} buffer: {3} created at: {4})",
            name, GetHashCode(), m_Shape, m_Buffer, allocationSource);
    }
}

public class SharedComputeTensorData : ComputeTensorData
{
    public SharedComputeTensorData(ComputeBuffer buffer, TensorShape shape, int offset = 0, string buffername = "") : base(buffer, shape, offset, buffername) {}
}

public class TextureFormatUtils
{
    public static bool IsRedOnly(TextureFormat format)
    {
        return  format == TextureFormat.R8 ||
                format == TextureFormat.R16 ||
                format == TextureFormat.RHalf ||
                format == TextureFormat.RFloat;
    }

    public static bool IsRedOnly(RenderTextureFormat format)
    {
        return  format == RenderTextureFormat.R8 ||
                format == RenderTextureFormat.R16 ||
                format == RenderTextureFormat.RHalf ||
                format == RenderTextureFormat.RFloat;
    }

    public static bool IsRedGreen(TextureFormat format)
    {
        return  format == TextureFormat.RG16 ||
                format == TextureFormat.RGHalf ||
                format == TextureFormat.RGFloat;
    }

    public static bool IsRedGreen(RenderTextureFormat format)
    {
        return  format == RenderTextureFormat.RG16 ||
                format == RenderTextureFormat.RGHalf ||
                format == RenderTextureFormat.RGFloat;
    }

    public static bool IsAlphaOnly(Texture tex)
    {
        var tex2D = tex as Texture2D;
        var texArr = tex as Texture2DArray;
        var tex3D = tex as Texture3D;
        if (tex2D != null)
            return tex2D.format == TextureFormat.Alpha8;
        else if (texArr != null)
            return texArr.format == TextureFormat.Alpha8;
        else if (tex3D != null)
            return tex3D.format == TextureFormat.Alpha8;
        else
            return false;
    }

    public static bool IsRedOnly(Texture tex)
    {
        var tex2D = tex as Texture2D;
        var texArr = tex as Texture2DArray;
        var tex3D = tex as Texture3D;
        var rt = tex as RenderTexture;

        if (tex2D != null)
            return IsRedOnly(tex2D.format);
        else if (texArr != null)
            return IsRedOnly(texArr.format);
        else if (tex3D != null)
            return IsRedOnly(tex3D.format);
        else if (rt != null)
            return IsRedOnly(rt.format);
        else
            return false;
    }

    public static bool IsRedGreen(Texture tex)
    {
        var tex2D = tex as Texture2D;
        var texArr = tex as Texture2DArray;
        var tex3D = tex as Texture3D;
        var rt = tex as RenderTexture;

        if (tex2D != null)
            return IsRedGreen(tex2D.format);
        else if (texArr != null)
            return IsRedGreen(texArr.format);
        else if (tex3D != null)
            return IsRedGreen(tex3D.format);
        else if (rt != null)
            return IsRedGreen(rt.format);
        else
            return false;
    }


    public static Color FormatToChannelMask(Texture tex, int interpretPixelAsChannels)
    {
        switch (interpretPixelAsChannels)
        {
            case 1:
                if (IsRedOnly(tex))
                    return new Color(1,0,0,0);
                if (IsAlphaOnly(tex))
                    return new Color(0,0,0,1);
                // TODO: known issue, doesn't handle RG textures properly
                return new Color(0,0,0,0); // see specialCaseWhenChannelMaskIsEmptyStoresAverage
            case 2:
                return new Color(1,1,0,0);
            case 3:
                return new Color(1,1,1,0);
            case 4:
            default:
                return new Color(1,1,1,1);
        }
    }

    public static Color FormatToChannelMask(Texture tex)
    {
        if (IsRedOnly(tex))
            return new Color(1,0,0,1);
        if (IsRedGreen(tex))
            return new Color(1,1,0,1);
        if (IsAlphaOnly(tex))
            return new Color(0,0,0,1);
        return new Color(1,1,1,1);
    }
}

public class TextureAsTensorData : ITensorData
{
    public enum Flip
    {
        None,
        Y,
    }

    public enum InterpretDepthAs
    {
        Batch,
        Channels,
    }

    public enum InterpretColorAs
    {
        AverageMultipleChannels,
        // TODO: PickFirstChannel,
    }

    private TensorShape m_Shape;
    private Texture[] m_Textures;
    private int m_InterpretPixelAsChannels;
    private InterpretDepthAs m_InterpretDepthAs;
    private InterpretColorAs m_InterpretColorAs;
    private Flip m_Flip;

    public TensorShape shape { get { return m_Shape; } }
    public Texture[] textures { get { return m_Textures; } }
    public int interpretPixelAsChannels { get { return m_InterpretPixelAsChannels; } }
    public InterpretDepthAs interpretDepthAs { get { return m_InterpretDepthAs; } }
    public InterpretColorAs interpretColorAs { get { return m_InterpretColorAs; } }
    public Flip flip { get { return m_Flip; } }


    public TextureAsTensorData(Texture[] textures, int interpretPixelAsChannels = 3,
        Flip flip = Flip.Y, InterpretDepthAs depthAs = InterpretDepthAs.Batch, InterpretColorAs colorAs = InterpretColorAs.AverageMultipleChannels)
    {
        m_InterpretPixelAsChannels = interpretPixelAsChannels;
        m_InterpretDepthAs = depthAs;
        m_InterpretColorAs = colorAs;
        m_Flip = flip;

        if (textures.Length < 1)
            throw new ArgumentException("Textures array must be non empty");

        var width = textures[0].width;
        var height = textures[0].height;

        var totalDepth = 0;
        foreach (var tex in textures)
        {
            if (tex.width != width || tex.height != height)
                throw new ArgumentException("All textures must have the same width and height dimensions");

            var tex2D = tex as Texture2D;
            var texArr = tex as Texture2DArray;
            var tex3D = tex as Texture3D;
            var rt = tex as RenderTexture;
            if (tex2D)
                totalDepth += 1;
            else if (texArr)
                totalDepth += texArr.depth;
            else if (tex3D)
                totalDepth += tex3D.depth;
            else if (rt)
                totalDepth += rt.volumeDepth;
            else
                throw new InvalidOperationException("Unsupported texture type");
        }
        m_Textures = textures;

        int batch = 1;
        int channels = interpretPixelAsChannels;
        if (m_InterpretDepthAs == InterpretDepthAs.Batch)
            batch *= totalDepth;
        else if (m_InterpretDepthAs == InterpretDepthAs.Channels)
            channels *= totalDepth;

        m_Shape = new TensorShape(batch, height, width, channels);
    }

    public TextureAsTensorData(Texture texture, int interpretPixelAsChannels = 3,
        Flip flip = Flip.Y, InterpretDepthAs depthAs = InterpretDepthAs.Batch, InterpretColorAs colorAs = InterpretColorAs.AverageMultipleChannels)
    : this(new [] { texture }, interpretPixelAsChannels, flip, depthAs, colorAs) {}

    public virtual void Reserve(int count)
    {
        // currently always readonly
        throw new InvalidOperationException("TextureAsTensorData is readonly");
    }

    public virtual void Upload(float[] data, int offset = 0, int count = -1)
    {
        // currently always readonly
        throw new InvalidOperationException("TextureAsTensorData is readonly");
    }

    static void ProcessLine(Color[] pixels, int srcOffset, int srcWidth, Color srcChannelMask, float[] dstArray, int dstOffset, Color dstChannelMask, int channelStride)
    {
        for (var x = 0; x < srcWidth; ++x)
        {
            var p = pixels[srcOffset + x];
            var dst = dstOffset;
            if (dstChannelMask[0] > 0) dstArray[dst++] = p.r * srcChannelMask[0];
            if (dstChannelMask[1] > 0) dstArray[dst++] = p.g * srcChannelMask[1];
            if (dstChannelMask[2] > 0) dstArray[dst++] = p.b * srcChannelMask[2];
            if (dstChannelMask[3] > 0) dstArray[dst++] = p.a * srcChannelMask[3];
            var specialCaseWhenChannelMaskIsEmptyStoresAverage = (dst == dstOffset);
            if (specialCaseWhenChannelMaskIsEmptyStoresAverage)
                dstArray[dst++] = (p.r + p.g + p.b) / 3;

            dstOffset += channelStride;
        }
    }

    public virtual bool ScheduleAsyncDownload(int count)
    {
        return true;
    }

    public virtual float[] Download(int count)
    {
        //;;D.logStackTraceEnabled = true;
        //;;Debug.Log("Download TextureAsTensorData " + name + " " + count + " @ " + ToString());
        //;;D.logStackTraceEnabled = false;

        Assert.AreEqual(shape.length, count);
        var data = new float[shape.length];
        int batch = 0;
        var dstChannel = 0;
        foreach (var tex in m_Textures)
        {
            var tex2D = tex as Texture2D;
            var texArr = tex as Texture2DArray;
            var tex3D = tex as Texture3D;
            // Source channel mask is a workaround - since Unity API that does not adhere to DX/GL standard when reading from 1 channel textures!
            var srcChannelMask = TextureFormatUtils.FormatToChannelMask(tex);
            var dstChannelMask = TextureFormatUtils.FormatToChannelMask(tex, m_InterpretPixelAsChannels);

            if (tex2D)
            {
                var pixels = tex2D.GetPixels(0);

                for (var y = 0; y < tex.height; ++y)
                {
                    var srcOffset = y * tex.width;
                    var dstY = (m_Flip == Flip.Y) ? tex.height - y - 1: y;
                    var dstOffset = shape.Index(batch, dstY, 0, dstChannel);
                    ProcessLine(pixels, srcOffset, tex.width, srcChannelMask, data, dstOffset, dstChannelMask, shape.channels);
                }

                if (m_InterpretDepthAs == InterpretDepthAs.Batch)
                    batch += 1;
                else if (m_InterpretDepthAs == InterpretDepthAs.Channels)
                    dstChannel += m_InterpretPixelAsChannels;
            }
            else if (texArr)
            {
                for (var z = 0; z < texArr.depth; ++z)
                {
                    var pixels = texArr.GetPixels(z, 0);

                    D.Log(dstChannel);
                    for (var y = 0; y < tex.height; ++y)
                    {
                        var srcOffset = y * tex.width;
                        var dstY = (m_Flip == Flip.Y) ? tex.height - y - 1: y;
                        var dstOffset = shape.Index(batch, dstY, 0, dstChannel);
                        ProcessLine(pixels, srcOffset, tex.width, srcChannelMask, data, dstOffset, dstChannelMask, shape.channels);
                    }

                    if (m_InterpretDepthAs == InterpretDepthAs.Batch)
                        batch += 1;
                    else if (m_InterpretDepthAs == InterpretDepthAs.Channels)
                        dstChannel += m_InterpretPixelAsChannels;
                }
            }
            else if (tex3D)
            {
                var pixels = tex3D.GetPixels(0);
                for (var z = 0; z < tex3D.depth; ++z)
                {
                    for (var y = 0; y < tex.height; ++y)
                    {
                        var srcOffset = z * tex.height + y * tex.width;
                        var dstY = (m_Flip == Flip.Y) ? tex.height - y - 1: y;
                        var dstOffset = shape.Index(batch, dstY, 0, dstChannel);
                        ProcessLine(pixels, srcOffset, tex.width, srcChannelMask, data, dstOffset, dstChannelMask, shape.channels);
                    }

                    if (m_InterpretDepthAs == InterpretDepthAs.Batch)
                        batch += 1;
                    else if (m_InterpretDepthAs == InterpretDepthAs.Channels)
                        dstChannel += m_InterpretPixelAsChannels;
                }
            }
            else
                throw new InvalidOperationException("Unsupported texture type for automatic readback to CPU");
        }

        return data;
    }

    public virtual float[] SharedAccess(out int offset)
    {
        offset = 0;
        return Download(shape.length);
    }

    public virtual int GetMaxCount()
    {
        return m_Shape.length;
    }

    public virtual void Dispose()
    {
    }
}

public class ReferenceComputeOps : ReferenceCPUOps
{
    private ComputeShader m_Kernels;
    private float[] m_SyncBuffer = new float[1];

    public ReferenceComputeOps(ComputeShader kernels, ITensorAllocator allocator = null)
    : base(allocator)
    {
        m_Kernels = kernels;
    }

    public ComputeTensorData Pin(Tensor X)
    {
        X.FlushCache();

        var onDevice = X.tensorOnDevice as ComputeTensorData;
        if (onDevice == null)
        {
            var asTexture = X.tensorOnDevice as TextureAsTensorData;
            if (asTexture != null)
                X.PinToDeviceAndDownloadFromIt(TextureToTensorData(asTexture, X.name));
            else
                X.PinToDeviceAndUploadToIt(new ComputeTensorData(X.shape, X.name));
        }

        Assert.IsNotNull(X.tensorOnDevice as ComputeTensorData);
        Assert.IsNotNull((X.tensorOnDevice as ComputeTensorData).buffer);

        return X.tensorOnDevice as ComputeTensorData;
    }

    public override void WaitForCompletion(Tensor x)
    {
        var data = x.tensorOnDevice as ComputeTensorData;

        if (data != null)
        {
            data.buffer.GetData(m_SyncBuffer, 0, 0, 1);
        }
    }

    public void SetTensor(ComputeFunc fn, string name, Tensor X)
    {
        var XonDevice = Pin(X);
        fn.SetTensor(name, X.shape, XonDevice.buffer, XonDevice.offset);
    }

    public Tensor NewTensor(ComputeFunc fn, string name, TensorShape shape)
    {
        var o = NewTensor(shape, name);
        fn.SetTensor(name, shape, Pin(o).buffer);
        return o;
    }

    public Tensor Dispatch(ComputeFunc fn, TensorShape outputShape, int workItemsX, int workItemsY, int workItemsZ, string outputName = "O")
    {
        var o = NewTensor(fn, outputName, outputShape);
        fn.Dispatch(workItemsX, workItemsY, workItemsZ);
        return o;
    }

    // ---------------------------------------------------------------------------------

    protected ITensorData TextureToTensorData(TextureAsTensorData texData, string name)
    {
        var fn = new ComputeFunc(m_Kernels, "TextureToTensor");
        var tensorData = new ComputeTensorData(texData.shape, name, false);

        fn.SetTensor("O", texData.shape, tensorData.buffer);
        fn.shader.SetBool("_FlipY", texData.flip == TextureAsTensorData.Flip.Y);

        var offsets = new int[] { 0,0,0,0 };
        foreach (var tex in texData.textures)
        {
            var texArr = tex as Texture2DArray;
            var tex3D = tex as Texture3D;
            var rt = tex as RenderTexture;

            var texDepth = 1;
            if (texArr)
                texDepth = texArr.depth;
            else if (tex3D)
                texDepth = tex3D.depth;
            else if (rt)
                texDepth = rt.volumeDepth;

            var srcChannelMask = TextureFormatUtils.FormatToChannelMask(tex, texData.interpretPixelAsChannels);

            fn.SetTexture("X", tex);
            fn.shader.SetInts("_Pool", new int [] {tex.width, tex.height});
            fn.shader.SetInts("_Pad", offsets);
            fn.shader.SetInts("_ChannelWriteMask", new [] {(int)srcChannelMask[0], (int)srcChannelMask[1], (int)srcChannelMask[2], (int)srcChannelMask[3] });

            fn.Dispatch(texData.shape.width, texData.shape.height, texDepth);

            if (texData.interpretDepthAs == TextureAsTensorData.InterpretDepthAs.Batch)
                offsets[0] += texDepth;
            else if (texData.interpretDepthAs == TextureAsTensorData.InterpretDepthAs.Channels)
                offsets[3] += texDepth * texData.interpretPixelAsChannels;
        }

        return tensorData;
    }

    public void TensorToRenderTexture(Tensor X, RenderTexture target,
                                        int batch = 0, int fromChannel = 0, float scale = 1.0f, float bias = 0f)
    {
        if (!target.enableRandomWrite || !target.IsCreated())
        {
            target.Release();
            target.enableRandomWrite = true;
            target.Create();
        }

        var fn = new ComputeFunc(m_Kernels, "TensorToTexture");
        SetTensor(fn, "X", X);
        fn.SetTexture("O", target);
        fn.shader.SetFloat("_Alpha", scale);
        fn.shader.SetFloat("_Beta", bias);
        fn.shader.SetInts("_Pad", new int[] { batch, 0, 0, fromChannel });
        fn.shader.SetBool("_FlipY", true);
        fn.Dispatch(target.width, target.height, 1);
    }

    // ---------------------------------------------------------------------------------

    public override Tensor Dense(Tensor X, Tensor W, Tensor B)
    {
        Assert.IsTrue(W.dimensions <= 2);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(X.flatWidth, W.flatHeight);

        var O = new TensorShape(X.flatHeight, W.flatWidth);

        var fn = new ComputeFunc(m_Kernels, "Dense");

        SetTensor(fn, "X", X);
        SetTensor(fn, "W", W);
        SetTensor(fn, "B", B);

        return Dispatch(fn, O, O.flatWidth, O.flatHeight, 1);
    }

    static public int IDivC(int v, int div)
    {
        return (v + div - 1) / div;
    }

    public override Tensor Conv2D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad)
    {
        Assert.AreEqual(X.channels, K.kernelDepth);
        Assert.AreEqual(K.kernelCount, B.flatWidth);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);

        var O = X.shape.ApplyKernel(K.shape, stride, pad);

        bool useWinograd = K.kernelWidth == 3 && K.kernelHeight == 3 && stride[0] == 1 && stride[1] == 1 && O.width % 2 == 0 && O.height % 2 == 0;
        if( useWinograd )
        {
            var fnw = new ComputeFunc(m_Kernels, "Conv2DWinograd_2x2_3x3");
            SetTensor(fnw, "X", X);
            SetTensor(fnw, "K", K);
            SetTensor(fnw, "B", B);
            fnw.shader.SetInts("_Pad", pad);

            var ow = Dispatch(fnw, O, K.kernelCount, IDivC(O.width,2), IDivC(O.height,2));
            return ow;
        }

        var fn = new ComputeFunc(m_Kernels, "Conv2D");

        SetTensor(fn, "X", X);
        SetTensor(fn, "K", K);
        SetTensor(fn, "B", B);
        fn.shader.SetInts("_Stride", stride);
        fn.shader.SetInts("_Pad", pad);

        var o = Dispatch(fn, O, K.kernelCount, O.width, O.height);
        return o;
    }

    public override Tensor DepthwiseConv2D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad)
    {
        if (K.kernelDepth != 1)
            return base.DepthwiseConv2D(X, K, B, stride, pad);

        Assert.AreEqual(K.kernelDepth, 1);
        Assert.AreEqual(K.kernelCount, X.channels);
        Assert.AreEqual(K.kernelCount, B.flatWidth);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);

        var O = X.shape.ApplyKernel(K.shape, stride, pad);

        var fn = new ComputeFunc(m_Kernels, "DepthwiseConv2D");

        SetTensor(fn, "X", X);
        SetTensor(fn, "K", K);
        SetTensor(fn, "B", B);
        fn.shader.SetInts("_Stride", stride);
        fn.shader.SetInts("_Pad", pad);

        var o = Dispatch(fn, O, K.kernelCount, O.width, O.height);
        return o;
    }

    public override Tensor Conv2DTrans(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, int[] outputAdjustment)
    {
        Assert.AreEqual(X.channels, K.kernelDepth);
        Assert.AreEqual(K.kernelCount, B.flatWidth);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);

        var O = X.shape.ApplyKernelInverse(K.shape, stride, pad, outputAdjustment);

        // one pass version
        pad = new int[]
        {
            K.kernelWidth - pad[0] - 1, K.kernelHeight - pad[1] - 1,
            K.kernelWidth - pad[2] - 1, K.kernelHeight - pad[3] - 1
        };

        var fn = new ComputeFunc(m_Kernels, "Conv2DTrans");

        SetTensor(fn, "X", X);
        SetTensor(fn, "K", K);
        SetTensor(fn, "B", B);
        fn.shader.SetInts("_Stride", stride);
        fn.shader.SetInts("_Pad", pad);

        return Dispatch(fn, O, K.kernelCount, O.width, O.height);
    }

    public override Tensor Upsample2D(Tensor X, int[] size)
    {
        Assert.AreEqual(size.Length, 2);

        var O = new TensorShape(X.batch, X.height*size[1], X.width*size[0], X.channels);

        var fn = new ComputeFunc(m_Kernels, "Upsample2D");

        SetTensor(fn, "X", X);

        fn.shader.SetInts("_Pool", size);

        return Dispatch(fn, O, X.channels, X.width, X.height);
    }

    protected virtual Tensor ApplyPadding(Tensor X, int[] pad, string kernelName, float constant = 0.0f)
    {
        Assert.AreEqual(pad.Length, 4);

        var O = X.shape.ApplyBorder(pad);

        var fn = new ComputeFunc(m_Kernels, kernelName);

        SetTensor(fn, "X", X);

        fn.shader.SetInts("_Pad", pad);

        if (kernelName == "Border2D")
        {
            // NOTE: negative "pad" variable will crop X tensor
            int croppedWidth = X.width - Math.Max(0, -pad[2]);
            int croppedHeight = X.height - Math.Max(0, -pad[3]);
            var croppedSize = new int[] { 0, 0 };
            croppedSize[0] = croppedWidth;
            croppedSize[1] = croppedHeight;

            fn.shader.SetInts("_Pool", croppedSize);
            fn.shader.SetFloat("_Beta", constant);
        }

        return Dispatch(fn, O, O.channels, O.width, O.height);
    }

    public override Tensor Border2D(Tensor X, int[] pad, float constant)
    {
        return ApplyPadding(X, pad, "Border2D", constant);
    }

    public override Tensor Pad2DEdge(Tensor X, int[] pad)
    {
        return ApplyPadding(X, pad, "Pad2DEdge");
    }

    public override Tensor Pad2DReflect(Tensor X, int[] pad)
    {
        return ApplyPadding(X, pad, "Pad2DReflect");
    }

    public override Tensor Pad2DSymmetric(Tensor X, int[] pad)
    {
        return ApplyPadding(X, pad, "Pad2DSymmetric");
    }

    protected virtual Tensor Pool2D(string kernelName, Tensor X, int[] pool, int[] stride, int[] pad)
    {
        Assert.AreEqual(pool.Length, 2);
        Assert.AreEqual(stride.Length, 2);

        var O = X.shape.ApplyPool(pool, stride, pad);

        var fn = new ComputeFunc(m_Kernels, kernelName);

        SetTensor(fn, "X", X);
        fn.shader.SetInts("_Pool", pool);
        fn.shader.SetInts("_Stride", stride);
        fn.shader.SetInts("_Pad", pad);

        return Dispatch(fn, O, O.channels, O.width, O.height);
    }

    public override Tensor MaxPool2D(Tensor X, int[] pool, int[] stride, int[] pad)
    {
        return Pool2D("MaxPool2D", X, pool, stride, pad);
    }

    public override Tensor AvgPool2D(Tensor X, int[] pool, int[] stride, int[] pad)
    {
        return Pool2D("AvgPool2D", X, pool, stride, pad);
    }

    protected virtual Tensor GlobalPool2D(string kernelName, Tensor X)
    {
        var O = new TensorShape(X.batch, 1, 1, X.channels);

        var fn = new ComputeFunc(m_Kernels, kernelName);

        SetTensor(fn, "X", X);

        return Dispatch(fn, O, O.channels, 1, 1);
    }

    public override Tensor GlobalMaxPool2D(Tensor X)
    {
        return GlobalPool2D("GlobalMaxPool2D", X);
    }

    public override Tensor GlobalAvgPool2D(Tensor X)
    {
        return GlobalPool2D("GlobalAvgPool2D", X);
    }

    public override Tensor GlobalAvgVariancePool2D(Tensor X)
    {
        var O = new TensorShape(X.batch, 2, 1, X.channels);

        var fn = new ComputeFunc(m_Kernels, "GlobalAvgVariancePool2D");

        SetTensor(fn, "X", X);

        return Dispatch(fn, O, O.channels, 1, 1);
    }

    public override Tensor ScaleBias(Tensor X, Tensor S, Tensor B)
    {
        Assert.AreEqual(X.channels, B.channels); Assert.AreEqual(X.channels, S.channels);
        Assert.AreEqual(B.length, B.channels); Assert.AreEqual(S.length, S.channels);

        var O = X.shape;
        var fn = new ComputeFunc(m_Kernels, "ScaleBias");

        SetTensor(fn, "X", X);
        SetTensor(fn, "W", S);
        SetTensor(fn, "B", B);

        return Dispatch(fn, O, O.channels, O.width, O.height);
    }

    public override Tensor Normalization(Tensor X, Tensor S, Tensor B, int pool, int axis, float epsilon)
    {
        if (axis != 3 && axis != -1)
            return base.Normalization(X, S, B, pool, axis, epsilon);

        if (pool == 1 && X.batch != 1)
            return base.Normalization(X, S, B, pool, axis, epsilon); // @TODO: Instance Normalization with batch > 1

        if (pool <= 0)
            pool = X.batch;

        var O = X.shape;
        var fn = new ComputeFunc(m_Kernels, "InstanceNorm");
        fn.shader.SetFloat("_Epsilon", epsilon);

        SetTensor(fn, "X", X);
        SetTensor(fn, "W", S);
        SetTensor(fn, "B", B);

        return Dispatch(fn, O, O.channels, 1, 1);
    }

    // @TODO: debug & fix
    public override Tensor Dropout(Tensor X, float alpha)
    {
        Assert.IsTrue(alpha >= 0f && alpha <= 1f);

        var O = X.shape;
        var fn = new ComputeFunc(m_Kernels, "Dropout");

        SetTensor(fn, "X", X);
        fn.shader.SetFloat("_Alpha", alpha);
        fn.shader.SetFloat("_Seed", UnityEngine.Random.value);

        return Dispatch(fn, O, O.channels, O.width, O.height);
    }

    protected virtual Tensor Activation(string kernelName, Tensor X, float alpha = 0f, float beta = 0f)
    {
        var O = X.shape;
        var fn = new ComputeFunc(m_Kernels, kernelName);

        SetTensor(fn, "X", X);
        fn.shader.SetFloat("_Alpha", alpha);
        fn.shader.SetFloat("_Beta",  beta);

        return Dispatch(fn, O, O.channels, O.width, O.height);
    }

    public override Tensor Relu(Tensor X)
    {
        return Activation("Relu", X);
    }

    public override Tensor Selu(Tensor X, float alpha, float gamma)
    {
        return Activation("Selu", X, alpha, gamma);
    }

    public override Tensor Neg(Tensor X)
    {
        return Activation("Neg", X);
    }

    public override Tensor Swish(Tensor X)
    {
        return Activation("Swish", X);
    }

    public override Tensor Tanh(Tensor X)
    {
        return Activation("Tanh", X);
    }

    public override Tensor Sigmoid(Tensor X)
    {
        return Activation("Sigmoid", X);
    }

    public override Tensor Elu(Tensor X, float alpha)
    {
        return Activation("Elu", X, alpha);
    }

    public override Tensor Relu6(Tensor X)
    {
        return Activation("Relu6", X);
    }

    public override Tensor LeakyRelu(Tensor X, float alpha)
    {
        return Activation("LeakyRelu", X, alpha);
    }

    public override Tensor PRelu(Tensor X, Tensor S)
    {
        Assert.IsTrue((X.flatWidth == S.flatWidth) || (S.flatWidth == 1));

        var O = X.shape;
        var fn = new ComputeFunc(m_Kernels, "PRelu");

        SetTensor(fn, "X", X);
        SetTensor(fn, "W", S);

        return Dispatch(fn, O, O.channels, O.width, O.height);
    }

    public override Tensor Exp(Tensor X)
    {
        return Activation("Exp", X);
    }

    public override Tensor Log(Tensor X)
    {
        return Activation("Log", X);
    }

    public override Tensor Sqrt(Tensor X)
    {
        return Activation("Sqrt", X);
    }

    public override Tensor Pow(Tensor X, float alpha)
    {
        return Activation("Pow", X, alpha);
    }

    public override Tensor Clip(Tensor X, float min, float max)
    {
        return Activation("Clip", X, min, max);
    }

    public override Tensor Softmax(Tensor X)
    {
        var O = X.shape.Flatten();

        var fn = new ComputeFunc(m_Kernels, "Softmax");

        SetTensor(fn, "X", X);

        return Dispatch(fn, O, O.flatWidth, O.flatHeight, 1);
    }

    public override Tensor LogSoftmax(Tensor X)
    {
        var O = X.shape.Flatten();

        var fn = new ComputeFunc(m_Kernels, "LogSoftmax");

        SetTensor(fn, "X", X);

        return Dispatch(fn, O, O.flatWidth, O.flatHeight, 1);
    }

    public override Tensor Concat(Tensor[] tensors, int axis)
    {
        if (axis != 3 && axis != -1)
            return base.Concat(tensors, axis);

        foreach (var X in tensors)
            if (X.shape.rank != 4)
                return base.Concat(tensors, axis);

        var O = TensorExtensions.Concat(tensors.Select(t => t.shape).ToArray(), axis);
        var offsets = new int[] { 0,0,0,0 };
        axis = O.Axis(axis);

        var fn = new ComputeFunc(m_Kernels, "Copy");
        var result = NewTensor(fn, "O", O);
        foreach (var X in tensors)
        {
            SetTensor(fn, "X", X);
            fn.shader.SetInts("_Pad", offsets);
            fn.Dispatch(X.channels, X.width, X.height);

            offsets[axis] += X.shape[axis];
        }
        return result;
    }

    public override Tensor Gather(Tensor[] tensors, int axis)
    {
        Tensor X = tensors[0];
        Tensor indices = tensors[1];

        int[] shape = X.shape.ToArray();
        shape[axis] = indices.flatWidth;

        var O = new TensorShape(shape);

        var fn = new ComputeFunc(m_Kernels, "Gather");
        SetTensor(fn, "X", X);
        SetTensor(fn, "K", indices);
        fn.shader.SetInts("_Axis", axis);

        return Dispatch(fn, O, O.channels, O.width, O.height);
    }

    public virtual Tensor ElementwiseWithBroadcast(string kernelName, Tensor[] tensors)
    {
        var O = TensorExtensions.MaxShape(tensors);

        Assert.IsTrue(tensors.Length > 0);
        var X = tensors[0];

        var fn = new ComputeFunc(m_Kernels, kernelName);
        for (int t = 1; t < tensors.Length; ++t)
        {
            var B = tensors[t];

            SetTensor(fn, "X", X);
            SetTensor(fn, "B", B);
            X = Dispatch(fn, O, O.channels, O.width, O.height);
        }

        return X;
    }

    public override Tensor Add(Tensor[] tensors)
    {
        return ElementwiseWithBroadcast("BroadcastAdd", tensors);
    }

    public override Tensor Sub(Tensor[] tensors)
    {
        return ElementwiseWithBroadcast("BroadcastSub", tensors);
    }

    public override Tensor Mul(Tensor[] tensors)
    {
        return ElementwiseWithBroadcast("BroadcastMul", tensors);
    }

    public override Tensor Div(Tensor[] tensors)
    {
        return ElementwiseWithBroadcast("BroadcastDiv", tensors);
    }

    public override Tensor Pow(Tensor[] tensors)
    {
        return ElementwiseWithBroadcast("BroadcastPow", tensors);
    }

    public override Tensor Min(Tensor[] tensors)
    {
        return ElementwiseWithBroadcast("BroadcastMin", tensors);
    }

    public override Tensor Max(Tensor[] tensors)
    {
        return ElementwiseWithBroadcast("BroadcastMax", tensors);
    }

    public override Tensor Greater(Tensor A, Tensor B)
    {
        return ElementwiseWithBroadcast("BroadcastGreater", new Tensor[] { A, B });
    }

    public override Tensor GreaterEqual(Tensor A, Tensor B)
    {
        return ElementwiseWithBroadcast("BroadcastGreaterEqual", new Tensor[] { A, B });
    }

    public override Tensor Less(Tensor A, Tensor B)
    {
        return ElementwiseWithBroadcast("BroadcastLess", new Tensor[] { A, B });
    }

    public override Tensor LessEqual(Tensor A, Tensor B)
    {
        return ElementwiseWithBroadcast("BroadcastLessEqual", new Tensor[] { A, B });
    }

    public override Tensor Equal(Tensor A, Tensor B)
    {
        return ElementwiseWithBroadcast("BroadcastEqual", new Tensor[] { A, B });
    }

    public override Tensor LogicalOr(Tensor A, Tensor B)
    {
        return ElementwiseWithBroadcast("BroadcastLogicalOr", new Tensor[] { A, B });
    }

    public override Tensor LogicalAnd(Tensor A, Tensor B)
    {
        return ElementwiseWithBroadcast("BroadcastLogicalAnd", new Tensor[] { A, B });
    }

    public override Tensor LogicalXor(Tensor A, Tensor B)
    {
        return ElementwiseWithBroadcast("BroadcastLogicalXor", new Tensor[] { A, B });
    }

    public override Tensor LogicalNot(Tensor X)
    {
        return Activation("LogicalNot", X);
    }

    public virtual Tensor Reduce(string kernelName, Tensor X, int axis)
    {
        if (axis != 3 && axis != -1)
            throw new NotImplementedException();

        var O = X.shape.Reduce(axis);
		Assert.AreEqual(O.channels, 1);

        var fn = new ComputeFunc(m_Kernels, kernelName);
        SetTensor(fn, "X", X);

        return Dispatch(fn, O, O.width, O.height, 1);
    }

    public override Tensor ReduceMin(Tensor X, int axis)
    {
    	return Reduce("ReduceMin", X, axis);
    }

	public override Tensor ReduceMax(Tensor X, int axis)
    {
    	return Reduce("ReduceMax", X, axis);
    }

    public override Tensor ReduceSum(Tensor X, int axis)
    {
    	return Reduce("ReduceSum", X, axis);
    }

    public override Tensor ReduceMean(Tensor X, int axis)
    {
    	return Reduce("ReduceMean", X, axis);
    }

    public override Tensor ReduceProd(Tensor X, int axis)
    {
    	return Reduce("ReduceProd", X, axis);
    }

    public override Tensor Prepare(Tensor X)
    {
        Pin(X);
        return X;
    }
}

public struct ComputeFunc
{
    // dispatch dimension limitation coming from D3D11
    const uint SafeDispatchLimit = 65535;

    public struct TensorDecl
    {
        public int ShapeId { get; }
        public int InfoId { get; }

        public TensorDecl(int shapeId, int infoId)
        {
            ShapeId = shapeId;
            InfoId = infoId;
        }
    }

    readonly public ComputeShader shader;
    readonly public string kernelName;
    readonly public int kernelIndex;
    readonly public uint threadGroupSizeX;
    readonly public uint threadGroupSizeY;
    readonly public uint threadGroupSizeZ;
    public uint threadGroupSize { get { return threadGroupSizeX * threadGroupSizeY * threadGroupSizeZ; } }

    public int width { get { return (int)threadGroupSizeX; } }
    public int height { get { return (int)threadGroupSizeY; } }
    public int depth { get { return (int)threadGroupSizeZ; } }

    static public TensorDecl GetTensorDecl(string name)
    {
        var shapeId = Shader.PropertyToID(s_StringCache.Lookup(name, "declShape"));
        var infoId = Shader.PropertyToID(s_StringCache.Lookup(name, "declInfo"));
        return new TensorDecl(shapeId, infoId);
    }
    static public int GetTensorData(string name ) { return Shader.PropertyToID(s_StringCache.Lookup(name, "data")); }

    static private StringCache s_StringCache = new StringCache();

    static private Texture2D s_DummyTexture2D;
    static private Texture3D s_DummyTexture3D;
    static private Texture2DArray s_DummyTexture2DArray;

    static private Texture2D dummyTexture2D {
        get
        {
            if (s_DummyTexture2D == null)
                s_DummyTexture2D = new Texture2D(8, 8);
            return s_DummyTexture2D;
        }
    }

    static private Texture3D dummyTexture3D
    {
        get
        {
            if (s_DummyTexture3D == null)
                s_DummyTexture3D = new Texture3D(8, 8, 1, TextureFormat.ARGB32, false);
            return s_DummyTexture3D;
        }
    }

    static private Texture2DArray dummyTexture2DArray
    {
        get
        {
            if (s_DummyTexture2DArray == null)
                s_DummyTexture2DArray = new Texture2DArray(8, 8, 1, TextureFormat.ARGB32, false);
            return s_DummyTexture2DArray;
        }
    }

    // ---------------------------------------------------------------------------------

    public ComputeFunc(ComputeShader cs, string[] kns, int x, int y = 1, int z = 1)
        : this(cs, FindBestKernelMatchingDimensions(new [] { cs }, kns, x, y, z))
    {
    }

    public ComputeFunc(ComputeShader cs, string kn)
        : this(new [] { cs }, kn)
    {
    }

    public ComputeFunc(ComputeShader[] cs, string[] kns, int x, int y = 1, int z = 1)
        : this(cs, FindBestKernelMatchingDimensions(cs, kns, x, y, z))
    {
    }

    public ComputeFunc(ComputeShader[] cs, string kn)
    {
        foreach (ComputeShader s in cs)
            if (s != null && s.HasKernel(kn))
            {
                shader = s;
                kernelName = kn;
                kernelIndex = shader.FindKernel(kernelName);
                shader.GetKernelThreadGroupSizes(kernelIndex, out threadGroupSizeX, out threadGroupSizeY, out threadGroupSizeZ);
                return;
            }
        throw new ArgumentException("Kernel " + kn + " is missing");
    }

    // ---------------------------------------------------------------------------------

    public void SetTensor(string name, TensorShape shape, ComputeBuffer buffer, Int64 dataOffset = 0)
    {
        SetTensorDecl(name, shape, dataOffset);
        SetTensorBuffer(name, buffer);
    }
    public void SetTensor(ComputeFunc.TensorDecl tensorDecl, int dataPropId, TensorShape shape, ComputeBuffer buffer, Int64 dataOffset = 0)
    {
        SetTensorDecl(tensorDecl, shape, dataOffset);
        SetTensorBuffer(dataPropId, buffer);
    }

    public void SetTensor(string name, TensorShape shape, Texture texture, Int64 dataOffset = 0)
    {
        SetTensorDecl(name, shape, dataOffset);
        SetTexture(name, texture);
    }

    public void SetTensorDecl(string name, TensorShape shape, Int64 dataOffset)
    {
        ComputeFunc.TensorDecl tensorDecl = GetTensorDecl(name);
        shader.SetInts(tensorDecl.ShapeId, shape.batch, shape.height, shape.width, shape.channels );
        shader.SetInts(tensorDecl.InfoId, (int)dataOffset, shape.length);
    }

    // WARN: SetTensorDecl() is not multi-thread safe due to s_TensorDeclScratchpad usage
    // However there is no plan to call SetTensorDecl() from multiple threads
    // NOTE: s_TensorDeclScratchpad is used to avoid memory allocation
	static private int[] s_tTensorDeclScratchpadShape = new int[4];
    static private int[] s_tTensorDeclScratchpadInfo = new int[2];
    public void SetTensorDecl(ComputeFunc.TensorDecl tensorDecl, TensorShape shape, Int64 dataOffset)
    {
        s_tTensorDeclScratchpadShape[0] = shape.batch;
        s_tTensorDeclScratchpadShape[1] = shape.height;
        s_tTensorDeclScratchpadShape[2] = shape.width;
        s_tTensorDeclScratchpadShape[3] = shape.channels;
        s_tTensorDeclScratchpadInfo[0] = (int)dataOffset;
        s_tTensorDeclScratchpadInfo[1] = shape.length;
        shader.SetInts(tensorDecl.ShapeId, s_tTensorDeclScratchpadShape);
        shader.SetInts(tensorDecl.InfoId, s_tTensorDeclScratchpadInfo);
    }

    public void SetTensorBuffer(string name, ComputeBuffer buffer)
    {
        shader.SetBuffer(kernelIndex, GetTensorData(name), buffer);
    }
    public void SetTensorBuffer(int propId, ComputeBuffer buffer)
    {
        shader.SetBuffer(kernelIndex, propId, buffer);
    }

    public void SetTexture(string name, Texture tex)
    {
        // set dummy textures for slots that are not used - to make API validation layers happy
        Texture tex2D = dummyTexture2D;
        Texture tex2Darray = dummyTexture2DArray;
        Texture tex3D = dummyTexture3D;

        if (tex.dimension == TextureDimension.Tex2D)
            tex2D = tex;
        else if (tex.dimension == TextureDimension.Tex2DArray)
            tex2Darray = tex;
        else if (tex.dimension == TextureDimension.Tex3D)
            tex3D = tex;
        else
            throw new InvalidOperationException("Unsupported texture type");

        shader.SetTexture(kernelIndex, name + "tex2D", tex2D);
        shader.SetTexture(kernelIndex, name + "tex3D", tex3D);
        shader.SetTexture(kernelIndex, name + "tex2DArray", tex2Darray);
    }

    public void Dispatch(int[] workItems)
    {
        Assert.IsTrue(workItems.Length >= 3);
        Dispatch(workItems[0], workItems[1], workItems[2]);
    }

    public void Dispatch(int workItemsX, int workItemsY, int workItemsZ)
    {
        Profiler.BeginSample(kernelName);
        var x = IntDivCeil(workItemsX, (int) threadGroupSizeX);
        var y = IntDivCeil(workItemsY, (int) threadGroupSizeY);
        var z = IntDivCeil(workItemsZ, (int) threadGroupSizeZ);

        // some GFX APIs / GPU hw/drivers have limitation of 65535 per dimension
        if (x > SafeDispatchLimit || y > SafeDispatchLimit || z > SafeDispatchLimit)
            D.LogWarning($"Exceeded safe compute dispatch group count limit per dimension [{x}, {y}, {z}] for {kernelName}");

        shader.Dispatch(kernelIndex, x, y, z);
        Profiler.EndSample();
    }

    // ---------------------------------------------------------------------------------

    static public int IntDivCeil(int v, int div)
    {
        return (v + div - 1) / div;
    }

    static public string FindBestKernelMatchingDimensions(ComputeShader[] cs, string[] kns, int x, int y = 1, int z = 1)
    {
        Assert.IsTrue(kns.Length > 0);

        foreach (var kernelName in kns)
        {
            foreach (var shader in cs)
            {
                int kernelIndex = shader.FindKernel(kernelName);
                uint threadGroupSizeX, threadGroupSizeY, threadGroupSizeZ;
                shader.GetKernelThreadGroupSizes(kernelIndex, out threadGroupSizeX, out threadGroupSizeY, out threadGroupSizeZ);

                if (x % threadGroupSizeX == 0 &&
                    y % threadGroupSizeY == 0 &&
                    z % threadGroupSizeZ == 0)
                    return kernelName;
            }
        }
        // pick the last one
        return kns[kns.Length - 1];
    }
}

} // namespace Barracuda
