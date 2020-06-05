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
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

[assembly: InternalsVisibleTo("Barracuda.EditorTests")]

namespace Unity.Barracuda {

public static class ComputeHelper
{
    public static int IDivC(int v, int div)
    {
        return (v + div - 1) / div;
    }
}

public class ComputeTensorData : ITensorData
{
    private bool m_DisposeBufferAfterUse;
    private ComputeBuffer m_Buffer;
    private TensorShape m_FullBufferShape;
    private int m_Offset;
    private ComputeInfo.ChannelsOrder m_OnDeviceChannelsOrder;

    public ComputeBuffer buffer { get { return m_Buffer; } }
    public int offset { get { return m_Offset; } }
    public string name;

#if DEBUG_TRACK_ALLOCATIONS
    protected StackTrace m_AllocationTrace;
#endif

    public ComputeTensorData(TensorShape shape, string buffername, ComputeInfo.ChannelsOrder onDeviceChannelsOrder, bool clearOnInit = true)
    {
        m_OnDeviceChannelsOrder = onDeviceChannelsOrder;
        name = buffername;
        m_Buffer = new ComputeBuffer(shape.length, sizeof(float));

        // @TODO: consider zero initialization only for "debug" mode
        if (clearOnInit)
        {
            float[] zeros = new float[shape.length];
            m_Buffer.SetData(zeros);
        }

        m_FullBufferShape = shape;
        m_Offset = 0;

        m_DisposeBufferAfterUse = true;

#if DEBUG_TRACK_ALLOCATIONS
        m_AllocationTrace = new System.Diagnostics.StackTrace();
#endif
    }

    public ComputeTensorData(ComputeBuffer buffer, TensorShape shape, int offset, string buffername, ComputeInfo.ChannelsOrder onDeviceChannelsOrder)
    {
        m_OnDeviceChannelsOrder = onDeviceChannelsOrder;
        name = buffername;
        m_Buffer = buffer;
        m_FullBufferShape = shape;
        m_Offset = offset;

        m_DisposeBufferAfterUse = false;
    }

    ~ComputeTensorData()
    {
        if (m_Buffer == null)
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
            m_Buffer.Dispose();
            m_Buffer = null;
        }
        m_DisposeBufferAfterUse = false;
    }

    public virtual void Reserve(int count)
    {
        if (m_Offset + count > maxCapacity)
            throw new ArgumentException("ComputeTensorData buffer is too small to reserve " + count + " elements.");
    }

    public virtual void Upload(float[] data, TensorShape shape, int managedBufferStartIndex = 0)
    {
        var count = shape.length;
        Assert.IsTrue(managedBufferStartIndex >= 0);
        Assert.IsTrue(managedBufferStartIndex + count <= data.Length);

        if (m_OnDeviceChannelsOrder == ComputeInfo.ChannelsOrder.NCHW)
        {
            //Transpose from HWC to CHW, TODO use a compute shader or threaded code.
            float[] chwData = new float[count];
            for (int readIndex=0; readIndex < count; ++readIndex)
            {
                int b = 0, h = 0, w = 0, ch = 0;
                shape.GetPositionsFromIndex(readIndex, ref b, ref h, ref w, ref ch);
                int writeIndex = shape.IndexNCHW(b, h, w, ch);
                chwData[writeIndex] = data[managedBufferStartIndex+readIndex];
            }
            m_Buffer.SetData(chwData, 0, m_Offset, count);
        }
        else
        {
            m_Buffer.SetData(data, managedBufferStartIndex, m_Offset, count);
        }

        m_AsyncDownloadSchedulingFrame = -1;
        #if UNITY_2018_2_OR_NEWER
        m_AsyncDownloadRequested = false;
        #endif
    }

    public virtual bool ScheduleAsyncDownload(int count)
    {
        #if UNITY_2018_2_OR_NEWER
        if (SystemInfo.supportsAsyncGPUReadback)
            return WaitForAsyncReadback(count);
        #endif

        return WaitFor3Frames(count);
    }

    private int m_AsyncDownloadSchedulingFrame = -1;
    private bool WaitFor3Frames(int count)
    {
        if (m_AsyncDownloadSchedulingFrame < 0)
            m_AsyncDownloadSchedulingFrame = Time.frameCount;
        var framesPassed = Time.frameCount - m_AsyncDownloadSchedulingFrame;
        return framesPassed > 3;
    }

    #if UNITY_2018_2_OR_NEWER
    private bool m_AsyncDownloadRequested = false;
    private AsyncGPUReadbackRequest m_AsyncDownloadRequest;
    private bool WaitForAsyncReadback(int count)
    {
        if (m_AsyncDownloadRequested)
        {
            if (m_AsyncDownloadRequest.hasError)
                m_AsyncDownloadRequested = false;
            else
                m_AsyncDownloadRequest.Update();
        }

        if (!m_AsyncDownloadRequested)
        {
            m_AsyncDownloadRequest = AsyncGPUReadback.Request(m_Buffer, count * sizeof(float), m_Offset * sizeof(float));
            m_AsyncDownloadRequested = true;
        }

        return m_AsyncDownloadRequest.done;
    }
    #endif

    private ConvertFromOnDeviceFormatHelper m_ConvertFromOnDeviceFormatHelper = new ConvertFromOnDeviceFormatHelper();
    private float[] ConvertFromOnDeviceFormat(TensorShape shape, float[] data)
    {
        return m_ConvertFromOnDeviceFormatHelper.GetNHWCData(shape, data, m_OnDeviceChannelsOrder);
    }

    private unsafe class ConvertFromOnDeviceFormatHelper
    {
        private float* oPtr;
        private float* xPtr;
        private TensorShape shape;
        private int unrollSize = 4;
        public Action<long> unrolledInnerLoopDelegate;

        internal ConvertFromOnDeviceFormatHelper()
        {
            unrolledInnerLoopDelegate = UnrolledInnerLoop;
        }

        internal float[] GetNHWCData(TensorShape shape, float[] data, ComputeInfo.ChannelsOrder onDeviceFormat, bool useRefImplementation = false)
        {
            //tensor is HWC on device, no need to concert.
            if (onDeviceFormat == ComputeInfo.ChannelsOrder.NHWC)
                return data;

            //tensor is flat in regard to CHW, no need to convert.
            var channelOrderRelatedDimensions = (shape.height > 1 ? 1 : 0) + (shape.width > 1 ? 1 : 0) + (shape.channels > 1 ? 1 : 0);
            if (channelOrderRelatedDimensions < 2)
                return data;

            //else allocate new buffer, apply conversion and return it.
            float[] hwcData = new float[shape.length];
            if (!useRefImplementation)
            {
                unsafe
                {
                    fixed (float* xPtr = &data[0], oPtr = &hwcData[0])
                    {
                        this.oPtr = oPtr;
                        this.xPtr = xPtr;
                        this.shape = shape;
                        ApplyConversion();
                    }
                }
            }
            else
            {
                for (int readIndex=0; readIndex < data.Length; ++readIndex)
                {
                    int b = 0, h = 0, w = 0, ch = 0;
                    shape.GetPositionsFromIndexNCHW(readIndex, ref b, ref h, ref w, ref ch);
                    int writeIndex = shape.Index(b, h, w, ch);
                    hwcData[writeIndex] = data[readIndex];
                }
            }

            return hwcData;
        }

        private void ApplyConversion()
        {
            UnsafeArrayCPUOps.Parallel_For(0L, shape.length / unrollSize, unrolledInnerLoopDelegate);

            // Remainder
            for (int i = (shape.length / unrollSize) * unrollSize; i < shape.length; ++i)
            {
                int b = 0, h = 0, w = 0, ch = 0;
                shape.GetPositionsFromIndexNCHW(i, ref b, ref h, ref w, ref ch);
                int writeIndex = shape.Index(b, h, w, ch);
                oPtr[writeIndex] = xPtr[i];
            }
        }

        private void UnrolledInnerLoop(long n)
        {
            int baseIndex = (int)n * 4;
            int b0 = 0, h0 = 0, w0 = 0, ch0 = 0;
            int b1 = 0, h1 = 0, w1 = 0, ch1 = 0;
            int b2 = 0, h2 = 0, w2 = 0, ch2 = 0;
            int b3 = 0, h3 = 0, w3 = 0, ch3 = 0;
            shape.GetPositionsFromIndexNCHW(baseIndex+0, ref b0, ref h0, ref w0, ref ch0);
            shape.GetPositionsFromIndexNCHW(baseIndex+1, ref b1, ref h1, ref w1, ref ch1);
            shape.GetPositionsFromIndexNCHW(baseIndex+2, ref b2, ref h2, ref w2, ref ch2);
            shape.GetPositionsFromIndexNCHW(baseIndex+3, ref b3, ref h3, ref w3, ref ch3);
            int writeIndex0 = shape.Index(b0, h0, w0, ch0);
            int writeIndex1 = shape.Index(b1, h1, w1, ch1);
            int writeIndex2 = shape.Index(b2, h2, w2, ch2);
            int writeIndex3 = shape.Index(b3, h3, w3, ch3);
            oPtr[writeIndex0] = xPtr[baseIndex+0];
            oPtr[writeIndex1] = xPtr[baseIndex+1];
            oPtr[writeIndex2] = xPtr[baseIndex+2];
            oPtr[writeIndex3] = xPtr[baseIndex+3];
        }
    }

    public virtual float[] Download(TensorShape shape)
    {
        //;;D.logStackTraceEnabled = true;
        //;;Debug.Log("Download ComputeTensorData " + name + " " + maxCapacity + " " + count);
        //;;D.logStackTraceEnabled = false;

        var count = shape.length;

        Profiler.BeginSample("Barracuda.DownloadDataFromGPU");
        Assert.IsTrue(maxCapacity >= count);
        count = Math.Min(maxCapacity, count);

        m_AsyncDownloadSchedulingFrame = -1;
        #if UNITY_2018_2_OR_NEWER
        if (m_AsyncDownloadRequested)
        {
            m_AsyncDownloadRequested = false;
            if (!m_AsyncDownloadRequest.done)
                m_AsyncDownloadRequest.WaitForCompletion();

            if (!m_AsyncDownloadRequest.hasError)
            {
                var reqData = m_AsyncDownloadRequest.GetData<float>().ToArray();
                if (reqData.Length >= count)
                { // if we have retrieved enough data
                    reqData = ConvertFromOnDeviceFormat(shape, reqData);
                    Profiler.EndSample();
                    return reqData;
                }
            }
        }
        #endif

        var data = new float[count];
        m_Buffer.GetData(data, 0, m_Offset, count);
        data = ConvertFromOnDeviceFormat(shape, data);
        Profiler.EndSample();

        return data;
    }

    public virtual float[] SharedAccess(out int offset)
    {
        offset = m_Offset;
        return Download(new TensorShape(0,0,0,maxCapacity));
    }

    public virtual int maxCapacity { get
    {
        return m_Buffer.count;
    } }

    public override string ToString()
    {
        string allocationSource = "";

#if DEBUG_TRACK_ALLOCATIONS
        allocationSource += "\nSource:\n" + m_AllocationTrace;
#endif

        return string.Format("(GPU:{0}#{1} {2} buffer: {3} created at: {4})",
            name, GetHashCode(), m_FullBufferShape, m_Buffer, allocationSource);
    }
}

internal class SharedComputeTensorData : ComputeTensorData
{
    public SharedComputeTensorData(ComputeBuffer buffer, TensorShape shape, int offset = 0, string buffername = "", ComputeInfo.ChannelsOrder channelsOrder = ComputeInfo.ChannelsOrder.NHWC) : base(buffer, shape, offset, buffername, channelsOrder) {}
}

internal class TextureFormatUtils
{
    public static bool IsRedOnly(TextureFormat format)
    {
        return  format == TextureFormat.R8 ||
                format == TextureFormat.R16 ||
                format == TextureFormat.RHalf ||
                format == TextureFormat.RFloat ||
                format == TextureFormat.BC4 ||
                format == TextureFormat.EAC_R ||
                format == TextureFormat.EAC_R_SIGNED;
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
                format == TextureFormat.RGFloat ||
                format == TextureFormat.BC5 ||
                format == TextureFormat.EAC_RG ||
                format == TextureFormat.EAC_RG_SIGNED;
    }

    public static bool IsRedGreen(RenderTextureFormat format)
    {
        return  format == RenderTextureFormat.RG16 ||
                format == RenderTextureFormat.RGHalf ||
                format == RenderTextureFormat.RGFloat;
    }

    public static bool IsRedGreenBlue(TextureFormat format)
    {
        return  format == TextureFormat.RGB565 ||
                format == TextureFormat.RGB24 ||
                format == TextureFormat.DXT1 ||
                #if !UNITY_IOS
                format == TextureFormat.DXT1Crunched ||
                #endif
                format == TextureFormat.PVRTC_RGB2 ||
                format == TextureFormat.PVRTC_RGB4 ||
                format == TextureFormat.ETC_RGB4 ||
                #if !UNITY_IOS
                format == TextureFormat.ETC_RGB4Crunched ||
                #endif
                format == TextureFormat.ETC2_RGB ||
                format == TextureFormat.ASTC_RGB_4x4 ||
                format == TextureFormat.ASTC_RGB_5x5 ||
                format == TextureFormat.ASTC_RGB_6x6 ||
                format == TextureFormat.ASTC_RGB_8x8 ||
                format == TextureFormat.ASTC_RGB_10x10 ||
                format == TextureFormat.ASTC_RGB_12x12 ||
                format == TextureFormat.BC6H;
    }

    public static bool IsRedGreenBlue(RenderTextureFormat format)
    {
        return  format == RenderTextureFormat.RGB565 ||
                format == RenderTextureFormat.BGR101010_XR;
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

    public static bool IsRedGreenBlue(Texture tex)
    {
        var tex2D = tex as Texture2D;
        var texArr = tex as Texture2DArray;
        var tex3D = tex as Texture3D;
        var rt = tex as RenderTexture;

        if (tex2D != null)
            return IsRedGreenBlue(tex2D.format);
        else if (texArr != null)
            return IsRedGreenBlue(texArr.format);
        else if (tex3D != null)
            return IsRedGreenBlue(tex3D.format);
        else if (rt != null)
            return IsRedGreenBlue(rt.format);
        else
            return false;
    }

    public static int FormatToChannelCount(Texture tex)
    {
        if (IsRedOnly(tex))
            return 1;
        if (IsAlphaOnly(tex))
            return 1;
        if (IsRedGreen(tex))
            return 2;
        if (IsRedGreenBlue(tex))
            return 3;
        return 4;
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


    public TextureAsTensorData(Texture[] textures, int interpretPixelAsChannels = -1,
        Flip flip = Flip.Y, InterpretDepthAs depthAs = InterpretDepthAs.Batch, InterpretColorAs colorAs = InterpretColorAs.AverageMultipleChannels)
    {
        if (textures.Length < 1)
            throw new ArgumentException("Textures array must be non empty");

        if (interpretPixelAsChannels < 0)
        {
            interpretPixelAsChannels = TextureFormatUtils.FormatToChannelCount(textures[0]);

            // check that all textures have the same number of channels
            foreach (var tex in textures)
                if (interpretPixelAsChannels != TextureFormatUtils.FormatToChannelCount(tex))
                    throw new ArgumentException("All textures must have the same number of channels");
        }

        m_InterpretPixelAsChannels = interpretPixelAsChannels;
        m_InterpretDepthAs = depthAs;
        m_InterpretColorAs = colorAs;
        m_Flip = flip;

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

    public TextureAsTensorData(Texture texture, int interpretPixelAsChannels = -1,
        Flip flip = Flip.Y, InterpretDepthAs depthAs = InterpretDepthAs.Batch, InterpretColorAs colorAs = InterpretColorAs.AverageMultipleChannels)
    : this(new [] { texture }, interpretPixelAsChannels, flip, depthAs, colorAs) {}

    public virtual void Reserve(int count)
    {
        // currently always readonly
        throw new InvalidOperationException("TextureAsTensorData is readonly");
    }

    public virtual void Upload(float[] data, TensorShape shape, int managedBufferStartIndex = 0)
    {
        // currently always readonly
        throw new InvalidOperationException("TextureAsTensorData is readonly");
    }

    public virtual bool ScheduleAsyncDownload(int count)
    {
        // @TODO: cache compute tensor data and request async
        return true;
    }

    public virtual float[] Download(TensorShape shape)
    {
        var gpuBackend = new ReferenceComputeOps(ComputeShaderSingleton.Instance.referenceKernels);
        // @TODO: cache compute buffer
        using(var computeTensorData = gpuBackend.TextureToTensorData(this, "__internalDownloadTextureToTensorData"))
        {
            return computeTensorData.Download(shape);
        }
    }

    public virtual float[] SharedAccess(out int offset)
    {
        offset = 0;
        return Download(shape);
    }

    public virtual int maxCapacity { get
    {
        return m_Shape.length;
    } }

    public virtual void Dispose()
    {
    }
}

public class ReferenceComputeOps : ReferenceCPUOps
{
    private ComputeShader[] m_Kernels;

    public ReferenceComputeOps(ComputeShader kernels, ITensorAllocator allocator = null)
    : base(allocator)
    {
        m_Kernels = new [] {kernels};
    }

    public ComputeTensorData Pin(Tensor X)
    {
        X.FlushCache();

        var onDevice = X.tensorOnDevice as ComputeTensorData;
        if (onDevice == null)
        {
            var asTexture = X.tensorOnDevice as TextureAsTensorData;
            if (asTexture != null)
                X.AttachToDevice(TextureToTensorData(asTexture, X.name));
            else
                X.UploadToDevice(new ComputeTensorData(X.shape, X.name, ComputeInfo.channelsOrder));
        }

        Assert.IsNotNull(X.tensorOnDevice as ComputeTensorData);
        Assert.IsNotNull((X.tensorOnDevice as ComputeTensorData).buffer);

        return X.tensorOnDevice as ComputeTensorData;
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

    internal ITensorData TextureToTensorData(TextureAsTensorData texData, string name)
    {
        var fn = new ComputeFunc(m_Kernels, "TextureToTensor");
        var tensorData = new ComputeTensorData(texData.shape, name, ComputeInfo.channelsOrder, false);

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

    public void TensorToRenderTexture(Tensor X, RenderTexture target, int batch, int fromChannel, Vector4 scale, Vector4 bias, Texture3D lut)
    {
        if (!target.enableRandomWrite || !target.IsCreated())
        {
            target.Release();
            target.enableRandomWrite = true;
            target.Create();
        }

        var fn = new ComputeFunc(m_Kernels, "TensorToTexture"+ (lut == null?"NoLUT":"3DLUT"));
        SetTensor(fn, "X", X);
        fn.SetTexture("O", target);
        fn.shader.SetVector("_Scale", scale);
        fn.shader.SetVector("_Bias", bias);
        fn.shader.SetInts("_Pad", new int[] { batch, 0, 0, fromChannel });
        fn.shader.SetBool("_FlipY", true);
        if (lut != null)
        {
            fn.SetTexture("X", lut);
            fn.shader.SetVector("_LutParams", new Vector2(1f / lut.width, lut.width - 1f));
        }

        fn.Dispatch(target.width, target.height, 1);
    }

    protected bool ShouldFlattenInputForDenseLayer(TensorShape X)
    {
        //In HWC flatten is a no-op memory wise.
        if (ComputeInfo.channelsOrder == ComputeInfo.ChannelsOrder.NHWC)
            return false;

        //In CHW flatten is return a tensor with items linearized in memory in regards to HWC layout.
        int flattenDimensions = (X.height > 1 ? 1 : 0) +
                                (X.width > 1 ? 1 : 0) +
                                (X.channels > 1 ? 1 : 0);
        return flattenDimensions > 1;
    }

    protected bool IsFusedActivationSupported(Layer.FusedActivation fusedActivation)
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

    // ---------------------------------------------------------------------------------

    public override Tensor MatMul(Tensor X, bool xTranspose, Tensor Y, bool yTranspose)
    {
        // MatMul implementation in terms of Dense
        var A = (xTranspose) ? Transpose(X): X;
        var B = (yTranspose) ? Transpose(Y): Y;
        var C = NewTensor(1, B.flatWidth);
        var Z = Sub(new[] { C, C }); // intialize bias with zeros

        var O = Dense(A, B, Z, Layer.FusedActivation.None);
        if (A != X) A.Dispose();
        if (B != Y) B.Dispose();
        C.Dispose();
        Z.Dispose();

        return O;
    }

    public override Tensor Dense(Tensor X, Tensor W, Tensor B, Layer.FusedActivation fusedActivation)
    {
        Assert.IsTrue(W.dimensions <= 2);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(X.flatWidth, W.flatHeight);

        if (ShouldFlattenInputForDenseLayer(X.shape))
            X = Flatten(X);

        var Oshape = new TensorShape(X.flatHeight, W.flatWidth);

        var fn = new ComputeFunc(m_Kernels, "Dense");

        SetTensor(fn, "X", X);
        SetTensor(fn, "W", W);
        SetTensor(fn, "B", B);
        fn.shader.SetInt("_ActivationMode", (int)fusedActivation);

        var O = Dispatch(fn, Oshape, Oshape.flatWidth, Oshape.flatHeight, 1);

        if (!IsFusedActivationSupported(fusedActivation))
            O = Activation(fusedActivation.ToString(), O);

        return O;
    }

    private Tensor Conv2DWinograd(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, Layer.FusedActivation fusedActivation)
    {
        Assert.AreEqual(X.channels, K.kernelDepth);
        Assert.AreEqual(K.kernelCount, B.flatWidth);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);

        var Oshape = X.shape.ApplyKernel(K.shape, stride, pad);

        var fn = new ComputeFunc(m_Kernels, "Conv2DWinograd_2x2_3x3");

        SetTensor(fn, "X", X);
        SetTensor(fn, "K", K);
        SetTensor(fn, "B", B);

        fn.shader.SetInts("_Pad", pad);
        fn.shader.SetInt("_ActivationMode", (int)fusedActivation);

        var O = Dispatch(fn, Oshape, K.kernelCount, ComputeHelper.IDivC(Oshape.width, 2), ComputeHelper.IDivC(Oshape.height, 2));

        if (!IsFusedActivationSupported(fusedActivation))
            O = Activation(fusedActivation.ToString(), O);

        return O;
    }

    public override Tensor Conv2D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, Layer.FusedActivation fusedActivation)
    {
        Assert.AreEqual(X.channels, K.kernelDepth);
        Assert.AreEqual(K.kernelCount, B.flatWidth);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);

        var Oshape = X.shape.ApplyKernel(K.shape, stride, pad);

        bool useWinograd = (K.kernelWidth == 3) && (K.kernelHeight == 3) && (stride[0] == 1) && (stride[1] == 1) && ((Oshape.height % 2) == 0) && ((Oshape.width % 2) == 0);
        if( useWinograd )
        {
            return Conv2DWinograd(X, K, B, stride, pad, fusedActivation);
        }

        var fn = new ComputeFunc(m_Kernels, "Conv2D");

        SetTensor(fn, "X", X);
        SetTensor(fn, "K", K);
        SetTensor(fn, "B", B);
        fn.shader.SetInts("_Stride", stride);
        fn.shader.SetInts("_Pad", pad);
        fn.shader.SetInt("_ActivationMode", (int)fusedActivation);

        var O = Dispatch(fn, Oshape, K.kernelCount, Oshape.width, Oshape.height);

        if (!IsFusedActivationSupported(fusedActivation))
            O = Activation(fusedActivation.ToString(), O);

        return O;
    }

    public override Tensor DepthwiseConv2D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, Layer.FusedActivation fusedActivation)
    {
        if (K.kernelDepth != 1)
            return base.DepthwiseConv2D(X, K, B, stride, pad, fusedActivation);

        Assert.AreEqual(K.kernelDepth, 1);
        Assert.AreEqual(K.kernelCount, X.channels);
        Assert.AreEqual(K.kernelCount, B.flatWidth);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);

        var Oshape = X.shape.ApplyKernel(K.shape, stride, pad);

        var fn = new ComputeFunc(m_Kernels, "DepthwiseConv2D");

        SetTensor(fn, "X", X);
        SetTensor(fn, "K", K);
        SetTensor(fn, "B", B);
        fn.shader.SetInts("_Stride", stride);
        fn.shader.SetInts("_Pad", pad);
        fn.shader.SetInt("_ActivationMode", (int)fusedActivation);

        var O = Dispatch(fn, Oshape, K.kernelCount, Oshape.width, Oshape.height);

        if (!IsFusedActivationSupported(fusedActivation))
            O = Activation(fusedActivation.ToString(), O);

        return O;
    }

    public override Tensor Conv2DTrans(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, int[] outputAdjustment, Layer.FusedActivation fusedActivation)
    {
        Assert.AreEqual(X.channels, K.kernelDepth);
        Assert.AreEqual(K.kernelCount, B.flatWidth);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);

        var Oshape = X.shape.ApplyKernelInverse(K.shape, stride, pad, outputAdjustment);

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
        fn.shader.SetInt("_ActivationMode", (int)fusedActivation);

        var O = Dispatch(fn, Oshape, K.kernelCount, Oshape.width, Oshape.height);

        if (!IsFusedActivationSupported(fusedActivation))
            O = Activation(fusedActivation.ToString(), O);

        return O;
    }

    public override Tensor Upsample2D(Tensor X, int[] scale, bool bilinear)
    {
        Assert.AreEqual(scale.Length, 2);

        var O = new TensorShape(X.batch, X.height*scale[1], X.width*scale[0], X.channels);

        var fn = new ComputeFunc(m_Kernels, bilinear ? "UpsampleBilinear2D": "Upsample2D");

        SetTensor(fn, "X", X);

        fn.shader.SetInts("_Pool", scale);

        if (bilinear) // dispatches over output dimensions (O)
            return Dispatch(fn, O, O.channels, O.width, O.height);
        else // dispatches over input dimensions (X)
            return Dispatch(fn, O, X.channels, X.width, X.height);
    }

    public override Tensor Resample2D(Tensor X, int[] size, bool bilinear)
    {
        Assert.AreEqual(size.Length, 2);

        var O = new TensorShape(X.batch, size[1], size[0], X.channels);

        var fn = new ComputeFunc(m_Kernels, bilinear ? "ResampleBilinear2D" : "Resample2D");

        SetTensor(fn, "X", X);

        return Dispatch(fn, O, O.channels, O.width, O.height);
    }

    public override Tensor DepthToSpace(Tensor X, int[] blocksize, Layer.DepthToSpaceMode mode)
    {
        Assert.AreEqual(blocksize.Length, 2);
        Assert.AreEqual(X.channels % (blocksize[0] * blocksize[1]), 0);

        var O = new TensorShape(X.batch, X.height * blocksize[1], X.width * blocksize[0], X.channels / (blocksize[0] * blocksize[1]));

        var fn = new ComputeFunc(m_Kernels, "DepthToSpace_" + mode);

        SetTensor(fn, "X", X);

        fn.shader.SetInts("_Pool", blocksize);

        return Dispatch(fn, O, O.channels, O.width, O.height);
    }

    public override Tensor SpaceToDepth(Tensor X, int[] blocksize)
    {
        Assert.AreEqual(blocksize.Length, 2);
        Assert.AreEqual(X.channels % (blocksize[0] * blocksize[1]), 0);

        var O = new TensorShape(X.batch, X.height / blocksize[1], X.width / blocksize[0], X.channels * (blocksize[0] * blocksize[1]));

        var fn = new ComputeFunc(m_Kernels, "SpaceToDepth");

        SetTensor(fn, "X", X);

        fn.shader.SetInts("_Pool", blocksize);

        return Dispatch(fn, O, O.channels, O.width, O.height);
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

    public override Tensor Pad2DReflect(Tensor X, int[] pad)
    {
        return ApplyPadding(X, pad, "Pad2DReflect");
    }

    public override Tensor Pad2DSymmetric(Tensor X, int[] pad)
    {
        return ApplyPadding(X, pad, "Pad2DSymmetric");
    }

    public override Tensor Pad2DEdge(Tensor X, int[] pad)
    {
        return ApplyPadding(X, pad, "Pad2DEdge");
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

    public override Tensor Normalization(Tensor X, Tensor S, Tensor B, int pool, int axis, float epsilon, Layer.FusedActivation fusedActivation)
    {
        if (axis != 3 && axis != -1)
            return base.Normalization(X, S, B, pool, axis, epsilon, fusedActivation);

        if (pool == 1 && X.batch != 1)
            return base.Normalization(X, S, B, pool, axis, epsilon, fusedActivation); // @TODO: Instance Normalization with batch > 1

        if (pool <= 0)
            pool = X.batch;

        var Oshape = X.shape;
        var fn = new ComputeFunc(m_Kernels, "InstanceNorm");
        fn.shader.SetFloat("_Epsilon", epsilon);
        fn.shader.SetInt("_ActivationMode", (int)fusedActivation);

        SetTensor(fn, "X", X);
        SetTensor(fn, "W", S);
        SetTensor(fn, "B", B);

        var O = Dispatch(fn, Oshape, Oshape.channels, 1, 1);

        if (!IsFusedActivationSupported(fusedActivation))
            O = Activation(fusedActivation.ToString(), O);

        return O;
    }

    public override Tensor LRN(Tensor X, float alpha, float beta, float bias, int size)
    {
        var O = X.shape;
        var fn = new ComputeFunc(m_Kernels, "LRN");

        SetTensor(fn, "X", X);
        fn.shader.SetFloat("_Alpha", alpha);
        fn.shader.SetFloat("_Beta",  beta);
        fn.shader.SetFloat("_Epsilon",  bias);
        fn.shader.SetInt("_Axis", size);

        return Dispatch(fn, O, O.channels, O.width, O.height);
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

    public override Tensor PRelu(Tensor X, Tensor S)
    {
        Assert.IsTrue((X.flatWidth == S.flatWidth) || (S.flatWidth == 1));

        var O = X.shape;
        var fn = new ComputeFunc(m_Kernels, "PRelu");

        SetTensor(fn, "X", X);
        SetTensor(fn, "W", S);

        return Dispatch(fn, O, O.channels, O.width, O.height);
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

    public override Tensor Tanh(Tensor X)
    {
        return Activation("Tanh", X);
    }

    public override Tensor Sigmoid(Tensor X)
    {
        return Activation("Sigmoid", X);
    }

    public override Tensor Relu6(Tensor X)
    {
        return Activation("Relu6", X);
    }

    public override Tensor Elu(Tensor X, float alpha)
    {
        return Activation("Elu", X, alpha);
    }

    public override Tensor LeakyRelu(Tensor X, float alpha)
    {
        return Activation("LeakyRelu", X, alpha);
    }

    public override Tensor Selu(Tensor X, float alpha, float gamma)
    {
        return Activation("Selu", X, alpha, gamma);
    }

    public override Tensor Swish(Tensor X)
    {
        return Activation("Swish", X);
    }

    public override Tensor Abs(Tensor X)
    {
        return Activation("Abs", X);
    }

    public override Tensor Neg(Tensor X)
    {
        return Activation("Neg", X);
    }

    public override Tensor Ceil(Tensor X)
    {
        return Activation("Ceil", X);
    }

    public override Tensor Clip(Tensor X, float min, float max)
    {
        return Activation("Clip", X, min, max);
    }

    public override Tensor Floor(Tensor X)
    {
        return Activation("Floor", X);
    }

    public override Tensor Reciprocal(Tensor X)
    {
        return Activation("Reciprocal", X);
    }

    public override Tensor Pow(Tensor X, float alpha)
    {
        return Activation("Pow", X, alpha);
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

    public override Tensor Expand(Tensor X, TensorShape newShape)
    {
        Assert.IsTrue(newShape.batch == X.batch || X.batch == 1);
        Assert.IsTrue(newShape.height == X.height || X.height == 1);
        Assert.IsTrue(newShape.width == X.width || X.width == 1);
        Assert.IsTrue(newShape.channels == X.channels || X.channels == 1);

        var fn = new ComputeFunc(m_Kernels, "Expand");
        SetTensor(fn, "X", X);

        return Dispatch(fn, newShape, newShape.channels, newShape.width, newShape.height);
    }

    protected virtual Tensor ElementwiseWithBroadcast(string kernelName, Tensor[] tensors)
    {
        var O = TensorExtensions.MaxShape(tensors);

        Assert.IsTrue(tensors.Length > 0);
        var X = tensors[0];

        var fn = new ComputeFunc(m_Kernels, kernelName);
        bool isFirstDispatch = true;
        for (int t = 1; t < tensors.Length; ++t)
        {
            var B = tensors[t];

            SetTensor(fn, "X", X);
            SetTensor(fn, "B", B);
            fn.shader.SetFloat("_Alpha", 1.0f/(float)tensors.Length);
            fn.shader.SetInt("_IsFirstDispatch", isFirstDispatch ? 1 : 0);

            X = Dispatch(fn, O, O.channels, O.width, O.height);
            isFirstDispatch = false;
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

    public override Tensor Mean(Tensor[] tensors)
    {
        return ElementwiseWithBroadcast("BroadcastMean", tensors);
    }

    protected virtual Tensor Reduce(string kernelName, Tensor X, int axis)
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

    protected virtual Tensor CopyAndReshape_NCHW(Tensor X, TensorShape newShape)
    {
        Assert.AreEqual(X.length, newShape.length);
        Assert.AreEqual(ComputeInfo.ChannelsOrder.NCHW, ComputeInfo.channelsOrder);

        var O = NewTensor(newShape, "O");

        var fn = new ComputeFunc(m_Kernels, "ReshapeFromNHWCModel_NCHW");
        SetTensor(fn, "X", X);
        SetTensor(fn, "O", O);
        fn.Dispatch(O.channels, O.width, O.height);

        return O;
    }

    protected override Tensor CopyAndReshape(Tensor X, TensorShape newShape)
    {
        Assert.AreEqual(X.length, newShape.length);
        if (X.shape != newShape)
        {
            //In CHW mode on should call CopyAndReshape_NCHW if shape is modified
            Assert.AreEqual(ComputeInfo.ChannelsOrder.NHWC, ComputeInfo.channelsOrder);
        }

        var copyShape = X.shape;
        var fn = new ComputeFunc(m_Kernels, "Copy");
        SetTensor(fn, "X", X);

        // NOTE: "Copy" kernel copies tensor data while preserving the shape
        // However here in CopyAndReshape we want to both copy and change the shape,
        // To be able to piggyback "Copy" kernel we specify new shape when allocating destination tensor,
        // but use shape identical to source when copying.

        var O = NewTensor(newShape, "O");
        fn.SetTensor("O", copyShape, Pin(O).buffer);

        var offsets = new int[] { 0,0,0,0 };
        fn.shader.SetInts("_Pad", offsets);
        fn.Dispatch(X.channels, X.width, X.height);

        return O;
    }

    public override Tensor Flatten(Tensor X)
    {
        if (ComputeInfo.channelsOrder == ComputeInfo.ChannelsOrder.NHWC)
            return base.Flatten(X);

        var newShape = X.shape.Flatten();
        return CopyAndReshape_NCHW(X, newShape);
    }

    public override Tensor Reshape(Tensor X, TensorShape newShape)
    {
        if (ComputeInfo.channelsOrder == ComputeInfo.ChannelsOrder.NHWC)
            return base.Reshape(X, newShape);

        return CopyAndReshape_NCHW(X, newShape);
    }

    public override Tensor Transpose(Tensor X)
    {
        Assert.IsTrue(X.dimensions <= 2);
        var O = new TensorShape(X.flatWidth, X.flatHeight);

        var fn = new ComputeFunc(m_Kernels, "Transpose");
        SetTensor(fn, "X", X);
        return Dispatch(fn, O, O.flatWidth, O.flatHeight, 1);
    }

    protected static int[] s_ConcatOffsets = new int[] {0, 0, 0, 0};
    public override Tensor Concat(Tensor[] tensors, int axis)
    {
        if (axis != 3 && axis != -1)
            return base.Concat(tensors, axis);

        foreach (var X in tensors)
            if (X.shape.rank != 4)
                return base.Concat(tensors, axis);

        var O = TensorExtensions.Concat(tensors, axis);
        var offsets = s_ConcatOffsets;
        Array.Clear(offsets, 0, offsets.Length);
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

    static private int[] s_StridedSliceStart = new int[4];
    public override Tensor StridedSlice(Tensor X, int[] starts, int[] ends, int[] stride)
    {
        Assert.AreEqual(starts.Length, 4);
        Assert.AreEqual(ends.Length, 4);
        Assert.AreEqual(stride.Length, 4);

        var O = X.shape.ApplyStridedSlice(starts, ends, stride);

        s_StridedSliceStart[0] = TensorExtensions.WrapIndex(starts[0], X.batch);
        s_StridedSliceStart[1] = TensorExtensions.WrapIndex(starts[1], X.height);
        s_StridedSliceStart[2] = TensorExtensions.WrapIndex(starts[2], X.width);
        s_StridedSliceStart[3] = TensorExtensions.WrapIndex(starts[3], X.channels);


        var fn = new ComputeFunc(m_Kernels, "StridedSlice");
        SetTensor(fn, "X", X);
        fn.shader.SetInts("_Stride", stride);
        fn.shader.SetInts("_Pad", s_StridedSliceStart);

        return Dispatch(fn, O, O.channels, O.width, O.height);
    }

    public override Tensor Tile(Tensor X, int[] repeats)
    {
        // @TODO: GPU implementation
        return base.Tile(X, repeats);
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
        fn.shader.SetInt("_Axis", axis);

        return Dispatch(fn, O, O.channels, O.width, O.height);
    }

    public override Tensor Copy(Tensor X)
    {
        return base.Copy(X);
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
        string kernelNameWithChannelsOrder = s_StringCache.Lookup(kn,
                            (ComputeInfo.channelsOrder == ComputeInfo.ChannelsOrder.NHWC) ? "_NHWC" : "_NCHW");
        foreach (ComputeShader s in cs)
            if (s != null && (s.HasKernel(kernelNameWithChannelsOrder) || s.HasKernel(kn)))
            {
                shader = s;
                kernelName = s.HasKernel(kernelNameWithChannelsOrder)?kernelNameWithChannelsOrder:kn;
                kernelIndex = shader.FindKernel(kernelName);
                shader.GetKernelThreadGroupSizes(kernelIndex, out threadGroupSizeX, out threadGroupSizeY, out threadGroupSizeZ);
                return;
            }
        throw new ArgumentException($"Kernel {kn} and {kernelNameWithChannelsOrder} are both missing");
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
        SetTensorDecl(tensorDecl, shape, dataOffset);
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

    public void Dispatch(ValueTuple<int,int,int> workItems)
    {
        Dispatch(workItems.Item1, workItems.Item2, workItems.Item3);
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


        ComputeDebugUtils.PrepareDispatch();

        shader.Dispatch(kernelIndex, x, y, z);

        ComputeDebugUtils.VerifyDispatch(kernelName);

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

} // namespace Unity.Barracuda
