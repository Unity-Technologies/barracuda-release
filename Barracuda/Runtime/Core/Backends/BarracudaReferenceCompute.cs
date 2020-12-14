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

internal static class ComputeHelper
{
    public static int IDivC(int v, int div)
    {
        return (v + div - 1) / div;
    }
}

/// <summary>
/// `Tensor` data storage for GPU backends
/// </summary>
public class ComputeTensorData : ITensorData
{
    private bool m_DisposeBufferAfterUse;
    private ComputeBuffer m_Buffer;
    private TensorShape m_FullBufferShape;
    private int m_Offset;
    private ComputeInfo.ChannelsOrder m_OnDeviceChannelsOrder;

    /// <summary>
    /// Data storage as `ComputeBuffer`
    /// </summary>
    public ComputeBuffer buffer { get { return m_Buffer; } }

    /// <summary>
    /// Offset in the data storage buffer
    /// </summary>
    public int offset { get { return m_Offset; } }

    /// <summary>
    /// Parent `Tensor` name
    /// </summary>
    public string name;

    /// <summary>
    /// Channel order channels-first vs channels-last
    /// </summary>
    public ComputeInfo.ChannelsOrder channelsOrder { get { return m_OnDeviceChannelsOrder; } }

#if DEBUG_TRACK_ALLOCATIONS
    protected StackTrace m_AllocationTrace;
#endif

    /// <summary>
    /// Create `ComputeTensorData`
    /// </summary>
    /// <param name="shape">shape</param>
    /// <param name="buffername">buffer name</param>
    /// <param name="onDeviceChannelsOrder">channel order</param>
    /// <param name="clearOnInit">clear on init</param>
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

    /// <summary>
    /// Create `ComputeTensorData` with specified `buffer`
    /// </summary>
    /// <param name="buffer">buffer</param>
    /// <param name="shape">shape</param>
    /// <param name="offset">offset</param>
    /// <param name="buffername">buffer name</param>
    /// <param name="onDeviceChannelsOrder">channels order</param>
    public ComputeTensorData(ComputeBuffer buffer, TensorShape shape, int offset, string buffername, ComputeInfo.ChannelsOrder onDeviceChannelsOrder)
    {
        m_OnDeviceChannelsOrder = onDeviceChannelsOrder;
        name = buffername;
        m_Buffer = buffer;
        m_FullBufferShape = shape;
        m_Offset = offset;

        m_DisposeBufferAfterUse = false;
    }

    /// <summary>
    /// Finalizer
    /// </summary>
    ~ComputeTensorData()
    {
        if (m_Buffer == null)
            return;
        if (!m_DisposeBufferAfterUse)
            return;

        D.LogWarning($"Found unreferenced, but undisposed Tensor data which might lead to GPU resource leak: {ToString()}");

        Dispose();
    }

    /// <summary>
    /// Dispose internal storage
    /// </summary>
    public virtual void Dispose()
    {
        if (m_DisposeBufferAfterUse)
        {
            m_Buffer.Dispose();
            m_Buffer = null;
        }
        m_DisposeBufferAfterUse = false;
    }

    /// <inheritdoc/>
    public virtual void Reserve(int count)
    {
        if (m_Offset + count > maxCapacity)
            throw new ArgumentException("ComputeTensorData buffer is too small to reserve " + count + " elements.");
    }

    /// <inheritdoc/>
    public virtual void Upload(float[] data, TensorShape shape, int managedBufferStartIndex = 0)
    {
        var count = shape.length;
        Assert.IsTrue(managedBufferStartIndex >= 0);
        Assert.IsTrue(managedBufferStartIndex + count <= data.Length);

        if (m_OnDeviceChannelsOrder == ComputeInfo.ChannelsOrder.NCHW)
        {
            //Transpose from HWC to CHW, TODO use a compute shader or threaded code.
            Profiler.BeginSample("Tensor.Upload_ChannelFirstTranpose");
            float[] chwData = new float[count];
            if (shape.IsNHWC())
            {
                for (int readIndex=0; readIndex < count; ++readIndex)
                {
                    int b = 0, h = 0, w = 0, ch = 0;
                    shape.GetPositionsFromIndex(readIndex, ref b, ref h, ref w, ref ch);
                    int writeIndex = shape.IndexChannelFirst(b, h, w, ch);
                    chwData[writeIndex] = data[managedBufferStartIndex+readIndex];
                }
            }
            else
            {
                for (int readIndex=0; readIndex < count; ++readIndex)
                {
                    int s = 0, r = 0, n = 0, t = 0, d = 0, h = 0, w = 0, ch = 0;
                    shape.GetPositionsFromIndex(readIndex, ref s, ref r, ref n, ref t, ref d, ref h, ref w, ref ch);
                    int writeIndex = shape.IndexChannelFirst(s, r, n, t, d, h, w, ch);
                    chwData[writeIndex] = data[managedBufferStartIndex+readIndex];
                }
            }
            Profiler.EndSample();
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

    /// <inheritdoc/>
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
            var channelOrderRelatedDimensions = 0;
            for (int i = TensorShape.DataBatch + 1; i < TensorShape.MaxRank; ++i)
            {
                if (shape[i] > 1)
                    ++channelOrderRelatedDimensions;
            }
            if (channelOrderRelatedDimensions == 1)
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
                    int s = 0, r = 0, n = 0, t = 0, d = 0, h = 0, w = 0, c = 0;
                    shape.GetPositionsFromIndexChannelFirst(readIndex, ref s, ref r, ref n, ref t, ref d, ref h, ref w, ref c);
                    int writeIndex = shape.Index(s,r,n,t,d,h,w,c);
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
                int s = 0, r = 0, n = 0, t = 0, d = 0, h = 0, w = 0, c = 0;
                shape.GetPositionsFromIndexChannelFirst(i, ref s, ref r, ref n, ref t, ref d, ref h, ref w, ref c);
                int writeIndex = shape.Index(s,r,n,t,d,h,w,c);
                oPtr[writeIndex] = xPtr[i];
            }
        }

        private void UnrolledInnerLoop(long n)
        {
            int baseIndex = (int)n * 4;
            int s0 = 0, r0 = 0, n0 = 0, t0 = 0, d0 = 0, h0 = 0, w0 = 0, c0 = 0;
            int s1 = 0, r1 = 0, n1 = 0, t1 = 0, d1 = 0, h1 = 0, w1 = 0, c1 = 0;
            int s2 = 0, r2 = 0, n2 = 0, t2 = 0, d2 = 0, h2 = 0, w2 = 0, c2 = 0;
            int s3 = 0, r3 = 0, n3 = 0, t3 = 0, d3 = 0, h3 = 0, w3 = 0, c3 = 0;
            shape.GetPositionsFromIndexChannelFirst(baseIndex+0, ref s0, ref r0, ref n0, ref t0, ref d0, ref h0, ref w0, ref c0);
            shape.GetPositionsFromIndexChannelFirst(baseIndex+1, ref s1, ref r1, ref n1, ref t1, ref d1, ref h1, ref w1, ref c1);
            shape.GetPositionsFromIndexChannelFirst(baseIndex+2, ref s2, ref r2, ref n2, ref t2, ref d2, ref h2, ref w2, ref c2);
            shape.GetPositionsFromIndexChannelFirst(baseIndex+3, ref s3, ref r3, ref n3, ref t3, ref d3, ref h3, ref w3, ref c3);
            int writeIndex0 = shape.Index(s0, r0, n0, t0, d0, h0, w0, c0);
            int writeIndex1 = shape.Index(s1, r1, n1, t1, d1, h1, w1, c1);
            int writeIndex2 = shape.Index(s2, r2, n2, t2, d2, h2, w2, c2);
            int writeIndex3 = shape.Index(s3, r3, n3, t3, d3, h3, w3, c3);
            oPtr[writeIndex0] = xPtr[baseIndex+0];
            oPtr[writeIndex1] = xPtr[baseIndex+1];
            oPtr[writeIndex2] = xPtr[baseIndex+2];
            oPtr[writeIndex3] = xPtr[baseIndex+3];
        }
    }

    /// <inheritdoc/>
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

    /// <inheritdoc/>
    public virtual float[] SharedAccess(out int offset)
    {
        offset = m_Offset;
        return Download(new TensorShape(0,0,0,maxCapacity));
    }

    /// <inheritdoc/>
    public virtual int maxCapacity { get
    {
        return m_Buffer.count;
    } }

    /// <summary>
    /// Summary
    /// </summary>
    /// <returns>summary</returns>
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
                #if UNITY_2019_1_OR_NEWER
                format == TextureFormat.ASTC_4x4 ||
                format == TextureFormat.ASTC_5x5 ||
                format == TextureFormat.ASTC_6x6 ||
                format == TextureFormat.ASTC_8x8 ||
                format == TextureFormat.ASTC_10x10 ||
                format == TextureFormat.ASTC_12x12 ||
                #else
                format == TextureFormat.ASTC_RGB_4x4 ||
                format == TextureFormat.ASTC_RGB_5x5 ||
                format == TextureFormat.ASTC_RGB_6x6 ||
                format == TextureFormat.ASTC_RGB_8x8 ||
                format == TextureFormat.ASTC_RGB_10x10 ||
                format == TextureFormat.ASTC_RGB_12x12 ||
                #endif
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

/// <summary>
/// Texture based `Tensor` storage
/// </summary>
public class TextureAsTensorData : ITensorData
{
    /// <summary>
    /// Flip flag enum
    /// </summary>
    public enum Flip
    {
        /// <summary>
        /// None
        /// </summary>
        None,
        /// <summary>
        /// Flip Y
        /// </summary>
        Y,
    }

    /// <summary>
    /// Interpret depth as enum
    /// </summary>
    public enum InterpretDepthAs
    {
        /// <summary>
        /// Batch
        /// </summary>
        Batch,

        /// <summary>
        /// Channels
        /// </summary>
        Channels,
    }

    /// <summary>
    /// Interpret color enum
    /// </summary>
    public enum InterpretColorAs
    {
        /// <summary>
        /// Average multiple channels
        /// </summary>
        AverageMultipleChannels,
        // TODO: PickFirstChannel,
    }

    private TensorShape m_Shape;
    private Texture[] m_Textures;
    private int m_InterpretPixelAsChannels;
    private InterpretDepthAs m_InterpretDepthAs;
    private InterpretColorAs m_InterpretColorAs;
    private Flip m_Flip;

    /// <summary>
    /// Shape
    /// </summary>
    public TensorShape shape { get { return m_Shape; } }

    /// <summary>
    /// Backing textures
    /// </summary>
    public Texture[] textures { get { return m_Textures; } }

    /// <summary>
    /// Interpret pixel as channels
    /// </summary>
    public int interpretPixelAsChannels { get { return m_InterpretPixelAsChannels; } }

    /// <summary>
    /// Interpret depth as
    /// </summary>
    public InterpretDepthAs interpretDepthAs { get { return m_InterpretDepthAs; } }

    /// <summary>
    /// Interpret color as
    /// </summary>
    public InterpretColorAs interpretColorAs { get { return m_InterpretColorAs; } }

    /// <summary>
    /// Flip flag
    /// </summary>
    public Flip flip { get { return m_Flip; } }

    /// <summary>
    /// Create `TextureAsTensorData` from supplied `textures`
    /// </summary>
    /// <param name="textures">backing textures</param>
    /// <param name="interpretPixelAsChannels">interpret pixel as channels</param>
    /// <param name="flip">flip</param>
    /// <param name="depthAs">depth as</param>
    /// <param name="colorAs">color as</param>
    /// <exception cref="ArgumentException">thrown if textures array is empty or texture types are different</exception>
    /// <exception cref="InvalidOperationException">thrown if unsupported texture type is supplied</exception>
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

    /// <summary>
    /// Create `TextureAsTensorData` from supplied `texture`
    /// </summary>
    /// <param name="texture">texture</param>
    /// <param name="interpretPixelAsChannels">interpret pixel as channels</param>
    /// <param name="flip">flip</param>
    /// <param name="depthAs">depth as</param>
    /// <param name="colorAs">color as</param>
    public TextureAsTensorData(Texture texture, int interpretPixelAsChannels = -1,
        Flip flip = Flip.Y, InterpretDepthAs depthAs = InterpretDepthAs.Batch, InterpretColorAs colorAs = InterpretColorAs.AverageMultipleChannels)
    : this(new [] { texture }, interpretPixelAsChannels, flip, depthAs, colorAs) {}

    /// <inheritdoc/>
    public virtual void Reserve(int count)
    {
        // currently always readonly
        throw new InvalidOperationException("TextureAsTensorData is readonly");
    }

    /// <inheritdoc/>
    public virtual void Upload(float[] data, TensorShape shape, int managedBufferStartIndex = 0)
    {
        // currently always readonly
        throw new InvalidOperationException("TextureAsTensorData is readonly");
    }

    /// <inheritdoc/>
    public virtual bool ScheduleAsyncDownload(int count)
    {
        // @TODO: cache compute tensor data and request async
        return true;
    }

    /// <inheritdoc/>
    public virtual float[] Download(TensorShape shape)
    {
        var gpuBackend = new ReferenceComputeOps(null);
        // @TODO: cache compute buffer
        using(var computeTensorData = gpuBackend.TextureToTensorData(this, "__internalDownloadTextureToTensorData"))
        {
            return computeTensorData.Download(shape);
        }
    }

    /// <inheritdoc/>
    public virtual float[] SharedAccess(out int offset)
    {
        offset = 0;
        return Download(shape);
    }

    /// <inheritdoc/>
    public virtual int maxCapacity { get
    {
        return m_Shape.length;
    } }

    /// <summary>
    /// Dispose
    /// </summary>
    public virtual void Dispose()
    {
    }
}

/// <summary>
/// Reference GPU compute `IOps` implementation
/// </summary>
public class ReferenceComputeOps : ReferenceCPUOps
{
    private ComputeShader[] m_Kernels;

    /// <summary>
    /// Create `ReferenceComputeOps`
    /// </summary>
    /// <param name="kernels">compute kernels</param>
    /// <param name="allocator">allocator</param>
    public ReferenceComputeOps(ComputeShader kernels, ITensorAllocator allocator = null)
    : base(allocator)
    {
        m_Kernels = new [] {kernels};
    }

    /// <summary>
    /// Pin `Tensor` to GPU compute device
    /// </summary>
    /// <param name="X">`Tensor`</param>
    /// <returns>`ComputeTensorData`</returns>
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

    internal void SetTensor(ComputeFunc fn, string name, Tensor X)
    {
        var XonDevice = Pin(X);
        fn.SetTensor(name, X.shape, XonDevice.buffer, XonDevice.offset);
    }

    internal Tensor NewTensor(ComputeFunc fn, string name, TensorShape shape)
    {
        var o = NewTensor(shape, name);
        fn.SetTensor(name, shape, Pin(o).buffer);
        return o;
    }

    internal Tensor Dispatch(ComputeFunc fn, TensorShape outputShape, int workItemsX, int workItemsY, int workItemsZ, string outputName = "O")
    {
        var o = NewTensor(fn, outputName, outputShape);
        fn.Dispatch(workItemsX, workItemsY, workItemsZ);
        return o;
    }

    // ---------------------------------------------------------------------------------

    internal ITensorData TextureToTensorData(TextureAsTensorData texData, string name)
    {
        var fn = new ComputeFunc(ComputeShaderSingleton.Instance.texureKernels, "TextureToTensor");
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
        if (!target.enableRandomWrite || !target.IsCreated())
        {
            target.Release();
            target.enableRandomWrite = true;
            target.Create();
        }

        var fn = new ComputeFunc(ComputeShaderSingleton.Instance.texureKernels, "TensorToTexture"+ (lut == null?"NoLUT":"3DLUT"));
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

    /// <summary>
    /// Check if `Flatten` is needed for `Dense` layer input
    /// </summary>
    /// <param name="X">input shape</param>
    /// <returns>`true` if `Flatten` is needed</returns>
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

    /// <summary>
    /// Check if `fusedActivation` type is supported
    /// </summary>
    /// <param name="fusedActivation">fused activation type</param>
    /// <returns>`true` if supported</returns>
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
    /// <inheritdoc/>
    public override Tensor MatMul(Tensor X, bool xTranspose, Tensor Y, bool yTranspose)
    {
        // N.B: Current implementation is inefficient as it introduces Transposes/Slice and Concat.
        // => consider refactoring dense to support batch
        // X and Y can be constants, in that cases the internal layout does not match ComputeInfo.channelsOrder and will allways be NHWC
        // => permute them if there is a layout mismatch
        if (Pin(X).channelsOrder == ComputeInfo.ChannelsOrder.NHWC && ComputeInfo.channelsOrder == ComputeInfo.ChannelsOrder.NCHW)
            X = TransposeToNCHW(X);
        if (Pin(Y).channelsOrder == ComputeInfo.ChannelsOrder.NHWC && ComputeInfo.channelsOrder == ComputeInfo.ChannelsOrder.NCHW)
            Y = TransposeToNCHW(Y);

        // V-Table magic, ReferenceCPU.MaMul is calls MatMul2D, Concat & Slice all which are overloaded by all respective IOps, so will call the correct backend
        return base.MatMul(X, xTranspose, Y, yTranspose);
    }

    /// <inheritdoc/>
    protected override Tensor MatMul2D(Tensor X, bool xTranspose, Tensor Y, bool yTranspose)
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

    /// <inheritdoc/>
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

    /// <summary>
    /// Convolution implementation via Winograd transform
    /// </summary>
    /// <param name="X">input</param>
    /// <param name="K">convolution kernel</param>
    /// <param name="B">bias</param>
    /// <param name="stride">stride</param>
    /// <param name="pad">padding</param>
    /// <param name="fusedActivation">fused activation type</param>
    /// <returns>output `Tensor`</returns>
    private Tensor Conv2DWinograd(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, Layer.FusedActivation fusedActivation)
    {
        Assert.IsTrue(X.shape.IsNHWC());
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

    /// <inheritdoc/>
    public override Tensor Conv2D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, Layer.FusedActivation fusedActivation)
    {
        Assert.IsTrue(X.shape.IsNHWC());
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

    /// <inheritdoc/>
    public override Tensor DepthwiseConv2D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, Layer.FusedActivation fusedActivation)
    {
        if (K.kernelDepth != 1)
            return base.DepthwiseConv2D(X, K, B, stride, pad, fusedActivation);

        Assert.IsTrue(X.shape.IsNHWC());
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

    /// <inheritdoc/>
    public override Tensor Conv2DTrans(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, int[] outputAdjustment, Layer.FusedActivation fusedActivation)
    {
        Assert.IsTrue(X.shape.IsNHWC());
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

    /// <inheritdoc/>
    public override Tensor Upsample2D(Tensor X, int[] scale, bool bilinear)
    {
        Assert.IsTrue(X.shape.IsNHWC());
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

    /// <inheritdoc/>
    public override Tensor Resample2D(Tensor X, int[] size, bool bilinear)
    {
        Assert.IsTrue(X.shape.IsNHWC());
        Assert.AreEqual(size.Length, 2);

        var O = new TensorShape(X.batch, size[1], size[0], X.channels);

        var fn = new ComputeFunc(m_Kernels, bilinear ? "ResampleBilinear2D" : "Resample2D");

        SetTensor(fn, "X", X);

        return Dispatch(fn, O, O.channels, O.width, O.height);
    }

    /// <inheritdoc/>
    public override Tensor DepthToSpace(Tensor X, int[] blocksize, Layer.DepthToSpaceMode mode)
    {
        Assert.IsTrue(X.shape.IsNHWC());
        Assert.AreEqual(blocksize.Length, 2);
        Assert.AreEqual(X.channels % (blocksize[0] * blocksize[1]), 0);

        var O = new TensorShape(X.batch, X.height * blocksize[1], X.width * blocksize[0], X.channels / (blocksize[0] * blocksize[1]));

        var fn = new ComputeFunc(m_Kernels, "DepthToSpace_" + mode);

        SetTensor(fn, "X", X);

        fn.shader.SetInts("_Pool", blocksize);

        return Dispatch(fn, O, O.channels, O.width, O.height);
    }

    /// <inheritdoc/>
    public override Tensor SpaceToDepth(Tensor X, int[] blocksize)
    {
        Assert.IsTrue(X.shape.IsNHWC());
        Assert.AreEqual(blocksize.Length, 2);
        Assert.AreEqual(X.channels % (blocksize[0] * blocksize[1]), 0);

        var O = new TensorShape(X.batch, X.height / blocksize[1], X.width / blocksize[0], X.channels * (blocksize[0] * blocksize[1]));

        var fn = new ComputeFunc(m_Kernels, "SpaceToDepth");

        SetTensor(fn, "X", X);

        fn.shader.SetInts("_Pool", blocksize);

        return Dispatch(fn, O, O.channels, O.width, O.height);
    }

    /// <inheritdoc/>
    protected virtual Tensor Pool2D(string kernelName, Tensor X, int[] pool, int[] stride, int[] pad)
    {
        Assert.IsTrue(X.shape.IsNHWC());
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

    /// <inheritdoc/>
    public override Tensor MaxPool2D(Tensor X, int[] pool, int[] stride, int[] pad)
    {
        return Pool2D("MaxPool2D", X, pool, stride, pad);
    }

    /// <inheritdoc/>
    public override Tensor AvgPool2D(Tensor X, int[] pool, int[] stride, int[] pad)
    {
        return Pool2D("AvgPool2D", X, pool, stride, pad);
    }

    /// <summary>
    /// Generic pooling 2D
    /// </summary>
    /// <param name="kernelName">kernel name</param>
    /// <param name="X">input</param>
    /// <returns>output `Tensor`</returns>
    protected virtual Tensor GlobalPool2D(string kernelName, Tensor X)
    {
        var O = new TensorShape(X.batch, 1, 1, X.channels);

        var fn = new ComputeFunc(m_Kernels, kernelName);

        SetTensor(fn, "X", X);

        return Dispatch(fn, O, O.channels, 1, 1);
    }

    /// <inheritdoc/>
    public override Tensor GlobalMaxPool2D(Tensor X)
    {
        return GlobalPool2D("GlobalMaxPool2D", X);
    }

    /// <inheritdoc/>
    public override Tensor GlobalAvgPool2D(Tensor X)
    {
        return GlobalPool2D("GlobalAvgPool2D", X);
    }

    /// <inheritdoc/>
    public override Tensor GlobalAvgVariancePool2D(Tensor X)
    {
        Assert.IsTrue(X.shape.IsNHWC());
        var O = new TensorShape(X.batch, 2, 1, X.channels);

        var fn = new ComputeFunc(m_Kernels, "GlobalAvgVariancePool2D");

        SetTensor(fn, "X", X);

        return Dispatch(fn, O, O.channels, 1, 1);
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
        Assert.IsTrue(X.shape.IsNHWC());
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

    /// <inheritdoc/>
    public override Tensor Border2D(Tensor X, int[] pad, float constant)
    {
        return ApplyPadding(X, pad, "Border2D", constant);
    }

    /// <inheritdoc/>
    public override Tensor Pad2DReflect(Tensor X, int[] pad)
    {
        return ApplyPadding(X, pad, "Pad2DReflect");
    }

    /// <inheritdoc/>
    public override Tensor Pad2DSymmetric(Tensor X, int[] pad)
    {
        return ApplyPadding(X, pad, "Pad2DSymmetric");
    }

    /// <inheritdoc/>
    public override Tensor Pad2DEdge(Tensor X, int[] pad)
    {
        return ApplyPadding(X, pad, "Pad2DEdge");
    }

    /// <inheritdoc/>
    public override Tensor ScaleBias(Tensor X, Tensor S, Tensor B)
    {
        if (!X.shape.IsNHWC())
            throw new NotImplementedException();

        Assert.AreEqual(X.channels, B.channels); Assert.AreEqual(X.channels, S.channels);
        Assert.AreEqual(B.length, B.channels); Assert.AreEqual(S.length, S.channels);

        var O = X.shape;
        var fn = new ComputeFunc(m_Kernels, "ScaleBias");

        SetTensor(fn, "X", X);
        SetTensor(fn, "W", S);
        SetTensor(fn, "B", B);

        return Dispatch(fn, O, O.channels, O.width, O.height);
    }

    /// <inheritdoc/>
    public override Tensor Normalization(Tensor X, Tensor S, Tensor B, int pool, int axis, float epsilon, Layer.FusedActivation fusedActivation)
    {
        if (!X.shape.IsNHWC())
            throw new NotImplementedException();

        if (axis != TensorShape.C && axis != -1)
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

    /// <inheritdoc/>
    public override Tensor LRN(Tensor X, float alpha, float beta, float bias, int size)
    {
        if (!X.shape.IsNHWC())
            throw new NotImplementedException();

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
    /// <inheritdoc/>
    public override Tensor Dropout(Tensor X, float alpha)
    {
        if (!X.shape.IsNHWC())
            throw new NotImplementedException();

        Assert.IsTrue(alpha >= 0f && alpha <= 1f);

        var O = X.shape;
        var fn = new ComputeFunc(m_Kernels, "Dropout");

        SetTensor(fn, "X", X);
        fn.shader.SetFloat("_Alpha", alpha);
        fn.shader.SetFloat("_Seed", UnityEngine.Random.value);

        return Dispatch(fn, O, O.channels, O.width, O.height);
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
        var O = X.shape;
        var fn = new ComputeFunc(m_Kernels, kernelName);

        SetTensor(fn, "X", X);
        fn.shader.SetFloat("_Alpha", alpha);
        fn.shader.SetFloat("_Beta",  beta);

        return Dispatch(fn, O, O.channels, O.width, O.height);
    }

    /// <inheritdoc/>
    public override Tensor Relu(Tensor X)
    {
        if (!X.shape.IsNHWC())
            return base.Relu(X);

        return Activation("Relu", X);
    }

    /// <inheritdoc/>
    public override Tensor PRelu(Tensor X, Tensor S)
    {
        if (!X.shape.IsNHWC())
            return base.PRelu(X, S);

        Assert.IsTrue((X.flatWidth == S.flatWidth) || (S.flatWidth == 1));

        var O = X.shape;
        var fn = new ComputeFunc(m_Kernels, "PRelu");

        SetTensor(fn, "X", X);
        SetTensor(fn, "W", S);

        return Dispatch(fn, O, O.channels, O.width, O.height);
    }

    /// <inheritdoc/>
    public override Tensor Softmax(Tensor X, int axis)
    {
        if (!X.shape.IsNHWC() || axis != TensorExtensions.NHWCTo8DAxis(1))
            return base.Softmax(X, axis);

        var O = X.shape.Flatten();

        var fn = new ComputeFunc(m_Kernels, "Softmax");

        SetTensor(fn, "X", X);

        return Dispatch(fn, O, O.flatWidth, O.flatHeight, 1);
    }

    /// <inheritdoc/>
    public override Tensor LogSoftmax(Tensor X)
    {
        if (!X.shape.IsNHWC())
            return base.LogSoftmax(X);

        var O = X.shape.Flatten();

        var fn = new ComputeFunc(m_Kernels, "LogSoftmax");

        SetTensor(fn, "X", X);

        return Dispatch(fn, O, O.flatWidth, O.flatHeight, 1);
    }

    /// <inheritdoc/>
    public override Tensor Tanh(Tensor X)
    {
        if (!X.shape.IsNHWC())
            return base.Tanh(X);

        return Activation("Tanh", X);
    }

    /// <inheritdoc/>
    public override Tensor Sigmoid(Tensor X)
    {
        if (!X.shape.IsNHWC())
            return base.Sigmoid(X);

        return Activation("Sigmoid", X);
    }

    /// <inheritdoc/>
    public override Tensor Relu6(Tensor X)
    {
        if (!X.shape.IsNHWC())
            return base.Relu6(X);

        return Activation("Relu6", X);
    }

    /// <inheritdoc/>
    public override Tensor Elu(Tensor X, float alpha)
    {
        if (!X.shape.IsNHWC())
            return base.Elu(X, alpha);

        return Activation("Elu", X, alpha);
    }

    /// <inheritdoc/>
    public override Tensor LeakyRelu(Tensor X, float alpha)
    {
        if (!X.shape.IsNHWC())
            return base.LeakyRelu(X, alpha);

        return Activation("LeakyRelu", X, alpha);
    }

    /// <inheritdoc/>
    public override Tensor Selu(Tensor X, float alpha, float gamma)
    {
        if (!X.shape.IsNHWC())
            return base.Selu(X, alpha, gamma);

        return Activation("Selu", X, alpha, gamma);
    }

    /// <inheritdoc/>
    public override Tensor Swish(Tensor X)
    {
        if (!X.shape.IsNHWC())
            return base.Swish(X);

        return Activation("Swish", X);
    }

    /// <inheritdoc/>
    public override Tensor Abs(Tensor X)
    {
        if (!X.shape.IsNHWC())
            return base.Abs(X);

        return Activation("Abs", X);
    }

    /// <inheritdoc/>
    public override Tensor Neg(Tensor X)
    {
        if (!X.shape.IsNHWC())
            return base.Neg(X);

        return Activation("Neg", X);
    }

    /// <inheritdoc/>
    public override Tensor Ceil(Tensor X)
    {
        if (!X.shape.IsNHWC())
            return base.Ceil(X);

        return Activation("Ceil", X);
    }

    /// <inheritdoc/>
    public override Tensor Clip(Tensor X, float min, float max)
    {
        if (!X.shape.IsNHWC())
            return base.Clip(X, min, max);

        return Activation("Clip", X, min, max);
    }

    /// <inheritdoc/>
    public override Tensor Floor(Tensor X)
    {
        if (!X.shape.IsNHWC())
            return base.Floor(X);

        return Activation("Floor", X);
    }

    /// <inheritdoc/>
    public override Tensor Reciprocal(Tensor X)
    {
        if (!X.shape.IsNHWC())
            return base.Reciprocal(X);

        return Activation("Reciprocal", X);
    }

    /// <inheritdoc/>
    public override Tensor Pow(Tensor X, float alpha)
    {
        if (!X.shape.IsNHWC())
            return base.Pow(X, alpha);

        return Activation("Pow", X, alpha);
    }

    /// <inheritdoc/>
    public override Tensor Exp(Tensor X)
    {
        if (!X.shape.IsNHWC())
            return base.Exp(X);

        return Activation("Exp", X);
    }

    /// <inheritdoc/>
    public override Tensor Log(Tensor X)
    {
        if (!X.shape.IsNHWC())
            return base.Log(X);

        return Activation("Log", X);
    }

    /// <inheritdoc/>
    public override Tensor Sqrt(Tensor X)
    {
        if (!X.shape.IsNHWC())
            return base.Sqrt(X);

        return Activation("Sqrt", X);
    }

    /// <inheritdoc/>
    public override Tensor Acos(Tensor X)
    {
        if (!X.shape.IsNHWC())
            return base.Acos(X);

        return Activation("Acos", X);
    }

    /// <inheritdoc/>
    public override Tensor Acosh(Tensor X)
    {
        if (!X.shape.IsNHWC())
            return base.Acosh(X);

        return Activation("Acosh", X);
    }

    /// <inheritdoc/>
    public override Tensor Asin(Tensor X)
    {
        if (!X.shape.IsNHWC())
            return base.Asin(X);

        return Activation("Asin", X);
    }

    /// <inheritdoc/>
    public override Tensor Asinh(Tensor X)
    {
        if (!X.shape.IsNHWC())
            return base.Asinh(X);

        return Activation("Asinh", X);
    }

    /// <inheritdoc/>
    public override Tensor Atan(Tensor X)
    {
        if (!X.shape.IsNHWC())
            return base.Atan(X);

        return Activation("Atan", X);
    }

    /// <inheritdoc/>
    public override Tensor Atanh(Tensor X)
    {
        if (!X.shape.IsNHWC())
            return base.Atanh(X);

        return Activation("Atanh", X);
    }

    /// <inheritdoc/>
    public override Tensor Cos(Tensor X)
    {
        if (!X.shape.IsNHWC())
            return base.Cos(X);

        return Activation("Cos", X);
    }

    /// <inheritdoc/>
    public override Tensor Cosh(Tensor X)
    {
        if (!X.shape.IsNHWC())
            return base.Cosh(X);

        return Activation("Cosh", X);
    }

    /// <inheritdoc/>
    public override Tensor Sin(Tensor X)
    {
        if (!X.shape.IsNHWC())
            return base.Sin(X);

        return Activation("Sin", X);
    }

    /// <inheritdoc/>
    public override Tensor Sinh(Tensor X)
    {
        if (!X.shape.IsNHWC())
            return base.Sinh(X);

        return Activation("Sinh", X);
    }

    /// <inheritdoc/>
    public override Tensor Tan(Tensor X)
    {
        if (!X.shape.IsNHWC())
            return base.Tan(X);

        return Activation("Tan", X);
    }

    /// <inheritdoc/>
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

    /// <summary>
    /// Elementwise broadcast for specified kernel
    /// </summary>
    /// <param name="kernelName">kernel name</param>
    /// <param name="tensors">input tensors</param>
    /// <returns>output `Tensor`</returns>
    /// <exception cref="NotImplementedException">thrown if input `Tensor` is not compatible with 4D shape</exception>
    protected virtual Tensor ElementwiseWithBroadcast(string kernelName, Tensor[] tensors)
    {
        if (!TensorExtensions.AreAllTensorsConvertibleToNCHW(tensors))
            throw new NotImplementedException();

        var O = TensorExtensions.MaxShape(tensors);

        Assert.IsTrue(tensors.Length > 0);
        var X = tensors[0];

        var fn = new ComputeFunc(m_Kernels, kernelName);
        bool isFirstDispatch = true;
        for (int t = 1; t < tensors.Length; ++t)
        {
            var B = tensors[t];

            // B and X can be constants, in that cases the internal layout does not match ComputeInfo.channelsOrder and will allways be NHWC
            // => permute them if there is a layout mismatch
            if (Pin(X).channelsOrder == ComputeInfo.ChannelsOrder.NHWC && ComputeInfo.channelsOrder == ComputeInfo.ChannelsOrder.NCHW)
                X = TransposeToNCHW(X);
            if (Pin(B).channelsOrder == ComputeInfo.ChannelsOrder.NHWC && ComputeInfo.channelsOrder == ComputeInfo.ChannelsOrder.NCHW)
                B = TransposeToNCHW(B);

            SetTensor(fn, "X", X);
            SetTensor(fn, "B", B);
            fn.shader.SetFloat("_Alpha", 1.0f/(float)tensors.Length);
            fn.shader.SetInt("_IsFirstDispatch", isFirstDispatch ? 1 : 0);

            X = Dispatch(fn, O, O.channels, O.width, O.height);
            isFirstDispatch = false;
        }

        return X;
    }

    /// <inheritdoc/>
    public override Tensor Add(Tensor[] tensors)
    {
        return ElementwiseWithBroadcast("BroadcastAdd", tensors);
    }

    /// <inheritdoc/>
    public override Tensor Sub(Tensor[] tensors)
    {
        return ElementwiseWithBroadcast("BroadcastSub", tensors);
    }

    /// <inheritdoc/>
    public override Tensor Mul(Tensor[] tensors)
    {
        return ElementwiseWithBroadcast("BroadcastMul", tensors);
    }

    /// <inheritdoc/>
    public override Tensor Div(Tensor[] tensors)
    {
        return ElementwiseWithBroadcast("BroadcastDiv", tensors);
    }

    /// <inheritdoc/>
    public override Tensor Pow(Tensor[] tensors)
    {
        return ElementwiseWithBroadcast("BroadcastPow", tensors);
    }

    /// <inheritdoc/>
    public override Tensor Min(Tensor[] tensors)
    {
        return ElementwiseWithBroadcast("BroadcastMin", tensors);
    }

    /// <inheritdoc/>
    public override Tensor Max(Tensor[] tensors)
    {
        return ElementwiseWithBroadcast("BroadcastMax", tensors);
    }

    /// <inheritdoc/>
    public override Tensor Mean(Tensor[] tensors)
    {
        return ElementwiseWithBroadcast("BroadcastMean", tensors);
    }

    /// <summary>
    /// Reduce with specified kernel
    /// </summary>
    /// <param name="kernelName">kernel name</param>
    /// <param name="X">input</param>
    /// <param name="axis">axis</param>
    /// <returns>output `Tensor`</returns>
    /// <exception cref="NotImplementedException">thrown if `axis` is not 4D compatible (depends on compute backend limitations)</exception>
    protected virtual Tensor Reduce(string kernelName, Tensor X, int axis)
    {
        axis = X.shape.Axis(axis);

        if (!X.shape.IsNHWC() || !TensorExtensions.Is8DAxisConvertibleToNHWC(axis))
            throw new NotImplementedException();

        //TODO optimize when reducing not on channel.
        bool needTranpose = axis != TensorShape.C;
        var axisNCHW = TensorExtensions.Convert8DAxisToNHWC(axis);
        var permuteTargetAxisAndC = new int[] {0, 1, 2, axisNCHW};
        permuteTargetAxisAndC[axisNCHW] = 3;
        if (needTranpose)
            X = Transpose(X, permuteTargetAxisAndC);

        var oShape = X.shape.Reduce(TensorShape.C);
        Assert.AreEqual(oShape.channels, 1);

        var fn = new ComputeFunc(m_Kernels, kernelName);
        SetTensor(fn, "X", X);

        var O = Dispatch(fn, oShape, oShape.width, oShape.height, 1);

        if (needTranpose)
            O = Transpose(O, permuteTargetAxisAndC);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor ArgMax(Tensor X, int axis)
    {
        return Reduce("ArgMax", X, axis);
    }

    /// <inheritdoc/>
    public override Tensor ArgMin(Tensor X, int axis)
    {
        return Reduce("ArgMin", X, axis);
    }

    /// <inheritdoc/>
    public override Tensor ReduceMin(Tensor X, int axis)
    {
        return Reduce("ReduceMin", X, axis);
    }

    /// <inheritdoc/>
    public override Tensor ReduceMax(Tensor X, int axis)
    {
        return Reduce("ReduceMax", X, axis);
    }

    /// <inheritdoc/>
    public override Tensor ReduceSum(Tensor X, int axis)
    {
        return Reduce("ReduceSum", X, axis);
    }

    /// <inheritdoc/>
    public override Tensor ReduceMean(Tensor X, int axis)
    {
        return Reduce("ReduceMean", X, axis);
    }

    /// <inheritdoc/>
    public override Tensor ReduceProd(Tensor X, int axis)
    {
        return Reduce("ReduceProd", X, axis);
    }

    /// <inheritdoc/>
    public override Tensor Greater(Tensor A, Tensor B)
    {
        return ElementwiseWithBroadcast("BroadcastGreater", new Tensor[] { A, B });
    }

    /// <inheritdoc/>
    public override Tensor GreaterEqual(Tensor A, Tensor B)
    {
        return ElementwiseWithBroadcast("BroadcastGreaterEqual", new Tensor[] { A, B });
    }

    /// <inheritdoc/>
    public override Tensor Less(Tensor A, Tensor B)
    {
        return ElementwiseWithBroadcast("BroadcastLess", new Tensor[] { A, B });
    }

    /// <inheritdoc/>
    public override Tensor LessEqual(Tensor A, Tensor B)
    {
        return ElementwiseWithBroadcast("BroadcastLessEqual", new Tensor[] { A, B });
    }

    /// <inheritdoc/>
    public override Tensor Equal(Tensor A, Tensor B)
    {
        return ElementwiseWithBroadcast("BroadcastEqual", new Tensor[] { A, B });
    }

    /// <inheritdoc/>
    public override Tensor LogicalOr(Tensor A, Tensor B)
    {
        return ElementwiseWithBroadcast("BroadcastLogicalOr", new Tensor[] { A, B });
    }

    /// <inheritdoc/>
    public override Tensor LogicalAnd(Tensor A, Tensor B)
    {
        return ElementwiseWithBroadcast("BroadcastLogicalAnd", new Tensor[] { A, B });
    }

    /// <inheritdoc/>
    public override Tensor LogicalXor(Tensor A, Tensor B)
    {
        return ElementwiseWithBroadcast("BroadcastLogicalXor", new Tensor[] { A, B });
    }

    /// <inheritdoc/>
    public override Tensor LogicalNot(Tensor X)
    {
        return Activation("LogicalNot", X);
    }

    /// <summary>
    /// Copy and reshape tensor for NCHW layout
    /// </summary>
    /// <param name="X">input</param>
    /// <param name="newShape">new shape</param>
    /// <returns>output `Tensor`</returns>
    protected virtual Tensor CopyAndReshape_NCHW(Tensor X, TensorShape newShape)
    {
        Assert.AreEqual(X.length, newShape.length);
        Assert.AreEqual(ComputeInfo.ChannelsOrder.NCHW, ComputeInfo.channelsOrder);

        var O = NewTensor(newShape, "O");

        if (X.shape.IsNHWC() && newShape.IsNHWC())
        {
            var fn = new ComputeFunc(m_Kernels, "ReshapeFromNHWCModel_NCHW");
            SetTensor(fn, "X", X);
            SetTensor(fn, "O", O);
            fn.Dispatch( O.width, O.height, O.channels);
        }
        else
        {
            var fn = new ComputeFunc(m_Kernels, "Reshape8DFromChannelFirstModel_NCHW");
            SetTensor(fn, "X", X);
            SetTensor(fn, "O", O);
            var xD  = new[] {X.shape[0], X.shape[1],X.shape[3],X.shape[4]};
            var oD  = new[] {O.shape[0], O.shape[1],O.shape[3],O.shape[4]};
            fn.shader.SetInts("_Pad", xD);
            fn.shader.SetInts("_Pool", oD);
            fn.Dispatch( O.width, O.height, O.channels);
        }

        return O;
    }

    /// <inheritdoc/>
    protected override Tensor CopyAndReshape(Tensor X, TensorShape newShape)
    {
        Assert.AreEqual(X.length, newShape.length);
        if (X.shape != newShape)
        {
            //In CHW mode one should call CopyAndReshape_NCHW if shape is modified
            Assert.AreEqual(ComputeInfo.ChannelsOrder.NHWC, ComputeInfo.channelsOrder);
        }
        bool isNHWCCopy = X.shape.IsNHWC() && newShape.IsNHWC();

        // NOTE: "Copy" kernel copies tensor data while preserving the shape
        // However here in CopyAndReshape we want to both copy and change the shape,
        // To be able to piggyback "Copy" kernel we specify new shape when allocating destination tensor,
        // but use shape identical to source when copying.
        var O = NewTensor(newShape, "O");
        var fn = new ComputeFunc(m_Kernels, isNHWCCopy?"Copy":"Copy8D");
        SetTensor(fn, "X", X);
        var copyShape = X.shape;
        fn.SetTensor("O", copyShape, Pin(O).buffer);

        if (isNHWCCopy)
        {
            var offsets = new int[] {0, 0, 0, 0};
            fn.shader.SetInts("_Pad", offsets);
        }
        else
        {
            var XonDeviceShape = GetOnDeviceShape(X.shape);
            var d0_3  = new[] {XonDeviceShape[0], XonDeviceShape[1],XonDeviceShape[2],XonDeviceShape[3]};
            var d4_7  = new[] {XonDeviceShape[4], XonDeviceShape[5],XonDeviceShape[6],XonDeviceShape[7]};
            fn.shader.SetInts("_Stride", d0_3);
            fn.shader.SetInts("_Pool", d4_7);
        }

        fn.Dispatch(X.channels, X.width, X.height);
        return O;
    }

    /// <inheritdoc/>
    public override Tensor Flatten(Tensor X)
    {
        var newShape = X.shape.Flatten();
        if (X.shape == newShape)
            return base.Flatten(X);

        return CopyAndReshape_NCHW(X, newShape);
    }

    /// <inheritdoc/>
    public override Tensor Reshape(Tensor X, TensorShape newShape)
    {
        if (ComputeInfo.channelsOrder == ComputeInfo.ChannelsOrder.NHWC || X.shape == newShape)
            return base.Reshape(X, newShape);

        return CopyAndReshape_NCHW(X, newShape);
    }

    /// <inheritdoc/>
    public override Tensor Transpose(Tensor X)
    {
        Assert.IsTrue(X.dimensions <= 2);
        var O = new TensorShape(X.flatWidth, X.flatHeight);

        var fn = new ComputeFunc(m_Kernels, "Transpose2D");
        SetTensor(fn, "X", X);
        return Dispatch(fn, O, O.flatWidth, O.flatHeight, 1);
    }

    /// <summary>
    /// Get `Tensor` shape on GPU device
    /// </summary>
    /// <param name="shape">shape</param>
    /// <returns>ouput shape as int array</returns>
    protected int[] GetOnDeviceShape(TensorShape shape)
    {
        var onDeviceShape = shape.ToArray();
        if (ComputeInfo.channelsOrder == ComputeInfo.ChannelsOrder.NCHW)
        {
            //SRNTDHWC --> SRNCTDHW
            var numChannel = onDeviceShape[7];
            onDeviceShape[7] = onDeviceShape[6];
            onDeviceShape[6] = onDeviceShape[5];
            onDeviceShape[5] = onDeviceShape[4];
            onDeviceShape[4] = onDeviceShape[3];
            onDeviceShape[3] = numChannel;
        }
        return onDeviceShape;
    }

    /// <summary>
    /// Convert permutation list to device specific layout
    /// </summary>
    /// <param name="permutationChannelLast">permutations channels last</param>
    /// <returns>new permutation list</returns>
    protected int[] ConvertPermutationToDeviceLayout(int[] permutationChannelLast)
    {
        if (ComputeInfo.channelsOrder == ComputeInfo.ChannelsOrder.NHWC)
            return permutationChannelLast;

        var permutationChannelFirst = new int[TensorShape.MaxRank];
        var channelLastToFirst = new[] {0, 1, 2, 7, 3, 4, 5, 6};
        for (int i = 0; i < TensorShape.MaxRank; ++i)
        {
            int sourceDestinationSemanticIndex = channelLastToFirst[i];
            int sourcePermutationSemanticIndex = permutationChannelLast[sourceDestinationSemanticIndex];
            permutationChannelFirst[i] = Array.IndexOf(channelLastToFirst, sourcePermutationSemanticIndex);
        }

        return permutationChannelFirst;
    }

    /// <inheritdoc/>
    public Tensor Transpose8D(Tensor X, int[] permutations)
    {
        permutations = TensorExtensions.Get8DPermutationsForNHWCPermutationsAndShape(X.shape, permutations);

        // See: Permute() in ONNXTensor.cs and https://stackoverflow.com/a/32034565
        var Oshape = X.shape.Permute(permutations);

        var OonDeviceShape = GetOnDeviceShape(Oshape);
        var XonDeviceShape = GetOnDeviceShape(X.shape);
        var onDevicePermutation = ConvertPermutationToDeviceLayout(permutations);

        // outTensor strides
        var reversePermute = new int[permutations.Length];
        for (var i = 0; i < permutations.Length; ++i)
            reversePermute[i] = Array.IndexOf(onDevicePermutation, i);
        var tempOutStrides = new int[TensorShape.MaxRank+1];
        tempOutStrides[8] = 1;
        for (int i = 7; i >= 0; --i)
            tempOutStrides[i] = tempOutStrides[i+1] * OonDeviceShape[i];
        var outStride = new int[reversePermute.Length];
        for (var i = 0; i < reversePermute.Length; ++i)
            outStride[i] = tempOutStrides[reversePermute[i] + 1];

        var d0_3  = new[] {XonDeviceShape[0], XonDeviceShape[1],XonDeviceShape[2],XonDeviceShape[3]};
        var d4_7  = new[] {XonDeviceShape[4], XonDeviceShape[5],XonDeviceShape[6],XonDeviceShape[7]};
        var outStride0_3 = new[] {outStride[0],outStride[1],outStride[2],outStride[3]};
        var outStride4_7 = new[] {outStride[4],outStride[5],outStride[6],outStride[7]};

        var fn = new ComputeFunc(m_Kernels, "Transpose8D");
        SetTensor(fn, "X", X);
        fn.shader.SetInts("_Pad", d0_3);
        fn.shader.SetInts("_Pool", d4_7);
        fn.shader.SetInts("_Stride", outStride0_3);
        fn.shader.SetInts("_ChannelWriteMask", outStride4_7);

        if (ComputeInfo.channelsOrder == ComputeInfo.ChannelsOrder.NCHW)
            return Dispatch(fn, Oshape, X.width, X.height, X.depth);
        else
            return Dispatch(fn, Oshape, X.channels, X.width, X.height);

    }

    /// <inheritdoc/>
    public override Tensor Transpose(Tensor X, int[] permutations)
    {
        if (!X.shape.IsNHWC() || permutations.Length != 4)
            return Transpose8D(X, permutations);

        Assert.AreEqual(permutations.Length, 4);

        var O = X.shape.Permute(permutations);

        var fn = new ComputeFunc(m_Kernels, "Transpose");
        SetTensor(fn, "X", X);
        fn.shader.SetInts("_Pool", permutations);
        return Dispatch(fn, O, X.channels, X.width, X.height);
    }

    Tensor TransposeToNCHW(Tensor X)
    {
        var O = X.shape;
        var fn = new ComputeFunc(m_Kernels, "TransposeToNCHW");
        SetTensor(fn, "X", X);
        return Dispatch(fn, O, X.channels, X.width, X.height);
    }

    internal static int[] s_ConcatOffsets = new int[4];
    /// <inheritdoc/>
    public override Tensor Concat(Tensor[] tensors, int axis)
    {
        if (axis != TensorShape.C && axis != -1)
            return base.Concat(tensors, axis);

        if (!TensorExtensions.AreAllTensorsConvertibleToNCHW(tensors) || !TensorExtensions.Is8DAxisConvertibleToNHWC(axis))
            return base.Concat(tensors, axis);

        var O = TensorExtensions.Concat(tensors, axis);
        var offsets = s_ConcatOffsets;
        Array.Clear(offsets, 0, offsets.Length);
        axis = O.Axis(axis);
        var axisNCHW = TensorExtensions.Convert8DAxisToNHWC(axis);

        var fn = new ComputeFunc(m_Kernels, "Copy");
        var result = NewTensor(fn, "O", O);
        foreach (var X in tensors)
        {
            SetTensor(fn, "X", X);
            fn.shader.SetInts("_Pad", offsets);
            fn.Dispatch(X.channels, X.width, X.height);

            offsets[axisNCHW] += X.shape[axis];
        }
        return result;
    }

    static private int[] s_StridedSliceStart = new int[4];
    /// <inheritdoc/>
    public override Tensor StridedSlice(Tensor X, int[] starts, int[] ends, int[] stride)
    {
        if (!X.shape.IsNHWC())
        {
            //TODO implement 8D support GPU path.
            return base.StridedSlice(X, starts, ends, stride);
        }

        starts = TensorExtensions.GetNHWCParametersFrom8DParameterAndShape(X.shape, starts);
        ends = TensorExtensions.GetNHWCParametersFrom8DParameterAndShape(X.shape, ends);
        stride = TensorExtensions.GetNHWCParametersFrom8DParameterAndShape(X.shape, stride);

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

    /// <inheritdoc/>
    public override Tensor Tile(Tensor X, int[] repeats)
    {
        // @TODO: GPU implementation
        return base.Tile(X, repeats);
    }

    /// <inheritdoc/>
    public override Tensor Gather(Tensor[] tensors, int axis)
    {
        if (!TensorExtensions.Is8DAxisConvertibleToNHWC(axis) || !tensors[0].shape.IsNHWC())
        {
            //TODO implement 8D support GPU path.
            return base.Gather(tensors, axis);
        }

        Tensor X = tensors[0];
        Tensor indices = tensors[1];

        var outputShape = X.shape;
        outputShape[axis] = indices.length;

        var fn = new ComputeFunc(m_Kernels, "Gather");
        SetTensor(fn, "X", X);
        SetTensor(fn, "K", indices);
        fn.shader.SetInt("_Axis", TensorExtensions.Convert8DAxisToNHWC(axis));

        return Dispatch(fn, outputShape, outputShape.channels, outputShape.width, outputShape.height);
    }

    /// <inheritdoc/>
    public override Tensor Copy(Tensor X)
    {
        return base.Copy(X);
    }

    /// <inheritdoc/>
    public override Tensor Prepare(Tensor X)
    {
        Pin(X);
        return X;
    }
}

internal struct ComputeFunc
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
