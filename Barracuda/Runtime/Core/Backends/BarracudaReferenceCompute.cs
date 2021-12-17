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
public class ComputeTensorData : UniqueResourceId, ITensorData
{
    private bool m_DisposeBufferAfterUse;
    private ComputeBuffer m_Buffer;
    private TensorShape m_Shape;
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

        m_Shape = shape;
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
    internal ComputeTensorData(ComputeBuffer buffer, TensorShape shape, int offset, string buffername, ComputeInfo.ChannelsOrder onDeviceChannelsOrder)
    {
        m_OnDeviceChannelsOrder = onDeviceChannelsOrder;
        name = buffername;
        m_Buffer = buffer;
        m_Shape = shape;
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
        if (count > maxCapacity)
            throw new ArgumentException("ComputeTensorData buffer is too small to reserve " + count + " elements.");
    }

    /// <inheritdoc/>
    public virtual void Upload(float[] data, TensorShape shape, int managedBufferStartIndex = 0)
    {
        var numItemToCopy = shape.length;
        var numItemAvailableInData = data.Length - managedBufferStartIndex;

        Assert.IsTrue(managedBufferStartIndex >= 0);
        Assert.IsTrue(numItemToCopy <= numItemAvailableInData);

        if (m_OnDeviceChannelsOrder == ComputeInfo.ChannelsOrder.NCHW)
        {
            //Transpose from HWC to CHW, TODO use a compute shader or threaded code.
            Profiler.BeginSample("Tensor.Upload_ChannelFirstTranpose");
            float[] chwData = new float[numItemToCopy];
            if (shape.Is4D())
            {
                for (int readIndex=0; readIndex < numItemToCopy; ++readIndex)
                {
                    int b = 0, h = 0, w = 0, ch = 0;
                    shape.GetPositionsFromIndex(readIndex, ref b, ref h, ref w, ref ch);
                    int writeIndex = shape.IndexChannelFirst(b, h, w, ch);
                    chwData[writeIndex] = data[managedBufferStartIndex+readIndex];
                }
            }
            else
            {
                for (int readIndex=0; readIndex < numItemToCopy; ++readIndex)
                {
                    int s = 0, r = 0, n = 0, t = 0, d = 0, h = 0, w = 0, ch = 0;
                    shape.GetPositionsFromIndex(readIndex, ref s, ref r, ref n, ref t, ref d, ref h, ref w, ref ch);
                    int writeIndex = shape.IndexChannelFirst(s, r, n, t, d, h, w, ch);
                    chwData[writeIndex] = data[managedBufferStartIndex+readIndex];
                }
            }
            Profiler.EndSample();
            m_Buffer.SetData(chwData, 0, m_Offset, numItemToCopy);
        }
        else
        {
            m_Buffer.SetData(data, managedBufferStartIndex, m_Offset, numItemToCopy);
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

        bool isAndroidPlayer = false;
        #if UNITY_ANDROID
            isAndroidPlayer = true;
        #endif

        var data = new float[count];
        if (isAndroidPlayer && m_Offset != 0)
        {
            //On mobile GetData does not take m_Offset into account, need a full download.
            var fullData = new float[m_Buffer.count];
            m_Buffer.GetData(fullData);
            Array.Copy(fullData, m_Offset, data, 0, count);
        }
        else
        {
            m_Buffer.GetData(data, 0, m_Offset, count);
        }

        data = ConvertFromOnDeviceFormat(shape, data);
        Profiler.EndSample();

        return data;
    }

    /// <inheritdoc/>
    public virtual BarracudaArray SharedAccess(out int offset)
    {
        offset = 0;
        return new BarracudaArrayFromManagedArray(Download(new TensorShape(0, 0, 0, maxCapacity)));//TODO fp16
    }

    /// <inheritdoc/>
    public virtual int maxCapacity => m_Shape.length;

    /// <inheritdoc/>
    public virtual DataType dataType => DataType.Float; //todo fp16

    /// <inheritdoc/>
    public virtual bool inUse => true;

    /// <inheritdoc/>
    public virtual bool isGPUMem => true;

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
            name, GetHashCode(), m_Shape, m_Buffer, allocationSource);
    }
}

internal class SharedComputeTensorData : ComputeTensorData
{
    public SharedComputeTensorData(ComputeBuffer buffer, TensorShape shape, int offset, string buffername = "", ComputeInfo.ChannelsOrder channelsOrder = ComputeInfo.ChannelsOrder.NHWC) : base(buffer, shape, offset, buffername, channelsOrder) {}
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

    public static int[] FormatToChannelMask(Texture tex, int interpretPixelAsChannels)
    {
        switch (interpretPixelAsChannels)
        {
            case 1:
                if (IsRedOnly(tex))
                    return new [] { 1,0,0,0 };
                if (IsAlphaOnly(tex))
                    return new [] { 0,0,0,1 };
                // TODO: known issue, doesn't handle RG textures properly
                return new [] { 0,0,0,0 }; // see specialCaseWhenChannelMaskIsEmptyStoresAverage
            case 2:
                return new [] { 1,1,0,0 };
            case 3:
                return new [] { 1,1,1,0 };
            default:
                return new [] { 1,1,1,1 };
        }
    }

    public static int[] FormatToChannelReadMap(Texture tex, int interpretPixelAsChannels)
    {
        // -1 == use default channel value, otherwise channel index

        if (IsRedOnly(tex))
            return new[] { 0, -1, -1, -1 };
        if (IsAlphaOnly(tex))
            return new[] { -1, -1, -1, 3 };

        switch (interpretPixelAsChannels)
        {
            case 1:
                // TODO: known issue, doesn't handle RG textures properly
                return new [] { -1,-1,-1,-1 }; // see specialCaseWhenChannelMaskIsEmptyStoresAverage
            case 2:
                return new[] { 0, 1, -1, -1 };
            case 3:
                return new[] { 0, 1, 2, -1 };
            default:
                return new[] { 0, 1, 2, 3 };
        }
    }
}

/// <summary>
/// Reference GPU compute `IOps` implementation
/// </summary>
public class ReferenceComputeOps : ReferenceCPUOps
{
    /// <summary>
    /// Create `ReferenceComputeOps`
    /// </summary>
    /// <param name="allocator">allocator</param>
    public ReferenceComputeOps(ITensorAllocator allocator = null)
    : base(allocator)
    {
    }

    /// <summary>
    /// Pin `Tensor` to GPU compute device, if `uploadCache` is false, data is not uploaded to device and `Tensor` is not 0-filled
    /// </summary>
    /// <param name="X">`Tensor`</param>
    /// <param name="uploadCache">`bool`</param>
    /// <returns>`ComputeTensorData`</returns>
    /// <summary>
    public ComputeTensorData Pin(Tensor X, bool uploadCache = true)
    {
        X.FlushCache(uploadCache);

        var onDevice = X.tensorOnDevice as ComputeTensorData;
        if (onDevice == null)
        {
            var asTexture = X.tensorOnDevice as TextureAsTensorData;
            if (asTexture != null)
                X.AttachToDevice(TextureToTensorData(asTexture, X.name));
            else
            {
                if (uploadCache)
                    X.UploadToDevice(new ComputeTensorData(X.shape, X.name, ComputeInfo.channelsOrder)); // device is not compatible, create new array and upload
                else
                    X.AllocateOnDevice(new ComputeTensorData(X.shape, X.name, ComputeInfo.channelsOrder, false)); // device is not compatible, create new array but do not upload nor 0-fill
            }
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

    internal Tensor NewTensor(ComputeFunc fn, string name, DataType dataType, TensorShape shape, AllocScope scope = AllocScope.LayerOutput)
    {
        var o = NewTensor(dataType, shape, scope, name);
        fn.SetTensor(name, shape, Pin(o).buffer);
        return o;
    }

    internal Tensor Dispatch(ComputeFunc fn, DataType dataType, TensorShape outputShape, int workItemsX, int workItemsY, int workItemsZ, string outputName = "O")
    {
        var o = NewTensor(fn, outputName, dataType, outputShape);
        fn.Dispatch(workItemsX, workItemsY, workItemsZ);
        return o;
    }

    // ---------------------------------------------------------------------------------

    internal ITensorData TextureToTensorData(TextureAsTensorData texData, string name)
    {
        var fn = new ComputeFunc(ComputeShaderContext.Optimized, "TextureToTensor", GetModelExecutionsReporter());
        var tensorData = new ComputeTensorData(texData.shape, name, ComputeInfo.channelsOrder, false);

        fn.SetTensor("O", texData.shape, tensorData.buffer);
        fn.shader.SetBool("_FlipY", texData.flip == TextureAsTensorData.Flip.Y);
        fn.shader.SetVector("_Scale", texData.scale);
        fn.shader.SetVector("_Bias", texData.bias);

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

            fn.SetTexture("X", tex);
            fn.shader.SetInts("_Pool", new int [] {tex.width, tex.height});
            fn.shader.SetInts("_Pad", offsets);
            fn.shader.SetInts("_ChannelWriteMask",
                TextureFormatUtils.FormatToChannelMask(tex, texData.interpretPixelAsChannels));
            fn.shader.SetInts("_ChannelReadMap",
                TextureFormatUtils.FormatToChannelReadMap(tex, texData.interpretPixelAsChannels));

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
    /// <param name="flipY">flips the texture along the Y dimension (optional, default: true)</param>
    public void TensorToRenderTexture(Tensor X, RenderTexture target, int batch, int fromChannel, Vector4 scale, Vector4 bias, Texture3D lut, bool flipY = true)
    {
        if (!target.enableRandomWrite || !target.IsCreated())
        {
            target.Release();
            target.enableRandomWrite = true;
            target.Create();
        }

        var fn = new ComputeFunc(ComputeShaderContext.Optimized, "TensorToTexture"+ (lut == null?"NoLUT":"3DLUT"), GetModelExecutionsReporter());
        SetTensor(fn, "X", X);
        fn.SetTexture("O", target);
        fn.shader.SetVector("_Scale", scale);
        fn.shader.SetVector("_Bias", bias);
        fn.shader.SetInts("_Pad", new int[] { batch, 0, 0, fromChannel });
        fn.shader.SetBool("_FlipY", flipY);
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
    /// Check if `fusedActivation` type is supported in place
    /// </summary>
    /// <param name="fusedActivation">fused activation type</param>
    /// <returns>`true` if supported</returns>
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

    // ---------------------------------------------------------------------------------
    /// <inheritdoc/>
    public override Tensor MatMul(Tensor X, int rankX, Tensor Y, int rankY)
    {
        // N.B: Current implementation is inefficient as it introduces Transposes/Slice and Concat.
        // => consider refactoring dense to support batch

        // X and Y can be constants, in that cases the internal layout does not match ComputeInfo.channelsOrder and will allways be NHWC
        // => permute them if there is a layout mismatch
        X = GetTensorInCurrentMemoryLayoutHelper(X);
        Y = GetTensorInCurrentMemoryLayoutHelper(Y);

        // V-Table magic, ReferenceCPU.MaMul is calls MatMul2D, Concat & Slice all which are overloaded by all respective IOps, so will call the correct backend
        return base.MatMul(X, rankX, Y, rankY);
    }

    /// <inheritdoc/>
    public override Tensor MatMul(Tensor X, bool xTranspose, Tensor Y, bool yTranspose)
    {
        X = GetTensorInCurrentMemoryLayoutHelper(X);
        Y = GetTensorInCurrentMemoryLayoutHelper(Y);

        // MatMul implementation in terms of Dense
        var A = (xTranspose) ? Transpose(X): X;
        var B = (yTranspose) ? Transpose(Y): Y;
        var C = NewTempTensor(X.dataType, new TensorShape(1, B.flatWidth));
        var Z = Sub(new[] { C, C }); // initialize bias with zeros, TODO will fragment ping pong allocator

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

        var fn = new ComputeFunc(ComputeShaderContext.Reference, "Dense", GetModelExecutionsReporter());

        SetTensor(fn, "X", X);
        SetTensor(fn, "W", W);
        SetTensor(fn, "B", B);
        fn.shader.SetInt("_ActivationMode", (int)fusedActivation);

        var O = Dispatch(fn, X.dataType, Oshape, Oshape.flatWidth, Oshape.flatHeight, 1);

        if (!IsFusedActivationSupported(fusedActivation))
            O = Activation(fusedActivation.ToString(), O);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Dense3(Tensor X, Tensor W, Tensor B)
    {
        var Oshape = new TensorShape(X.batch, 1, W.channels, X.channels);

        var fn = new ComputeFunc(ComputeShaderContext.Reference, "Dense3", GetModelExecutionsReporter());

        SetTensor(fn, "X", X);
        SetTensor(fn, "W", W);
        SetTensor(fn, "B", B);

        var O = Dispatch(fn, X.dataType, Oshape, Oshape.width, Oshape.channels, Oshape.batch);

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
        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(X.channels, K.kernelDepth);
        Assert.AreEqual(K.kernelCount, B.flatWidth);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);

        var Oshape = X.shape.ApplyKernel(K.shape, stride, pad);

        var fn = new ComputeFunc(ComputeShaderContext.Reference, "Conv2DWinograd_2x2_3x3", GetModelExecutionsReporter());

        SetTensor(fn, "X", X);
        SetTensor(fn, "K", K);
        SetTensor(fn, "B", B);

        fn.shader.SetInts("_Pad", pad);
        fn.shader.SetInt("_ActivationMode", (int)fusedActivation);

        var O = Dispatch(fn, X.dataType, Oshape, K.kernelCount, ComputeHelper.IDivC(Oshape.width, 2), ComputeHelper.IDivC(Oshape.height, 2));

        if (!IsFusedActivationSupported(fusedActivation))
            O = Activation(fusedActivation.ToString(), O);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Conv3D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, Layer.FusedActivation fusedActivation)
    {
        Assert.IsTrue(X.shape.IsNDHWC());
        Assert.AreEqual(X.channels, K.kernelDepth);
        Assert.AreEqual(K.kernelCount, B.flatWidth);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(stride.Length, 3);//WHD
        Assert.AreEqual(pad.Length, 6);

        var Oshape = X.shape.ApplyKernel(K.shape, stride, pad);

        var fn = new ComputeFunc(ComputeShaderContext.Reference, "Conv3D", GetModelExecutionsReporter());

        SetTensor(fn, "X", X);
        SetTensor(fn, "K", K);
        SetTensor(fn, "B", B);
        fn.shader.SetInts("_Stride", stride);
        fn.shader.SetInts("_Pad", pad.Take(3).ToArray());
        fn.shader.SetInt("_ActivationMode", (int)fusedActivation);

        var O = Dispatch(fn, X.dataType, Oshape, K.kernelCount, Oshape.width, Oshape.height);

        if (!IsFusedActivationSupported(fusedActivation))
            O = Activation(fusedActivation.ToString(), O);

        return O;
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

        bool useWinograd = (K.kernelWidth == 3) && (K.kernelHeight == 3) && (stride[0] == 1) && (stride[1] == 1) && ((Oshape.height % 2) == 0) && ((Oshape.width % 2) == 0);
        if( useWinograd )
        {
            return Conv2DWinograd(X, K, B, stride, pad, fusedActivation);
        }

        var fn = new ComputeFunc(ComputeShaderContext.Reference, "Conv2D", GetModelExecutionsReporter());

        SetTensor(fn, "X", X);
        SetTensor(fn, "K", K);
        SetTensor(fn, "B", B);
        fn.shader.SetInts("_Stride", stride);
        fn.shader.SetInts("_Pad", pad);
        fn.shader.SetInt("_ActivationMode", (int)fusedActivation);

        var O = Dispatch(fn, X.dataType, Oshape, K.kernelCount, Oshape.width, Oshape.height);

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

        var fn = new ComputeFunc(ComputeShaderContext.Reference, "DepthwiseConv2D", GetModelExecutionsReporter());

        SetTensor(fn, "X", X);
        SetTensor(fn, "K", K);
        SetTensor(fn, "B", B);
        fn.shader.SetInts("_Stride", stride);
        fn.shader.SetInts("_Pad", pad);
        fn.shader.SetInt("_ActivationMode", (int)fusedActivation);

        var O = Dispatch(fn, X.dataType, Oshape, K.kernelCount, Oshape.width, Oshape.height);

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

        // one pass version
        pad = new int[]
        {
            K.kernelWidth - pad[0] - 1, K.kernelHeight - pad[1] - 1,
            K.kernelWidth - pad[2] - 1, K.kernelHeight - pad[3] - 1
        };

        var fn = new ComputeFunc(ComputeShaderContext.Reference, "Conv2DTrans", GetModelExecutionsReporter());

        SetTensor(fn, "X", X);
        SetTensor(fn, "K", K);
        SetTensor(fn, "B", B);
        fn.shader.SetInts("_Stride", stride);
        fn.shader.SetInts("_Pad", pad);
        fn.shader.SetInt("_ActivationMode", (int)fusedActivation);

        var O = Dispatch(fn, X.dataType, Oshape, K.kernelCount, Oshape.width, Oshape.height);

        if (!IsFusedActivationSupported(fusedActivation))
            O = Activation(fusedActivation.ToString(), O);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Upsample2D(Tensor X, int[] scale, bool bilinear)
    {
        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(scale.Length, 2);

        var O = new TensorShape(X.batch, X.height*scale[1], X.width*scale[0], X.channels);

        var fn = new ComputeFunc(ComputeShaderContext.Reference, bilinear ? "UpsampleBilinear2D": "Upsample2D", GetModelExecutionsReporter());

        SetTensor(fn, "X", X);

        fn.shader.SetInts("_Pool", scale);

        if (bilinear) // dispatches over output dimensions (O)
            return Dispatch(fn, X.dataType, O, O.channels, O.width, O.height);
        else // dispatches over input dimensions (X)
            return Dispatch(fn, X.dataType, O, X.channels, X.width, X.height);
    }

    /// <inheritdoc/>
    public override Tensor Upsample3D(Tensor X, int[] scale, bool trilinear)
    {
        Assert.IsTrue(X.shape.IsNDHWC());
        Assert.AreEqual(scale.Length, 3);

        var O = new TensorShape(1, 1, X.batch, 1, X.depth*scale[2], X.height*scale[1], X.width*scale[0], X.channels);

        var fn = new ComputeFunc(ComputeShaderContext.Reference, trilinear ? "UpsampleTrilinear3D": "Upsample3D", GetModelExecutionsReporter());

        SetTensor(fn, "X", X);

        fn.shader.SetInts("_Pool", scale);

        if (trilinear) // dispatches over output dimensions (O)
            return Dispatch(fn, X.dataType, O, O.channels, O.width, O.height);
        else // dispatches over input dimensions (X)
            return Dispatch(fn, X.dataType, O, X.channels, X.width, X.height);
    }

    /// <inheritdoc/>
    public override Tensor Resample2D(Tensor X, int[] size, bool bilinear)
    {
        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(size.Length, 2);

        var O = new TensorShape(X.batch, size[1], size[0], X.channels);

        var fn = new ComputeFunc(ComputeShaderContext.Reference, bilinear ? "ResampleBilinear2D" : "Resample2D", GetModelExecutionsReporter());

        SetTensor(fn, "X", X);

        return Dispatch(fn, X.dataType, O, O.channels, O.width, O.height);
    }

    /// <inheritdoc/>
    public override Tensor DepthToSpace(Tensor X, int[] blocksize, Layer.DepthToSpaceMode mode)
    {
        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(blocksize.Length, 2);

        var O = new TensorShape(X.batch, X.height * blocksize[1], X.width * blocksize[0], X.channels / (blocksize[0] * blocksize[1]));

        var fn = new ComputeFunc(ComputeShaderContext.Reference, "DepthToSpace_" + mode, GetModelExecutionsReporter());

        SetTensor(fn, "X", X);

        fn.shader.SetInts("_Pool", blocksize);

        return Dispatch(fn, X.dataType, O, O.channels, O.width, O.height);
    }

    /// <inheritdoc/>
    public override Tensor SpaceToDepth(Tensor X, int[] blocksize)
    {
        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(blocksize.Length, 2);

        var O = new TensorShape(X.batch, X.height / blocksize[1], X.width / blocksize[0], X.channels * (blocksize[0] * blocksize[1]));

        var fn = new ComputeFunc(ComputeShaderContext.Reference, "SpaceToDepth", GetModelExecutionsReporter());

        SetTensor(fn, "X", X);

        fn.shader.SetInts("_Pool", blocksize);

        return Dispatch(fn, X.dataType, O, O.channels, O.width, O.height);
    }

    /// <inheritdoc/>
    protected virtual Tensor Pool2D(string kernelName, Tensor X, int[] pool, int[] stride, int[] pad)
    {
        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(pool.Length, 2);
        Assert.AreEqual(stride.Length, 2);

        var O = X.shape.ApplyPool(pool, stride, pad);

        var fn = new ComputeFunc(ComputeShaderContext.Reference, kernelName, GetModelExecutionsReporter());

        SetTensor(fn, "X", X);
        fn.shader.SetInts("_Pool", pool);
        fn.shader.SetInts("_Stride", stride);
        fn.shader.SetInts("_Pad", pad);

        return Dispatch(fn, X.dataType, O, O.channels, O.width, O.height);
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
        Assert.IsTrue(X.shape.Is4D());
        var O = new TensorShape(X.batch, 1, 1, X.channels);

        var fn = new ComputeFunc(ComputeShaderContext.Reference, kernelName, GetModelExecutionsReporter());

        SetTensor(fn, "X", X);

        return Dispatch(fn, X.dataType, O, O.channels, 1, 1);
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
        Assert.IsTrue(X.shape.Is4D());
        var O = new TensorShape(X.batch, 2, 1, X.channels);

        var fn = new ComputeFunc(ComputeShaderContext.Reference, "GlobalAvgVariancePool2D", GetModelExecutionsReporter());

        SetTensor(fn, "X", X);

        return Dispatch(fn, X.dataType, O, O.channels, 1, 1);
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

        var O = X.shape.ApplyBorder(pad);

        var fn = new ComputeFunc(ComputeShaderContext.Reference, kernelName, GetModelExecutionsReporter());

        SetTensor(fn, "X", X);

        fn.shader.SetInts("_Pad", pad.Take(3).ToArray());

        if (kernelName == "Border2D")
        {
            // NOTE: negative "pad" variable will crop X tensor
            int croppedWidth = X.width - Math.Max(0, -pad[3]);
            int croppedHeight = X.height - Math.Max(0, -pad[4]);
            int croppedChannels = X.channels - Math.Max(0, -pad[5]);
            var croppedSize = new int[] { 0, 0, 0 };
            croppedSize[0] = croppedWidth;
            croppedSize[1] = croppedHeight;
            croppedSize[2] = croppedChannels;

            fn.shader.SetInts("_Pool", croppedSize);
            fn.shader.SetFloat("_Beta", constant);
        }

        return Dispatch(fn, X.dataType, O, O.channels, O.width, O.height);
    }


    /// <summary>
    /// Apply 3D padding
    /// </summary>
    /// <param name="X">input</param>
    /// <param name="pad">padding</param>
    /// <param name="kernelName">kernel name</param>
    /// <param name="constant">padding constant</param>
    /// <returns>output `Tensor`</returns>
    protected virtual Tensor ApplyPadding3D(Tensor X, int[] pad, string kernelName, float constant = 0.0f)
    {
        Assert.IsTrue(X.shape.IsNDHWC());
        Assert.AreEqual(pad.Length, 8);

        var O = X.shape.ApplyBorder(pad);

        var fn = new ComputeFunc(ComputeShaderContext.Reference, kernelName, GetModelExecutionsReporter());

        SetTensor(fn, "X", X);

        fn.shader.SetInts("_Pad", pad.Take(4).ToArray());

        if (kernelName == "Border3D")
        {
            // NOTE: negative "pad" variable will crop X tensor
            int croppedWidth = X.width - Math.Max(0, -pad[4]);
            int croppedHeight = X.height - Math.Max(0, -pad[5]);
            int croppedDepth = X.depth - Math.Max(0, -pad[6]);
            int croppedChannels = X.channels - Math.Max(0, -pad[7]);

            var croppedSize = new int[] { 0, 0, 0, 0 };
            croppedSize[0] = croppedWidth;
            croppedSize[1] = croppedHeight;
            croppedSize[2] = croppedDepth;
            croppedSize[3] = croppedChannels;

            fn.shader.SetInts("_Pool", croppedSize);
            fn.shader.SetFloat("_Beta", constant);
        }

        return Dispatch(fn, X.dataType, O, O.channels, O.width, O.height);
    }

    /// <inheritdoc/>
    public override Tensor Border2D(Tensor X, int[] pad, float constant)
    {
        return ApplyPadding(X, pad, "Border2D", constant);
    }

    /// <inheritdoc/>
    public override Tensor Border3D(Tensor X, int[] pad, float constant)
    {
        return ApplyPadding3D(X, pad, "Border3D", constant);
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
        Assert.AreEqual(X.channels, B.channels); Assert.AreEqual(X.channels, S.channels);
        Assert.AreEqual(B.length, B.channels); Assert.AreEqual(S.length, S.channels);

        var O = X.shape;
        var fn = new ComputeFunc(ComputeShaderContext.Reference, "ScaleBias", GetModelExecutionsReporter());

        SetTensor(fn, "X", X);
        SetTensor(fn, "W", S);
        SetTensor(fn, "B", B);

        return Dispatch(fn, X.dataType, O, O.channels, O.width, O.height);
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

        var Oshape = X.shape;
        var fn = new ComputeFunc(ComputeShaderContext.Reference, "InstanceNorm", GetModelExecutionsReporter());
        fn.shader.SetFloat("_Epsilon", epsilon);
        fn.shader.SetInt("_ActivationMode", (int)fusedActivation);

        SetTensor(fn, "X", X);
        SetTensor(fn, "W", S);
        SetTensor(fn, "B", B);

        var O = Dispatch(fn, X.dataType, Oshape, Oshape.channels, 1, 1);

        if (!IsFusedActivationSupported(fusedActivation))
            O = Activation(fusedActivation.ToString(), O);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor LRN(Tensor X, float alpha, float beta, float bias, int size)
    {
        var O = X.shape;
        var fn = new ComputeFunc(ComputeShaderContext.Reference, "LRN", GetModelExecutionsReporter());

        SetTensor(fn, "X", X);
        fn.shader.SetFloat("_Alpha", alpha);
        fn.shader.SetFloat("_Beta",  beta);
        fn.shader.SetFloat("_Epsilon",  bias);
        fn.shader.SetInt("_Axis", size);

        return Dispatch(fn, X.dataType, O, O.channels, O.width, O.height);
    }

    // @TODO: debug & fix
    /// <inheritdoc/>
    public override Tensor Dropout(Tensor X, float alpha)
    {
        Assert.IsTrue(alpha >= 0f && alpha <= 1f);

        var O = X.shape;
        var fn = new ComputeFunc(ComputeShaderContext.Reference, "Dropout", GetModelExecutionsReporter());

        SetTensor(fn, "X", X);

        fn.shader.SetFloat("_Alpha", alpha);

        using (var seedOverride = new Seed(ref m_DropoutSeed, 1337))
        {
            fn.shader.SetFloat("_Seed", UnityEngine.Random.value);
        }

        return Dispatch(fn, X.dataType, O, O.channels, O.width, O.height);
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
        var fn = new ComputeFunc(ComputeShaderContext.Reference, kernelName, GetModelExecutionsReporter());

        SetTensor(fn, "X", X);
        fn.shader.SetFloat("_Alpha", alpha);
        fn.shader.SetFloat("_Beta",  beta);

        return Dispatch(fn, X.dataType, O, O.channels, O.width, O.height);
    }

    /// <inheritdoc/>
    public override Tensor Relu(Tensor X)
    {
        return Activation("Relu", X);
    }

    /// <inheritdoc/>
    public override Tensor PRelu(Tensor X, Tensor S)
    {
        Assert.IsTrue((X.flatWidth == S.flatWidth) || (S.flatWidth == 1));

        var O = X.shape;
        var fn = new ComputeFunc(ComputeShaderContext.Reference, "PRelu", GetModelExecutionsReporter());

        SetTensor(fn, "X", X);
        SetTensor(fn, "W", S);

        return Dispatch(fn, X.dataType, O, O.channels, O.width, O.height);
    }

    /// <inheritdoc/>
    public override Tensor Softmax(Tensor X, int axis)
    {
        axis = X.shape.Axis(axis);
        var Oshape = X.shape;

        int reducedDim = X.shape[axis];
        var XShape = X.shape.ToArray();

        if (ComputeInfo.channelsOrder == ComputeInfo.ChannelsOrder.NCHW)
        {
            XShape[TensorShape.DataBatch + 1] = Oshape[TensorShape.C];
            for (int i = TensorShape.DataBatch + 1; i < TensorShape.C; i++)
                XShape[i + 1] = Oshape[i];

            if (axis == TensorShape.C)
                axis = TensorShape.DataBatch + 1;
            else if (axis > TensorShape.DataBatch)
                axis += 1;
        }

        int height = 1;
        for (var i = 0; i < axis; i++)
            height *= XShape[i];

        int width = 1;
        for (var i = axis + 1; i < X.shape.rank; i++)
            width *= XShape[i];

        var fn = new ComputeFunc(ComputeShaderContext.Reference, "Softmax", GetModelExecutionsReporter());

        var strides = new[] { height, reducedDim, width, 0, 0 };
        fn.shader.SetInts("_Stride", strides);

        SetTensor(fn, "X", X);

        var O =  Dispatch(fn, X.dataType, Oshape, height, width, 1);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor LogSoftmax(Tensor X, int axis)
    {
        axis = X.shape.Axis(axis);
        var Oshape = X.shape;

        int reducedDim = X.shape[axis];
        var XShape = X.shape.ToArray();

        if (ComputeInfo.channelsOrder == ComputeInfo.ChannelsOrder.NCHW)
        {
            XShape[TensorShape.DataBatch + 1] = Oshape[TensorShape.C];
            for (int i = TensorShape.DataBatch + 1; i < TensorShape.C; i++)
                XShape[i + 1] = Oshape[i];

            if (axis == TensorShape.C)
                axis = TensorShape.DataBatch + 1;
            else if (axis > TensorShape.DataBatch)
                axis += 1;
        }

        int height = 1;
        for (var i = 0; i < axis; i++)
            height *= XShape[i];

        int width = 1;
        for (var i = axis + 1; i < X.shape.rank; i++)
            width *= XShape[i];

        var fn = new ComputeFunc(ComputeShaderContext.Reference, "LogSoftmax", GetModelExecutionsReporter());

        var strides = new[] { height, reducedDim, width, 0, 0 };
        fn.shader.SetInts("_Stride", strides);

        SetTensor(fn, "X", X);

        var O =  Dispatch(fn, X.dataType, Oshape, height, width, 1);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Tanh(Tensor X)
    {
        return Activation("Tanh", X);
    }

    /// <inheritdoc/>
    public override Tensor Softplus(Tensor X)
    {
        return Activation("Softplus", X);
    }

    /// <inheritdoc/>
    public override Tensor Sigmoid(Tensor X)
    {
        return Activation("Sigmoid", X);
    }

    /// <inheritdoc/>
    public override Tensor HardSigmoid(Tensor X, float alpha, float beta)
    {
        return Activation("HardSigmoid", X, alpha, beta);
    }

    /// <inheritdoc/>
    public override Tensor Relu6(Tensor X)
    {
        return Activation("Relu6", X);
    }

    /// <inheritdoc/>
    public override Tensor Elu(Tensor X, float alpha)
    {
        return Activation("Elu", X, alpha);
    }

    /// <inheritdoc/>
    public override Tensor LeakyRelu(Tensor X, float alpha)
    {
        return Activation("LeakyRelu", X, alpha);
    }

    /// <inheritdoc/>
    public override Tensor Selu(Tensor X, float alpha, float gamma)
    {
        return Activation("Selu", X, alpha, gamma);
    }

    /// <inheritdoc/>
    public override Tensor Swish(Tensor X)
    {
        return Activation("Swish", X);
    }

    /// <inheritdoc/>
    public override Tensor Abs(Tensor X)
    {
        return Activation("Abs", X);
    }

    /// <inheritdoc/>
    public override Tensor Neg(Tensor X)
    {
        return Activation("Neg", X);
    }

    /// <inheritdoc/>
    public override Tensor Ceil(Tensor X)
    {
        return Activation("Ceil", X);
    }

    /// <inheritdoc/>
    public override Tensor Clip(Tensor X, float min, float max)
    {
        return Activation("Clip", X, min, max);
    }

    /// <inheritdoc/>
    public override Tensor Floor(Tensor X)
    {
        return Activation("Floor", X);
    }

    /// <inheritdoc/>
    public override Tensor Round(Tensor X)
    {
        return Activation("Round", X);
    }

    /// <inheritdoc/>
    public override Tensor Reciprocal(Tensor X)
    {
        return Activation("Reciprocal", X);
    }

    /// <inheritdoc/>
    public override Tensor Pow(Tensor X, float alpha)
    {
        return Activation("Pow", X, alpha);
    }

    /// <inheritdoc/>
    public override Tensor Exp(Tensor X)
    {
        return Activation("Exp", X);
    }

    /// <inheritdoc/>
    public override Tensor Log(Tensor X)
    {
        return Activation("Log", X);
    }

    /// <inheritdoc/>
    public override Tensor Sqrt(Tensor X)
    {
        return Activation("Sqrt", X);
    }

    /// <inheritdoc/>
    public override Tensor Acos(Tensor X)
    {
        return Activation("Acos", X);
    }

    /// <inheritdoc/>
    public override Tensor Acosh(Tensor X)
    {
        return Activation("Acosh", X);
    }

    /// <inheritdoc/>
    public override Tensor Asin(Tensor X)
    {
        return Activation("Asin", X);
    }

    /// <inheritdoc/>
    public override Tensor Asinh(Tensor X)
    {
        return Activation("Asinh", X);
    }

    /// <inheritdoc/>
    public override Tensor Atan(Tensor X)
    {
        return Activation("Atan", X);
    }

    /// <inheritdoc/>
    public override Tensor Atanh(Tensor X)
    {
        return Activation("Atanh", X);
    }

    /// <inheritdoc/>
    public override Tensor Cos(Tensor X)
    {
        return Activation("Cos", X);
    }

    /// <inheritdoc/>
    public override Tensor Cosh(Tensor X)
    {
        return Activation("Cosh", X);
    }

    /// <inheritdoc/>
    public override Tensor Sin(Tensor X)
    {
        return Activation("Sin", X);
    }

    /// <inheritdoc/>
    public override Tensor Sinh(Tensor X)
    {
        return Activation("Sinh", X);
    }

    /// <inheritdoc/>
    public override Tensor Tan(Tensor X)
    {
        return Activation("Tan", X);
    }

    /// <inheritdoc/>
    public override Tensor Erf(Tensor X)
    {
        return Activation("Erf", X);
    }

    /// <inheritdoc/>
    public override Tensor ConstantOfShape(TensorShape X, DataType type, float value = 0.0f)
    {
        var fn = new ComputeFunc(ComputeShaderContext.Reference, "ConstantOfShape", GetModelExecutionsReporter());
        fn.shader.SetFloat("_Alpha", value);

        return Dispatch(fn, type, X, X.channels, X.width, X.height);
    }

    /// <inheritdoc/>
    public override Tensor Expand(Tensor X, TensorShape newShape)
    {
        Assert.IsTrue(newShape.sequenceLength == X.sequenceLength || X.sequenceLength == 1);
        Assert.IsTrue(newShape.numberOfDirections == X.numberOfDirections || X.numberOfDirections == 1);
        Assert.IsTrue(newShape.batch == X.batch || X.batch == 1);
        Assert.IsTrue(newShape.extraDimension == X.extraDimension || X.extraDimension == 1);
        Assert.IsTrue(newShape.depth == X.depth || X.depth == 1);
        Assert.IsTrue(newShape.height == X.height || X.height == 1);
        Assert.IsTrue(newShape.width == X.width || X.width == 1);
        Assert.IsTrue(newShape.channels == X.channels || X.channels == 1);

        X = GetTensorInCurrentMemoryLayoutHelper(X);

        var fn = new ComputeFunc(ComputeShaderContext.Reference, "Expand", GetModelExecutionsReporter());
        SetTensor(fn, "X", X);

        return Dispatch(fn, X.dataType, newShape, newShape.channels, newShape.width, newShape.height);
    }

    internal static Tensor[] s_ElementwiseBroadcastTensors = new Tensor[2];

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

        var fn = new ComputeFunc(ComputeShaderContext.Reference, kernelName, GetModelExecutionsReporter());
        bool isFirstDispatch = true;
        for (int t = 1; t < tensors.Length; ++t)
        {
            var B = tensors[t];

            // B and X can be constants, in that cases the internal layout does not match ComputeInfo.channelsOrder and will allways be NHWC
            // => permute them if there is a layout mismatch
            X = GetTensorInCurrentMemoryLayoutHelper(X);
            B = GetTensorInCurrentMemoryLayoutHelper(B);

            SetTensor(fn, "X", X);
            SetTensor(fn, "B", B);
            fn.shader.SetFloat("_Alpha", 1.0f/(float)tensors.Length);
            fn.shader.SetInt("_IsFirstDispatch", isFirstDispatch ? 1 : 0);

            X = Dispatch(fn, X.dataType, O, O.channels, O.width, O.height);
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

    internal static int[] s_ReducePermute = new int[8];

    internal static void FillReducePermute(int axis)
    {
        for (var idx = 0; idx < s_ReducePermute.Length; idx++)
            s_ReducePermute[idx] = idx;
        s_ReducePermute[7] = axis;
        s_ReducePermute[axis] = 7;
    }

    /// <summary>
    /// Reduce with specified kernel
    /// </summary>
    /// <param name="kernelName">kernel name</param>
    /// <param name="X">input</param>
    /// <param name="axis">axis</param>
    /// <returns>output `Tensor`</returns>
    internal static readonly Dictionary<Layer.Type, string> s_ReduceRefKernelNames = new Dictionary<Layer.Type, string> {
        {Layer.Type.ReduceMax, "ReduceMax"}, {Layer.Type.ReduceMean, "ReduceMean"},
        {Layer.Type.ReduceMin, "ReduceMin"}, {Layer.Type.ReduceProd, "ReduceProd"},
        {Layer.Type.ReduceSum, "ReduceSum"}, {Layer.Type.ArgMax, "ArgMax"},
        {Layer.Type.ArgMin, "ArgMin"}
    };

    private Tensor ReduceHelper(Layer.Type kernelName, Tensor X, int axis)
    {
        axis = X.shape.Axis(axis);

        bool needTranpose = axis != TensorShape.C;
        FillReducePermute(axis);

        if (needTranpose)
            X = Transpose(X, s_ReducePermute);

        var oShape = X.shape.Reduce(TensorShape.C);
        Assert.AreEqual(oShape.channels, 1);

        var fn = new ComputeFunc(ComputeShaderContext.Reference, s_ReduceRefKernelNames[kernelName], GetModelExecutionsReporter());
        SetTensor(fn, "X", X);

        var O = Dispatch(fn, X.dataType, oShape, oShape.width, oShape.height, 1);

        if (needTranpose)
            O = Transpose(O, s_ReducePermute);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor ArgMax(Tensor X, int axis)
    {
        return ReduceHelper(Layer.Type.ArgMax, X, axis);
    }

    /// <inheritdoc/>
    public override Tensor ArgMin(Tensor X, int axis)
    {
        return ReduceHelper(Layer.Type.ArgMin, X, axis);
    }

    /// <inheritdoc/>
    public override Tensor ReduceMin(Tensor X, int axis)
    {
        return ReduceHelper(Layer.Type.ReduceMin, X, axis);
    }

    /// <inheritdoc/>
    public override Tensor ReduceMax(Tensor X, int axis)
    {
        return ReduceHelper(Layer.Type.ReduceMax, X, axis);
    }

    /// <inheritdoc/>
    public override Tensor ReduceSum(Tensor X, int axis)
    {
        return ReduceHelper(Layer.Type.ReduceSum, X, axis);
    }

    /// <inheritdoc/>
    public override Tensor ReduceMean(Tensor X, int axis)
    {
        return ReduceHelper(Layer.Type.ReduceMean, X, axis);
    }

    /// <inheritdoc/>
    public override Tensor ReduceProd(Tensor X, int axis)
    {
        return ReduceHelper(Layer.Type.ReduceProd, X, axis);
    }

    /// <inheritdoc/>
    public override Tensor Greater(Tensor A, Tensor B)
    {
        s_ElementwiseBroadcastTensors[0] = A;
        s_ElementwiseBroadcastTensors[1] = B;
        return ElementwiseWithBroadcast("BroadcastGreater", s_ElementwiseBroadcastTensors);
    }

    /// <inheritdoc/>
    public override Tensor GreaterEqual(Tensor A, Tensor B)
    {
        s_ElementwiseBroadcastTensors[0] = A;
        s_ElementwiseBroadcastTensors[1] = B;
        return ElementwiseWithBroadcast("BroadcastGreaterEqual", s_ElementwiseBroadcastTensors);
    }

    /// <inheritdoc/>
    public override Tensor Less(Tensor A, Tensor B)
    {
        s_ElementwiseBroadcastTensors[0] = A;
        s_ElementwiseBroadcastTensors[1] = B;
        return ElementwiseWithBroadcast("BroadcastLess", s_ElementwiseBroadcastTensors);
    }

    /// <inheritdoc/>
    public override Tensor LessEqual(Tensor A, Tensor B)
    {
        s_ElementwiseBroadcastTensors[0] = A;
        s_ElementwiseBroadcastTensors[1] = B;
        return ElementwiseWithBroadcast("BroadcastLessEqual", s_ElementwiseBroadcastTensors);
    }

    /// <inheritdoc/>
    public override Tensor Equal(Tensor A, Tensor B)
    {
        s_ElementwiseBroadcastTensors[0] = A;
        s_ElementwiseBroadcastTensors[1] = B;
        return ElementwiseWithBroadcast("BroadcastEqual", s_ElementwiseBroadcastTensors);
    }

    /// <inheritdoc/>
    public override Tensor LogicalOr(Tensor A, Tensor B)
    {
        s_ElementwiseBroadcastTensors[0] = A;
        s_ElementwiseBroadcastTensors[1] = B;
        return ElementwiseWithBroadcast("BroadcastLogicalOr", s_ElementwiseBroadcastTensors);
    }

    /// <inheritdoc/>
    public override Tensor LogicalAnd(Tensor A, Tensor B)
    {
        s_ElementwiseBroadcastTensors[0] = A;
        s_ElementwiseBroadcastTensors[1] = B;
        return ElementwiseWithBroadcast("BroadcastLogicalAnd", s_ElementwiseBroadcastTensors);
    }

    /// <inheritdoc/>
    public override Tensor LogicalXor(Tensor A, Tensor B)
    {
        s_ElementwiseBroadcastTensors[0] = A;
        s_ElementwiseBroadcastTensors[1] = B;
        return ElementwiseWithBroadcast("BroadcastLogicalXor", s_ElementwiseBroadcastTensors);
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
        var fn = new ComputeFunc(ComputeShaderContext.Reference, "BroadcastWhere", GetModelExecutionsReporter());

        var O = TensorExtensions.MaxShape(new[] { C, A, B });

        SetTensor(fn, "X", C);
        SetTensor(fn, "W", A);
        SetTensor(fn, "K", B);

        return Dispatch(fn, C.dataType, O, O.channels, O.width, O.height);
    }

    /// <inheritdoc/>
    public override Tensor OneHot(Tensor X, int depth, float onValue, float offValue, int inputRank=-1)
    {
        if (inputRank == -1)
            inputRank = X.dimensions;

        if (inputRank >= 4)
            throw new NotImplementedException();

        TensorShape O = new TensorShape();
        if (inputRank == 1)
            O = new TensorShape(X.flatHeight, depth);
        else if (inputRank == 2)
            O = new TensorShape(X.flatHeight, 1, depth, X.channels);
        else
            O = new TensorShape(X.batch, X.width, depth, X.channels);

        var fn = new ComputeFunc(ComputeShaderContext.Reference, "OneHot", GetModelExecutionsReporter());

        SetTensor(fn, "X", X);
        fn.shader.SetFloat("_Alpha", onValue);
        fn.shader.SetFloat("_Beta", offValue);
        fn.shader.SetInt("_Axis", depth);
        fn.shader.SetInts("_Pad", new int[] { inputRank, 0, 0, 0 });

        return Dispatch(fn, X.dataType, O, X.width, depth, X.channels);
    }

    /// <inheritdoc/>
    public override Tensor RoiAlign(Tensor X, Tensor Rois, Tensor Indices, int outputHeight, int outputWidth, int samplingRatio, float spatialScale)
    {
        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(Rois.flatHeight, Indices.batch);
        Assert.AreEqual(Rois.flatWidth, 4);

        TensorShape O = new TensorShape(Rois.flatHeight, outputHeight, outputWidth, X.channels);
        var fn = new ComputeFunc(ComputeShaderContext.Reference, "RoiAlign", GetModelExecutionsReporter());

        SetTensor(fn, "X", X);
        SetTensor(fn, "K", Rois);
        SetTensor(fn, "B", Indices);

        fn.shader.SetFloat("_Alpha", spatialScale);
        fn.shader.SetInt("_Axis", samplingRatio);

        return Dispatch(fn, X.dataType, O, outputHeight, outputWidth, X.channels);
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

        var O = NewTensor(X.dataType, newShape, AllocScope.LayerOutput, "O");

        if (X.shape.Is4D() && newShape.Is4D())
        {
            var fn = new ComputeFunc(ComputeShaderContext.Reference, "ReshapeFromNHWCModel_NCHW", GetModelExecutionsReporter());
            SetTensor(fn, "X", X);
            SetTensor(fn, "O", O);
            fn.Dispatch( O.width, O.height, O.channels);
        }
        else
        {
            var fn = new ComputeFunc(ComputeShaderContext.Reference, "Reshape8DFromChannelFirstModel_NCHW", GetModelExecutionsReporter());
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
        bool isNHWCCopy = X.shape.Is4D() && newShape.Is4D();

        // NOTE: "Copy" kernel copies tensor data while preserving the shape
        // However here in CopyAndReshape we want to both copy and change the shape,
        // To be able to piggyback "Copy" kernel we specify new shape when allocating destination tensor,
        // but use shape identical to source when copying.
        var O = NewTensor(X.dataType, newShape, AllocScope.LayerOutput, "O");
        var fn = new ComputeFunc(ComputeShaderContext.Reference, isNHWCCopy?"Copy":"Copy8D", GetModelExecutionsReporter());
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
        if (X.shape == newShape || ComputeInfo.channelsOrder == ComputeInfo.ChannelsOrder.NHWC)
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
        // TODO: reshape when possible
        Assert.IsTrue(X.dimensions <= 2);
        var O = new TensorShape(X.flatWidth, X.flatHeight);

        var fn = new ComputeFunc(ComputeShaderContext.Reference, "Transpose2D", GetModelExecutionsReporter());
        SetTensor(fn, "X", X);
        return Dispatch(fn, X.dataType, O, O.flatWidth, O.flatHeight, 1);
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

    private Tensor Transpose8DHelper(Tensor X, int[] permutations)
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

        var fn = new ComputeFunc(ComputeShaderContext.Reference, "Transpose8D", GetModelExecutionsReporter());
        SetTensor(fn, "X", X);
        fn.shader.SetInts("_Pad", d0_3);
        fn.shader.SetInts("_Pool", d4_7);
        fn.shader.SetInts("_Stride", outStride0_3);
        fn.shader.SetInts("_ChannelWriteMask", outStride4_7);

        if (ComputeInfo.channelsOrder == ComputeInfo.ChannelsOrder.NCHW)
            return Dispatch(fn, X.dataType, Oshape, X.width, X.height, X.depth);
        else
            return Dispatch(fn, X.dataType, Oshape, X.channels, X.width, X.height);

    }

    /// <inheritdoc/>
    public override Tensor Transpose(Tensor X, int[] permutations)
    {
        if (!X.shape.Is4D() || permutations.Length != 4)
            return Transpose8DHelper(X, permutations);

        Assert.AreEqual(permutations.Length, 4);

        X = GetTensorInCurrentMemoryLayoutHelper(X);
        var O = X.shape.Permute(permutations);

        var fn = new ComputeFunc(ComputeShaderContext.Reference, "Transpose", GetModelExecutionsReporter());
        SetTensor(fn, "X", X);
        fn.shader.SetInts("_Pool", permutations);
        return Dispatch(fn, X.dataType, O, X.channels, X.width, X.height);
    }

    internal Tensor GetTensorInCurrentMemoryLayoutHelper(Tensor tensor)
    {
        //Return a tensor in the current memory layout from ComputeInfo.channelsOrder.
        //Noop in the general case it will transpose constant tensor when ComputeInfo.channelsOrder == NCHW
        //as those tensor are always in channel last layout.
        //This is needed for kernel that can accept both input and constant tensor in the same argument.
        if (ComputeInfo.channelsOrder == ComputeInfo.ChannelsOrder.NCHW &&
            Pin(tensor).channelsOrder == ComputeInfo.ChannelsOrder.NHWC)
            return TransposeToChannelFirstHelper(tensor);
        else
            return tensor;
    }

    internal virtual Tensor TransposeToChannelFirstHelper(Tensor X)
    {
        var O = X.shape;
        var fn = new ComputeFunc(ComputeShaderContext.Reference, "TransposeToChannelFirst", GetModelExecutionsReporter());
        SetTensor(fn, "X", X);
        return Dispatch(fn, X.dataType, O, X.channels, X.width, X.height);
    }

    internal static int[] s_ConcatOffsets = new int[4];
    /// <inheritdoc/>
    public override Tensor Concat(Tensor[] tensors, int axis)
    {
        if (axis != TensorShape.C && axis != -1)
            return base.Concat(tensors, axis);

        if (!TensorExtensions.AreAllTensorsConvertibleTo4D(tensors) || !TensorExtensions.Is8DAxisConvertibleTo4D(axis))
            return base.Concat(tensors, axis);

        var fn = new ComputeFunc(ComputeShaderContext.Reference, "Copy", GetModelExecutionsReporter());

        var dataType = tensors.Length > 0 ? tensors[0].dataType : DataType.Float;
        var O = NewTensor(dataType, TensorExtensions.Concat(tensors, axis), AllocScope.LayerOutput);

        var offsets = s_ConcatOffsets;
        Array.Clear(offsets, 0, offsets.Length);
        axis = O.shape.Axis(axis);
        var axisNHWC = TensorExtensions.Convert8DAxisTo4D(axis);

        foreach (var inputTensor in tensors)
        {
            // input can be constants, in that cases the internal layout does not match ComputeInfo.channelsOrder and will allways be NHWC
            // => permute if there is a layout mismatch
            var X = GetTensorInCurrentMemoryLayoutHelper(inputTensor);

            SetTensor(fn, "X", X);
            SetTensor(fn, "O", O);

            fn.shader.SetInts("_Pad", offsets);

            fn.Dispatch(X.channels, X.width, X.height);

            offsets[axisNHWC] += X.shape[axis];
        }

        return O;
    }

    private void Set8DParamsForShader(int[] srcValues, int[] firstSplit, int[] secondSplit)
    {
        Assert.IsTrue(srcValues.Length == 8);
        Assert.IsTrue(firstSplit.Length == 4);
        Assert.IsTrue(secondSplit.Length == 4);
        firstSplit[0] =   srcValues[TensorShape.DataBatch];
        firstSplit[1] =   srcValues[TensorShape.H];
        firstSplit[2] =   srcValues[TensorShape.W];
        firstSplit[3] =   srcValues[TensorShape.C];
        secondSplit[0] = srcValues[TensorShape.SequenceLength];
        secondSplit[1] = srcValues[TensorShape.NumberOfDirections];
        secondSplit[2] = srcValues[TensorShape.DataFeature3];
        secondSplit[3] = srcValues[TensorShape.D];
    }

    private unsafe void Set8DParamsForShader(int* srcValues, int[] firstSplit, int[] secondSplit)
    {
        Assert.IsTrue(firstSplit.Length == 4);
        Assert.IsTrue(secondSplit.Length == 4);
        firstSplit[0] =   srcValues[TensorShape.DataBatch];
        firstSplit[1] =   srcValues[TensorShape.H];
        firstSplit[2] =   srcValues[TensorShape.W];
        firstSplit[3] =   srcValues[TensorShape.C];
        secondSplit[0] = srcValues[TensorShape.SequenceLength];
        secondSplit[1] = srcValues[TensorShape.NumberOfDirections];
        secondSplit[2] = srcValues[TensorShape.DataFeature3];
        secondSplit[3] = srcValues[TensorShape.D];
    }

    static private int[] s_StridedSliceStart = new int[4];
    static private int[] s_StridedSliceStart8D = new int[4];
    static private int[] s_StridedSliceStride = new int[4];
    static private int[] s_StridedSliceStride8D = new int[4];
    /// <inheritdoc/>
    public override Tensor StridedSlice(Tensor X, int[] starts4Dor8D, int[] ends4Dor8D, int[] strides4Dor8D)
    {
        X = GetTensorInCurrentMemoryLayoutHelper(X);

        unsafe
        {
            int* starts = stackalloc int[TensorShape.MaxRank];
            int* ends = stackalloc int[TensorShape.MaxRank];
            int* strides = stackalloc int[TensorShape.MaxRank];
            TensorExtensions.Get8DParametersNoAlloc(X.shape, starts4Dor8D, starts, 0);
            TensorExtensions.Get8DParametersNoAlloc(X.shape, ends4Dor8D, ends, 1);
            TensorExtensions.Get8DParametersNoAlloc(X.shape, strides4Dor8D, strides, 1);

            var O = X.shape.ApplyStridedSlice8DUnsafeNoAlloc(starts, ends, strides);

            for (int i = 0; i < TensorShape.MaxRank; ++i)
                starts[i] = Math.Min(TensorExtensions.WrapIndex(starts[i], X.shape[i]), X.shape[i] - 1);

            Set8DParamsForShader(strides, s_StridedSliceStride, s_StridedSliceStride8D);
            Set8DParamsForShader(starts, s_StridedSliceStart, s_StridedSliceStart8D);

            var fn = new ComputeFunc(ComputeShaderContext.Reference, "StridedSlice", GetModelExecutionsReporter());
            SetTensor(fn, "X", X);
            fn.shader.SetInts("_Stride4D", s_StridedSliceStride);
            fn.shader.SetInts("_Stride8D", s_StridedSliceStride8D);
            fn.shader.SetInts("_Pad", s_StridedSliceStart);
            fn.shader.SetInts("_Pool", s_StridedSliceStart8D);

            return Dispatch(fn, X.dataType, O, O.channels, O.width, O.height);
        }
    }

    /// <inheritdoc/>
    public override Tensor Tile(Tensor X, int[] repeats)
    {
        X = GetTensorInCurrentMemoryLayoutHelper(X);

        var O = X.shape.Scale(repeats);
        var fn = new ComputeFunc(ComputeShaderContext.Reference, "Tile", GetModelExecutionsReporter());
        SetTensor(fn, "X", X);

        return Dispatch(fn, X.dataType, O, O.channels, O.width, O.height);
    }

    /// <inheritdoc/>
    public override Tensor Gather(Tensor[] tensors, int axis)
    {
        Tensor X = tensors[0];
        Tensor indices = tensors[1];

        var outputShape = X.shape;
        outputShape[axis] = indices.length;

        var fn = new ComputeFunc(ComputeShaderContext.Reference, "Gather", GetModelExecutionsReporter());
        SetTensor(fn, "X", X);
        SetTensor(fn, "K", indices);
        fn.shader.SetInt("_Axis", axis);

        return Dispatch(fn, X.dataType, outputShape, outputShape.channels, outputShape.width, outputShape.height);
    }

    /// <inheritdoc/>
    public override Tensor ScatterND(Tensor X, Tensor indices, Tensor updates, Layer.ScatterNDReductionMode reduction)
    {
        // only support for scattering on C for now
        Assert.IsTrue(indices.batch == X.batch);
        Assert.IsTrue(updates.width == X.width && updates.height == X.height);
        var outputShape = X.shape;

        var fn = new ComputeFunc(ComputeShaderContext.Reference, "ScatterND", GetModelExecutionsReporter());
        SetTensor(fn, "X", X);
        SetTensor(fn, "K", indices);
        SetTensor(fn, "W", updates);

        fn.shader.SetInt("_Axis", (int)reduction);

        return Dispatch(fn, X.dataType, outputShape, outputShape.channels, outputShape.width, outputShape.height);
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

    /// <inheritdoc/>
    public override Tensor PrepareNoAlloc(Tensor X)
    {
        Pin(X, uploadCache: false);
        return X;
    }
}

internal struct ComputeFunc
{
    // dispatch dimension limitation coming from D3D11
    public static uint SafeDispatchLimit = 65535;

    public struct TensorDecl
    {
        public int ShapeId { get; }
        public int ShapeId8D { get; }
        public int InfoId { get; }

        public TensorDecl(int shapeId, int shapeId8D, int infoId)
        {
            ShapeId = shapeId;
            ShapeId8D = shapeId8D;
            InfoId = infoId;
        }
    }

    private readonly IModelExecutionsReporter executionReporter;
    readonly public ComputeShader shader;
    readonly public string kernelName;
    readonly public ComputeShaderContext computeShaderContext;
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
        var shapeId8D = Shader.PropertyToID(s_StringCache.Lookup(name, "declShape8D"));
        var infoId = Shader.PropertyToID(s_StringCache.Lookup(name, "declInfo"));
        return new TensorDecl(shapeId, shapeId8D, infoId);
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
    public ComputeFunc(ComputeShaderContext ctx, string kn, IModelExecutionsReporter reporter)
    {
        executionReporter = reporter;
        string kernelNameWithChannelsOrder = s_StringCache.Lookup(kn,
                            (ComputeInfo.channelsOrder == ComputeInfo.ChannelsOrder.NHWC) ? "_NHWC" : "_NCHW");

        var s = ComputeShaderSingleton.Instance.FindComputeShader(ctx, kernelNameWithChannelsOrder) ??
                                ComputeShaderSingleton.Instance.FindComputeShader(ctx, kn);

        if (s != null && (s.HasKernel(kernelNameWithChannelsOrder) || s.HasKernel(kn)))
        {
            shader = s;
            kernelName = s.HasKernel(kernelNameWithChannelsOrder)?kernelNameWithChannelsOrder:kn;
            computeShaderContext = ctx;
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
    static private int[] s_tTensorDeclScratchpadShape8D = new int[4];
    static private int[] s_tTensorDeclScratchpadInfo = new int[2];
    public void SetTensorDecl(ComputeFunc.TensorDecl tensorDecl, TensorShape shape, Int64 dataOffset)
    {
        s_tTensorDeclScratchpadShape[0] = shape.batch;
        s_tTensorDeclScratchpadShape[1] = shape.height;
        s_tTensorDeclScratchpadShape[2] = shape.width;
        s_tTensorDeclScratchpadShape[3] = shape.channels;
        s_tTensorDeclScratchpadShape8D[0] = shape.sequenceLength;
        s_tTensorDeclScratchpadShape8D[1] = shape.numberOfDirections;
        s_tTensorDeclScratchpadShape8D[2] = shape.extraDimension;
        s_tTensorDeclScratchpadShape8D[3] = shape.depth;
        s_tTensorDeclScratchpadInfo[0] = (int)dataOffset;
        s_tTensorDeclScratchpadInfo[1] = shape.length;
        shader.SetInts(tensorDecl.ShapeId8D, s_tTensorDeclScratchpadShape8D);
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

#if ENABLE_BARRACUDA_STATS
        if (executionReporter != null)
        {
            var dispatchInfo = DispatchInfo.CreateFromComputeFunc(this, workItemsX, workItemsY, workItemsZ);
            executionReporter.AddLayerDispatch(dispatchInfo);
        }
#endif //ENABLE_BARRACUDA_STATS

        shader.Dispatch(kernelIndex, x, y, z);

        ComputeDebugUtils.VerifyDispatch(kernelName);

        Profiler.EndSample();
    }

    // ---------------------------------------------------------------------------------

    static public int IntDivCeil(int v, int div)
    {
        return (v + div - 1) / div;
    }
}

} // namespace Unity.Barracuda
