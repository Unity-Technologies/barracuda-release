using UnityEngine;
using UnityEngine.Experimental.Rendering; // AsyncGPUReadback
using UnityEngine.Assertions;
using System;

namespace Unity.Barracuda
{

/// <summary>
/// Texture based `Tensor` storage
/// </summary>
public class TextureAsTensorData : UniqueResourceId, ITensorData
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

    /// <summary>
    /// multiplies scales texture value
    /// </summary>
    public Vector4 scale
    {
        get { return m_scale; }
    }

    /// <summary>
    /// subtracts bias texture value
    /// </summary>
    public Vector4 bias
    {
        get { return m_bias; }
    }


    private TensorShape m_Shape;
    private Texture[] m_Textures;
    private int m_InterpretPixelAsChannels;
    private InterpretDepthAs m_InterpretDepthAs;
    private InterpretColorAs m_InterpretColorAs;
    private Flip m_Flip;
    private Vector4 m_scale, m_bias;


    /// <summary>
    /// Shape
    /// </summary>
    public TensorShape shape
    {
        get { return m_Shape; }
    }

    /// <summary>
    /// Backing textures
    /// </summary>
    public Texture[] textures
    {
        get { return m_Textures; }
    }

    /// <summary>
    /// Interpret pixel as channels
    /// </summary>
    public int interpretPixelAsChannels
    {
        get { return m_InterpretPixelAsChannels; }
    }

    /// <summary>
    /// Interpret depth as
    /// </summary>
    public InterpretDepthAs interpretDepthAs
    {
        get { return m_InterpretDepthAs; }
    }

    /// <summary>
    /// Interpret color as
    /// </summary>
    public InterpretColorAs interpretColorAs
    {
        get { return m_InterpretColorAs; }
    }

    /// <summary>
    /// Flip flag
    /// </summary>
    public Flip flip
    {
        get { return m_Flip; }
    }

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
        Flip flip = Flip.Y, InterpretDepthAs depthAs = InterpretDepthAs.Batch,
        InterpretColorAs colorAs = InterpretColorAs.AverageMultipleChannels) :
        this(textures, flip, depthAs, colorAs, Vector4.one, Vector4.zero, interpretPixelAsChannels)
    {
    }

    /// <summary>
    /// Create `TextureAsTensorData` from supplied `textures`
    /// </summary>
    /// <param name="textures">backing textures</param>
    /// <param name="interpretPixelAsChannels">interpret pixel as channels</param>
    /// <param name="flip">flip</param>
    /// <param name="depthAs">depth as</param>
    /// <param name="colorAs">color as</param>
    /// <param name="scale">multiplies `scale` to texture values</param>
    /// <param name="bias">substracts `bias` from texture values</param>
    /// <exception cref="ArgumentException">thrown if textures array is empty or texture types are different</exception>
    /// <exception cref="InvalidOperationException">thrown if unsupported texture type is supplied</exception>
    public TextureAsTensorData(Texture[] textures,
        Flip flip, InterpretDepthAs depthAs, InterpretColorAs colorAs, Vector4 scale, Vector4 bias,
        int interpretPixelAsChannels)
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

        m_scale = scale;
        m_bias = bias;

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
        Flip flip = Flip.Y, InterpretDepthAs depthAs = InterpretDepthAs.Batch,
        InterpretColorAs colorAs = InterpretColorAs.AverageMultipleChannels)
        : this(new[] { texture }, interpretPixelAsChannels, flip, depthAs, colorAs)
    {
    }

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

    private static void FillCacheFromTexture(float[] output, Texture tex,
        int batchOffset, int channelOffset, int[] channelWriteMask, int[] channelReadMap,
        bool flipY, Vector4 scale4, Vector4 bias4, TensorShape texDataShape)
    {
        var tex2D = tex as Texture2D;
        var texArr = tex as Texture2DArray;
        var tex3D = tex as Texture3D;
        var rt = tex as RenderTexture;

        Color[] colors = null;
        var texDepth = 1;
        if (tex2D)
        {
            colors = tex2D.GetPixels(0);
            texDepth = 1;
        }
        else if (texArr)
        {
            colors = texArr.GetPixels(0, 0);
            texDepth = texArr.depth;
        }
        else if (tex3D)
        {
            colors = tex3D.GetPixels(0);
            texDepth = tex3D.depth;
        }
        else if (rt)
        {
            var currentRT = RenderTexture.active;
            RenderTexture.active = rt;
            Texture2D tmpTexture = new Texture2D(rt.width, rt.height, tex.graphicsFormat, TextureCreationFlags.None);
            tmpTexture.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
            tmpTexture.Apply();
            colors = tmpTexture.GetPixels(0);
            RenderTexture.active = currentRT;
            texDepth = rt.volumeDepth;
            if (rt.format == RenderTextureFormat.RHalf)
                Debug.LogError(
                    "Texture to Tensor does not support RHalf format for source rendertarget when Compute shader are not available on platform.");
        }

        if (texDepth != 1)
        {
            Debug.LogError(
                "Texture to Tensor only support texture resource with one slice when Compute shader are not available on platform!");
        }

        Assert.IsNotNull(colors);

        for (int x = 0; x < texDataShape.width; ++x)
        for (int yTex = 0; yTex < texDataShape.height; ++yTex)
        {
            int c = channelOffset;
            int y = flipY ? texDataShape.height - yTex - 1 : yTex;

            var pixelIndex = yTex * texDataShape.width + x;
            Vector4 v = colors[pixelIndex];
            bool specialCaseWhenChannelMaskIsEmptyStoresAverage = true;
            for (int i = 0; i < 4; ++i)
            {
                if (channelWriteMask[i] == 1)
                {
                    int readFrom = channelReadMap[i];
                    float value = i < 3 ? 0 : 1; // default values for channels R,G,B=0 and A=1
                    float scale = 1.0f;
                    float bias = 0.0f;
                    if (readFrom >= 0)
                    {
                        value = v[readFrom];
                        scale = scale4[readFrom];
                        bias = bias4[readFrom];
                    }

                    output[texDataShape.Index(batchOffset, y, x, c)] = scale * value + bias;
                    specialCaseWhenChannelMaskIsEmptyStoresAverage = false;
                    c += 1;
                }
            }

            if (specialCaseWhenChannelMaskIsEmptyStoresAverage)
            {
                v = Vector4.Scale(v, scale4) + bias4;
                float avg = (v.x + v.y + v.z) / 3.0f;
                output[texDataShape.Index(batchOffset, y, x, c)] = avg;
            }
        }
    }

    // TODO@: expose now that Download necesarrily goes via the gpu (compute/pixel) ?
    private float[] TextureToTensorDataCache(TensorShape shape)
    {
        float[] tensorDataCache = new float[shape.length];
        bool flipY = flip == Flip.Y;

        int batchOffset = 0;
        int channelOffset = 0;
        foreach (var tex in textures)
        {
            var channelWriteMask = TextureFormatUtils.FormatToChannelMask(tex, interpretPixelAsChannels);
            var channelReadMap = TextureFormatUtils.FormatToChannelReadMap(tex, interpretPixelAsChannels);

            FillCacheFromTexture(tensorDataCache, tex, batchOffset, channelOffset, channelWriteMask, channelReadMap,
                flipY, scale, bias, shape);

            if (interpretDepthAs == InterpretDepthAs.Batch)
                batchOffset += 1;
            else if (interpretDepthAs == InterpretDepthAs.Channels)
                channelOffset += interpretPixelAsChannels;
        }

        return tensorDataCache;
    }

    /// <inheritdoc/>
    public virtual float[] Download(TensorShape shape)
    {
        if (ComputeInfo.supportsCompute && SystemInfo.supportsComputeShaders)
        {
            var gpuBackend = new ReferenceComputeOps(null);
            // @TODO: cache compute buffer
            using (var computeTensorData =
                gpuBackend.TextureToTensorData(this, "__internalDownloadTextureToTensorData"))
            {
                return computeTensorData.Download(shape);
            }
        }
        else
        {
            var gpuBackend = new PixelShaderOps(null);
            using (var pixelShaderTensorData =
                gpuBackend.TextureToTensorData(this, "__internalDownloadTextureToTensorData"))
            {
                return pixelShaderTensorData.Download(shape);
            }
        }
    }

    /// <inheritdoc/>
    public virtual BarracudaArray SharedAccess(out int offset)
    {
        offset = 0;
        return new BarracudaArrayFromManagedArray(Download(shape)); //TODO fp16
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
    /// Dispose
    /// </summary>
    public virtual void Dispose()
    {
    }
}

} //namespace Barracuda
