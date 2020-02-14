using UnityEngine.Assertions;
using System;

namespace Barracuda {

/// <summary>
/// TensorShape are immutable representation of a Tensor dimensions and rank.
/// At the moment a TensorShape is always of rank 4 and channels last ie B,H,W,C.
/// However an axis can be a size 1. For example a tensor without spatial information will be B,1,1,C
/// </summary>
[Serializable]
public struct TensorShape
{
    /// <summary>
    /// Return the number of batch.
    /// </summary>
    public readonly int batch;
    /// <summary>
    /// Return the spatial height.
    /// </summary>
    public readonly int height;
    /// <summary>
    /// Return the spatial width.
    /// </summary>
    public readonly int width;
    /// <summary>
    /// Return the number of channels.
    /// </summary>
    public readonly int channels;

    #region Constructors
    /// <summary>
    /// Create a TensorShape of shape B,H,W,C.
    /// </summary>
    public TensorShape(int b, int h, int w, int ch)
    {
        batch = b > 0 ? b : 1;
        height = h > 0 ? h : 1;
        width = w > 0 ? w : 1;
        channels = ch > 0 ? ch : 1;
    }
    /// <summary>
    /// Create a TensorShape of shape B,1,1,C.
    /// </summary>
    public TensorShape(int b, int ch)
    {
        batch = b > 0 ? b : 1;
        height = 1;
        width = 1;
        channels = ch > 0 ? ch : 1;
    }
    /// <summary>
    /// Create a TensorShape of arbitrary `shape`.
    /// Currently `shape` can have only up to 4 dimensions.
    /// </summary>
    public TensorShape(int[] shape)
        : this(
            shape.Length > 0 ? shape[0] : 0,
            shape.Length > 1 ? shape[1] : 0,
            shape.Length > 2 ? shape[2] : 0,
            shape.Length > 3 ? shape[3] : 0)
    {
    }
    #endregion

    #region Properties
    /// <summary>
    /// Kernel dimension ordering is [H,W,C,K] for efficiency purpose.
    /// Return kernel height.
    /// </summary>
    public int kernelHeight { get { return batch; } }
    /// <summary>
    /// Kernel dimension ordering is [H,W,C,K] for efficiency purpose.
    /// Return kernel width.
    /// </summary>
    public int kernelWidth { get { return height; } }
    /// <summary>
    /// Kernel dimension ordering is [H,W,C,K] for efficiency purpose.
    /// Return kernel depth (aka the number of input channels of the associated operator).
    /// </summary>
    public int kernelDepth { get { return width; } }
    /// <summary>
    /// Kernel dimension ordering is [H,W,C,K] for efficiency purpose.
    /// Return kernel count (aka the number of output channels of the associated operator).
    /// </summary>
    public int kernelCount { get { return channels; } }
    /// <summary>
    /// Return the number of batch.
    /// </summary>
    public int flatHeight { get { return batch; } }
    /// <summary>
    /// Return the H*W*C.
    /// </summary>
    public int flatWidth { get { return height * width * channels; } }
    /// <summary>
    /// Return the total number of elements represented by this shape.
    /// </summary>
    public int length { get { return batch * height * width * channels; } }
    /// <summary>
    /// Always 4, look also at the `dimensions` property.
    /// </summary>
    public int rank { get { return 4; } }
    /// <summary>
    /// Return the count of non-unit dimension of this shape.
    /// For example [B,1,1,C] dimensions is 2.
    /// </summary>
    public int dimensions { get {
            return
                (batch > 1 ? 1 : 0) +
                (height > 1 ? 1 : 0) +
                (width > 1 ? 1 : 0) +
                (channels > 1 ? 1 : 0);
        }
    }
    #endregion

    #region Helpers
    /// <summary>
    /// Allow to use negative axis to access tensorShape backward.
    /// `axis` should be from -rank to rank (exclusive).
    /// </summary>
    public int Axis(int axis)
    {
        Assert.IsTrue(axis > -rank && axis < rank);
        return axis >= 0 ? axis: rank + axis;
    }
    /// <summary>
    /// Given an offset in memory return the dimensions indices of the element as [b,h,w,c].
    /// </summary>
    public void GetPositionsFromIndex(int index, ref int b, ref int h, ref int w, ref int ch)
    {
        ch = index % channels;
        w = (index / channels) % width;
        h = (index / (channels * width)) % height;
        b = (index / (channels * width * height)) % batch;
    }
    /// <summary>
    /// Given an element dimensions indices [b,h,w,c] with broadcast support, return this element offset in memory.
    /// </summary>
    public int IndexWithBroadcast(int b, int h, int w, int ch)
    {
        b %= batch;
        h %= height;
        w %= width;
        ch %= channels;
        return Index(b, h, w, ch);
    }
    /// <summary>
    /// Given an element dimensions indices [b,h,w,c] return this element offset in memory.
    /// </summary>
    public int Index(int b, int h, int w, int ch)
    {
        int index =
            b * height * width * channels +
            h * width * channels +
            w * channels +
            ch;
        return index;
    }
    /// <summary>
    /// Given an element dimensions indices [b,0,0,c] return this element offset in memory.
    /// </summary>
    public int Index(int b, int c)
    {
        int index =
            b * height * width * channels +
            c;
        return index;
    }
    /// <summary>
    /// Indexer to return a dimension of this tensorShape as [B,H,W,C]
    /// Prefer this over ToArray() to avoid GC allocation/collection.
    /// </summary>
    public int this[int axis]
    {
        get
        {
            //switch case rather than `ToArray` to avoid GC allocation
            switch(axis)
            {
                case 0:
                    return batch;
                case 1:
                    return height;
                case 2:
                    return width;
                default:
                    return channels;
            }
        }
    }
    /// <summary>
    /// Return an array representation of this tensorShape as [B,H,W,C]
    /// Prefer tensorShape[x] to avoid GC allocation/collection.
    /// </summary>
    public int[] ToArray()
    {
        return new[] { batch, height, width, channels };
    }

    /// <summary>
    /// Remove single-dimensional entries from the shape.
    /// [b=4,h=1,w=1,c=128] => [b=1,h=1,w=4,c=128]
    /// </summary>
    public TensorShape Squeeze()
    {
        var dims = ToArray();

        var squeezed = new[] { 1,1,1,1 };
        Assert.IsTrue(dims.Length == squeezed.Length);
        var index = squeezed.Length;
        foreach (var dim in dims)
            if (dim > 1)
                squeezed[--index] = dim;
        return new TensorShape(squeezed);
    }

    /// <summary>
    /// Return a TensorShape of dimensions [B,1,1,H*W*C]
    /// </summary>
    public TensorShape Flatten()
    {
        return new TensorShape(batch, height * width * channels);
    }
    #endregion

    #region Comparison operators
    public static bool operator ==(TensorShape a, TensorShape b)
    {
        return
            a.batch == b.batch &&
            a.height == b.height &&
            a.width == b.width &&
            a.channels == b.channels;
    }

    public static bool operator !=(TensorShape a, TensorShape b)
    {
        return !(a == b);
    }

    public override bool Equals(Object obj)
    {
        // Check for null values and compare run-time types.
        if (obj == null || GetType() != obj.GetType())
            return false;

        return this == (TensorShape)obj;
    }

    public override int GetHashCode()
    {
        return batch ^ height ^ width ^ channels;
    }
    #endregion

    public override string ToString()
    {
        return $"({batch}, {height}, {width}, {channels})";
    }
}


// @TODO: most likely Tensor should still be struct - that way passing Tensor as argument into IOps would be safer (no hidden state mods), and Flatten & Reshape could return modified Tensor
// ITensorData & Dispose mechanism should however allow Tensors to share the same ITensorData
public class Tensor : IDisposable
{
    private ITensorData m_TensorOnDevice;
    private ITensorAllocator m_TensorAllocator;
    private float[] m_Cache;
    private bool m_CacheIsDirty;

    /// <summary>
    /// Return this tensor name.
    /// </summary>
    public string name;
    /// <summary>
    /// Return this tensor allocator, see interface `ITensorAllocator`.
    /// </summary>
    public ITensorAllocator allocator { get { return m_TensorAllocator; } }

    #region Shape
    /// <summary>
    /// Return this tensor shape as [B,H,W,C].
    /// </summary>
    public readonly TensorShape shape;
    /// <summary>
    /// Return the number of batch.
    /// </summary>
    public int batch { get { return shape.batch; } }
    /// <summary>
    /// Return the spatial height.
    /// </summary>
    public int height { get { return shape.height; } }
    /// <summary>
    /// Return the spatial width.
    /// </summary>
    public int width { get { return shape.width; } }
    /// <summary>
    /// Return the number of channels.
    /// </summary>
    public int channels { get { return shape.channels; } }
    /// <summary>
    /// Kernel dimension ordering is [H,W,C,K] for efficiency purpose.
    /// Return kernel width.
    /// </summary>
    public int kernelWidth { get { return shape.kernelWidth; } }
    /// <summary>
    /// Kernel dimension ordering is [H,W,C,K] for efficiency purpose.
    /// Return kernel height.
    /// </summary>
    public int kernelHeight { get { return shape.kernelHeight; } }
    /// <summary>
    /// Kernel dimension ordering is [H,W,C,K] for efficiency purpose.
    /// Return kernel depth (aka the number of input channels of the associated operator).
    /// </summary>
    public int kernelDepth { get { return shape.kernelDepth; } }
    /// <summary>
    /// Kernel dimension ordering is [H,W,C,K] for efficiency purpose.
    /// Return kernel count (aka the number of output channels of the associated operator).
    /// </summary>
    public int kernelCount { get { return shape.kernelCount; } }
    /// <summary>
    /// Return the number of batch.
    /// </summary>
    public int flatHeight { get { return shape.flatHeight; } }
    /// <summary>
    /// Return the H*W*C.
    /// </summary>
    public int flatWidth { get { return shape.flatWidth; } }
    /// <summary>
    /// Return the total number of elements in this tensor.
    /// </summary>
    public int length { get { return shape.length; } }
    /// <summary>
    /// Return the count of non-unit dimension of this tensor shape.
    /// For example [B,1,1,C] dimensions is 2.
    /// </summary>
    public int dimensions { get { return shape.dimensions; } }
    #endregion

    #region Constructors
    /// <summary>
    /// Create a Tensor from a shape `s`, an array of data `srcData` and an optional name `n`
    /// `s` should be of size 4, order is [b,h,w,ch].
    /// `srcData` should be of size s[0]*s[1]*s[2]*s[3].
    /// </summary>
    public Tensor(int[] s, float[] srcData, string n = "") : this(new TensorShape(s), srcData, n) {}
    /// <summary>
    /// Create a Tensor of shape [b,h,w,ch], an array of data `srcData` and an optional name `n`
    /// `srcData` should be of size b*h*w*ch
    /// </summary>
    public Tensor(int b, int h, int w, int ch, float[] srcData, string n = "") : this(new TensorShape(b, h, w, ch), srcData, n) {}
    /// <summary>
    /// Create a Tensor of shape [b,1,1,ch], an array of data `srcData` and an optional name `n`
    /// `srcData` should be of size b*ch
    /// </summary>
    public Tensor(int b, int ch, float[] srcData, string n = "") : this(new TensorShape(b, ch), srcData, n) {}
    /// <summary>
    /// Create a Tensor of shape `s`, an array of data `srcData` and an optional name `n`
    /// `srcData` should be of size `s.length`.
    /// </summary>
    public Tensor(TensorShape s, float[] srcData, string n = "")
    {
        //;;UnityEngine.Debug.Log("Tensor::Tensor " + n + " " + s + " []-> " + srcData);
        name = n;
        shape = s;
        m_TensorOnDevice = new ArrayTensorData(shape);
        m_TensorOnDevice.Upload(srcData, 0, Math.Min(length, srcData.Length));
        m_TensorAllocator = null;
        m_Cache = null;
        m_CacheIsDirty = false;
    }
    /// <summary>
    /// Create a Tensor from a shape `s`, an array of data `srcData` and an optional name `n`
    /// `s` should be of size 4, order is [b,h,w,ch].
    /// `srcData` should be of size s[0]*s[1]*s[2]*s[3].
    /// </summary>
    public Tensor(int[] s, float[][] srcData, string n = "") : this(new TensorShape(s), srcData, n) {}
    /// <summary>
    /// Create a Tensor of shape [b,h,w,ch], an array of data `srcData` and an optional name `n`
    /// `srcData` should be of size b*h*w*ch
    /// </summary>
    public Tensor(int b, int h, int w, int ch, float[][] srcData, string n = "") : this(new TensorShape(b, h, w, ch), srcData, n) {}
    /// <summary>
    /// Create a Tensor of shape [b,1,1,ch], an array of data `srcData` and an optional name `n`
    /// `srcData` should be of size b*ch
    /// </summary>
    public Tensor(int b, int ch, float[][] srcData, string n = "") : this(new TensorShape(b, ch), srcData, n) {}
    /// <summary>
    /// Create a Tensor of shape `s`, an array of data `srcData` and an optional name `n`
    /// `srcData` should be of size `s.length`.
    /// </summary>
    public Tensor(TensorShape s, float[][] srcData, string n = "")
    {
        //;;UnityEngine.Debug.Log("Tensor::Tensor " + n + " " + s + " [][]-> " + srcData);
        name = n;
        shape = s;
        var arrayTensorData = new ArrayTensorData(shape);
        for (var i = 0; i < Math.Min(flatHeight, srcData.Length); ++i)
        {
            var src = srcData[i];
            var dstOffset = i * flatWidth;
            Array.Copy(src, 0, arrayTensorData.array, dstOffset, Math.Min(flatWidth, src.Length));
        }
        m_TensorOnDevice = arrayTensorData;
        m_TensorAllocator = null;
        m_Cache = null;
        m_CacheIsDirty = false;
    }
    /// <summary>
    /// Create a Tensor from a shape `s`, associated ComputeBuffer `srcBuffer` filled with tensor values, and an optional name `n`
    /// `s` should be of size 4, order is [b,h,w,ch].
    /// `srcBuffer` should be larger than s[0]*s[1]*s[2]*s[3].
    /// </summary>
    public Tensor(int[] s, UnityEngine.ComputeBuffer srcBuffer, string n = "") : this(new TensorShape(s), srcBuffer, n) {}
    /// <summary>
    /// Create a Tensor of shape [b,h,w,ch], associated ComputeBuffer `srcBuffer` filled with tensor values, and an optional name `n`
    /// `srcBuffer` should be larger than b*h*w*ch
    /// </summary>
    public Tensor(int b, int h, int w, int ch, UnityEngine.ComputeBuffer srcBuffer, string n = "") : this(new TensorShape(b, h, w, ch), srcBuffer, n) {}
    /// <summary>
    /// Create a Tensor of shape [b,1,1,ch], associated ComputeBuffer `srcBuffer` filled with tensor values, and an optional name `n`
    /// `srcBuffer` should be larger than b*ch
    /// </summary>
    public Tensor(int b, int ch, UnityEngine.ComputeBuffer srcBuffer, string n = "") : this(new TensorShape(b, ch), srcBuffer, n) {}
    /// <summary>
    /// Create a Tensor of shape `s`, associated ComputeBuffer `srcBuffer` filled with tensor values, and an optional name `n`
    /// `srcBuffer` should be larger than `s.length`.
    /// </summary>
    public Tensor(TensorShape s, UnityEngine.ComputeBuffer srcBuffer, string n = "")
    {
        name = n;
        shape = s;
        if (srcBuffer.count < s.length)
            throw new ArgumentException($"Compute buffer {name} capacity is {srcBuffer.count} less than {s.length} required for shape {s}");
        if (srcBuffer.stride == 4)
            throw new ArgumentException($"Currently only compute buffers with stride of 4 are supported. Compute buffer {name} stride is {srcBuffer.stride} instead");
        m_TensorOnDevice = new ComputeTensorData(srcBuffer, shape, offset:0, name);
        m_TensorAllocator = null;
        m_Cache = null;
        m_CacheIsDirty = false;
    }

    /// <summary>
    /// Create a Tensor from a texture, shape is [1, texture.height, texture.width, `channels=3`]
    /// </summary>
    public Tensor(UnityEngine.Texture srcTexture, int channels = 3, string n = "") : this(new [] { srcTexture }, channels, n) {}
    /// <summary>
    /// Create a Tensor from multiple texture, shape is [srcTextures.length, texture.height, texture.width, `channels=3`]
    /// All textures must be of the same size and dimension.
    /// </summary>
    public Tensor(UnityEngine.Texture[] srcTextures, int channels = 3, string n = "")
    {
        name = n;
        var tensorData = new TextureAsTensorData(srcTextures, channels);
        //;;UnityEngine.Debug.Log("Tensor::Tensor " + n + " " + tensorData.shape + " [TEX] " + srcTextures);
        shape = tensorData.shape;
        Assert.IsTrue(tensorData.GetMaxCount() >= length);
        m_TensorOnDevice = tensorData;
        m_TensorAllocator = null;
        m_Cache = null;
        m_CacheIsDirty = false;
    }
    /// <summary>
    /// Create a Tensor from a shape `s`, a ITensorData `d` and an optional name `n`
    /// `s` should be of size 4, order is [b,h,w,ch].
    /// </summary>
    public Tensor(int[] s, ITensorData d, string n = "") : this(new TensorShape(s), d, n) {}
    /// <summary>
    /// Create a Tensor of shape [b,h,w,ch], a ITensorData `d` and an optional name `n`
    /// `srcData` should be of size b*h*w*ch
    /// </summary>
    public Tensor(int b, int h, int w, int ch, ITensorData d, string n = "") : this(new TensorShape(b, h, w, ch), d, n) {}
    /// <summary>
    /// Create a Tensor of shape [b,1,1,ch], a ITensorData `d` and an optional name `n`
    /// `srcData` should be of size b*ch
    /// </summary>
    public Tensor(int b, int ch, ITensorData d, string n = "") : this(new TensorShape(b, ch), d, n) {}
    /// <summary>
    /// Create a Tensor of shape `s`, a ITensorData `d` and an optional name `n`
    /// </summary>
    public Tensor(TensorShape s, ITensorData d, string n = "")
    {
        //;;UnityEngine.Debug.Log("Tensor::Tensor " + n + " " + s + " @ " + ((d != null) ? d.GetType().Name : "null"));
        name = n;
        shape = s;
        m_TensorOnDevice = d;
        m_TensorAllocator = null;
        m_Cache = null;
        m_CacheIsDirty = false;
    }
    /// <summary>
    /// Create an uninitialized Tensor with a shape of [1,1,1,1].
    /// </summary>
    public Tensor(string n = "") : this(new TensorShape(1,1,1,1), n) {}
    /// <summary>
    /// Create an uninitialized Tensor from a shape `s`. `s` should be of size 4, order is [b,h,w,ch]
    /// </summary>
    public Tensor(int[] s, string n = "") : this(new TensorShape(s), n) {}
    /// <summary>
    /// Create an uninitialized Tensor of shape [b,h,w,ch].
    /// </summary>
    public Tensor(int b, int h, int w, int ch, string n = "") : this(new TensorShape(b, h, w, ch), n) {}
    /// <summary>
    /// Create an uninitialized Tensor of shape [b,1,1,ch].
    /// </summary>
    public Tensor(int b, int ch, string n = "") : this(new TensorShape(b, ch), n) {}
    /// <summary>
    /// Create an uninitialized Tensor of shape `s`.
    /// </summary>
    public Tensor(TensorShape s, string n = "")
    {
        //;;UnityEngine.Debug.Log("Tensor::Tensor " + n + " " + s);
        name = n;
        shape = s;
        m_TensorOnDevice = null;
        m_TensorAllocator = null;
        m_Cache = null;
        m_CacheIsDirty = false;
    }
    /// <summary>
    /// Create a Tensor from a shape `s`, a ITensorData `d` and a ITensorAllocator `a`
    /// `s` should be of size 4, order is [b,h,w,ch].
    /// </summary>
    public Tensor(int[] s, ITensorData d, ITensorAllocator a) : this(new TensorShape(s), d, a) {}
    /// <summary>
    /// Create a Tensor of shape [b,h,w,ch], a ITensorData `d` and a ITensorAllocator `a`
    /// </summary>
    public Tensor(int b, int h, int w, int ch, ITensorData d, ITensorAllocator a) : this(new TensorShape(b, h, w, ch), d, a) {}
    /// <summary>
    /// Create a Tensor of shape [b,1,1,ch], a ITensorData `d` and a ITensorAllocator `a`
    /// `srcData` should be of size b*ch
    /// </summary>
    public Tensor(int b, int ch, ITensorData d, ITensorAllocator a) : this(new TensorShape(b, ch), d, a) {}
    /// <summary>
    /// Create a Tensor of shape `s`, a ITensorData `d` and a ITensorAllocator `a`
    /// </summary>
    public Tensor(TensorShape s, ITensorData d, ITensorAllocator a)
    {
        //;;UnityEngine.Debug.Log("Tensor::Tensor " + s + " " + d + " " + a);
        name = "";
        shape = s;
        m_TensorOnDevice = d;
        m_TensorAllocator = a;
        m_Cache = null;
        m_CacheIsDirty = false;
    }

    /// <summary>
    /// Create an uninitialized Tensor with a shape of [1,1,1,1] and ITensorAllocator `a`
    /// </summary>
    public Tensor(ITensorAllocator a) : this(new TensorShape(1,1,1,1), a) {}
    /// <summary>
    /// Create an uninitialized Tensor from a shape `s` and ITensorAllocator `a`
    /// `s` should be of size 4, order is [b,h,w,ch].
    ///
    /// </summary>
    public Tensor(int[] s, ITensorAllocator a) : this(new TensorShape(s), a) {}
    /// <summary>
    /// Create an uninitialized Tensor of shape [b,h,w,ch] and ITensorAllocator `a`.
    /// </summary>
    public Tensor(int b, int h, int w, int ch, ITensorAllocator a) : this(new TensorShape(b, h, w, ch), a) {}
    /// <summary>
    /// Create an uninitialized Tensor of shape [b,1,1,ch] and ITensorAllocator `a`.
    /// </summary>
    public Tensor(int b, int ch, ITensorAllocator a) : this(new TensorShape(b, ch), a) {}
    /// <summary>
    /// Create an uninitialized Tensor of shape `s` and ITensorAllocator `a`.
    /// </summary>
    public Tensor(TensorShape s, ITensorAllocator a)
    {
        //;;UnityEngine.Debug.Log("Tensor::Tensor " + n + " " + s + " " + a);
        name = "";
        shape = s;
        m_TensorOnDevice = null;
        m_TensorAllocator = a;
        m_Cache = null;
        m_CacheIsDirty = false;
    }
    #endregion

    /// <summary>
    /// Destructor will also dispose associated memories.
    /// </summary>
    ~Tensor()
    {
        Dispose();
    }

    /// <summary>
    /// Allocate tensor on device if needed and update data.
    /// By default cached copy of the data will be discarded when doing so, set `forceInvalidateCache` to false to keep the cache.
    /// </summary>
    public void PinToDeviceAndUploadToIt(ITensorData onDevice, bool forceInvalidateCache = true)
    {
        if (m_TensorOnDevice == onDevice && !m_CacheIsDirty)
            return;

        PrepareCacheForAccess();
        PinToDevice(onDevice, disposeUnpinned: true);

        m_CacheIsDirty = true;
        if (forceInvalidateCache)
            UploadAndInvalidateCache();
        else
            UploadIfDirty();
    }

    /// <summary>
    /// Allocate tensor on device if needed and download data to cache.
    /// See also `PrepareCacheForAccess()`.
    /// </summary>
    public void PinToDeviceAndDownloadFromIt(ITensorData onDevice)
    {
        if (m_TensorOnDevice == onDevice && !m_CacheIsDirty)
            return;

        UploadIfDirty();
        PinToDevice(onDevice, disposeUnpinned: true);
        if (m_Cache != null)
            PrepareCacheForAccess();
    }

    private void PinToDevice(ITensorData onDevice, bool disposeUnpinned = true)
    {
        Assert.IsTrue(onDevice?.GetMaxCount() >= length || onDevice == null);

        if (m_TensorAllocator != null)
            m_TensorAllocator.Repin(this, onDevice, m_TensorOnDevice, disposeUnpinned);
        else if (disposeUnpinned)
            m_TensorOnDevice?.Dispose();

        m_TensorOnDevice = onDevice;
    }

    /// <summary>
    /// Cast a tensorData to this tensor, transferring ownership of on tensorData device memory to this tensor.
    /// </summary>
    public void CastOnDevice(ITensorData onDevice)
    {
        if (m_TensorOnDevice == onDevice)
            return;

        Assert.IsNotNull(onDevice);
        Assert.IsNotNull(m_TensorOnDevice);
        Assert.IsTrue(onDevice.GetMaxCount() >= length);

        if (m_TensorAllocator != null)
            m_TensorAllocator.Cast(this, onDevice, m_TensorOnDevice);

        m_TensorOnDevice = onDevice;
    }

    /// <summary>
    /// Remove tensor from device, will first sync the cache with device data.
    /// </summary>
    public ITensorData Unpin(bool disposeUnpinned = true)
    {
        PrepareCacheForAccess();

        ITensorData unpinned = (disposeUnpinned) ? null : m_TensorOnDevice;
        PinToDevice(null, disposeUnpinned);
        return unpinned;
    }

    private bool m_Disposing = false;    // to protect from infinite-loop. in case UnpinAndDisposeTensor() is called from Dispose()
    /// <summary>
    /// Remove tensor from device, and dispose it.
    /// </summary>
    public ITensorData UnpinAndDisposeTensor()
    {
        // NOTE: since this Tensor is going to be Disposed
        // there is no need to populate cache with data from tensorOnDevice
        // we can save on skipping PrepareCacheForAccess() call
        ITensorData unpinned = m_TensorOnDevice;
        PinToDevice(null, false);
        if (!m_Disposing)
            Dispose();
        return unpinned;
    }

    private void UploadIfDirty()
    {
        if (m_CacheIsDirty && m_TensorOnDevice != null)
            m_TensorOnDevice.Upload(m_Cache);
        m_CacheIsDirty = false;
    }

    private void UploadAndInvalidateCache()
    {
        UploadIfDirty();

        // remove cache only, if pinned to device
        // otherwise cache holds the only copy of the tensor data and we can not loose it
        if (m_TensorOnDevice == null)
            return;

        m_Cache = null;
        m_CacheIsDirty = false;
    }

    /// <summary>
    /// Populate the cache with on device data.
    /// Blocking read if `blocking` is true (default)
    /// </summary>
    public bool PrepareCacheForAccess(bool blocking = true)
    {
        // non-blocking, schedule download for later
        if (!blocking && m_TensorOnDevice != null && m_Cache == null)
            if (!m_TensorOnDevice.ScheduleAsyncDownload(length))
                return false;

        // blocking, have to get data now!
        if (m_Cache == null)
        {
            if (m_TensorOnDevice != null)
                m_Cache = m_TensorOnDevice.Download(length);
            else
                m_Cache = new float[length];
            m_CacheIsDirty = false;
        }

        return true;
    }

    /// <summary>
    /// Upload cache to device memory and delete it.
    /// </summary>
    public void FlushCache()
    {
        UploadAndInvalidateCache();
    }

    // @TODO: choose approach to handle case when tensors after Flatten/Reshape are written into OR taken ownership of
    // 1) owns data, copy on PrepareCacheForAccess() and PinForWrite()
    // 2) always copy data in Flatten()/Reshape(), remove from Tensor interface
    // 2) always copy data in Flatten()/Reshape(), implement ICloneable for GPU ITensorData

    /// <summary>
    /// Create a flattened copy of the current Tensor ie of shape [B,1,1,H*W*CH]
    /// </summary>
    public Tensor Flatten()
    {
        var newShape = shape.Flatten();

        Tensor copy;
        if (m_TensorAllocator != null)
            copy = m_TensorAllocator.Alloc(newShape, m_TensorOnDevice);
        else
            copy = new Tensor(newShape, m_TensorOnDevice);

        copy.name = $"flatten of {name}";
        copy.m_Cache = m_Cache;
        copy.m_CacheIsDirty = m_CacheIsDirty;
        return copy;
    }

    /// <summary>
    /// Create a reshaped copy of the current Tensor.
    /// `newShape`.length must be equal to this.shape.length.
    /// </summary>
    public Tensor Reshape(TensorShape newShape)
    {
        Assert.AreEqual(shape.length, newShape.length);
        Tensor copy;
        if (m_TensorAllocator != null)
            copy = m_TensorAllocator.Alloc(newShape, m_TensorOnDevice);
        else
            copy = new Tensor(newShape, m_TensorOnDevice);

        copy.name = $"reshape of {name}";
        copy.m_Cache = m_Cache;
        copy.m_CacheIsDirty = m_CacheIsDirty;
        return copy;
    }

    /// <summary>
    /// Create a copy of the current Tensor, sharing data storage with original tensor.
    /// </summary>
    public Tensor ShallowCopy()
    {
        Tensor copy;
        if (m_TensorAllocator != null)
            copy = m_TensorAllocator.Alloc(shape, m_TensorOnDevice);
        else
            copy = new Tensor(shape, m_TensorOnDevice);

        copy.name = $"copy of {name}";
        copy.m_Cache = m_Cache;
        copy.m_CacheIsDirty = m_CacheIsDirty;

        return copy;
    }

    /// <summary>
    /// Create a copy of the current Tensor, actively syncing there data in a blocking way.
    /// </summary>
    public Tensor DeepCopy()
    {
        // @TODO: use Tensor allocator
        var copy = new Tensor(shape, $"clone of {name}");
        if (m_TensorOnDevice is ICloneable)
        {
            UploadIfDirty();
            var copyOfTensorData = (m_TensorOnDevice as ICloneable).Clone() as ITensorData;
            copy.PinToDeviceAndDownloadFromIt(copyOfTensorData);
        }
        else
        {
            PrepareCacheForAccess();
            copy.PrepareCacheForAccess();
            Array.Copy(m_Cache, 0, copy.m_Cache, 0, length);
        }

        return copy;
    }

    /// <summary>
    /// Remove system reference to this tensor, caller assume ownership.
    /// </summary>
    public void TakeOwnership()
    {
        m_TensorAllocator?.WaiveOwnership(this);
        m_TensorAllocator = null;
    }

    /// Called from ITensorAllocator, puts Tensor in the ready for reuse state.
    internal ITensorData Invalidate()
    {
        ITensorData unpinned = m_TensorOnDevice;
        PinToDevice(null, false);
        Assert.AreEqual(m_TensorOnDevice, null);
        m_Cache = null;
        m_CacheIsDirty = false;
        m_TensorOnDevice = null;
        m_TensorAllocator = null;
        return unpinned;
    }

    /// <summary>
    /// Dispose Tensor and associated memories.
    /// </summary>
    public virtual void Dispose()
    {
        m_Disposing = true;
        if (m_TensorAllocator != null)
        {
            m_TensorAllocator.Release(this, true);
        }
        else if (m_TensorOnDevice != null)
        {
            //;;UnityEngine.D.Log("DISPOSE " + name + " " + shape + " @ " + m_TensorOnDevice.GetType().Name);
            m_TensorOnDevice.Dispose();
        }

        m_Cache = null;
        m_CacheIsDirty = false;
        m_TensorOnDevice = null;
        m_TensorAllocator = null;
        m_Disposing = false;
    }


    #region Render Texture
    /// <summary>
    /// Fill a RenderTexture with a slice/batch of a tensor.
    /// </summary>
    public void ToRenderTexture(UnityEngine.RenderTexture target, int batch = 0, int fromChannel = 0, float scale = 1.0f, float bias = 0f)
    {
        var gpuBackend = new ReferenceComputeOps(ComputeShaderSingleton.Instance.referenceKernels);
        gpuBackend.TensorToRenderTexture(this, target, batch, fromChannel, scale, bias);
    }

    /// <summary>
    /// Create a new RenderTexture from a slice/batch of a tensor.
    /// </summary>
    public UnityEngine.RenderTexture ToRenderTexture(int batch = 0, int fromChannel = 0, float scale = 1.0f, float bias = 0f)
    {
        var target = new UnityEngine.RenderTexture(width, height, 0);
        ToRenderTexture(target, batch, fromChannel, scale, bias);
        return target;
    }
    #endregion


    #region Data access
    /// <summary>
    /// Allow to use negative axis to access tensorShape backward.
    /// `axis` should be from -rank to rank (exclusive).
    /// </summary>
    public int Axis(int axis)
    {
        return shape.Axis(axis);
    }
    /// <summary>
    /// Given an element dimensions indices [b,h,w,c] return this element offset in memory.
    /// </summary>
    public int Index(int b, int h, int w, int ch)
    {
        return shape.Index(b, h, w, ch);
    }
    /// <summary>
    /// Given an element dimensions indices [b,h,w,c] with broadcast support, return this element offset in memory.
    /// </summary>
    public int IndexWithBroadcast(int b, int h, int w, int ch)
    {
        return shape.IndexWithBroadcast(b, h, w, ch);
    }
    /// <summary>
    /// Given an element dimensions indices [b,0,0,c] return this element offset in memory.
    /// </summary>
    public int Index(int y, int x)
    {
        return shape.Index(y, x);
    }
    /// <summary>
    /// Access element at offset `index` in this Tensor.
    /// This will create a blocking read if cache is dirty.
    /// </summary>
    public float this[int index]
    {
        get { PrepareCacheForAccess(); return m_Cache[index]; }
        set { PrepareCacheForAccess(); m_Cache[index] = value; m_CacheIsDirty = true; }
    }
    /// <summary>
    /// Access element at index [b,0,0,ch] in this Tensor.
    /// This will create a blocking read if cache is dirty.
    /// </summary>
    public float this[int b, int ch]
    {
        get { PrepareCacheForAccess(); return m_Cache[Index(b, ch)]; }
        set { PrepareCacheForAccess(); m_Cache[Index(b, ch)] = value; m_CacheIsDirty = true; }
    }
    /// <summary>
    /// Access element at index [b,h,w,ch] in this Tensor.
    /// This will create a blocking read if cache is dirty.
    /// </summary>
    public float this[int b, int h, int w, int ch]
    {
        get { PrepareCacheForAccess(); return m_Cache[Index(b, h, w, ch)]; }
        set { PrepareCacheForAccess(); m_Cache[Index(b, h, w, ch)] = value; m_CacheIsDirty = true; }
    }

    // @TODO: implement via ITensorData.SharedAccess()
    /// <summary>
    /// Return the cached linear memory representation of this tensor data.
    /// This will create a blocking read if cache is dirty.
    /// see also `readonlyArrayOffset`.
    /// IMPORTANT: This data should not be modified.
    /// </summary>
    public float[] readonlyArray { get { PrepareCacheForAccess(); return m_Cache; } }
    // @TODO: implement via ITensorData.SharedAccess()
    /// <summary>
    /// Return the offset to use when accessing `readonlyArray`
    /// Always 0 at the moment.
    /// </summary>
    public int readonlyArrayOffset { get { return 0; } }
    #endregion

    public ITensorData tensorOnDevice { get { return m_TensorOnDevice; } }
    public ITensorData data
    {
        get
        {
            if (m_TensorOnDevice == null)
                PinToDeviceAndUploadToIt(new ArrayTensorData(shape));
            return m_TensorOnDevice;
        }
    }

    public override string ToString()
    {
        return $"({name} {shape}, alloc: {m_TensorAllocator?.GetType()}, onDevice:{m_TensorOnDevice})";
    }

}

} // namespace Barracuda
