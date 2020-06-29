using UnityEngine.Assertions;
using System;
using UnityEngine;

namespace Unity.Barracuda {

/// <summary>
/// TensorShape are immutable representation of a Tensor dimensions and rank.
/// At the moment a TensorShape is always of rank 4 and channels last ie NHWC.
/// However an axis can be of size 1. For example a tensor without spatial information will be N,1,1,C
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
    /// Create a TensorShape of shape [N,H,W,C].
    /// </summary>
    public TensorShape(int n, int h, int w, int c)
    {
        batch = n > 0 ? n : 1;
        height = h > 0 ? h : 1;
        width = w > 0 ? w : 1;
        channels = c > 0 ? c : 1;
    }
    /// <summary>
    /// Create a TensorShape of shape [N,1,1,C].
    /// </summary>
    public TensorShape(int n, int c)
    {
        batch = n > 0 ? n : 1;
        height = 1;
        width = 1;
        channels = c > 0 ? c : 1;
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
    /// For example [N,1,1,C] dimensions is 2.
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
    /// Given an offset in memory return the dimensions indices of the element as [N,H,W,C].
    /// </summary>
    public void GetPositionsFromIndex(int index, ref int n, ref int h, ref int w, ref int c)
    {
        c = index % channels;
        w = (index / channels) % width;
        h = (index / (channels * width)) % height;
        n = (index / (channels * width * height)) % batch;
    }
    /// <summary>
    /// Given an offset in memory return the dimensions indices of the element as [N,H,W,C] in NCHW format.
    /// </summary>
    internal void GetPositionsFromIndexNCHW(int index, ref int n, ref int h, ref int w, ref int c)
    {
        w = index % width;
        h = (index / width) % height;
        c = (index / (width * height)) % channels;
        n = (index / (width * height * channels)) % batch;
    }
    /// <summary>
    /// Given an element dimensions indices [N,H,W,C] with broadcast support, return this element offset in memory.
    /// </summary>
    public int IndexWithBroadcast(int n, int h, int w, int c)
    {
        n %= batch;
        h %= height;
        w %= width;
        c %= channels;
        return Index(n, h, w, c);
    }
    /// <summary>
    /// Given an element dimensions indices [N,H,W,C] return this element offset in memory, clamping indices to tensor dimensions.
    /// </summary>
    public int IndexWithClamp(int n, int h, int w, int c)
    {
        n = Math.Max(n, 0);
        h = Math.Max(h, 0);
        w = Math.Max(w, 0);
        c = Math.Max(c, 0);
        n = Math.Min(n, batch - 1);
        h = Math.Min(h, height - 1);
        w = Math.Min(w, width - 1);
        c = Math.Min(c, channels - 1);
        return Index(n, h, w, c);
    }
    /// <summary>
    /// Given an element dimensions indices [N,H,W,C] return this element offset in memory.
    /// </summary>
    public int Index(int n, int h, int w, int c)
    {
        int index =
            n * height * width * channels +
            h * width * channels +
            w * channels +
            c;
        return index;
    }
    /// <summary>
    /// Given an element dimensions indices [N,H,W,C] return this element offset in memory in NCHW format.
    /// </summary>
    internal int IndexNCHW(int n, int h, int w, int c)
    {
        int index =
            n * channels * height * width +
            c * height * width +
            h * width +
            w;
        return index;
    }
    /// <summary>
    /// Given an element dimensions indices [N,0,0,C] return this element offset in memory.
    /// </summary>
    public int Index(int n, int c)
    {
        int index =
            n * height * width * channels +
            c;
        return index;
    }
    /// <summary>
    /// Indexer to return a dimension of this tensorShape as [N,H,W,C]
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

        internal set
        {
            unsafe
            {
                fixed (TensorShape* thiz = &this)
                {
                    int* p = (int*)thiz;

                    if (axis < 0 || axis > 3)
                        axis = 3;

                    p[axis] = value > 0 ? value : 1;
                }
            }
        }
    }
    /// <summary>
    /// Return an array representation of this tensorShape as [N,H,W,C]
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
    /// Return a TensorShape of dimensions [N,1,1,H*W*C]
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

    public override bool Equals(System.Object obj)
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
    private bool m_Disposed = false;

    #region Debug
    /// <summary>
    /// Return this tensor name.
    /// </summary>
    public string name;
    /// <summary>
    /// Return if tensor was already disposed.
    /// </summary>
    internal bool disposed { get { return m_Disposed; } }
    #endregion

    /// <summary>
    /// Return this tensor allocator, see interface `ITensorAllocator`.
    /// </summary>
    public ITensorAllocator allocator { get { return m_TensorAllocator; } }

    #region Shape
    /// <summary>
    /// Return this tensor shape as [N,H,W,C].
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
    /// For example [N,1,1,C] dimensions is 2.
    /// </summary>
    public int dimensions { get { return shape.dimensions; } }
    #endregion

    #region Constructors
    /// <summary>
    /// Create a Tensor from a `shape`, an array of data `srcData` and an optional debug `name`.
    /// `shape` must be of size 4, the order is [N,H,W,C].
    /// `srcData` must be of size `s[0]*s[1]*s[2]*s[3]`.
    /// </summary>
    public Tensor(int[] shape, float[] srcData, string name = "") : this(new TensorShape(shape), srcData, name) {}
    /// <summary>
    /// Create a Tensor of shape [N,H,W,C], an array of data `srcData` and an optional debug `name`.
    /// `srcData` must be of size `n*h*w*c`.
    /// </summary>
    public Tensor(int n, int h, int w, int c, float[] srcData, string name = "") : this(new TensorShape(n, h, w, c), srcData, name) {}
    /// <summary>
    /// Create a Tensor of shape [N,1,1,C], an array of data `srcData` and an optional debug `name`.
    /// `srcData` must be of size `n*c`.
    /// </summary>
    public Tensor(int n, int c, float[] srcData, string name = "") : this(new TensorShape(n, c), srcData, name) {}
    /// <summary>
    /// Create a Tensor with specified `shape`, an array of data `srcData` and an optional debug `name`.
    /// `srcData` must be of size `shape.length`.
    /// </summary>
    public Tensor(TensorShape shape, float[] srcData, string name = "")
    {
        this.name = name;
        this.shape = shape;
        m_TensorOnDevice = new ArrayTensorData(shape);
        Assert.IsTrue(srcData.Length == length);
        m_TensorOnDevice.Upload(srcData, shape, 0);
        m_TensorAllocator = null;
        m_Cache = null;
        m_CacheIsDirty = false;
    }
    /// <summary>
    /// Create a Tensor from a `shape`, an array of data `srcData` and an optional name debug `name`.
    /// `shape` must be of size 4, the order is [N,H,W,C].
    /// `srcData` must be of size `s[0]*s[1]*s[2]*s[3]`.
    /// </summary>
    public Tensor(int[] shape, float[][] srcData, string name = "") : this(new TensorShape(shape), srcData, name) {}
    /// <summary>
    /// Create a Tensor of shape [N,H,W,C], an array of data `srcData` and an optional debug `name`.
    /// `srcData` must be of size `n*h*w*c`.
    /// </summary>
    public Tensor(int n, int h, int w, int c, float[][] srcData, string name = "") : this(new TensorShape(n, h, w, c), srcData, name) {}
    /// <summary>
    /// Create a Tensor of shape [N,1,1,C], an array of data `srcData` and an optional debug `name`.
    /// `srcData` must be of size `n*c`.
    /// </summary>
    public Tensor(int n, int c, float[][] srcData, string name = "") : this(new TensorShape(n, c), srcData, name) {}
    /// <summary>
    /// Create a Tensor with specified `shape`, an array of data `srcData` and an optional debug `name`.
    /// `srcData` must be of size `shape.length`.
    /// </summary>
    public Tensor(TensorShape shape, float[][] srcData, string name = "")
    {
        this.name = name;
        this.shape = shape;
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
    /// Create a Tensor from a `shape`, associated ComputeBuffer `srcBuffer` filled with tensor values, and an optional debug `name`.
    /// `shape` must be of size 4, the order is [N,H,W,C].
    /// `srcBuffer` must be larger than `s[0]*s[1]*s[2]*s[3]`.
    /// </summary>
    public Tensor(int[] shape, UnityEngine.ComputeBuffer srcBuffer, string name = "") : this(new TensorShape(shape), srcBuffer, name) {}
    /// <summary>
    /// Create a Tensor of shape [N,H,W,C], associated ComputeBuffer `srcBuffer` filled with tensor values, and an optional debug `name`.
    /// `srcBuffer` must be larger than `n*h*w*c`.
    /// </summary>
    public Tensor(int n, int h, int w, int c, UnityEngine.ComputeBuffer srcBuffer, string name = "") : this(new TensorShape(n, h, w, c), srcBuffer, name) {}
    /// <summary>
    /// Create a Tensor of shape [N,1,1,C], associated ComputeBuffer `srcBuffer` filled with tensor values, and an optional debug `name`.
    /// `srcBuffer` must be larger than `n*c`.
    /// </summary>
    public Tensor(int n, int c, UnityEngine.ComputeBuffer srcBuffer, string name = "") : this(new TensorShape(n, c), srcBuffer, name) {}
    /// <summary>
    /// Create a Tensor with specified `shape`, associated ComputeBuffer `srcBuffer` filled with tensor values, and an optional debug `name`.
    /// `srcBuffer` must be larger than `shape.length`.
    /// </summary>
    public Tensor(TensorShape shape, UnityEngine.ComputeBuffer srcBuffer, string name = "")
    {
        this.name = name;
        this.shape = shape;
        if (srcBuffer.count < shape.length)
            throw new ArgumentException($"Compute buffer `{name}` capacity is {srcBuffer.count} less than {shape.length} required for shape {shape}");
        if (srcBuffer.stride != 4)
            throw new ArgumentException($"Currently only compute buffers with stride of 4 are supported. Compute buffer `{name}` stride is {srcBuffer.stride} instead");
        m_TensorOnDevice = new ComputeTensorData(srcBuffer, shape, offset:0, name, ComputeInfo.channelsOrder);
        m_TensorAllocator = null;
        m_Cache = null;
        m_CacheIsDirty = false;
    }

    /// <summary>
    /// Create a Tensor from a texture, shape is [1, `texture.height`, `texture.width`, `channels`].
    /// If `channels` is set to -1 (default value), then number of channels in the new Tensor will match the number of channels in the texture.
    /// Just like `Texture2D.GetPixels` when reading from LDR texture (RGBA32, ARGB32, RGB24, Alpha8, RG16, R8, etc) this function will remap pixel values from byte values to the range of [0.0 .. 1.0]. Pixel values from HDR textures (such as ARGBFloat or ARGBHalf) will be left unchanged.
    /// </summary>
    public Tensor(UnityEngine.Texture srcTexture, int channels = -1, string name = "") : this(new [] { srcTexture }, channels, name) {}
    /// <summary>
    /// Create a Tensor from multiple texture, shape is [`srcTextures.length`, `texture.height`, `texture.width`, `channels`].
    /// If `channels` is set to -1 (default value), then number of channels in the new Tensor will match the number of channels in the texture.
    /// All textures must be of the same size and dimension.
    /// Just like `Texture2D.GetPixels` when reading from LDR texture (RGBA32, ARGB32, RGB24, Alpha8, RG16, R8, etc) this function will remap pixel values from byte values to the range of [0.0 .. 1.0]. Pixel values from HDR textures (such as ARGBFloat or ARGBHalf) will be left unchanged.
    /// </summary>
    public Tensor(UnityEngine.Texture[] srcTextures, int channels = -1, string name = "")
    {
        this.name = name;
        var tensorData = new TextureAsTensorData(srcTextures, channels);
        //;;UnityEngine.Debug.Log("Tensor::Tensor " + n + " " + tensorData.shape + " [TEX] " + srcTextures);
        shape = tensorData.shape;
        Assert.IsTrue(tensorData.maxCapacity >= length);
        m_TensorOnDevice = tensorData;
        m_TensorAllocator = null;
        m_Cache = null;
        m_CacheIsDirty = false;
    }
    /// <summary>
    /// Create a Tensor from a `shape`, an ITensorData `data` and an optional debug `name`.
    /// `shape` must be of size 4, the order is [N,H,W,C].
    /// </summary>
    public Tensor(int[] shape, ITensorData data, string name = "") : this(new TensorShape(shape), data, name) {}
    /// <summary>
    /// Create a Tensor of shape [N,H,W,C], an ITensorData `data` and an optional debug `name`.
    /// `srcData` must be of size `n*h*w*c`.
    /// </summary>
    public Tensor(int n, int h, int w, int c, ITensorData data, string name = "") : this(new TensorShape(n, h, w, c), data, name) {}
    /// <summary>
    /// Create a Tensor of shape [N,1,1,C], an ITensorData `data` and an optional debug `name`.
    /// `srcData` must be of size `n*c`.
    /// </summary>
    public Tensor(int n, int c, ITensorData data, string name = "") : this(new TensorShape(n, c), data, name) {}
    /// <summary>
    /// Create a Tensor with specified `shape`, an ITensorData `data` and an optional debug `name`.
    /// </summary>
    public Tensor(TensorShape shape, ITensorData data, string name = "")
    {
        this.name = name;
        this.shape = shape;
        m_TensorOnDevice = data;
        m_TensorAllocator = null;
        m_Cache = null;
        m_CacheIsDirty = false;
    }
    /// <summary>
    /// Create an uninitialized Tensor with a shape of [1,1,1,1] and an optional debug `name`.
    /// </summary>
    public Tensor(string name = "") : this(new TensorShape(1,1,1,1), name) {}
    /// <summary>
    /// Create an uninitialized Tensor from a `shape` and an optional debug `name`.
    /// `shape` must be of size 4, the order is [N,H,W,C]
    /// </summary>
    public Tensor(int[] shape, string name = "") : this(new TensorShape(shape), name) {}
    /// <summary>
    /// Create an uninitialized Tensor of shape [N,H,W,C] and an optional debug `name`.
    /// </summary>
    public Tensor(int n, int h, int w, int c, string name = "") : this(new TensorShape(n, h, w, c), name) {}
    /// <summary>
    /// Create an uninitialized Tensor of shape [N,1,1,C] and an optional debug `name`.
    /// </summary>
    public Tensor(int n, int c, string name = "") : this(new TensorShape(n, c), name) {}
    /// <summary>
    /// Create an uninitialized Tensor with specified `shape` and an optional debug `name`.
    /// </summary>
    public Tensor(TensorShape shape, string name = "")
    {
        this.name = name;
        this.shape = shape;
        m_TensorOnDevice = null;
        m_TensorAllocator = null;
        m_Cache = null;
        m_CacheIsDirty = false;
    }
    /// <summary>
    /// Create a Tensor from a `shape`, an ITensorData `data` and an ITensorAllocator `allocator`.
    /// `shape` must be of size 4, the order is [N,H,W,C].
    /// </summary>
    public Tensor(int[] shape, ITensorData data, ITensorAllocator allocator) : this(new TensorShape(shape), data, allocator) {}
    /// <summary>
    /// Create a Tensor of shape [N,H,W,C], an ITensorData `data` and an ITensorAllocator `allocator`.
    /// `data` must be of size `n*h*w*c`.
    /// </summary>
    public Tensor(int n, int h, int w, int c, ITensorData data, ITensorAllocator allocator) : this(new TensorShape(n, h, w, c), data, allocator) {}
    /// <summary>
    /// Create a Tensor of shape [N,1,1,C], an ITensorData `data` and an ITensorAllocator `allocator`.
    /// `srcData` must be of size `n*c`.
    /// </summary>
    public Tensor(int n, int c, ITensorData data, ITensorAllocator allocator) : this(new TensorShape(n, c), data, allocator) {}
    /// <summary>
    /// Create a Tensor with specified `shape`, an ITensorData `data` and an ITensorAllocator `allocator`
    /// </summary>
    public Tensor(TensorShape shape, ITensorData data, ITensorAllocator allocator)
    {
        this.name = "";
        this.shape = shape;
        m_TensorOnDevice = data;
        m_TensorAllocator = allocator;
        m_Cache = null;
        m_CacheIsDirty = false;
    }

    /// <summary>
    /// Create an uninitialized Tensor with a shape of [1,1,1,1] and an ITensorAllocator `allocator`.
    /// </summary>
    public Tensor(ITensorAllocator allocator) : this(new TensorShape(1,1,1,1), allocator) {}
    /// <summary>
    /// Create an uninitialized Tensor from a `shape` and an ITensorAllocator `allocator`.
    /// `shape` must be of size 4, the order is [N,H,W,C].
    /// </summary>
    public Tensor(int[] shape, ITensorAllocator allocator) : this(new TensorShape(shape), allocator) {}
    /// <summary>
    /// Create an uninitialized Tensor of shape [N,H,W,C] and an ITensorAllocator `allocator`.
    /// </summary>
    public Tensor(int n, int h, int w, int c, ITensorAllocator allocator) : this(new TensorShape(n, h, w, c), allocator) {}
    /// <summary>
    /// Create an uninitialized Tensor of shape [N,1,1,C] and an ITensorAllocator `allocator`.
    /// </summary>
    public Tensor(int n, int c, ITensorAllocator allocator) : this(new TensorShape(n, c), allocator) {}
    /// <summary>
    /// Create an uninitialized Tensor with specified `shape` and ITensorAllocator `allocator`.
    /// </summary>
    public Tensor(TensorShape shape, ITensorAllocator allocator)
    {
        this.name = "";
        this.shape = shape;
        m_TensorOnDevice = null;
        m_TensorAllocator = allocator;
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

    private void PinToDevice(ITensorData onDevice, bool disposeUnpinned = true)
    {
        Assert.IsTrue(onDevice?.maxCapacity >= length || onDevice == null);

        if (m_TensorAllocator != null)
            m_TensorAllocator.MoveToDevice(this, onDevice, m_TensorOnDevice, disposeUnpinned);
        else if (disposeUnpinned)
            m_TensorOnDevice?.Dispose();

        m_TensorOnDevice = onDevice;
    }

    /// <summary>
    /// Upload tensor values to the device.
    /// This call associates tensor with the uninitialized block of data residing on a device.
    /// `destination` should be allocated on a target device. Previous contents of `destination` will be overwritten after this call.
    /// By default local cache will be discarded after this call, set `invalidateCacheAfterUpload` to false to keep the cache.
    /// </summary>
    public void UploadToDevice(ITensorData destination, bool invalidateCacheAfterUpload = true)
    {
        if (m_TensorOnDevice == destination && !m_CacheIsDirty)
            return;

        PrepareCacheForAccess();
        PinToDevice(destination, disposeUnpinned: true);

        m_CacheIsDirty = true;
        if (invalidateCacheAfterUpload)
            UploadAndInvalidateCache();
        else
            UploadIfDirty();
    }

    /// <summary>
    /// Associates tensor with the block of data residing on a device.
    /// Tensor values will be downloaded from the `source` upon the first access.
    /// `source` should contain initialized and valid data representing tensor values.
    /// See also `PrepareCacheForAccess()` to schedule download as soon as possible.
    /// </summary>
    public void AttachToDevice(ITensorData source)
    {
        if (m_TensorOnDevice == source && !m_CacheIsDirty)
            return;

        UploadIfDirty();
        PinToDevice(source, disposeUnpinned: true);
        if (m_Cache != null)
            PrepareCacheForAccess();
    }

    /// <summary>
    /// Remove tensor from device, will first sync the cache with device data.
    /// </summary>
    public ITensorData DetachFromDevice(bool disposeDeviceData = true)
    {
        PrepareCacheForAccess();

        ITensorData unpinned = (disposeDeviceData) ? null : m_TensorOnDevice;
        PinToDevice(null, disposeDeviceData);
        return unpinned;
    }

    private void UploadIfDirty()
    {
        if (m_CacheIsDirty && m_TensorOnDevice != null)
            m_TensorOnDevice.Upload(m_Cache, shape);
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
                m_Cache = m_TensorOnDevice.Download(shape);
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

    private Tensor ShallowCopy(TensorShape newShape, string newName)
    {
        Tensor copy;
        if (m_TensorAllocator != null)
            copy = m_TensorAllocator.Alloc(newShape, m_TensorOnDevice);
        else
            copy = new Tensor(newShape, m_TensorOnDevice);

        copy.name = newName;
        copy.m_Cache = m_Cache;
        copy.m_CacheIsDirty = m_CacheIsDirty;

        return copy;
    }

    /// <summary>
    /// Create a copy of the current Tensor, sharing data storage with original tensor.
    /// </summary>
    public Tensor ShallowCopy(string newName = null)
    {
        return ShallowCopy(shape, $"copy of {name}");
    }

    /// <summary>
    /// Create a flattened copy of the current Tensor ie of shape [N,1,1,H*W*C]
    /// </summary>
    public Tensor Flatten(string newName = null)
    {
        var newShape = shape.Flatten();
        return ShallowCopy(newShape, newName ?? $"flatten of {name}");
    }

    /// <summary>
    /// Create a reshaped copy of the current Tensor.
    /// `newShape`.length must be equal to this.shape.length.
    /// </summary>
    public Tensor Reshape(TensorShape newShape, string newName = null)
    {
        Assert.AreEqual(shape.length, newShape.length);
        return ShallowCopy(newShape, newName ?? $"reshape of {name}");
    }

    /// <summary>
    /// Create a copy of the current Tensor.
    /// </summary>
    public Tensor DeepCopy()
    {
        // @TODO: use Tensor allocator
        var copy = new Tensor(shape, $"clone of {name}");
        if (m_TensorOnDevice is ICloneable)
        {
            UploadIfDirty();
            var copyOfTensorData = (m_TensorOnDevice as ICloneable).Clone() as ITensorData;
            copy.AttachToDevice(copyOfTensorData);
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
        m_Disposed = true;
    }


    #region Render Texture
    /// <summary>
    /// Fill a `target` RenderTexture with a portion of the tensor applying `scale` and `bias`. Portion of the target is specified by `batch` and `fromChannel`.
    /// `batch` specifies the tensor batch to read values from.
    /// `fromChannel` specifies the first tensor channel to start reading values from.
    /// Number of channels in the `target` texture specifies how many channels to read from the tensor, starting from index `fromChannel`.
    /// Resolution of the `target` must match the spatial dimensions of the tensor.
    /// `scale` multiplier and `bias` addition is applied to the values read from the tensor and, if `target` is LDR texture (RGBA32, ARGB32, RGB24, Alpha8, RG16, R8, etc), clamped to the range from 0.0 to 1.0.
    /// </summary>
    public void ToRenderTexture(UnityEngine.RenderTexture target, int batch, int fromChannel, Vector4 scale, Vector4 bias, Texture3D lut = null)
    {
        var gpuBackend = new ReferenceComputeOps(ComputeShaderSingleton.Instance.referenceKernels);
        gpuBackend.TensorToRenderTexture(this, target, batch, fromChannel, scale, bias, lut);
    }

    /// <summary>
    /// Fill a `target` RenderTexture with a portion of the tensor applying `scale` and `bias`. Portion of the target is specified by `batch` and `fromChannel`.
    /// `batch` specifies the tensor batch to read values from.
    /// `fromChannel` specifies the first tensor channel to start reading values from.
    /// Number of channels in the `target` texture specifies how many channels to read from the tensor, starting from index `fromChannel`.
    /// Resolution of the `target` must match the spatial dimensions of the tensor.
    /// `scale` multiplier and `bias` addition is applied to the values read from the tensor and, if `target` is LDR texture (RGBA32, ARGB32, RGB24, Alpha8, RG16, R8, etc), clamped to the range from 0.0 to 1.0.
    /// </summary>
    public void ToRenderTexture(UnityEngine.RenderTexture target, int batch = 0, int fromChannel = 0, float scale = 1.0f, float bias = 0f, Texture3D lut = null)
    {
        ToRenderTexture(target, batch, fromChannel, new Vector4(scale,scale,scale,scale), new Vector4(bias,bias,bias,bias), lut);
    }

    /// <summary>
    /// Create new RenderTexture and fill it with a portion of the tensor applying `scale` and `bias`. Portion of the target is specified by `batch` and `fromChannel`.
    /// `format` specifies the type of the new RenderTexture.
    /// `batch` specifies the tensor batch to read values from.
    /// `fromChannel` specifies the first tensor channel to start reading values from.
    /// Number of channels in the `target` texture specifies how many channels to read from the tensor, starting from index `fromChannel`.
    /// `scale` multiplier and `bias` addition is applied to the values read from the tensor and, if `format` is LDR (RGBA32, ARGB32, RGB24, Alpha8, RG16, R8, etc), clamped to the range from 0.0 to 1.0.
    /// </summary>
    public UnityEngine.RenderTexture ToRenderTexture(RenderTextureFormat format, int batch = 0, int fromChannel = 0, float scale = 1.0f, float bias = 0f, Texture3D lut = null)
    {
        var target = new UnityEngine.RenderTexture(width, height, 0, format);
        ToRenderTexture(target, batch, fromChannel, scale, bias, lut);
        return target;
    }

    /// <summary>
    /// Create new RenderTexture and fill it with a portion of the tensor applying `scale` and `bias`. Portion of the target is specified by `batch` and `fromChannel`.
    /// `batch` specifies the tensor batch to read values from.
    /// `fromChannel` specifies the first tensor channel to start reading values from.
    /// Number of channels in the `target` texture specifies how many channels to read from the tensor, starting from index `fromChannel`.
    /// Resolution of the `target` must match the spatial dimensions of the tensor.
    /// `scale` multiplier and `bias` addition is applied to the values read from the tensor and clamped to the range from 0.0 to 1.0.
    /// </summary>
    public UnityEngine.RenderTexture ToRenderTexture(int batch = 0, int fromChannel = 0, float scale = 1.0f, float bias = 0f, Texture3D lut = null)
    {
        return ToRenderTexture(RenderTextureFormat.Default, batch, fromChannel, scale, bias, lut);
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
    /// Given an element dimensions indices [N,H,W,C] return this element offset in memory.
    /// </summary>
    public int Index(int b, int h, int w, int ch)
    {
        return shape.Index(b, h, w, ch);
    }
    /// <summary>
    /// Given an element dimensions indices [N,H,W,C] return this element offset in memory, clamping indices to tensor dimensions.
    /// </summary>
    public int IndexWithClamp(int b, int h, int w, int ch)
    {
        return shape.IndexWithClamp(b, h, w, ch);
    }
    /// <summary>
    /// Given an element dimensions indices [N,H,W,C] with broadcast support, return this element offset in memory.
    /// </summary>
    public int IndexWithBroadcast(int b, int h, int w, int ch)
    {
        return shape.IndexWithBroadcast(b, h, w, ch);
    }
    /// <summary>
    /// Given an element dimensions indices [N,0,0,C] return this element offset in memory.
    /// </summary>
    public int Index(int y, int x)
    {
        return shape.Index(y, x);
    }
    /// <summary>
    /// Access element at offset `index` in this Tensor.
    /// This will create a blocking read, if this Tensor is a result of a computation on a different device (GPU).
    /// </summary>
    public float this[int index]
    {
        get { PrepareCacheForAccess(); return m_Cache[index]; }
        set { PrepareCacheForAccess(); m_Cache[index] = value; m_CacheIsDirty = true; }
    }
    /// <summary>
    /// Access element at index [N,0,0,C] in this Tensor.
    /// This will create a blocking read, if this Tensor is a result of a computation on a different device (GPU).
    /// </summary>
    public float this[int b, int ch]
    {
        get { PrepareCacheForAccess(); return m_Cache[Index(b, ch)]; }
        set { PrepareCacheForAccess(); m_Cache[Index(b, ch)] = value; m_CacheIsDirty = true; }
    }
    /// <summary>
    /// Access element at index [N,H,W,C] in this Tensor.
    /// This will create a blocking read, if this Tensor is a result of a computation on a different device (GPU).
    /// </summary>
    public float this[int b, int h, int w, int ch]
    {
        get { PrepareCacheForAccess(); return m_Cache[Index(b, h, w, ch)]; }
        set { PrepareCacheForAccess(); m_Cache[Index(b, h, w, ch)] = value; m_CacheIsDirty = true; }
    }

    /// <summary>
    /// Return the cached linear memory representation of this tensor data.
    /// This will create a blocking read, if this Tensor is a result of a computation on a different device (GPU).
    /// IMPORTANT: Modifying contents of the returned array will have undefined behavior.
    /// </summary>
    public float[] ToReadOnlyArray()
    {
        // @TODO: implement via ITensorData.SharedAccess(), public float[] ToReadOnlyArray(ref int arrayOffset)
        PrepareCacheForAccess();
        return m_Cache;
    }
    #endregion

    public ITensorData tensorOnDevice { get { return m_TensorOnDevice; } }
    public ITensorData data
    {
        get
        {
            if (m_TensorOnDevice == null)
                UploadToDevice(new ArrayTensorData(shape));
            return m_TensorOnDevice;
        }
    }

    public override string ToString()
    {
        return $"(`{name}` {shape}, alloc: {m_TensorAllocator?.GetType()}, onDevice:{m_TensorOnDevice})";
    }

    #region Obsolete
    private bool m_Disposing = false;    // to protect from infinite-loop. in case UnpinAndDisposeTensor() is called from Dispose()
    [ObsoleteAttribute("Use Dispose instead.", false)]
    public ITensorData UnpinAndDisposeTensor()
    {
        // NOTE: since this Tensor is going to be Disposed
        // there is no need to populate cache with data from tensorOnDevice
        // we can save on skipping PrepareCacheForAccess() call
        ITensorData unpinned = tensorOnDevice;
        PinToDevice(null, false);
        if (!m_Disposing)
            Dispose();
        return unpinned;
    }

    [ObsoleteAttribute("Use ToReadOnlyArray instead.", false)]
    public float[] readonlyArray { get { PrepareCacheForAccess(); return m_Cache; } }
    [ObsoleteAttribute("Use ToReadOnlyArray instead.", false)]
    public int readonlyArrayOffset { get { return 0; } }
    #endregion

}

} // namespace Barracuda
