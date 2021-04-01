using UnityEngine.Assertions;
using System;
using System.Runtime.InteropServices;
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
    /// Max rank
    /// </summary>
    public const int MaxRank = 8;

    // The following dimension names are based on ONNX Dimension Denotation.
    // see: https://github.com/onnx/onnx/blob/master/docs/DimensionDenotation.md

    /// <summary>
    /// Data channel dimension index number
    /// </summary>
    public const int DataChannel  = 7;
    /// <summary>
    /// Channels dimension index number
    /// </summary>
    public const int C = DataChannel;

    /// <summary>
    /// Data feature 0 dimension index number
    /// </summary>
    public const int DataFeature0 = 6;
    /// <summary>
    /// Width dimension index number
    /// </summary>
    public const int W = DataFeature0;

    /// <summary>
    /// Data feature 1 dimension index number
    /// </summary>
    public const int DataFeature1 = 5;
    /// <summary>
    /// Height dimension index number
    /// </summary>
    public const int H = DataFeature1;

    /// <summary>
    /// Data feature 2 dimension index number
    /// </summary>
    public const int DataFeature2 = 4;
    /// <summary>
    /// Depth dimension index number
    /// </summary>
    public const int D = DataFeature2;

    /// <summary>
    /// Data feature 3 dimension index number
    /// </summary>
    public const int DataFeature3 = 3;
    /// <summary>
    /// Batch dimension index number
    /// </summary>
    public const int DataBatch = 2;

    /// <summary>
    /// Sequence length dimension index number
    /// </summary>
    public const int NumberOfDirections = 1;

    /// <summary>
    /// Sequence length dimension index number
    /// </summary>
    public const int SequenceLength = 0;

    /// <summary>
    /// Data features
    /// </summary>
    public readonly static int[] DataFeatures = new int[] { W, H, D, DataFeature3 };

    /// <summary>
    /// Kernel input channel dimension
    /// </summary>
    public const int KernelInChannel = 6;

    /// <summary>
    /// Kernel output channel dimension
    /// </summary>
    public const int KernelOutChannel = 7;

    /// <summary>
    /// Kernel spatial dimension 0
    /// </summary>
    public const int KernelSpatial0 = 5;

    /// <summary>
    /// Kernel spatial dimension 1
    /// </summary>
    public const int KernelSpatial1 = DataBatch;    // NOTE: maps to batch

    /// <summary>
    /// Kernel spatial dimension 2
    /// </summary>
    public const int KernelSpatial2 = DataBatch-1;  // NOTE: maps to numDirections

    /// <summary>
    /// Kernel spatial dimension 3
    /// </summary>
    public const int KernelSpatial3 = SequenceLength;  // NOTE: maps to sequenceLength

    /// <summary>
    /// Kernel spatial dimensions
    /// </summary>
    public readonly static int[] KernelSpatials = new int[] { KernelSpatial0, KernelSpatial1, KernelSpatial2, KernelSpatial3 };

    /// <summary>
    /// Return the number of sequence.
    /// </summary>
    public readonly int sequenceLength;
    /// <summary>
    /// Return the number of direction.
    /// </summary>
    public readonly int numberOfDirections;
    /// <summary>
    /// Return the number of batch.
    /// </summary>
    public readonly int batch;
    /// <summary>
    /// Return the size of 3rd spatial dimension (axis is DataFeature3)
    /// Internal for now, please use myTensorShape[DataFeature3] instead.
    /// </summary>
    internal readonly int extraDimension;
    /// <summary>
    /// Return the spatial depth (axis is DataFeature2).
    /// </summary>
    public readonly int depth;
    /// <summary>
    /// Return the spatial height (axis is DataFeature1).
    /// </summary>
    public readonly int height;
    /// <summary>
    /// Return the spatial width (axis is DataFeature0).
    /// </summary>
    public readonly int width;
    /// <summary>
    /// Return the number of channels.
    /// </summary>
    public readonly int channels;

    #region Constructors
    /// <summary>
    /// Create a TensorShape of shape [S,R,N,T,D,H,W,C].
    /// Currently seqLen must be 1.
    /// </summary>
    /// <param name="s">sequence</param>
    /// <param name="r">direction</param>
    /// <param name="n">batch</param>
    /// <param name="t">time</param>
    /// <param name="d">depth</param>
    /// <param name="h">height</param>
    /// <param name="w">width</param>
    /// <param name="c">channels</param>
    public TensorShape(int s, int r, int n, int t, int d, int h, int w, int c)
    {
        sequenceLength = s > 0 ? s : 1;
        numberOfDirections = r > 0 ? r : 1;
        batch = n > 0 ? n : 1;
        extraDimension = t > 0 ? t : 1;
        depth = d > 0 ? d : 1;
        height = h > 0 ? h : 1;
        width = w > 0 ? w : 1;
        channels = c > 0 ? c : 1;
    }

    /// <summary>
    /// Create a TensorShape of shape [1,1,N,1,D,H,W,C].
    /// </summary>
    /// <param name="n">batch</param>
    /// <param name="d">depth</param>
    /// <param name="h">height</param>
    /// <param name="w">width</param>
    /// <param name="c">channels</param>
    public TensorShape(int n, int d, int h, int w, int c)
    {
        sequenceLength = 1;
        numberOfDirections = 1;
        batch = n > 0 ? n : 1;
        extraDimension = 1;
        depth = d > 0 ? d : 1;
        height = h > 0 ? h : 1;
        width = w > 0 ? w : 1;
        channels = c > 0 ? c : 1;
    }

    /// <summary>
    /// Create a TensorShape of shape [1,1,N,1,1,H,W,C].
    /// </summary>
    /// <param name="n">batch</param>
    /// <param name="h">height</param>
    /// <param name="w">width</param>
    /// <param name="c">channels</param>
    public TensorShape(int n, int h, int w, int c)
    {
        sequenceLength = 1;
        numberOfDirections = 1;
        batch = n > 0 ? n : 1;
        extraDimension = 1;
        depth = 1;
        height = h > 0 ? h : 1;
        width = w > 0 ? w : 1;
        channels = c > 0 ? c : 1;
    }

    /// <summary>
    /// Create a TensorShape of shape [1,1,N,1,1,1,W,C].
    /// </summary>
    /// <param name="n">batch</param>
    /// <param name="w">width</param>
    /// <param name="c">channels</param>
    public TensorShape(int n, int w, int c)
    {
        sequenceLength = 1;
        numberOfDirections = 1;
        batch = n > 0 ? n : 1;
        extraDimension = 1;
        depth = 1;
        height = 1;
        width = w > 0 ? w : 1;
        channels = c > 0 ? c : 1;
    }
    /// <summary>
    /// Create a TensorShape of shape [1,1,N,1,1,1,1,C].
    /// </summary>
    /// <param name="n">batch</param>
    /// <param name="c">channels</param>
    public TensorShape(int n, int c)
    {
        sequenceLength = 1;
        numberOfDirections = 1;
        batch = n > 0 ? n : 1;
        extraDimension = 1;
        depth = 1;
        height = 1;
        width = 1;
        channels = c > 0 ? c : 1;
    }

    /// <summary>
    /// Create a TensorShape of shape [1,1,N,1,1,1,1,1].
    /// </summary>
    /// <param name="n">batch</param>
    public TensorShape(int n)
    {
        sequenceLength = 1;
        numberOfDirections = 1;
        batch = n > 0 ? n : 1;
        extraDimension = 1;
        depth = 1;
        height = 1;
        width = 1;
        channels = 1;
    }
    /// <summary>
    /// Create a TensorShape of arbitrary `shape`.
    /// `shape` must be of length 4 [N,H,W,C] or 8 [S,R,N,T,D,H,W,C].
    /// If `shape.length` is 4 then the dimensions S,R,T and D will be defaulted to 1.
    /// </summary>
    /// <param name="shape">shape as int array</param>
    public TensorShape(int[] shape)
        : this( shape.Length > 4 ? shape[0] : 1,
                shape.Length > 4 ? shape[1] : 1,
                shape.Length > 4 ? shape[2] : shape[0],
                shape.Length > 4 ? shape[3] : 1,
                shape.Length > 4 ? shape[4] : 1,
                shape.Length > 4 ? shape[5] : (shape.Length > 1?shape[1]:1),
                shape.Length > 4 ? shape[6] : (shape.Length > 2?shape[2]:1),
                shape.Length > 4 ? shape[7] : (shape.Length > 3?shape[3]:1))
    {
    }
    #endregion

    #region Properties
    /// <summary>
    /// Kernel dimension ordering is [D,H,W,C,K] for efficiency purpose.
    /// Return kernel intermediate dimension 0.
    /// </summary>
    public int kernelSpatialDepth { get { return numberOfDirections; } }
    /// <summary>
    /// Kernel dimension ordering is [D,H,W,C,K] for efficiency purpose.
    /// Return kernel height.
    /// </summary>
    public int kernelHeight { get { return batch; } }//Use .batch so HWCK weight use 4D constructor for backward compatibility with 4D tensorShape.
    /// <summary>
    /// Kernel dimension ordering is [D,H,W,C,K] for efficiency purpose.
    /// Return kernel width.
    /// </summary>
    public int kernelWidth { get { return height; } }
    /// <summary>
    /// Kernel dimension ordering is [D,H,W,C,K] for efficiency purpose.
    /// Return kernel depth (aka the number of input channels of the associated operator).
    /// </summary>
    public int kernelDepth { get { return width; } }
    /// <summary>
    /// Kernel dimension ordering is [D,H,W,C,K] for efficiency purpose.
    /// Return kernel count (aka the number of output channels of the associated operator).
    /// </summary>
    public int kernelCount { get { return channels; } }
    /// <summary>
    /// Return the number of batch.
    /// </summary>
    public int flatHeight { get { return batch; } }
    /// <summary>
    /// Return the T*D*H*W*C.
    /// </summary>
    public int flatWidth { get { return extraDimension * depth * height * width * channels; } }
    /// <summary>
    /// Return the total number of elements represented by this shape.
    /// </summary>
    public int length { get { return sequenceLength * numberOfDirections *flatHeight * flatWidth; } }
    /// <summary>
    /// Always 8, look also at the `dimensions` property.
    /// </summary>
    public int rank { get { return 8; } }
    /// <summary>
    /// Return the count of non-unit dimension of this shape.
    /// For example [N,1,1,C] dimensions is 2.
    /// </summary>
    public int dimensions { get {
            return
                (sequenceLength > 1 ? 1 : 0) +
                (numberOfDirections > 1 ? 1 : 0) +
                (batch > 1 ? 1 : 0) +
                (extraDimension > 1 ? 1 : 0) +
                (depth > 1 ? 1 : 0) +
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
    /// <param name="axis">axis</param>
    /// <returns>adjusted axis</returns>
    public int Axis(int axis)
    {
        Assert.IsTrue(axis > -rank && axis < rank);
        return axis >= 0 ? axis: rank + axis;
    }

    /// <summary>
    /// Given an offset in memory return the dimensions indices of the element as [_,_,N,_,_,H,W,C].
    /// </summary>
    /// <param name="index">one dimensional index (offset) in the memory</param>
    /// <param name="n">batch</param>
    /// <param name="h">height</param>
    /// <param name="w">width</param>
    /// <param name="c">channels</param>
    public void GetPositionsFromIndex(int index, ref int n, ref int h, ref int w, ref int c)
    {
        c = index % channels;
        w = (index / channels) % width;
        h = (index / (channels * width)) % height;
        n = (index / (channels * width * height * depth * extraDimension)) % batch;
    }

    /// <summary>
    /// Given an offset in memory return the dimensions indices of the element as [S,R,N,T,D,H,W,C].
    /// </summary>
    /// <param name="index">one dimensional index (offset) in the memory</param>
    /// <param name="s">sequence</param>
    /// <param name="r">direction</param>
    /// <param name="n">batch</param>
    /// <param name="t">time</param>
    /// <param name="d">depth</param>
    /// <param name="h">height</param>
    /// <param name="w">width</param>
    /// <param name="c">channels</param>
    public void GetPositionsFromIndex(int index, ref int s, ref int r, ref int n, ref int t, ref int d, ref int h, ref int w, ref int c)
    {
        c =  index % channels;
        w = (index / channels) % width;
        h = (index / (channels * width)) % height;
        d = (index / (channels * width * height)) % depth;
        t = (index / (channels * width * height * depth)) % extraDimension;
        n = (index / (channels * width * height * depth * extraDimension)) % batch;
        r = (index / (channels * width * height * depth * extraDimension * batch)) % numberOfDirections;
        s = (index / (channels * width * height * depth * extraDimension * batch * numberOfDirections)) % sequenceLength;
    }

    /// <summary>
    /// Given an offset in memory return the dimensions indices of the element as [S,R,N,T,D,H,W,C] in ChannelFirst memory layout.
    /// </summary>
    /// <param name="index">one dimensional index (offset) in the memory</param>
    /// <param name="s">sequence</param>
    /// <param name="r">direction</param>
    /// <param name="n">batch</param>
    /// <param name="t">time</param>
    /// <param name="d">depth</param>
    /// <param name="h">height</param>
    /// <param name="w">width</param>
    /// <param name="c">channels</param>
    internal void GetPositionsFromIndexChannelFirst(int index, ref int s, ref int r, ref int n, ref int t, ref int d, ref int h, ref int w, ref int c)
    {
        w  = index % width;
        h  = (index / width) % height;
        d  = (index / (width * height)) % depth;
        t  = (index / (width * height * depth)) % extraDimension;
        c  = (index / (width * height * depth * extraDimension)) % channels;
        n  = (index / (width * height * depth * extraDimension * channels)) % batch;
        r  = (index / (width * height * depth * extraDimension * channels * batch)) % numberOfDirections;
        s  = (index / (width * height * depth * extraDimension * channels * batch * numberOfDirections)) % sequenceLength;
    }

    /// <summary>
    /// Given an offset in memory return the dimensions indices of the element as [_,_,N,_,_,H,W,C] in ChannelFirst format.
    /// </summary>
    /// <param name="index">one dimensional index (offset) in the memory</param>
    /// <param name="n">batch</param>
    /// <param name="h">height</param>
    /// <param name="w">width</param>
    /// <param name="c">channels</param>
    internal void GetPositionsFromIndexChannelFirst(int index, ref int n, ref int h, ref int w, ref int c)
    {
        w = index % width;
        h = (index / width) % height;
        c  = (index / (width * height * depth * extraDimension)) % channels;
        n  = (index / (width * height * depth * extraDimension * channels)) % batch;
    }

    /// <summary>
    /// Given an element dimensions indices [0,0,N,0,0,H,W,C] with broadcast support, return this element offset in memory.
    /// </summary>
    /// <param name="n">batch</param>
    /// <param name="h">height</param>
    /// <param name="w">width</param>
    /// <param name="c">channels</param>
    /// <returns></returns>
    public int IndexWithBroadcast(int n, int h, int w, int c)
    {
        n %= batch;
        h %= height;
        w %= width;
        c %= channels;
        return Index(n, h, w, c);
    }

    /// <summary>
    /// Given an element dimensions indices [S,R,N,T,D,H,W,C] with broadcast support, return this element offset in memory.
    /// </summary>
    /// <param name="s">sequence</param>
    /// <param name="r">direction</param>
    /// <param name="n">batch</param>
    /// <param name="t">time</param>
    /// <param name="d">depth</param>
    /// <param name="h">height</param>
    /// <param name="w">width</param>
    /// <param name="c">channels</param>
    /// <returns>one dimensional index (offset in the flat memory region)</returns>
    public int IndexWithBroadcast(int s, int r, int n, int t, int d, int h, int w, int c)
    {
        s %= sequenceLength;
        r %= numberOfDirections;
        n %= batch;
        t %= extraDimension;
        d %= depth;
        h %= height;
        w %= width;
        c %= channels;
        return Index(s, r, n, t, d, h, w, c);
    }

    /// <summary>
    /// Given an element dimensions indices [1,N,1,1,1,H,W,C] return this element offset in memory, clamping indices to tensor dimensions.
    /// </summary>
    /// <param name="n">batch</param>
    /// <param name="h">height</param>
    /// <param name="w">width</param>
    /// <param name="c">channels</param>
    /// <returns>one dimensional index (offset in the flat memory region)</returns>
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
    /// Given an element dimensions indices [1,N,1,1,D,H,W,C] return this element offset in memory, clamping indices to tensor dimensions.
    /// </summary>
    /// <param name="n">batch</param>
    /// <param name="d">depth</param>
    /// <param name="h">height</param>
    /// <param name="w">width</param>
    /// <param name="c">channels</param>
    /// <returns>one dimensional index (offset in the flat memory region)</returns>
    public int IndexWithClamp(int n, int d, int h, int w, int c)
    {
        n = Math.Max(n, 0);
        d = Math.Max(d, 0);
        h = Math.Max(h, 0);
        w = Math.Max(w, 0);
        c = Math.Max(c, 0);
        n = Math.Min(n, batch - 1);
        d = Math.Min(d, depth - 1);
        h = Math.Min(h, height - 1);
        w = Math.Min(w, width - 1);
        c = Math.Min(c, channels - 1);
        return Index(n, d, h, w, c);
    }

    /// <summary>
    /// Given an element dimensions indices [S,R,N,T,D,H,W,C] return this element offset in memory, clamping indices to tensor dimensions.
    /// </summary>
    /// <param name="s">sequence</param>
    /// <param name="r">direction</param>
    /// <param name="n">batch</param>
    /// <param name="t">time</param>
    /// <param name="d">depth</param>
    /// <param name="h">height</param>
    /// <param name="w">width</param>
    /// <param name="c">channels</param>
    /// <returns>one dimensional index (offset in the flat memory region)</returns>
    public int IndexWithClamp(int s, int r, int n, int t, int d, int h, int w, int c)
    {
        s = Math.Max(s, 0);
        r = Math.Max(r, 0);
        n = Math.Max(n, 0);
        t = Math.Max(t, 0);
        d = Math.Max(d, 0);
        h = Math.Max(h, 0);
        w = Math.Max(w, 0);
        c = Math.Max(c, 0);
        s = Math.Min(s, sequenceLength - 1);
        r = Math.Min(r, numberOfDirections - 1);
        n = Math.Min(n, batch - 1);
        t = Math.Min(t, extraDimension - 1);
        d = Math.Min(d, depth - 1);
        h = Math.Min(h, height - 1);
        w = Math.Min(w, width - 1);
        c = Math.Min(c, channels - 1);
        return Index(s,r,n,t,d,h,w,c);
    }

    /// <summary>
    /// Given an element dimensions indices [S,R,N,T,D,H,W,C] return this element offset in memory.
    /// </summary>
    /// <param name="s">sequence</param>
    /// <param name="r">direction</param>
    /// <param name="n">batch</param>
    /// <param name="t">time</param>
    /// <param name="d">depth</param>
    /// <param name="h">height</param>
    /// <param name="w">width</param>
    /// <param name="c">channels</param>
    /// <returns>one dimensional index (offset in the flat memory region)</returns>
    public int Index(int s, int r, int n, int t, int d, int h, int w, int c)
    {
        int index =
            s * numberOfDirections * batch * extraDimension * depth * height * width * channels +
            r * batch * extraDimension * depth * height * width * channels +
            n * extraDimension * depth * height * width * channels +
            t * depth * height * width * channels +
            d * height * width * channels +
            h * width * channels +
            w * channels +
            c;
        return index;
    }

    /// <summary>
    /// Given an element dimensions indices [0,0,N,0,D,H,W,C] return this element offset in memory.
    /// </summary>
    /// <param name="n">batch</param>
    /// <param name="d">depth</param>
    /// <param name="h">height</param>
    /// <param name="w">width</param>
    /// <param name="c">channels</param>
    /// <returns>one dimensional index (offset in the flat memory region)</returns>
    public int Index(int n, int d, int h, int w, int c)
    {
        int index =
            n * extraDimension * depth * height * width * channels +
            d * height * width * channels +
            h * width * channels +
            w * channels +
            c;
        return index;
    }

    /// <summary>
    /// Given an element dimensions indices [0,0,N,0,0,H,W,C] return this element offset in memory.
    /// </summary>
    /// <param name="n">batch</param>
    /// <param name="h">height</param>
    /// <param name="w">width</param>
    /// <param name="c">channels</param>
    /// <returns>one dimensional index (offset in the flat memory region)</returns>
    public int Index(int n, int h, int w, int c)
    {
        int index =
            n * extraDimension * depth * height * width * channels +
            h * width * channels +
            w * channels +
            c;
        return index;
    }

    /// <summary>
    /// Given an element dimensions indices [S,R,N,T,D,H,W,C] return this element offset in memory in ChannelFirst format.
    /// </summary>
    /// <param name="s">sequence</param>
    /// <param name="r">direction</param>
    /// <param name="n">batch</param>
    /// <param name="t">time</param>
    /// <param name="d">depth</param>
    /// <param name="h">height</param>
    /// <param name="w">width</param>
    /// <param name="c">channels</param>
    /// <returns>one dimensional index (offset in the flat memory region)</returns>
    internal int IndexChannelFirst(int s, int r, int n, int t, int d, int h, int w, int c)
    {
        int index =
            s * numberOfDirections * batch * channels * extraDimension * depth * height * width +
            r * batch * channels * extraDimension * depth * height * width +
            n * channels * extraDimension * depth * height * width +
            c * extraDimension * depth * height * width +
            t * depth * height * width +
            d * height * width +
            h * width +
            w;
        return index;
    }

    /// <summary>
    /// Given an element dimensions indices [0,0,N,0,0,H,W,C] return this element offset in memory in ChannelFirst format.
    /// </summary>
    /// <param name="n">batch</param>
    /// <param name="h">height</param>
    /// <param name="w">width</param>
    /// <param name="c">channels</param>
    /// <returns>one dimensional index (offset in the flat memory region)</returns>
    internal int IndexChannelFirst(int n, int h, int w, int c)
    {
        int index =
            n * channels * extraDimension * depth * height * width +
            c * extraDimension * depth * height * width +
            h * width +
            w;
        return index;
    }

    /// <summary>
    /// Given an element dimensions indices [0,0,N,0,0,0,0,C] return this element offset in memory.
    /// </summary>
    /// <param name="n">batch</param>
    /// <param name="c">channels</param>
    /// <returns>one dimensional index (offset in the flat memory region)</returns>
    public int Index(int n, int c)
    {
        int index =
            n * flatWidth +
            c;
        return index;
    }

    /// <summary>
    /// Indexer to return a dimension of this tensorShape as [S,R,N,T,D,H,W,C]
    /// Prefer this over ToArray() to avoid GC allocation/collection.
    /// </summary>
    /// <param name="axis">axis</param>
    public int this[int axis]
    {
        get
        {
            // switch case instead of ToArray() avoids GC allocation
            switch(axis)
            {
                case 0:
                    return sequenceLength;
                case 1:
                    return numberOfDirections;
                case 2:
                    return batch;
                case 3:
                    return extraDimension;
                case 4:
                    return depth;
                case 5:
                    return height;
                case 6:
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

                    if (axis < 0 || axis > 7)
                        axis = 7;

                    p[axis] = value > 0 ? value : 1;
                }
            }
        }
    }

    /// <summary>
    /// Return an array representation of this tensorShape as [S,R,N,T,D,H,W,C]
    /// Prefer tensorShape[x] to avoid GC allocation/collection.
    /// </summary>
    /// <returns>shape as int array</returns>
    public int[] ToArray()
    {
        return new[] { sequenceLength, numberOfDirections, batch, extraDimension, depth, height, width, channels };
    }

    /// <summary>
    /// Remove single-dimensional entries from the shape.
    /// [s=1,r=1,b=4,t=1,d=1h=1,w=1,c=128] => [s=1,r=1,b=1,t=1,d=1,h=1,w=4,c=128]
    /// </summary>
    /// <returns>new TensorShape</returns>
    public TensorShape Squeeze()
    {
        var dims = ToArray();

        var squeezed = new TensorShape( 1,1,1,1,1,1,1,1 );
        Assert.IsTrue(dims.Length == squeezed.rank);
        var index = squeezed.rank;
        foreach (var dim in dims)
            if (dim > 1)
                squeezed[--index] = dim;
        return squeezed;
    }

    /// <summary>
    /// Return a TensorShape of dimensions [S,R,N,1,1,1,1,T*D*H*W*C]
    /// </summary>
    /// <returns>new TensorShape</returns>
    public TensorShape Flatten()
    {
        return new TensorShape(sequenceLength, numberOfDirections, batch, 1, 1, 1, 1, flatWidth);
    }
    #endregion

    #region Comparison operators
    /// <summary>
    /// Compares two `TensorShape` objects
    /// </summary>
    /// <param name="a">left object</param>
    /// <param name="b">right object</param>
    /// <returns>`true` if contents of the objects `a` and `b` are equal</returns>
    public static bool operator ==(TensorShape a, TensorShape b)
    {
        for (var i = 0; i < TensorShape.MaxRank; ++i)
            if (a[i] != b[i])
                return false;
        return true;
    }

    /// <summary>
    /// Compares two `TensorShape` objects
    /// </summary>
    /// <param name="a">left object</param>
    /// <param name="b">right object</param>
    /// <returns>`true` if contents of the objects `a` and `b` are not equal</returns>
    public static bool operator !=(TensorShape a, TensorShape b)
    {
        return !(a == b);
    }

    /// <summary>
    /// Compares `this` object to other object
    /// </summary>
    /// <param name="obj">other object</param>
    /// <returns>`true` if contents of the objects `a` and `b` are equal</returns>
    public override bool Equals(System.Object obj)
    {
        // Check for null values and compare run-time types.
        if (obj == null || GetType() != obj.GetType())
            return false;

        return this == (TensorShape)obj;
    }

    /// <summary>
    /// Object hash code
    /// </summary>
    /// <returns>object hash code</returns>
    public override int GetHashCode()
    {
        return sequenceLength ^ numberOfDirections ^ batch ^ extraDimension ^ depth ^ height ^ width ^ channels;
    }
    #endregion

    /// <summary>
    /// Object summary
    /// </summary>
    /// <returns>object summary as a string</returns>
    public override string ToString()
    {
        if (this.Is4D())
            return $"(n:{batch}, h:{height}, w:{width}, c:{channels})";
        else
            return $"(s:{sequenceLength}, r:{numberOfDirections}, n:{batch}, t:{extraDimension}, d:{depth}, h:{height}, w:{width}, c:{channels})";
    }
}
/// <summary>
/// Helper structure to iterate over tensor shape
/// </summary>
public struct TensorIterator
{
    /// <summary>
    /// Tensor shape
    /// </summary>
    public readonly TensorShape shape;
    private readonly int m_shapeLength;

    /// <summary>
    /// Index
    /// </summary>
    public int index;

    /// <summary>
    /// dimension 0
    /// </summary>
    public int d0;

    /// <summary>
    /// dimension 1
    /// </summary>
    public int d1;

    /// <summary>
    /// dimension 2
    /// </summary>
    public int d2;

    /// <summary>
    /// dimension 3
    /// </summary>
    public int d3;

    /// <summary>
    /// dimension 4
    /// </summary>
    public int d4;

    /// <summary>
    /// dimension 5
    /// </summary>
    public int d5;

    /// <summary>
    /// dimension 6
    /// </summary>
    public int d6;

    /// <summary>
    /// dimension 7
    /// </summary>
    public int d7;

    /// <summary>
    /// Constructs Tensor shape iterator
    /// </summary>
    /// <param name="shape">shape</param>
    /// <param name="index">starting index</param>
    public TensorIterator(TensorShape shape, int index = 0)
    {
        this.shape = shape;
        m_shapeLength = shape.length;
        this.index = index;
        d0 = 0; d1 = 0; d2 = 0; d3 = 0; d4 = 0; d5 = 0; d6 = 0; d7 = 0;
        AssignIndexAndInvalidateDimensions(index);
    }

    /// <summary>
    /// Constructs Tensor shape iterator
    /// </summary>
    /// <param name="tensor">Tensor</param>
    /// <param name="index">starting index</param>
    public TensorIterator(Tensor tensor, int index = 0) : this(tensor.shape, index)
    {
    }

    internal void AssignIndexAndInvalidateDimensions(int index)
    {
        this.index = index;
        d0 = 0; d1 = 0; d2 = 0; d3 = 0; d4 = 0; d5 = 0; d6 = 0; d7 = 0;
        if (index != 0)
            shape.GetPositionsFromIndex(index,
                ref d0, ref d1, ref d2, ref d3, ref d4, ref d5, ref d6, ref d7);
    }

    /// <summary>
    /// Next element in the Tensor shape space
    /// </summary>
    public void Next()
    {
        ++index;
        ++d7;
        // carry-over chain
        if (d7 < shape[7]) return; d7 = 0; ++d6;
        if (d6 < shape[6]) return; d6 = 0; ++d5;
        if (d5 < shape[5]) return; d5 = 0; ++d4;
        if (d4 < shape[4]) return; d4 = 0; ++d3;
        if (d3 < shape[3]) return; d3 = 0; ++d2;
        if (d2 < shape[2]) return; d2 = 0; ++d1;
        if (d1 < shape[1]) return; d1 = 0; ++d0;
    }

    /// <summary>
    /// Advance iterator by `step`
    /// </summary>
    /// <param name="step">step count</param>
    public void Advance(int step)
    {
        index += step;
        d7 += step;
        Assert.IsTrue(index >= 0);
        if (d7 >= shape[7] * 2 || d7 < 0)
        { // step is too large and would overflow the carry-over into the next dimension
          // or step is negative and would require a borrow from the next dimension
            AssignIndexAndInvalidateDimensions(index);
            return;
        }

        // carry-over chain
        if (d7 < shape[7]) return; d7 -= shape[7]; Assert.IsTrue(d7 < shape[7]); ++d6;
        if (d6 < shape[6]) return; d6 = 0; ++d5;
        if (d5 < shape[5]) return; d5 = 0; ++d4;
        if (d4 < shape[4]) return; d4 = 0; ++d3;
        if (d3 < shape[3]) return; d3 = 0; ++d2;
        if (d2 < shape[2]) return; d2 = 0; ++d1;
        if (d1 < shape[1]) return; d1 = 0; ++d0;
    }

    /// <summary>
    /// Is iterator in valid state
    /// </summary>
    /// <returns>`true` if iterator is still within shape</returns>
    public bool IsValid()
    {
        return index < m_shapeLength;
    }

    /// <summary>
    /// Index in reduced shape
    /// </summary>
    /// <param name="reducedShape">reduced shape</param>
    /// <returns>index</returns>
    public int IndexInReducedShape(TensorShape reducedShape)
    {
        int rd0 = Math.Min(d0, reducedShape[0]-1);
        int rd1 = Math.Min(d1, reducedShape[1]-1);
        int rd2 = Math.Min(d2, reducedShape[2]-1);
        int rd3 = Math.Min(d3, reducedShape[3]-1);
        int rd4 = Math.Min(d4, reducedShape[4]-1);
        int rd5 = Math.Min(d5, reducedShape[5]-1);
        int rd6 = Math.Min(d6, reducedShape[6]-1);
        int rd7 = Math.Min(d7, reducedShape[7]-1);
        return reducedShape.Index(rd0, rd1, rd2, rd3, rd4, rd5, rd6, rd7);
    }

    /// <summary>
    /// Index with replaced `axis` value
    /// </summary>
    /// <param name="axis">axis to replace</param>
    /// <param name="newDimensionValue">new value for specific axis</param>
    /// <returns>index</returns>
    public int IndexWithReplacedAxis(int axis, int newDimensionValue)
    {
        int nd0 = axis == 0 ? newDimensionValue : d0;
        int nd1 = axis == 1 ? newDimensionValue : d1;
        int nd2 = axis == 2 ? newDimensionValue : d2;
        int nd3 = axis == 3 ? newDimensionValue : d3;
        int nd4 = axis == 4 ? newDimensionValue : d4;
        int nd5 = axis == 5 ? newDimensionValue : d5;
        int nd6 = axis == 6 ? newDimensionValue : d6;
        int nd7 = axis == 7 ? newDimensionValue : d7;
        return shape.Index(nd0, nd1, nd2, nd3, nd4, nd5, nd6, nd7);
    }

    /// <summary>
    /// Access specific axis value
    /// </summary>
    /// <param name="axis">axis</param>
    public int this[int axis]
    {
        get
        {
            // switch case instead of ToArray() avoids GC allocation
            switch(axis)
            {
                case 0: return d0;
                case 1: return d1;
                case 2: return d2;
                case 3: return d3;
                case 4: return d4;
                case 5: return d5;
                case 6: return d6;
                default:return d7;
            }
        }
    }
}


// @TODO: most likely Tensor should still be struct - that way passing Tensor as argument into IOps would be safer (no hidden state mods), and Flatten & Reshape could return modified Tensor
// ITensorData & Dispose mechanism should however allow Tensors to share the same ITensorData
/// <summary>
/// Multidimensional array-like data storage
/// </summary>
public class Tensor : IDisposable
{
    private ITensorData m_TensorOnDevice;
    private ITensorAllocator m_TensorAllocator;
    private float[] m_Cache;
    private bool m_CacheIsDirty;
    private bool m_Disposed = false;

    public static event Action<Tensor> tensorDisposed;

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
    /// Return this tensor shape as [S,R,N,T,D,H,W,C].
    /// </summary>
    public readonly TensorShape shape;
    /// <summary>
    /// Return the number of sequences.
    /// </summary>
    public int sequenceLength { get { return shape.sequenceLength; } }
    /// <summary>
    /// Return the number of directions.
    /// </summary>
    public int numberOfDirections { get { return shape.numberOfDirections; } }
    /// <summary>
    /// Return the number of batches.
    /// </summary>
    public int batch { get { return shape.batch; } }
    /// <summary>
    /// Return the size of 3rd spatial dimension (axis is DataFeature3)
    /// Internal for now, please use myTensor.shape[DataFeature3] instead.
    /// </summary>
    internal int extraDimension { get { return shape.extraDimension; } }
    /// <summary>
    /// Return the spatial depth.
    /// </summary>
    public int depth { get { return shape.depth; } }
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
    /// Kernel dimension ordering is [D,H,W,C,K] for efficiency purpose.
    /// Return kernel spatial depth.
    /// </summary>
    public int kernelSpatialDepth { get { return shape.kernelSpatialDepth; } }
    /// <summary>
    /// Kernel dimension ordering is [D,H,W,C,K] for efficiency purpose.
    /// Return kernel spatial width.
    /// </summary>
    public int kernelWidth { get { return shape.kernelWidth; } }
    /// <summary>
    /// Kernel dimension ordering is [D,H,W,C,K] for efficiency purpose.
    /// Return kernel spatial height.
    /// </summary>
    public int kernelHeight { get { return shape.kernelHeight; } }
    /// <summary>
    /// Kernel dimension ordering is [D,H,W,C,K] for efficiency purpose.
    /// Return kernel depth (aka the number of input channels of the associated operator).
    /// </summary>
    public int kernelDepth { get { return shape.kernelDepth; } }
    /// <summary>
    /// Kernel dimension ordering is [D,H,W,C,K] for efficiency purpose.
    /// Return kernel count (aka the number of output channels of the associated operator).
    /// </summary>
    public int kernelCount { get { return shape.kernelCount; } }
    /// <summary>
    /// Return the number of batch.
    /// </summary>
    public int flatHeight { get { return shape.flatHeight; } }
    /// <summary>
    /// Return T*D*H*W*C.
    /// </summary>
    public int flatWidth { get { return shape.flatWidth; } }
    /// <summary>
    /// Return the total number of elements in this tensor.
    /// </summary>
    public int length { get { return shape.length; } }
    /// <summary>
    /// Return the count of non-unit dimension of this tensor shape.
    /// For example [1,1,N,1,1,1,1,C] dimensions is 2.
    /// </summary>
    public int dimensions { get { return shape.dimensions; } }
    #endregion

    #region Constructors
    /// <summary>
    /// Create a Tensor from a `shape`, an array of data `srcData` and an optional debug `name`.
    /// `shape` must be of size 8, the order is [S,R,N,T,D,H,W,C].
    /// S and R must be 1.
    /// `srcData` must be of size `s[0]*s[1]*s[2]*s[3]*s[4]*s[5]*s[6]*s[7]`.
    /// </summary>
    /// <param name="shape">shape</param>
    /// <param name="srcData">source data</param>
    /// <param name="name">name</param>
    public Tensor(int[] shape, float[] srcData, string name = "") : this(new TensorShape(shape), srcData, name) {}

    /// <summary>
    /// Create a Tensor of shape [N,H,W,C], an array of data `srcData` and an optional debug `name`.
    /// `srcData` must be of size `n*h*w*c`.
    /// </summary>
    /// <param name="n">batch</param>
    /// <param name="h">height</param>
    /// <param name="w">width</param>
    /// <param name="c">channels</param>
    /// <param name="srcData">source data</param>
    /// <param name="name">name</param>
    public Tensor(int n, int h, int w, int c, float[] srcData, string name = "") : this(new TensorShape(n, h, w, c), srcData, name) {}

    /// <summary>
    /// Create a Tensor of shape [N,1,1,C], an array of data `srcData` and an optional debug `name`.
    /// `srcData` must be of size `n*c`.
    /// </summary>
    /// <param name="n">batch</param>
    /// <param name="c">channels</param>
    /// <param name="srcData">source data</param>
    /// <param name="name">name</param>
    public Tensor(int n, int c, float[] srcData, string name = "") : this(new TensorShape(n, c), srcData, name) {}

    /// <summary>
    /// Create a Tensor with specified `shape`, an array of data `srcData` and an optional debug `name`.
    /// `srcData` must be of size `shape.length`.
    /// </summary>
    /// <param name="shape">shape</param>
    /// <param name="srcData">source data</param>
    /// <param name="name">name</param>
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
    /// `shape` must be of size 8, the order is [S,R,N,T,D,H,W,C].
    /// S and R must be 1.
    /// `srcData` must be of size `s[0]*s[1]*s[2]*s[3]*s[4]*s[5]*s[6]*s[7]`.
    /// </summary>
    /// <param name="shape">shape</param>
    /// <param name="srcData">source data</param>
    /// <param name="name">name</param>
    public Tensor(int[] shape, float[][] srcData, string name = "") : this(new TensorShape(shape), srcData, name) {}

    /// <summary>
    /// Create a Tensor of shape [1,1,N,1,1,H,W,C], an array of data `srcData` and an optional debug `name`.
    /// `srcData` must be of size `n*h*w*c`.
    /// </summary>
    /// <param name="n">batch</param>
    /// <param name="h">height</param>
    /// <param name="w">width</param>
    /// <param name="c">channels</param>
    /// <param name="srcData">source data</param>
    /// <param name="name">name</param>
    public Tensor(int n, int h, int w, int c, float[][] srcData, string name = "") : this(new TensorShape(n, h, w, c), srcData, name) {}

    /// <summary>
    /// Create a Tensor of shape [1,1,N,1,1,1,1,C], an array of data `srcData` and an optional debug `name`.
    /// `srcData` must be of size `n*c`.
    /// </summary>
    /// <param name="n">batch</param>
    /// <param name="c">channels</param>
    /// <param name="srcData">source data</param>
    /// <param name="name">name</param>
    public Tensor(int n, int c, float[][] srcData, string name = "") : this(new TensorShape(n, c), srcData, name) {}

    /// <summary>
    /// Create a Tensor with specified `shape`, an array of data `srcData` and an optional debug `name`.
    /// `srcData` must be of size `shape.length`.
    /// </summary>
    /// <param name="shape">shape</param>
    /// <param name="srcData">source data</param>
    /// <param name="name">name</param>
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
    /// Create a Tensor from a `shape`, an array of data `srcData` and an optional name debug `name`.
    /// `shape` must be of size 8, the order is [S,R,N,T,D,H,W,C].
    /// S and R must be 1.
    /// `srcData` must be of size `s[0]*s[1]*s[2]*s[3]*s[4]*s[5]*s[6]*s[7]`.
    /// </summary>
    /// <param name="shape">shape</param>
    /// <param name="srcData">source data</param>
    /// <param name="name">name</param>
    public Tensor(int[] shape, float[,] srcData, string name = "") : this(new TensorShape(shape), srcData, name) {}

    /// <summary>
    /// Create a Tensor of shape [1,1,N,1,1,1,1,C], an array of data `srcData` and an optional debug `name`.
    /// `srcData` must be of size `n*c`.
    /// </summary>
    /// <param name="n">batch</param>
    /// <param name="c">channels</param>
    /// <param name="srcData">source data</param>
    /// <param name="name">name</param>
    public Tensor(int n, int c, float[,] srcData, string name = "") : this(new TensorShape(n, c), srcData, name) {}

    /// <summary>
    /// Create a Tensor with specified `shape`, an array of data `srcData` and an optional debug `name`.
    /// `srcData` must be of size `shape.length`.
    /// </summary>
    /// <param name="shape">shape</param>
    /// <param name="srcData">source data</param>
    /// <param name="name">name</param>
    public Tensor(TensorShape shape, float[,] srcData, string name = "") : this(shape, (Array)srcData, name) {}

    internal Tensor(TensorShape shape, Array srcData, string name = "")
    {
        this.name = name;
        this.shape = shape;
        var arrayTensorData = new ArrayTensorData(shape);

        Buffer.BlockCopy(srcData, 0, arrayTensorData.array, 0,
            Math.Min(shape.length, srcData.Length)*Marshal.SizeOf<float>());

        m_TensorOnDevice = arrayTensorData;
        m_TensorAllocator = null;
        m_Cache = null;
        m_CacheIsDirty = false;
    }

    /// <summary>
    /// Create a Tensor from a `shape`, an array of data `srcData` and an optional name debug `name`.
    /// `shape` must be of size 8, the order is [S,R,N,T,D,H,W,C].
    /// S and R must be 1.
    /// `srcData` must be of size `s[0]*s[1]*s[2]*s[3]*s[4]*s[5]*s[6]*s[7]`.
    /// </summary>
    /// <param name="shape">shape</param>
    /// <param name="srcData">source data</param>
    /// <param name="name">name</param>
    public Tensor(int[] shape, float[,,,] srcData, string name = "") : this(new TensorShape(shape), srcData, name) {}

    /// <summary>
    /// Create a Tensor of shape [1,1,N,1,1,H,W,C], an array of data `srcData` and an optional debug `name`.
    /// `srcData` must be of size `n*h*w*c`.
    /// </summary>
    /// <param name="n">batch</param>
    /// <param name="h">height</param>
    /// <param name="w">width</param>
    /// <param name="c">channels</param>
    /// <param name="srcData">source data</param>
    /// <param name="name">name</param>
    public Tensor(int n, int h, int w, int c, float[,,,] srcData, string name = "") : this(new TensorShape(n, h, w, c), srcData, name) {}

    /// <summary>
    /// Create a Tensor with specified `shape`, an array of data `srcData` and an optional debug `name`.
    /// `srcData` must be of size `shape.length`.
    /// </summary>
    /// <param name="shape">shape</param>
    /// <param name="srcData">source data</param>
    /// <param name="name">name</param>
    public Tensor(TensorShape shape, float[,,,] srcData, string name = "") : this(shape, (Array)srcData, name) {}

    /// <summary>
    /// Create a Tensor from a `shape`, associated ComputeBuffer `srcBuffer` filled with tensor values, and an optional debug `name`.
    /// `shape` must be of size 8, the order is [S,R,N,T,D,H,W,C].
    /// S and R must be 1.
    /// `srcBuffer` must be larger than `s[0]*s[1]*s[2]*s[3]*s[4]*s[5]*s[6]*s[7]`.
    /// </summary>
    /// <param name="shape">shape</param>
    /// <param name="srcBuffer">source buffer</param>
    /// <param name="name">name</param>
    public Tensor(int[] shape, UnityEngine.ComputeBuffer srcBuffer, string name = "") : this(new TensorShape(shape), srcBuffer, name) {}

    /// <summary>
    /// Create a Tensor of shape [1,1,N,1,1,H,W,C], associated ComputeBuffer `srcBuffer` filled with tensor values, and an optional debug `name`.
    /// `srcBuffer` must be larger than `n*h*w*c`.
    /// </summary>
    /// <param name="n">batch</param>
    /// <param name="h">height</param>
    /// <param name="w">width</param>
    /// <param name="c">channels</param>
    /// <param name="srcBuffer">source buffer</param>
    /// <param name="name">name</param>
    public Tensor(int n, int h, int w, int c, UnityEngine.ComputeBuffer srcBuffer, string name = "") : this(new TensorShape(n, h, w, c), srcBuffer, name) {}

    /// <summary>
    /// Create a Tensor of shape [1,1,N,1,1,1,1,C], associated ComputeBuffer `srcBuffer` filled with tensor values, and an optional debug `name`.
    /// `srcBuffer` must be larger than `n*c`.
    /// </summary>
    /// <param name="n">batch</param>
    /// <param name="c">channels</param>
    /// <param name="srcBuffer">source buffer</param>
    /// <param name="name">name</param>
    public Tensor(int n, int c, UnityEngine.ComputeBuffer srcBuffer, string name = "") : this(new TensorShape(n, c), srcBuffer, name) {}

    /// <summary>
    /// Create a Tensor with specified `shape`, associated ComputeBuffer `srcBuffer` filled with tensor values, and an optional debug `name`.
    /// `srcBuffer` must be larger than `shape.length`.
    /// </summary>
    /// <param name="shape">shape</param>
    /// <param name="srcBuffer">source buffer</param>
    /// <param name="name">name</param>
    /// <exception cref="ArgumentException">thrown if specified buffer is too small or stride is mismatched</exception>
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
    /// Create a Tensor from a texture, shape is [1,1,1,1,1, `texture.height`, `texture.width`, `channels`].
    /// If `channels` is set to -1 (default value), then number of channels in the new Tensor will match the number of channels in the texture.
    /// Just like `Texture2D.GetPixels` when reading from LDR texture (RGBA32, ARGB32, RGB24, Alpha8, RG16, R8, etc) this function will remap pixel values from byte values to the range of [0.0 .. 1.0]. Pixel values from HDR textures (such as ARGBFloat or ARGBHalf) will be left unchanged.
    /// </summary>
    /// <param name="srcTexture">source texture</param>
    /// <param name="channels">channels</param>
    /// <param name="name">name</param>
    public Tensor(UnityEngine.Texture srcTexture, int channels = -1, string name = "") : this(new [] { srcTexture }, channels, name) {}

    /// <summary>
    /// Create a Tensor from multiple texture, shape is [1,1,1,1, `srcTextures.length`, `texture.height`, `texture.width`, `channels`].
    /// If `channels` is set to -1 (default value), then number of channels in the new Tensor will match the number of channels in the texture.
    /// All textures must be of the same size and dimension.
    /// Just like `Texture2D.GetPixels` when reading from LDR texture (RGBA32, ARGB32, RGB24, Alpha8, RG16, R8, etc) this function will remap pixel values from byte values to the range of [0.0 .. 1.0]. Pixel values from HDR textures (such as ARGBFloat or ARGBHalf) will be left unchanged.
    /// </summary>
    /// <param name="srcTextures">source textures</param>
    /// <param name="channels">channels</param>
    /// <param name="name">name</param>
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
    /// `shape` must be of size 8, the order is [S,R,N,T,D,H,W,C].
    /// S and R must be 1.
    /// </summary>
    /// <param name="shape">shape</param>
    /// <param name="data">data</param>
    /// <param name="name">name</param>
    public Tensor(int[] shape, ITensorData data, string name = "") : this(new TensorShape(shape), data, name) {}

    /// <summary>
    /// Create a Tensor of shape [1,1,N,1,1,H,W,C], an ITensorData `data` and an optional debug `name`.
    /// `srcData` must be of size `n*h*w*c`.
    /// </summary>
    /// <param name="n">batch</param>
    /// <param name="h">height</param>
    /// <param name="w">width</param>
    /// <param name="c">channels</param>
    /// <param name="data">data</param>
    /// <param name="name">name</param>
    public Tensor(int n, int h, int w, int c, ITensorData data, string name = "") : this(new TensorShape(n, h, w, c), data, name) {}

    /// <summary>
    /// Create a Tensor of shape [1,1,N,1,1,1,1,C], an ITensorData `data` and an optional debug `name`.
    /// `srcData` must be of size `n*c`.
    /// </summary>
    /// <param name="n">batch</param>
    /// <param name="c">channels</param>
    /// <param name="data">data</param>
    /// <param name="name">name</param>
    public Tensor(int n, int c, ITensorData data, string name = "") : this(new TensorShape(n, c), data, name) {}

    /// <summary>
    /// Create a Tensor with specified `shape`, an ITensorData `data` and an optional debug `name`.
    /// </summary>
    /// <param name="shape">shape</param>
    /// <param name="data">data</param>
    /// <param name="name">name</param>
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
    /// Create an uninitialized Tensor with a shape of [1,1,1,1,1,1,1,1] and an optional debug `name`.
    /// </summary>
    /// <param name="name">name</param>
    public Tensor(string name = "") : this(new TensorShape(1,1,1,1), name) {}

    /// <summary>
    /// Create an uninitialized Tensor from a `shape` and an optional debug `name`.
    /// `shape` must be of size 8, the order is [S,R,N,T,D,H,W,C]
    /// S and R must be 1.
    /// </summary>
    /// <param name="shape">shape</param>
    /// <param name="name">name</param>
    public Tensor(int[] shape, string name = "") : this(new TensorShape(shape), name) {}

    /// <summary>
    /// Create an uninitialized Tensor of shape [1,1,N,1,1,H,W,C] and an optional debug `name`.
    /// </summary>
    /// <param name="n">batch</param>
    /// <param name="h">height</param>
    /// <param name="w">width</param>
    /// <param name="c">channels</param>
    /// <param name="name">name</param>
    public Tensor(int n, int h, int w, int c, string name = "") : this(new TensorShape(n, h, w, c), name) {}

    /// <summary>
    /// Create an uninitialized Tensor of shape [1,1,N,1,1,1,1,C] and an optional debug `name`.
    /// </summary>
    /// <param name="n">batch</param>
    /// <param name="c">channels</param>
    /// <param name="name">name</param>
    public Tensor(int n, int c, string name = "") : this(new TensorShape(n, c), name) {}

    /// <summary>
    /// Create an uninitialized Tensor with specified `shape` and an optional debug `name`.
    /// </summary>
    /// <param name="shape">shape</param>
    /// <param name="name">name</param>
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
    /// `shape` must be of size 8, the order is [S,R,N,T,D,H,W,C].
    /// S and R must be 1.
    /// </summary>
    /// <param name="shape">shape</param>
    /// <param name="data">data</param>
    /// <param name="allocator">allocator</param>
    public Tensor(int[] shape, ITensorData data, ITensorAllocator allocator) : this(new TensorShape(shape), data, allocator) {}

    /// <summary>
    /// Create a Tensor of shape [1,1,N,1,1,H,W,C], an ITensorData `data` and an ITensorAllocator `allocator`.
    /// `data` must be of size `n*h*w*c`.
    /// </summary>
    /// <param name="n">batch</param>
    /// <param name="h">height</param>
    /// <param name="w">width</param>
    /// <param name="c">channels</param>
    /// <param name="data">data</param>
    /// <param name="allocator">allocator</param>
    public Tensor(int n, int h, int w, int c, ITensorData data, ITensorAllocator allocator) : this(new TensorShape(n, h, w, c), data, allocator) {}

    /// <summary>
    /// Create a Tensor of shape [1,1,N,1,1,1,1,C], an ITensorData `data` and an ITensorAllocator `allocator`.
    /// `srcData` must be of size `n*c`.
    /// </summary>
    /// <param name="n">batch</param>
    /// <param name="c">channels</param>
    /// <param name="data">data</param>
    /// <param name="allocator">allocator</param>
    public Tensor(int n, int c, ITensorData data, ITensorAllocator allocator) : this(new TensorShape(n, c), data, allocator) {}

    /// <summary>
    /// Create a Tensor with specified `shape`, an ITensorData `data` and an ITensorAllocator `allocator`
    /// </summary>
    /// <param name="shape">shape</param>
    /// <param name="data">data</param>
    /// <param name="allocator">allocator</param>
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
    /// Create an uninitialized Tensor with a shape of [1,1,1,1,1,1,1,1] and an ITensorAllocator `allocator`.
    /// </summary>
    /// <param name="allocator">allocator</param>
    public Tensor(ITensorAllocator allocator) : this(new TensorShape(1,1,1,1,1,1,1,1), allocator) {}


    /// <summary>
    /// Create an uninitialized Tensor from a `shape` and an ITensorAllocator `allocator`.
    /// `shape` must be of size 8, the order is [S,R,N,T,D,H,W,C].
    /// S and R must be 1.
    /// </summary>
    /// <param name="shape">shape</param>
    /// <param name="allocator">allocator</param>
    public Tensor(int[] shape, ITensorAllocator allocator) : this(new TensorShape(shape), allocator) {}

    /// <summary>
    /// Create an uninitialized Tensor of shape [1,1,N,1,1,H,W,C] and an ITensorAllocator `allocator`.
    /// </summary>
    /// <param name="n">batch</param>
    /// <param name="h">height</param>
    /// <param name="w">width</param>
    /// <param name="c">channels</param>
    /// <param name="allocator">allocator</param>
    public Tensor(int n, int h, int w, int c, ITensorAllocator allocator) : this(new TensorShape(n, h, w, c), allocator) {}

    /// <summary>
    /// Create an uninitialized Tensor of shape [1,1,N,1,1,1,1,C] and an ITensorAllocator `allocator`.
    /// </summary>
    /// <param name="n">batch</param>
    /// <param name="c">channels</param>
    /// <param name="allocator">allocator</param>
    public Tensor(int n, int c, ITensorAllocator allocator) : this(new TensorShape(n, c), allocator) {}

    /// <summary>
    /// Create an uninitialized Tensor with specified `shape` and ITensorAllocator `allocator`.
    /// </summary>
    /// <param name="shape">shape</param>
    /// <param name="allocator">allocator</param>
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
    /// <param name="destination">destination</param>
    /// <param name="invalidateCacheAfterUpload">invalidate cache after upload</param>
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
    /// <param name="source">source</param>
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
    /// <param name="disposeDeviceData">dispose device data</param>
    /// <returns>Tensor data</returns>
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
    /// <param name="blocking">blocking read if `true`</param>
    /// <returns>`true` if data is ready</returns>
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
    /// <param name="newName">new name</param>
    /// <returns>shallow copy of the Tensor</returns>
    public Tensor ShallowCopy(string newName = null)
    {
        return ShallowCopy(shape, newName ?? $"shallowcopy of {name}");
    }

    /// <summary>
    /// Create a flattened copy of the current Tensor ie of shape [1,1,N,1,1,1,1,T*D*H*W*C]
    /// </summary>
    /// <param name="newName">new name</param>
    /// <returns>shallow copy of the Tensor with new shape</returns>
    public Tensor Flatten(string newName = null)
    {
        var newShape = shape.Flatten();
        return ShallowCopy(newShape, newName ?? $"flatten of {name}");
    }

    /// <summary>
    /// Create a reshaped copy of the current Tensor.
    /// `newShape`.length must be equal to this.shape.length.
    /// </summary>
    /// <param name="newShape">new shape</param>
    /// <param name="newName">new name</param>
    /// <returns>shallow copy of the Tensor with new shape and name</returns>
    public Tensor Reshape(TensorShape newShape, string newName = null)
    {
        Assert.AreEqual(shape.length, newShape.length);
        return ShallowCopy(newShape, newName ?? $"reshape of {name}");
    }

    /// <summary>
    /// Create a copy of the current Tensor.
    /// </summary>
    /// <returns>new copy of the Tensor</returns>
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

        tensorDisposed?.Invoke(this);
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
    /// <param name="target">target RenderTexture</param>
    /// <param name="batch">batch</param>
    /// <param name="fromChannel">from channel</param>
    /// <param name="scale">scale</param>
    /// <param name="bias">bias</param>
    /// <param name="lut">lut table</param>
    public void ToRenderTexture(UnityEngine.RenderTexture target, int batch, int fromChannel, Vector4 scale, Vector4 bias, Texture3D lut = null)
    {
        var gpuBackend = new ReferenceComputeOps(null);
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
    /// <param name="target">target RenderTexture</param>
    /// <param name="batch">batch</param>
    /// <param name="fromChannel">from channel</param>
    /// <param name="scale">scale</param>
    /// <param name="bias">bias</param>
    /// <param name="lut">lut table</param>
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
    /// <param name="format">RenderTexture format</param>
    /// <param name="batch">batch</param>
    /// <param name="fromChannel">from channel</param>
    /// <param name="scale">scale</param>
    /// <param name="bias">bias</param>
    /// <param name="lut">lut table</param>
    /// <returns>created RenderTexture</returns>
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
    /// <param name="batch">batch</param>
    /// <param name="fromChannel">from channel</param>
    /// <param name="scale">scale</param>
    /// <param name="bias">bias</param>
    /// <param name="lut">lut table</param>
    /// <returns></returns>
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
    /// <param name="axis">axis</param>
    /// <returns>remapped axis</returns>
    public int Axis(int axis)
    {
        return shape.Axis(axis);
    }

    /// <summary>
    /// Given an element dimensions indices [0,0,N,0,0,H,W,C] return this element offset in memory.
    /// </summary>
    /// <param name="b">batch</param>
    /// <param name="h">height</param>
    /// <param name="w">width</param>
    /// <param name="ch">channels</param>
    /// <returns>flat index (offset in memory)</returns>
    public int Index(int b, int h, int w, int ch)
    {
        return shape.Index(b, h, w, ch);
    }

    /// <summary>
    /// Given an element dimensions indices [0,0,N,0,D,H,W,C] return this element offset in memory.
    /// </summary>
    /// <param name="b">batch</param>
    /// <param name="d">depth</param>
    /// <param name="h">height</param>
    /// <param name="w">width</param>
    /// <param name="ch">channels</param>
    /// <returns></returns>
    public int Index(int b, int d, int h, int w, int ch)
    {
        return shape.Index(b, d, h, w, ch);
    }
    /// <summary>
    /// Given an element dimensions indices [S,R,N,T,D,H,W,C] return this element offset in memory.
    /// </summary>
    /// <param name="s">sequence</param>
    /// <param name="r">direction</param>
    /// <param name="n">batch</param>
    /// <param name="t">time</param>
    /// <param name="d">depth</param>
    /// <param name="h">height</param>
    /// <param name="w">width</param>
    /// <param name="c">channels</param>
    /// <returns>flat index (offset in memory)</returns>
    public int Index(int s, int r, int n, int t, int d, int h, int w, int c)
    {
        return shape.Index(s, r, n, t, d, h, w, c);
    }

    /// <summary>
    /// Given an element dimensions indices [0,0,N,0,0,H,W,C] return this element offset in memory, clamping indices to tensor dimensions.
    /// </summary>
    /// <param name="n">batch</param>
    /// <param name="h">height</param>
    /// <param name="w">width</param>
    /// <param name="c">channels</param>
    /// <returns>flat index (offset in memory)</returns>
    public int IndexWithClamp(int n, int h, int w, int c)
    {
        return shape.IndexWithClamp(n, h, w, c);
    }

    /// <summary>
    /// Given an element dimensions indices [0,0,N,0,D,H,W,C] return this element offset in memory, clamping indices to tensor dimensions.
    /// </summary>
    /// <param name="n">batch</param>
    /// <param name="d">depth</param>
    /// <param name="h">height</param>
    /// <param name="w">width</param>
    /// <param name="c">channels</param>
    /// <returns>flat index (offset in memory)</returns>
    public int IndexWithClamp(int n, int d, int h, int w, int c)
    {
        return shape.IndexWithClamp(n, d, h, w, c);
    }
    /// <summary>
    /// Given an element dimensions indices[0,0,N,0,0,H,W,C] with broadcast support, return this element offset in memory.
    /// </summary>
    /// <param name="n">batch</param>
    /// <param name="h">height</param>
    /// <param name="w">width</param>
    /// <param name="c">channels</param>
    /// <returns>flat index (offset in memory)</returns>
    public int IndexWithBroadcast(int n, int h, int w, int c)
    {
        return shape.IndexWithBroadcast(n, h, w, c);
    }

    /// <summary>
    /// Given an element dimensions indices [S,R,N,T,D,H,W,C] with broadcast support, return this element offset in memory.
    /// </summary>
    /// <param name="s">sequence</param>
    /// <param name="r">direction</param>
    /// <param name="n">batch</param>
    /// <param name="t">time</param>
    /// <param name="d">depth</param>
    /// <param name="h">height</param>
    /// <param name="w">width</param>
    /// <param name="c">channels</param>
    /// <returns>flat index (offset in memory)</returns>
    public int IndexWithBroadcast(int s, int r, int n, int t, int d, int h, int w, int c)
    {
        return shape.IndexWithBroadcast(s,r,n,t,d,h,w,c);
    }
    /// <summary>
    /// Given an element dimensions indices [0,0,N,0,0,0,0,C] return this element offset in memory.
    /// </summary>
    /// <param name="y">y</param>
    /// <param name="x">x</param>
    /// <returns>flat index (offset in memory)</returns>
    public int Index(int y, int x)
    {
        return shape.Index(y, x);
    }

    /// <summary>
    /// Access element at offset `index` in this Tensor.
    /// This will create a blocking read, if this Tensor is a result of a computation on a different device (GPU).
    /// </summary>
    /// <param name="index">flat index</param>
    public float this[int index]
    {
        get { PrepareCacheForAccess(); return m_Cache[index]; }
        set { PrepareCacheForAccess(); m_Cache[index] = value; m_CacheIsDirty = true; }
    }

    /// <summary>
    /// Access element at index [0,0,N,0,0,0,0,C] in this Tensor.
    /// This will create a blocking read, if this Tensor is a result of a computation on a different device (GPU).
    /// </summary>
    /// <param name="b">batch</param>
    /// <param name="ch">channels</param>
    public float this[int b, int ch]
    {
        get { PrepareCacheForAccess(); return m_Cache[Index(b, ch)]; }
        set { PrepareCacheForAccess(); m_Cache[Index(b, ch)] = value; m_CacheIsDirty = true; }
    }

    /// <summary>
    /// Access element at index [0,0,N,0,0,H,W,C] in this Tensor.
    /// This will create a blocking read, if this Tensor is a result of a computation on a different device (GPU).
    /// </summary>
    /// <param name="b">batch</param>
    /// <param name="h">height</param>
    /// <param name="w">width</param>
    /// <param name="ch">channels</param>
    public float this[int b, int h, int w, int ch]
    {
        get { PrepareCacheForAccess(); return m_Cache[Index(b, h, w, ch)]; }
        set { PrepareCacheForAccess(); m_Cache[Index(b, h, w, ch)] = value; m_CacheIsDirty = true; }
    }
    /// <summary>
    /// Access element at index [0,0,N,0,D,H,W,C] in this Tensor.
    /// This will create a blocking read, if this Tensor is a result of a computation on a different device (GPU).
    /// </summary>
    public float this[int b, int d, int h, int w, int ch]
    {
        get { PrepareCacheForAccess(); return m_Cache[Index(b, d, h, w, ch)]; }
        set { PrepareCacheForAccess(); m_Cache[Index(b, d, h, w, ch)] = value; m_CacheIsDirty = true; }
    }


    /// <summary>
    /// Access element at index [S,R,N,T,D,H,W,C] in this Tensor.
    /// This will create a blocking read, if this Tensor is a result of a computation on a different device (GPU).
    /// </summary>
    /// <param name="s">sequence</param>
    /// <param name="r">direction</param>
    /// <param name="n">batch</param>
    /// <param name="t">time</param>
    /// <param name="d">depth</param>
    /// <param name="h">height</param>
    /// <param name="w">width</param>
    /// <param name="c">channels</param>
    public float this[int s, int r, int n, int t, int d, int h, int w, int c]
    {
        get { PrepareCacheForAccess(); return m_Cache[Index(s, r, n, t , d, h, w, c)]; }
        set { PrepareCacheForAccess(); m_Cache[Index(s, r, n, t , d, h, w, c)] = value; m_CacheIsDirty = true; }
    }

    /// <summary>
    /// Return the cached linear memory representation of this tensor data.
    /// This will create a blocking read, if this Tensor is a result of a computation on a different device (GPU).
    /// IMPORTANT: Modifying contents of the returned array will have undefined behavior.
    /// </summary>
    /// <returns>cached linear memory representation of this tensor data</returns>
    public float[] ToReadOnlyArray()
    {
        // @TODO: implement via ITensorData.SharedAccess(), public float[] ToReadOnlyArray(ref int arrayOffset)
        PrepareCacheForAccess();
        return m_Cache;
    }
    #endregion

    /// <summary>
    /// Device specific internal representation of Tensor data
    /// </summary>
    public ITensorData tensorOnDevice { get { return m_TensorOnDevice; } }

    /// <summary>
    /// Upload data to device and return its instance
    /// </summary>
    public ITensorData data
    {
        get
        {
            if (m_TensorOnDevice == null)
                UploadToDevice(new ArrayTensorData(shape));
            return m_TensorOnDevice;
        }
    }

    /// <summary>
    /// Tensor metadata summary
    /// </summary>
    /// <returns>Tensor metadata summary</returns>
    public override string ToString()
    {
        return $"(`{name}` {shape}, alloc: {m_TensorAllocator?.GetType()}, onDevice:{m_TensorOnDevice})";
    }

    #region Obsolete
    private bool m_Disposing = false;    // to protect from infinite-loop. in case UnpinAndDisposeTensor() is called from Dispose()

    /// <summary>
    /// Unload tensor data from device and dispose this Tensor
    /// </summary>
    /// <returns>device specific Tensor data</returns>
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

    /// <summary>
    /// Read-only array of Tensor data
    /// </summary>
    [ObsoleteAttribute("Use ToReadOnlyArray instead.", false)]
    public float[] readonlyArray { get { PrepareCacheForAccess(); return m_Cache; } }

    /// <summary>
    /// Offset into read-only array of Tensor data
    /// </summary>
    [ObsoleteAttribute("Use ToReadOnlyArray instead.", false)]
    public int readonlyArrayOffset { get { return 0; } }
    #endregion

}

} // namespace Barracuda
