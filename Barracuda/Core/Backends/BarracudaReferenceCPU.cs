using UnityEngine;
using UnityEngine.Assertions;
using System;
using System.Linq;

namespace Barracuda {

public class ArrayTensorData : ITensorData
{
    protected float[] m_Array;
    public float[] array { get { return m_Array; } }

    public ArrayTensorData(int count)
    {
        m_Array = new float[count];
    }

    public ArrayTensorData(TensorShape shape) : this(shape.length)
    {
    }

    ~ArrayTensorData()
    {
        Dispose();
    }

    public virtual void Dispose()
    {
        m_Array = null;
    }

    public virtual void Reserve(int count)
    {
        if (count > m_Array.Length)
            m_Array = new float[count];
    }

    public virtual void Upload(float[] data, int offset = 0, int count = -1)
    {
        Assert.IsTrue(offset >= 0);
        if (count < 0)
            count = data.Length - offset;

        if (m_Array == data && offset == 0)
        {
            Assert.IsTrue(count == data.Length);
            return;
        }

        Reserve(count);

        Array.Copy(data, offset, m_Array, 0, count);
    }

    public virtual bool ScheduleAsyncDownload(int count)
    {
        return true;
    }

    public virtual float[] Download(int count)
    {
        //;;D.logStackTraceEnabled = true;
        //;;D.Log("Download ArrayTensorData " + count + " from " + m_Array.Length + " @ " + ToString());
        //;;D.logStackTraceEnabled = false;

        Assert.IsTrue(m_Array.Length >= count);
        count = Math.Min(m_Array.Length, count);

        if (count <= m_Array.Length)
            return m_Array;

        var dest = new float[count];
        Array.Copy(m_Array, 0, dest, 0, count);
        return dest;
    }

    public virtual float[] SharedAccess(out int offset)
    {
        offset = 0;
        return m_Array;
    }

    public virtual int GetMaxCount()
    {
        return m_Array.Length;
    }

    public override string ToString()
    {
        return string.Format("(CPU array: {0} max: {1})",
            GetHashCode(), m_Array?.Length);
    }
}

public class SharedArrayTensorData : ITensorData
{
    protected float[] m_Array;
    protected int m_Offset;
    protected int m_Count;

    public float[] array { get { return m_Array; } }
    public int offset { get { return m_Offset; } }
    public int count { get { return m_Count; } }

    public SharedArrayTensorData(float[] data, int offset = 0, int count = -1)
    {
        Assert.IsTrue(offset >= 0);
        if (count < 0)
            count = data.Length - offset;

        m_Array = data;
        m_Offset = offset;
        Assert.IsTrue(count >= 0);
        Assert.IsTrue(offset + count <= m_Array.Length);
        m_Count = count;
    }

    ~SharedArrayTensorData()
    {
        Dispose();
    }

    public virtual void Dispose()
    {
    }

    public virtual void Reserve(int count)
    {
        // currently always readonly
        throw new InvalidOperationException("SharedArrayTensorData is readonly!");
    }

    public virtual void Upload(float[] data, int offset = 0, int count = -1)
    {
        // currently always readonly
        throw new InvalidOperationException("SharedArrayTensorData is readonly!");
    }

    public virtual bool ScheduleAsyncDownload(int count)
    {
        return true;
    }

    public virtual float[] Download(int count)
    {
        //;;D.logStackTraceEnabled = true;
        //;;D.Log("Download SharedArrayTensorData " + count + " from " + m_Count + " @ " + ToString());
        //;;D.logStackTraceEnabled = false;

        Assert.IsTrue(m_Count >= count);
        count = Math.Min(m_Count, count);

        var dest = new float[count];
        Array.Copy(m_Array, m_Offset, dest, 0, count);
        return dest;
    }

    public virtual float[] SharedAccess(out int offset)
    {
        offset = m_Offset;
        return m_Array;
    }

    public virtual int GetMaxCount()
    {
        return m_Array.Length - m_Offset;
    }

    public override string ToString()
    {
        return string.Format("(CPU shared: {0} max: {1} offset: {2} count: {3})",
            GetHashCode(), m_Array.Length, m_Offset, m_Count);
    }
}


public class ReferenceCPUOps : IOps
{
    private ITensorAllocator m_Allocator;

    public ReferenceCPUOps(ITensorAllocator allocator = null)
    {
        if (allocator == null)
            allocator = new TensorCachingAllocator();
        m_Allocator = allocator;
    }

    protected Tensor NewTensor(TensorShape s, string name = "")
    {
        var tensor = m_Allocator.Alloc(s);
        tensor.name = name;
        return tensor;
    }

    protected Tensor NewTensorLike(Tensor t)
    {
        return NewTensor(t.shape);
    }

    protected Tensor NewTensor(int b, int ch, string name = "")
    {
        return NewTensor(new TensorShape(b, ch), name);
    }

    protected Tensor NewTensor(int b, int h, int w, int ch, string name = "")
    {
        return NewTensor(new TensorShape(b, h, w, ch), name);
    }

    public virtual void WaitForCompletion(Tensor x)
    {
        // do nothing on CPU
    }

    public virtual void ResetAllocator(bool keepCachedMemory = true)
    {
        m_Allocator.Reset(keepCachedMemory);
    }

    // ---------------------------------------------------------------------------------

    public virtual Tensor MatMul(Tensor X, bool xTranspose, Tensor Y, bool yTranspose)
    {
        Assert.IsTrue(X.dimensions <= 2);
        Assert.IsTrue(Y.dimensions <= 2);
        X = X.Flatten();
        Y = Y.Flatten();

        if (xTranspose)
            X = Transpose(X);
        if (yTranspose)
            Y = Transpose(Y);

        Assert.AreEqual(X.flatWidth, Y.flatHeight);
        var O = NewTensor(X.flatHeight, Y.flatWidth);

        for (int y = 0; y < O.flatHeight; ++y)
            for (int x = 0; x < O.flatWidth; ++x)
            {
                float v = 0;
                for (int i = 0; i < X.flatWidth; ++i)
                {
                    v += X[y, i] * Y[i, x];
                }
                O[y, x] = v;
            }
        return O;
    }

    public virtual Tensor Dense(Tensor X, Tensor W, Tensor B)
    {
        Assert.IsTrue(W.dimensions <= 2);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(X.flatWidth, W.flatHeight);

        var O = NewTensor(X.flatHeight, W.flatWidth);

        for (int y = 0; y < O.flatHeight; ++y)
            for (int x = 0; x < O.flatWidth; ++x)
            {
                float v = B[x];
                for (int i = 0; i < X.flatWidth; ++i)
                {
                    v += X[y, i] * W[i, x];
                }
                O[y, x] = v;
            }
        return O;
    }

    public virtual Tensor Conv2D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad)
    {
        Assert.AreEqual(X.channels, K.kernelDepth);
        Assert.AreEqual(K.kernelCount, B.flatWidth);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);

        var O = NewTensor(X.shape.ApplyKernel(K.shape, stride, pad));

        for (var n = 0; n < O.batch; ++n)
            for (var y = 0; y < O.height; ++y)
                for (var x = 0; x < O.width; ++x)
                    for (var k = 0; k < K.kernelCount; ++k)
                    {
                        float v = B[k];
                        for (int dy = 0; dy < K.kernelHeight; ++dy)
                        {
                            for (int dx = 0; dx < K.kernelWidth; ++dx)
                            {
                                int oy = y * stride[1] + dy - pad[1];
                                int ox = x * stride[0] + dx - pad[0];

                                if (oy < 0) continue;
                                if (oy >= X.height) continue;
                                if (ox < 0) continue;
                                if (ox >= X.width) continue;

                                for (var c = 0; c < X.channels; ++c)
                                {
                                    float xv = X[n, oy, ox, c
                                        //n  * X.height * X.width * X.channels +
                                        //oy * X.width * X.channels +
                                        //ox * X.channels +
                                        //c  +
                                        //X.offset
                                    ];

                                    float kv = K[dy, dx, c, k
                                        //dy * K.height * K.width * K.channels +
                                        //dx * K.width * K.channels +
                                        //c  * K.channels +
                                        //k  +
                                        //K.offset
                                    ];

                                    v += xv * kv;
                                }
                            }
                        }
                        O[n, y, x, k
                            //n * O.height * O.width * O.channels +
                            //y * O.width * O.channels +
                            //x * O.channels +
                            //k +
                            //O.offset
                        ] = v;
                    }
        return O;
    }

    public virtual Tensor DepthwiseConv2D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad)
    {
        if (K.kernelDepth != 1)
            throw new NotImplementedException();

        Assert.AreEqual(K.kernelDepth, 1);
        Assert.AreEqual(K.kernelCount, X.channels);
        Assert.AreEqual(K.kernelCount, B.flatWidth);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);

        // ONNX: (M x C/group x kH x kW)
        // TF: [H, W, in_channels, channel_multiplier]

        // TF pseudocode:
        // output[b, i, j, k * channel_multiplier + q] =
        // sum_{di, dj}
        //      input [b, i + di, j + dj, k] *
        //      filter[di, dj, k, q] *

        var O = NewTensor(X.shape.ApplyKernel(K.shape, stride, pad));

        for (var n = 0; n < O.batch; ++n)
            for (var y = 0; y < O.height; ++y)
                for (var x = 0; x < O.width; ++x)
                    for (var k = 0; k < K.kernelCount; ++k)
                    {
                        float v = B[k];
                        for (int dy = 0; dy < K.kernelHeight; ++dy)
                            for (int dx = 0; dx < K.kernelWidth; ++dx)
                            {
                                int oy = y * stride[1] + dy - pad[1];
                                int ox = x * stride[0] + dx - pad[0];

                                if (oy < 0) continue;
                                if (oy >= X.height) continue;
                                if (ox < 0) continue;
                                if (ox >= X.width) continue;

                                float xv = X[n, oy, ox, k];
                                float kv = K[dy, dx, 0, k];
                                v += xv * kv;
                            }
                        O[n, y, x, k] = v;
                    }
        return O;
    }

    public virtual Tensor Conv2DTrans(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, int[] outputAdjustment)
    {
        Assert.AreEqual(X.channels, K.kernelDepth);
        Assert.AreEqual(K.kernelCount, B.flatWidth);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);
        Assert.AreEqual(pad[0],pad[2]);
        Assert.AreEqual(pad[1],pad[3]);

        var O = NewTensor(X.shape.ApplyKernelInverse(K.shape, stride, pad, outputAdjustment));
        int prePadW = K.kernelWidth  - pad[0] - 1;
        int prePadH = K.kernelHeight - pad[1] - 1;
        int strideH = 1;
        int strideW = 1;

        for (var n = 0; n < O.batch; ++n)
            for (var y = 0; y < O.height; ++y)
                for (var x = 0; x < O.width; ++x)
                    for (var k = 0; k < K.kernelCount; ++k)
                    {
                        float v = B[k];
                        for (int dy = 0; dy < K.kernelHeight; dy += strideH)
                            for (int dx = 0; dx < K.kernelWidth; dx += strideW)
                            {
                                int readX = (x + dx - prePadW) / stride[0];
                                int readY = (y + dy - prePadH) / stride[1];

                                if ((x + dx - prePadW) % stride[0] != 0) continue;
                                if ((y + dy - prePadH) % stride[0] != 0) continue;
                                if (readX < 0) continue;
                                if (readX >= X.width) continue;
                                if (readY < 0) continue;
                                if (readY >= X.height) continue;

                                for (var c = 0; c < X.channels; ++c)
                                {
                                    float xv = X[n, readY, readX, c];
                                    float kv = K[K.kernelHeight - 1 - dy,
                                                 K.kernelWidth  - 1 - dx, c, k];
                                    v += xv * kv;
                                }
                            }
                        
                        O[n, y, x, k] = v;
                    }
        return O;
    }

    public virtual Tensor Upsample2D(Tensor X, int[] size)
    {
        Assert.AreEqual(size.Length, 2);
        var O = NewTensor(X.batch, X.height*size[1], X.width*size[0], X.channels);

        for (int b = 0; b < O.batch; ++b)
            for (int y = 0; y < O.height; ++y)
                for (int x = 0; x < O.width; ++x)
                    for (int c = 0; c < O.channels; ++c)
                    {
                        int oy = y / size[1];
                        int ox = x / size[0];
                        float v = X[b, oy, ox, c
                            //b  * X.height * X.width * X.channels +
                            //oy * X.width * X.channels +
                            //ox * X.channels +
                            //c +
                            //X.offset
                        ];

                        O[b, y, x, c
                            //b * O.height * O.width * O.channels +
                            //y * O.width * O.channels +
                            //x * O.channels +
                            //c +
                            //O.offset
                        ] = v;
                    }
        return O;
    }

    public virtual Tensor MaxPool2D(Tensor X, int[] pool, int[] stride, int[] pad)
    {
        Assert.AreEqual(pool.Length, 2);
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);

        var O = NewTensor(X.shape.ApplyPool(pool, stride, pad));

        for (int b = 0; b < O.batch; ++b)
            for (int y = 0; y < O.height; ++y)
                for (int x = 0; x < O.width; ++x)
                    for (int c = 0; c < O.channels; ++c)
                    {
                        float maxVal = float.MinValue;
                        for (int dy = 0; dy < pool[1]; ++dy)
                            for (int dx = 0; dx < pool[0]; ++dx)
                            {
                                int oy = y * stride[1] + dy - pad[1];
                                int ox = x * stride[0] + dx - pad[0];

                                if (oy < 0) continue;
                                if (oy >= X.height) continue;
                                if (ox < 0) continue;
                                if (ox >= X.width) continue;

                                float v = X[b, oy, ox, c
                                    //b  * X.height * X.width * X.channels +
                                    //oy * X.width * X.channels +
                                    //ox * X.channels +
                                    //c +
                                    //X.offset
                                ];
                                maxVal = Mathf.Max(v, maxVal);
                            }

                        O[b, y, x, c
                            //b * O.height * O.width * O.channels +
                            //y * O.width * O.channels +
                            //x * O.channels +
                            //c +
                            //O.offset
                        ] = maxVal;
                    }
        return O;
    }

    public virtual Tensor AvgPool2D(Tensor X, int[] pool, int[] stride, int[] pad)
    {
        Assert.AreEqual(pool.Length, 2);
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);

        var O = NewTensor(X.shape.ApplyPool(pool, stride, pad));

        for (int b = 0; b < O.batch; ++b)
            for (int y = 0; y < O.height; ++y)
                for (int x = 0; x < O.width; ++x)
                    for (int c = 0; c < O.channels; ++c)
                    {
                        float accum = 0.0f;
                        float counter = 0.0f;
                        for (int dy = 0; dy < pool[1]; ++dy)
                            for (int dx = 0; dx < pool[0]; ++dx)
                            {
                                int oy = y * stride[1] + dy - pad[1];
                                int ox = x * stride[0] + dx - pad[0];

                                if (oy < 0) continue;
                                if (oy >= X.height) continue;
                                if (ox < 0) continue;
                                if (ox >= X.width) continue;

                                float v = X[b, oy, ox, c
                                    //b  * X.height * X.width * X.channels +
                                    //oy * X.width * X.channels +
                                    //ox * X.channels +
                                    //c +
                                    //X.offset
                                ];
                                accum += v;
                                ++counter;
                            }

                        O[b, y, x, c
                            //b * O.height * O.width * O.channels +
                            //y * O.width * O.channels +
                            //x * O.channels +
                            //c +
                            //O.offset
                        ] = accum / counter;
                    }
        return O;
    }

    public virtual Tensor GlobalMaxPool2D(Tensor X)
    {
        var O = NewTensor(X.batch, 1, 1, X.channels);

        for (int b = 0; b < X.batch; ++b)
            for (int c = 0; c < X.channels; ++c)
            {
                float maxVal = float.MinValue;
                for (int y = 0; y < X.height; ++y)
                    for (int x = 0; x < X.width; ++x)
                    {
                        float v = X[b, y, x, c
                            //b * X.height * X.width * X.channels +
                            //y * X.width * X.channels +
                            //x * X.channels +
                            //c +
                            //X.offset
                        ];
                        maxVal = Mathf.Max(v, maxVal);
                    }

                O[b, 0, 0, c
                    //b * O.channels +
                    //c +
                    //O.offset
                ] = maxVal;
            }
        return O;
    }

    public virtual Tensor GlobalAvgPool2D(Tensor X)
    {
        var O = NewTensor(X.batch, 1, 1, X.channels);

        for (int b = 0; b < X.batch; ++b)
            for (int c = 0; c < X.channels; ++c)
            {
                float accum = 0.0f;
                for (int y = 0; y < X.height; ++y)
                    for (int x = 0; x < X.width; ++x)
                    {
                        float v = X[b, y, x, c
                            //b * X.height * X.width * X.channels +
                            //y * X.width * X.channels +
                            //x * X.channels +
                            //c +
                            //X.offset
                        ];
                        accum += v;
                    }

                O[b, 0, 0, c
                    //b * O.channels +
                    //c +
                    //O.offset
                ] = accum / (X.width * X.height);
            }
        return O;
    }

    public virtual Tensor GlobalAvgVariancePool2D(Tensor X)
    {
        var O = NewTensor(X.batch, 2, 1, X.channels);

        for (int b = 0; b < X.batch; ++b)
            for (int c = 0; c < X.channels; ++c)
            {
                float mean = 0.0f;
                float mean2 = 0.0f;
                for (int y = 0; y < X.height; ++y)
                    for (int x = 0; x < X.width; ++x)
                    {
                        float v = X[b, y, x, c
                            //b * X.height * X.width * X.channels +
                            //y * X.width * X.channels +
                            //x * X.channels +
                            //c +
                            //X.offset
                        ];
                        mean  += v;
                        mean2 += v*v;
                    }

                mean  /= (X.width * X.height);
                mean2 /= (X.width * X.height);

                O[b, 0, 0, c
                    //b * O.channels +
                    //c +
                    //O.offset
                ] = mean;

                O[b, 1, 0, c
                    //b * O.channels +
                    //c +
                    //O.offset
                ] = mean2 - mean * mean;
            }
        return O;
    }
        

    private Tensor ApplyPadding(Tensor X, int[] pad, Func<Tensor, int, int, int, int, float> paddingOp)
    {
        Assert.AreEqual(pad.Length, 4);

        var O = NewTensor(X.shape.ApplyBorder(pad));

        // NOTE: negative "pad" variable will crop X tensor
        int croppedWidth = X.width - Math.Max(0, -pad[2]);
        int croppedHeight = X.height - Math.Max(0, -pad[3]);

        for (int b = 0; b < O.batch; ++b)
            for (int y = 0; y < O.height; ++y)
                for (int x = 0; x < O.width; ++x)
                {
                    int readX = x - pad[0];
                    int readY = y - pad[1];

                    if (readX < 0 || readX >= croppedWidth ||
                        readY < 0 || readY >= croppedHeight)
                    {
                        for (int c = 0; c < O.channels; ++c)
                            O[b, y, x, c] = paddingOp(X, b, readY, readX, c);
                    }
                    else
                    {
                        for (int c = 0; c < O.channels; ++c)
                            O[b, y, x, c] = X[b, readY, readX, c];
                    }
                }

        return O;
    }

    public virtual Tensor Border2D(Tensor X, int[] pad, float value)
    {
        Func<Tensor, int, int, int, int, float> padOp = (tensor, b, h, w, c) => value;
        return ApplyPadding(X, pad, padOp);
    }

    private static void ClampHWToTensorShape(TensorShape shape, ref int height, ref int width)
    {
        width = Math.Max(width, 0);
        height = Math.Max(height, 0);
        width = Math.Min(width, shape.width - 1);
        height = Math.Min(height, shape.height - 1);
    }

    public virtual Tensor Pad2DReflect(Tensor X, int[] pad)
    {
        float GetReflectPadding(Tensor tensorX, int b, int readY, int readX, int c)
        {
            int lastXIndex = tensorX.shape.width - 1;
            int lastYIndex = tensorX.shape.height - 1;

            if (readX < 0)
                readX = -readX;
            else if (readX > lastXIndex)
                readX = lastXIndex - (readX - lastXIndex);

            if (readY < 0)
                readY = -readY;
            else if (readY > lastYIndex)
                readY = lastYIndex - (readY - lastYIndex);

            ClampHWToTensorShape(tensorX.shape, ref readY, ref readX);
            return tensorX[b,readY, readX,c];
        }


        return ApplyPadding(X, pad, GetReflectPadding);
    }

    public virtual Tensor Pad2DSymmetric(Tensor X, int[] pad)
    {
        float GetSymmetricPadding(Tensor tensorX, int b, int readY, int readX, int c)
        {
            int lastXIndex = tensorX.shape.width - 1;
            int lastYIndex = tensorX.shape.height - 1;

            if (readX < 0)
                readX = -readX - 1;
            else if (readX > lastXIndex)
                readX = lastXIndex - (readX - lastXIndex) + 1;

            if (readY < 0)
                readY = -readY - 1;
            else if (readY > lastYIndex)
                readY = lastYIndex - (readY - lastYIndex) + 1;

            ClampHWToTensorShape(tensorX.shape, ref readY, ref readX);
            return tensorX[b,readY, readX,c];
        }

        return ApplyPadding(X, pad, GetSymmetricPadding);
    }

    public virtual Tensor Pad2DEdge(Tensor X, int[] pad)
    {
        float GetEdgePadding(Tensor tensorX, int b, int readY, int readX, int c)
        {
            ClampHWToTensorShape(tensorX.shape, ref readY, ref readX);
            return tensorX[b,readY, readX,c];
        }


        return ApplyPadding(X, pad, GetEdgePadding);
    }

    public virtual Tensor ScaleBias(Tensor X, Tensor S, Tensor B)
    {
        Assert.AreEqual(X.channels, B.channels); Assert.AreEqual(X.channels, S.channels);
        Assert.AreEqual(B.length, B.channels); Assert.AreEqual(S.length, S.channels);

        var O = NewTensorLike(X);

        for (int b = 0; b < X.batch; ++b)
            for (int y = 0; y < X.height; ++y)
                for (int x = 0; x < X.width; ++x)
                    for (int c = 0; c < X.channels; ++c)
                    {
                        float beta = B[0, 0, 0, c];//.array[c + B.offset];
                        float gamma = S[0, 0, 0, c];//S.array[c + S.offset];

                        //var i = X.IndexWithOffset(b, y, x, c);
                        float v = X[b, y, x, c];//.array[i];
                        O[b, y, x, c] = v * gamma + beta;
                    }
        return O;
    }

    public virtual Tensor LRN(Tensor X, float alpha, float beta, float bias, int size)
    {
        // https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
        throw new NotImplementedException();
    }

    public virtual Tensor Normalization(Tensor X, Tensor S, Tensor B, int pool, int axis, float epsilon)
    {
        Assert.AreEqual(X.channels, B.channels); Assert.AreEqual(X.channels, S.channels);

        if (axis != 3 && axis != -1)
            throw new NotImplementedException();

        // Special cases of Normalization:
        // 1) Instance Normalization, if pool == 1
        // 2) Batch Normalization, if pool <= 0
        if (pool <= 0)
            pool = X.batch;

        var O = NewTensorLike(X);

        var channels = X.channels;
        var width = X.width;
        var height = X.height;

        for (int subBatch = 0; subBatch < O.batch; subBatch += pool)
            for (int c = 0; c < channels; ++c)
            {
                int bBegin = subBatch;
                int bEnd = Math.Min(subBatch + pool, O.batch);

                float gamma = S[0, 0, 0, c];//.array[c + S.offset];
                float beta = B[0, 0, 0, c];//B.array[c + B.offset];

                // calc mean
                double sum = 0;
                for (int b = bBegin; b < bEnd; ++b)
                    for (int y = 0; y < height; ++y)
                        for (int x = 0; x < width; ++x)
                        {
                            double v = X[b, y, x, c];
                            sum += v;
                        }
                double mean = sum / (width * height);

                // calc variance
                sum = 0;
                for (int b = bBegin; b < bEnd; ++b)
                    for (int y = 0; y < height; ++y)
                        for (int x = 0; x < width; ++x)
                        {
                            double v = X[b, y, x, c];
                            sum += (v - mean) * (v - mean);
                        }
                double var = sum / (width * height);

                // apply normalization
                for (int b = bBegin; b < bEnd; ++b)
                    for (int y = 0; y < height; ++y)
                        for (int x = 0; x < width; ++x)
                        {
                            float v = X[b, y, x, c];
                            O[b, y, x, c] = (float)(gamma * (v - mean) / Math.Sqrt(var + epsilon) + beta);
                        }
            }
        return O;
    }

    protected float Bernoulli(float p)
    {
        return (UnityEngine.Random.value <= p) ? 1f: 0f;
    }

    protected float Gaussian(float mean, float stdDev)
    {
        float u, v, s;
        do {
            u = UnityEngine.Random.value * 2 - 1;
            v = UnityEngine.Random.value * 2 - 1;
            s = u * u + v * v;
        } while (s >= 1 || s == 0);
        float mul = Mathf.Sqrt(-2.0f * Mathf.Log(s) / s);
        return mean + stdDev * u * mul;
    }

    protected class Seed : IDisposable
    {
        UnityEngine.Random.State[] m_SeedStorage;
        UnityEngine.Random.State m_EngineSeed;
        public Seed(ref UnityEngine.Random.State[] storage, int initialSeed)
        {
            m_EngineSeed = UnityEngine.Random.state;
            if (storage == null)
            {
                storage = new UnityEngine.Random.State[1];
                UnityEngine.Random.InitState(initialSeed);
                storage[0] = UnityEngine.Random.state;
            }
            else
                UnityEngine.Random.state = storage[0];
            m_SeedStorage = storage;
        }

        public virtual void Dispose()
        {
            m_SeedStorage[0] = UnityEngine.Random.state;
            UnityEngine.Random.state = m_EngineSeed;
        }
    }

    private UnityEngine.Random.State[] m_DropoutSeed;
    public virtual Tensor Dropout(Tensor X, float alpha)
    {
        Assert.IsTrue(alpha >= 0f && alpha <= 1f);
        var O = NewTensorLike(X);

        // Based on PyTorch Dropout implementation
        // See: https://github.com/pytorch/pytorch/blob/master/torch/nn/_functions/dropout.py

        using (var seedOverride = new Seed(ref m_DropoutSeed, 1337))
        {
            var end = X.length;
            for (int i = 0; i < end; ++i)
            {
                float v = X[i];
                v *= Bernoulli(1f - alpha) / (1f - alpha);
                O[i] = v;
            }
        }
        return O;
    }

    private UnityEngine.Random.State[] m_RandomNormalSeed;
    public virtual Tensor RandomNormal(TensorShape s, float mean, float scale, int seed)
    {
        var O = NewTensor(s);

        using (var seedOverride = new Seed(ref m_RandomNormalSeed, seed))
        {
            var end = O.length;
            for (int i = 0; i < end; ++i)
                O[i] = Gaussian(mean, scale);
        }

        return O;
    }

    private UnityEngine.Random.State[] m_RandomUniformSeed;
    public virtual Tensor RandomUniform(TensorShape s, float mean, float scale, int seed)
    {
        var O = NewTensor(s);

        using (var seedOverride = new Seed(ref m_RandomUniformSeed, seed))
        {
            var end = O.length;
            for (int i = 0; i < end; ++i)
                O[i] = mean + scale * UnityEngine.Random.value;
        }

        return O;
    }

    private UnityEngine.Random.State[] m_MultinomialSeed;
    public virtual Tensor Multinomial(Tensor X, int count, int seed)
    {
        var O = NewTensor(X.flatHeight, count);

        // Tensorflow Multinomial for reference
        // See: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/multinomial_op.cc

        using (var seedOverride = new Seed(ref m_MultinomialSeed, seed))
        {
            for (int n = 0; n < X.flatHeight; ++n)
            {
                var maxLogP = Mathf.NegativeInfinity;
                for (int i = 0; i < X.flatWidth; ++i)
                    maxLogP = Mathf.Max(X[n, i], maxLogP);

                float sumOfProbabilities = 0f;
                for (int i = 0; i < X.flatWidth; ++i)
                    sumOfProbabilities += Mathf.Exp(X[n, i] - maxLogP); // NOTE: X contains log-probabilities

                for (int sample = 0; sample < count; ++sample)
                {
                    float p = UnityEngine.Random.value * sumOfProbabilities;

                    int i = 0;
                    float cumulativeP = 0f;
                    while (i < X.flatWidth && p > cumulativeP)
                    {
                        cumulativeP += Mathf.Exp(X[n, i] - maxLogP);
                        i++;
                    }
                    Assert.IsTrue(i > 0);
                    O[n, sample] = (float)(i - 1);
                }
            }
        }

        return O;
    }

    public virtual Tensor OneHot(Tensor X, int depth, float onValue, float offValue)
    {
        var O = NewTensor(X.flatHeight, 1, X.flatWidth, depth);

        for (int n = 0; n < X.flatHeight; ++n)
        {
            for (int j = 0; j < X.flatWidth; ++j)
            {
                int index = (int)X[n, j];
                for (int i = 0; i < depth; ++i)
                {
                    float v = (i == index) ? onValue: offValue;
                    O[n, 0, j, i] = v;
                }
            }
        }
        return O;
    }

    public virtual Tensor Relu(Tensor X)
    {
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = Mathf.Max(v, 0.0f);
            O[i] = v;
        }
        return O;
    }

    public virtual Tensor PRelu(Tensor X, Tensor S)
    {
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            float slope = S[i % S.length];

            v = Mathf.Max(0.0f, v) + slope * Mathf.Min(0.0f, v);
            O[i] = v;
        }

        return O;
    }

    public virtual Tensor Softmax(Tensor X)
    {
        var O = NewTensor(X.shape.Flatten());

        //e_x = np.exp(X - X.max(axis=1, keepdims=True))
        //X = e_x / e_x.sum(axis=1, keepdims=True)
        for (int y = 0; y < X.flatHeight; ++y)
        {
            float maxV = Mathf.NegativeInfinity;
            for (int x = 0; x < X.channels; ++x)
            {
                float v = X[y, x
                    //b * X.channels +
                    //x +
                    //X.offset
                ];

                if (v > maxV)
                    maxV = v;
            }

            float sum = 0.0f;
            for (int x = 0; x < X.flatWidth; ++x)
            {
                float v = X[y, x
                    // y * X.channels +
                    // x +
                    // X.offset
                ];
                sum += Mathf.Exp(v - maxV);
            }

            for (int x = 0; x < X.flatWidth; ++x)
            {
                float v = X[y, x
                    //y * X.channels +
                    //x +
                    //X.offset
                ];
                v = Mathf.Exp(v - maxV) / sum;
                O[y, x
                    //y * O.width +
                    //x +
                    //O.offset
                ] = v;
            }
        }

        return O;
    }

    public virtual Tensor LogSoftmax(Tensor X)
    {
        var O = NewTensor(X.shape.Flatten());

        //e_x = np.exp(X - X.max(axis=1, keepdims=True))
        //X = log( e_x / e_x.sum(axis=1, keepdims=True) )
        for (int y = 0; y < X.flatHeight; ++y)
        {
            float maxV = Mathf.NegativeInfinity;
            for (int x = 0; x < X.channels; ++x)
            {
                float v = X[y, x
                    //b * X.channels +
                    //x +
                    //X.offset
                ];

                if (v > maxV)
                    maxV = v;
            }

            float sum = 0.0f;
            for (int x = 0; x < X.flatWidth; ++x)
            {
                float v = X[y, x
                    // y * X.channels +
                    // x +
                    // X.offset
                ];
                sum += Mathf.Exp(v - maxV);
            }

            for (int x = 0; x < X.flatWidth; ++x)
            {
                float v = X[y, x
                    //y * X.channels +
                    //x +
                    //X.offset
                ];
                v = Mathf.Log( Mathf.Exp(v - maxV) / sum );
                O[y, x
                    //y * O.width +
                    //x +
                    //O.offset
                ] = v;
            }
        }

        return O;
    }

    public virtual Tensor Tanh(Tensor X)
    {
        // f(x) = tanh(x) = sinh(x) / cosh(x) = (exp(2*x) - 1) / (exp(2*x) + 1)
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            O[i] = MathfEx.tanh(X[i]);
        }
        return O;
    }

    public virtual Tensor Sigmoid(Tensor X)
    {
        // f(x) = 1 / (1 + exp(-x))
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = 1f / (1f + Mathf.Exp(-v));
            O[i] = v;
        }
        return O;
    }

    public virtual Tensor Relu6(Tensor X)
    {
        // f(x) = min(max(x, 0), 6)
        // "Convolutional Deep Belief Networks on CIFAR-10", A Krizhevsky, 2010
        // http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = Mathf.Min(Mathf.Max(0f, v), 6f);
            O[i] = v;
        }
        return O;
    }

    public virtual Tensor Elu(Tensor X, float alpha)
    {
        // f(x) = alpha * (exp(x) - 1.) for x < 0, f(x) = x for x >= 0
        // "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)", DA Clevert, 2015
        // https://arxiv.org/abs/1511.07289
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            if (v <= 0)
                v = alpha * (Mathf.Exp(v) - 1f);
            O[i] = v;
        }
        return O;
    }

    public virtual Tensor LeakyRelu(Tensor X, float alpha)
    {
        // f(x) = alpha * x for x < 0, f(x) = x for x >= 0.
        // "Rectifier Nonlinearities Improve Neural Network Acoustic Models". AL Maas, 2013
        // http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf
        Assert.IsTrue(alpha <= 1);
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = Mathf.Max(v, alpha * v);
            // @TODO: doublecheck the following code
            // from Theano impl
            // https://github.com/Theano/theano/blob/d395439aec5a6ddde8ef5c266fd976412a5c5695/theano/tensor/nnet/nnet.py#L2209-L2251
            //float f1 = 0.5f * (1f + alpha)
            //float f2 = 0.5f * (1f - alpha)
            //v = f1 * v + f2 * Mathf.Abs(v);
            O[i] = v;
        }
        return O;
    }

    public virtual Tensor Selu(Tensor X, float alpha, float gamma)
    {
        // f(x) = gamma * (alpha * e^x - alpha) for x <= 0, f(x) = gamma * x for x > 0
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            if (v <= 0)
                v = gamma * (alpha * Mathf.Exp(v) - alpha);
            else
                v = gamma * v;
            O[i] = v;
        }
        return O;
    }

    public virtual Tensor Swish(Tensor X)
    {
        // f(x) = sigmoid(x) * x = x / (1 + exp(-x))
        // "Searching for Activation Functions". P Ramachandran, 2017
        // https://arxiv.org/abs/1710.05941
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = v / (1f + Mathf.Exp(-v));
            O[i] = v;
        }
        return O;
    }

    public virtual Tensor Abs(Tensor X)
    {
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = Mathf.Abs(v);
            O[i] = v;
        }
        return O;
    }

    public virtual Tensor Neg(Tensor X)
    {
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = -v;
            O[i] = v;
        }
        return O;
    }

    public virtual Tensor Ceil(Tensor X)
    {
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = Mathf.Ceil(v);
            O[i] = v;
        }
        return O;
    }

    public virtual Tensor Clip(Tensor X, float min, float max)
    {
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = Mathf.Clamp(v, min, max);

            O[i] = v;
        }
        return O;
    }

    public virtual Tensor Floor(Tensor X)
    {
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = Mathf.Floor(v);
            O[i] = v;
        }
        return O;
    }

    public virtual Tensor Reciprocal(Tensor X)
    {
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = 1.0f / v;
            O[i] = v;
        }
        return O;
    }

    public virtual Tensor Pow(Tensor X, float alpha)
    {
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = Mathf.Pow(v, alpha);
            O[i] = v;
        }
        return O;
    }

    public virtual Tensor Exp(Tensor X)
    {
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = Mathf.Exp(v);
            O[i] = v;
        }
        return O;
    }

    public virtual Tensor Log(Tensor X)
    {
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = Mathf.Log(v);
            O[i] = v;
        }
        return O;
    }

    public virtual Tensor Sqrt(Tensor X)
    {
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = Mathf.Sqrt(v);
            O[i] = v;
        }
        return O;
    }

    public virtual Tensor Concat(Tensor[] tensors, int axis)
    {
        var concatShape = TensorExtensions.ConcatShapes(tensors, axis);
        var O = NewTensor(concatShape);

        var srcIndices = new long[tensors.Length];
        for (int i = 0; i < tensors.Length; ++i)
            srcIndices[i] = tensors[i].readonlyArrayOffset;

        // product of all tensor dimensions starting from axis
        var copyBlockLengths = new long[tensors.Length];
        for (int i = 0; i < tensors.Length; ++i)
            copyBlockLengths[i] = tensors[i].shape.ToArray().Skip(tensors[i].shape.Axis(axis)).Aggregate(1L, (a, b) => (long)a * (long)b);

        // copy tensor data interleaved into O
        int intDstIndex = 0;
        var dstArray = O.data.SharedAccess(out intDstIndex);
        long dstIndex = intDstIndex;
        long takes = concatShape.ToArray().Take(concatShape.Axis(axis)).Aggregate(1L, (a, b) => (long)a * (long)b);
        for (int take = 0; take < takes; ++take)
            for (int i = 0; i < tensors.Length; ++i)
            {
                var copyLength = copyBlockLengths[i];

                Array.Copy(tensors[i].readonlyArray, srcIndices[i], // from
                    dstArray, dstIndex, copyLength);                // to

                srcIndices[i] += copyLength;
                dstIndex += copyLength;
            }

        O.data.Upload(dstArray, 0);
        return O;
    }

    public virtual Tensor StridedSlice(Tensor X, int[] starts, int[] ends, int[] stride)
    {
        Assert.AreEqual(starts.Length, 4);
        Assert.AreEqual(ends.Length, 4);
        Assert.AreEqual(stride.Length, 4);

        var O = NewTensor(X.shape.ApplyStridedSlice(starts, ends, stride));

        var startB = TensorExtensions.WrapIndex(starts[0], X.batch);
        var startY = TensorExtensions.WrapIndex(starts[1], X.height);
        var startX = TensorExtensions.WrapIndex(starts[2], X.width);
        var startC = TensorExtensions.WrapIndex(starts[3], X.channels);

        for (int b = 0; b < O.batch; ++b)
            for (int y = 0; y < O.height; ++y)
                for (int x = 0; x < O.width; ++x)
                    for (int c = 0; c < O.channels; ++c)
                        O[b, y, x, c] = X[
                            startB + b * stride[0],
                            startY + y * stride[1],
                            startX + x * stride[2],
                            startC + c * stride[3]];
        return O;
    }

    public virtual Tensor Tile(Tensor X, int[] repeats)
    {
        Assert.AreEqual(repeats.Length, 4);
        var O = NewTensor(X.shape.Scale(repeats));

        for (int b = 0; b < O.batch; ++b)
            for (int y = 0; y < O.height; ++y)
                for (int x = 0; x < O.width; ++x)
                    for (int c = 0; c < O.channels; ++c)
                        O[b, y, x, c] = X[
                            b % repeats[0],
                            y % repeats[1],
                            x % repeats[2],
                            c % repeats[3]];
        return O;
    }

    private Tensor ApplyElementwiseWithBroadcast(Tensor[] tensors, Func<float, float, float> operation)
    {
        var O = GetOutputTensorFromBroadcast(tensors);
        var A = tensors[0];
        for (int t = 1; t < tensors.Length; ++t)
        {
            var B = tensors[t];
            for (int b = 0; b < O.shape.batch; ++b)
            {
                for (int h = 0; h < O.shape.height; ++h)
                {
                    for (int w = 0; w < O.shape.width; ++w)
                    {
                        for (int c = 0; c < O.shape.channels; ++c)
                        {
                            var valueA = A[A.IndexWithBroadcast(b, h, w, c)];
                            var valueB = B[B.IndexWithBroadcast(b, h, w, c)];
                            O[O.Index(b, h, w, c)] = operation(valueA, valueB);
                        }
                    }
                }
            }
            A = O;
        }
        return O;
    }

    // O = tensors[0] + tensors[1] + ... + tensors[N-1]
    public virtual Tensor Add(Tensor[] tensors)
    {
        Func<float, float, float> op = (a, b) => a + b;
        return ApplyElementwiseWithBroadcast(tensors, op);
    }

    // O = tensors[0] - tensors[1] - ... - tensors[N-1]
    public virtual Tensor Sub(Tensor[] tensors)
    {
        Func<float, float, float> op = (a, b) => a - b;
        return ApplyElementwiseWithBroadcast(tensors, op);
    }

    // O = tensors[0] * tensors[1] * ... * tensors[N-1]
    public virtual Tensor Mul(Tensor[] tensors)
    {
        Func<float, float, float> op = (a, b) => a * b;
        return ApplyElementwiseWithBroadcast(tensors, op);
    }
    // O = tensors[0] / tensors[1] / ... / tensors[N-1]
    public virtual Tensor Div(Tensor[] tensors)
    {
        Func<float, float, float> op = (a, b) => a / b;
        return ApplyElementwiseWithBroadcast(tensors, op);
    }

    // O = tensors[0] ^ tensors[1] ^ ... ^ tensors[N-1]
    public virtual Tensor Pow(Tensor[] tensors)
    {
        Func<float, float, float> op = (a, b) => Mathf.Pow(a, b);
        return ApplyElementwiseWithBroadcast(tensors, op);
    }

    // O = min(tensors[0], tensors[1],  ... , tensors[N-1])
    public virtual Tensor Min(Tensor[] tensors)
    {
        Func<float, float, float> op = (a, b) => Mathf.Min(a, b);
        return ApplyElementwiseWithBroadcast(tensors, op);
    }

    // O = max(tensors[0], tensors[1],  ... , tensors[N-1])
    public virtual Tensor Max(Tensor[] tensors)
    {
        Func<float, float, float> op = (a, b) => Mathf.Max(a, b);
        return ApplyElementwiseWithBroadcast(tensors, op);
    }

    // O = (1/N) * (tensors[0] + tensors[1] + ... + tensors[N-1])
    public virtual Tensor Mean(Tensor[] tensors)
    {
        // accumulate
        Func<float, float, float> op = (a, b) => a + b;
        var O = ApplyElementwiseWithBroadcast(tensors, op);

        // div by N
        var invN = 1.0f / tensors.Length;
        var end = O.length;
        for (int i = 0; i < O.length; ++i)
        {
            float v = O[i];
            v *= invN;
            O[i] = v;
        }
        return O;
    }

    public virtual Tensor ReduceMin(Tensor X, int axis)
    {
        if (axis != 3 && axis != -1)
            throw new NotImplementedException();

        var O = NewTensor(X.shape.Reduce(axis));
        var n = X.channels;

        for (int b = 0; b < O.batch; ++b)
            for (int y = 0; y < O.height; ++y)
                for (int x = 0; x < O.width; ++x)
                {
                    float acc = float.MaxValue;
                    for (int c = 0; c < n; ++c)
                        acc = Mathf.Min(acc, X[b, y, x, c]);
                    O[b, y, x, 0] = acc;
                }
        return O;
    }

    public virtual Tensor ReduceMax(Tensor X, int axis)
    {
        if (axis != 3 && axis != -1)
            throw new NotImplementedException();

        var O = NewTensor(X.shape.Reduce(axis));
        var n = X.channels;

        for (int b = 0; b < O.batch; ++b)
            for (int y = 0; y < O.height; ++y)
                for (int x = 0; x < O.width; ++x)
                {
                    float acc = float.MinValue;
                    for (int c = 0; c < n; ++c)
                        acc = Mathf.Max(acc, X[b, y, x, c]);
                    O[b, y, x, 0] = acc;
                }
        return O;
    }

    public virtual Tensor ReduceSum(Tensor X, int axis)
    {
        if (axis != 3 && axis != -1)
            throw new NotImplementedException();

        var O = NewTensor(X.shape.Reduce(axis));
        var n = X.channels;

        for (int b = 0; b < O.batch; ++b)
            for (int y = 0; y < O.height; ++y)
                for (int x = 0; x < O.width; ++x)
                {
                    float acc = 0.0f;
                    for (int c = 0; c < n; ++c)
                        acc += X[b, y, x, c];
                    O[b, y, x, 0] = acc;
                }
        return O;
    }

    public virtual Tensor ReduceMean(Tensor X, int axis)
    {
        if (axis != 3 && axis != -1)
            throw new NotImplementedException();

        var O = NewTensor(X.shape.Reduce(axis));
        var n = X.channels;

        for (int b = 0; b < O.batch; ++b)
            for (int y = 0; y < O.height; ++y)
                for (int x = 0; x < O.width; ++x)
                {
                    float acc = 0.0f;
                    for (int c = 0; c < n; ++c)
                        acc += X[b, y, x, c];
                    O[b, y, x, 0] = acc / n;
                }
        return O;
    }

    public virtual Tensor ReduceProd(Tensor X, int axis)
    {
        if (axis != 3 && axis != -1)
            throw new NotImplementedException();

        var O = NewTensor(X.shape.Reduce(axis));
        var n = X.channels;

        for (int b = 0; b < O.batch; ++b)
            for (int y = 0; y < O.height; ++y)
                for (int x = 0; x < O.width; ++x)
                {
                    float acc = 1.0f;
                    for (int c = 0; c < n; ++c)
                        acc *= X[b, y, x, c];
                    O[b, y, x, 0] = acc;
                }
        return O;
    }

    private Tensor GetOutputTensorFromBroadcast(Tensor[] tensors)
    {
        Assert.IsTrue(tensors.Length > 0);

        var O = NewTensor(TensorExtensions.MaxShape(tensors));
        foreach (var t in tensors)
        {
            Assert.IsTrue((t.batch    == 1) || (t.batch    == O.batch));
            Assert.IsTrue((t.height   == 1) || (t.height   == O.height));
            Assert.IsTrue((t.width    == 1) || (t.width    == O.width));
            Assert.IsTrue((t.channels == 1) || (t.channels == O.channels));
        }

        return O;
    }

    private Tensor ApplyLogicalOperator(Tensor tensorA, Tensor tensorB, Func<float, float, float> logicOp)
    {
        var O = GetOutputTensorFromBroadcast(new Tensor[] { tensorA, tensorB });
        for (int b = 0; b < O.shape.batch; ++b)
        {
            for (int h = 0; h < O.shape.height; ++h)
            {
                for (int w = 0; w < O.shape.width; ++w)
                {
                    for (int c = 0; c < O.shape.channels; ++c)
                    {
                            var A = tensorA[tensorA.IndexWithBroadcast(b, h, w, c)];
                            var B = tensorB[tensorB.IndexWithBroadcast(b, h, w, c)];
                            O[O.Index(b,h,w,c)] = logicOp(A,B);
                    }
                }
            }
        }

        return O;
    }

    public virtual Tensor Greater(Tensor A, Tensor B)
    {
        Func<float, float, float> logicOp = (a, b) => Convert.ToSingle(a > b);
        return ApplyLogicalOperator(A, B, logicOp);
    }
    public virtual Tensor GreaterEqual(Tensor A, Tensor B)
    {
        Func<float, float, float> logicOp = (a, b) => Convert.ToSingle(a >= b);
        return ApplyLogicalOperator(A, B, logicOp);
    }
    public virtual Tensor Less(Tensor A, Tensor B)
    {
        Func<float, float, float> logicOp = (a, b) => Convert.ToSingle(a < b);
        return ApplyLogicalOperator(A, B, logicOp);
    }
    public virtual Tensor LessEqual(Tensor A, Tensor B)
    {
        Func<float, float, float> logicOp = (a, b) => Convert.ToSingle(a <= b);
        return ApplyLogicalOperator(A, B, logicOp);
    }
    public virtual Tensor Equal(Tensor A, Tensor B)
    {
        Func<float, float, float> logicOp = (a, b) => Convert.ToSingle(a == b);
        return ApplyLogicalOperator(A, B, logicOp);
    }
    public virtual Tensor LogicalOr(Tensor A, Tensor B)
    {
        Func<float, float, float> logicOp = (a, b) => Convert.ToSingle( Convert.ToBoolean(a) || Convert.ToBoolean(b) );
        return ApplyLogicalOperator(A, B, logicOp);
    }
    public virtual Tensor LogicalAnd(Tensor A, Tensor B)
    {
        Func<float, float, float> logicOp = (a, b) => Convert.ToSingle( Convert.ToBoolean(a) && Convert.ToBoolean(b) );
        return ApplyLogicalOperator(A, B, logicOp);
    }
    public virtual Tensor LogicalXor(Tensor A, Tensor B)
    {
        Func<float, float, float> logicOp = (a, b) => Convert.ToSingle( Convert.ToBoolean(a) ^ Convert.ToBoolean(b) );
        return ApplyLogicalOperator(A, B, logicOp);
    }
    public virtual Tensor LogicalNot(Tensor X)
    {
        var O = NewTensorLike(X);
        var end = O.length;
        for (int i = 0; i < end; ++i)
            O[i] = Convert.ToSingle( !Convert.ToBoolean(X[i]) );
        return O;
    }

    public virtual Tensor Flatten(Tensor X)
    {
        return X.Flatten();
    }

    public virtual Tensor Reshape(Tensor X, TensorShape newShape)
    {
        return X.Reshape(newShape);
    }

    public virtual Tensor Gather(Tensor[] tensors, int axis)
    {
        Tensor X = tensors[0];
        Tensor indices = tensors[1];

        int[] shape = X.shape.ToArray();
        shape[axis] = indices.flatWidth;

        var O = NewTensor(new TensorShape(shape));

        for (int b = 0; b < shape[0]; ++b)
            for (int y = 0; y < shape[1]; ++y)
                for (int x = 0; x < shape[2]; ++x)
                    for (int c = 0; c < shape[3]; ++c)
                    {
                        if (axis == 0)
                            O[b, y, x, c] = X[(int)indices[b], y, x, c];
                        else if (axis == 1)
                            O[b, y, x, c] = X[b, (int)indices[y], x, c];
                        else if (axis == 2)
                            O[b, y, x, c] = X[b, y, (int)indices[x], c];
                        else
                            O[b, y, x, c] = X[b, y, x, (int)indices[c]];
                    }
        return O;
    }

    public virtual Tensor Transpose(Tensor X)
    {
        Assert.IsTrue(X.dimensions <= 2);
        X = X.Flatten();

        var O = NewTensor(X.flatWidth, X.flatHeight);

        for (int y = 0; y < O.flatHeight; ++y)
            for (int x = 0; x < O.flatWidth; ++x)
                O[y, x] = X[x, y];

        return O;
    }

    public virtual Tensor Prepare(Tensor X)
    {
        X.PrepareCacheForAccess();
        return X;
    }
}

    public class MathfEx
    {
        public static float tanh(float x)
        {
            // tanh = (exp(2*x) - 1) / (exp(2*x) + 1)

            // Constant taken from http://llvm.org/svn/llvm-project/libclc/trunk/generic/lib/math/tanh.cl
            // const float large_threshold = 0x1.0a2b24p+3f;
            const float LargeThreshold = 8.317766f;

            // See also: https://stackoverflow.com/questions/34835641/tanh-returning-nan-for-large-input

            // Handle edge-cases to prevent NaNs creeping in
            if (x >= LargeThreshold || x <= -LargeThreshold)
                return Mathf.Sign(x);

            float exp2 = Mathf.Exp(2f * x);
            return (exp2 - 1f) / (exp2 + 1f);
        }
    }

} // namespace Barracuda
