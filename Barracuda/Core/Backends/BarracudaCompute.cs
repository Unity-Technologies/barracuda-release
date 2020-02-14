using UnityEngine;
using UnityEngine.Assertions;
using System;
using System.Linq;
using System.Collections.Generic;

/*
PERFORMANCE COMPARISON after the latest OPTIMIZATION pass
default @ be623ff20d72 VS compute-optimizations2 @ 13946c6c7e50

NOTES:
1) 33% in 1 batch cases and over 100% for 16 batch cases in most models
2) Most models saw boost with large batches due to "unrolling" of images over N,W,H dimensions in optimized Convolution kernel
3) INCEPTION saw large performance boost due to introduction of Convolution kernel that efficiently supports arbitrary input/output channel counts

-------------------------------------------------------------
BASELINE: default @ be623ff20d72
log comment: “Added Conv2d_L1Cache32 variant, removed extra check in the kernel, restored performance on older Radeons + Intel”

VGG
@1    Exec #50:  95.2 ms, cpu: 1.0 ms, avg:  64.8 ms, result:OK
@16   Exec #8: 1108.1 ms, cpu: 1.2 ms, avg: 1112.6 ms, result:OK

MOBILENET
@1    Exec #100:  37.9 ms, cpu: 7.9 ms, avg:  22.5 ms, result:OK
@16   Exec #32: 213.0 ms, cpu: 9.3 ms, avg: 216.3 ms, result:OK

RES
@1    Exec #50:  42.4 ms, cpu: 7.0 ms, avg:  43.2 ms, result:OK
@16   Exec #15: 654.8 ms, cpu: 16.0 ms, avg: 682.6 ms, result:OK

INCEPTION
@1    Exec #32:  86.8 ms, cpu: 21.8 ms, avg:  92.6 ms, result:OK
@16   Exec #8: 1344.2 ms, cpu: 26.4 ms, avg: 1349.7 ms, result:OK


PIX2PIX
@1    Exec #15: 279.0 ms, cpu: 2.5 ms, avg: 239.6 ms, result:OK
PIX2PIX_T
@1   Exec #32: 114.3 ms, cpu: 2.3 ms, avg: 117.2 ms, result:OK


-------------------------------------------------------------
OPTIMIZED: compute-optimizations2 @ 13946c6c7e50
log comment: “Optimizations: added path that support arbitrary number of input and ouptut channels in Convolutions (toggled via STRICT_CHANNELS)”

VGG
@1    Exec #50:  45.8 ms, cpu: 1.0 ms, avg:  46.5 ms, result:OK      39%
@16   Exec #16: 529.1 ms, cpu: 1.1 ms, avg: 539.6 ms, result:OK     106%

MOBILENET
@1    Exec #100:  28.6 ms, cpu: 6.7 ms, avg:  16.8 ms, result:OK     33%
@16   Exec #48: 138.2 ms, cpu: 9.4 ms, avg: 116.4 ms, result:OK      85%

RES
@1    Exec #50:  32.7 ms, cpu: 6.6 ms, avg:  33.6 ms, result:OK      28%
@16   Exec #31: 312.2 ms, cpu: 8.3 ms, avg: 319.4 ms, result:OK     113%

INCEPTION
@1    Exec #50:  48.0 ms, cpu: 21.9 ms, avg:  55.2 ms, result:OK     67%
@16   Exec #32: 188.7 ms, cpu: 25.7 ms, avg: 198.4 ms, result:OK    580%

PIX2PIX
@1   Exec #32: 152.2 ms, cpu: 2.6 ms, avg: 154.6 ms, result:OK       55%
PIX2PIX_T
@1   Exec #32: 123.1 ms, cpu: 2.4 ms, avg: 107.1 ms, result:OK      9.4%


*/

namespace Barracuda {

public sealed class ComputeKernelLibrary
{
    static public int IDivC(int v, int div)
    {
        return (v + div - 1) / div;
    }

    static public Entry[] Dense(TensorShape X, TensorShape W, TensorShape O, int type)
    {
        var h = O.flatHeight;
        var w = O.flatWidth;

        return new[] {
            new[] { // float16
                new Entry("DenseFP16Div2",
                    Int3(w / 2, h),                                 BigO(X.flatWidth)
                    // @TODO: w % 2 == 0
                ),
            },
            new[] { // float32
                new Entry("Dense_Tilled2x2_Cached",
                    Int3(w/2, h/2),                                 BigO(X.flatWidth)/2,
                    StrictAnd(w % 2 == 0 && h % 2 == 0 && X.flatWidth % 32 == 0),
                    (Application.platform == RuntimePlatform.Android) || (Application.platform == RuntimePlatform.IPhonePlayer) || (SystemInfo.graphicsDeviceVendor.Contains("Intel"))
                ),
                new Entry("Dense_Tilled4x4_Cached",
                    Int3(w/4, h/4),                                 BigO(X.flatWidth)/4,
                    StrictAnd(w % 4 == 0 && h % 4 == 0 && X.flatWidth % 32 == 0),
                    (Application.platform == RuntimePlatform.Android) || (Application.platform == RuntimePlatform.IPhonePlayer) || (SystemInfo.graphicsDeviceVendor.Contains("Intel"))
                ),
                new Entry("Dense_T8x8_R8x8",
                    Int3(w / 8, h / 8),                             BigO(X.flatWidth)/8,
                    StrictAnd(w % 64 == 0 && h % 64 == 0 && X.flatWidth % 64 == 0)
                ),
                new Entry("Dense_T16x16_R4x4",
                    Int3(w / 4, h / 4),                             BigO(X.flatWidth)/4,
                    StrictAnd(w % 64 == 0 && h % 64 == 0 && X.flatWidth % 64 == 0)
                ),
                new Entry("Dense_T8x8_R4x4",
                    Int3(w / 4, h / 4),                             BigO(X.flatWidth)/4,
                    StrictAnd(w % 32 == 0 && h % 32 == 0 && X.flatWidth % 32 == 0)
                ),
                // old
                new Entry("DenseTiled64x64",
                    Int3(w / 4, h / 4),                             BigO(X.flatWidth)*1.33f/4,
                    StrictAnd(w % 4 == 0 && h % 4 == 0
                        && X.flatWidth % 64 == 0 && ComputeInfo.supportsDense64x64)
                ),
                new Entry("DenseTiled32x32",
                    Int3(w / 2, h / 2),                             BigO(X.flatWidth)*1.33f/2,
                    StrictAnd(w % 2 == 0 && h % 2 == 0
                        && X.flatWidth % 32 == 0 && ComputeInfo.supportsDense32x32)
                ),
                new Entry("DenseTiled16x16",
                    Int3(w, h),                                     BigO(X.flatWidth)*1.33f,
                    StrictAnd(X.flatWidth % 16 == 0)
                    // @TODO: relax Strict constraint, only And part should be necessary due to mask
                ),
                new Entry("Dense_L1Cached64",
                    Int3(w, h),                                     BigO(X.flatWidth)
                ),

            },
        } [type];
    }

    static public Entry[] Conv2D(TensorShape X, TensorShape K, TensorShape O, int[] stride, int[] pad)
    {
        var n = O.batch;
        var h = O.height;
        var w = O.width;
        var k = K.kernelCount;
        var c = X.channels;

        return new[] {
            new Entry("Conv2DWinograd_2x2_3x3",
                Int3(k, IDivC(w, 2), IDivC(h, 2)),  BigO(X.channels) * 1 / (2*2.25f),
                StrictAnd(K.kernelWidth == 3 && K.kernelHeight == 3 && stride[0] == 1 && stride[1] == 1 && h % 2 == 0 && w % 2 == 0),
                (Application.platform == RuntimePlatform.Android) || (Application.platform == RuntimePlatform.IPhonePlayer) || (SystemInfo.graphicsDeviceVendor.Contains("Intel"))
            ),
            new Entry("Conv2DKernelKxK_StrictC16K64_T8x8_R8x8",
                Int3(IDivC(k, 8), IDivC(w*h, 8)),      BigO(X.channels) * 1.05f / 4,
                stride[0] == 1 && stride[1] == 1 && k % 64 == 0 && X.channels % 16 == 0 && n == 1 && w%4 == 0 && ComputeInfo.supportsComputeSharedMemory,
                (Application.platform == RuntimePlatform.PS4) || (Application.platform == RuntimePlatform.XboxOne)
            ),
            new Entry("Conv2DKernel1x1_StrictC16K64_T16x16_R4x4",
                Int3(IDivC(k, 4), IDivC(n*w*h, 4)),    BigO(X.channels) * 0.8f / 4,
                K.kernelWidth == 1 && K.kernelHeight == 1 &&
                stride[0] == 1 && stride[1] == 1 &&
                k % 64 == 0 && X.channels % 16 == 0 &&
                ComputeInfo.supportsComputeSharedMemory
            ),
            new Entry("Conv2DKernelKxK_StrictC16K64_T16x16_R4x4",
                Int3(IDivC(k, 4), IDivC(n*w*h, 4)),    BigO(X.channels) * 0.9f / 4,
                k % 64 == 0 && X.channels % 16 == 0 && ComputeInfo.supportsComputeSharedMemory
            ),
            new Entry("Conv2DKernelKxK_T16x16_R4x4",
                Int3(IDivC(k, 4), IDivC(n*w*h, 4)),    BigO(X.channels) * 1.0f / 4,
                k >= 16 && c >= 16 && ComputeInfo.supportsComputeSharedMemory
            ),
//            new Entry("Conv2DKernelKxK_T16x16_R4x4",
//                Int3(IDivC(k, 4), IDivC(n*w*h, 4)),                 BigO(X.channels) * 1.1f / 4
//            ),
            // old
            new Entry("Conv2D_L1Cached64_RegisterBlock4x4",
                Int3(K.kernelCount, w/4+1, h/4+1),                  BigO(O.batch * X.channels) * 1.1f / 4,
                K.kernelCount % 64 == 0 && X.channels % 64 == 0 && ComputeInfo.supportsComputeSharedMemory
            ),
            new Entry("Conv2D_L1Cached32_RegisterBlock4x4",
                Int3(K.kernelCount, w/4+1, h/4+1),                  BigO(O.batch * X.channels) / 3,
                K.kernelCount % 32 == 0 && X.channels % 32 == 0 && ComputeInfo.supportsComputeSharedMemory
            ),
            new Entry("Conv2D_RegisterBlock4x2",
                Int3(K.kernelCount, w/4, h/2),                      BigO(O.batch * X.channels) * 1.1f / 2,
                StrictAnd(
                    w % 4 == 0 && h % 2 == 0)
            ),
            new Entry("Conv2D",
                Int3(k, w, h),                                             BigO(O.batch * X.channels)
            ),
        };
    }

    static public Entry[] DepthwiseConv2D(TensorShape X, TensorShape K, TensorShape O)
    {
        var h = O.height;
        var w = O.width;

        return new[] {

            new Entry("DepthwiseConv2D",
                Int3(K.kernelCount, w, h),                          BigO(O.batch * X.channels)
            ),
        };
    }

    static public Entry[] Conv2DTrans(TensorShape X, TensorShape K, TensorShape O)
    {
        return new[] {
            new Entry("Conv2DTrans_KernelCached_K5x5_T16x16",
                dispatch_: Int3(K.kernelCount, O.width, O.height), bigO_: BigO(O.batch * O.channels * X.channels) / 3,
                valid_: (X.channels <= 256 && K.kernelHeight <= 5 && K.kernelWidth <= 5)
            ),
            new Entry("Conv2DTrans",
                dispatch_: Int3(K.kernelCount, O.width, O.height), bigO_: BigO(O.batch * O.channels * X.channels)
            ),
        };
    }

    static public Entry[] Activation(TensorShape X, TensorShape O, string kernelName)
    {
        return
            new[] {
            new Entry(kernelName + "_FlatStrict",
                dispatch_: Int3(O.length/2),
                bigO_: 0.8f* BigO(1),
                strictDims: StrictAnd(O.length % 128 == 0)
            ),
            new Entry(kernelName + "_Flat",
                dispatch_: Int3(O.length),
                bigO_: BigO(1)
            ),
            new Entry(kernelName + "_Loop",
                dispatch_: Int3(O.length),
                bigO_: BigO(2),
                loopStride_: 256
            )
        };
    }

    static public Entry[] PRelu(TensorShape X, TensorShape O)
    {
        return new[] {
            new Entry("PRelu_CNyx2",
                Int3(O.channels, O.batch * O.height * O.width)
            ),
            new Entry("PRelu_Flat",
                Int3(O.length)
            ),
            new Entry("PRelu_Loop",
                Int3(O.length), BigO(2), 256
            )
        };
    }

    static public Entry[] Softmax(TensorShape X, TensorShape O)
    {
        return new[] {
            new Entry("Softmax",
                Int3(O.flatWidth, O.flatHeight)
            ),
        };
    }

    static public Entry[] LogSoftmax(TensorShape X, TensorShape O)
    {
        return new[] {
            new Entry("LogSoftmax",
                Int3(O.flatWidth, O.flatHeight)
            ),
        };
    }

    static public Entry[] ScaleBias(TensorShape X, TensorShape O)
    {
        return new[] {
            new Entry("ScaleBias_CNyx2",
                Int3(O.channels, O.batch * O.height * O.width)
            ),
            new Entry("ScaleBias_Flat",
                Int3(O.length)
            ),
            new Entry("ScaleBias_Loop",
                Int3(O.length), BigO(2), 256
            )
        };
    }

    static public Entry[] Upsample2D(TensorShape X, TensorShape O)
    {
        return new[] {
            // NOTE: dispatched over X (not O)
            new Entry("Upsample2D",
                Int3(X.channels, X.width, X.height),                BigO(X.batch)
            ),
        };
    }

    static public Entry[] Pool2D(TensorShape X, TensorShape O, string kernelName)
    {
        return new[] {
            //new Entry(kernelName + "_16x4x4",
            //    Int3(O.channels, O.width, O.height),            BigO(O.batch)
            //),
            new Entry(kernelName,
                Int3(O.channels, O.width, O.height),                BigO(O.batch)
            ),
        };
    }

    static public Entry[] GlobalPool2D(TensorShape X, TensorShape O, string kernelName)
    {
        return new[] {
            new Entry(kernelName,
                Int3(O.channels),                                   BigO(O.batch)
            ),
        };
    }

    static public Entry[] Normalization(TensorShape X, TensorShape O)
    {
        return new[] {
            new Entry("InstanceNorm",
                Int3(O.channels),                                   BigO(O.batch * O.width * O.height)
            ),
        };
    }
    static public Entry[] NormalizationTail(TensorShape X, TensorShape O)
    {
        return new[] {
            new Entry("InstanceNormTail_CNyx2",
                Int3(O.channels, O.batch * O.height * O.width)
            ),
            new Entry("InstanceNormTail_Flat",
                Int3(O.length)
            ),
            new Entry("InstanceNormTail_Loop",
                Int3(O.length), BigO(2), 256
            )
        };
    }

    static public Entry[] Copy(TensorShape X, TensorShape O)
    {
        return new[] {
            // NOTE: dispatched over X (not O)
            new Entry("Copy",
                Int3(X.channels, X.width, X.height),                BigO(O.batch)
            ),
        };
    }

    static public Entry[] Padding(TensorShape X, TensorShape O, string kernelName)
    {
        return new[] {
            new Entry(kernelName,
                Int3(O.channels, O.width, O.height),                BigO(O.batch)
            ),
        };
    }

    static public Entry[] Broadcast(TensorShape X, TensorShape O, string kernelName)
    {
        return new[] {
            new Entry(kernelName,
                Int3(O.channels, O.width, O.height),                BigO(O.batch)
            ),
        };
    }

    static int[] Int3(int x, int y = 1, int z = 1) { return new[] { x, y, z }; }
    static float BigO(int o) { return (float)o; }
    public struct StrictDimensions { public bool valid; }
    static StrictDimensions StrictAnd(bool valid_) { return new StrictDimensions { valid = valid_ }; }
    static StrictDimensions Strict() { return new StrictDimensions { valid = true }; }

    public struct Entry
    {
        public readonly string name;
        public readonly int[] dispatch;
        public readonly float bigO;
        public readonly bool valid;
        public readonly bool strict;
        public readonly uint loopStride; // > 0 indicates looping kernel
        public readonly bool devicePriority;

        public Entry(string name_, int[] dispatch_, float bigO_ = 1.0f, bool valid_ = true, bool devicePriority_ = false)
        {
            name = name_;
            dispatch = dispatch_;
            bigO = bigO_;
            valid = valid_;
            strict = false;
            loopStride = 0;
            devicePriority = devicePriority_;
        }

        public Entry(string name_, int[] dispatch_, float bigO_, uint loopStride_) :
            this(name_, dispatch_, bigO_)
        {
            loopStride = loopStride_;
        }

        public Entry(string name_, int[] dispatch_, float bigO_, StrictDimensions strictDims) :
            this(name_, dispatch_, bigO_, strictDims.valid)
        {
            strict = true;
        }

        public Entry(string name_, int[] dispatch_, float bigO_, StrictDimensions strictDims, bool devicePriority_) :
            this(name_, dispatch_, bigO_, strictDims.valid, devicePriority_)
        {
            strict = true;
        }
    }
}

public struct ComputeKernel
{
    readonly public ComputeFunc func;
    readonly public int[] dispatch;
    public ComputeShader shader { get { return func.shader; } }

    public ComputeKernel(ComputeFunc func_, int[] dispatch_)
    {
        func = func_;
        dispatch = dispatch_;
    }

    public void SetTensor(string name, TensorShape shape, ComputeBuffer buffer, Int64 dataOffset = 0)
    {
        func.SetTensor(name, shape, buffer, dataOffset);
    }
    public void SetTensor(ComputeFunc.TensorDecl tensorDecl, int dataPropId, TensorShape shape, ComputeBuffer buffer, Int64 dataOffset = 0)
    {
        func.SetTensor(tensorDecl, dataPropId, shape, buffer, dataOffset);
    }

    public void SetTensorDecl(string name, TensorShape shape, Int64 dataOffset)
    {
        func.SetTensorDecl(name, shape, dataOffset);
    }
    public void SetTensorDecl(ComputeFunc.TensorDecl tensorDecl, TensorShape shape, Int64 dataOffset)
    {
        func.SetTensorDecl(tensorDecl, shape, dataOffset);
    }

    public void SetTensorBuffer(string name, ComputeBuffer buffer)
    {
        func.SetTensorBuffer(name, buffer);
    }
    public void SetTensorBuffer(int propId, ComputeBuffer buffer)
    {
        func.SetTensorBuffer(propId, buffer);
    }

    public void Dispatch()
    {
        func.Dispatch(dispatch);
    }

    const long  InvalidEntry = long.MaxValue;
    internal static long CalculateEntryScore(ComputeShader[] kernels, ComputeKernelLibrary.Entry entry, bool verbose)
    {
        long work = InvalidEntry;
        try
        {
            if (!entry.valid)
                return InvalidEntry;

            // @TODO: @OPTIMIZE: cache threadGroupSize instead of creating ComputeFunc and querying every time
            var fn = new ComputeFunc(kernels, entry.name);

            if (fn.threadGroupSizeX * fn.threadGroupSizeY * fn.threadGroupSizeZ > ComputeInfo.maxComputeWorkGroupSize)
                return InvalidEntry;

            if (entry.strict)
            {
                if (entry.dispatch[0] % fn.threadGroupSizeX != 0 ||
                    entry.dispatch[1] % fn.threadGroupSizeY != 0 ||
                    entry.dispatch[2] % fn.threadGroupSizeZ != 0)
                    return InvalidEntry;
            }

            var x = (long) ComputeFunc.IntDivCeil(entry.dispatch[0], (int) fn.threadGroupSizeX);
            var y = (long) ComputeFunc.IntDivCeil(entry.dispatch[1], (int) fn.threadGroupSizeY);
            var z = (long) ComputeFunc.IntDivCeil(entry.dispatch[2], (int) fn.threadGroupSizeZ);

            if (entry.loopStride == 0 && (x > 65535 || y > 65535 || z > 65535))
            {
                if (verbose)
                    D.LogWarning($"Kernel {entry.name} dispatch arguments out of range (any [{x},{y},{z}] > 65535), skipping..");

                return InvalidEntry;
            }

            work = x * y * z;

            work *= (int) fn.threadGroupSize;
            work = (long) (entry.bigO * work);
        }
        catch (ArgumentException)
        {
            if (verbose)
                D.LogWarning($"Kernel processing failed, skipping {entry.name}");
        }
        return work;
    }

    public static ComputeKernel BestKernel(ComputeShader[] kernels, ComputeKernelLibrary.Entry[] entrees, bool verbose)
    {
        var bestEntry = entrees[0];
        var bestScore = InvalidEntry;
        bool foundKernelWithDevicePriority = false;
        for (int i = 0; i < entrees.Length; i++)
        {
            var score = CalculateEntryScore(kernels, entrees[i], verbose);
            bool entryDevicePriority = entrees[i].devicePriority;

            if (score == InvalidEntry)
                continue;

            // first time we encounter a kernel with device priority
            if (!foundKernelWithDevicePriority && entryDevicePriority)
            {
                bestScore = score;
                bestEntry = entrees[i];
            }
            // compute best entry: sort only on priority kernels (if some exist), else sort on non priority
            else if ( (!foundKernelWithDevicePriority && !entryDevicePriority) || (foundKernelWithDevicePriority && entryDevicePriority))
            {
                bestScore = (score <= bestScore) ? score : bestScore;
                bestEntry = (score <= bestScore) ? entrees[i] : bestEntry;
            }

            foundKernelWithDevicePriority = foundKernelWithDevicePriority || entryDevicePriority;
        }

        if (verbose)
            D.Log(bestEntry.name);

        var func = new ComputeFunc(kernels, bestEntry.name);

        if (bestEntry.loopStride > 0)
        {
            int preferedDispatch = (int)bestEntry.loopStride * (int)func.threadGroupSizeX;
            var kernel = new ComputeKernel(func, new int[] {preferedDispatch, 1, 1});
            kernel.shader.SetInt("_LoopStride", preferedDispatch);
            return kernel;
        }
        else
        {
            return new ComputeKernel(func, bestEntry.dispatch);
        }
    }

}

public class ComputeOps : ReferenceComputeOps
{
    // ---------------------------------------------------------------------------------
    private bool printKernels = false;

    // ---------------------------------------------------------------------------------
    private ComputeShader[] m_Kernels;
    private bool m_Verbose = false;

    public ComputeOps(ComputeShader[] kernels, ComputeShader referenceKernel,  ITensorAllocator allocator = null, bool verbose = false)
    : base(referenceKernel, allocator)
    {
        m_Kernels = kernels;
        m_Verbose = verbose;
    }

    // ---------------------------------------------------------------------------------

    protected ComputeKernel BestKernel(ComputeKernelLibrary.Entry[] entrees)
    {
        return ComputeKernel.BestKernel(m_Kernels, entrees, m_Verbose);
    }

    // ---------------------------------------------------------------------------------
    public override Tensor Dense(Tensor X, Tensor W, Tensor B)
    {
        Assert.IsTrue(W.dimensions <= 2);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(X.flatWidth, W.flatHeight);

        var O = NewTensor(X.flatHeight, W.flatWidth);

        var itemSize = 4; // @TODO: itemSizeInBytes == 2 | float16
        var fn = BestKernel(ComputeKernelLibrary.Dense(X.shape, W.shape, O.shape, itemSize >> 2));

        if (printKernels)
            Debug.Log($"{fn.func.kernelName}: {O.shape} = {X.shape} * {W.shape}" );

        fn.SetTensor("X", X.shape, Pin(X).buffer);
        fn.SetTensor("O", O.shape, Pin(O).buffer);
        fn.SetTensorDecl("W", W.shape, Pin(W).offset);
        fn.SetTensorDecl("B", B.shape, Pin(B).offset);
        Assert.AreEqual(Pin(W).buffer, Pin(B).buffer);
        fn.SetTensorBuffer("WBK", Pin(W).buffer);

        fn.Dispatch();
        return O;
    }

    Tensor Conv2DWinograd(Tensor X, Tensor K, Tensor B, Tensor O, int[] stride, int[] pad)
    {
        Assert.AreEqual(X.channels, K.kernelDepth);
        Assert.AreEqual(K.kernelCount, B.flatWidth);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);

        // Winograd
        // transform kernel
        TensorShape Kws = new TensorShape(K.batch + 1, K.height + 1, K.width, K.channels);
        var fn_wk = new ComputeFunc(m_Kernels, "KernelWinograd_3x3");
        fn_wk.SetTensor("X", K.shape, Pin(K).buffer);

        var Kw = Dispatch(fn_wk, Kws, K.kernelCount, X.channels, 1);

        var fn_w = new ComputeFunc(m_Kernels, "Conv2DWinograd_2x2_3x3");

        SetTensor(fn_w, "X", X);
        SetTensor(fn_w, "K", Kw);
        SetTensor(fn_w, "B", B);

        fn_w.shader.SetInts("_Pad", pad);

        var OW = Dispatch(fn_w, O.shape, Kw.kernelCount, IDivC(O.width, 2), IDivC(O.height, 2));
        return OW;
    }

    public override Tensor Conv2D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad)
    {
        Assert.AreEqual(X.channels, K.kernelDepth);
        Assert.AreEqual(K.kernelCount, B.flatWidth);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);

        var O = NewTensor(X.shape.ApplyKernel(K.shape, stride, pad));
        var fn = BestKernel(ComputeKernelLibrary.Conv2D(X.shape, K.shape, O.shape, stride, pad));

        if (printKernels)
            Debug.Log($"{fn.func.kernelName}: {O.shape} = {X.shape} # {K.shape} stride: {stride[0]},{stride[1]} pad:{pad[0]},{pad[1]}" );

        if (fn.func.kernelName == "Conv2DWinograd_2x2_3x3")
        {
            return Conv2DWinograd(X, K, B, O, stride, pad);
        }

        fn.SetTensor("X", X.shape, Pin(X).buffer);
        fn.SetTensor("O", O.shape, Pin(O).buffer);
        fn.SetTensorDecl("K", K.shape, Pin(K).offset);
        fn.SetTensorDecl("B", B.shape, Pin(B).offset);
        Assert.AreEqual(Pin(K).buffer, Pin(B).buffer);
        fn.SetTensorBuffer("WBK", Pin(K).buffer);

        fn.shader.SetInts("_Pad", pad);
        fn.shader.SetInts("_Stride", stride);

        fn.Dispatch();
        return O;
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

        var O = NewTensor(X.shape.ApplyKernel(K.shape, stride, pad));
        var fn = BestKernel(ComputeKernelLibrary.DepthwiseConv2D(X.shape, K.shape, O.shape));

        if (printKernels)
            Debug.Log($"{fn.func.kernelName}: {O.shape} = {X.shape} ∆ {K.shape} stride: {stride[0]},{stride[1]} pad:{pad[0]},{pad[1]}" );

        fn.SetTensor("X", X.shape, Pin(X).buffer);
        fn.SetTensor("O", O.shape, Pin(O).buffer);
        fn.SetTensorDecl("K", K.shape, Pin(K).offset);
        fn.SetTensorDecl("B", B.shape, Pin(B).offset);
        Assert.AreEqual(Pin(K).buffer, Pin(B).buffer);
        fn.SetTensorBuffer("WBK", Pin(K).buffer);

        fn.shader.SetInts("_Stride", stride);
        fn.shader.SetInts("_Pad", pad);

        fn.Dispatch();
        return O;
    }

    public override Tensor Conv2DTrans(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, int[] outputAdjustment)
    {
        Assert.AreEqual(X.channels, K.kernelDepth);
        Assert.AreEqual(K.kernelCount, B.flatWidth);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);

        var O = NewTensor(X.shape.ApplyKernelInverse(K.shape, stride, pad, outputAdjustment));
        var fn = BestKernel(ComputeKernelLibrary.Conv2DTrans(X.shape, K.shape, O.shape));

        if (printKernels)
            D.Log($"{fn.func.kernelName}: {O.shape} = {X.shape} @ {K.shape} stride: {stride[0]},{stride[1]} pad:{pad[0]},{pad[1]}" );

        pad = new int[]
        {
            K.kernelWidth - pad[0] - 1, K.kernelHeight - pad[1] - 1,
            K.kernelWidth - pad[2] - 1, K.kernelHeight - pad[3] - 1
        };

        fn.SetTensor("X", X.shape, Pin(X).buffer);
        fn.SetTensor("O", O.shape, Pin(O).buffer);
        fn.SetTensorDecl("K", K.shape, Pin(K).offset);
        fn.SetTensorDecl("B", B.shape, Pin(B).offset);
        Assert.AreEqual(Pin(K).buffer, Pin(B).buffer);
        fn.SetTensorBuffer("WBK", Pin(K).buffer);

        fn.shader.SetInts("_Pad", pad);
        fn.shader.SetInts("_Stride", stride);

        fn.Dispatch();

        return O;
    }

    public override Tensor Upsample2D(Tensor X, int[] size)
    {
        Assert.AreEqual(size.Length, 2);

        var O = NewTensor(X.batch, X.height*size[1], X.width*size[0], X.channels);
        var fn = BestKernel(ComputeKernelLibrary.Upsample2D(X.shape, O.shape));

        if (printKernels)
            D.Log($"{fn.func.kernelName}: {O.shape} = {X.shape} ^ size: {size[0]},{size[1]}" );

        fn.SetTensor("X", X.shape, Pin(X).buffer);
        fn.SetTensor("O", O.shape, Pin(O).buffer);

        fn.shader.SetInts("_Pool", size);

        fn.Dispatch();
        return O;
    }

    protected override Tensor Pool2D(string kernelName, Tensor X, int[] pool, int[] stride, int[] pad)
    {
        Assert.AreEqual(pool.Length, 2);
        Assert.AreEqual(stride.Length, 2);

        if (pad[0] == 0 && pad[1] == 0 && pad[2] == 0 && pad[3] == 0)
            kernelName += "_NoPads";

        var O = NewTensor(X.shape.ApplyPool(pool, stride, pad));
        var fn = BestKernel(ComputeKernelLibrary.Pool2D(X.shape, O.shape, kernelName));

        if (printKernels)
            D.Log($"{fn.func.kernelName}: {O.shape} = {X.shape} ^ pool: {pool[0]},{pool[1]} stride: {stride[0]},{stride[1]} pad:{pad[0]},{pad[1]}" );

        fn.SetTensor("X", X.shape, Pin(X).buffer);
        fn.SetTensor("O", O.shape, Pin(O).buffer);

        fn.shader.SetInts("_Pool", pool);
        fn.shader.SetInts("_Stride", stride);
        fn.shader.SetInts("_Pad", pad);

        fn.Dispatch();
        return O;
    }

    public override Tensor GlobalMaxPool2D(Tensor X)
    {
        return GlobalPool2D("MaxPool2D", "GlobalMaxPool2D", X);
    }

    public override Tensor GlobalAvgPool2D(Tensor X)
    {
        return GlobalPool2D("AvgPool2D", "GlobalAvgPool2D", X);
    }

    public override Tensor GlobalAvgVariancePool2D(Tensor X)
    {
        var O = NewTensor(X.batch, 2, 1, X.channels);
        var fn = BestKernel(ComputeKernelLibrary.GlobalPool2D(X.shape, O.shape, "GlobalAvgVariancePool2D"));

        fn.SetTensor("X", X.shape, Pin(X).buffer);
        fn.SetTensor("O", O.shape, Pin(O).buffer);

        fn.Dispatch();
        return O;
    }

    protected virtual Tensor GlobalPool2D(string smallKernelName, string globalKernelName, Tensor X)
    {
        // downsample with pyramid approach
        while (X.height * X.width >= 256)
        {
            var pool = new [] {4, 4};
            var stride = pool;
            var noPad = new[] {0, 0, 0, 0};

            var lastLength = X.length;
            X = Pool2D(smallKernelName, X, pool, stride, noPad);
            Assert.IsTrue(X.length < lastLength);
        }

        var O = NewTensor(X.batch, 1, 1, X.channels);
        var fn = BestKernel(ComputeKernelLibrary.GlobalPool2D(X.shape, O.shape, globalKernelName));

        fn.SetTensor("X", X.shape, Pin(X).buffer);
        fn.SetTensor("O", O.shape, Pin(O).buffer);

        fn.Dispatch();
        return O;
    }

    public override Tensor ScaleBias(Tensor X, Tensor S, Tensor B)
    {
        Assert.AreEqual(X.channels, B.channels); Assert.AreEqual(X.channels, S.channels);
        Assert.AreEqual(B.length, B.channels); Assert.AreEqual(S.length, S.channels);

        var O = NewTensor(X.shape);
        var fn = BestKernel(ComputeKernelLibrary.ScaleBias(X.shape, O.shape));

        if (printKernels)
            D.Log(fn.func.kernelName);

        fn.SetTensor("X", X.shape, Pin(X).buffer);
        fn.SetTensor("O", O.shape, Pin(O).buffer);
        fn.SetTensorDecl("W", S.shape, Pin(S).offset);
        fn.SetTensorDecl("B", B.shape, Pin(B).offset);
        Assert.AreEqual(Pin(S).buffer, Pin(B).buffer);
        fn.SetTensorBuffer("WBK", Pin(S).buffer);

        fn.Dispatch();
        return O;
    }

    public override Tensor Normalization(Tensor X, Tensor S, Tensor B, int pool, int axis, float epsilon)
    {
        if (axis != 3 && axis != -1)
            throw new NotImplementedException();

        if (pool <= 0)
            pool = X.batch;

        if (pool > 1)
            throw new NotImplementedException(); // @TODO: support other types of Normalization at test time
                                                 // Currently supported only pool=1 (InstanceNormalization)

        var meanVariance = GlobalAvgVariancePool2D(X);

        var O = NewTensor(X.shape);
        var fn = BestKernel(ComputeKernelLibrary.NormalizationTail(X.shape, O.shape));
        fn.SetTensor("X", X.shape, Pin(X).buffer);
        fn.SetTensor("O", O.shape, Pin(O).buffer);
        fn.SetTensor("W", meanVariance.shape, Pin(meanVariance).buffer);

        fn.shader.SetFloat("_Epsilon", epsilon);

        fn.Dispatch();

        return ScaleBias(O, S, B);
    }

    protected override Tensor Activation(string kernelName, Tensor X, float alpha = 0f, float beta = 0f)
    {
        var O = NewTensor(X.shape);
        var fn = BestKernel(ComputeKernelLibrary.Activation(X.shape, O.shape, kernelName));

        if (printKernels)
            D.Log(fn.func.kernelName);

        fn.SetTensor("X", X.shape, Pin(X).buffer);
        fn.SetTensor("O", O.shape, Pin(O).buffer);

        fn.shader.SetFloat("_Alpha", alpha);
        fn.shader.SetFloat("_Beta",  beta);

        fn.Dispatch();
        return O;
    }

    public override Tensor PRelu(Tensor X, Tensor S)
    {
        Assert.IsTrue((X.flatWidth == S.flatWidth) || (S.flatWidth == 1));

        var O = NewTensor(X.shape);
        var fn = BestKernel(ComputeKernelLibrary.PRelu(X.shape, O.shape));

        if (printKernels)
            D.Log(fn.func.kernelName);

        fn.SetTensor("X", X.shape, Pin(X).buffer);
        fn.SetTensor("O", O.shape, Pin(O).buffer);
        fn.SetTensor("W", S.shape, Pin(S).buffer);

        fn.Dispatch();
        return O;
    }

    public override Tensor Softmax(Tensor X)
    {
        var O = NewTensor(X.shape);
        var fn = BestKernel(ComputeKernelLibrary.Softmax(X.shape, O.shape));

        fn.SetTensor("X", X.shape, Pin(X).buffer);
        fn.SetTensor("O", O.shape, Pin(O).buffer);

        fn.Dispatch();
        return O;
    }

    public override Tensor LogSoftmax(Tensor X)
    {
        var O = NewTensor(X.shape);
        var fn = BestKernel(ComputeKernelLibrary.LogSoftmax(X.shape, O.shape));

        fn.SetTensor("X", X.shape, Pin(X).buffer);
        fn.SetTensor("O", O.shape, Pin(O).buffer);

        fn.Dispatch();
        return O;
    }

    // @TODO: implement Dropout in terms of RandomUniform by preparing random values on CPU upfront and multiplying result on GPU later on
    // public override Tensor Dropout(Tensor X, float alpha)

    private UnityEngine.Random.State[] m_RandomNormalSeed;
    public override Tensor RandomNormal(TensorShape s, float mean, float scale, int seed)
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
    public override Tensor RandomUniform(TensorShape s, float mean, float scale, int seed)
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


    public override Tensor Concat(Tensor[] tensors, int axis)
    {
        var O = NewTensor(TensorExtensions.Concat(tensors.Select(t => t.shape).ToArray(), axis));

        var offsets = new int[] { 0,0,0,0 };
        axis = O.shape.Axis(axis);

        foreach (var X in tensors)
        {
            var fn = BestKernel(ComputeKernelLibrary.Copy(X.shape, O.shape));

            fn.SetTensor("X", X.shape, Pin(X).buffer);
            fn.SetTensor("O", O.shape, Pin(O).buffer);

            fn.shader.SetInts("_Pad", offsets);

            fn.Dispatch();

            offsets[axis] += X.shape[axis];
        }

        return O;
    }

    public override Tensor ElementwiseWithBroadcast(string kernelName, Tensor[] tensors)
    {
        Assert.IsTrue(tensors.Length > 0);

        Tensor outputTensor1 = NewTensor(TensorExtensions.MaxShape(tensors));
        Tensor outputTensor2 = null;
        if (tensors.Length > 2)
            outputTensor2 = NewTensor(TensorExtensions.MaxShape(tensors));

        var X = tensors[0];
        var fn = BestKernel(ComputeKernelLibrary.Broadcast(X.shape, outputTensor1.shape, kernelName));

        Tensor O = null;
        for (int t = 1; t < tensors.Length; ++t)
        {
            var B = tensors[t];
            O = (t%2 == 1)?outputTensor1:outputTensor2;
            fn.SetTensor("X", X.shape, Pin(X).buffer);
            fn.SetTensor("O", O.shape, Pin(O).buffer);
            fn.SetTensor("B", B.shape, Pin(B).buffer, Pin(B).offset);

            fn.Dispatch();

            X = O;
        }

        return O;
    }

    protected override Tensor ApplyPadding(Tensor X, int[] pad, string kernelName, float constant = 0.0f)
    {
        Assert.AreEqual(pad.Length, 4);

        var O = NewTensor(X.shape.ApplyBorder(pad));
        var fn = BestKernel(ComputeKernelLibrary.Padding(X.shape, O.shape, kernelName));

        fn.SetTensor("X", X.shape, Pin(X).buffer);
        fn.SetTensor("O", O.shape, Pin(O).buffer);

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

        fn.Dispatch();
        return O;
    }

    public override Tensor LogicalNot(Tensor X)
    {
        var O = NewTensor(X.shape);
        var fn = BestKernel(ComputeKernelLibrary.Activation(X.shape, O.shape, "LogicalNot"));

        fn.SetTensor("X", X.shape, Pin(X).buffer);
        fn.SetTensor("O", O.shape, Pin(O).buffer);

        fn.Dispatch();
        return O;
    }
}

public class ComputeVarsWithSharedModel : DefaultVars
{
    private Dictionary<string, ComputeBuffer> m_ModelBuffers = new Dictionary<string, ComputeBuffer>();
    private Dictionary<string, Int64> m_OffsetsIntoModelWeights = new Dictionary<string, long>();

    public override void Dispose()
    {
        base.Dispose();

        foreach (var key in m_ModelBuffers.Keys)
            m_ModelBuffers[key].Dispose();
        m_ModelBuffers.Clear();
        m_OffsetsIntoModelWeights.Clear();
    }

    protected override Tensor[] PrepareLayerInputTensors(Model model, Layer layer, IOps ops)
    {
        var tensorIndex = 0;
        var tensors = new Tensor[layer.inputs.Length + layer.datasets.Length];

        foreach (var name in layer.inputs)
        {
            var tensor = new Tensor(1, 1, 1, 1, m_StringCache.Lookup(layer.name, "_dummy_in", tensorIndex));
            tensors[tensorIndex++] = tensor;
        }

        Int64 offsetIntoModelWeights = m_OffsetsIntoModelWeights.ContainsKey(layer.name) ?
                                       m_OffsetsIntoModelWeights[layer.name]: 0;
        ComputeBuffer buffer = m_ModelBuffers.ContainsKey(layer.name) ? m_ModelBuffers[layer.name] : null;

        if (buffer == null)
        {
            buffer = CreateComputeBufferForModelTensors(layer, out offsetIntoModelWeights);
            if (buffer != null)
            {
                m_ModelBuffers[layer.name] = buffer;
                m_OffsetsIntoModelWeights[layer.name] = offsetIntoModelWeights;
            }
        }

        foreach (var arg in layer.datasets)
        {
            Assert.IsNotNull(buffer);
            var tensor = new Tensor(arg.shape,
                new SharedComputeTensorData(buffer, arg.shape, (int)(arg.offset - offsetIntoModelWeights)),
                m_StringCache.Lookup(layer.name, "_arg", tensorIndex));
            tensors[tensorIndex++] = tensor;
            m_ModelTensors.Add(tensor);
        }

        Assert.AreEqual(tensorIndex, tensors.Length);
        return tensors;
    }

    protected ComputeBuffer CreateComputeBufferForModelTensors(Layer layer, out Int64 offsetIntoModelWeights)
    {
        Int64 minOffset = layer.weights.LongLength;
        Int64 maxOffset = 0;
        foreach (var t in layer.datasets)
        {
            minOffset = Math.Min(minOffset, t.offset);
            maxOffset = Math.Max(maxOffset, t.offset + t.length);
        }
        var length = Convert.ToInt32(maxOffset - minOffset);
        if (length <= 0)
        {
            offsetIntoModelWeights = 0;
            return null;
        }

        var buffer = new ComputeBuffer(length, sizeof(float));
        // @WARN: looks like Unity ComputeBuffer.SetData API take "computeBufferStartIndex" and "length" arguments in floats, instead of buffer element size aka stride
        // as would be expected per API documentation
        // @TODO: bugreport documentation discrepancy!
        offsetIntoModelWeights = minOffset;
        buffer.SetData(layer.weights, Convert.ToInt32(offsetIntoModelWeights), 0, length);
        return buffer;
    }
}

} // namespace Barracuda
