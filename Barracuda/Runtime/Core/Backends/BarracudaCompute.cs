using UnityEngine;
using UnityEngine.Assertions;
using System;
using System.Collections.Generic;
using Unity.Collections;

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

namespace Unity.Barracuda {

internal sealed class ComputeKernelLibrary
{
    static private StringCache s_StringCache = new StringCache();
    static private List<Entry> s_DenseFP16Entries = new List<Entry>(1);
    static private List<Entry> s_DenseFP32Entries = new List<Entry>(10);
    static public List<Entry> Dense(TensorShape X, TensorShape W, TensorShape O, int type)
    {
        var h = O.flatHeight;
        var w = O.flatWidth;

        var entries = type > 0 ? s_DenseFP32Entries : s_DenseFP16Entries;
        entries.Clear();

        if (type == 0) // FP16
        {
            entries.Add(new Entry("DenseFP16Div2",
                    Int3(w / 2, h),                                 BigO(X.flatWidth)
                    // @TODO: w % 2 == 0
            ));
        }
        else // FP32
        {
            entries.Add(new Entry("Dense_Tilled2x2_Cached",
                    Int3(ComputeHelper.IDivC(w, 2), ComputeHelper.IDivC(h, 2)),                                 BigO(X.flatWidth)/2,
                    StrictAnd(w % 2 == 0 && h % 2 == 0 && X.flatWidth % 32 == 0),
                (Application.platform == RuntimePlatform.Android) ||
                (Application.platform == RuntimePlatform.IPhonePlayer) ||
                (ComputeInfo.graphicsDeviceVendor.Contains("Intel"))
            ));
            entries.Add(new Entry("Dense_Tilled4x4_Cached",
                    Int3(ComputeHelper.IDivC(w, 4), ComputeHelper.IDivC(h, 4)),                                 BigO(X.flatWidth)/4,
                    StrictAnd(w % 4 == 0 && h % 4 == 0 && X.flatWidth % 32 == 0),
                (Application.platform == RuntimePlatform.Android) ||
                (Application.platform == RuntimePlatform.IPhonePlayer) ||
                (ComputeInfo.graphicsDeviceVendor.Contains("Intel"))
            ));
            entries.Add(new Entry("Dense_T8x8_R8x8",
                    Int3(w / 8, h / 8),                             BigO(X.flatWidth)/8,
                    StrictAnd(w % 64 == 0 && h % 64 == 0 && X.flatWidth % 64 == 0)
            ));
            entries.Add(new Entry("Dense_T16x16_R4x4",
                    Int3(w / 4, h / 4),                             BigO(X.flatWidth)/4,
                    StrictAnd(w % 64 == 0 && h % 64 == 0 && X.flatWidth % 64 == 0)
            ));
            entries.Add(new Entry("Dense_T8x8_R4x4",
                    Int3(w / 4, h / 4),                             BigO(X.flatWidth)/4,
                    StrictAnd(w % 32 == 0 && h % 32 == 0 && X.flatWidth % 32 == 0)
            ));

                // old
            entries.Add(
                new Entry("DenseTiled64x64",
                    Int3(w / 4, h / 4),                             BigO(X.flatWidth)*1.33f/4,
                    StrictAnd(w % 4 == 0 && h % 4 == 0
                        && X.flatWidth % 64 == 0 && ComputeInfo.supportsDense64x64)
                ));
            entries.Add(new Entry("DenseTiled32x32",
                    Int3(w / 2, h / 2),                             BigO(X.flatWidth)*1.33f/2,
                    StrictAnd(w % 2 == 0 && h % 2 == 0
                        && X.flatWidth % 32 == 0 && ComputeInfo.supportsDense32x32)
            ));
            entries.Add(new Entry("DenseTiled16x16",
                    Int3(w, h),                                     BigO(X.flatWidth)*1.33f,
                    StrictAnd(X.flatWidth % 16 == 0)
                    // @TODO: relax Strict constraint, only And part should be necessary due to mask
            ));
            entries.Add(new Entry("Dense_L1Cached64",
                    Int3(w, h),                                     BigO(X.flatWidth)
            ));
        }

        return entries;
    }

    private enum ChannelMode
    {
        Strict,
        Lax
    }

    private enum KernelMode
    {
        Strict,
        Lax
    }

    private const int k_MinimumThreads = 4096;//Heuristic to try to avoid R8x8 path when number of GPU threads would be to low for parallelism.
    private const int k_MinimumKernelCountForT8x8_R8x8 = 32;
    private const int k_MinimumPixelCountForT8x8_R8x8 = 64;
    private const int k_MinimumPixelCountForT2x32_R8x8 = k_MinimumPixelCountForT8x8_R8x8 * 4;//T2_32 consume 4x more pixels per TG than T8x8
    private static bool IsT8x8_R8x8KernelValid(ChannelMode channelMode, KernelMode kernelMode, int c, int k, int h, int w, int n)
    {
        bool valid;
        if (ComputeInfo.channelsOrder == ComputeInfo.ChannelsOrder.NCHW)
        {
            valid = ComputeInfo.supportsComputeSharedMemory;
            if (channelMode==ChannelMode.Strict)
                valid &= (c % 8) == 0;

            if (kernelMode==KernelMode.Strict)
                valid &= (k % 64) == 0;
            else
                valid &= (k % 16) == 0;
        }
        else
        {
            //Conv2DKernelKxK_StrictC4K16_T8x8_R8x8 is only enabled in NCHW mode.
            //The kernel was tested to be faster than R4x4 at various workload in NHWC too. However to avoid
            //any potential regression and maintenance, the NHWC path is disabled of this kernel is disabled.
            valid = false;
        }

        //Performance wise this kernel will drop fast when k < 64 or w*h < 64.
        valid &= k >= k_MinimumKernelCountForT8x8_R8x8;
        valid &= (w*h) >= k_MinimumPixelCountForT8x8_R8x8;

        //If this kernel can't go wide enough we will probably waste GPU parallelism should prefer another kernel.
        int numThreadsR8x8 = ComputeHelper.IDivC(k,8 ) * ComputeHelper.IDivC(w * h , 8) * n;
        valid &= numThreadsR8x8 >= k_MinimumThreads;

        //valid &= (h*w) > (64*64);

        return valid;
    }

    private static bool IsT2x32_R8x8KernelValid(ChannelMode channelMode, KernelMode kernelMode, int c, int k, int h, int w, int n)
    {
        bool valid;
        if (ComputeInfo.channelsOrder == ComputeInfo.ChannelsOrder.NCHW)
        {
            valid = ComputeInfo.supportsComputeSharedMemory;
            if (channelMode==ChannelMode.Strict)
                valid &= (c % 4) == 0;

            if (kernelMode == KernelMode.Strict)
            {
                valid &= (k % 16) == 0;
            }
        }
        else
        {
            //Conv2DKernelKxK_StrictC4K16_T2x32_R8x8 Only viable in NCHW mode perf wise.
            valid = false;
        }

        //Performance wise this kernel will drop fast when h*w < 128*128.
        valid &= (h*w) > k_MinimumPixelCountForT2x32_R8x8;

        //If this kernel can't go wide enough we will probably waste GPU parallelism should prefer another kernel.
        int numThreadsR8x8 = ComputeHelper.IDivC(k,8 ) * ComputeHelper.IDivC(w * h , 8) * n;
        valid &= numThreadsR8x8 >= k_MinimumThreads;

        return valid;
    }

    private static bool IsWinograd16x16_R4x4KernelValid(ChannelMode channelMode, KernelMode kernelMode, int c, int k, int h, int w, int n)
    {
        bool valid = (ComputeInfo.channelsOrder == ComputeInfo.ChannelsOrder.NCHW); // NHWC not implemented

        valid &= ComputeInfo.supportsComputeSharedMemory;
        if (channelMode == ChannelMode.Strict)
            valid &= (c % 8) == 0;

        if (kernelMode == KernelMode.Strict)
            valid &= (k % 16) == 0;

        bool isMobile = (Application.platform == RuntimePlatform.Android) || (Application.platform == RuntimePlatform.IPhonePlayer);
        bool isOSX = (Application.platform == RuntimePlatform.OSXEditor) || (Application.platform == RuntimePlatform.OSXPlayer);
        bool isIntelUHD = ComputeInfo.graphicsDeviceVendor.Contains("Intel");
        // winograd always better on these platforms
        if (isMobile || isOSX || isIntelUHD)
            return valid;

        // Performance wise this kernel is less efficient than T8x8_R8x8 for lower channels count and big pixel dims
        if ((k % 64) == 0)
            valid &= (c >= 64) || (h*w <= 128*128);

        return valid;
    }

    private static List<Entry> s_Conv3DEntries = new List<Entry>(4);
    internal static List<Entry> Conv3D(TensorShape X, TensorShape K, TensorShape O, int[] stride, int[] pad)
    {
        var n = O.batch;
        var d = O.depth;
        var h = O.height;
        var w = O.width;
        var k = K.kernelCount;
        var c = X.channels;

        var entries = s_Conv3DEntries;
        entries.Clear();

        entries.Add(new Entry("Conv3D",
            Int3(k, w, h), BigO(O.batch * X.depth * X.channels)));

        entries.Add(new Entry("Conv3DKernelKxK_LaxC8LaxK32_T8x16_R4x4",
            Int3(ComputeHelper.IDivC(k, 4), ComputeHelper.IDivC(d*w*h, 4), n), BigO(X.channels) * 0.8f,
            valid_: (k>=8) && ComputeInfo.supportsComputeSharedMemory));

        entries.Add(new Entry("Conv3DKernelKxK_StrictC8LaxK32_T8x16_R4x4",
            Int3(ComputeHelper.IDivC(k, 4), ComputeHelper.IDivC(d*w*h, 4), n), BigO(X.channels) * 0.7f,
            valid_: (c % 8 == 0) && (k>=8) && ComputeInfo.supportsComputeSharedMemory));

        entries.Add(new Entry("Conv3DKernelKxK_StrictC8StrictK32_T8x16_R4x4",
            Int3(ComputeHelper.IDivC(k, 4), ComputeHelper.IDivC(d*w*h, 4), n), BigO(X.channels) * 0.6f,
            valid_: (c % 8 == 0) && (k % 32 == 0) && ComputeInfo.supportsComputeSharedMemory));

        return entries;
    }

    private static List<Entry> s_Conv2DEntries = new List<Entry>(16);
    internal static List<Entry> Conv2D(TensorShape X, TensorShape K, TensorShape O, int[] stride, int[] pad)
    {
        var n = O.batch;
        var h = O.height;
        var w = O.width;
        var k = K.kernelCount;
        var c = X.channels;

        var entries = s_Conv2DEntries;
        entries.Clear();

        // Winograd
        // R4x4_T16x16 : R4x4 T16x(4x4)
        entries.Add(new Entry("Conv2DWinograd_2x2_Kernel3x3_StrictC8StrictK16_T16x16_R4x4",
                Int3(16*16 * ComputeHelper.IDivC(k, 16), ComputeHelper.IDivC(ComputeHelper.IDivC(w, 2) * ComputeHelper.IDivC(h, 2), 16), n),      BigO(X.channels) * (0.8f / 64) * (1.0f/2.25f),
                 valid_: K.kernelWidth == 3 && K.kernelHeight == 3 &&
                         stride[0] == 1 && stride[1] == 1 &&
                         IsWinograd16x16_R4x4KernelValid(ChannelMode.Strict, KernelMode.Strict, c, k, h, w, n)));
        entries.Add(new Entry("Conv2DWinograd_2x2_Kernel3x3_StrictC8LaxK16_T16x16_R4x4",
                Int3(16*16 * ComputeHelper.IDivC(k, 16), ComputeHelper.IDivC(ComputeHelper.IDivC(w, 2) * ComputeHelper.IDivC(h, 2), 16), n),      BigO(X.channels) * (0.9f / 64) * (1.0f/2.25f),
                 valid_: K.kernelWidth == 3 && K.kernelHeight == 3 &&
                         stride[0] == 1 && stride[1] == 1 &&
                         IsWinograd16x16_R4x4KernelValid(ChannelMode.Strict, KernelMode.Lax, c, k, h, w, n)));
        // R8x8_16k
        entries.Add(
            new Entry("Conv2DKernelKxK_LaxC4StrictK16_T2x32_R8x8",
                Int3(ComputeHelper.IDivC(k, 8), ComputeHelper.IDivC(w*h, 8), n),      BigO(X.channels) * 1.3f,
                 valid_: IsT2x32_R8x8KernelValid(ChannelMode.Lax,KernelMode.Strict,c,k,h,w,n)));

        entries.Add(new Entry("Conv2DKernelKxK_StrictC4LaxK16_T2x32_R8x8",
                Int3(ComputeHelper.IDivC(k, 8), ComputeHelper.IDivC(w*h, 8), n),      BigO(X.channels) * 1.2f,
                 valid_: IsT2x32_R8x8KernelValid(ChannelMode.Strict,KernelMode.Lax,c,k,h,w,n)));

        entries.Add(new Entry("Conv2DKernelKxK_StrictC4StrictK16_T2x32_R8x8",
                Int3(ComputeHelper.IDivC(k, 8), ComputeHelper.IDivC(w*h, 8), n),      BigO(X.channels) * 1.1f,
                 valid_: IsT2x32_R8x8KernelValid(ChannelMode.Strict,KernelMode.Strict,c,k,h,w,n)));

        // R8x8_64k
        entries.Add(new Entry("Conv2DKernelKxK_StrictC16StrictK64_T8x8_R8x8",
                Int3(ComputeHelper.IDivC(k, 8), ComputeHelper.IDivC(w*h, 8), n),      BigO(X.channels) * 0.7f,
                 valid_: IsT8x8_R8x8KernelValid(ChannelMode.Strict, KernelMode.Strict,c,k,h,w,n)));

        entries.Add(new Entry("Conv2DKernelKxK_StrictC16LaxK64_T8x8_R8x8",
                Int3(ComputeHelper.IDivC(k, 8), ComputeHelper.IDivC(w*h, 8), n),      BigO(X.channels) * 0.75f,
                 valid_: IsT8x8_R8x8KernelValid(ChannelMode.Strict, KernelMode.Lax,c,k,h,w,n)));

        // R4x4
        int r4x4dispatchY = (ComputeInfo.channelsOrder == ComputeInfo.ChannelsOrder.NHWC) ? n * w * h : w * h;
        int r4x4dispatchZ = (ComputeInfo.channelsOrder == ComputeInfo.ChannelsOrder.NHWC) ? 1 : n;
        entries.Add(new Entry("Conv2DKernel1x1_StrictC16K64_T16x16_R4x4",
                Int3(ComputeHelper.IDivC(k, 4), ComputeHelper.IDivC(r4x4dispatchY, 4), r4x4dispatchZ),    BigO(X.channels) * 0.8f / 4,
                K.kernelWidth == 1 && K.kernelHeight == 1 &&
                stride[0] == 1 && stride[1] == 1 &&
                (k % 64) == 0 && (c % 16) == 0 &&
                ComputeInfo.supportsComputeSharedMemory));

        entries.Add(new Entry("Conv2DKernelKxK_StrictC16K64_T16x16_R4x4",
                Int3(ComputeHelper.IDivC(k, 4), ComputeHelper.IDivC(r4x4dispatchY, 4), r4x4dispatchZ),    BigO(X.channels) * 0.9f / 4,
                (k % 64) == 0 && (c % 16) == 0 && ComputeInfo.supportsComputeSharedMemory));

        entries.Add(new Entry("Conv2DKernelKxK_T16x16_R4x4",
                Int3(ComputeHelper.IDivC(k, 4), ComputeHelper.IDivC(r4x4dispatchY, 4), r4x4dispatchZ),    BigO(X.channels) * 1.0f / 4,
                k >= 16 && c >= 16 && ComputeInfo.supportsComputeSharedMemory));
//      entries.Add(new Entry("Conv2DKernelKxK_T16x16_R4x4",
//                Int3(ComputeHelper.IDivC(k, 4), ComputeHelper.IDivC(n*w*h, 4)),                 BigO(X.channels) * 1.1f / 4));

        // Old
//        entries.Add(new Entry("Conv2D_L1Cached64_RegisterBlock4x4",
//                Int3(K.kernelCount, w/4+1, h/4+1),                  BigO(O.batch * X.channels) * 1.1f / 4,
//                (k % 64) == 0 && (c % 64) == 0 && ComputeInfo.supportsComputeSharedMemory));
//
//        entries.Add(new Entry("Conv2D_L1Cached32_RegisterBlock4x4",
//                Int3(K.kernelCount, w/4+1, h/4+1),                  BigO(O.batch * X.channels) / 3,
//            (k % 32) == 0 && (c % 32) == 0 && ComputeInfo.supportsComputeSharedMemory));

        entries.Add(new Entry("Conv2D_RegisterBlock4x2",
                Int3(K.kernelCount, w/4, h/2),                      BigO(O.batch * X.channels) * 1.1f / 2,
                StrictAnd(
                (w % 4) == 0 && (h % 2) == 0)));

        entries.Add(new Entry("Conv2D",
            Int3(k, w, h), BigO(O.batch * X.channels)));

        return entries;
    }

    private static List<Entry> s_DepthwiseConv2DEntries = new List<Entry>(1);
    internal static List<Entry> DepthwiseConv2D(TensorShape X, TensorShape K, TensorShape O)
    {
        var h = O.height;
        var w = O.width;

        var entries = s_DepthwiseConv2DEntries;
        entries.Clear();

        entries.Add(new Entry("DepthwiseConv2D",
                Int3(K.kernelCount, w, h), BigO(O.batch * X.channels)));

        return entries;
    }

    private static List<Entry> s_Conv2DTransEntries = new List<Entry>(2);
    internal static List<Entry> Conv2DTrans(TensorShape X, TensorShape K, TensorShape O)
    {
        var entries = s_Conv2DTransEntries;
        entries.Clear();

        entries.Add(new Entry("Conv2DTrans_KernelCached_K5x5_T16x16",
                dispatch_: Int3(K.kernelCount, O.width, O.height), bigO_: BigO(O.batch * O.channels * X.channels) / 3,
            valid_: (X.channels <= 256 && K.kernelHeight <= 5 && K.kernelWidth <= 5)));

        entries.Add(new Entry("Conv2DTrans",
            dispatch_: Int3(K.kernelCount, O.width, O.height), bigO_: BigO(O.batch * O.channels * X.channels)));

        return entries;
    }

    private static List<Entry> s_ActivationEntries = new List<Entry>(3);
    internal static List<Entry> Activation(TensorShape X, TensorShape O, string kernelName)
    {
        var entries = s_ActivationEntries;
        entries.Clear();

        entries.Add(new Entry(s_StringCache.Lookup(kernelName, "_FlatStrict"),
                dispatch_: Int3(O.length/2),
                bigO_: 0.8f* BigO(1),
            strictDims: StrictAnd(O.length % 128 == 0)));

        entries.Add( new Entry(s_StringCache.Lookup(kernelName, "_Flat"),
                dispatch_: Int3(O.length),
            bigO_: BigO(1)));

        entries.Add(new Entry(s_StringCache.Lookup(kernelName, "_Loop"),
                dispatch_: Int3(O.length),
                bigO_: BigO(2),
            loopStride_: 256));

        return entries;
    }

    private static List<Entry> s_PReluEntries = new List<Entry>(3);
    internal static List<Entry> PRelu(TensorShape X, TensorShape O)
    {
        var entries = s_PReluEntries;
        entries.Clear();

        entries.Add(new Entry("PRelu_CNyx2",
            Int3(O.channels, O.batch * O.height * O.width), 1.0f, ComputeInfo.channelsOrder==ComputeInfo.ChannelsOrder.NHWC));

        entries.Add(new Entry("PRelu_Flat",
            Int3(O.length)));

        entries.Add(new Entry("PRelu_Loop",
            Int3(O.length), BigO(2), 256));

        return entries;
    }

    private static List<Entry> s_SoftmaxEntries = new List<Entry>(1);
    internal static List<Entry> Softmax(TensorShape X, TensorShape O)
    {
        var entries = s_SoftmaxEntries;
        entries.Clear();

        entries.Add(new Entry("Softmax",
            Int3(O.flatWidth, O.flatHeight)));

        return entries;
    }

    private static List<Entry> s_LogSoftmaxEntries = new List<Entry>(1);
    internal static List<Entry> LogSoftmax(TensorShape X, TensorShape O)
    {
        var entries = s_LogSoftmaxEntries;
        entries.Clear();

        entries.Add(new Entry("LogSoftmax",
            Int3(O.flatWidth, O.flatHeight)));

        return entries;
    }

    private static List<Entry> s_ScaleBiasEntries = new List<Entry>(3);
    internal static List<Entry> ScaleBias(TensorShape X, TensorShape O)
    {
        var entries = s_ScaleBiasEntries;
        entries.Clear();

        entries.Add(new Entry("ScaleBias_CNyx2",
            Int3(O.channels, O.batch * O.height * O.width), 1.0f, ComputeInfo.channelsOrder==ComputeInfo.ChannelsOrder.NHWC));

        entries.Add(new Entry("ScaleBias_Flat",
            Int3(O.length)));

        entries.Add(new Entry("ScaleBias_Loop",
            Int3(O.length), BigO(2), 256));

        return entries;
    }

    private static List<Entry> s_Upsample2DEntries = new List<Entry>(2);
    internal static List<Entry> Upsample2D(TensorShape X, TensorShape O, int[] scale, bool bilinear)
    {
        var entries = s_Upsample2DEntries;
        entries.Clear();

        if (bilinear)
        {
            entries.Add(
                new Entry("UpsampleBilinear2D_2x2",
                    Int3(O.width, O.height, O.channels), BigO(O.batch) * 0.8f,
                    (scale[0] == 2 && scale[1] == 2)));
            entries.Add(
                new Entry("UpsampleBilinear2D",
                    Int3(O.channels, O.width, O.height), BigO(O.batch)));
        }
        else
        {
            entries.Add(
                // NOTE: dispatched over X (not O)
                new Entry("Upsample2D",
                    Int3(X.channels, X.width, X.height), BigO(X.batch)));
        }

        return entries;
    }

    private static List<Entry> s_Pool2DReduceEntries = new List<Entry>(1);
    internal static List<Entry> Pool2DReduce(TensorShape X, TensorShape O, string kernelName)
    {
        var entries = s_Pool2DReduceEntries;
        entries.Clear();

        entries.Add(new Entry(kernelName,
            Int3(O.channels, ComputeHelper.IDivC(X.width, 2), ComputeHelper.IDivC(X.height, 2)), BigO(O.batch)));

        return entries;
    }

    private static List<Entry> s_Pool2DEntries = new List<Entry>(1);
    internal static List<Entry> Pool2D(TensorShape X, TensorShape O, string kernelName)
    {
        var entries = s_Pool2DEntries;
        entries.Clear();

        entries.Add(
            //new Entry(kernelName + "_16x4x4",
            //    Int3(O.channels, O.width, O.height),            BigO(O.batch)
            //),
            new Entry(kernelName,
                Int3(O.channels, O.width, O.height), BigO(O.batch)));

        return entries;
    }

    private static List<Entry> s_PoolAvgVar2DEntries = new List<Entry>(1);
    internal static List<Entry> PoolAvgVar2D(TensorShape X, TensorShape O, string kernelName)
    {
        var entries = s_PoolAvgVar2DEntries;
        entries.Clear();

        entries.Add(
            //new Entry(kernelName + "_16x4x4",
            //    Int3(O.channels, O.width, O.height),            BigO(O.batch)
            //),
            new Entry(kernelName,
                Int3(O.channels, ComputeHelper.IDivC(X.width, 2), ComputeHelper.IDivC(X.height, 2)), BigO(O.batch)));

        return entries;
    }

    private static List<Entry> s_GlobalPool2DEntries = new List<Entry>(1);
    internal static List<Entry> GlobalPool2D(TensorShape X, TensorShape O, string kernelName)
    {
        var entries = s_GlobalPool2DEntries;
        entries.Clear();

        entries.Add(new Entry(kernelName,
            Int3(O.channels), BigO(O.batch)));

        return entries;
    }

    private static List<Entry> s_NormalizationTailEntries = new List<Entry>(3);
    internal static List<Entry> NormalizationTail(TensorShape X, TensorShape O)
    {
        var entries = s_NormalizationTailEntries;
        entries.Clear();

        entries.Add(new Entry("InstanceNormTail_CNyx2",
            Int3(O.channels, O.batch * O.height * O.width), 1.0f, ComputeInfo.channelsOrder==ComputeInfo.ChannelsOrder.NHWC));

        entries.Add(new Entry("InstanceNormTail_Flat",
            Int3(O.length)));

        entries.Add(new Entry("InstanceNormTail_Loop",
            Int3(O.length), BigO(2), 256));

        return entries;
    }

    private static List<Entry> s_CopyEntries = new List<Entry>(1);
    internal static List<Entry> Copy(TensorShape X, TensorShape O)
    {
        var entries = s_CopyEntries;
        entries.Clear();

        entries.Add( // NOTE: dispatched over X (not O)
            new Entry("Copy",
                Int3(X.channels, X.width, X.height), BigO(O.batch)));

        return entries;
    }

    private static List<Entry> s_TransposeToChannelFirst = new List<Entry>(1);
    internal static List<Entry> TransposeToChannelFirst(TensorShape X, TensorShape O)
    {
        var entries = s_TransposeToChannelFirst;
        entries.Clear();

        entries.Add( // NOTE: dispatched over X (not O)
            new Entry("TransposeToChannelFirst",
                Int3(X.channels, X.width, X.height), BigO(O.batch)));

        return entries;
    }

    private static List<Entry> s_ReshapeFromNHWCModelEntries = new List<Entry>(2);
    internal static List<Entry> ReshapeFromNHWCModel(TensorShape O)
    {
        var entries = s_ReshapeFromNHWCModelEntries;
        entries.Clear();

        entries.Add(
            new Entry("ReshapeFromNHWCModel_Flat",
                Int3(O.channels, O.width, O.height)));

        entries.Add(
            new Entry("ReshapeFromNHWCModel_Loop",
            Int3(O.length), BigO(2), 256));

        return entries;
    }

    private static List<Entry> s_PaddingEntries = new List<Entry>(1);
    internal static List<Entry> Padding(TensorShape X, TensorShape O, string kernelName)
    {
        var entries = s_PaddingEntries;
        entries.Clear();

        entries.Add(new Entry(kernelName,
            Int3(O.channels, O.width, O.height), BigO(O.batch)));

        return entries;
    }

    private static List<Entry> s_BroadcastEntries = new List<Entry>(1);
    internal static List<Entry> Broadcast(TensorShape X, TensorShape O, string kernelName)
    {
        var entries = s_BroadcastEntries;
        entries.Clear();

        if (ComputeInfo.channelsOrder == ComputeInfo.ChannelsOrder.NHWC)
            entries.Add(new Entry(kernelName, Int3(O.channels, O.width, O.height), BigO(O.batch)));
        else
            entries.Add(new Entry(kernelName, Int3(O.width, O.height, O.channels), BigO(O.batch)));
        return entries;
    }

    static ValueTuple<int,int,int> Int3(int x, int y = 1, int z = 1) { return ValueTuple.Create(x, y, z); }
    static float BigO(int o) { return (float)o; }
    internal struct StrictDimensions { public bool valid; }
    static StrictDimensions StrictAnd(bool valid_) { return new StrictDimensions { valid = valid_ }; }
    static StrictDimensions Strict() { return new StrictDimensions { valid = true }; }

    internal struct Entry
    {
        public readonly string name;
        public readonly ValueTuple<int,int,int> dispatch;
        public readonly float bigO;
        public readonly bool valid;
        public readonly bool strict;
        public readonly uint loopStride; // > 0 indicates looping kernel
        public readonly bool devicePriority;

        public Entry(string name_, ValueTuple<int,int,int> dispatch_, float bigO_ = 1.0f, bool valid_ = true, bool devicePriority_ = false)
        {
            name = name_;
            dispatch = dispatch_;
            bigO = bigO_;
            valid = valid_;
            strict = false;
            loopStride = 0;
            devicePriority = devicePriority_;
        }

        public Entry(string name_, ValueTuple<int,int,int> dispatch_, float bigO_, uint loopStride_) :
            this(name_, dispatch_, bigO_)
        {
            loopStride = loopStride_;
        }

        public Entry(string name_, ValueTuple<int,int,int> dispatch_, float bigO_, StrictDimensions strictDims) :
            this(name_, dispatch_, bigO_, strictDims.valid)
        {
            strict = true;
        }

        public Entry(string name_, ValueTuple<int,int,int> dispatch_, float bigO_, StrictDimensions strictDims, bool devicePriority_) :
            this(name_, dispatch_, bigO_, strictDims.valid, devicePriority_)
        {
            strict = true;
        }
    }
}

internal struct ComputeKernel
{
    readonly public ComputeFunc func;
    readonly public ValueTuple<int,int,int> dispatch;
    public ComputeShader shader { get { return func.shader; } }

    public ComputeKernel(ComputeFunc func_, ValueTuple<int,int,int> dispatch_)
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
    internal static long CalculateEntryScore(ComputeShaderContext ctx, ComputeKernelLibrary.Entry entry, bool verbose)
    {
        long work = InvalidEntry;
        try
        {
            if (!entry.valid)
                return InvalidEntry;

            // @TODO: @OPTIMIZE: cache threadGroupSize instead of creating ComputeFunc and querying every time
            var fn = new ComputeFunc(ctx, entry.name);

            if (fn.threadGroupSizeX * fn.threadGroupSizeY * fn.threadGroupSizeZ > ComputeInfo.maxComputeWorkGroupSize)
                return InvalidEntry;

            if (entry.strict)
            {
                if (entry.dispatch.Item1 % fn.threadGroupSizeX != 0 ||
                    entry.dispatch.Item2 % fn.threadGroupSizeY != 0 ||
                    entry.dispatch.Item3 % fn.threadGroupSizeZ != 0)
                    return InvalidEntry;
            }

            var x = (long) ComputeFunc.IntDivCeil(entry.dispatch.Item1, (int) fn.threadGroupSizeX);
            var y = (long) ComputeFunc.IntDivCeil(entry.dispatch.Item2, (int) fn.threadGroupSizeY);
            var z = (long) ComputeFunc.IntDivCeil(entry.dispatch.Item3, (int) fn.threadGroupSizeZ);

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

    internal static ComputeKernel BestKernel(ComputeShaderContext ctx, List<ComputeKernelLibrary.Entry> entrees, bool verbose)
    {
        var bestEntry = entrees[0];
        var bestScore = InvalidEntry;
        bool foundKernelWithDevicePriority = false;
        for (int i = 0; i < entrees.Count; i++)
        {
            var score = CalculateEntryScore(ctx, entrees[i], verbose);
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

        var func = new ComputeFunc(ctx, bestEntry.name);

        if (bestEntry.loopStride > 0)
        {
            int preferedDispatch = (int)bestEntry.loopStride * (int)func.threadGroupSizeX;
            var kernel = new ComputeKernel(func, (preferedDispatch, 1, 1));
            kernel.shader.SetInt("_LoopStride", preferedDispatch);
            return kernel;
        }
        else
        {
            return new ComputeKernel(func, bestEntry.dispatch);
        }
    }

}

/// <summary>
/// GPU compute implementation of `IOps`
/// </summary>
public class ComputeOps : ReferenceComputeOps
{
    // ---------------------------------------------------------------------------------
    private bool printKernels = false;

    // ---------------------------------------------------------------------------------
    private bool m_Verbose = false;

    /// <summary>
    /// Create `ComputeOps`
    /// </summary>
    /// <param name="allocator">allocator</param>
    /// <param name="verbose">verbose flag</param>
    public ComputeOps(ITensorAllocator allocator = null, bool verbose = false)
    : base(allocator)
    {
        m_Verbose = verbose;
    }

    // ---------------------------------------------------------------------------------

    internal ComputeKernel BestKernel(List<ComputeKernelLibrary.Entry> entrees)
    {
        return ComputeKernel.BestKernel(ComputeShaderContext.Optimized, entrees, m_Verbose);
    }

    internal ComputeKernel CompileKernel(ComputeKernelLibrary.Entry entry)
    {
        var func = new ComputeFunc(ComputeShaderContext.Optimized, entry.name);
        if (entry.loopStride > 0)
        {
            int preferedDispatch = (int)entry.loopStride * (int)func.threadGroupSizeX;
            var kernel = new ComputeKernel(func, (preferedDispatch, 1, 1));
            kernel.shader.SetInt("_LoopStride", preferedDispatch);
            return kernel;
        }
        else
        {
            return new ComputeKernel(func, entry.dispatch);
        }
    }

    // ---------------------------------------------------------------------------------

    /// <inheritdoc/>
    public override Tensor MatMul(Tensor X, bool xTranspose, Tensor Y, bool yTranspose)
    {
        // MatMul implementation in terms of Dense
        var A = (xTranspose) ? Transpose(X): X;
        var B = (yTranspose) ? Transpose(Y): Y;
        var Cshape = new TensorShape(1, B.flatWidth); // intialize bias with zeros

        ComputeBuffer buffer = new ComputeBuffer(B.shape.length + Cshape.length, sizeof(float));

        var Bpacked = new Tensor(B.shape, new SharedComputeTensorData(buffer, B.shape, 0));
        var Cpacked = new Tensor(Cshape, new SharedComputeTensorData(buffer, Cshape, B.shape.length));

        var fn_pack = new ComputeKernel(new ComputeFunc(ComputeShaderContext.Optimized, "MatMulPackB0Bias"), (B.flatWidth, B.flatHeight, 1));
        fn_pack.SetTensor("X", B.shape, Pin(B).buffer);
        fn_pack.SetTensor("O", Bpacked.shape, Pin(Bpacked).buffer);

        fn_pack.Dispatch();

        var O = Dense(A, Bpacked, Cpacked, Layer.FusedActivation.None);
        if (A != X) A.Dispose();
        if (B != Y) B.Dispose();

        buffer.Dispose();

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
        fn.shader.SetInt("_ActivationMode", (int)fusedActivation);

        fn.Dispatch();

        if (!IsFusedActivationSupported(fusedActivation))
            O = Activation(fusedActivation.ToString(), O);

        return O;
    }

    Tensor Conv2DWinograd(Tensor X, Tensor K, Tensor B, Tensor O, int[] stride, int[] pad, Layer.FusedActivation fusedActivation, ComputeKernel fn)
    {
        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(X.channels, K.kernelDepth);
        Assert.AreEqual(K.kernelCount, B.flatWidth);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);

        // Winograd
        // transform kernel
        TensorShape Kws = new TensorShape(K.kernelHeight + 1, K.kernelWidth + 1, K.kernelDepth, K.kernelCount);

        ComputeBuffer buffer = new ComputeBuffer(Kws.length + B.shape.length, sizeof(float));
        var Ktransformed = new Tensor(Kws, new SharedComputeTensorData(buffer, Kws, 0));
        var Bpacked = new Tensor(B.shape, new SharedComputeTensorData(buffer, B.shape, Kws.length));

        var fn_wk = new ComputeKernel(new ComputeFunc(ComputeShaderContext.Optimized, "KernelWinograd_3x3"), (K.kernelCount, X.channels, B.length));

        fn_wk.SetTensorDecl("K", K.shape, Pin(K).offset);
        fn_wk.SetTensorDecl("B", B.shape, Pin(B).offset);
        Assert.AreEqual(Pin(K).buffer, Pin(B).buffer);
        fn_wk.SetTensorBuffer("WBK", Pin(K).buffer);
        fn_wk.SetTensor("O", Ktransformed.shape, Pin(Ktransformed).buffer);
        fn_wk.Dispatch();

        fn.SetTensor("X", X.shape, Pin(X).buffer);
        fn.SetTensor("O", O.shape, Pin(O).buffer);
        fn.SetTensorDecl("K", Ktransformed.shape, Pin(Ktransformed).offset);
        fn.SetTensorDecl("B", Bpacked.shape, Pin(Bpacked).offset);
        Assert.AreEqual(Pin(Ktransformed).buffer, Pin(Bpacked).buffer);
        fn.SetTensorBuffer("WBK", Pin(Ktransformed).buffer);
        fn.shader.SetInts("_Pad", pad);
        fn.shader.SetInt("_ActivationMode", (int)fusedActivation);
        fn.Dispatch();

        if (!IsFusedActivationSupported(fusedActivation))
            O = Activation(fusedActivation.ToString(), O);

        buffer.Dispose();
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

        var O = NewTensor(X.shape.ApplyKernel(K.shape, stride, pad));
        var fn = BestKernel(ComputeKernelLibrary.Conv3D(X.shape, K.shape, O.shape, stride, pad));

        if (printKernels)
            Debug.Log($"{fn.func.kernelName}: {O.shape} = {X.shape} # {K.shape} stride: {stride[0]},{stride[1]},,{stride[2]} pad:{pad[0]},{pad[1]}, ,{stride[2]}" );

        fn.SetTensor("X", X.shape, Pin(X).buffer);
        fn.SetTensor("O", O.shape, Pin(O).buffer);
        fn.SetTensorDecl("K", K.shape, Pin(K).offset);
        fn.SetTensorDecl("B", B.shape, Pin(B).offset);
        Assert.AreEqual(Pin(K).buffer, Pin(B).buffer);
        fn.SetTensorBuffer("WBK", Pin(K).buffer);

        fn.shader.SetInts("_Pad", pad);
        fn.shader.SetInts("_Stride", stride);
        fn.shader.SetInt("_ActivationMode", (int)fusedActivation);

        fn.Dispatch();

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
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);

        var O = NewTensor(X.shape.ApplyKernel(K.shape, stride, pad));
        var fn = BestKernel(ComputeKernelLibrary.Conv2D(X.shape, K.shape, O.shape, stride, pad));

        if (printKernels)
            Debug.Log($"{fn.func.kernelName}: {O.shape} = {X.shape} # {K.shape} stride: {stride[0]},{stride[1]} pad:{pad[0]},{pad[1]}" );

        if (fn.func.kernelName.StartsWith("Conv2DWinograd"))
        {
            return Conv2DWinograd(X, K, B, O, stride, pad, fusedActivation, fn);
        }

        fn.SetTensor("X", X.shape, Pin(X).buffer);
        fn.SetTensor("O", O.shape, Pin(O).buffer);
        fn.SetTensorDecl("K", K.shape, Pin(K).offset);
        fn.SetTensorDecl("B", B.shape, Pin(B).offset);
        Assert.AreEqual(Pin(K).buffer, Pin(B).buffer);
        fn.SetTensorBuffer("WBK", Pin(K).buffer);

        fn.shader.SetInts("_Pad", pad);
        fn.shader.SetInts("_Stride", stride);
        fn.shader.SetInt("_ActivationMode", (int)fusedActivation);

        fn.Dispatch();

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
        fn.shader.SetInt("_ActivationMode", (int)fusedActivation);

        fn.Dispatch();

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

        // conv2d trans as conv2d
        pad = new int[]
        {
            K.kernelWidth - pad[0] - 1, K.kernelHeight - pad[1] - 1,
            K.kernelWidth - pad[2] - 1, K.kernelHeight - pad[3] - 1
        };

        // Unwrap ConvTrans as a call to Conv2D:
        // https://arxiv.org/abs/1603.07285
        // Two pass algorithm:
        // O-pad X, flip kernel and call Conv2D

        // 0-pad X accordingly:
        // stride number of 0 between values of X
        // outputAdjustment number of 0 at the end of X
        // regular padding will be done in Conv2D
        var XpaddedShape = new TensorShape(X.batch, stride[1] * (X.height - 1) + 1 + outputAdjustment[1], stride[0] * (X.width - 1) + 1 + outputAdjustment[0], X.channels);
        var fn = new ComputeFunc(ComputeShaderContext.Optimized, "Conv2DTransPadFill");
        fn.shader.SetInts("_Stride", stride);
        fn.shader.SetInts("_Pad", outputAdjustment);
        fn.SetTensor("X", X.shape, Pin(X).buffer);
        var Xpadded = Dispatch(fn, XpaddedShape, X.channels, X.width, X.height);

        // Flip kernel
        // handle WBK case (K and B data share the same CB), copy B at the same time as flipping K
        ComputeBuffer buffer = new ComputeBuffer(K.shape.length + B.shape.length, sizeof(float));

        var Kflipped = new Tensor(K.shape, new SharedComputeTensorData(buffer, K.shape, 0));
        var Bpacked = new Tensor(B.shape, new SharedComputeTensorData(buffer, B.shape, K.shape.length));

        var fn_flip = new ComputeKernel(new ComputeFunc(ComputeShaderContext.Optimized, "Conv2DTransFlipKernel"), (K.kernelCount, X.channels, (K.kernelWidth*K.kernelHeight)));
        fn_flip.SetTensorDecl("K", K.shape, Pin(K).offset);
        fn_flip.SetTensorDecl("B", B.shape, Pin(B).offset);
        Assert.AreEqual(Pin(K).buffer, Pin(B).buffer);
        fn_flip.SetTensorBuffer("WBK", Pin(K).buffer);
        fn_flip.SetTensor("O", Kflipped.shape, Pin(Kflipped).buffer);
        fn_flip.shader.SetInts("_Stride", stride);
        fn_flip.shader.SetInts("_Pad", outputAdjustment);

        fn_flip.Dispatch();

        var O = Conv2D(Xpadded, Kflipped, Bpacked, new int[] { 1, 1 }, pad, fusedActivation);
        buffer.Dispose();
        return O;
    }

    /// <inheritdoc/>
    public override Tensor Upsample2D(Tensor X, int[] scale, bool bilinear)
    {
        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(scale.Length, 2);

        var O = NewTensor(X.batch, X.height*scale[1], X.width*scale[0], X.channels);
        var fn = BestKernel(ComputeKernelLibrary.Upsample2D(X.shape, O.shape, scale, bilinear));

        if (printKernels)
            D.Log($"{fn.func.kernelName}: {O.shape} = {X.shape} ^ size: {scale[0]},{scale[1]}" );

        fn.SetTensor("X", X.shape, Pin(X).buffer);
        fn.SetTensor("O", O.shape, Pin(O).buffer);

        fn.shader.SetInts("_Pool", scale);


        fn.Dispatch();
        return O;
    }

    /// <inheritdoc/>
    protected override Tensor Pool2D(string kernelName, Tensor X, int[] pool, int[] stride, int[] pad)
    {
        Assert.AreEqual(pool.Length, 2);
        Assert.AreEqual(stride.Length, 2);

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

    /// <inheritdoc/>
    public override Tensor GlobalMaxPool2D(Tensor X)
    {
        return GlobalPool2D("MaxPool2DReduce", "GlobalMaxPool2D", X);
    }

    /// <inheritdoc/>
    public override Tensor GlobalAvgPool2D(Tensor X)
    {
        return GlobalPool2D("AvgPool2DReduce", "GlobalAvgPool2D", X);
    }

    Tuple<Tensor, Tensor> GlobalAvgVariancePool2DReduce(Tensor X, Tensor X2, bool isFirstDispatch)
    {
        var pool = new[] { 8, 8 };
        var stride = pool;
        var pad = new[] { 0, 0, 0, 0 };
        string kernelName = "AvgVariancePool2DReduce";

        var Oshape = X.shape.ApplyPool(pool, stride, pad, ceilMode: true);
        var O = NewTensor(new TensorShape(Oshape.batch, ComputeHelper.IDivC(Oshape.height, 2), ComputeHelper.IDivC(Oshape.width, 2), Oshape.channels));
        var O2 = NewTensor(O.shape);

        var fn = BestKernel(ComputeKernelLibrary.PoolAvgVar2D(X.shape, O.shape, kernelName));

        if (printKernels)
            D.Log($"{fn.func.kernelName}: {O.shape} = {X.shape} ^ pool: {pool[0]},{pool[1]} stride: {stride[0]},{stride[1]} pad:{pad[0]},{pad[1]}" );

        fn.SetTensor("X", X.shape, Pin(X).buffer);
        fn.SetTensor("X2", X2.shape, Pin(X2).buffer);
        fn.SetTensor("O", O.shape, Pin(O).buffer);
        fn.SetTensor("O2", O2.shape, Pin(O2).buffer);

        fn.shader.SetInts("_Pool", pool);
        fn.shader.SetInts("_Stride", stride);
        fn.shader.SetInts("_Pad", pad);
        fn.shader.SetInt("_IsFirstDispatch", isFirstDispatch ? 1 : 0);

        fn.Dispatch();
        return new Tuple<Tensor,Tensor>(O,O2);
    }

    /// <inheritdoc/>
    public override Tensor GlobalAvgVariancePool2D(Tensor X)
    {
        Assert.IsTrue(X.shape.Is4D());
        var inputDim = new [] {X.height, X.width};
        var X2 = X; // save a X^2 and do it in the first dispatch
        bool isFirstDispatch = true;
        // downsample with pyramid approach
        while (X.height * X.width >= 8*8*2)
        {
            var lastLength = X.length;
            var XX2 = GlobalAvgVariancePool2DReduce(X, X2, isFirstDispatch);
            X = XX2.Item1;
            X2 = XX2.Item2;
            Assert.IsTrue(X.length < lastLength);
            isFirstDispatch = false;
        }

        var O = NewTensor(X.batch, 2, 1, X.channels);
        var fn = BestKernel(ComputeKernelLibrary.GlobalPool2D(X.shape, O.shape, "GlobalAvgVariancePool2D"));

        fn.SetTensor("X", X.shape, Pin(X).buffer);
        fn.SetTensor("X2", X2.shape, Pin(X2).buffer);
        fn.SetTensor("O", O.shape, Pin(O).buffer);
        fn.shader.SetInts("_Pool", inputDim);
        fn.shader.SetInt("_IsFirstDispatch", isFirstDispatch ? 1 : 0);

        fn.Dispatch();
        return O;
    }

    Tensor GlobalPool2DReduce(string kernelName, Tensor X)
    {
        var pool = new[] { 8, 8 };
        var stride = pool;
        var pad = new[] { 0, 0, 0, 0 };

        var Oshape = X.shape.ApplyPool(pool, stride, pad, ceilMode: true);
        var O = NewTensor(new TensorShape(Oshape.batch, ComputeHelper.IDivC(Oshape.height, 2), ComputeHelper.IDivC(Oshape.width, 2), Oshape.channels));
        var fn = BestKernel(ComputeKernelLibrary.Pool2DReduce(X.shape, O.shape, kernelName));

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

    internal static int[] s_GlobalPool2DInputDim = new int[2];

    /// <summary>
    /// Generic global 2D pooling
    /// </summary>
    /// <param name="smallKernelName">small kernel name</param>
    /// <param name="globalKernelName">global kernel name</param>
    /// <param name="X">input</param>
    /// <returns>output `Tensor`</returns>
    protected virtual Tensor GlobalPool2D(string smallKernelName, string globalKernelName, Tensor X)
    {
        Assert.IsTrue(X.shape.Is4D());
        s_GlobalPool2DInputDim[0] = X.height;
        s_GlobalPool2DInputDim[1] = X.width;

        // downsample with pyramid approach
        while (X.height * X.width >= 8*8*2)
        {
            var lastLength = X.length;
            X = GlobalPool2DReduce(smallKernelName, X);
            Assert.IsTrue(X.length < lastLength);
        }

        var O = NewTensor(X.batch, 1, 1, X.channels);
        var fn = BestKernel(ComputeKernelLibrary.GlobalPool2D(X.shape, O.shape, globalKernelName));

        fn.SetTensor("X", X.shape, Pin(X).buffer);
        fn.SetTensor("O", O.shape, Pin(O).buffer);
        fn.shader.SetInts("_Pool", s_GlobalPool2DInputDim);

        fn.Dispatch();
        return O;
    }

    /// <inheritdoc/>
    public override Tensor ScaleBias(Tensor X, Tensor S, Tensor B)
    {
        if (!X.shape.Is4D())
            return base.ScaleBias(X, S, B);

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

    /// <inheritdoc/>
    public override Tensor Normalization(Tensor X, Tensor S, Tensor B, int pool, int axis, float epsilon, Layer.FusedActivation fusedActivation)
    {
        if (!X.shape.Is4D())
            throw new NotImplementedException();

        if (axis != TensorShape.C && axis != -1)
            return base.Normalization(X, S, B, pool, axis, epsilon, fusedActivation);

        if (pool <= 0)
            pool = X.batch;

        if (pool > 1)
            throw new NotImplementedException(); // @TODO: support other types of Normalization at test time
                                                 // Currently supported only pool=1 (InstanceNormalization)
        var meanVariance = GlobalAvgVariancePool2D(X);

        Assert.AreEqual(X.channels, B.channels); Assert.AreEqual(X.channels, S.channels);
        Assert.AreEqual(B.length, B.channels); Assert.AreEqual(S.length, S.channels);

        var O = NewTensor(X.shape);
        var fn = BestKernel(ComputeKernelLibrary.NormalizationTail(X.shape, O.shape));
        fn.SetTensor("X", X.shape, Pin(X).buffer);
        fn.SetTensor("O", O.shape, Pin(O).buffer);
        fn.SetTensor("W", meanVariance.shape, Pin(meanVariance).buffer);


        fn.SetTensorDecl("S", S.shape, Pin(S).offset);
        fn.SetTensorDecl("B", B.shape, Pin(B).offset);
        Assert.AreEqual(Pin(S).buffer, Pin(B).buffer);
        fn.SetTensorBuffer("WBK", Pin(S).buffer);
        fn.shader.SetFloat("_Epsilon", epsilon);
        fn.shader.SetInt("_ActivationMode", (int)fusedActivation);

        fn.Dispatch();

        if (!IsFusedActivationSupported(fusedActivation))
            O = Activation(fusedActivation.ToString(), O);

        return O;
    }

    /// <inheritdoc/>
    protected override Tensor Reduce(string kernelName, Tensor X, int axis)
    {
        axis = X.shape.Axis(axis);

        //TODO optimize when reducing not on channel.
        bool needTranpose = axis != TensorShape.C;
        FillReducePermute(axis);

        if (needTranpose)
            X = Transpose(X, s_ReducePermute);

        var oShape = X.shape.Reduce(TensorShape.C);
        Assert.AreEqual(oShape.channels, 1);

        var O = NewTensor(oShape);

        var fn = new ComputeKernel(new ComputeFunc(ComputeShaderContext.Optimized, kernelName),
                                    (oShape.width, oShape.height, 1));

        if (printKernels)
            D.Log(fn.func.kernelName);

        fn.SetTensor("X", X.shape, Pin(X).buffer);
        fn.SetTensor("O", O.shape, Pin(O).buffer);

        fn.Dispatch();

        if (needTranpose)
            O = Transpose(O, s_ReducePermute);

        return O;
    }

    /// <inheritdoc/>
    protected override Tensor Activation(string kernelName, Tensor X, float alpha = 0f, float beta = 0f)
    {
        if (!X.shape.Is4D())
            return base.Activation(kernelName, X, alpha, beta);

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

    /// <inheritdoc/>
    public override Tensor PRelu(Tensor X, Tensor S)
    {
        if (!X.shape.Is4D() || !S.shape.Is4D())
            return base.PRelu(X, S);

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

    /// <inheritdoc/>
    public override Tensor Softmax(Tensor X, int axis)
    {
        if (X.shape.sequenceLength != 1 || X.shape.numberOfDirections != 1 || axis > X.shape.FirstNotIdentityFeatureDimensionIndex())
            return base.Softmax(X, axis);

        var O = NewTensor(X.shape);
        var fn = BestKernel(ComputeKernelLibrary.Softmax(X.shape, O.shape));

        fn.SetTensor("X", X.shape, Pin(X).buffer);
        fn.SetTensor("O", O.shape, Pin(O).buffer);

        fn.Dispatch();
        return O;
    }

    /// <inheritdoc/>
    public override Tensor LogSoftmax(Tensor X)
    {
        if (X.shape.sequenceLength != 1 || X.shape.numberOfDirections != 1)
            return base.LogSoftmax(X);

        var O = NewTensor(X.shape);
        var fn = BestKernel(ComputeKernelLibrary.LogSoftmax(X.shape, O.shape));

        fn.SetTensor("X", X.shape, Pin(X).buffer);
        fn.SetTensor("O", O.shape, Pin(O).buffer);

        fn.Dispatch();
        return O;
    }

    // @TODO: implement Dropout in terms of RandomUniform by preparing random values on CPU upfront and multiplying result on GPU later on
    // public override Tensor Dropout(Tensor X, float alpha)

    protected override Tensor TransposeToChannelFirst(Tensor X)
    {
        var O = NewTensor(X.shape);
        var fn = BestKernel(ComputeKernelLibrary.TransposeToChannelFirst(X.shape, O.shape));

        fn.SetTensor("X", X.shape, Pin(X).buffer);
        fn.SetTensor("O", O.shape, Pin(O).buffer);

        fn.Dispatch();
        return O;
    }

    /// <inheritdoc/>
    public override Tensor Concat(Tensor[] tensors, int axis)
    {
        if (!TensorExtensions.AreAllTensorsConvertibleTo4D(tensors) || !TensorExtensions.Is8DAxisConvertibleTo4D(axis))
            return base.Concat(tensors, axis);

        var O = NewTensor(TensorExtensions.Concat(tensors, axis));

        var offsets = s_ConcatOffsets;
        Array.Clear(offsets, 0, offsets.Length);
        axis = O.shape.Axis(axis);
        var axisNHWC = TensorExtensions.Convert8DAxisTo4D(axis);

        foreach (var inputTensor in tensors)
        {
            // input can be constants, in that cases the internal layout does not match ComputeInfo.channelsOrder and will allways be NHWC
            // => permute if there is a layout mismatch
            var X = GetTensorInCurrentMemoryLayout(inputTensor);

            var fn = BestKernel(ComputeKernelLibrary.Copy(X.shape, O.shape));

            fn.SetTensor("X", X.shape, Pin(X).buffer);
            fn.SetTensor("O", O.shape, Pin(O).buffer);

            fn.shader.SetInts("_Pad", offsets);

            fn.Dispatch();

            offsets[axisNHWC] += X.shape[axis];
        }

        return O;
    }

    // Requires `output` to be allocated by the calling code to avoid unnecessary GC allocations
    internal int[] GetInputTensorStridesOnDevice(TensorShape shape, ComputeInfo.ChannelsOrder channelOrder, int[] output)
    {
        Assert.IsNotNull(output);
        Assert.AreEqual(4, output.Length);

        output[0] = (shape.batch == 1) ? 0 : shape.height * shape.width * shape.channels;

        if (channelOrder == ComputeInfo.ChannelsOrder.NHWC)
        {
            output[1] = (shape.height == 1) ? 0 : shape.width * shape.channels;
            output[2] = (shape.width == 1) ? 0 : shape.channels;
            output[3] = (shape.channels == 1) ? 0 : 1;
        }
        else
        {
            output[1] = (shape.height == 1) ? 0 : shape.width;
            output[2] = (shape.width == 1) ? 0 : 1;
            output[3] = (shape.channels == 1) ? 0 : shape.height * shape.width;
        }

        return output;
    }

    internal static int[] s_XStrides = new int[4];
    internal static int[] s_BStrides = new int[4];
    /// <inheritdoc/>
    protected override Tensor ElementwiseWithBroadcast(string kernelName, Tensor[] tensors)
    {
        Assert.IsTrue(tensors.Length > 0);

        if (!TensorExtensions.AreAllTensorsConvertibleTo4D(tensors))
            return base.ElementwiseWithBroadcast(kernelName, tensors);

        Tensor outputTensor1 = NewTensor(TensorExtensions.MaxShape(tensors));
        Tensor outputTensor2 = null;
        if (tensors.Length > 2)
            outputTensor2 = NewTensor(TensorExtensions.MaxShape(tensors));

        var X = tensors[0];
        var fn = BestKernel(ComputeKernelLibrary.Broadcast(X.shape, outputTensor1.shape, kernelName));

        Tensor O = null;
        bool isFirstDispatch = true;
        for (int t = 1; t < tensors.Length; ++t)
        {
            var B = tensors[t];
            O = (t % 2 == 1) ? outputTensor1 : outputTensor2;
            fn.SetTensor("X", X.shape, Pin(X).buffer);
            fn.SetTensor("O", O.shape, Pin(O).buffer);
            fn.SetTensor("B", B.shape, Pin(B).buffer, Pin(B).offset);
            fn.shader.SetFloat("_Alpha", 1.0f / (float)tensors.Length);
            fn.shader.SetInt("_IsFirstDispatch", isFirstDispatch ? 1 : 0);

            fn.shader.SetInts("_XStrides", GetInputTensorStridesOnDevice(X.shape, Pin(X).channelsOrder, s_XStrides));
            fn.shader.SetInts("_BStrides", GetInputTensorStridesOnDevice(B.shape, Pin(B).channelsOrder, s_BStrides));

            fn.Dispatch();

            X = O;
            isFirstDispatch = false;
        }

        if (O != outputTensor1) outputTensor1.Dispose();
        if (O != outputTensor2) outputTensor2?.Dispose();
        return O;
    }


    internal static int[] s_ApplyPaddingCroppedSize = new int[2];
    /// <inheritdoc/>
    protected override Tensor ApplyPadding(Tensor X, int[] pad, string kernelName, float constant = 0.0f)
    {
        Assert.IsTrue(X.shape.Is4D());
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

            s_ApplyPaddingCroppedSize[0] = croppedWidth;
            s_ApplyPaddingCroppedSize[1] = croppedHeight;

            fn.shader.SetInts("_Pool", s_ApplyPaddingCroppedSize);
            fn.shader.SetFloat("_Beta", constant);
        }

        fn.Dispatch();
        return O;
    }

    /// <inheritdoc/>
    public override Tensor LogicalNot(Tensor X)
    {
        var O = NewTensor(X.shape);
        var fn = BestKernel(ComputeKernelLibrary.Activation(X.shape, O.shape, "LogicalNot"));

        fn.SetTensor("X", X.shape, Pin(X).buffer);
        fn.SetTensor("O", O.shape, Pin(O).buffer);

        fn.Dispatch();
        return O;
    }

    internal static int[] s_SStrides = new int[4];
    /// <inheritdoc/>
    public override Tensor Where(Tensor C, Tensor A, Tensor B)
    {
        if (!C.shape.Is4D() || !A.shape.Is4D() || !B.shape.Is4D())
            return base.Where(C, A, B);

        Tensor O = NewTensor(C.shape);
        var fn = BestKernel(ComputeKernelLibrary.Broadcast(C.shape, O.shape, "BroadcastWhere"));

        fn.SetTensor("X", C.shape, Pin(C).buffer);
        fn.SetTensor("O", O.shape, Pin(O).buffer);
        fn.SetTensor("S", A.shape, Pin(A).buffer, Pin(A).offset);
        fn.SetTensor("B", B.shape, Pin(B).buffer, Pin(B).offset);

        fn.shader.SetInts("_XStrides", GetInputTensorStridesOnDevice(C.shape, Pin(C).channelsOrder, s_XStrides));
        fn.shader.SetInts("_SStrides", GetInputTensorStridesOnDevice(A.shape, Pin(A).channelsOrder, s_SStrides));
        fn.shader.SetInts("_BStrides", GetInputTensorStridesOnDevice(B.shape, Pin(B).channelsOrder, s_BStrides));

        fn.Dispatch();
        return O;
    }

    /// <inheritdoc/>
    protected override Tensor CopyAndReshape_NCHW(Tensor X, TensorShape newShape)
    {
        //8D reshape only supported on reference backend. No optimized 8D version as
        //the goal is rather to have a `channelFirst` model were reshape is a noop.
        if (!X.shape.Is4D() || !newShape.Is4D())
            return base.CopyAndReshape_NCHW(X, newShape);

        Assert.AreEqual(X.length, newShape.length);
        Assert.AreEqual(ComputeInfo.ChannelsOrder.NCHW, ComputeInfo.channelsOrder);

        var O = NewTensor(newShape, "O");
        var fn = BestKernel(ComputeKernelLibrary.ReshapeFromNHWCModel(O.shape));

        fn.SetTensor("X", X.shape, Pin(X).buffer);
        fn.SetTensor("O", O.shape, Pin(O).buffer);

        fn.Dispatch();
        return O;
    }

    /// <inheritdoc/>
    protected override Tensor CopyAndReshape(Tensor X, TensorShape newShape)
    {
        //8D reshape only supported on reference backend atm.
        if (!X.shape.Is4D() || !newShape.Is4D())
            return base.CopyAndReshape(X, newShape);

        var copyShape = X.shape;
        Assert.AreEqual(copyShape.length, newShape.length);
        if (X.shape != newShape)
        {
            //In CHW mode one should call CopyAndReshape_NCHW if shape is modified
            Assert.AreEqual(ComputeInfo.ChannelsOrder.NHWC, ComputeInfo.channelsOrder);
        }

        // NOTE: "Copy" kernel copies tensor data while preserving the shape
        // However here in CopyAndReshape we want to both copy and change the shape,
        // To be able to piggyback "Copy" kernel we specify new shape when allocating destination tensor,
        // but use shape identical to source when copying.

        var O = NewTensor(newShape);
        var fn = BestKernel(ComputeKernelLibrary.Copy(copyShape, copyShape));

        fn.SetTensor("X", copyShape, Pin(X).buffer);
        fn.SetTensor("O", copyShape, Pin(O).buffer);

        fn.shader.SetInts("_Pad", new int[] { 0,0,0,0 });

        fn.Dispatch();
        return O;
    }
}

internal class ComputeVarsWithSharedModel : DefaultVars
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

} // namespace Unity.Barracuda
