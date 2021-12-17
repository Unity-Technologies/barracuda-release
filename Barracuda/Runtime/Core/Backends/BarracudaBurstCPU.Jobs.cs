using UnityEngine;
using System;
using System.Collections.Generic;
using System.Threading;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Burst;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;
using Unity.Mathematics;

[assembly: BurstCompile(OptimizeFor = OptimizeFor.FastCompilation)]
namespace Unity.Barracuda {

// BarracudaBurstCPU.Core.cs -- definition of class BurstCPUOps, Pin(), BurstTensorData
// BarracudaBurstCPU.Ops.cs  -- impl. IOps, job schedulers
// BarracudaBurstCPU.Jobs.cs -- impl. jobs

public partial class BurstCPUOps
{
    internal static readonly Thread MainThread = Thread.CurrentThread;

    #region Job resources declaration

    internal unsafe struct ReadOnlyMemResource
    {
        [NoAlias][NativeDisableUnsafePtrRestriction][ReadOnly] public void* ptr;
        public float* ptrfloat { get { return (float*)ptr; } }
        public half* ptrhalf { get { return (half*)ptr; } }
    }

    internal unsafe struct ReadWriteMemResource
    {
        [NoAlias][NativeDisableUnsafePtrRestriction] public void* ptr;
        public float* ptrfloat { get { return (float*)ptr; } }
        public half* ptrhalf { get { return (half*)ptr; } }
    }

    internal interface IJobResourceDeclarationO
    {
        ReadWriteMemResource O { get; set; }
    }

    internal interface IJobResourceDeclarationXO
    {
        ReadOnlyMemResource X { get; set; }
        ReadWriteMemResource O { get; set; }
    }

    internal interface IJobResourceDeclarationXBO
    {
        ReadOnlyMemResource X { get; set; }
        ReadOnlyMemResource B { get; set; }
        ReadWriteMemResource O { get; set; }
    }

    internal interface IJobResourceDeclarationXSBO
    {
        ReadOnlyMemResource X { get; set; }
        ReadOnlyMemResource S { get; set; }
        ReadOnlyMemResource B { get; set; }
        ReadWriteMemResource O { get; set; }
    }

    #endregion

    #region Job inner data declaration

    internal partial struct HardSigmoidJobHelper
    {
        [ReadOnly] public float alpha, beta;
    }

    internal partial struct ClipJobHelper
    {
        [ReadOnly] public float min, max;
    }

    internal partial struct PowJobHelper
    {
        [ReadOnly] public float alpha;
    }

    internal partial struct EluJobHelper
    {
        [ReadOnly] public float alpha;
    }

    internal partial struct SeluJobHelper
    {
        [ReadOnly] public float alpha, gamma;
    }

    internal partial struct PReluJobHelper
    {
        [ReadOnly] public int inOutChannels;
        [ReadOnly] public int isGammaAVector; //1 if true, 0 if false
    }

    internal partial struct LeakyReluJobHelper
    {
        // from Theano impl
        // https://github.com/Theano/theano/blob/d395439aec5a6ddde8ef5c266fd976412a5c5695/theano/tensor/nnet/nnet.py#L2209-L2251
        [ReadOnly] public float f1, f2, alpha_;
        public float alpha { get { return alpha_; } set {
            alpha_ = value;
            f1 = 0.5f * (1f + alpha_);
            f2 = 0.5f * (1f - alpha_);
        } }
    }

    internal partial struct CopyJobHelper
    {
        [ReadOnly] public int length;
    }

    internal partial struct CopyStrideJobHelper
    {
        [ReadOnly] public int XStride;
        [ReadOnly] public int OStride;
        [ReadOnly] public int count;
        [ReadOnly] public int length;
    }

    internal partial struct GenericSliceJobHelper
    {
        [ReadOnly] public TensorShape shapeO;
        [ReadOnly] public TensorShape shapeX;
        [ReadOnly] public int strideS, strideR, strideN, strideT;
        [ReadOnly] public int strideD, strideH, strideW, strideC;
        [ReadOnly] public int startS, startR, startN, startT;
        [ReadOnly] public int startD, startH, startW, startC;
    }

    internal partial struct GenericStridedSliceJobHelper
    {
        [ReadOnly] public TensorShape shapeO;
        [ReadOnly] public TensorShape shapeX;
        [ReadOnly] public int strideS, strideR, strideN, strideT;
        [ReadOnly] public int strideD, strideH, strideW, strideC;
        [ReadOnly] public int startS, startR, startN, startT;
        [ReadOnly] public int startD, startH, startW, startC;
    }

    internal partial struct Border2DJobHelper
    {
        [ReadOnly] public TensorShape shapeO;
        [ReadOnly] public TensorShape shapeX;
        [ReadOnly] public int PadWidth;
        [ReadOnly] public int PadHeight;
        [ReadOnly] public int PadChannels;
        [ReadOnly] public int CroppedWidth;
        [ReadOnly] public int CroppedHeight;
        [ReadOnly] public int CroppedChannels;
        [ReadOnly] public float Beta;
    }

    internal unsafe partial struct TransposeJobHelper
    {
        [ReadOnly] public TensorShape shapeO;
        [ReadOnly] public TensorShape shapeX;
        [ReadOnly] public fixed int permutations[8];
    }

    internal partial struct Pad2DEdgeJobHelper
    {
        [ReadOnly] public TensorShape shapeO;
        [ReadOnly] public TensorShape shapeX;
        [ReadOnly] public int PadWidth;
        [ReadOnly] public int PadHeight;
        [ReadOnly] public int PadChannels;
    }

    internal partial struct Pad2DReflectJobHelper
    {
        [ReadOnly] public TensorShape shapeO;
        [ReadOnly] public TensorShape shapeX;
        [ReadOnly] public int PadWidth;
        [ReadOnly] public int PadHeight;
        [ReadOnly] public int PadChannels;
    }

    internal partial struct Pad2DSymmetricJobHelper
    {
        [ReadOnly] public TensorShape shapeO;
        [ReadOnly] public TensorShape shapeX;
        [ReadOnly] public int PadWidth;
        [ReadOnly] public int PadHeight;
        [ReadOnly] public int PadChannels;
    }

    internal partial struct TileJobHelper
    {
        [ReadOnly] public TensorShape shapeO;
        [ReadOnly] public TensorShape shapeX;
    }

    internal partial struct GatherJobHelper
    {
        [ReadOnly] public TensorShape shapeO;
        [ReadOnly] public TensorShape shapeX;
        [ReadOnly] public int axis;
    }

    internal partial struct OneHotJobHelper
    {
        [ReadOnly] public TensorShape shapeO;
        [ReadOnly] public TensorShape shapeX;
        [ReadOnly] public int depth;
        [ReadOnly] public int inputRank;
        [ReadOnly] public float onValue;
        [ReadOnly] public float offValue;
    }

    internal partial struct RandomNormalJobHelper
    {
        public Unity.Mathematics.Random rng;
        public float mean;
        public float scale;
    }

    internal partial struct RandomUniformJobHelper
    {
        public Unity.Mathematics.Random rng;
        public float mean;
        public float scale;
    }

    internal partial struct TestXOJobHelper
    {
        public int offset;
        public float bias;
    }

    internal partial struct TestXBOJobHelper
    {
        public int offset;
    }

    internal partial struct VectorBroadcastScaleBiasJobHelper
    {
        [ReadOnly] public int inOutChannels;
        [ReadOnly] public float alpha;
    }

    internal partial struct DepthwiseConv2DJobHelper
    {
        [ReadOnly] public int strideX, strideY, padX, padY;
        [ReadOnly] public int inHeight, inWidth, inChannels, inStrideN, inStrideH, inStrideW;
        [ReadOnly] public int kernelCount, kernelHeight, kernelWidth, kernelStrideH, kernelStrideW;
        [ReadOnly] public int outBatch, outWidth, outStrideN, outStrideH, outStrideW;
    }

    internal partial struct Dense3JobHelper
    {
        public int AM, AN;
        public int BM, BN;
        public int SM, SN;
        public int dispatchThreadX, dispatchThreadY, dispatchThreadZ;
    }

    internal partial struct ReduceMaxJobHelper
    {
        [ReadOnly] public int offsetReduce;
        [ReadOnly] public int reduceDim;
    }

    internal partial struct ReduceSumJobHelper
    {
        [ReadOnly] public int offsetReduce;
        [ReadOnly] public int reduceDim;
    }

    internal partial struct ReduceMeanJobHelper
    {
        [ReadOnly] public int offsetReduce;
        [ReadOnly] public int reduceDim;
    }

    internal partial struct ExpBiasReduceJobHelper
    {
        [ReadOnly] public int offsetReduce;
        [ReadOnly] public int reduceDim;
    }

    internal partial struct SoftmaxEndJobHelper
    {
        [ReadOnly] public int offsetReduce;
        [ReadOnly] public int reduceDim;
    }

    internal partial struct LogSoftmaxEndJobHelper
    {
        [ReadOnly] public int offsetReduce;
        [ReadOnly] public int reduceDim;
    }

    internal partial struct MaxPool2DJobHelper
    {
        [ReadOnly] public int strideX, strideY, padX, padY;
        [ReadOnly] public int kernelHeight, kernelWidth;
        [ReadOnly] public int inHeight, inWidth, inChannels, inStrideN, inStrideH, inStrideW;
        [ReadOnly] public int outBatch, outWidth, outStrideN, outStrideH, outStrideW;
    }

    internal partial struct AvgPool2DJobHelper
    {
        [ReadOnly] public int strideX, strideY, padX, padY;
        [ReadOnly] public int kernelHeight, kernelWidth;
        [ReadOnly] public int inHeight, inWidth, inChannels, inStrideN, inStrideH, inStrideW;
        [ReadOnly] public int outBatch, outWidth, outStrideN, outStrideH, outStrideW;
    }


    #endregion


    static unsafe float* AllocBlock(int blockSizeM, int blockSizeN)
    {
        int sz = blockSizeM * blockSizeN * sizeof(float);
        // Allocator.Temp is the fastest allocator, but can only be used within jobs; No explicit need to deallocate
        // Source: https://docs.unity3d.com/Packages/com.unity.collections@1.0/manual/allocation.html#allocatortemp
        return (float*)UnsafeUtility.Malloc(sz, JobsUtility.CacheLineSize, Allocator.Temp);
    }

    static unsafe half* AllocBlockHalf(int blockSizeM, int blockSizeN)
    {
        int sz = blockSizeM * blockSizeN * sizeof(half);
        // Allocator.Temp is the fastest allocator, but can only be used within jobs; No explicit need to deallocate
        // Source: https://docs.unity3d.com/Packages/com.unity.collections@1.0/manual/allocation.html#allocatortemp
        return (half*)UnsafeUtility.Malloc(sz, JobsUtility.CacheLineSize, Allocator.Temp);
    }

    static unsafe void FreeBlock(void* ptr)
    {
        // We are using Allocator.Temp, so there is no explicit need to deallocate
        // if (ptr != null)
        //     UnsafeUtility.Free(ptr, Allocator.Temp);
    }

    static unsafe void CopyBlock(float* blockOut, float* matrixIn, int row, int M, int col, int N, int blockSizeM, int blockSizeN)
    {
        var rowFinal = Math.Min(row + blockSizeM, M);
        var count = Math.Min(col + blockSizeN, N) - col;

        for (var i = row; i < rowFinal; i++)
            MatrixUtils.CopyFloatArray(blockOut + (i - row) * blockSizeN, matrixIn + i * N + col, count);
    }

    static unsafe int CopyBlockWithPadding(float* matrixIn, int row, int M, int col, int N, float* blockOut, int blockSizeM, int blockSizeN, bool transpose = false)
    {
        MatrixUtils.ClearFloatArray(blockOut, 0, blockSizeM * blockSizeN);
        var blockOutStride = blockSizeN;

        var rowFinal = Math.Min(row + blockSizeM, M);
        var count = Math.Min(col + blockSizeN, N) - col;

        // @TODO: measure which one is better - sequential access over matrix memory or blockOut cache
        if (transpose)
        {
            // sequential access over matrixIn, strided over blockOut
            for (var j = 0; j < count; ++j)
            for (var i = row; i < rowFinal; i++)
                blockOut[(i - row) * blockOutStride + j] = matrixIn[i + (col + j) * M];
        }
        else
            for (var i = row; i < rowFinal; i++)
            {
                MatrixUtils.CopyFloatArray(matrixIn + i * N + col, blockOut + (i - row) * blockOutStride, count);
            }
        return blockOutStride;
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    internal unsafe struct MatrixMultiplyJob : IJobParallelFor
    {
        // Convention: M x N matrices (other areas in our code may be N x M)
        [NoAlias][NativeDisableUnsafePtrRestriction][ReadOnly] public unsafe float* A;
        public int AM, AN;
        [NoAlias][NativeDisableUnsafePtrRestriction][ReadOnly] public unsafe float* B;
        public int BM, BN;
        [NoAlias][NativeDisableUnsafePtrRestriction]           public unsafe float* C;
        public int CM, CN;
        public bool transposeA;
        public bool transposeB;

        public int blockSizeM;
        public int blockSizeN;
        public int blockSizeK;

        public JobHandle Schedule(JobHandle dependsOn)
        {
            return Schedule(blocksBatchCount:1, dependsOn);
        }

        public JobHandle Schedule(int blocksBatchCount, JobHandle dependsOn)
        {
            if (transposeA)
            {
                int tmp = AM; AM = AN; AN = tmp;
            }
            if (transposeB)
            {
                int tmp = BM; BM = BN; BN = tmp;
            }

            // TODO: Determine optimal kernel / block sizes for mobile/console; This code path is currently not used
            // in production and instead MatrixMultiplyLegacyJob; However, this kernel size seemed to work best with
            // mobile; An alternative is have codegen generate the whole job + kernel, so we can switch dynamically
            // at runtime.
#if UNITY_ANDROID || UNITY_IOS || UNITY_WSA || UNITY_PS4 || UNITY_PS5 || UNITY_XBOXONE
            if (blockSizeM == 0 || blockSizeN == 0 || blockSizeK == 0)
            {
                blockSizeM = 64;
                blockSizeN = 64;
                blockSizeK = 16;
            }
#else
            if (blockSizeM == 0 || blockSizeN == 0 || blockSizeK == 0)
            {
                // Profiling across a range of matrices for best block size revealed:
                // (32, 384, 16) was the best common block size for matrices <= 576
                // (32, 768, 32) for matrices > 576 and <= 1152
                // (64, 96, 32) for matrices > 1200
                int maxM = 32;
                int maxN = 384;
                int maxK = 16;

                if (AM > 1200)
                {
                    maxM = 64;
                    maxN = 96;
                    maxK = 32;
                }
                else if (AM > 576)
                {
                    maxM = 32;
                    maxN = 768;
                    maxK = 32;
                }

                blockSizeM = Mathf.Min(AM, maxM);

                const int kernelWidth = 24;
                var sizeN = Mathf.ClosestPowerOfTwo(AN);
                sizeN = (sizeN / kernelWidth) * kernelWidth;
                sizeN = Mathf.Max(sizeN, kernelWidth);
                blockSizeN = Mathf.Min(sizeN, maxN);

                // Adjust block size down to the actual count of rows, so no allocation takes place needlessly
                blockSizeK = Mathf.Min(BM, maxK);
            }
#endif

            // Distribute jobs over a single axis
            int longerAxis = AM;
            int blockSizeForLongerAxis = blockSizeM;
            if (BN > AM)
            {
                longerAxis = BN; blockSizeForLongerAxis = blockSizeN;
            }

            var workElements = (longerAxis + blockSizeForLongerAxis - 1) / blockSizeForLongerAxis;
            return IJobParallelForExtensions.Schedule(this, workElements, blocksBatchCount, dependsOn);
        }

        public void Execute(int i)
        {
            int shorterAxis = BN;
            int blockSizeForShorterAxis = blockSizeN;
            if (BN > AM)
            {
                shorterAxis = AM; blockSizeForShorterAxis = blockSizeM;
            }

            float* blockTempA = null;
            float* blockTempB = null;
            float* blockTempC = null;

            // this job is scheduled over the Max(AN, BM)
            // need to pick the remaining (shorter) axis
            for (int j = 0; j < shorterAxis; j += blockSizeForShorterAxis)
            {
                int rowA = (AM >= BN) ? i * blockSizeM: j;
                int colB = (AM >= BN) ? j             : i * blockSizeN;

                float* blockC = C + rowA * CN + colB;
                int strideC = CN;

                if (rowA + blockSizeM > CM || colB + blockSizeN > CN) // copy remainder of C into zero-padded block
                {
                    if (blockTempC == null)
                        blockTempC = AllocBlock(blockSizeM, blockSizeN);
                    blockC = blockTempC;
                    strideC = CopyBlockWithPadding(C, rowA, CM, colB, CN, blockC, blockSizeM, blockSizeN);
                }

                for (int l = 0; l < AN; l += blockSizeK) // inner-loop
                {
                    float* blockA = A + rowA * AN + l;
                    float* blockB = B + l * BN + colB;
                    int strideA = AN;
                    int strideB = BN;

                    if (rowA + blockSizeM > AM || l + blockSizeK > AN || transposeA) // copy remainder of A or transposed A into zero-padded block
                    {
                        if (blockTempA == null)
                            blockTempA = AllocBlock(blockSizeM, blockSizeK);
                        blockA = blockTempA;
                        strideA = CopyBlockWithPadding(A, rowA, AM, l, AN, blockA, blockSizeM, blockSizeK, transposeA);
                    }

                    if (colB + blockSizeN > BN || l + blockSizeK > BM || transposeB) // copy remainder of A or transposed A into zero-padded block
                    {
                        if (blockTempB == null)
                            blockTempB = AllocBlock(blockSizeK, blockSizeN);
                        blockB = blockTempB;
                        strideB = CopyBlockWithPadding(B, l, BM, colB, BN, blockB, blockSizeK, blockSizeN, transposeB);
                    }

// Use defines instead of Application.isMobilePlatform || Application.isConsolePlatform, so we don't interrupt Burst
// inlining or introduce a branch here in the inner loop
#if UNITY_ANDROID || UNITY_IOS || UNITY_WSA || UNITY_PS4 || UNITY_PS5 || UNITY_XBOXONE
                    MultiplyBlockUnroll1x8(blockA, strideA, blockB, strideB, blockC, strideC,
                        blockSizeM, blockSizeK, Math.Min(blockSizeN, BN - colB));
#else
                    MultiplyBlockUnroll3x24(blockA, strideA, blockB, strideB, blockC, strideC,
                        blockSizeM, blockSizeK, Math.Min(blockSizeN, BN - colB));
#endif
                }

                if (blockC == blockTempC) // copy back
                    CopyBlock(blockC, C, rowA, CM, colB, CN, blockSizeM, blockSizeN);

                FreeBlock(blockTempA);
                FreeBlock(blockTempB);
                FreeBlock(blockTempC);
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    unsafe struct MatrixMultiplyLegacyJob : IJobParallelFor
    {
        [NoAlias][NativeDisableUnsafePtrRestriction][ReadOnly] public unsafe float* A;
        public int AM, AN;
        [NoAlias][NativeDisableUnsafePtrRestriction][ReadOnly] public unsafe float* B;
        public int BM, BN;
        [NoAlias][NativeDisableUnsafePtrRestriction]           public unsafe float* C;
        public int CM, CN;
        public bool transposeA;
        public bool transposeB;

        public const int blockSize = 16;

        public JobHandle Schedule(JobHandle dependsOn)
        {
            return Schedule(blocksBatchCount:1, dependsOn);
        }
        public JobHandle Schedule(int blocksBatchCount, JobHandle dependsOn)
        {
            if (transposeA)
            {
                int tmp = AM; AM = AN; AN = tmp;
            }
            if (transposeB)
            {
                int tmp = BM; BM = BN; BN = tmp;
            }

            int n = math.max(AM, BN);
            int workElements = (n + blockSize - 1) / blockSize;
            return IJobParallelForExtensions.Schedule(this, workElements, blocksBatchCount, dependsOn);
        }

        public void Execute(int i)
        {
            int bs = blockSize;
            unsafe
            {
                float* blockTempA = null;
                float* blockTempB = null;
                float* blockTempC = null;

                // this job is scheduled over the Max(AN, BM)
                // need to pick the remaining (shorter) axis
                for (int j = 0; j < Math.Min(AM, BN); j += bs)
                {
                    int rowA = (AM > BN) ? i * bs: j;
                    int colB = (AM > BN) ? j     : i * bs;

                    float* blockC = C + rowA * CN + colB;
                    int strideC = CN;

                    if (rowA + bs > CM || colB + bs > CN) // copy remainder of C into zero-padded block
                    {
                        if (blockTempC == null)
                            blockTempC = AllocBlock();
                        blockC = blockTempC;
                        strideC = bs;
                        MatrixUtils.CopyBlockWithPadding(C, rowA, CM, colB, CN, blockC, bs);
                    }

                    for (int l = 0; l < AN; l += bs) // inner-loop
                    {
                        float* blockA = A + rowA * AN +    l;
                        float* blockB = B +    l * BN + colB;
                        int strideA = AN;
                        int strideB = BN;

                        if (rowA + bs > AM || l + bs > AN || transposeA) // copy remainder of A or transposed A into zero-padded block
                        {
                            if (blockTempA == null)
                                blockTempA = AllocBlock();
                            blockA = blockTempA;
                            strideA = bs;
                            MatrixUtils.CopyBlockWithPadding(A, rowA, AM,    l, AN, blockA, bs, transposeA);
                        }

                        if (colB + bs > BN || l + bs > BM || transposeB) // copy remainder of A or transposed A into zero-padded block
                        {
                            if (blockTempB == null)
                                blockTempB = AllocBlock();
                            blockB = blockTempB;
                            strideB = bs;
                            MatrixUtils.CopyBlockWithPadding(B,    l, BM, colB, BN, blockB, bs, transposeB);
                        }

						MultiplyBlockUnrollHx16(blockA, strideA, blockB, strideB, blockC, strideC);
                    }

                    if (blockC == blockTempC) // copy back
                        MatrixUtils.CopyBlockWithPadding(blockC, C, rowA, CM, colB, CN, bs);
                }

                FreeBlock(blockTempA);
                FreeBlock(blockTempB);
                FreeBlock(blockTempC);
            }
        }

        static unsafe float* AllocBlock()
        {
            const int sz = blockSize * blockSize * sizeof(float);
            return (float*)UnsafeUtility.Malloc(sz, JobsUtility.CacheLineSize, Allocator.TempJob);
        }

        static unsafe void FreeBlock(float* ptr)
        {
            if (ptr != null)
                UnsafeUtility.Free(ptr, Allocator.TempJob);
        }

        static unsafe void MultiplyBlockUnrollHx16(float* Ap, int Astride, float* Bp, int Bstride, float* Cp, int Cstride)
        {
            for (int i = 0; i < blockSize; i++)
            {
                for (int j = 0; j < blockSize; j += 16)
                {
                    int baseC = i * Cstride + j;
                    float sum0 = *(Cp + baseC + 0);
                    float sum1 = *(Cp + baseC + 1);
                    float sum2 = *(Cp + baseC + 2);
                    float sum3 = *(Cp + baseC + 3);
                    float sum4 = *(Cp + baseC + 4);
                    float sum5 = *(Cp + baseC + 5);
                    float sum6 = *(Cp + baseC + 6);
                    float sum7 = *(Cp + baseC + 7);
                    float sum8 = *(Cp + baseC + 8);
                    float sum9 = *(Cp + baseC + 9);
                    float sumA = *(Cp + baseC +10);
                    float sumB = *(Cp + baseC +11);
                    float sumC = *(Cp + baseC +12);
                    float sumD = *(Cp + baseC +13);
                    float sumE = *(Cp + baseC +14);
                    float sumF = *(Cp + baseC +15);

                    for (int l = 0; l < blockSize; l++)
                    {
                        float A = *(Ap + i * Astride + l);
                        int baseB = l * Bstride + j;

                        sum0 += A * (*(Bp + baseB + 0));
                        sum1 += A * (*(Bp + baseB + 1));
                        sum2 += A * (*(Bp + baseB + 2));
                        sum3 += A * (*(Bp + baseB + 3));
                        sum4 += A * (*(Bp + baseB + 4));
                        sum5 += A * (*(Bp + baseB + 5));
                        sum6 += A * (*(Bp + baseB + 6));
                        sum7 += A * (*(Bp + baseB + 7));
                        sum8 += A * (*(Bp + baseB + 8));
                        sum9 += A * (*(Bp + baseB + 9));
                        sumA += A * (*(Bp + baseB +10));
                        sumB += A * (*(Bp + baseB +11));
                        sumC += A * (*(Bp + baseB +12));
                        sumD += A * (*(Bp + baseB +13));
                        sumE += A * (*(Bp + baseB +14));
                        sumF += A * (*(Bp + baseB +15));
                    }

                    *(Cp + baseC + 0) = sum0;
                    *(Cp + baseC + 1) = sum1;
                    *(Cp + baseC + 2) = sum2;
                    *(Cp + baseC + 3) = sum3;
                    *(Cp + baseC + 4) = sum4;
                    *(Cp + baseC + 5) = sum5;
                    *(Cp + baseC + 6) = sum6;
                    *(Cp + baseC + 7) = sum7;
                    *(Cp + baseC + 8) = sum8;
                    *(Cp + baseC + 9) = sum9;
                    *(Cp + baseC +10) = sumA;
                    *(Cp + baseC +11) = sumB;
                    *(Cp + baseC +12) = sumC;
                    *(Cp + baseC +13) = sumD;
                    *(Cp + baseC +14) = sumE;
                    *(Cp + baseC +15) = sumF;
                }
            }
		}
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    unsafe struct MatrixMultiply3x2Job : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Aptr => X.ptrfloat;
        public ReadOnlyMemResource B { get; set; } float* Bptr => B.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Cptr => O.ptrfloat;
        public int AM, AN;
        public int BM, BN;
        public int CM, CN;

        public int dispatchThreadX, dispatchThreadY, dispatchThreadZ;
        public const int blockSize = 16;

        public void Execute(int threadID)
        {

            int dispatchThreadXY = dispatchThreadX * dispatchThreadY;

            int batch = (threadID / dispatchThreadXY);
            int i = (threadID % dispatchThreadXY) % dispatchThreadX;
            int j = (threadID % dispatchThreadXY) / dispatchThreadX;

            int batchOffSetA = (batch * AM * AN);
            int batchOffSetC = (batch * CM * CN);

            int rowA = i * blockSize;
            int colB = j * blockSize;

            unsafe
            {
                float* blockTempA = null;
                float* blockTempB = null;
                float* blockTempC = null;

                float* blockC = Cptr + rowA + CM * colB + batchOffSetC;
                int strideC = CM;

                if (rowA + blockSize > CM || colB + blockSize > CN) // copy remainder of C into zero-padded block
                {
                    blockTempC = AllocBlock(blockSize, blockSize);
                    strideC = blockSize;
                    blockC = blockTempC;
                }
                for (int y = 0; y < blockSize; y++)
                    for (int x = 0; x < blockSize; x++)
                        blockC[x + strideC * y] = 0.0f;

                for (int l = 0; l < AN; l += blockSize) // inner-loop
                {
                    float* blockA = Aptr + rowA + AM * l + batchOffSetA;
                    float* blockB = Bptr + l * BN + colB;
                    int strideA = AM;
                    int strideB = BN;

                    if (rowA + blockSize > AM || l + blockSize > AN) // copy remainder of A into zero-padded block
                    {
                        if (blockTempA == null)
                            blockTempA = AllocBlock(blockSize, blockSize);
                        strideA = blockSize;

                        for (int y = 0; y < blockSize; y++)
                            for (int x = 0; x < blockSize; x++)
                                blockTempA[x + blockSize * y] = ((rowA + x) < AM && (l + y < AN)) ? blockA[x + AM * y] : 0.0f;

                        blockA = blockTempA;
                    }

                    if (colB + blockSize > BN || l + blockSize > BM) // copy remainder of B into zero-padded block
                    {
                        if (blockTempB == null)
                            blockTempB = AllocBlock(blockSize, blockSize);
                        strideB = blockSize;

                        for (int y = 0; y < blockSize; y++)
                            for (int x = 0; x < blockSize; x++)
                                blockTempB[x + blockSize * y] = ((colB + x) < BN && (l + y < BM)) ? blockB[x + BN * y] : 0.0f;

                        blockB = blockTempB;
                    }

                    MultiplyBlockUnrollHx16(blockA, strideA, blockB, strideB, blockC, strideC);
                }

                if (blockC == blockTempC) // copy back
                {
                    for (int y = 0; y < blockSize; y++)
                        for (int x = 0; x < blockSize; x++)
                        {
                            if (((rowA + x) < CM) && ((colB + y) < CN))
                                Cptr[(rowA + x) + CM * (colB + y) + batchOffSetC] = blockTempC[x + blockSize * y];
                        }
                }

                FreeBlock(blockTempA);
                FreeBlock(blockTempB);
                FreeBlock(blockTempC);
            }
        }

        static void MultiplyBlockUnrollHx16(float* Ap, int Astride, float* Bp, int Bstride, float* Cp, int Cstride)
        {
            for (int i = 0; i < blockSize; i++)
            {
                float sum0 = *(Cp + i + Cstride * 0);
                float sum1 = *(Cp + i + Cstride * 1);
                float sum2 = *(Cp + i + Cstride * 2);
                float sum3 = *(Cp + i + Cstride * 3);
                float sum4 = *(Cp + i + Cstride * 4);
                float sum5 = *(Cp + i + Cstride * 5);
                float sum6 = *(Cp + i + Cstride * 6);
                float sum7 = *(Cp + i + Cstride * 7);
                float sum8 = *(Cp + i + Cstride * 8);
                float sum9 = *(Cp + i + Cstride * 9);
                float sumA = *(Cp + i + Cstride * 10);
                float sumB = *(Cp + i + Cstride * 11);
                float sumC = *(Cp + i + Cstride * 12);
                float sumD = *(Cp + i + Cstride * 13);
                float sumE = *(Cp + i + Cstride * 14);
                float sumF = *(Cp + i + Cstride * 15);

                for (int l = 0; l < blockSize; l++)
                {
                    float A = *(Ap + i + Astride * l);

                    float B0 = *(Bp + l * Bstride + 0);
                    float B1 = *(Bp + l * Bstride + 1);
                    float B2 = *(Bp + l * Bstride + 2);
                    float B3 = *(Bp + l * Bstride + 3);
                    float B4 = *(Bp + l * Bstride + 4);
                    float B5 = *(Bp + l * Bstride + 5);
                    float B6 = *(Bp + l * Bstride + 6);
                    float B7 = *(Bp + l * Bstride + 7);
                    float B8 = *(Bp + l * Bstride + 8);
                    float B9 = *(Bp + l * Bstride + 9);
                    float BA = *(Bp + l * Bstride + 10);
                    float BB = *(Bp + l * Bstride + 11);
                    float BC = *(Bp + l * Bstride + 12);
                    float BD = *(Bp + l * Bstride + 13);
                    float BE = *(Bp + l * Bstride + 14);
                    float BF = *(Bp + l * Bstride + 15);


                    sum0 += A * B0;
                    sum1 += A * B1;
                    sum2 += A * B2;
                    sum3 += A * B3;
                    sum4 += A * B4;
                    sum5 += A * B5;
                    sum6 += A * B6;
                    sum7 += A * B7;
                    sum8 += A * B8;
                    sum9 += A * B9;
                    sumA += A * BA;
                    sumB += A * BB;
                    sumC += A * BC;
                    sumD += A * BD;
                    sumE += A * BE;
                    sumF += A * BF;
                }

                *(Cp + i + Cstride * 0 ) = sum0;
                *(Cp + i + Cstride * 1 ) = sum1;
                *(Cp + i + Cstride * 2 ) = sum2;
                *(Cp + i + Cstride * 3 ) = sum3;
                *(Cp + i + Cstride * 4 ) = sum4;
                *(Cp + i + Cstride * 5 ) = sum5;
                *(Cp + i + Cstride * 6 ) = sum6;
                *(Cp + i + Cstride * 7 ) = sum7;
                *(Cp + i + Cstride * 8 ) = sum8;
                *(Cp + i + Cstride * 9 ) = sum9;
                *(Cp + i + Cstride * 10) = sumA;
                *(Cp + i + Cstride * 11) = sumB;
                *(Cp + i + Cstride * 12) = sumC;
                *(Cp + i + Cstride * 13) = sumD;
                *(Cp + i + Cstride * 14) = sumE;
                *(Cp + i + Cstride * 15) = sumF;
            }
        }
    }


    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    unsafe struct MatrixMultiply4x4Job : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Aptr => X.ptrfloat;
        public ReadOnlyMemResource B { get; set; } float* Bptr => B.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Cptr => O.ptrfloat;
        public int AB0, AB1, AM, AN;
        public int BB0, BB1, BM, BN;
        public int CB1, CM, CN;

        public int dispatchThreadX, dispatchThreadY, dispatchThreadZ;
        public const int blockSize = 16;

        public void Execute(int threadID)
        {
            int dispatchThreadXY = dispatchThreadX * dispatchThreadY;

            int batch1 = (threadID % CB1);
            int batch0 = (threadID / CB1) / dispatchThreadXY;
            int i = ((threadID / CB1) % dispatchThreadXY) % dispatchThreadX;
            int j = ((threadID / CB1) % dispatchThreadXY) / dispatchThreadX;

            int batchOffSetA = ((batch0 % AB0) * AM * AN * AB1 + (batch1 % AB1));
            int batchOffSetB = ((batch0 % BB0) * BM * BN * BB1 + (batch1 % BB1));
            int batchOffSetC = (batch0 * CM * CN * CB1 + batch1);

            int rowA = i * blockSize;
            int colB = j * blockSize;

            unsafe
            {
                float* blockTempA = null;
                float* blockTempB = null;
                float* blockTempC = null;

                float* blockC = Cptr + (rowA * CN + colB)*CB1 + batchOffSetC;
                int strideC = CN;
                int strideBatchC = CB1;

                if (rowA + blockSize > CM || colB + blockSize > CN) // copy remainder of A into zero-padded block
                {
                    blockTempC = AllocBlock(blockSize, blockSize);
                    strideC = blockSize;
                    strideBatchC = 1;
                    blockC = blockTempC;
                }
                for (int y = 0; y < blockSize; y++)
                    for (int x = 0; x < blockSize; x++)
                        blockC[(x + strideC * y) * strideBatchC] = 0.0f;

                for (int l = 0; l < AN; l += blockSize) // inner-loop
                {
                    float* blockA = Aptr + (rowA * AN + l)*AB1 + batchOffSetA;
                    float* blockB = Bptr + (l * BN + colB)*BB1 + batchOffSetB;
                    int strideA = AN;
                    int strideBatchA = AB1;
                    int strideB = BN;
                    int strideBatchB = BB1;

                    if (rowA + blockSize > AM || l + blockSize > AN) // copy remainder of A into zero-padded block
                    {
                        if (blockTempA == null)
                            blockTempA = AllocBlock(blockSize, blockSize);
                        strideA = blockSize;
                        strideBatchA = 1;

                        for (int y = 0; y < blockSize; y++)
                            for (int x = 0; x < blockSize; x++)
                                blockTempA[x + blockSize * y] = ((rowA + y) < AM && (l + x < AN)) ? blockA[(x + AN * y)*AB1] : 0.0f;

                        blockA = blockTempA;
                    }

                    if (colB + blockSize > BN || l + blockSize > BM) // copy remainder of A into zero-padded block
                    {
                        if (blockTempB == null)
                            blockTempB = AllocBlock(blockSize, blockSize);
                        strideB = blockSize;
                        strideBatchB = 1;

                        for (int y = 0; y < blockSize; y++)
                            for (int x = 0; x < blockSize; x++)
                                blockTempB[x + blockSize * y] = ((colB + x) < BN && (l + y < BM)) ? blockB[(x + BN * y)*BB1] : 0.0f;

                        blockB = blockTempB;
                    }

                    MultiplyBlockUnrollHx16(blockA, strideA, strideBatchA, blockB, strideB, strideBatchB, blockC, strideC, strideBatchC);
                }

                if (blockC == blockTempC) // copy back
                {
                    for (int y = 0; y < blockSize; y++)
                    for (int x = 0; x < blockSize; x++)
                    {
                        if (((rowA + y) < CM) && (colB + x < CN))
                            Cptr[((rowA + y) * CN + (colB + x)) * CB1 + batchOffSetC] = blockTempC[x + blockSize * y];
                    }
                }

                FreeBlock(blockTempA);
                FreeBlock(blockTempB);
                FreeBlock(blockTempC);
            }
        }

        static void MultiplyBlockUnrollHx16(float* Ap, int Astride, int ABatchStride, float* Bp, int Bstride, int BBatchStride, float* Cp, int Cstride, int CBatchStride)
        {
            for (int i = 0; i < blockSize; i++)
            {
                float sum0 = *(Cp + (i * Cstride + 0 )*CBatchStride);
                float sum1 = *(Cp + (i * Cstride + 1 )*CBatchStride);
                float sum2 = *(Cp + (i * Cstride + 2 )*CBatchStride);
                float sum3 = *(Cp + (i * Cstride + 3 )*CBatchStride);
                float sum4 = *(Cp + (i * Cstride + 4 )*CBatchStride);
                float sum5 = *(Cp + (i * Cstride + 5 )*CBatchStride);
                float sum6 = *(Cp + (i * Cstride + 6 )*CBatchStride);
                float sum7 = *(Cp + (i * Cstride + 7 )*CBatchStride);
                float sum8 = *(Cp + (i * Cstride + 8 )*CBatchStride);
                float sum9 = *(Cp + (i * Cstride + 9 )*CBatchStride);
                float sumA = *(Cp + (i * Cstride + 10)*CBatchStride);
                float sumB = *(Cp + (i * Cstride + 11)*CBatchStride);
                float sumC = *(Cp + (i * Cstride + 12)*CBatchStride);
                float sumD = *(Cp + (i * Cstride + 13)*CBatchStride);
                float sumE = *(Cp + (i * Cstride + 14)*CBatchStride);
                float sumF = *(Cp + (i * Cstride + 15)*CBatchStride);

                for (int l = 0; l < blockSize; l++)
                {
                    float A = *(Ap + (i * Astride + l)*ABatchStride);

                    float B0 = *(Bp + (l * Bstride + 0 )*BBatchStride);
                    float B1 = *(Bp + (l * Bstride + 1 )*BBatchStride);
                    float B2 = *(Bp + (l * Bstride + 2 )*BBatchStride);
                    float B3 = *(Bp + (l * Bstride + 3 )*BBatchStride);
                    float B4 = *(Bp + (l * Bstride + 4 )*BBatchStride);
                    float B5 = *(Bp + (l * Bstride + 5 )*BBatchStride);
                    float B6 = *(Bp + (l * Bstride + 6 )*BBatchStride);
                    float B7 = *(Bp + (l * Bstride + 7 )*BBatchStride);
                    float B8 = *(Bp + (l * Bstride + 8 )*BBatchStride);
                    float B9 = *(Bp + (l * Bstride + 9 )*BBatchStride);
                    float BA = *(Bp + (l * Bstride + 10)*BBatchStride);
                    float BB = *(Bp + (l * Bstride + 11)*BBatchStride);
                    float BC = *(Bp + (l * Bstride + 12)*BBatchStride);
                    float BD = *(Bp + (l * Bstride + 13)*BBatchStride);
                    float BE = *(Bp + (l * Bstride + 14)*BBatchStride);
                    float BF = *(Bp + (l * Bstride + 15)*BBatchStride);

                    sum0 += A * B0;
                    sum1 += A * B1;
                    sum2 += A * B2;
                    sum3 += A * B3;
                    sum4 += A * B4;
                    sum5 += A * B5;
                    sum6 += A * B6;
                    sum7 += A * B7;
                    sum8 += A * B8;
                    sum9 += A * B9;
                    sumA += A * BA;
                    sumB += A * BB;
                    sumC += A * BC;
                    sumD += A * BD;
                    sumE += A * BE;
                    sumF += A * BF;
                }

                *(Cp + (i * Cstride + 0 )*CBatchStride) = sum0;
                *(Cp + (i * Cstride + 1 )*CBatchStride) = sum1;
                *(Cp + (i * Cstride + 2 )*CBatchStride) = sum2;
                *(Cp + (i * Cstride + 3 )*CBatchStride) = sum3;
                *(Cp + (i * Cstride + 4 )*CBatchStride) = sum4;
                *(Cp + (i * Cstride + 5 )*CBatchStride) = sum5;
                *(Cp + (i * Cstride + 6 )*CBatchStride) = sum6;
                *(Cp + (i * Cstride + 7 )*CBatchStride) = sum7;
                *(Cp + (i * Cstride + 8 )*CBatchStride) = sum8;
                *(Cp + (i * Cstride + 9 )*CBatchStride) = sum9;
                *(Cp + (i * Cstride + 10)*CBatchStride) = sumA;
                *(Cp + (i * Cstride + 11)*CBatchStride) = sumB;
                *(Cp + (i * Cstride + 12)*CBatchStride) = sumC;
                *(Cp + (i * Cstride + 13)*CBatchStride) = sumD;
                *(Cp + (i * Cstride + 14)*CBatchStride) = sumE;
                *(Cp + (i * Cstride + 15)*CBatchStride) = sumF;
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    unsafe struct ConvertHalfToFloatJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;

        public void Execute(int threadID)
        {
            Optr[threadID] = (float)(Xptr[threadID]);
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    unsafe struct ConvertFloatToHalfJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;

        public void Execute(int threadID)
        {
            Optr[threadID] = (half)(Xptr[threadID]);
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    unsafe struct Im2ColSliceJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        [ReadOnly] public int inOutBatch, inOutChannels;
        [ReadOnly] public int inHeight,  inStrideN,  inStrideH, inStrideW;
        [ReadOnly] public int outWidth, outStrideN, outStrideH;
        [ReadOnly] public int strideX, strideY, offsetY;
        [ReadOnly] public int padLeft, padRight, skipFromInputRow, copyFromInputRow;
        public void Execute(int y)
        {
            for (int n = 0; n < inOutBatch; ++n)
            {
                int readY = strideY * y + offsetY;
                float* from = X.ptrfloat + n *  inStrideN + readY *  inStrideH + skipFromInputRow * inStrideW;
                float* to   = O.ptrfloat + n * outStrideN +     y * outStrideH;

                if (readY < 0 ||
                    readY >= inHeight)
                {
                    // pad-0 top or bottom line, len = outWidth
                    UnsafeUtility.MemClear(destination: to,
                                           size:        inOutChannels * outWidth * sizeof(float));
                    to += inOutChannels * outWidth;
                }
                else
                {
                    // pad-0 left, len = padLeft
                    UnsafeUtility.MemClear(destination: to,
                                           size:        inOutChannels * padLeft * sizeof(float));
                    to += inOutChannels * padLeft;

                    // copy from X with stride, if necessary
                    if (strideX == 1)
                    {
                        UnsafeUtility.MemCpy(destination: to,
                                             source:      from,
                                             size:        inOutChannels * copyFromInputRow * sizeof(float));
                        to += inOutChannels * copyFromInputRow;
                    }
                    else
                    {
                        UnsafeUtility.MemCpyStride(destination: to,     destinationStride:        inOutChannels * sizeof(float),
                                                   source:      from,   sourceStride:   strideX * inOutChannels * sizeof(float),
                                                   elementSize: inOutChannels * sizeof(float),
                                                   count:       copyFromInputRow);
                        to += inOutChannels * copyFromInputRow;
                    }

                    // pad-0 right, len = padRight
                    UnsafeUtility.MemClear(destination: to,
                                           size:        inOutChannels * padRight * sizeof(float));
                    to += inOutChannels * padRight;
                }
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    unsafe struct ZeroBroadcastJob : IJob, IJobResourceDeclarationO
    {
        public ReadWriteMemResource O { get; set; }
        [ReadOnly] public int repeat;
        public void Execute()
        {
            UnsafeUtility.MemClear(destination: O.ptr, size: repeat * sizeof(float));
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    unsafe struct VectorBroadcastJob : IJob, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        [ReadOnly] public int channels;
        [ReadOnly] public int repeat;
        public void Execute()
        {
            UnsafeUtility.MemCpyReplicate(destination: O.ptr,
                                          source:      X.ptr,
                                          size:        channels * sizeof(float),
                                          count:       repeat);
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    unsafe struct MemFreeJob : IJob
    {
        [NoAlias] [NativeDisableUnsafePtrRestriction]           public void* buffer0;
        [NoAlias] [NativeDisableUnsafePtrRestriction]           public void* buffer1;
                                                     [ReadOnly] public Allocator allocator;
        public void Execute()
        {
            if (buffer0 != null)
                UnsafeUtility.Free(buffer0, allocator);
            if (buffer1 != null)
                UnsafeUtility.Free(buffer1, allocator);
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct LSTMEndJob : IJobParallelFor
    {
        [NoAlias][NativeDisableUnsafePtrRestriction][ReadOnly] public unsafe float* i_mad_w;
        [NoAlias][NativeDisableUnsafePtrRestriction][ReadOnly] public unsafe float* j_mad_w;
        [NoAlias][NativeDisableUnsafePtrRestriction][ReadOnly] public unsafe float* f_mad_w;
        [NoAlias][NativeDisableUnsafePtrRestriction][ReadOnly] public unsafe float* o_mad_w;
        [NoAlias][NativeDisableUnsafePtrRestriction][ReadOnly] public unsafe float* i_mad_r;
        [NoAlias][NativeDisableUnsafePtrRestriction][ReadOnly] public unsafe float* j_mad_r;
        [NoAlias][NativeDisableUnsafePtrRestriction][ReadOnly] public unsafe float* f_mad_r;
        [NoAlias][NativeDisableUnsafePtrRestriction][ReadOnly] public unsafe float* o_mad_r;

        [NoAlias][NativeDisableUnsafePtrRestriction][ReadOnly] public unsafe float* cell;

        [NoAlias][NativeDisableUnsafePtrRestriction] public unsafe float* O;
        [NoAlias][NativeDisableUnsafePtrRestriction] public unsafe float* cell_out;
        [NoAlias][NativeDisableUnsafePtrRestriction] public unsafe float* hidden_out;

        public int sequenceIndexO, sequenceIndexI;
        public int batchSize, hiddenSize;
        public int batchSizeR;

        public JobHandle Schedule(int arrayLength, int innerloopBatchCount, JobHandle dependsOn)
        {
            return IJobParallelForExtensions.Schedule(this, arrayLength, innerloopBatchCount, dependsOn);
        }

        public void Execute(int threadId)
        {
            int b_tID = (threadId / hiddenSize);
            int h_tID = (threadId % hiddenSize);
            int threadId_r = (b_tID % batchSizeR) * hiddenSize + h_tID;
            float i_mad = i_mad_w[batchSize * hiddenSize * sequenceIndexI + threadId] + i_mad_r[threadId_r];
            float j_mad = j_mad_w[batchSize * hiddenSize * sequenceIndexI + threadId] + j_mad_r[threadId_r];
            float f_mad = f_mad_w[batchSize * hiddenSize * sequenceIndexI + threadId] + f_mad_r[threadId_r];
            float o_mad = o_mad_w[batchSize * hiddenSize * sequenceIndexI + threadId] + o_mad_r[threadId_r];

            float i = 1f / (1f + math.exp(-i_mad));
            float j = math.tanh(j_mad);
            float f = 1f / (1f + math.exp(-f_mad));
            float o = 1f / (1f + math.exp(-o_mad));

            float state_c_mul = cell[threadId_r] * f;
            float i_j_mul = i * j;
            float state_c = state_c_mul + i_j_mul;
            float state_c_tanh = math.tanh(state_c);
            float state_h = o * state_c_tanh;

            O[batchSize * hiddenSize * sequenceIndexO + threadId] = state_h;
            hidden_out[threadId] = state_h;
            cell_out[threadId] = state_c;
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    unsafe struct LSTMDense3Job : IJobParallelFor
    {
        [NoAlias][NativeDisableUnsafePtrRestriction][ReadOnly] public unsafe float* A;
        public int AM, AN;
        [NoAlias][NativeDisableUnsafePtrRestriction][ReadOnly] public unsafe float* B;
        public int BM, BN;
        [NoAlias][NativeDisableUnsafePtrRestriction][ReadOnly] public unsafe float* C;
        public int CN;

        [NoAlias][NativeDisableUnsafePtrRestriction] public unsafe float* S;
        public int SM, SN;

        public int dispatchThreadX, dispatchThreadY, dispatchThreadZ;
        public const int blockSize = 16;

        public JobHandle Schedule(JobHandle dependsOn)
        {
            return Schedule(blocksBatchCount:1, dependsOn);
        }
        public JobHandle Schedule(int blocksBatchCount, JobHandle dependsOn)
        {
            return IJobParallelForExtensions.Schedule(this, dispatchThreadX * dispatchThreadY * dispatchThreadZ, blocksBatchCount, dependsOn);
        }

        public void Execute(int threadID)
        {
            int dispatchThreadXY = dispatchThreadX * dispatchThreadY;

            int batch = (threadID / dispatchThreadXY);
            int i = (threadID % dispatchThreadXY) % dispatchThreadX;
            int j = (threadID % dispatchThreadXY) / dispatchThreadX;

            int batchOffSetA = (batch * AM * AN);
            int batchOffSetS = (batch * SM * SN);

            int rowA = i * blockSize;
            int colB = j * blockSize;

            unsafe
            {
                float* blockTempA = null;
                float* blockTempB = null;
                float* blockTempS = null;

                float* blockS = S + rowA * SN + colB + batchOffSetS;
                int strideS = SN;

                if (rowA + blockSize > SM || colB + blockSize > SN) // copy remainder of C into zero-padded block
                {
                    blockTempS = AllocBlock(blockSize, blockSize);
                    strideS = blockSize;
                    blockS = blockTempS;
                }
                for (int y = 0; y < blockSize; y++)
                    for (int x = 0; x < blockSize; x++)
                        blockS[x + strideS * y] = (colB + x) < BN ? C[(colB + x)%CN] : 0.0f;

                for (int l = 0; l < AN; l += blockSize) // inner-loop
                {
                    float* blockA = A + rowA * AN + l + batchOffSetA;
                    float* blockB = B + l * BN + colB;
                    int strideA = AN;
                    int strideB = BN;

                    if (rowA + blockSize > AM || l + blockSize > AN) // copy remainder of A into zero-padded block
                    {
                        if (blockTempA == null)
                            blockTempA = AllocBlock(blockSize, blockSize);
                        strideA = blockSize;

                        for (int y = 0; y < blockSize; y++)
                            for (int x = 0; x < blockSize; x++)
                                blockTempA[x + blockSize * y] = ((rowA + y) < AM && (l + x < AN)) ? blockA[x + AN * y] : 0.0f;

                        blockA = blockTempA;
                    }

                    if (colB + blockSize > BN || l + blockSize > BM) // copy remainder of B into zero-padded block
                    {
                        if (blockTempB == null)
                            blockTempB = AllocBlock(blockSize, blockSize);
                        strideB = blockSize;

                        for (int y = 0; y < blockSize; y++)
                            for (int x = 0; x < blockSize; x++)
                                blockTempB[x + blockSize * y] = ((colB + x) < BN && (l + y < BM)) ? blockB[x + BN * y] : 0.0f;

                        blockB = blockTempB;
                    }

                    MultiplyBlockUnrollHx16(blockA, strideA, blockB, strideB, blockS, strideS);
                }

                if (blockS == blockTempS) // copy back
                {
                    for (int y = 0; y < blockSize; y++)
                        for (int x = 0; x < blockSize; x++)
                        {
                            if (((rowA + y) < SM) && ((colB + x) < SN))
                                S[(rowA + y) * SN + (colB + x) + batchOffSetS] = blockTempS[x + blockSize * y];
                        }
                }

                FreeBlock(blockTempA);
                FreeBlock(blockTempB);
                FreeBlock(blockTempS);
            }
        }

        static void MultiplyBlockUnrollHx16(float* Ap, int Astride, float* Bp, int Bstride, float* Sp, int Sstride)
        {
            for (int i = 0; i < blockSize; i++)
            {
                float sum0 = *(Sp + i * Sstride + 0);
                float sum1 = *(Sp + i * Sstride + 1);
                float sum2 = *(Sp + i * Sstride + 2);
                float sum3 = *(Sp + i * Sstride + 3);
                float sum4 = *(Sp + i * Sstride + 4);
                float sum5 = *(Sp + i * Sstride + 5);
                float sum6 = *(Sp + i * Sstride + 6);
                float sum7 = *(Sp + i * Sstride + 7);
                float sum8 = *(Sp + i * Sstride + 8);
                float sum9 = *(Sp + i * Sstride + 9);
                float sumA = *(Sp + i * Sstride + 10);
                float sumB = *(Sp + i * Sstride + 11);
                float sumC = *(Sp + i * Sstride + 12);
                float sumD = *(Sp + i * Sstride + 13);
                float sumE = *(Sp + i * Sstride + 14);
                float sumF = *(Sp + i * Sstride + 15);

                for (int l = 0; l < blockSize; l++)
                {
                    float A = *(Ap + i * Astride + l);

                    float B0 = *(Bp + l * Bstride + 0);
                    float B1 = *(Bp + l * Bstride + 1);
                    float B2 = *(Bp + l * Bstride + 2);
                    float B3 = *(Bp + l * Bstride + 3);
                    float B4 = *(Bp + l * Bstride + 4);
                    float B5 = *(Bp + l * Bstride + 5);
                    float B6 = *(Bp + l * Bstride + 6);
                    float B7 = *(Bp + l * Bstride + 7);
                    float B8 = *(Bp + l * Bstride + 8);
                    float B9 = *(Bp + l * Bstride + 9);
                    float BA = *(Bp + l * Bstride + 10);
                    float BB = *(Bp + l * Bstride + 11);
                    float BC = *(Bp + l * Bstride + 12);
                    float BD = *(Bp + l * Bstride + 13);
                    float BE = *(Bp + l * Bstride + 14);
                    float BF = *(Bp + l * Bstride + 15);


                    sum0 += A * B0;
                    sum1 += A * B1;
                    sum2 += A * B2;
                    sum3 += A * B3;
                    sum4 += A * B4;
                    sum5 += A * B5;
                    sum6 += A * B6;
                    sum7 += A * B7;
                    sum8 += A * B8;
                    sum9 += A * B9;
                    sumA += A * BA;
                    sumB += A * BB;
                    sumC += A * BC;
                    sumD += A * BD;
                    sumE += A * BE;
                    sumF += A * BF;
                }

                *(Sp + i * Sstride + 0 ) = sum0;
                *(Sp + i * Sstride + 1 ) = sum1;
                *(Sp + i * Sstride + 2 ) = sum2;
                *(Sp + i * Sstride + 3 ) = sum3;
                *(Sp + i * Sstride + 4 ) = sum4;
                *(Sp + i * Sstride + 5 ) = sum5;
                *(Sp + i * Sstride + 6 ) = sum6;
                *(Sp + i * Sstride + 7 ) = sum7;
                *(Sp + i * Sstride + 8 ) = sum8;
                *(Sp + i * Sstride + 9 ) = sum9;
                *(Sp + i * Sstride + 10) = sumA;
                *(Sp + i * Sstride + 11) = sumB;
                *(Sp + i * Sstride + 12) = sumC;
                *(Sp + i * Sstride + 13) = sumD;
                *(Sp + i * Sstride + 14) = sumE;
                *(Sp + i * Sstride + 15) = sumF;
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    unsafe struct LSTMDenseJob : IJobParallelFor
    {
        [NoAlias][NativeDisableUnsafePtrRestriction][ReadOnly] public unsafe float* A;
        public int AM, AN;
        [NoAlias][NativeDisableUnsafePtrRestriction][ReadOnly] public unsafe float* B;
        public int BM, BN;
        [NoAlias][NativeDisableUnsafePtrRestriction][ReadOnly] public unsafe float* C;
        public int CN;

        [NoAlias][NativeDisableUnsafePtrRestriction] public unsafe float* S;
        public int SM, SN;

        public int dispatchThreadX, dispatchThreadY;
        public const int blockSize = 16;

        public JobHandle Schedule(JobHandle dependsOn)
        {
            return Schedule(blocksBatchCount: 1, dependsOn);
        }
        public JobHandle Schedule(int blocksBatchCount, JobHandle dependsOn)
        {
            return IJobParallelForExtensions.Schedule(this, dispatchThreadX * dispatchThreadY, blocksBatchCount, dependsOn);
        }


        public void Execute(int threadID)
        {
            int i = (threadID % dispatchThreadX);
            int j = (threadID / dispatchThreadX);

            int rowA = i * blockSize;
            int colB = j * blockSize;

            unsafe
            {
                float* blockTempA = null;
                float* blockTempB = null;
                float* blockTempS = null;

                float* blockS = S + rowA * SN + colB;
                int strideS = SN;

                if (rowA + blockSize > SM || colB + blockSize > SN) // copy remainder of C into zero-padded block
                {
                    blockTempS = AllocBlock(blockSize, blockSize);
                    strideS = blockSize;
                    blockS = blockTempS;
                }
                for (int y = 0; y < blockSize; y++)
                    for (int x = 0; x < blockSize; x++)
                        blockS[x + strideS * y] = (colB + x) < BN ? C[(colB + x)%CN] : 0.0f;

                for (int l = 0; l < AN; l += blockSize) // inner-loop
                {
                    float* blockA = A + rowA * AN + l;
                    float* blockB = B + l * BN + colB;
                    int strideA = AN;
                    int strideB = BN;

                    if (rowA + blockSize > AM || l + blockSize > AN) // copy remainder of A into zero-padded block
                    {
                        if (blockTempA == null)
                            blockTempA = AllocBlock(blockSize, blockSize);
                        strideA = blockSize;

                        for (int y = 0; y < blockSize; y++)
                            for (int x = 0; x < blockSize; x++)
                                blockTempA[x + blockSize * y] = ((rowA + y) < AM && (l + x < AN)) ? blockA[x + AN * y] : 0.0f;

                        blockA = blockTempA;
                    }

                    if (colB + blockSize > BN || l + blockSize > BM) // copy remainder of B into zero-padded block
                    {
                        if (blockTempB == null)
                            blockTempB = AllocBlock(blockSize, blockSize);
                        strideB = blockSize;

                        for (int y = 0; y < blockSize; y++)
                            for (int x = 0; x < blockSize; x++)
                                blockTempB[x + blockSize * y] = ((colB + x) < BN && (l + y < BM)) ? blockB[x + BN * y] : 0.0f;

                        blockB = blockTempB;
                    }

                    MultiplyBlockUnrollHx16(blockA, strideA, blockB, strideB, blockS, strideS);
                }

                if (blockS == blockTempS) // copy back
                {
                    for (int y = 0; y < blockSize; y++)
                        for (int x = 0; x < blockSize; x++)
                        {
                            if (((rowA + y) < SM) && ((colB + x) < SN))
                                S[(rowA + y) * SN + (colB + x)] = blockTempS[x + blockSize * y];
                        }
                }

                FreeBlock(blockTempA);
                FreeBlock(blockTempB);
                FreeBlock(blockTempS);
            }
        }

        static void MultiplyBlockUnrollHx16(float* Ap, int Astride, float* Bp, int Bstride, float* Sp, int Sstride)
        {
            for (int i = 0; i < blockSize; i++)
            {
                float sum0 = *(Sp + i * Sstride + 0);
                float sum1 = *(Sp + i * Sstride + 1);
                float sum2 = *(Sp + i * Sstride + 2);
                float sum3 = *(Sp + i * Sstride + 3);
                float sum4 = *(Sp + i * Sstride + 4);
                float sum5 = *(Sp + i * Sstride + 5);
                float sum6 = *(Sp + i * Sstride + 6);
                float sum7 = *(Sp + i * Sstride + 7);
                float sum8 = *(Sp + i * Sstride + 8);
                float sum9 = *(Sp + i * Sstride + 9);
                float sumA = *(Sp + i * Sstride + 10);
                float sumB = *(Sp + i * Sstride + 11);
                float sumC = *(Sp + i * Sstride + 12);
                float sumD = *(Sp + i * Sstride + 13);
                float sumE = *(Sp + i * Sstride + 14);
                float sumF = *(Sp + i * Sstride + 15);

                for (int l = 0; l < blockSize; l++)
                {
                    float A = *(Ap + i * Astride + l);

                    float B0 = *(Bp + l * Bstride + 0);
                    float B1 = *(Bp + l * Bstride + 1);
                    float B2 = *(Bp + l * Bstride + 2);
                    float B3 = *(Bp + l * Bstride + 3);
                    float B4 = *(Bp + l * Bstride + 4);
                    float B5 = *(Bp + l * Bstride + 5);
                    float B6 = *(Bp + l * Bstride + 6);
                    float B7 = *(Bp + l * Bstride + 7);
                    float B8 = *(Bp + l * Bstride + 8);
                    float B9 = *(Bp + l * Bstride + 9);
                    float BA = *(Bp + l * Bstride + 10);
                    float BB = *(Bp + l * Bstride + 11);
                    float BC = *(Bp + l * Bstride + 12);
                    float BD = *(Bp + l * Bstride + 13);
                    float BE = *(Bp + l * Bstride + 14);
                    float BF = *(Bp + l * Bstride + 15);


                    sum0 += A * B0;
                    sum1 += A * B1;
                    sum2 += A * B2;
                    sum3 += A * B3;
                    sum4 += A * B4;
                    sum5 += A * B5;
                    sum6 += A * B6;
                    sum7 += A * B7;
                    sum8 += A * B8;
                    sum9 += A * B9;
                    sumA += A * BA;
                    sumB += A * BB;
                    sumC += A * BC;
                    sumD += A * BD;
                    sumE += A * BE;
                    sumF += A * BF;
                }

                *(Sp + i * Sstride + 0 ) = sum0;
                *(Sp + i * Sstride + 1 ) = sum1;
                *(Sp + i * Sstride + 2 ) = sum2;
                *(Sp + i * Sstride + 3 ) = sum3;
                *(Sp + i * Sstride + 4 ) = sum4;
                *(Sp + i * Sstride + 5 ) = sum5;
                *(Sp + i * Sstride + 6 ) = sum6;
                *(Sp + i * Sstride + 7 ) = sum7;
                *(Sp + i * Sstride + 8 ) = sum8;
                *(Sp + i * Sstride + 9 ) = sum9;
                *(Sp + i * Sstride + 10) = sumA;
                *(Sp + i * Sstride + 11) = sumB;
                *(Sp + i * Sstride + 12) = sumC;
                *(Sp + i * Sstride + 13) = sumD;
                *(Sp + i * Sstride + 14) = sumE;
                *(Sp + i * Sstride + 15) = sumF;
            }
        }
    }
}

} // namespace Barracuda
