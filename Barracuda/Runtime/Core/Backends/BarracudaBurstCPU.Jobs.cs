using UnityEngine;
using System;
using System.Threading;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Burst;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;
using Unity.Mathematics;

namespace Unity.Barracuda {

// BarracudaBurstCPU.Core.cs -- definition of class BurstCPUOps, Pin(), BurstTensorData
// BarracudaBurstCPU.Ops.cs  -- impl. IOps, job schedulers
// BarracudaBurstCPU.Jobs.cs -- impl. jobs

public partial class BurstCPUOps
{
    internal static readonly Thread MainThread = Thread.CurrentThread;

    #region Job resources declaration

    [BurstCompile]
    internal unsafe struct ReadOnlyMemResource
    {
        [NoAlias][NativeDisableUnsafePtrRestriction][ReadOnly] public float* ptr;
        internal ReadOnlyMemResource(float* _ptr) => ptr = _ptr;
    }

    [BurstCompile]
    internal unsafe struct ReadWriteMemResource
    {
        [NoAlias][NativeDisableUnsafePtrRestriction] public float* ptr;
        internal ReadWriteMemResource(float* _ptr) => ptr = _ptr;
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

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct MatrixMultiplyJob : IJobParallelFor
    {
        [NoAlias][NativeDisableUnsafePtrRestriction][ReadOnly] public unsafe float* A;
        public int AN, AM;
        [NoAlias][NativeDisableUnsafePtrRestriction][ReadOnly] public unsafe float* B;
        public int BN, BM;
        [NoAlias][NativeDisableUnsafePtrRestriction]           public unsafe float* C;
        public int CN, CM;
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
                int tmp = AN; AN = AM; AM = tmp;
            }
            if (transposeB)
            {
                int tmp = BN; BN = BM; BM = tmp;
            }

            int n = math.max(AN, BM);
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
                for (int j = 0; j < Math.Min(AN, BM); j += bs)
                {
                    int rowA = (AN > BM) ? i * bs: j;
                    int colB = (AN > BM) ? j     : i * bs;

                    float* blockC = C + rowA * CM + colB;
                    int strideC = CM;

                    if (rowA + bs > CN || colB + bs > CM) // copy remainder of C into zero-padded block
                    {
                        if (blockTempC == null)
                            blockTempC = AllocBlock();
                        blockC = blockTempC;
                        strideC = bs;
                        MatrixUtils.CopyBlockWithPadding(C, rowA, CN, colB, CM, blockC, bs);
                    }

                    for (int l = 0; l < AM; l += bs) // inner-loop
                    {
                        float* blockA = A + rowA * AM +    l;
                        float* blockB = B +    l * BM + colB;
                        int strideA = AM;
                        int strideB = BM;

                        if (rowA + bs > AN || l + bs > AM || transposeA) // copy remainder of A or transposed A into zero-padded block
                        {
                            if (blockTempA == null)
                                blockTempA = AllocBlock();
                            blockA = blockTempA;
                            strideA = bs;
                            MatrixUtils.CopyBlockWithPadding(A, rowA, AN,    l, AM, blockA, bs, transposeA);
                        }

                        if (colB + bs > BM || l + bs > BN || transposeB) // copy remainder of A or transposed A into zero-padded block
                        {
                            if (blockTempB == null)
                                blockTempB = AllocBlock();
                            blockB = blockTempB;
                            strideB = bs;
                            MatrixUtils.CopyBlockWithPadding(B,    l, BN, colB, BM, blockB, bs, transposeB);
                        }

                        MultiplyBlockUnroll16xh(blockA, strideA, blockB, strideB, blockC, strideC);
                    }

                    if (blockC == blockTempC) // copy back
                        MatrixUtils.CopyBlockWithPadding(blockC, C, rowA, CN, colB, CM, bs);
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

        static unsafe void MultiplyBlockUnroll16xh(float* Ap, int Astride, float* Bp, int Bstride, float* Cp, int Cstride)
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

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct MatrixMultiply3x2Job : IJobParallelFor
    {
        [NoAlias][NativeDisableUnsafePtrRestriction][ReadOnly] public unsafe float* A;
        public int AB, AN, AM;
        [NoAlias][NativeDisableUnsafePtrRestriction][ReadOnly] public unsafe float* B;
        public int BN, BM;
        [NoAlias][NativeDisableUnsafePtrRestriction]           public unsafe float* C;
        public int CN, CM;

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

            int batchOffSetA = (batch * AN * AM);
            int batchOffSetC = (batch * CN * CM);

            int rowA = i * blockSize;
            int colB = j * blockSize;

            unsafe
            {
                float* blockTempA = null;
                float* blockTempB = null;
                float* blockTempC = null;

                float* blockC = C + rowA + CN * colB + batchOffSetC;
                int strideC = CN;

                if (rowA + blockSize > CN || colB + blockSize > CM) // copy remainder of C into zero-padded block
                {
                    blockTempC = AllocBlock();
                    strideC = blockSize;
                    blockC = blockTempC;
                }
                for (int y = 0; y < blockSize; y++)
                    for (int x = 0; x < blockSize; x++)
                        blockC[x + strideC * y] = 0.0f;

                for (int l = 0; l < AM; l += blockSize) // inner-loop
                {
                    float* blockA = A + rowA + AN * l + batchOffSetA;
                    float* blockB = B + l * BM + colB;
                    int strideA = AN;
                    int strideB = BM;

                    if (rowA + blockSize > AN || l + blockSize > AM) // copy remainder of A into zero-padded block
                    {
                        if (blockTempA == null)
                            blockTempA = AllocBlock();
                        strideA = blockSize;

                        for (int y = 0; y < blockSize; y++)
                            for (int x = 0; x < blockSize; x++)
                                blockTempA[x + blockSize * y] = ((rowA + x) < AN && (l + y < AM)) ? blockA[x + AN * y] : 0.0f;

                        blockA = blockTempA;
                    }

                    if (colB + blockSize > BM || l + blockSize > BN) // copy remainder of B into zero-padded block
                    {
                        if (blockTempB == null)
                            blockTempB = AllocBlock();
                        strideB = blockSize;

                        for (int y = 0; y < blockSize; y++)
                            for (int x = 0; x < blockSize; x++)
                                blockTempB[x + blockSize * y] = ((colB + x) < BM && (l + y < BN)) ? blockB[x + BM * y] : 0.0f;

                        blockB = blockTempB;
                    }

                    MultiplyBlockUnroll16xh(blockA, strideA, blockB, strideB, blockC, strideC);
                }

                if (blockC == blockTempC) // copy back
                {
                    for (int y = 0; y < blockSize; y++)
                        for (int x = 0; x < blockSize; x++)
                        {
                            if (((rowA + x) < CN) && ((colB + y) < CM))
                                C[(rowA + x) + CN * (colB + y) + batchOffSetC] = blockTempC[x + blockSize * y];
                        }
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

        static unsafe void MultiplyBlockUnroll16xh(float* Ap, int Astride, float* Bp, int Bstride, float* Cp, int Cstride)
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


    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct MatrixMultiply4x4Job : IJobParallelFor
    {
        [NoAlias][NativeDisableUnsafePtrRestriction][ReadOnly] public unsafe float* A;
        public int AB0, AB1, AN, AM;
        [NoAlias][NativeDisableUnsafePtrRestriction][ReadOnly] public unsafe float* B;
        public int BB0, BB1, BN, BM;
        [NoAlias][NativeDisableUnsafePtrRestriction]           public unsafe float* C;
        public int CB0, CB1, CN, CM;

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

            int batch1 = (threadID % CB1);
            int batch0 = (threadID / CB1) / dispatchThreadXY;
            int i = ((threadID / CB1) % dispatchThreadXY) % dispatchThreadX;
            int j = ((threadID / CB1) % dispatchThreadXY) / dispatchThreadX;

            int batchOffSetA = ((batch0 % AB0) * AN * AM * AB1 + (batch1 % AB1));
            int batchOffSetB = ((batch0 % BB0) * BN * BM * BB1 + (batch1 % BB1));
            int batchOffSetC = (batch0 * CN * CM * CB1 + batch1);

            int rowA = i * blockSize;
            int colB = j * blockSize;

            unsafe
            {
                float* blockTempA = null;
                float* blockTempB = null;
                float* blockTempC = null;

                float* blockC = C + (rowA * CM + colB)*CB1 + batchOffSetC;
                int strideC = CM;
                int strideBatchC = CB1;

                if (rowA + blockSize > CN || colB + blockSize > CM) // copy remainder of A into zero-padded block
                {
                    blockTempC = AllocBlock();
                    strideC = blockSize;
                    strideBatchC = 1;
                    blockC = blockTempC;
                }
                for (int y = 0; y < blockSize; y++)
                    for (int x = 0; x < blockSize; x++)
                        blockC[(x + strideC * y) * strideBatchC] = 0.0f;

                for (int l = 0; l < AM; l += blockSize) // inner-loop
                {
                    float* blockA = A + (rowA * AM + l)*AB1 + batchOffSetA;
                    float* blockB = B + (l * BM + colB)*BB1 + batchOffSetB;
                    int strideA = AM;
                    int strideBatchA = AB1;
                    int strideB = BM;
                    int strideBatchB = BB1;

                    if (rowA + blockSize > AN || l + blockSize > AM) // copy remainder of A into zero-padded block
                    {
                        if (blockTempA == null)
                            blockTempA = AllocBlock();
                        strideA = blockSize;
                        strideBatchA = 1;

                        for (int y = 0; y < blockSize; y++)
                            for (int x = 0; x < blockSize; x++)
                                blockTempA[x + blockSize * y] = ((rowA + y) < AN && (l + x < AM)) ? blockA[(x + AM * y)*AB1] : 0.0f;

                        blockA = blockTempA;
                    }

                    if (colB + blockSize > BM || l + blockSize > BN) // copy remainder of A into zero-padded block
                    {
                        if (blockTempB == null)
                            blockTempB = AllocBlock();
                        strideB = blockSize;
                        strideBatchB = 1;

                        for (int y = 0; y < blockSize; y++)
                            for (int x = 0; x < blockSize; x++)
                                blockTempB[x + blockSize * y] = ((colB + x) < BM && (l + y < BN)) ? blockB[(x + BM * y)*BB1] : 0.0f;

                        blockB = blockTempB;
                    }

                    MultiplyBlockUnroll16xh(blockA, strideA, strideBatchA, blockB, strideB, strideBatchB, blockC, strideC, strideBatchC);
                }

                if (blockC == blockTempC) // copy back
                {
                    for (int y = 0; y < blockSize; y++)
                    for (int x = 0; x < blockSize; x++)
                    {
                        if (((rowA + y) < CN) && (colB + x < CM))
                            C[((rowA + y) * CM + (colB + x)) * CB1 + batchOffSetC] = blockTempC[x + blockSize * y];
                    }
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

        static unsafe void MultiplyBlockUnroll16xh(float* Ap, int Astride, int ABatchStride, float* Bp, int Bstride, int BBatchStride, float* Cp, int Cstride, int CBatchStride)
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

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct Dense3Job : IJobParallelFor
    {
        [NoAlias][NativeDisableUnsafePtrRestriction][ReadOnly] public unsafe float* A;
        public int AB, AN, AM;
        [NoAlias][NativeDisableUnsafePtrRestriction][ReadOnly] public unsafe float* B;
        public int BN, BM;
        [NoAlias][NativeDisableUnsafePtrRestriction][ReadOnly] public unsafe float* C;
        [NoAlias][NativeDisableUnsafePtrRestriction]           public unsafe float* S;
        public int SN, SM;

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

            int batchOffSetA = (batch * AN * AM);
            int batchOffSetS = (batch * SN * SM);

            int rowA = i * blockSize;
            int colB = j * blockSize;

            unsafe
            {
                float* blockTempA = null;
                float* blockTempB = null;
                float* blockTempS = null;

                float* blockS = S + rowA + SN * colB + batchOffSetS;
                int strideS = SN;

                if (rowA + blockSize > SN || colB + blockSize > SM) // copy remainder of C into zero-padded block
                {
                    blockTempS = AllocBlock();
                    strideS = blockSize;
                    blockS = blockTempS;
                }
                for (int y = 0; y < blockSize; y++)
                    for (int x = 0; x < blockSize; x++)
                        blockS[x + strideS * y] = (colB + y) < BM ? C[colB + y] : 0.0f;

                for (int l = 0; l < AM; l += blockSize) // inner-loop
                {
                    float* blockA = A + rowA + AN * l + batchOffSetA;
                    float* blockB = B + l * BM + colB;
                    int strideA = AN;
                    int strideB = BM;

                    if (rowA + blockSize > AN || l + blockSize > AM) // copy remainder of A into zero-padded block
                    {
                        if (blockTempA == null)
                            blockTempA = AllocBlock();
                        strideA = blockSize;

                        for (int y = 0; y < blockSize; y++)
                            for (int x = 0; x < blockSize; x++)
                                blockTempA[x + blockSize * y] = ((rowA + x) < AN && (l + y < AM)) ? blockA[x + AN * y] : 0.0f;

                        blockA = blockTempA;
                    }

                    if (colB + blockSize > BM || l + blockSize > BN) // copy remainder of B into zero-padded block
                    {
                        if (blockTempB == null)
                            blockTempB = AllocBlock();
                        strideB = blockSize;

                        for (int y = 0; y < blockSize; y++)
                            for (int x = 0; x < blockSize; x++)
                                blockTempB[x + blockSize * y] = ((colB + x) < BM && (l + y < BN)) ? blockB[x + BM * y] : 0.0f;

                        blockB = blockTempB;
                    }

                    MultiplyBlockUnroll16xh(blockA, strideA, blockB, strideB, blockS, strideS);
                }

                if (blockS == blockTempS) // copy back
                {
                    for (int y = 0; y < blockSize; y++)
                        for (int x = 0; x < blockSize; x++)
                        {
                            if (((rowA + x) < SN) && ((colB + y) < SM))
                                S[(rowA + x) + SN * (colB + y) + batchOffSetS] = blockTempS[x + blockSize * y];
                        }
                }

                FreeBlock(blockTempA);
                FreeBlock(blockTempB);
                FreeBlock(blockTempS);
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

        static unsafe void MultiplyBlockUnroll16xh(float* Ap, int Astride, float* Bp, int Bstride, float* Sp, int Sstride)
        {
            for (int i = 0; i < blockSize; i++)
            {
                float sum0 = *(Sp + i + Sstride * 0);
                float sum1 = *(Sp + i + Sstride * 1);
                float sum2 = *(Sp + i + Sstride * 2);
                float sum3 = *(Sp + i + Sstride * 3);
                float sum4 = *(Sp + i + Sstride * 4);
                float sum5 = *(Sp + i + Sstride * 5);
                float sum6 = *(Sp + i + Sstride * 6);
                float sum7 = *(Sp + i + Sstride * 7);
                float sum8 = *(Sp + i + Sstride * 8);
                float sum9 = *(Sp + i + Sstride * 9);
                float sumA = *(Sp + i + Sstride * 10);
                float sumB = *(Sp + i + Sstride * 11);
                float sumC = *(Sp + i + Sstride * 12);
                float sumD = *(Sp + i + Sstride * 13);
                float sumE = *(Sp + i + Sstride * 14);
                float sumF = *(Sp + i + Sstride * 15);

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

                *(Sp + i + Sstride * 0 ) = sum0;
                *(Sp + i + Sstride * 1 ) = sum1;
                *(Sp + i + Sstride * 2 ) = sum2;
                *(Sp + i + Sstride * 3 ) = sum3;
                *(Sp + i + Sstride * 4 ) = sum4;
                *(Sp + i + Sstride * 5 ) = sum5;
                *(Sp + i + Sstride * 6 ) = sum6;
                *(Sp + i + Sstride * 7 ) = sum7;
                *(Sp + i + Sstride * 8 ) = sum8;
                *(Sp + i + Sstride * 9 ) = sum9;
                *(Sp + i + Sstride * 10) = sumA;
                *(Sp + i + Sstride * 11) = sumB;
                *(Sp + i + Sstride * 12) = sumC;
                *(Sp + i + Sstride * 13) = sumD;
                *(Sp + i + Sstride * 14) = sumE;
                *(Sp + i + Sstride * 15) = sumF;
            }
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
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
                float* from = X.ptr + n *  inStrideN + readY *  inStrideH + skipFromInputRow * inStrideW;
                float* to   = O.ptr + n * outStrideN +     y * outStrideH;

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

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct MaxPool2DJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        [ReadOnly] public int strideX, strideY, padX, padY;
        [ReadOnly] public int kernelHeight, kernelWidth;
        [ReadOnly] public int inHeight, inWidth, inChannels,    inStrideN,  inStrideH,     inStrideW;
        [ReadOnly] public int outBatch, outWidth,               outStrideN, outStrideH,    outStrideW;
        const int unrollSize = 16;
        public void Execute(int y)
        {
            int accumulatorMemSize = inChannels * sizeof(float);
            float* outputAccumulators = (float*)UnsafeUtility.Malloc(accumulatorMemSize, JobsUtility.CacheLineSize, Allocator.TempJob);
            for (int n = 0; n < outBatch; ++n)
            for (int x = 0; x < outWidth; ++x)
            {
                bool firstNotRejectedPixelInKernel = true;
                // gather max results in accumulators
                for (int dy = 0; dy < kernelHeight; ++dy)
                {
                    int readY = y * strideY + dy - padY;
                    if (readY < 0) continue;
                    if (readY >= inHeight) continue;

                    for (int dx = 0; dx < kernelWidth; ++dx)
                    {
                        int readX = x * strideX + dx - padY;
                        if (readX < 0) continue;
                        if (readX >= inWidth) continue;

                        float* dst    = outputAccumulators;
                        float* src    = X.ptr + n * inStrideN + readY * inStrideH     + readX * inStrideW;

                        int k = 0;
                        if (firstNotRejectedPixelInKernel) // first pass, write-through
                        {
                            for (; k < inChannels - unrollSize + 1; k += unrollSize) // unroll of inChannels loop
                                for (int q = 0; q < unrollSize; q++, src++, dst++)
                                    *dst = *src;
                            for (; k < inChannels; k++, src++, dst++) // remainder of inChannels loop
                                *dst = *src;
                        }
                        else
                        {
                            for (; k < inChannels - unrollSize + 1; k += unrollSize) // unroll of inChannels loop
                                for (int q = 0; q < unrollSize; q++, src++, dst++)
                                    *dst = (*dst) > (*src) ? (*dst) : (*src);
                            for (; k < inChannels; k++, src++, dst++) // remainder of inChannels loop
                                *dst = (*dst) > (*src) ? (*dst) : (*src);
                        }
                        firstNotRejectedPixelInKernel = false;
                    }
                }

                // safety net, if kernel was completely outside of X
                // fill with padding_value (0) to avoid uninitialized memory
                if (firstNotRejectedPixelInKernel)
                    UnsafeUtility.MemClear(outputAccumulators, accumulatorMemSize);

                { // write accumulators to memory
                    int k = 0;
                    float* src  = outputAccumulators;
                    float* dst  = O.ptr + n * outStrideN + y * outStrideH + x * outStrideW;
                    for (; k < inChannels - unrollSize + 1; k += unrollSize)  // unroll of inChannels loop
                        for (int q = 0; q < unrollSize; q++, src++, dst++)
                            *dst = *src;
                    for (; k < inChannels; k++, src++, dst++) // remainder of inChannels loop
                        *dst = *src;
                }
            }

            UnsafeUtility.Free(outputAccumulators, Allocator.TempJob);
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct AvgPool2DJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        [ReadOnly] public int strideX, strideY, padX, padY;
        [ReadOnly] public int kernelHeight, kernelWidth;
        [ReadOnly] public int inHeight, inWidth, inChannels,    inStrideN,  inStrideH,     inStrideW;
        [ReadOnly] public int outBatch, outWidth,               outStrideN, outStrideH,    outStrideW;

        const int unrollSize = 16;
        public void Execute(int y)
        {
            int accumulatorMemSize = inChannels * sizeof(float);
            float* outputAccumulators = (float*)UnsafeUtility.Malloc(accumulatorMemSize, JobsUtility.CacheLineSize, Allocator.TempJob);

            for (int n = 0; n < outBatch; ++n)
            for (int x = 0; x < outWidth; ++x)
            {
                // reset accumulators & counter
                int counter = 0;
                UnsafeUtility.MemClear(outputAccumulators, accumulatorMemSize);

                // gather sums in accumulators
                for (int dy = 0; dy < kernelHeight; ++dy)
                {
                    int readY = y * strideY + dy - padY;
                    if (readY < 0) continue;
                    if (readY >= inHeight) continue;

                    for (int dx = 0; dx < kernelWidth; ++dx)
                    {
                        int readX = x * strideX + dx - padY;
                        if (readX < 0) continue;
                        if (readX >= inWidth) continue;

                        float* dst    = outputAccumulators;
                        float* src    = X.ptr + n * inStrideN + readY * inStrideH     + readX * inStrideW;

                        int k = 0;
                        for (; k < inChannels - unrollSize + 1; k += unrollSize) // unroll of inChannels loop
                            for (int q = 0; q < unrollSize; q++, src++, dst++)
                                *dst += *src;
                        for (; k < inChannels; k++, src++, dst++) // remainder of inChannels loop
                            *dst += *src;
                        counter++;
                    }
                }

                // safety net, if kernel was completely outside of X
                counter = math.max(1, counter);

                { // write accumulators to memory
                    int k = 0;
                    float invCounter = 1f / (float)counter;
                    float* src  = outputAccumulators;
                    float* dst  = O.ptr + n * outStrideN + y * outStrideH + x * outStrideW;
                    for (; k < inChannels - unrollSize + 1; k += unrollSize)  // unroll of inChannels loop
                        for (int q = 0; q < unrollSize; q++, src++, dst++)
                            *dst = *src * invCounter;
                    for (; k < inChannels; k++, src++, dst++) // remainder of inChannels loop
                        *dst = *src * invCounter;
                }
            }

            UnsafeUtility.Free(outputAccumulators, Allocator.TempJob);
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct DepthwiseConv2DJob : IJobParallelFor, IJobResourceDeclarationXSBO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadOnlyMemResource S { get; set; }
        public ReadOnlyMemResource B { get; set; }
        public ReadWriteMemResource O { get; set; }
        [ReadOnly] public int strideX, strideY, padX, padY;
        [ReadOnly] public int inHeight, inWidth, inChannels,           inStrideN,  inStrideH,     inStrideW;
        [ReadOnly] public int kernelCount, kernelHeight, kernelWidth,          kernelStrideH, kernelStrideW;
        [ReadOnly] public int outBatch, outWidth,                     outStrideN, outStrideH,    outStrideW;
        const int unrollSize = 16;
        public void Execute(int y)
        {
            int accumulatorMemSize = kernelCount * sizeof(float);
            float* outputAccumulators = (float*)UnsafeUtility.Malloc(accumulatorMemSize, JobsUtility.CacheLineSize, Allocator.TempJob);
            for (int n = 0; n < outBatch; ++n)
            for (int x = 0; x < outWidth; ++x)
            {
                // reset accumulators to 0
                UnsafeUtility.MemClear(outputAccumulators, accumulatorMemSize);

                // gather X * K results in accumulators
                for (int dy = 0; dy < kernelHeight; ++dy)
                {
                    int readY = y * strideY + dy - padY;
                    if (readY < 0) continue;
                    if (readY >= inHeight) continue;

                    for (int dx = 0; dx < kernelWidth; ++dx)
                    {
                        int readX = x * strideX + dx - padY;
                        if (readX < 0) continue;
                        if (readX >= inWidth) continue;

                        float* dst    = outputAccumulators;
                        float* src    = X.ptr + n * inStrideN + readY * inStrideH     + readX * inStrideW;
                        float* kernel = S.ptr                 +    dy * kernelStrideH +    dx * kernelStrideW;

                        int k = 0;
                        for (; k < kernelCount - unrollSize + 1; k += unrollSize) // unroll of kernelCount loop
                            for (int q = 0; q < unrollSize; q++, src++, dst++, kernel++)
                                *dst += (*src) * (*kernel);
                        for (; k < kernelCount; k++, src++, dst++, kernel++) // remainder of kernelCount loop
                            *dst += (*src) * (*kernel);
                    }
                }

                { // write accumulators to memory and add bias
                    int k = 0;
                    float* src  = outputAccumulators;
                    float* dst  = O.ptr + n * outStrideN + y * outStrideH + x * outStrideW;
                    float* bias = B.ptr;
                    for (; k < kernelCount - unrollSize + 1; k += unrollSize)  // unroll of kernelCount loop
                        for (int q = 0; q < unrollSize; q++, src++, dst++, bias++)
                            *dst = (*src) + (*bias);
                    for (; k < kernelCount; k++, src++, dst++, bias++) // remainder of kernelCount loop
                        *dst = (*src) + (*bias);
                }
            }

            UnsafeUtility.Free(outputAccumulators, Allocator.TempJob);
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct PReluJob : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadOnlyMemResource B { get; set; }
        public ReadWriteMemResource O { get; set; }
        [ReadOnly] public int inOutChannels;
        [ReadOnly] public int isGammaAVector;//1 if true, 0 if false

        const int unrollSize = 32;
        public void Execute(int i)
        {
            float* src   = X.ptr + i * inOutChannels;
            float* dst   = O.ptr + i * inOutChannels;
            float* gamma = B.ptr + i * inOutChannels * isGammaAVector;

            int j = 0;
            for (; j < inOutChannels - unrollSize + 1; j += unrollSize) // unroll of inOutChannels loop
                for (int q = 0; q < unrollSize; q++, src++, dst++, gamma+=isGammaAVector)
                    *dst = PRelu(*src, *gamma);
            for (; j < inOutChannels; j++, src++, dst++, gamma+=isGammaAVector) // remainder of inOutChannels loop
                *dst = PRelu(*src, *gamma);

        }

        public static float PRelu(float v, float gamma)
        {
            // from Theano impl
            // https://github.com/Theano/theano/blob/d395439aec5a6ddde8ef5c266fd976412a5c5695/theano/tensor/nnet/nnet.py#L2209-L2251
            // @TODO: precompute f1 and f2 for all S before this job
            float f1 = 0.5f * (1f + gamma);
            float f2 = 0.5f * (1f - gamma);
            // NOTE: burst-1.2.3 has troubles with Math.Min/Max generating poorly vectorized and branch code
            // Instead Math.Abs based code is used instead. (Math.Abs just flips 1 bit)
            return f1 * v + f2 * math.abs(v);
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct ReluJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        public void Execute(int i)
        {
            float v = X.ptr[i];
            // NOTE: burst-1.2.3 has troubles with Math.Min/Max generating poorly vectorized and branch code
            // Instead Math.Abs based code is used instead. (Math.Abs just flips 1 bit)
            O.ptr[i] = 0.5f * (v + math.abs(v));
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct Relu6Job : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        public void Execute(int i)
        {
            // f(x) = min(max(x, 0), 6)
            // "Convolutional Deep Belief Networks on CIFAR-10", A Krizhevsky, 2010
            // http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf
            float v = X.ptr[i];

            // NOTE: burst-1.2.3 has troubles with Math.Min/Max generating poorly vectorized and branch code
            // Instead Math.Abs based code is used instead. (Math.Abs just flips 1 bit)
            O.ptr[i] = 0.5f * (-math.abs(v - 6f) + math.abs(v) + 6f);
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct LeakyReluJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        // from Theano impl
        // https://github.com/Theano/theano/blob/d395439aec5a6ddde8ef5c266fd976412a5c5695/theano/tensor/nnet/nnet.py#L2209-L2251
        [ReadOnly] float f1, f2, alpha_;
        public float alpha { get { return alpha_; } set {
            alpha_ = value;
            f1 = 0.5f * (1f + alpha_);
            f2 = 0.5f * (1f - alpha_);
        } }
        public void Execute(int i)
        {
            float v = X.ptr[i];
            // NOTE: burst-1.2.3 has troubles with Math.Min/Max generating poorly vectorized and branch code
            // Instead Math.Abs based code is used instead. (Math.Abs just flips 1 bit)
            O.ptr[i] = f1 * v + f2 * math.abs(v);
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct TanhJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        public void Execute(int i)
        {
            O.ptr[i] = math.tanh(X.ptr[i]);
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct SoftplusJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        public void Execute(int i)
        {
            O.ptr[i] = math.log(math.exp(X.ptr[i]) + 1f);
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct SigmoidJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        public void Execute(int i)
        {
            O.ptr[i] = 1f / (1f + math.exp(-X.ptr[i]));
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct EluJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        [ReadOnly] public float alpha;
        public void Execute(int i)
        {
            // f(x) = alpha * (exp(x) - 1.) for x < 0, f(x) = x for x >= 0
            // "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)", DA Clevert, 2015
            // https://arxiv.org/abs/1511.07289
            float v = X.ptr[i];
            if (v <= 0)
                v = alpha * (math.exp(v) - 1f);
            O.ptr[i] = v;
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct SeluJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        [ReadOnly] public float alpha, gamma;
        public void Execute(int i)
        {
            // f(x) = gamma * (alpha * e^x - alpha) for x <= 0, f(x) = gamma * x for x > 0
            float v = X.ptr[i];
            if (v <= 0)
                v = gamma * (alpha * math.exp(v) - alpha);
            else
                v = gamma * v;
            O.ptr[i] = v;
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct SwishJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        public void Execute(int i)
        {
            // f(x) = sigmoid(x) * x = x / (1 + exp(-x))
            // "Searching for Activation Functions". P Ramachandran, 2017
            // https://arxiv.org/abs/1710.05941
            float v = X.ptr[i];
            v = v / (1f + math.exp(-v));
            O.ptr[i] = v;
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct ExpBiasReduceJob : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadOnlyMemResource B { get; set; }
        public ReadWriteMemResource O { get; set; }
        [ReadOnly] public int inChannels;
        public void Execute(int y)
        {
            float accum = 0f;
            for (int i = 0; i < inChannels; ++i)
                accum += math.exp(X.ptr[y * inChannels + i] - B.ptr[y]);
            O.ptr[y] = accum;
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct SoftmaxEndJob : IJobParallelFor, IJobResourceDeclarationXSBO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadOnlyMemResource S { get; set; }
        public ReadOnlyMemResource B { get; set; }
        public ReadWriteMemResource O { get; set; }
        [ReadOnly] public int inChannels;
        public void Execute(int i)
        {
            int n = (i / inChannels);
            O.ptr[i] = math.exp(X.ptr[i] - B.ptr[n]) / S.ptr[n];
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct LogSoftmaxEndJob : IJobParallelFor, IJobResourceDeclarationXSBO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadOnlyMemResource S { get; set; }
        public ReadOnlyMemResource B { get; set; }
        public ReadWriteMemResource O { get; set; }
        [ReadOnly] public int inChannels;
        public void Execute(int i)
        {
            int n = (i / inChannels);
            O.ptr[i] = (X.ptr[i] - B.ptr[n]) - math.log(S.ptr[n]);
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct AbsJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        public void Execute(int i)
        {
            O.ptr[i] = Math.Abs(X.ptr[i]);
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct NegJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        public void Execute(int i)
        {
            O.ptr[i] = -X.ptr[i];
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct CeilJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        public void Execute(int i)
        {
            O.ptr[i] = math.ceil(X.ptr[i]);
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct ClipJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        [ReadOnly] public float min, max;
        public void Execute(int i)
        {
            O.ptr[i] = math.clamp(X.ptr[i], min, max);
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct FloorJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        public void Execute(int i)
        {
            O.ptr[i] = math.floor(X.ptr[i]);
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct RoundJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        public void Execute(int i)
        {
            O.ptr[i] = math.round(X.ptr[i]);
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct ReciprocalJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        public void Execute(int i)
        {
            O.ptr[i] = 1.0f / X.ptr[i];
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct PowJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        [ReadOnly] public float alpha;
        public void Execute(int i)
        {
            O.ptr[i] = math.pow(X.ptr[i], alpha);
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct ExpJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        public void Execute(int i)
        {
            O.ptr[i] = math.exp(X.ptr[i]);
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct LogJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        public void Execute(int i)
        {
            O.ptr[i] = math.log(X.ptr[i]);
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct SqrtJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        public void Execute(int i)
        {
            O.ptr[i] = math.sqrt(X.ptr[i]);
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct AcosJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        public void Execute(int i)
        {
            O.ptr[i] = math.acos(X.ptr[i]);
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct AcoshJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        public void Execute(int i)
        {
            float x = X.ptr[i];
            O.ptr[i] = math.log(x + math.sqrt(x*x - 1.0f));
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct AsinJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        public void Execute(int i)
        {
            O.ptr[i] = math.asin(X.ptr[i]);
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct AsinhJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        public void Execute(int i)
        {
            float x = X.ptr[i];
            O.ptr[i] = math.log( x + math.sqrt(x*x + 1.0f));
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct AtanJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        public void Execute(int i)
        {
            O.ptr[i] = math.atan(X.ptr[i]);
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct AtanhJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        public void Execute(int i)
        {
            O.ptr[i] = 0.5f * math.log((1.0f + X.ptr[i])/(1.0f - X.ptr[i]));
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct CosJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        public void Execute(int i)
        {
            O.ptr[i] = math.cos(X.ptr[i]);
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct CoshJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        public void Execute(int i)
        {
            O.ptr[i] = 0.5f * (math.exp(X.ptr[i]) + math.exp(-X.ptr[i]));
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct SinJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        public void Execute(int i)
        {
            O.ptr[i] = math.sin(X.ptr[i]);
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct SinhJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        public void Execute(int i)
        {
            O.ptr[i] = 0.5f * (math.exp(X.ptr[i]) - math.exp(-X.ptr[i]));
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct TanJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        public void Execute(int i)
        {
            O.ptr[i] = math.tan(X.ptr[i]);
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct ElementwiseAddJob : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadOnlyMemResource B { get; set; }
        public ReadWriteMemResource O { get; set; }
        [ReadOnly] public TensorShape shapeO;
        [ReadOnly] public fixed int stridesX[8];
        [ReadOnly] public fixed int stridesY[8];
        [ReadOnly] public float alpha;

        public void Execute(int i)
        {
            int s = 0, r = 0, n = 0, t = 0, d = 0, h = 0, w = 0, c = 0;
            shapeO.GetPositionsFromIndex(i, ref s, ref r, ref n, ref t, ref d, ref h, ref w, ref c);

            float x = X.ptr[stridesX[0] * s + stridesX[1] * r + stridesX[2] * n + stridesX[3] * t + stridesX[4] * d + stridesX[5] * h + stridesX[6] * w + stridesX[7] * c];
            float y = B.ptr[stridesY[0] * s + stridesY[1] * r + stridesY[2] * n + stridesY[3] * t + stridesY[4] * d + stridesY[5] * h + stridesY[6] * w + stridesY[7] * c];

            O.ptr[i] = alpha * y + x;
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct ElementwiseMulJob : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadOnlyMemResource B { get; set; }
        public ReadWriteMemResource O { get; set; }
        [ReadOnly] public TensorShape shapeO;
        [ReadOnly] public fixed int stridesX[8];
        [ReadOnly] public fixed int stridesY[8];

        public void Execute(int i)
        {
            int s = 0, r = 0, n = 0, t = 0, d = 0, h = 0, w = 0, c = 0;
            shapeO.GetPositionsFromIndex(i, ref s, ref r, ref n, ref t, ref d, ref h, ref w, ref c);

            float x = X.ptr[stridesX[0] * s + stridesX[1] * r + stridesX[2] * n + stridesX[3] * t + stridesX[4] * d + stridesX[5] * h + stridesX[6] * w + stridesX[7] * c];
            float y = B.ptr[stridesY[0] * s + stridesY[1] * r + stridesY[2] * n + stridesY[3] * t + stridesY[4] * d + stridesY[5] * h + stridesY[6] * w + stridesY[7] * c];

            O.ptr[i] = x * y;
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct ElementwiseDivJob : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadOnlyMemResource B { get; set; }
        public ReadWriteMemResource O { get; set; }
        [ReadOnly] public TensorShape shapeO;
        [ReadOnly] public fixed int stridesX[8];
        [ReadOnly] public fixed int stridesY[8];

        public void Execute(int i)
        {
            int s = 0, r = 0, n = 0, t = 0, d = 0, h = 0, w = 0, c = 0;
            shapeO.GetPositionsFromIndex(i, ref s, ref r, ref n, ref t, ref d, ref h, ref w, ref c);

            float x = X.ptr[stridesX[0] * s + stridesX[1] * r + stridesX[2] * n + stridesX[3] * t + stridesX[4] * d + stridesX[5] * h + stridesX[6] * w + stridesX[7] * c];
            float y = B.ptr[stridesY[0] * s + stridesY[1] * r + stridesY[2] * n + stridesY[3] * t + stridesY[4] * d + stridesY[5] * h + stridesY[6] * w + stridesY[7] * c];

            O.ptr[i] = x/y;
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct ElementwisePowJob : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadOnlyMemResource B { get; set; }
        public ReadWriteMemResource O { get; set; }
        [ReadOnly] public TensorShape shapeO;
        [ReadOnly] public fixed int stridesX[8];
        [ReadOnly] public fixed int stridesY[8];

        public void Execute(int i)
        {
            int s = 0, r = 0, n = 0, t = 0, d = 0, h = 0, w = 0, c = 0;
            shapeO.GetPositionsFromIndex(i, ref s, ref r, ref n, ref t, ref d, ref h, ref w, ref c);

            float x = X.ptr[stridesX[0] * s + stridesX[1] * r + stridesX[2] * n + stridesX[3] * t + stridesX[4] * d + stridesX[5] * h + stridesX[6] * w + stridesX[7] * c];
            float y = B.ptr[stridesY[0] * s + stridesY[1] * r + stridesY[2] * n + stridesY[3] * t + stridesY[4] * d + stridesY[5] * h + stridesY[6] * w + stridesY[7] * c];

            O.ptr[i] = math.pow(x, y);
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct ElementwiseMaxJob : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadOnlyMemResource B { get; set; }
        public ReadWriteMemResource O { get; set; }
        [ReadOnly] public TensorShape shapeO;
        [ReadOnly] public fixed int stridesX[8];
        [ReadOnly] public fixed int stridesY[8];

        public void Execute(int i)
        {
            int s = 0, r = 0, n = 0, t = 0, d = 0, h = 0, w = 0, c = 0;
            shapeO.GetPositionsFromIndex(i, ref s, ref r, ref n, ref t, ref d, ref h, ref w, ref c);

            float x = X.ptr[stridesX[0] * s + stridesX[1] * r + stridesX[2] * n + stridesX[3] * t + stridesX[4] * d + stridesX[5] * h + stridesX[6] * w + stridesX[7] * c];
            float y = B.ptr[stridesY[0] * s + stridesY[1] * r + stridesY[2] * n + stridesY[3] * t + stridesY[4] * d + stridesY[5] * h + stridesY[6] * w + stridesY[7] * c];

            O.ptr[i] = math.max(x , y);
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct ElementwiseMinJob : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadOnlyMemResource B { get; set; }
        public ReadWriteMemResource O { get; set; }
        [ReadOnly] public TensorShape shapeO;
        [ReadOnly] public fixed int stridesX[8];
        [ReadOnly] public fixed int stridesY[8];

        public void Execute(int i)
        {
            int s = 0, r = 0, n = 0, t = 0, d = 0, h = 0, w = 0, c = 0;
            shapeO.GetPositionsFromIndex(i, ref s, ref r, ref n, ref t, ref d, ref h, ref w, ref c);

            float x = X.ptr[stridesX[0] * s + stridesX[1] * r + stridesX[2] * n + stridesX[3] * t + stridesX[4] * d + stridesX[5] * h + stridesX[6] * w + stridesX[7] * c];
            float y = B.ptr[stridesY[0] * s + stridesY[1] * r + stridesY[2] * n + stridesY[3] * t + stridesY[4] * d + stridesY[5] * h + stridesY[6] * w + stridesY[7] * c];

            O.ptr[i] = math.min(x, y);
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct SetConstantPaddingJob : IJobParallelFor, IJobResourceDeclarationO
    {
        public ReadWriteMemResource O { get; set; }
        [ReadOnly] public float constant;
        public void Execute(int i)
        {
            O.ptr[i] = constant;
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct SetConstantPaddingWithStrideJob : IJobParallelFor, IJobResourceDeclarationO
    {
        public ReadWriteMemResource O { get; set; }
        [ReadOnly] public float constant;
        [ReadOnly] public int length;
        [ReadOnly] public int stride;
        public void Execute(int i)
        {
            int indexStrideIndex = i / length;
            int indexStrideOffset = i % length;
            O.ptr[indexStrideIndex * stride + indexStrideOffset] = constant;
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct ZeroBroadcastJob : IJob, IJobResourceDeclarationO
    {
        public ReadWriteMemResource O { get; set; }
        [ReadOnly] public int repeat;
        public void Execute()
        {
            UnsafeUtility.MemClear(destination: O.ptr, size: repeat * sizeof(float));
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct CopyJob : IJob, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        [ReadOnly] public int length;
        public void Execute()
        {
            UnsafeUtility.MemCpy(destination: O.ptr, source: X.ptr, size: length * sizeof(float));
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct CopyStrideJob : IJob, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        [ReadOnly] public int XStride;
        [ReadOnly] public int OStride;
        [ReadOnly] public int count;
        [ReadOnly] public int length;
        public void Execute()
        {
            UnsafeUtility.MemCpyStride(destination: O.ptr, destinationStride: OStride * sizeof(float), source: X.ptr, sourceStride: XStride * sizeof(float), elementSize: length * sizeof(float), count: count);
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
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

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct GenericSliceJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        [ReadOnly] public TensorShape shapeO;
        [ReadOnly] public TensorShape shapeX;
        [ReadOnly] public int strideS, strideR, strideN, strideT;
        [ReadOnly] public int strideD, strideH, strideW;
        [ReadOnly] public int startS, startR, startN, startT;
        [ReadOnly] public int startD, startH, startW, startC;
        public void Execute(int threadIndex)
        {
            int indexO = threadIndex * shapeO.channels;
            int s = 0, r = 0, n = 0, t = 0;
            int d = 0, h = 0, w = 0, c = 0;
            shapeO.GetPositionsFromIndex(indexO, ref s, ref r, ref n, ref t, ref d, ref h, ref w, ref c);
            s = startS + s * strideS;
            r = startR + r * strideR;
            n = startN + n * strideN;
            t = startT + t * strideT;
            d = startD + d * strideD;
            h = startH + h * strideH;
            w = startW + w * strideW;
            c = startC + c;
            int indexX = shapeX.Index(s, r, n, t, d, h, w, c);
            UnsafeUtility.MemCpy(destination: O.ptr+indexO, source: X.ptr+indexX, size: shapeO.channels * sizeof(float));
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct GenericStridedSliceJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        [ReadOnly] public TensorShape shapeO;
        [ReadOnly] public TensorShape shapeX;
        [ReadOnly] public int strideS, strideR, strideN, strideT;
        [ReadOnly] public int strideD, strideH, strideW, strideC;
        [ReadOnly] public int startS, startR, startN, startT;
        [ReadOnly] public int startD, startH, startW, startC;
        public void Execute(int i)
        {
            int s = 0, r = 0, n = 0, t = 0;
            int d = 0, h = 0, w = 0, c = 0;
            shapeO.GetPositionsFromIndex(i, ref s, ref r, ref n, ref t, ref d, ref h, ref w, ref c);
            s = startS + s * strideS;
            r = startR + r * strideR;
            n = startN + n * strideN;
            t = startT + t * strideT;
            d = startD + d * strideD;
            h = startH + h * strideH;
            w = startW + w * strideW;
            c = startC + c * strideC;
            O.ptr[i] = X.ptr[shapeX.Index(s, r, n, t, d, h, w, c)];
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct ScalarBroadcastAddJob : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadOnlyMemResource B { get; set; }
        public ReadWriteMemResource O { get; set; }
        [ReadOnly] public float alpha;

        public void Execute(int i)
        {
            O.ptr[i] = B.ptr[0] * alpha + X.ptr[i];
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct BroadcastAddJob : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadOnlyMemResource B { get; set; }
        public ReadWriteMemResource O { get; set; }
        [ReadOnly] public float alpha;

        public void Execute(int i)
        {
            O.ptr[i] = B.ptr[i] * alpha + X.ptr[i];
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct ScalarBroadcastMulJob : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadOnlyMemResource B { get; set; }
        public ReadWriteMemResource O { get; set; }
        public void Execute(int i)
        {
            O.ptr[i] = X.ptr[i] * B.ptr[0];
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct BroadcastMulJob : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadOnlyMemResource B { get; set; }
        public ReadWriteMemResource O { get; set; }
        public void Execute(int i)
        {
            O.ptr[i] = X.ptr[i] * B.ptr[i];
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct ScalarBroadcastDivJob : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadOnlyMemResource B { get; set; }
        public ReadWriteMemResource O { get; set; }
        public void Execute(int i)
        {
            O.ptr[i] = X.ptr[i] / B.ptr[0];
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct BroadcastDivJob : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadOnlyMemResource B { get; set; }
        public ReadWriteMemResource O { get; set; }
        public void Execute(int i)
        {
            O.ptr[i] = X.ptr[i] / B.ptr[i];
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct ScalarBroadcastMinJob : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadOnlyMemResource B { get; set; }
        public ReadWriteMemResource O { get; set; }
        public void Execute(int i)
        {
            O.ptr[i] = math.min(X.ptr[i], B.ptr[0]);
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct BroadcastMinJob : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadOnlyMemResource B { get; set; }
        public ReadWriteMemResource O { get; set; }

        public void Execute(int i)
        {
            O.ptr[i] = math.min(X.ptr[i], B.ptr[i]);
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct ScalarBroadcastMaxJob : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadOnlyMemResource B { get; set; }
        public ReadWriteMemResource O { get; set; }
        public void Execute(int i)
        {
            O.ptr[i] = math.max(X.ptr[i], B.ptr[0]);
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct BroadcastMaxJob : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadOnlyMemResource B { get; set; }
        public ReadWriteMemResource O { get; set; }
        public void Execute(int i)
        {
            O.ptr[i] = math.max(X.ptr[i], B.ptr[i]);
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct ScalarBroadcastPowJob : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadOnlyMemResource B { get; set; }
        public ReadWriteMemResource O { get; set; }
        public void Execute(int i)
        {
            O.ptr[i] = math.pow(X.ptr[i], B.ptr[0]);
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct BroadcastPowJob : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadOnlyMemResource B { get; set; }
        public ReadWriteMemResource O { get; set; }

        public void Execute(int i)
        {
            O.ptr[i] = math.pow(X.ptr[i], B.ptr[i]);
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct VectorBroadcastScaleBiasJob : IJobParallelFor, IJobResourceDeclarationXSBO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadOnlyMemResource S { get; set; }
        public ReadOnlyMemResource B { get; set; }
        public ReadWriteMemResource O { get; set; }
        [ReadOnly] public int inOutChannels;
        [ReadOnly] public float alpha;

        const int unrollSize = 32;
        public void Execute(int i)
        {
            float* src   = X.ptr + i * inOutChannels;
            float* dst   = O.ptr + i * inOutChannels;
            float* gamma = S.ptr;
            float* beta  = B.ptr;

            int j = 0;
            for (; j < inOutChannels - unrollSize + 1; j += unrollSize) // unroll of inOutChannels loop
                for (int q = 0; q < unrollSize; q++, src++, dst++, gamma++, beta++)
                    *dst = (*src) * (*gamma) + (*beta) * alpha;
            for (; j < inOutChannels; j++, src++, dst++, gamma++, beta++) // remainder of inOutChannels loop
                *dst = (*src) * (*gamma) + (*beta) * alpha;
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct ReduceMeanJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        [ReadOnly] public int offsetReduce;
        [ReadOnly] public int reduceDim;
        public void Execute(int i)
        {
            int x = i % offsetReduce;
            int y = i / offsetReduce;

            float meanV = 0;
            for (int z = 0; z < reduceDim; ++z)
            {
                float v = X.ptr[y * offsetReduce * reduceDim + z * offsetReduce + x];
                meanV += v;
            }
            O.ptr[y * offsetReduce + x] = meanV / (float)reduceDim;
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct ReduceSumJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        [ReadOnly] public int offsetReduce;
        [ReadOnly] public int reduceDim;
        public void Execute(int i)
        {
            int x = i % offsetReduce;
            int y = i / offsetReduce;

            float meanV = 0;
            for (int z = 0; z < reduceDim; ++z)
            {
                float v = X.ptr[y * offsetReduce * reduceDim + z * offsetReduce + x];
                meanV += v;
            }
            O.ptr[y * offsetReduce + x] = meanV;
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct ReduceMaxJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        [ReadOnly] public int offsetReduce;
        [ReadOnly] public int reduceDim;
        public void Execute(int i)
        {
            int x = i % offsetReduce;
            int y = i / offsetReduce;

            float maxV = float.MinValue;
            for (int z = 0; z < reduceDim; ++z)
            {
                float v = X.ptr[y * offsetReduce * reduceDim + z * offsetReduce + x];
                maxV = math.max(maxV, v);
            }
            O.ptr[y * offsetReduce + x] = maxV;
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct TransposeJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        [ReadOnly] public TensorShape shapeO;
        [ReadOnly] public TensorShape shapeX;
        [ReadOnly] public fixed int permutations[8];
        public void Execute(int i)
        {
            int s = 0, r = 0, n = 0, t = 0, d = 0, h = 0, w = 0, c = 0;
            shapeX.GetPositionsFromIndex(i, ref s, ref r, ref n, ref t, ref d, ref h, ref w, ref c);

            int* index = stackalloc int[8];
            index[0] = s; index[1] = r; index[2] = n; index[3] = t; index[4] = d; index[5] = h; index[6] = w; index[7] = c;

            int indexO = shapeO.Index(index[permutations[0]], index[permutations[1]], index[permutations[2]], index[permutations[3]], index[permutations[4]], index[permutations[5]], index[permutations[6]], index[permutations[7]]);
            O.ptr[indexO] = X.ptr[i];
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct MemFreeJob : IJob
    {
        [NoAlias] [NativeDisableUnsafePtrRestriction]           public float* buffer0;
        [NoAlias] [NativeDisableUnsafePtrRestriction]           public float* buffer1;
                                                     [ReadOnly] public Allocator allocator;
        public void Execute()
        {
            if (buffer0 != null)
                UnsafeUtility.Free(buffer0, allocator);
            if (buffer1 != null)
                UnsafeUtility.Free(buffer1, allocator);
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct TileJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }
        [ReadOnly] public TensorShape shapeO;
        [ReadOnly] public TensorShape shapeX;
        public void Execute(int i)
        {
            int s = 0, r = 0, n = 0, t = 0, d = 0, h = 0, w = 0, c = 0;
            shapeO.GetPositionsFromIndex(i, ref s, ref r, ref n, ref t, ref d, ref h, ref w, ref c);

            s = s % shapeX[0]; r = r % shapeX[1]; n = n % shapeX[2]; t = t % shapeX[3]; d = d % shapeX[4]; h = h % shapeX[5]; w = w % shapeX[6]; c = c % shapeX[7];

            float x = X.ptr[shapeX.Index(s, r, n, t, d, h, w, c)];

            O.ptr[i] = x;
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct GatherJob : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadOnlyMemResource B { get; set; }
        public ReadWriteMemResource O { get; set; }

        [ReadOnly] public TensorShape shapeO;
        [ReadOnly] public TensorShape shapeX;
        [ReadOnly] public int axis;
        public void Execute(int i)
        {
            int s = 0, r = 0, n = 0, t = 0, d = 0, h = 0, w = 0, c = 0;
            shapeO.GetPositionsFromIndex(i, ref s, ref r, ref n, ref t, ref d, ref h, ref w, ref c);

            int d0 = (axis == 0) ? (int) B.ptr[s] : s;
            int d1 = (axis == 1) ? (int) B.ptr[r] : r;
            int d2 = (axis == 2) ? (int) B.ptr[n] : n;
            int d3 = (axis == 3) ? (int) B.ptr[t] : t;
            int d4 = (axis == 4) ? (int) B.ptr[d] : d;
            int d5 = (axis == 5) ? (int) B.ptr[h] : h;
            int d6 = (axis == 6) ? (int) B.ptr[w] : w;
            int d7 = (axis == 7) ? (int) B.ptr[c] : c;

            O.ptr[i] = X.ptr[shapeX.Index(d0, d1, d2, d3, d4, d5, d6, d7)];
        }
    }

    [BurstCompile(FloatPrecision.Low, FloatMode.Fast)]
    unsafe struct OneHotJob : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; }
        public ReadWriteMemResource O { get; set; }

        [ReadOnly] public TensorShape shapeO;
        [ReadOnly] public TensorShape shapeX;
        [ReadOnly] public int depth;
        [ReadOnly] public bool isInput1D;
        [ReadOnly] public float onValue;
        [ReadOnly] public float offValue;

        public void Execute(int idx)
        {
            int i = idx % shapeX.flatWidth;
            int j = (idx / shapeX.flatWidth) % depth;
            int n = ((idx / shapeX.flatWidth) / depth) % shapeX.flatHeight;

            int index = (int)X.ptr[n * shapeX.flatWidth + i];
            float v = (j == index) ? onValue: offValue;

            if (isInput1D)
                O.ptr[idx] = v;
            else
                O.ptr[idx] = v;
        }
    }
}

} // namespace Barracuda
