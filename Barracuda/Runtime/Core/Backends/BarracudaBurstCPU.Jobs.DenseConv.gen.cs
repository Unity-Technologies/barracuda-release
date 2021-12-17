// This is auto-generated -- do not modify directly
using UnityEngine;
using System;
using Unity.Burst;
using Unity.Burst.Intrinsics;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using static Unity.Burst.Intrinsics.X86.Avx;
using static Unity.Burst.Intrinsics.X86.Fma;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs.LowLevel.Unsafe;
using FencingHelperMode = Unity.Barracuda.BurstSchedulingHelper.FencingHelperMode;

namespace Unity.Barracuda {
public partial class BurstCPUOps
{
    #region Dense/Conv jobs declaration for mode: _Full_Float

    internal partial struct DepthwiseConv2DJobHelper
    {
        public JobHandle ScheduleXSBO(Tensor X, Tensor S, Tensor B, Tensor O, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            var pinX = Pin(X);
            var pinS = Pin(S);
            var pinB = Pin(B);
            var pinO = Pin(O, uploadCache: false);
            return ScheduleXSBO(pinX, pinS, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
        }
        public JobHandle ScheduleXSBO(BurstTensorData pinX, BurstTensorData pinS, BurstTensorData pinB, BurstTensorData pinO, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            bool AHalf = pinX.array.Type == DataType.Half;
            bool WHalf = pinS.array.Type == DataType.Half;
            bool BHalf = pinB.array.Type == DataType.Half;
            bool OHalf = pinO.array.Type == DataType.Half;
            UnityEngine.Assertions.Assert.AreEqual(AHalf, OHalf);
            UnityEngine.Assertions.Assert.AreEqual(WHalf, BHalf);
            if (AHalf && WHalf)
            {
                var job = new DepthwiseConv2DJob_Full_Half();
                job.data = this;
                return job.ScheduleXSBO(pinX, pinS, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else if (!AHalf && WHalf)
            {
                var job = new DepthwiseConv2DJob_ActAsFloat_WeightAsHalf();
                job.data = this;
                return job.ScheduleXSBO(pinX, pinS, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else if (!AHalf && !WHalf)
            {
                var job = new DepthwiseConv2DJob_Full_Float();
                job.data = this;
                return job.ScheduleXSBO(pinX, pinS, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else //if (AHalf && !WHalf)
            {
                UnityEngine.Assertions.Assert.IsTrue(false, "DepthwiseConv2DJob does not support activation as half while weights are floats.");
                return new JobHandle();
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    unsafe struct DepthwiseConv2DJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXSBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadOnlyMemResource S { get; set; } float* Sptr => S.ptrfloat;
        public ReadOnlyMemResource B { get; set; } float* Bptr => B.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public DepthwiseConv2DJobHelper data;

        const int unrollSize = 16;
        public void Execute(int y)
        {
            int accumulatorMemSize = data.kernelCount * sizeof(float);
            float* outputAccumulators = (float*)UnsafeUtility.Malloc(accumulatorMemSize, JobsUtility.CacheLineSize, Allocator.TempJob);
            for (int n = 0; n < data.outBatch; ++n)
            for (int x = 0; x < data.outWidth; ++x)
            {
                // reset accumulators to 0
                UnsafeUtility.MemClear(outputAccumulators, accumulatorMemSize);

                // gather X * K results in accumulators
                for (int dy = 0; dy < data.kernelHeight; ++dy)
                {
                    int readY = y * data.strideY + dy - data.padY;
                    if (readY < 0) continue;
                    if (readY >= data.inHeight) continue;

                    for (int dx = 0; dx < data.kernelWidth; ++dx)
                    {
                        int readX = x * data.strideX + dx - data.padY;
                        if (readX < 0) continue;
                        if (readX >= data.inWidth) continue;

                        float* dst    = outputAccumulators;
                        float* src    = Xptr + n * data.inStrideN + readY * data.inStrideH + readX * data.inStrideW;
                        float* kernel = Sptr + dy * data.kernelStrideH + dx * data.kernelStrideW;

                        int k = 0;
                        for (; k < data.kernelCount - unrollSize + 1; k += unrollSize) // unroll of kernelCount loop
                            for (int q = 0; q < unrollSize; q++, src++, dst++, kernel++)
                                *dst += (float)((*src) * (*kernel));
                        for (; k < data.kernelCount; k++, src++, dst++, kernel++) // remainder of kernelCount loop
                            *dst += (float)((*src) * (*kernel));
                    }
                }

                { // write accumulators to memory and add bias
                    int k = 0;
                    float* src  = outputAccumulators;
                    float* dst  = Optr + n * data.outStrideN + y * data.outStrideH + x * data.outStrideW;
                    float* bias = Bptr;
                    for (; k < data.kernelCount - unrollSize + 1; k += unrollSize)  // unroll of kernelCount loop
                        for (int q = 0; q < unrollSize; q++, src++, dst++, bias++)
                            *dst = (float)((*src) + (*bias));
                    for (; k < data.kernelCount; k++, src++, dst++, bias++) // remainder of kernelCount loop
                        *dst = (float)((*src) + (*bias));
                }
            }

            UnsafeUtility.Free(outputAccumulators, Allocator.TempJob);
        }
    }

    internal partial struct Dense3JobHelper
    {
        public JobHandle ScheduleXSBO(Tensor X, Tensor S, Tensor B, Tensor O, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            var pinX = Pin(X);
            var pinS = Pin(S);
            var pinB = Pin(B);
            var pinO = Pin(O, uploadCache: false);
            return ScheduleXSBO(pinX, pinS, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
        }
        public JobHandle ScheduleXSBO(BurstTensorData pinX, BurstTensorData pinS, BurstTensorData pinB, BurstTensorData pinO, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            bool AHalf = pinX.array.Type == DataType.Half;
            bool WHalf = pinS.array.Type == DataType.Half;
            bool BHalf = pinB.array.Type == DataType.Half;
            bool OHalf = pinO.array.Type == DataType.Half;
            UnityEngine.Assertions.Assert.AreEqual(AHalf, OHalf);
            UnityEngine.Assertions.Assert.AreEqual(WHalf, BHalf);
            if (AHalf && WHalf)
            {
                var job = new Dense3Job_Full_Half();
                job.data = this;
                return job.ScheduleXSBO(pinX, pinS, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else if (!AHalf && WHalf)
            {
                var job = new Dense3Job_ActAsFloat_WeightAsHalf();
                job.data = this;
                return job.ScheduleXSBO(pinX, pinS, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else if (!AHalf && !WHalf)
            {
                var job = new Dense3Job_Full_Float();
                job.data = this;
                return job.ScheduleXSBO(pinX, pinS, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else //if (AHalf && !WHalf)
            {
                UnityEngine.Assertions.Assert.IsTrue(false, "Dense3Job does not support activation as half while weights are floats.");
                return new JobHandle();
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    unsafe struct Dense3Job_Full_Float : IJobParallelFor, IJobResourceDeclarationXSBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadOnlyMemResource S { get; set; } float* Sptr => S.ptrfloat;
        public ReadOnlyMemResource B { get; set; } float* Bptr => B.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public Dense3JobHelper data;

        public const int blockSize = 16;
        public void Execute(int threadID)
        {
            float* A = this.Xptr;
            float* B = this.Sptr;
            float* C = this.Bptr;
            float* S = this.Optr;
            int AM = data.AM;
            int BM = data.BM;
            int SM = data.SM;
            int AN = data.AN;
            int BN = data.BN;
            int SN = data.SN;

            int dispatchThreadXY = data.dispatchThreadX * data.dispatchThreadY;

            int batch = (threadID / dispatchThreadXY);
            int i = (threadID % dispatchThreadXY) % data.dispatchThreadX;
            int j = (threadID % dispatchThreadXY) / data.dispatchThreadX;

            int batchOffSetA = (batch * AM * AN);
            int batchOffSetS = (batch * SM * SN);

            int rowA = i * blockSize;
            int colB = j * blockSize;

            unsafe
            {
                float* blockTempA = null;
                float* blockTempB = null;
                float* blockTempS = null;

                float* blockS = S + rowA + SM * colB + batchOffSetS;
                int strideS = SM;

                if (rowA + blockSize > SM || colB + blockSize > SN) // copy remainder of C into zero-padded block
                {
                    blockTempS = AllocBlock(blockSize, blockSize);
                    strideS = blockSize;
                    blockS = blockTempS;
                }
                for (int y = 0; y < blockSize; y++)
                    for (int x = 0; x < blockSize; x++)
                        blockS[x + strideS * y] = (float)((colB + y) < BN ? C[colB + y] : 0.0f);

                for (int l = 0; l < AN; l += blockSize) // inner-loop
                {
                    float* blockA = A + rowA + AM * l + batchOffSetA;
                    float* blockB = B + l * BN + colB;
                    int strideA = AM;
                    int strideB = BN;

                    if (rowA + blockSize > AM || l + blockSize > AN) // copy remainder of A into zero-padded block
                    {
                        if (blockTempA == null)
                            blockTempA = AllocBlock(blockSize, blockSize);
                        strideA = blockSize;

                        for (int y = 0; y < blockSize; y++)
                            for (int x = 0; x < blockSize; x++)
                                blockTempA[x + blockSize * y] = (float)(((rowA + x) < AM && (l + y < AN)) ? blockA[x + AM * y] : 0.0f);

                        blockA = blockTempA;
                    }

                    if (colB + blockSize > BN || l + blockSize > BM) // copy remainder of B into zero-padded block
                    {
                        if (blockTempB == null)
                            blockTempB = AllocBlock(blockSize, blockSize);
                        strideB = blockSize;

                        for (int y = 0; y < blockSize; y++)
                            for (int x = 0; x < blockSize; x++)
                                blockTempB[x + blockSize * y] = (float)(((colB + x) < BN && (l + y < BM)) ? blockB[x + BN * y] : 0.0f);

                        blockB = blockTempB;
                    }

                    MultiplyBlockUnrollHx16(blockA, strideA, blockB, strideB, blockS, strideS);
                }

                if (blockS == blockTempS) // copy back
                {
                    for (int y = 0; y < blockSize; y++)
                        for (int x = 0; x < blockSize; x++)
                        {
                            if (((rowA + x) < SM) && ((colB + y) < SN))
                                S[(rowA + x) + SM * (colB + y) + batchOffSetS] = blockTempS[x + blockSize * y];
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

                *(Sp + i + Sstride * 0 ) = (float)(sum0);
                *(Sp + i + Sstride * 1 ) = (float)(sum1);
                *(Sp + i + Sstride * 2 ) = (float)(sum2);
                *(Sp + i + Sstride * 3 ) = (float)(sum3);
                *(Sp + i + Sstride * 4 ) = (float)(sum4);
                *(Sp + i + Sstride * 5 ) = (float)(sum5);
                *(Sp + i + Sstride * 6 ) = (float)(sum6);
                *(Sp + i + Sstride * 7 ) = (float)(sum7);
                *(Sp + i + Sstride * 8 ) = (float)(sum8);
                *(Sp + i + Sstride * 9 ) = (float)(sum9);
                *(Sp + i + Sstride * 10) = (float)(sumA);
                *(Sp + i + Sstride * 11) = (float)(sumB);
                *(Sp + i + Sstride * 12) = (float)(sumC);
                *(Sp + i + Sstride * 13) = (float)(sumD);
                *(Sp + i + Sstride * 14) = (float)(sumE);
                *(Sp + i + Sstride * 15) = (float)(sumF);
            }
        }
    }

    #endregion
    #region Dense/Conv jobs declaration for mode: _ActAsFloat_WeightAsHalf

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    unsafe struct DepthwiseConv2DJob_ActAsFloat_WeightAsHalf : IJobParallelFor, IJobResourceDeclarationXSBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadOnlyMemResource S { get; set; } half* Sptr => S.ptrhalf;
        public ReadOnlyMemResource B { get; set; } half* Bptr => B.ptrhalf;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public DepthwiseConv2DJobHelper data;

        const int unrollSize = 16;
        public void Execute(int y)
        {
            int accumulatorMemSize = data.kernelCount * sizeof(float);
            float* outputAccumulators = (float*)UnsafeUtility.Malloc(accumulatorMemSize, JobsUtility.CacheLineSize, Allocator.TempJob);
            for (int n = 0; n < data.outBatch; ++n)
            for (int x = 0; x < data.outWidth; ++x)
            {
                // reset accumulators to 0
                UnsafeUtility.MemClear(outputAccumulators, accumulatorMemSize);

                // gather X * K results in accumulators
                for (int dy = 0; dy < data.kernelHeight; ++dy)
                {
                    int readY = y * data.strideY + dy - data.padY;
                    if (readY < 0) continue;
                    if (readY >= data.inHeight) continue;

                    for (int dx = 0; dx < data.kernelWidth; ++dx)
                    {
                        int readX = x * data.strideX + dx - data.padY;
                        if (readX < 0) continue;
                        if (readX >= data.inWidth) continue;

                        float* dst    = outputAccumulators;
                        float* src    = Xptr + n * data.inStrideN + readY * data.inStrideH + readX * data.inStrideW;
                        half* kernel = Sptr + dy * data.kernelStrideH + dx * data.kernelStrideW;

                        int k = 0;
                        for (; k < data.kernelCount - unrollSize + 1; k += unrollSize) // unroll of kernelCount loop
                            for (int q = 0; q < unrollSize; q++, src++, dst++, kernel++)
                                *dst += (float)((*src) * (*kernel));
                        for (; k < data.kernelCount; k++, src++, dst++, kernel++) // remainder of kernelCount loop
                            *dst += (float)((*src) * (*kernel));
                    }
                }

                { // write accumulators to memory and add bias
                    int k = 0;
                    float* src  = outputAccumulators;
                    float* dst  = Optr + n * data.outStrideN + y * data.outStrideH + x * data.outStrideW;
                    half* bias = Bptr;
                    for (; k < data.kernelCount - unrollSize + 1; k += unrollSize)  // unroll of kernelCount loop
                        for (int q = 0; q < unrollSize; q++, src++, dst++, bias++)
                            *dst = (float)((*src) + (*bias));
                    for (; k < data.kernelCount; k++, src++, dst++, bias++) // remainder of kernelCount loop
                        *dst = (float)((*src) + (*bias));
                }
            }

            UnsafeUtility.Free(outputAccumulators, Allocator.TempJob);
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    unsafe struct Dense3Job_ActAsFloat_WeightAsHalf : IJobParallelFor, IJobResourceDeclarationXSBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadOnlyMemResource S { get; set; } half* Sptr => S.ptrhalf;
        public ReadOnlyMemResource B { get; set; } half* Bptr => B.ptrhalf;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public Dense3JobHelper data;

        public const int blockSize = 16;
        public void Execute(int threadID)
        {
            float* A = this.Xptr;
            half* B = this.Sptr;
            half* C = this.Bptr;
            float* S = this.Optr;
            int AM = data.AM;
            int BM = data.BM;
            int SM = data.SM;
            int AN = data.AN;
            int BN = data.BN;
            int SN = data.SN;

            int dispatchThreadXY = data.dispatchThreadX * data.dispatchThreadY;

            int batch = (threadID / dispatchThreadXY);
            int i = (threadID % dispatchThreadXY) % data.dispatchThreadX;
            int j = (threadID % dispatchThreadXY) / data.dispatchThreadX;

            int batchOffSetA = (batch * AM * AN);
            int batchOffSetS = (batch * SM * SN);

            int rowA = i * blockSize;
            int colB = j * blockSize;

            unsafe
            {
                float* blockTempA = null;
                half* blockTempB = null;
                float* blockTempS = null;

                float* blockS = S + rowA + SM * colB + batchOffSetS;
                int strideS = SM;

                if (rowA + blockSize > SM || colB + blockSize > SN) // copy remainder of C into zero-padded block
                {
                    blockTempS = AllocBlock(blockSize, blockSize);
                    strideS = blockSize;
                    blockS = blockTempS;
                }
                for (int y = 0; y < blockSize; y++)
                    for (int x = 0; x < blockSize; x++)
                        blockS[x + strideS * y] = (float)((colB + y) < BN ? C[colB + y] : 0.0f);

                for (int l = 0; l < AN; l += blockSize) // inner-loop
                {
                    float* blockA = A + rowA + AM * l + batchOffSetA;
                    half* blockB = B + l * BN + colB;
                    int strideA = AM;
                    int strideB = BN;

                    if (rowA + blockSize > AM || l + blockSize > AN) // copy remainder of A into zero-padded block
                    {
                        if (blockTempA == null)
                            blockTempA = AllocBlock(blockSize, blockSize);
                        strideA = blockSize;

                        for (int y = 0; y < blockSize; y++)
                            for (int x = 0; x < blockSize; x++)
                                blockTempA[x + blockSize * y] = (float)(((rowA + x) < AM && (l + y < AN)) ? blockA[x + AM * y] : 0.0f);

                        blockA = blockTempA;
                    }

                    if (colB + blockSize > BN || l + blockSize > BM) // copy remainder of B into zero-padded block
                    {
                        if (blockTempB == null)
                            blockTempB = AllocBlockHalf(blockSize, blockSize);
                        strideB = blockSize;

                        for (int y = 0; y < blockSize; y++)
                            for (int x = 0; x < blockSize; x++)
                                blockTempB[x + blockSize * y] = (half)(((colB + x) < BN && (l + y < BM)) ? blockB[x + BN * y] : 0.0f);

                        blockB = blockTempB;
                    }

                    MultiplyBlockUnrollHx16(blockA, strideA, blockB, strideB, blockS, strideS);
                }

                if (blockS == blockTempS) // copy back
                {
                    for (int y = 0; y < blockSize; y++)
                        for (int x = 0; x < blockSize; x++)
                        {
                            if (((rowA + x) < SM) && ((colB + y) < SN))
                                S[(rowA + x) + SM * (colB + y) + batchOffSetS] = blockTempS[x + blockSize * y];
                        }
                }

                FreeBlock(blockTempA);
                FreeBlock(blockTempB);
                FreeBlock(blockTempS);
            }
        }

        static void MultiplyBlockUnrollHx16(float* Ap, int Astride, half* Bp, int Bstride, float* Sp, int Sstride)
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

                *(Sp + i + Sstride * 0 ) = (float)(sum0);
                *(Sp + i + Sstride * 1 ) = (float)(sum1);
                *(Sp + i + Sstride * 2 ) = (float)(sum2);
                *(Sp + i + Sstride * 3 ) = (float)(sum3);
                *(Sp + i + Sstride * 4 ) = (float)(sum4);
                *(Sp + i + Sstride * 5 ) = (float)(sum5);
                *(Sp + i + Sstride * 6 ) = (float)(sum6);
                *(Sp + i + Sstride * 7 ) = (float)(sum7);
                *(Sp + i + Sstride * 8 ) = (float)(sum8);
                *(Sp + i + Sstride * 9 ) = (float)(sum9);
                *(Sp + i + Sstride * 10) = (float)(sumA);
                *(Sp + i + Sstride * 11) = (float)(sumB);
                *(Sp + i + Sstride * 12) = (float)(sumC);
                *(Sp + i + Sstride * 13) = (float)(sumD);
                *(Sp + i + Sstride * 14) = (float)(sumE);
                *(Sp + i + Sstride * 15) = (float)(sumF);
            }
        }
    }

    #endregion
    #region Dense/Conv jobs declaration for mode: _Full_Half

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    unsafe struct DepthwiseConv2DJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXSBO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadOnlyMemResource S { get; set; } half* Sptr => S.ptrhalf;
        public ReadOnlyMemResource B { get; set; } half* Bptr => B.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public DepthwiseConv2DJobHelper data;

        const int unrollSize = 16;
        public void Execute(int y)
        {
            int accumulatorMemSize = data.kernelCount * sizeof(half);
            half* outputAccumulators = (half*)UnsafeUtility.Malloc(accumulatorMemSize, JobsUtility.CacheLineSize, Allocator.TempJob);
            for (int n = 0; n < data.outBatch; ++n)
            for (int x = 0; x < data.outWidth; ++x)
            {
                // reset accumulators to 0
                UnsafeUtility.MemClear(outputAccumulators, accumulatorMemSize);

                // gather X * K results in accumulators
                for (int dy = 0; dy < data.kernelHeight; ++dy)
                {
                    int readY = y * data.strideY + dy - data.padY;
                    if (readY < 0) continue;
                    if (readY >= data.inHeight) continue;

                    for (int dx = 0; dx < data.kernelWidth; ++dx)
                    {
                        int readX = x * data.strideX + dx - data.padY;
                        if (readX < 0) continue;
                        if (readX >= data.inWidth) continue;

                        half* dst    = outputAccumulators;
                        half* src    = Xptr + n * data.inStrideN + readY * data.inStrideH + readX * data.inStrideW;
                        half* kernel = Sptr + dy * data.kernelStrideH + dx * data.kernelStrideW;

                        int k = 0;
                        for (; k < data.kernelCount - unrollSize + 1; k += unrollSize) // unroll of kernelCount loop
                            for (int q = 0; q < unrollSize; q++, src++, dst++, kernel++)
                                *dst += (half)((*src) * (*kernel));
                        for (; k < data.kernelCount; k++, src++, dst++, kernel++) // remainder of kernelCount loop
                            *dst += (half)((*src) * (*kernel));
                    }
                }

                { // write accumulators to memory and add bias
                    int k = 0;
                    half* src  = outputAccumulators;
                    half* dst  = Optr + n * data.outStrideN + y * data.outStrideH + x * data.outStrideW;
                    half* bias = Bptr;
                    for (; k < data.kernelCount - unrollSize + 1; k += unrollSize)  // unroll of kernelCount loop
                        for (int q = 0; q < unrollSize; q++, src++, dst++, bias++)
                            *dst = (half)((*src) + (*bias));
                    for (; k < data.kernelCount; k++, src++, dst++, bias++) // remainder of kernelCount loop
                        *dst = (half)((*src) + (*bias));
                }
            }

            UnsafeUtility.Free(outputAccumulators, Allocator.TempJob);
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    unsafe struct Dense3Job_Full_Half : IJobParallelFor, IJobResourceDeclarationXSBO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadOnlyMemResource S { get; set; } half* Sptr => S.ptrhalf;
        public ReadOnlyMemResource B { get; set; } half* Bptr => B.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public Dense3JobHelper data;

        public const int blockSize = 16;
        public void Execute(int threadID)
        {
            half* A = this.Xptr;
            half* B = this.Sptr;
            half* C = this.Bptr;
            half* S = this.Optr;
            int AM = data.AM;
            int BM = data.BM;
            int SM = data.SM;
            int AN = data.AN;
            int BN = data.BN;
            int SN = data.SN;

            int dispatchThreadXY = data.dispatchThreadX * data.dispatchThreadY;

            int batch = (threadID / dispatchThreadXY);
            int i = (threadID % dispatchThreadXY) % data.dispatchThreadX;
            int j = (threadID % dispatchThreadXY) / data.dispatchThreadX;

            int batchOffSetA = (batch * AM * AN);
            int batchOffSetS = (batch * SM * SN);

            int rowA = i * blockSize;
            int colB = j * blockSize;

            unsafe
            {
                half* blockTempA = null;
                half* blockTempB = null;
                half* blockTempS = null;

                half* blockS = S + rowA + SM * colB + batchOffSetS;
                int strideS = SM;

                if (rowA + blockSize > SM || colB + blockSize > SN) // copy remainder of C into zero-padded block
                {
                    blockTempS = AllocBlockHalf(blockSize, blockSize);
                    strideS = blockSize;
                    blockS = blockTempS;
                }
                for (int y = 0; y < blockSize; y++)
                    for (int x = 0; x < blockSize; x++)
                        blockS[x + strideS * y] = (half)((colB + y) < BN ? C[colB + y] : 0.0f);

                for (int l = 0; l < AN; l += blockSize) // inner-loop
                {
                    half* blockA = A + rowA + AM * l + batchOffSetA;
                    half* blockB = B + l * BN + colB;
                    int strideA = AM;
                    int strideB = BN;

                    if (rowA + blockSize > AM || l + blockSize > AN) // copy remainder of A into zero-padded block
                    {
                        if (blockTempA == null)
                            blockTempA = AllocBlockHalf(blockSize, blockSize);
                        strideA = blockSize;

                        for (int y = 0; y < blockSize; y++)
                            for (int x = 0; x < blockSize; x++)
                                blockTempA[x + blockSize * y] = (half)(((rowA + x) < AM && (l + y < AN)) ? blockA[x + AM * y] : 0.0f);

                        blockA = blockTempA;
                    }

                    if (colB + blockSize > BN || l + blockSize > BM) // copy remainder of B into zero-padded block
                    {
                        if (blockTempB == null)
                            blockTempB = AllocBlockHalf(blockSize, blockSize);
                        strideB = blockSize;

                        for (int y = 0; y < blockSize; y++)
                            for (int x = 0; x < blockSize; x++)
                                blockTempB[x + blockSize * y] = (half)(((colB + x) < BN && (l + y < BM)) ? blockB[x + BN * y] : 0.0f);

                        blockB = blockTempB;
                    }

                    MultiplyBlockUnrollHx16(blockA, strideA, blockB, strideB, blockS, strideS);
                }

                if (blockS == blockTempS) // copy back
                {
                    for (int y = 0; y < blockSize; y++)
                        for (int x = 0; x < blockSize; x++)
                        {
                            if (((rowA + x) < SM) && ((colB + y) < SN))
                                S[(rowA + x) + SM * (colB + y) + batchOffSetS] = blockTempS[x + blockSize * y];
                        }
                }

                FreeBlock(blockTempA);
                FreeBlock(blockTempB);
                FreeBlock(blockTempS);
            }
        }

        static void MultiplyBlockUnrollHx16(half* Ap, int Astride, half* Bp, int Bstride, half* Sp, int Sstride)
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

                *(Sp + i + Sstride * 0 ) = (half)(sum0);
                *(Sp + i + Sstride * 1 ) = (half)(sum1);
                *(Sp + i + Sstride * 2 ) = (half)(sum2);
                *(Sp + i + Sstride * 3 ) = (half)(sum3);
                *(Sp + i + Sstride * 4 ) = (half)(sum4);
                *(Sp + i + Sstride * 5 ) = (half)(sum5);
                *(Sp + i + Sstride * 6 ) = (half)(sum6);
                *(Sp + i + Sstride * 7 ) = (half)(sum7);
                *(Sp + i + Sstride * 8 ) = (half)(sum8);
                *(Sp + i + Sstride * 9 ) = (half)(sum9);
                *(Sp + i + Sstride * 10) = (half)(sumA);
                *(Sp + i + Sstride * 11) = (half)(sumB);
                *(Sp + i + Sstride * 12) = (half)(sumC);
                *(Sp + i + Sstride * 13) = (half)(sumD);
                *(Sp + i + Sstride * 14) = (half)(sumE);
                *(Sp + i + Sstride * 15) = (half)(sumF);
            }
        }
    }

    #endregion
}
}
