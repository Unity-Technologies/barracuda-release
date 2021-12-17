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
    #region Reduce jobs declaration for mode: _Full_Float

    internal partial struct ReduceMaxJobHelper
    {
        public JobHandle ScheduleXO(BurstTensorData pinX, FencedMemoryAlloc pinO, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            bool AHalf = pinX.array.Type == DataType.Half;
            bool OHalf = pinO.type == DataType.Half;
            UnityEngine.Assertions.Assert.AreEqual(AHalf, OHalf);
            if (AHalf)
            {
                var job = new ReduceMaxJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new ReduceMaxJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    internal partial struct ReduceMaxJobHelper
    {
        public JobHandle ScheduleXO(Tensor X, Tensor O, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            var pinX = Pin(X);
            var pinO = Pin(O, uploadCache: false);
            return ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
        }
        public JobHandle ScheduleXO(BurstTensorData pinX, BurstTensorData pinO, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            bool AHalf = pinX.array.Type == DataType.Half;
            bool OHalf = pinO.array.Type == DataType.Half;
            UnityEngine.Assertions.Assert.AreEqual(AHalf, OHalf);
            if (AHalf)
            {
                var job = new ReduceMaxJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new ReduceMaxJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    unsafe struct ReduceMaxJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public ReduceMaxJobHelper data;

        public void Execute(int i)
        {
            int x = i % data.offsetReduce;
            int y = i / data.offsetReduce;

            float maxV = float.MinValue;
            for (int z = 0; z < data.reduceDim; ++z)
            {
                float v = Xptr[y * data.offsetReduce * data.reduceDim + z * data.offsetReduce + x];
                maxV = math.max(maxV, v);
            }
            Optr[y * data.offsetReduce + x] = (float)maxV;
        }
    }

    internal partial struct ReduceSumJobHelper
    {
        public JobHandle ScheduleXO(Tensor X, Tensor O, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            var pinX = Pin(X);
            var pinO = Pin(O, uploadCache: false);
            return ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
        }
        public JobHandle ScheduleXO(BurstTensorData pinX, BurstTensorData pinO, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            bool AHalf = pinX.array.Type == DataType.Half;
            bool OHalf = pinO.array.Type == DataType.Half;
            UnityEngine.Assertions.Assert.AreEqual(AHalf, OHalf);
            if (AHalf)
            {
                var job = new ReduceSumJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new ReduceSumJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    unsafe struct ReduceSumJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public ReduceSumJobHelper data;

        public void Execute(int i)
        {
            int x = i % data.offsetReduce;
            int y = i / data.offsetReduce;

            float sumV = 0;
            for (int z = 0; z < data.reduceDim; ++z)
            {
                float v = Xptr[y * data.offsetReduce * data.reduceDim + z * data.offsetReduce + x];
                sumV += v;
            }
            Optr[y * data.offsetReduce + x] = (float)(sumV);
        }
    }

    internal partial struct ReduceMeanJobHelper
    {
        public JobHandle ScheduleXO(Tensor X, Tensor O, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            var pinX = Pin(X);
            var pinO = Pin(O, uploadCache: false);
            return ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
        }
        public JobHandle ScheduleXO(BurstTensorData pinX, BurstTensorData pinO, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            bool AHalf = pinX.array.Type == DataType.Half;
            bool OHalf = pinO.array.Type == DataType.Half;
            UnityEngine.Assertions.Assert.AreEqual(AHalf, OHalf);
            if (AHalf)
            {
                var job = new ReduceMeanJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new ReduceMeanJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    unsafe struct ReduceMeanJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public ReduceMeanJobHelper data;

        public void Execute(int i)
        {
            int x = i % data.offsetReduce;
            int y = i / data.offsetReduce;

            float sumV = 0;
            for (int z = 0; z < data.reduceDim; ++z)
            {
                float v = Xptr[y * data.offsetReduce * data.reduceDim + z * data.offsetReduce + x];
                sumV += v;
            }
            Optr[y * data.offsetReduce + x] = (float)(sumV / (float)data.reduceDim);
        }
    }

    internal partial struct ExpBiasReduceJobHelper
    {
        public JobHandle ScheduleXBO(BurstTensorData pinX, FencedMemoryAlloc pinB, FencedMemoryAlloc pinO, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            bool AHalf = pinX.array.Type == DataType.Half;
            bool WHalf = pinB.type == DataType.Half;
            bool OHalf = pinO.type == DataType.Half;
            UnityEngine.Assertions.Assert.AreEqual(AHalf, OHalf);
            if (AHalf && WHalf)
            {
                var job = new ExpBiasReduceJob_Full_Half();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else if (!AHalf && WHalf)
            {
                var job = new ExpBiasReduceJob_ActAsFloat_WeightAsHalf();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else if (!AHalf && !WHalf)
            {
                var job = new ExpBiasReduceJob_Full_Float();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else //if (AHalf && !WHalf)
            {
                UnityEngine.Assertions.Assert.IsTrue(false, "ExpBiasReduceJob does not support activation as half while weights are floats.");
                return new JobHandle();
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    unsafe struct ExpBiasReduceJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadOnlyMemResource B { get; set; } float* Bptr => B.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public ExpBiasReduceJobHelper data;

        public void Execute(int i)
        {
            int x = i % data.offsetReduce;
            int y = i / data.offsetReduce;

            float accum = 0.0f;
            for (int z = 0; z < data.reduceDim; ++z)
            {
                float v = Xptr[y * data.offsetReduce * data.reduceDim + z * data.offsetReduce + x];
                float b = Bptr[y * data.offsetReduce + x];
                accum += math.exp(v - b);
            }
            Optr[y * data.offsetReduce + x] = (float)accum;
        }
    }

    internal partial struct SoftmaxEndJobHelper
    {
        public JobHandle ScheduleXSBO(BurstTensorData pinX, FencedMemoryAlloc pinS, FencedMemoryAlloc pinB, BurstTensorData pinO, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            bool AHalf = pinX.array.Type == DataType.Half;
            bool WHalf = pinS.type == DataType.Half;
            bool BHalf = pinB.type == DataType.Half;
            bool OHalf = pinO.array.Type == DataType.Half;
            UnityEngine.Assertions.Assert.AreEqual(AHalf, OHalf);
            UnityEngine.Assertions.Assert.AreEqual(WHalf, BHalf);
            if (AHalf && WHalf)
            {
                var job = new SoftmaxEndJob_Full_Half();
                job.data = this;
                return job.ScheduleXSBO(pinX, pinS, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else if (!AHalf && WHalf)
            {
                var job = new SoftmaxEndJob_ActAsFloat_WeightAsHalf();
                job.data = this;
                return job.ScheduleXSBO(pinX, pinS, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else if (!AHalf && !WHalf)
            {
                var job = new SoftmaxEndJob_Full_Float();
                job.data = this;
                return job.ScheduleXSBO(pinX, pinS, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else //if (AHalf && !WHalf)
            {
                UnityEngine.Assertions.Assert.IsTrue(false, "SoftmaxEndJob does not support activation as half while weights are floats.");
                return new JobHandle();
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct SoftmaxEndJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXSBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadOnlyMemResource S { get; set; } float* Sptr => S.ptrfloat;
        public ReadOnlyMemResource B { get; set; } float* Bptr => B.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public SoftmaxEndJobHelper data;

        public void Execute(int i)
        {
            int x = i % data.offsetReduce;
            int y = ((i / data.offsetReduce) % data.reduceDim);
            int z = ((i / data.offsetReduce) / data.reduceDim);

            Optr[i] = (float)(math.exp(Xptr[i] - Bptr[z * data.offsetReduce + x]) / Sptr[z * data.offsetReduce + x]);
        }
    }

    internal partial struct LogSoftmaxEndJobHelper
    {
        public JobHandle ScheduleXSBO(BurstTensorData pinX, FencedMemoryAlloc pinS, FencedMemoryAlloc pinB, BurstTensorData pinO, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            bool AHalf = pinX.array.Type == DataType.Half;
            bool WHalf = pinS.type == DataType.Half;
            bool BHalf = pinB.type == DataType.Half;
            bool OHalf = pinO.array.Type == DataType.Half;
            UnityEngine.Assertions.Assert.AreEqual(AHalf, OHalf);
            UnityEngine.Assertions.Assert.AreEqual(WHalf, BHalf);
            if (AHalf && WHalf)
            {
                var job = new LogSoftmaxEndJob_Full_Half();
                job.data = this;
                return job.ScheduleXSBO(pinX, pinS, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else if (!AHalf && WHalf)
            {
                var job = new LogSoftmaxEndJob_ActAsFloat_WeightAsHalf();
                job.data = this;
                return job.ScheduleXSBO(pinX, pinS, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else if (!AHalf && !WHalf)
            {
                var job = new LogSoftmaxEndJob_Full_Float();
                job.data = this;
                return job.ScheduleXSBO(pinX, pinS, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else //if (AHalf && !WHalf)
            {
                UnityEngine.Assertions.Assert.IsTrue(false, "LogSoftmaxEndJob does not support activation as half while weights are floats.");
                return new JobHandle();
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct LogSoftmaxEndJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXSBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadOnlyMemResource S { get; set; } float* Sptr => S.ptrfloat;
        public ReadOnlyMemResource B { get; set; } float* Bptr => B.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public LogSoftmaxEndJobHelper data;

        public void Execute(int i)
        {
            int x = i % data.offsetReduce;
            int y = ((i / data.offsetReduce) % data.reduceDim);
            int z = ((i / data.offsetReduce) / data.reduceDim);

            Optr[i] = (float)((Xptr[i] - Bptr[z * data.offsetReduce + x]) - math.log(Sptr[z * data.offsetReduce + x]));
        }
    }

    internal partial struct MaxPool2DJobHelper
    {
        public JobHandle ScheduleXO(Tensor X, Tensor O, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            var pinX = Pin(X);
            var pinO = Pin(O, uploadCache: false);
            return ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
        }
        public JobHandle ScheduleXO(BurstTensorData pinX, BurstTensorData pinO, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            bool AHalf = pinX.array.Type == DataType.Half;
            bool OHalf = pinO.array.Type == DataType.Half;
            UnityEngine.Assertions.Assert.AreEqual(AHalf, OHalf);
            if (AHalf)
            {
                var job = new MaxPool2DJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new MaxPool2DJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    unsafe struct MaxPool2DJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public MaxPool2DJobHelper data;

        const int unrollSize = 16;
        public void Execute(int y)
        {
            int accumulatorMemSize = data.inChannels * sizeof(float);
            float* outputAccumulators = (float*)UnsafeUtility.Malloc(accumulatorMemSize, JobsUtility.CacheLineSize, Allocator.TempJob);
            for (int n = 0; n < data.outBatch; ++n)
            for (int x = 0; x < data.outWidth; ++x)
            {
                bool firstNotRejectedPixelInKernel = true;
                // gather max results in accumulators
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

                        float* dst = outputAccumulators;
                        float* src = Xptr + n * data.inStrideN + readY * data.inStrideH + readX * data.inStrideW;

                        int k = 0;
                        if (firstNotRejectedPixelInKernel) // first pass, write-through
                        {
                            for (; k < data.inChannels - unrollSize + 1; k += unrollSize) // unroll of inChannels loop
                                for (int q = 0; q < unrollSize; q++, src++, dst++)
                                    *dst = *src;
                            for (; k < data.inChannels; k++, src++, dst++) // remainder of inChannels loop
                                *dst = *src;
                        }
                        else
                        {
                            for (; k < data.inChannels - unrollSize + 1; k += unrollSize) // unroll of inChannels loop
                                for (int q = 0; q < unrollSize; q++, src++, dst++)
                                    *dst = (*dst) > (*src) ? (*dst) : (*src);
                            for (; k < data.inChannels; k++, src++, dst++) // remainder of inChannels loop
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
                    float* dst  = Optr + n * data.outStrideN + y * data.outStrideH + x * data.outStrideW;
                    for (; k < data.inChannels - unrollSize + 1; k += unrollSize)  // unroll of inChannels loop
                        for (int q = 0; q < unrollSize; q++, src++, dst++)
                            *dst = *src;
                    for (; k < data.inChannels; k++, src++, dst++) // remainder of inChannels loop
                        *dst = *src;
                }
            }

            UnsafeUtility.Free(outputAccumulators, Allocator.TempJob);
        }
    }

    internal partial struct AvgPool2DJobHelper
    {
        public JobHandle ScheduleXO(Tensor X, Tensor O, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            var pinX = Pin(X);
            var pinO = Pin(O, uploadCache: false);
            return ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
        }
        public JobHandle ScheduleXO(BurstTensorData pinX, BurstTensorData pinO, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            bool AHalf = pinX.array.Type == DataType.Half;
            bool OHalf = pinO.array.Type == DataType.Half;
            UnityEngine.Assertions.Assert.AreEqual(AHalf, OHalf);
            if (AHalf)
            {
                var job = new AvgPool2DJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new AvgPool2DJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    unsafe struct AvgPool2DJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public AvgPool2DJobHelper data;

        const int unrollSize = 16;
        public void Execute(int y)
        {
            int accumulatorMemSize = data.inChannels * sizeof(float);
            float* outputAccumulators = (float*)UnsafeUtility.Malloc(accumulatorMemSize, JobsUtility.CacheLineSize, Allocator.TempJob);

            for (int n = 0; n < data.outBatch; ++n)
            for (int x = 0; x < data.outWidth; ++x)
            {
                // reset accumulators & counter
                int counter = 0;
                UnsafeUtility.MemClear(outputAccumulators, accumulatorMemSize);

                // gather sums in accumulators
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

                        float* dst = outputAccumulators;
                        float* src = Xptr + n * data.inStrideN + readY * data.inStrideH + readX * data.inStrideW;

                        int k = 0;
                        for (; k < data.inChannels - unrollSize + 1; k += unrollSize) // unroll of inChannels loop
                            for (int q = 0; q < unrollSize; q++, src++, dst++)
                                *dst += *src;
                        for (; k < data.inChannels; k++, src++, dst++) // remainder of inChannels loop
                            *dst += *src;
                        counter++;
                    }
                }

                // safety net, if kernel was completely outside of X
                counter = math.max(1, counter);

                { // write accumulators to memory
                    int k = 0;
                    float invCounter = 1f / counter;
                    float* src  = outputAccumulators;
                    float* dst  = Optr + n * data.outStrideN + y * data.outStrideH + x * data.outStrideW;
                    for (; k < data.inChannels - unrollSize + 1; k += unrollSize)  // unroll of inChannels loop
                        for (int q = 0; q < unrollSize; q++, src++, dst++)
                            *dst = (float)(*src * invCounter);
                    for (; k < data.inChannels; k++, src++, dst++) // remainder of inChannels loop
                        *dst = (float)(*src * invCounter);
                }
            }

            UnsafeUtility.Free(outputAccumulators, Allocator.TempJob);
        }
    }

    #endregion
    #region Reduce jobs declaration for mode: _ActAsFloat_WeightAsHalf




    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    unsafe struct ExpBiasReduceJob_ActAsFloat_WeightAsHalf : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadOnlyMemResource B { get; set; } half* Bptr => B.ptrhalf;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public ExpBiasReduceJobHelper data;

        public void Execute(int i)
        {
            int x = i % data.offsetReduce;
            int y = i / data.offsetReduce;

            float accum = 0.0f;
            for (int z = 0; z < data.reduceDim; ++z)
            {
                float v = Xptr[y * data.offsetReduce * data.reduceDim + z * data.offsetReduce + x];
                float b = Bptr[y * data.offsetReduce + x];
                accum += math.exp(v - b);
            }
            Optr[y * data.offsetReduce + x] = (float)accum;
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct SoftmaxEndJob_ActAsFloat_WeightAsHalf : IJobParallelFor, IJobResourceDeclarationXSBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadOnlyMemResource S { get; set; } half* Sptr => S.ptrhalf;
        public ReadOnlyMemResource B { get; set; } half* Bptr => B.ptrhalf;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public SoftmaxEndJobHelper data;

        public void Execute(int i)
        {
            int x = i % data.offsetReduce;
            int y = ((i / data.offsetReduce) % data.reduceDim);
            int z = ((i / data.offsetReduce) / data.reduceDim);

            Optr[i] = (float)(math.exp(Xptr[i] - Bptr[z * data.offsetReduce + x]) / Sptr[z * data.offsetReduce + x]);
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct LogSoftmaxEndJob_ActAsFloat_WeightAsHalf : IJobParallelFor, IJobResourceDeclarationXSBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadOnlyMemResource S { get; set; } half* Sptr => S.ptrhalf;
        public ReadOnlyMemResource B { get; set; } half* Bptr => B.ptrhalf;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public LogSoftmaxEndJobHelper data;

        public void Execute(int i)
        {
            int x = i % data.offsetReduce;
            int y = ((i / data.offsetReduce) % data.reduceDim);
            int z = ((i / data.offsetReduce) / data.reduceDim);

            Optr[i] = (float)((Xptr[i] - Bptr[z * data.offsetReduce + x]) - math.log(Sptr[z * data.offsetReduce + x]));
        }
    }



    #endregion
    #region Reduce jobs declaration for mode: _Full_Half

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    unsafe struct ReduceMaxJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public ReduceMaxJobHelper data;

        public void Execute(int i)
        {
            int x = i % data.offsetReduce;
            int y = i / data.offsetReduce;

            float maxV = float.MinValue;
            for (int z = 0; z < data.reduceDim; ++z)
            {
                float v = Xptr[y * data.offsetReduce * data.reduceDim + z * data.offsetReduce + x];
                maxV = math.max(maxV, v);
            }
            Optr[y * data.offsetReduce + x] = (half)maxV;
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    unsafe struct ReduceSumJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public ReduceSumJobHelper data;

        public void Execute(int i)
        {
            int x = i % data.offsetReduce;
            int y = i / data.offsetReduce;

            float sumV = 0;
            for (int z = 0; z < data.reduceDim; ++z)
            {
                float v = Xptr[y * data.offsetReduce * data.reduceDim + z * data.offsetReduce + x];
                sumV += v;
            }
            Optr[y * data.offsetReduce + x] = (half)(sumV);
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    unsafe struct ReduceMeanJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public ReduceMeanJobHelper data;

        public void Execute(int i)
        {
            int x = i % data.offsetReduce;
            int y = i / data.offsetReduce;

            float sumV = 0;
            for (int z = 0; z < data.reduceDim; ++z)
            {
                float v = Xptr[y * data.offsetReduce * data.reduceDim + z * data.offsetReduce + x];
                sumV += v;
            }
            Optr[y * data.offsetReduce + x] = (half)(sumV / (float)data.reduceDim);
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    unsafe struct ExpBiasReduceJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadOnlyMemResource B { get; set; } half* Bptr => B.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public ExpBiasReduceJobHelper data;

        public void Execute(int i)
        {
            int x = i % data.offsetReduce;
            int y = i / data.offsetReduce;

            float accum = 0.0f;
            for (int z = 0; z < data.reduceDim; ++z)
            {
                float v = Xptr[y * data.offsetReduce * data.reduceDim + z * data.offsetReduce + x];
                float b = Bptr[y * data.offsetReduce + x];
                accum += math.exp(v - b);
            }
            Optr[y * data.offsetReduce + x] = (half)accum;
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct SoftmaxEndJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXSBO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadOnlyMemResource S { get; set; } half* Sptr => S.ptrhalf;
        public ReadOnlyMemResource B { get; set; } half* Bptr => B.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public SoftmaxEndJobHelper data;

        public void Execute(int i)
        {
            int x = i % data.offsetReduce;
            int y = ((i / data.offsetReduce) % data.reduceDim);
            int z = ((i / data.offsetReduce) / data.reduceDim);

            Optr[i] = (half)(math.exp(Xptr[i] - Bptr[z * data.offsetReduce + x]) / Sptr[z * data.offsetReduce + x]);
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct LogSoftmaxEndJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXSBO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadOnlyMemResource S { get; set; } half* Sptr => S.ptrhalf;
        public ReadOnlyMemResource B { get; set; } half* Bptr => B.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public LogSoftmaxEndJobHelper data;

        public void Execute(int i)
        {
            int x = i % data.offsetReduce;
            int y = ((i / data.offsetReduce) % data.reduceDim);
            int z = ((i / data.offsetReduce) / data.reduceDim);

            Optr[i] = (half)((Xptr[i] - Bptr[z * data.offsetReduce + x]) - math.log(Sptr[z * data.offsetReduce + x]));
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    unsafe struct MaxPool2DJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public MaxPool2DJobHelper data;

        const int unrollSize = 16;
        public void Execute(int y)
        {
            int accumulatorMemSize = data.inChannels * sizeof(half);
            half* outputAccumulators = (half*)UnsafeUtility.Malloc(accumulatorMemSize, JobsUtility.CacheLineSize, Allocator.TempJob);
            for (int n = 0; n < data.outBatch; ++n)
            for (int x = 0; x < data.outWidth; ++x)
            {
                bool firstNotRejectedPixelInKernel = true;
                // gather max results in accumulators
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

                        half* dst = outputAccumulators;
                        half* src = Xptr + n * data.inStrideN + readY * data.inStrideH + readX * data.inStrideW;

                        int k = 0;
                        if (firstNotRejectedPixelInKernel) // first pass, write-through
                        {
                            for (; k < data.inChannels - unrollSize + 1; k += unrollSize) // unroll of inChannels loop
                                for (int q = 0; q < unrollSize; q++, src++, dst++)
                                    *dst = *src;
                            for (; k < data.inChannels; k++, src++, dst++) // remainder of inChannels loop
                                *dst = *src;
                        }
                        else
                        {
                            for (; k < data.inChannels - unrollSize + 1; k += unrollSize) // unroll of inChannels loop
                                for (int q = 0; q < unrollSize; q++, src++, dst++)
                                    *dst = (*dst) > (*src) ? (*dst) : (*src);
                            for (; k < data.inChannels; k++, src++, dst++) // remainder of inChannels loop
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
                    half* src  = outputAccumulators;
                    half* dst  = Optr + n * data.outStrideN + y * data.outStrideH + x * data.outStrideW;
                    for (; k < data.inChannels - unrollSize + 1; k += unrollSize)  // unroll of inChannels loop
                        for (int q = 0; q < unrollSize; q++, src++, dst++)
                            *dst = *src;
                    for (; k < data.inChannels; k++, src++, dst++) // remainder of inChannels loop
                        *dst = *src;
                }
            }

            UnsafeUtility.Free(outputAccumulators, Allocator.TempJob);
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    unsafe struct AvgPool2DJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public AvgPool2DJobHelper data;

        const int unrollSize = 16;
        public void Execute(int y)
        {
            int accumulatorMemSize = data.inChannels * sizeof(half);
            half* outputAccumulators = (half*)UnsafeUtility.Malloc(accumulatorMemSize, JobsUtility.CacheLineSize, Allocator.TempJob);

            for (int n = 0; n < data.outBatch; ++n)
            for (int x = 0; x < data.outWidth; ++x)
            {
                // reset accumulators & counter
                int counter = 0;
                UnsafeUtility.MemClear(outputAccumulators, accumulatorMemSize);

                // gather sums in accumulators
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

                        half* dst = outputAccumulators;
                        half* src = Xptr + n * data.inStrideN + readY * data.inStrideH + readX * data.inStrideW;

                        int k = 0;
                        for (; k < data.inChannels - unrollSize + 1; k += unrollSize) // unroll of inChannels loop
                            for (int q = 0; q < unrollSize; q++, src++, dst++)
                                *dst += *src;
                        for (; k < data.inChannels; k++, src++, dst++) // remainder of inChannels loop
                            *dst += *src;
                        counter++;
                    }
                }

                // safety net, if kernel was completely outside of X
                counter = math.max(1, counter);

                { // write accumulators to memory
                    int k = 0;
                    float invCounter = 1f / counter;
                    half* src  = outputAccumulators;
                    half* dst  = Optr + n * data.outStrideN + y * data.outStrideH + x * data.outStrideW;
                    for (; k < data.inChannels - unrollSize + 1; k += unrollSize)  // unroll of inChannels loop
                        for (int q = 0; q < unrollSize; q++, src++, dst++)
                            *dst = (half)(*src * invCounter);
                    for (; k < data.inChannels; k++, src++, dst++) // remainder of inChannels loop
                        *dst = (half)(*src * invCounter);
                }
            }

            UnsafeUtility.Free(outputAccumulators, Allocator.TempJob);
        }
    }

    #endregion
}
}
