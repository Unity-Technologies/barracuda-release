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
    #region Broadcast Jobs declaration for mode: _Full_Float

    internal partial struct VectorBroadcastScaleBiasJobHelper
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
                var job = new VectorBroadcastScaleBiasJob_Full_Half();
                job.data = this;
                return job.ScheduleXSBO(pinX, pinS, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else if (!AHalf && WHalf)
            {
                var job = new VectorBroadcastScaleBiasJob_ActAsFloat_WeightAsHalf();
                job.data = this;
                return job.ScheduleXSBO(pinX, pinS, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else if (!AHalf && !WHalf)
            {
                var job = new VectorBroadcastScaleBiasJob_Full_Float();
                job.data = this;
                return job.ScheduleXSBO(pinX, pinS, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else //if (AHalf && !WHalf)
            {
                UnityEngine.Assertions.Assert.IsTrue(false, "VectorBroadcastScaleBiasJob does not support activation as half while weights are floats.");
                return new JobHandle();
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    unsafe struct VectorBroadcastScaleBiasJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXSBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadOnlyMemResource S { get; set; } float* Sptr => S.ptrfloat;
        public ReadOnlyMemResource B { get; set; } float* Bptr => B.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public VectorBroadcastScaleBiasJobHelper data;

        const int unrollSize = 32;
        public void Execute(int i)
        {
            float* src   = Xptr + i * data.inOutChannels;
            float* dst   = Optr + i * data.inOutChannels;
            float* gamma = Sptr;
            float* beta  = Bptr;

            int j = 0;
            for (; j < data.inOutChannels - unrollSize + 1; j += unrollSize) // unroll of inOutChannels loop
                for (int q = 0; q < unrollSize; q++, src++, dst++, gamma++, beta++)
                    *dst = (float)((*src) * (*gamma) + (*beta) * data.alpha);
            for (; j < data.inOutChannels; j++, src++, dst++, gamma++, beta++) // remainder of inOutChannels loop
                *dst = (float)((*src) * (*gamma) + (*beta) * data.alpha);
        }
    }

    internal partial struct ScalarBroadcastAddJobHelper
    {
        [ReadOnly] public float alpha;
        public JobHandle ScheduleXBO(Tensor X, Tensor B, Tensor O, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            var pinX = Pin(X);
            var pinB = Pin(B);
            var pinO = Pin(O, uploadCache: false);
            bool AHalf = pinX.array.Type == DataType.Half;
            bool WHalf = pinB.array.Type == DataType.Half;
            bool OHalf = pinO.array.Type == DataType.Half;
            UnityEngine.Assertions.Assert.AreEqual(AHalf, OHalf);
            UnityEngine.Assertions.Assert.AreEqual(AHalf, WHalf);
            if (AHalf)
            {
                var job = new ScalarBroadcastAddJob_Full_Half();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else //if (!AHalf)
            {
                var job = new ScalarBroadcastAddJob_Full_Float();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ScalarBroadcastAddJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadOnlyMemResource B { get; set; } float* Bptr => B.ptrfloat;//Always use activation type
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public ScalarBroadcastAddJobHelper data;

        public void Execute(int i)
        {
            float v = Bptr[0] * data.alpha + Xptr[i];
            Optr[i] = (float)v;
        }
    }
    internal partial struct BroadcastAddJobHelper
    {
        [ReadOnly] public float alpha;
        public JobHandle ScheduleXBO(Tensor X, Tensor B, Tensor O, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            var pinX = Pin(X);
            var pinB = Pin(B);
            var pinO = Pin(O, uploadCache: false);
            bool AHalf = pinX.array.Type == DataType.Half;
            bool WHalf = pinB.array.Type == DataType.Half;
            bool OHalf = pinO.array.Type == DataType.Half;
            UnityEngine.Assertions.Assert.AreEqual(AHalf, OHalf);
            UnityEngine.Assertions.Assert.AreEqual(AHalf, WHalf);
            if (AHalf)
            {
                var job = new BroadcastAddJob_Full_Half();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else //if (!AHalf)
            {
                var job = new BroadcastAddJob_Full_Float();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct BroadcastAddJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadOnlyMemResource B { get; set; } float* Bptr => B.ptrfloat;//Always use activation type
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public BroadcastAddJobHelper data;

        public void Execute(int i)
        {
            float v = Bptr[i] * data.alpha + Xptr[i];
            Optr[i] = (float)v;
        }
    }
    internal partial struct ScalarBroadcastMulJobHelper
    {
        [ReadOnly] public float alpha;
        public JobHandle ScheduleXBO(Tensor X, Tensor B, Tensor O, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            var pinX = Pin(X);
            var pinB = Pin(B);
            var pinO = Pin(O, uploadCache: false);
            bool AHalf = pinX.array.Type == DataType.Half;
            bool WHalf = pinB.array.Type == DataType.Half;
            bool OHalf = pinO.array.Type == DataType.Half;
            UnityEngine.Assertions.Assert.AreEqual(AHalf, OHalf);
            UnityEngine.Assertions.Assert.AreEqual(AHalf, WHalf);
            if (AHalf)
            {
                var job = new ScalarBroadcastMulJob_Full_Half();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else //if (!AHalf)
            {
                var job = new ScalarBroadcastMulJob_Full_Float();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ScalarBroadcastMulJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadOnlyMemResource B { get; set; } float* Bptr => B.ptrfloat;//Always use activation type
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public ScalarBroadcastMulJobHelper data;

        public void Execute(int i)
        {
            float v = Xptr[i] * Bptr[0];
            Optr[i] = (float)v;
        }
    }
    internal partial struct BroadcastMulJobHelper
    {
        [ReadOnly] public float alpha;
        public JobHandle ScheduleXBO(Tensor X, Tensor B, Tensor O, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            var pinX = Pin(X);
            var pinB = Pin(B);
            var pinO = Pin(O, uploadCache: false);
            bool AHalf = pinX.array.Type == DataType.Half;
            bool WHalf = pinB.array.Type == DataType.Half;
            bool OHalf = pinO.array.Type == DataType.Half;
            UnityEngine.Assertions.Assert.AreEqual(AHalf, OHalf);
            UnityEngine.Assertions.Assert.AreEqual(AHalf, WHalf);
            if (AHalf)
            {
                var job = new BroadcastMulJob_Full_Half();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else //if (!AHalf)
            {
                var job = new BroadcastMulJob_Full_Float();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct BroadcastMulJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadOnlyMemResource B { get; set; } float* Bptr => B.ptrfloat;//Always use activation type
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public BroadcastMulJobHelper data;

        public void Execute(int i)
        {
            float v = Xptr[i] * Bptr[i];
            Optr[i] = (float)v;
        }
    }
    internal partial struct ScalarBroadcastDivJobHelper
    {
        [ReadOnly] public float alpha;
        public JobHandle ScheduleXBO(Tensor X, Tensor B, Tensor O, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            var pinX = Pin(X);
            var pinB = Pin(B);
            var pinO = Pin(O, uploadCache: false);
            bool AHalf = pinX.array.Type == DataType.Half;
            bool WHalf = pinB.array.Type == DataType.Half;
            bool OHalf = pinO.array.Type == DataType.Half;
            UnityEngine.Assertions.Assert.AreEqual(AHalf, OHalf);
            UnityEngine.Assertions.Assert.AreEqual(AHalf, WHalf);
            if (AHalf)
            {
                var job = new ScalarBroadcastDivJob_Full_Half();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else //if (!AHalf)
            {
                var job = new ScalarBroadcastDivJob_Full_Float();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ScalarBroadcastDivJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadOnlyMemResource B { get; set; } float* Bptr => B.ptrfloat;//Always use activation type
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public ScalarBroadcastDivJobHelper data;

        public void Execute(int i)
        {
            float v = Xptr[i] / Bptr[0];
            Optr[i] = (float)v;
        }
    }
    internal partial struct BroadcastDivJobHelper
    {
        [ReadOnly] public float alpha;
        public JobHandle ScheduleXBO(Tensor X, Tensor B, Tensor O, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            var pinX = Pin(X);
            var pinB = Pin(B);
            var pinO = Pin(O, uploadCache: false);
            bool AHalf = pinX.array.Type == DataType.Half;
            bool WHalf = pinB.array.Type == DataType.Half;
            bool OHalf = pinO.array.Type == DataType.Half;
            UnityEngine.Assertions.Assert.AreEqual(AHalf, OHalf);
            UnityEngine.Assertions.Assert.AreEqual(AHalf, WHalf);
            if (AHalf)
            {
                var job = new BroadcastDivJob_Full_Half();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else //if (!AHalf)
            {
                var job = new BroadcastDivJob_Full_Float();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct BroadcastDivJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadOnlyMemResource B { get; set; } float* Bptr => B.ptrfloat;//Always use activation type
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public BroadcastDivJobHelper data;

        public void Execute(int i)
        {
            float v = Xptr[i] / Bptr[i];
            Optr[i] = (float)v;
        }
    }
    internal partial struct ScalarBroadcastMinJobHelper
    {
        [ReadOnly] public float alpha;
        public JobHandle ScheduleXBO(Tensor X, Tensor B, Tensor O, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            var pinX = Pin(X);
            var pinB = Pin(B);
            var pinO = Pin(O, uploadCache: false);
            bool AHalf = pinX.array.Type == DataType.Half;
            bool WHalf = pinB.array.Type == DataType.Half;
            bool OHalf = pinO.array.Type == DataType.Half;
            UnityEngine.Assertions.Assert.AreEqual(AHalf, OHalf);
            UnityEngine.Assertions.Assert.AreEqual(AHalf, WHalf);
            if (AHalf)
            {
                var job = new ScalarBroadcastMinJob_Full_Half();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else //if (!AHalf)
            {
                var job = new ScalarBroadcastMinJob_Full_Float();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ScalarBroadcastMinJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadOnlyMemResource B { get; set; } float* Bptr => B.ptrfloat;//Always use activation type
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public ScalarBroadcastMinJobHelper data;

        public void Execute(int i)
        {
            float v = math.min(Xptr[i], Bptr[0]);
            Optr[i] = (float)v;
        }
    }
    internal partial struct BroadcastMinJobHelper
    {
        [ReadOnly] public float alpha;
        public JobHandle ScheduleXBO(Tensor X, Tensor B, Tensor O, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            var pinX = Pin(X);
            var pinB = Pin(B);
            var pinO = Pin(O, uploadCache: false);
            bool AHalf = pinX.array.Type == DataType.Half;
            bool WHalf = pinB.array.Type == DataType.Half;
            bool OHalf = pinO.array.Type == DataType.Half;
            UnityEngine.Assertions.Assert.AreEqual(AHalf, OHalf);
            UnityEngine.Assertions.Assert.AreEqual(AHalf, WHalf);
            if (AHalf)
            {
                var job = new BroadcastMinJob_Full_Half();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else //if (!AHalf)
            {
                var job = new BroadcastMinJob_Full_Float();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct BroadcastMinJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadOnlyMemResource B { get; set; } float* Bptr => B.ptrfloat;//Always use activation type
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public BroadcastMinJobHelper data;

        public void Execute(int i)
        {
            float v = math.min(Xptr[i], Bptr[i]);
            Optr[i] = (float)v;
        }
    }
    internal partial struct ScalarBroadcastMaxJobHelper
    {
        [ReadOnly] public float alpha;
        public JobHandle ScheduleXBO(Tensor X, Tensor B, Tensor O, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            var pinX = Pin(X);
            var pinB = Pin(B);
            var pinO = Pin(O, uploadCache: false);
            bool AHalf = pinX.array.Type == DataType.Half;
            bool WHalf = pinB.array.Type == DataType.Half;
            bool OHalf = pinO.array.Type == DataType.Half;
            UnityEngine.Assertions.Assert.AreEqual(AHalf, OHalf);
            UnityEngine.Assertions.Assert.AreEqual(AHalf, WHalf);
            if (AHalf)
            {
                var job = new ScalarBroadcastMaxJob_Full_Half();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else //if (!AHalf)
            {
                var job = new ScalarBroadcastMaxJob_Full_Float();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ScalarBroadcastMaxJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadOnlyMemResource B { get; set; } float* Bptr => B.ptrfloat;//Always use activation type
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public ScalarBroadcastMaxJobHelper data;

        public void Execute(int i)
        {
            float v = math.max(Xptr[i], Bptr[0]);
            Optr[i] = (float)v;
        }
    }
    internal partial struct BroadcastMaxJobHelper
    {
        [ReadOnly] public float alpha;
        public JobHandle ScheduleXBO(Tensor X, Tensor B, Tensor O, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            var pinX = Pin(X);
            var pinB = Pin(B);
            var pinO = Pin(O, uploadCache: false);
            bool AHalf = pinX.array.Type == DataType.Half;
            bool WHalf = pinB.array.Type == DataType.Half;
            bool OHalf = pinO.array.Type == DataType.Half;
            UnityEngine.Assertions.Assert.AreEqual(AHalf, OHalf);
            UnityEngine.Assertions.Assert.AreEqual(AHalf, WHalf);
            if (AHalf)
            {
                var job = new BroadcastMaxJob_Full_Half();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else //if (!AHalf)
            {
                var job = new BroadcastMaxJob_Full_Float();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct BroadcastMaxJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadOnlyMemResource B { get; set; } float* Bptr => B.ptrfloat;//Always use activation type
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public BroadcastMaxJobHelper data;

        public void Execute(int i)
        {
            float v = math.max(Xptr[i], Bptr[i]);
            Optr[i] = (float)v;
        }
    }
    internal partial struct ScalarBroadcastPowJobHelper
    {
        [ReadOnly] public float alpha;
        public JobHandle ScheduleXBO(Tensor X, Tensor B, Tensor O, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            var pinX = Pin(X);
            var pinB = Pin(B);
            var pinO = Pin(O, uploadCache: false);
            bool AHalf = pinX.array.Type == DataType.Half;
            bool WHalf = pinB.array.Type == DataType.Half;
            bool OHalf = pinO.array.Type == DataType.Half;
            UnityEngine.Assertions.Assert.AreEqual(AHalf, OHalf);
            UnityEngine.Assertions.Assert.AreEqual(AHalf, WHalf);
            if (AHalf)
            {
                var job = new ScalarBroadcastPowJob_Full_Half();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else //if (!AHalf)
            {
                var job = new ScalarBroadcastPowJob_Full_Float();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ScalarBroadcastPowJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadOnlyMemResource B { get; set; } float* Bptr => B.ptrfloat;//Always use activation type
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public ScalarBroadcastPowJobHelper data;

        public void Execute(int i)
        {
            float v = math.pow(Xptr[i], Bptr[0]);
            Optr[i] = (float)v;
        }
    }
    internal partial struct BroadcastPowJobHelper
    {
        [ReadOnly] public float alpha;
        public JobHandle ScheduleXBO(Tensor X, Tensor B, Tensor O, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            var pinX = Pin(X);
            var pinB = Pin(B);
            var pinO = Pin(O, uploadCache: false);
            bool AHalf = pinX.array.Type == DataType.Half;
            bool WHalf = pinB.array.Type == DataType.Half;
            bool OHalf = pinO.array.Type == DataType.Half;
            UnityEngine.Assertions.Assert.AreEqual(AHalf, OHalf);
            UnityEngine.Assertions.Assert.AreEqual(AHalf, WHalf);
            if (AHalf)
            {
                var job = new BroadcastPowJob_Full_Half();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else //if (!AHalf)
            {
                var job = new BroadcastPowJob_Full_Float();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct BroadcastPowJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadOnlyMemResource B { get; set; } float* Bptr => B.ptrfloat;//Always use activation type
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public BroadcastPowJobHelper data;

        public void Execute(int i)
        {
            float v = math.pow(Xptr[i], Bptr[i]);
            Optr[i] = (float)v;
        }
    }

    internal unsafe struct ElementwiseAddJobHelper
    {
        [ReadOnly] public TensorShape shapeO;
        [ReadOnly] public fixed int stridesX[8];
        [ReadOnly] public fixed int stridesY[8];
        [ReadOnly] public float alpha;
        public JobHandle ScheduleXBO(Tensor X, Tensor B, Tensor O, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            var pinX = Pin(X);
            var pinB = Pin(B);
            var pinO = Pin(O, uploadCache: false);
            bool AHalf = pinX.array.Type == DataType.Half;
            bool WHalf = pinB.array.Type == DataType.Half;
            bool OHalf = pinO.array.Type == DataType.Half;
            UnityEngine.Assertions.Assert.AreEqual(AHalf, OHalf);
            UnityEngine.Assertions.Assert.AreEqual(AHalf, WHalf);
            if (AHalf)
            {
                var job = new ElementwiseAddJob_Full_Half();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else //if (!AHalf)
            {
                var job = new ElementwiseAddJob_Full_Float();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ElementwiseAddJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadOnlyMemResource B { get; set; } float* Bptr => B.ptrfloat;//Always use activation type
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public ElementwiseAddJobHelper data;

        public void Execute(int i)
        {
            int s = 0, r = 0, n = 0, t = 0, d = 0, h = 0, w = 0, c = 0;
            data.shapeO.GetPositionsFromIndex(i, ref s, ref r, ref n, ref t, ref d, ref h, ref w, ref c);

            float x = Xptr[data.stridesX[0] * s + data.stridesX[1] * r + data.stridesX[2] * n + data.stridesX[3] * t + data.stridesX[4] * d + data.stridesX[5] * h + data.stridesX[6] * w + data.stridesX[7] * c];
            float y = Bptr[data.stridesY[0] * s + data.stridesY[1] * r + data.stridesY[2] * n + data.stridesY[3] * t + data.stridesY[4] * d + data.stridesY[5] * h + data.stridesY[6] * w + data.stridesY[7] * c];

            float v = data.alpha * y + x;
            Optr[i] = (float)v;
        }
    }
    internal unsafe struct ElementwiseMulJobHelper
    {
        [ReadOnly] public TensorShape shapeO;
        [ReadOnly] public fixed int stridesX[8];
        [ReadOnly] public fixed int stridesY[8];
        [ReadOnly] public float alpha;
        public JobHandle ScheduleXBO(Tensor X, Tensor B, Tensor O, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            var pinX = Pin(X);
            var pinB = Pin(B);
            var pinO = Pin(O, uploadCache: false);
            bool AHalf = pinX.array.Type == DataType.Half;
            bool WHalf = pinB.array.Type == DataType.Half;
            bool OHalf = pinO.array.Type == DataType.Half;
            UnityEngine.Assertions.Assert.AreEqual(AHalf, OHalf);
            UnityEngine.Assertions.Assert.AreEqual(AHalf, WHalf);
            if (AHalf)
            {
                var job = new ElementwiseMulJob_Full_Half();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else //if (!AHalf)
            {
                var job = new ElementwiseMulJob_Full_Float();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ElementwiseMulJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadOnlyMemResource B { get; set; } float* Bptr => B.ptrfloat;//Always use activation type
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public ElementwiseMulJobHelper data;

        public void Execute(int i)
        {
            int s = 0, r = 0, n = 0, t = 0, d = 0, h = 0, w = 0, c = 0;
            data.shapeO.GetPositionsFromIndex(i, ref s, ref r, ref n, ref t, ref d, ref h, ref w, ref c);

            float x = Xptr[data.stridesX[0] * s + data.stridesX[1] * r + data.stridesX[2] * n + data.stridesX[3] * t + data.stridesX[4] * d + data.stridesX[5] * h + data.stridesX[6] * w + data.stridesX[7] * c];
            float y = Bptr[data.stridesY[0] * s + data.stridesY[1] * r + data.stridesY[2] * n + data.stridesY[3] * t + data.stridesY[4] * d + data.stridesY[5] * h + data.stridesY[6] * w + data.stridesY[7] * c];

            float v = x * y;
            Optr[i] = (float)v;
        }
    }
    internal unsafe struct ElementwiseDivJobHelper
    {
        [ReadOnly] public TensorShape shapeO;
        [ReadOnly] public fixed int stridesX[8];
        [ReadOnly] public fixed int stridesY[8];
        [ReadOnly] public float alpha;
        public JobHandle ScheduleXBO(Tensor X, Tensor B, Tensor O, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            var pinX = Pin(X);
            var pinB = Pin(B);
            var pinO = Pin(O, uploadCache: false);
            bool AHalf = pinX.array.Type == DataType.Half;
            bool WHalf = pinB.array.Type == DataType.Half;
            bool OHalf = pinO.array.Type == DataType.Half;
            UnityEngine.Assertions.Assert.AreEqual(AHalf, OHalf);
            UnityEngine.Assertions.Assert.AreEqual(AHalf, WHalf);
            if (AHalf)
            {
                var job = new ElementwiseDivJob_Full_Half();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else //if (!AHalf)
            {
                var job = new ElementwiseDivJob_Full_Float();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ElementwiseDivJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadOnlyMemResource B { get; set; } float* Bptr => B.ptrfloat;//Always use activation type
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public ElementwiseDivJobHelper data;

        public void Execute(int i)
        {
            int s = 0, r = 0, n = 0, t = 0, d = 0, h = 0, w = 0, c = 0;
            data.shapeO.GetPositionsFromIndex(i, ref s, ref r, ref n, ref t, ref d, ref h, ref w, ref c);

            float x = Xptr[data.stridesX[0] * s + data.stridesX[1] * r + data.stridesX[2] * n + data.stridesX[3] * t + data.stridesX[4] * d + data.stridesX[5] * h + data.stridesX[6] * w + data.stridesX[7] * c];
            float y = Bptr[data.stridesY[0] * s + data.stridesY[1] * r + data.stridesY[2] * n + data.stridesY[3] * t + data.stridesY[4] * d + data.stridesY[5] * h + data.stridesY[6] * w + data.stridesY[7] * c];

            float v = x / y;
            Optr[i] = (float)v;
        }
    }
    internal unsafe struct ElementwiseMinJobHelper
    {
        [ReadOnly] public TensorShape shapeO;
        [ReadOnly] public fixed int stridesX[8];
        [ReadOnly] public fixed int stridesY[8];
        [ReadOnly] public float alpha;
        public JobHandle ScheduleXBO(Tensor X, Tensor B, Tensor O, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            var pinX = Pin(X);
            var pinB = Pin(B);
            var pinO = Pin(O, uploadCache: false);
            bool AHalf = pinX.array.Type == DataType.Half;
            bool WHalf = pinB.array.Type == DataType.Half;
            bool OHalf = pinO.array.Type == DataType.Half;
            UnityEngine.Assertions.Assert.AreEqual(AHalf, OHalf);
            UnityEngine.Assertions.Assert.AreEqual(AHalf, WHalf);
            if (AHalf)
            {
                var job = new ElementwiseMinJob_Full_Half();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else //if (!AHalf)
            {
                var job = new ElementwiseMinJob_Full_Float();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ElementwiseMinJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadOnlyMemResource B { get; set; } float* Bptr => B.ptrfloat;//Always use activation type
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public ElementwiseMinJobHelper data;

        public void Execute(int i)
        {
            int s = 0, r = 0, n = 0, t = 0, d = 0, h = 0, w = 0, c = 0;
            data.shapeO.GetPositionsFromIndex(i, ref s, ref r, ref n, ref t, ref d, ref h, ref w, ref c);

            float x = Xptr[data.stridesX[0] * s + data.stridesX[1] * r + data.stridesX[2] * n + data.stridesX[3] * t + data.stridesX[4] * d + data.stridesX[5] * h + data.stridesX[6] * w + data.stridesX[7] * c];
            float y = Bptr[data.stridesY[0] * s + data.stridesY[1] * r + data.stridesY[2] * n + data.stridesY[3] * t + data.stridesY[4] * d + data.stridesY[5] * h + data.stridesY[6] * w + data.stridesY[7] * c];

            float v = math.min(x , y);
            Optr[i] = (float)v;
        }
    }
    internal unsafe struct ElementwiseMaxJobHelper
    {
        [ReadOnly] public TensorShape shapeO;
        [ReadOnly] public fixed int stridesX[8];
        [ReadOnly] public fixed int stridesY[8];
        [ReadOnly] public float alpha;
        public JobHandle ScheduleXBO(Tensor X, Tensor B, Tensor O, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            var pinX = Pin(X);
            var pinB = Pin(B);
            var pinO = Pin(O, uploadCache: false);
            bool AHalf = pinX.array.Type == DataType.Half;
            bool WHalf = pinB.array.Type == DataType.Half;
            bool OHalf = pinO.array.Type == DataType.Half;
            UnityEngine.Assertions.Assert.AreEqual(AHalf, OHalf);
            UnityEngine.Assertions.Assert.AreEqual(AHalf, WHalf);
            if (AHalf)
            {
                var job = new ElementwiseMaxJob_Full_Half();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else //if (!AHalf)
            {
                var job = new ElementwiseMaxJob_Full_Float();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ElementwiseMaxJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadOnlyMemResource B { get; set; } float* Bptr => B.ptrfloat;//Always use activation type
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public ElementwiseMaxJobHelper data;

        public void Execute(int i)
        {
            int s = 0, r = 0, n = 0, t = 0, d = 0, h = 0, w = 0, c = 0;
            data.shapeO.GetPositionsFromIndex(i, ref s, ref r, ref n, ref t, ref d, ref h, ref w, ref c);

            float x = Xptr[data.stridesX[0] * s + data.stridesX[1] * r + data.stridesX[2] * n + data.stridesX[3] * t + data.stridesX[4] * d + data.stridesX[5] * h + data.stridesX[6] * w + data.stridesX[7] * c];
            float y = Bptr[data.stridesY[0] * s + data.stridesY[1] * r + data.stridesY[2] * n + data.stridesY[3] * t + data.stridesY[4] * d + data.stridesY[5] * h + data.stridesY[6] * w + data.stridesY[7] * c];

            float v = math.max(x , y);
            Optr[i] = (float)v;
        }
    }
    internal unsafe struct ElementwisePowJobHelper
    {
        [ReadOnly] public TensorShape shapeO;
        [ReadOnly] public fixed int stridesX[8];
        [ReadOnly] public fixed int stridesY[8];
        [ReadOnly] public float alpha;
        public JobHandle ScheduleXBO(Tensor X, Tensor B, Tensor O, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            var pinX = Pin(X);
            var pinB = Pin(B);
            var pinO = Pin(O, uploadCache: false);
            bool AHalf = pinX.array.Type == DataType.Half;
            bool WHalf = pinB.array.Type == DataType.Half;
            bool OHalf = pinO.array.Type == DataType.Half;
            UnityEngine.Assertions.Assert.AreEqual(AHalf, OHalf);
            UnityEngine.Assertions.Assert.AreEqual(AHalf, WHalf);
            if (AHalf)
            {
                var job = new ElementwisePowJob_Full_Half();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else //if (!AHalf)
            {
                var job = new ElementwisePowJob_Full_Float();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ElementwisePowJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadOnlyMemResource B { get; set; } float* Bptr => B.ptrfloat;//Always use activation type
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public ElementwisePowJobHelper data;

        public void Execute(int i)
        {
            int s = 0, r = 0, n = 0, t = 0, d = 0, h = 0, w = 0, c = 0;
            data.shapeO.GetPositionsFromIndex(i, ref s, ref r, ref n, ref t, ref d, ref h, ref w, ref c);

            float x = Xptr[data.stridesX[0] * s + data.stridesX[1] * r + data.stridesX[2] * n + data.stridesX[3] * t + data.stridesX[4] * d + data.stridesX[5] * h + data.stridesX[6] * w + data.stridesX[7] * c];
            float y = Bptr[data.stridesY[0] * s + data.stridesY[1] * r + data.stridesY[2] * n + data.stridesY[3] * t + data.stridesY[4] * d + data.stridesY[5] * h + data.stridesY[6] * w + data.stridesY[7] * c];

            float v = math.pow(x, y);
            Optr[i] = (float)v;
        }
    }

    #endregion
    #region Broadcast Jobs declaration for mode: _ActAsFloat_WeightAsHalf

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    unsafe struct VectorBroadcastScaleBiasJob_ActAsFloat_WeightAsHalf : IJobParallelFor, IJobResourceDeclarationXSBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadOnlyMemResource S { get; set; } half* Sptr => S.ptrhalf;
        public ReadOnlyMemResource B { get; set; } half* Bptr => B.ptrhalf;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public VectorBroadcastScaleBiasJobHelper data;

        const int unrollSize = 32;
        public void Execute(int i)
        {
            float* src   = Xptr + i * data.inOutChannels;
            float* dst   = Optr + i * data.inOutChannels;
            half* gamma = Sptr;
            half* beta  = Bptr;

            int j = 0;
            for (; j < data.inOutChannels - unrollSize + 1; j += unrollSize) // unroll of inOutChannels loop
                for (int q = 0; q < unrollSize; q++, src++, dst++, gamma++, beta++)
                    *dst = (float)((*src) * (*gamma) + (*beta) * data.alpha);
            for (; j < data.inOutChannels; j++, src++, dst++, gamma++, beta++) // remainder of inOutChannels loop
                *dst = (float)((*src) * (*gamma) + (*beta) * data.alpha);
        }
    }



    #endregion
    #region Broadcast Jobs declaration for mode: _Full_Half

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    unsafe struct VectorBroadcastScaleBiasJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXSBO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadOnlyMemResource S { get; set; } half* Sptr => S.ptrhalf;
        public ReadOnlyMemResource B { get; set; } half* Bptr => B.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public VectorBroadcastScaleBiasJobHelper data;

        const int unrollSize = 32;
        public void Execute(int i)
        {
            half* src   = Xptr + i * data.inOutChannels;
            half* dst   = Optr + i * data.inOutChannels;
            half* gamma = Sptr;
            half* beta  = Bptr;

            int j = 0;
            for (; j < data.inOutChannels - unrollSize + 1; j += unrollSize) // unroll of inOutChannels loop
                for (int q = 0; q < unrollSize; q++, src++, dst++, gamma++, beta++)
                    *dst = (half)((*src) * (*gamma) + (*beta) * data.alpha);
            for (; j < data.inOutChannels; j++, src++, dst++, gamma++, beta++) // remainder of inOutChannels loop
                *dst = (half)((*src) * (*gamma) + (*beta) * data.alpha);
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ScalarBroadcastAddJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadOnlyMemResource B { get; set; } half* Bptr => B.ptrhalf;//Always use activation type
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public ScalarBroadcastAddJobHelper data;

        public void Execute(int i)
        {
            float v = Bptr[0] * data.alpha + Xptr[i];
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct BroadcastAddJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadOnlyMemResource B { get; set; } half* Bptr => B.ptrhalf;//Always use activation type
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public BroadcastAddJobHelper data;

        public void Execute(int i)
        {
            float v = Bptr[i] * data.alpha + Xptr[i];
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ScalarBroadcastMulJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadOnlyMemResource B { get; set; } half* Bptr => B.ptrhalf;//Always use activation type
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public ScalarBroadcastMulJobHelper data;

        public void Execute(int i)
        {
            float v = Xptr[i] * Bptr[0];
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct BroadcastMulJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadOnlyMemResource B { get; set; } half* Bptr => B.ptrhalf;//Always use activation type
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public BroadcastMulJobHelper data;

        public void Execute(int i)
        {
            float v = Xptr[i] * Bptr[i];
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ScalarBroadcastDivJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadOnlyMemResource B { get; set; } half* Bptr => B.ptrhalf;//Always use activation type
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public ScalarBroadcastDivJobHelper data;

        public void Execute(int i)
        {
            float v = Xptr[i] / Bptr[0];
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct BroadcastDivJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadOnlyMemResource B { get; set; } half* Bptr => B.ptrhalf;//Always use activation type
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public BroadcastDivJobHelper data;

        public void Execute(int i)
        {
            float v = Xptr[i] / Bptr[i];
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ScalarBroadcastMinJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadOnlyMemResource B { get; set; } half* Bptr => B.ptrhalf;//Always use activation type
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public ScalarBroadcastMinJobHelper data;

        public void Execute(int i)
        {
            float v = math.min(Xptr[i], Bptr[0]);
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct BroadcastMinJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadOnlyMemResource B { get; set; } half* Bptr => B.ptrhalf;//Always use activation type
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public BroadcastMinJobHelper data;

        public void Execute(int i)
        {
            float v = math.min(Xptr[i], Bptr[i]);
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ScalarBroadcastMaxJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadOnlyMemResource B { get; set; } half* Bptr => B.ptrhalf;//Always use activation type
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public ScalarBroadcastMaxJobHelper data;

        public void Execute(int i)
        {
            float v = math.max(Xptr[i], Bptr[0]);
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct BroadcastMaxJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadOnlyMemResource B { get; set; } half* Bptr => B.ptrhalf;//Always use activation type
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public BroadcastMaxJobHelper data;

        public void Execute(int i)
        {
            float v = math.max(Xptr[i], Bptr[i]);
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ScalarBroadcastPowJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadOnlyMemResource B { get; set; } half* Bptr => B.ptrhalf;//Always use activation type
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public ScalarBroadcastPowJobHelper data;

        public void Execute(int i)
        {
            float v = math.pow(Xptr[i], Bptr[0]);
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct BroadcastPowJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadOnlyMemResource B { get; set; } half* Bptr => B.ptrhalf;//Always use activation type
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public BroadcastPowJobHelper data;

        public void Execute(int i)
        {
            float v = math.pow(Xptr[i], Bptr[i]);
            Optr[i] = (half)v;
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ElementwiseAddJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadOnlyMemResource B { get; set; } half* Bptr => B.ptrhalf;//Always use activation type
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public ElementwiseAddJobHelper data;

        public void Execute(int i)
        {
            int s = 0, r = 0, n = 0, t = 0, d = 0, h = 0, w = 0, c = 0;
            data.shapeO.GetPositionsFromIndex(i, ref s, ref r, ref n, ref t, ref d, ref h, ref w, ref c);

            float x = Xptr[data.stridesX[0] * s + data.stridesX[1] * r + data.stridesX[2] * n + data.stridesX[3] * t + data.stridesX[4] * d + data.stridesX[5] * h + data.stridesX[6] * w + data.stridesX[7] * c];
            float y = Bptr[data.stridesY[0] * s + data.stridesY[1] * r + data.stridesY[2] * n + data.stridesY[3] * t + data.stridesY[4] * d + data.stridesY[5] * h + data.stridesY[6] * w + data.stridesY[7] * c];

            float v = data.alpha * y + x;
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ElementwiseMulJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadOnlyMemResource B { get; set; } half* Bptr => B.ptrhalf;//Always use activation type
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public ElementwiseMulJobHelper data;

        public void Execute(int i)
        {
            int s = 0, r = 0, n = 0, t = 0, d = 0, h = 0, w = 0, c = 0;
            data.shapeO.GetPositionsFromIndex(i, ref s, ref r, ref n, ref t, ref d, ref h, ref w, ref c);

            float x = Xptr[data.stridesX[0] * s + data.stridesX[1] * r + data.stridesX[2] * n + data.stridesX[3] * t + data.stridesX[4] * d + data.stridesX[5] * h + data.stridesX[6] * w + data.stridesX[7] * c];
            float y = Bptr[data.stridesY[0] * s + data.stridesY[1] * r + data.stridesY[2] * n + data.stridesY[3] * t + data.stridesY[4] * d + data.stridesY[5] * h + data.stridesY[6] * w + data.stridesY[7] * c];

            float v = x * y;
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ElementwiseDivJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadOnlyMemResource B { get; set; } half* Bptr => B.ptrhalf;//Always use activation type
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public ElementwiseDivJobHelper data;

        public void Execute(int i)
        {
            int s = 0, r = 0, n = 0, t = 0, d = 0, h = 0, w = 0, c = 0;
            data.shapeO.GetPositionsFromIndex(i, ref s, ref r, ref n, ref t, ref d, ref h, ref w, ref c);

            float x = Xptr[data.stridesX[0] * s + data.stridesX[1] * r + data.stridesX[2] * n + data.stridesX[3] * t + data.stridesX[4] * d + data.stridesX[5] * h + data.stridesX[6] * w + data.stridesX[7] * c];
            float y = Bptr[data.stridesY[0] * s + data.stridesY[1] * r + data.stridesY[2] * n + data.stridesY[3] * t + data.stridesY[4] * d + data.stridesY[5] * h + data.stridesY[6] * w + data.stridesY[7] * c];

            float v = x / y;
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ElementwiseMinJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadOnlyMemResource B { get; set; } half* Bptr => B.ptrhalf;//Always use activation type
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public ElementwiseMinJobHelper data;

        public void Execute(int i)
        {
            int s = 0, r = 0, n = 0, t = 0, d = 0, h = 0, w = 0, c = 0;
            data.shapeO.GetPositionsFromIndex(i, ref s, ref r, ref n, ref t, ref d, ref h, ref w, ref c);

            float x = Xptr[data.stridesX[0] * s + data.stridesX[1] * r + data.stridesX[2] * n + data.stridesX[3] * t + data.stridesX[4] * d + data.stridesX[5] * h + data.stridesX[6] * w + data.stridesX[7] * c];
            float y = Bptr[data.stridesY[0] * s + data.stridesY[1] * r + data.stridesY[2] * n + data.stridesY[3] * t + data.stridesY[4] * d + data.stridesY[5] * h + data.stridesY[6] * w + data.stridesY[7] * c];

            float v = math.min(x , y);
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ElementwiseMaxJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadOnlyMemResource B { get; set; } half* Bptr => B.ptrhalf;//Always use activation type
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public ElementwiseMaxJobHelper data;

        public void Execute(int i)
        {
            int s = 0, r = 0, n = 0, t = 0, d = 0, h = 0, w = 0, c = 0;
            data.shapeO.GetPositionsFromIndex(i, ref s, ref r, ref n, ref t, ref d, ref h, ref w, ref c);

            float x = Xptr[data.stridesX[0] * s + data.stridesX[1] * r + data.stridesX[2] * n + data.stridesX[3] * t + data.stridesX[4] * d + data.stridesX[5] * h + data.stridesX[6] * w + data.stridesX[7] * c];
            float y = Bptr[data.stridesY[0] * s + data.stridesY[1] * r + data.stridesY[2] * n + data.stridesY[3] * t + data.stridesY[4] * d + data.stridesY[5] * h + data.stridesY[6] * w + data.stridesY[7] * c];

            float v = math.max(x , y);
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ElementwisePowJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadOnlyMemResource B { get; set; } half* Bptr => B.ptrhalf;//Always use activation type
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public ElementwisePowJobHelper data;

        public void Execute(int i)
        {
            int s = 0, r = 0, n = 0, t = 0, d = 0, h = 0, w = 0, c = 0;
            data.shapeO.GetPositionsFromIndex(i, ref s, ref r, ref n, ref t, ref d, ref h, ref w, ref c);

            float x = Xptr[data.stridesX[0] * s + data.stridesX[1] * r + data.stridesX[2] * n + data.stridesX[3] * t + data.stridesX[4] * d + data.stridesX[5] * h + data.stridesX[6] * w + data.stridesX[7] * c];
            float y = Bptr[data.stridesY[0] * s + data.stridesY[1] * r + data.stridesY[2] * n + data.stridesY[3] * t + data.stridesY[4] * d + data.stridesY[5] * h + data.stridesY[6] * w + data.stridesY[7] * c];

            float v = math.pow(x, y);
            Optr[i] = (half)v;
        }
    }

    #endregion
}
}
