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
    #region Activation jobs declaration for mode: _Full_Float

    internal partial struct ReluJobHelper
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
                var job = new ReluJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new ReluJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ReluJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public ReluJobHelper data;

        public void Execute(int i)
        {
            float v = Xptr[i];
            // NOTE: burst-1.2.3 has troubles with Math.Min/Max generating poorly vectorized and branch code
            // Instead Math.Abs based code is used instead. (Math.Abs just flips 1 bit)
            Optr[i] = (float)(0.5f * (v + math.abs(v)));
        }
    }

    internal partial struct Relu6JobHelper
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
                var job = new Relu6Job_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new Relu6Job_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct Relu6Job_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public Relu6JobHelper data;

        public void Execute(int i)
        {
            // f(x) = min(max(x, 0), 6)
            // "Convolutional Deep Belief Networks on CIFAR-10", A Krizhevsky, 2010
            // http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf
            float v = Xptr[i];

            // NOTE: burst-1.2.3 has troubles with Math.Min/Max generating poorly vectorized and branch code
            // Instead Math.Abs based code is used instead. (Math.Abs just flips 1 bit)
            Optr[i] = (float)(0.5f * (-math.abs(v - 6f) + math.abs(v) + 6f));
        }
    }

    internal partial struct LeakyReluJobHelper
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
                var job = new LeakyReluJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new LeakyReluJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct LeakyReluJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public LeakyReluJobHelper data;

        public void Execute(int i)
        {
            float v = Xptr[i];
            // NOTE: burst-1.2.3 has troubles with Math.Min/Max generating poorly vectorized and branch code
            // Instead Math.Abs based code is used instead. (Math.Abs just flips 1 bit)
            Optr[i] = (float)(data.f1 * v + data.f2 * math.abs(v));
        }
    }

    internal partial struct TanhJobHelper
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
                var job = new TanhJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new TanhJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct TanhJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public TanhJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = math.tanh(x);
            Optr[i] = (float)v;
        }
    }
    internal partial struct SoftplusJobHelper
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
                var job = new SoftplusJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new SoftplusJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct SoftplusJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public SoftplusJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = math.log(math.exp(x) + 1f);
            Optr[i] = (float)v;
        }
    }
    internal partial struct SigmoidJobHelper
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
                var job = new SigmoidJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new SigmoidJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct SigmoidJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public SigmoidJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = 1f / (1f + math.exp(-x));
            Optr[i] = (float)v;
        }
    }
    internal partial struct AbsJobHelper
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
                var job = new AbsJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new AbsJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct AbsJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public AbsJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = Math.Abs(x);
            Optr[i] = (float)v;
        }
    }
    internal partial struct NegJobHelper
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
                var job = new NegJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new NegJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct NegJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public NegJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = -x;
            Optr[i] = (float)v;
        }
    }
    internal partial struct CeilJobHelper
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
                var job = new CeilJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new CeilJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct CeilJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public CeilJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = math.ceil(x);
            Optr[i] = (float)v;
        }
    }
    internal partial struct FloorJobHelper
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
                var job = new FloorJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new FloorJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct FloorJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public FloorJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = math.floor(x);
            Optr[i] = (float)v;
        }
    }
    internal partial struct RoundJobHelper
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
                var job = new RoundJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new RoundJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct RoundJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public RoundJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = math.round(x);
            Optr[i] = (float)v;
        }
    }
    internal partial struct ReciprocalJobHelper
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
                var job = new ReciprocalJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new ReciprocalJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ReciprocalJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public ReciprocalJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = 1.0f / x;
            Optr[i] = (float)v;
        }
    }
    internal partial struct ExpJobHelper
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
                var job = new ExpJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new ExpJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ExpJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public ExpJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = math.exp(x);
            Optr[i] = (float)v;
        }
    }
    internal partial struct LogJobHelper
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
                var job = new LogJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new LogJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct LogJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public LogJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = math.log(x);
            Optr[i] = (float)v;
        }
    }
    internal partial struct SqrtJobHelper
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
                var job = new SqrtJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new SqrtJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct SqrtJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public SqrtJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = math.sqrt(x);
            Optr[i] = (float)v;
        }
    }
    internal partial struct AcosJobHelper
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
                var job = new AcosJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new AcosJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct AcosJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public AcosJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = math.acos(x);
            Optr[i] = (float)v;
        }
    }
    internal partial struct AcoshJobHelper
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
                var job = new AcoshJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new AcoshJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct AcoshJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public AcoshJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = math.log( x + math.sqrt(x*x - 1.0f));
            Optr[i] = (float)v;
        }
    }
    internal partial struct AsinJobHelper
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
                var job = new AsinJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new AsinJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct AsinJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public AsinJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = math.asin(x);
            Optr[i] = (float)v;
        }
    }
    internal partial struct AsinhJobHelper
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
                var job = new AsinhJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new AsinhJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct AsinhJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public AsinhJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = math.log( x + math.sqrt(x*x + 1.0f));
            Optr[i] = (float)v;
        }
    }
    internal partial struct AtanJobHelper
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
                var job = new AtanJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new AtanJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct AtanJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public AtanJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = math.atan(x);
            Optr[i] = (float)v;
        }
    }
    internal partial struct AtanhJobHelper
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
                var job = new AtanhJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new AtanhJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct AtanhJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public AtanhJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = 0.5f * math.log((1.0f + x)/(1.0f - x));
            Optr[i] = (float)v;
        }
    }
    internal partial struct CosJobHelper
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
                var job = new CosJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new CosJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct CosJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public CosJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = math.cos(x);
            Optr[i] = (float)v;
        }
    }
    internal partial struct CoshJobHelper
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
                var job = new CoshJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new CoshJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct CoshJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public CoshJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = 0.5f * (math.exp(x) + math.exp(-x));
            Optr[i] = (float)v;
        }
    }
    internal partial struct SinJobHelper
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
                var job = new SinJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new SinJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct SinJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public SinJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = math.sin(x);
            Optr[i] = (float)v;
        }
    }
    internal partial struct SinhJobHelper
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
                var job = new SinhJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new SinhJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct SinhJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public SinhJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = 0.5f * (math.exp(x) - math.exp(-x));
            Optr[i] = (float)v;
        }
    }
    internal partial struct TanJobHelper
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
                var job = new TanJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new TanJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct TanJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public TanJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = math.tan(x);
            Optr[i] = (float)v;
        }
    }

    internal partial struct HardSigmoidJobHelper
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
                var job = new HardSigmoidJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new HardSigmoidJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct HardSigmoidJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public HardSigmoidJobHelper data;

        public void Execute(int i)
        {
            Optr[i] = (float)(math.max(0.0f, math.min(1.0f, data.alpha * Xptr[i] + data.beta)));
        }
    }

    internal partial struct ClipJobHelper
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
                var job = new ClipJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new ClipJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ClipJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public ClipJobHelper data;

        public void Execute(int i)
        {
            Optr[i] = (float)(math.clamp(Xptr[i], data.min, data.max));
        }
    }

    internal partial struct PowJobHelper
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
                var job = new PowJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new PowJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct PowJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public PowJobHelper data;

        public void Execute(int i)
        {
            Optr[i] = (float)(math.pow(Xptr[i], data.alpha));
        }
    }

    internal partial struct ErfJobHelper
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
                var job = new ErfJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new ErfJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ErfJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public ErfJobHelper data;

        public void Execute(int i)
        {
            float v = Xptr[i];

            // Abramowitz/Stegun approximations
            // erf(x) = -erf(-x)
            float x = math.abs(v);

            float p = 0.3275911f;
            float a1 = 0.254829592f; float a2 = -0.284496736f; float a3 = 1.421413741f;
            float a4 = -1.453152027f; float a5 = 1.061405429f;

            float t = 1.0f / (1.0f + p * x);
            float t2 = t * t;
            float t3 = t2 * t;
            float t4 = t3 * t;
            float t5 = t4 * t;

            Optr[i] = (float)(math.sign(v) * (1 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * math.exp(-x * x)));
        }
    }

    internal partial struct EluJobHelper
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
                var job = new EluJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new EluJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct EluJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public EluJobHelper data;

        public void Execute(int i)
        {
            // f(x) = alpha * (exp(x) - 1.) for x < 0, f(x) = x for x >= 0
            // "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)", DA Clevert, 2015
            // https://arxiv.org/abs/1511.07289
            float v = Xptr[i];
            if (v <= 0)
                v = data.alpha * (math.exp(v) - 1f);
            Optr[i] = (float)(v);
        }
    }

    internal partial struct SeluJobHelper
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
                var job = new SeluJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new SeluJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct SeluJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public SeluJobHelper data;

        public void Execute(int i)
        {
            // f(x) = gamma * (alpha * e^x - alpha) for x <= 0, f(x) = gamma * x for x > 0
            float v = Xptr[i];
            if (v <= 0.0f)
                v = data.gamma * (data.alpha * math.exp(v) - data.alpha);
            else
                v = data.gamma * v;
            Optr[i] = (float)(v);
        }
    }

    internal partial struct PReluJobHelper
    {
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
                var job = new PReluJob_Full_Half();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else //if (!AHalf)
            {
                var job = new PReluJob_Full_Float();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct PReluJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadOnlyMemResource B { get; set; } float* Bptr => B.ptrfloat;//Always use activation type
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public PReluJobHelper data;

        const int unrollSize = 32;
        public void Execute(int i)
        {
            float* src   = Xptr + i * data.inOutChannels;
            float* dst   = Optr + i * data.inOutChannels;
            float* gamma = Bptr + i * data.inOutChannels * data.isGammaAVector;

            int j = 0;
            for (; j < data.inOutChannels - unrollSize + 1; j += unrollSize) // unroll of inOutChannels loop
                for (int q = 0; q < unrollSize; q++, src++, dst++, gamma+=data.isGammaAVector)
                    *dst = (float)(PRelu(*src, *gamma));
            for (; j < data.inOutChannels; j++, src++, dst++, gamma+=data.isGammaAVector) // remainder of inOutChannels loop
                *dst = (float)(PRelu(*src, *gamma));
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

    internal partial struct SwishJobHelper
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
                var job = new SwishJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new SwishJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct SwishJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public SwishJobHelper data;

        public void Execute(int i)
        {
            // f(x) = sigmoid(x) * x = x / (1 + exp(-x))
            // "Searching for Activation Functions". P Ramachandran, 2017
            // https://arxiv.org/abs/1710.05941
            float v = Xptr[i];
            v = v / (1f + math.exp(-v));
            Optr[i] = (float)(v);
        }
    }

    #endregion
    #region Activation jobs declaration for mode: _Full_Half

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ReluJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public ReluJobHelper data;

        public void Execute(int i)
        {
            float v = Xptr[i];
            // NOTE: burst-1.2.3 has troubles with Math.Min/Max generating poorly vectorized and branch code
            // Instead Math.Abs based code is used instead. (Math.Abs just flips 1 bit)
            Optr[i] = (half)(0.5f * (v + math.abs(v)));
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct Relu6Job_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public Relu6JobHelper data;

        public void Execute(int i)
        {
            // f(x) = min(max(x, 0), 6)
            // "Convolutional Deep Belief Networks on CIFAR-10", A Krizhevsky, 2010
            // http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf
            float v = Xptr[i];

            // NOTE: burst-1.2.3 has troubles with Math.Min/Max generating poorly vectorized and branch code
            // Instead Math.Abs based code is used instead. (Math.Abs just flips 1 bit)
            Optr[i] = (half)(0.5f * (-math.abs(v - 6f) + math.abs(v) + 6f));
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct LeakyReluJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public LeakyReluJobHelper data;

        public void Execute(int i)
        {
            float v = Xptr[i];
            // NOTE: burst-1.2.3 has troubles with Math.Min/Max generating poorly vectorized and branch code
            // Instead Math.Abs based code is used instead. (Math.Abs just flips 1 bit)
            Optr[i] = (half)(data.f1 * v + data.f2 * math.abs(v));
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct TanhJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public TanhJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = math.tanh(x);
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct SoftplusJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public SoftplusJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = math.log(math.exp(x) + 1f);
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct SigmoidJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public SigmoidJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = 1f / (1f + math.exp(-x));
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct AbsJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public AbsJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = Math.Abs(x);
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct NegJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public NegJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = -x;
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct CeilJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public CeilJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = math.ceil(x);
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct FloorJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public FloorJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = math.floor(x);
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct RoundJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public RoundJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = math.round(x);
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ReciprocalJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public ReciprocalJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = 1.0f / x;
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ExpJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public ExpJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = math.exp(x);
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct LogJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public LogJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = math.log(x);
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct SqrtJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public SqrtJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = math.sqrt(x);
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct AcosJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public AcosJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = math.acos(x);
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct AcoshJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public AcoshJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = math.log( x + math.sqrt(x*x - 1.0f));
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct AsinJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public AsinJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = math.asin(x);
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct AsinhJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public AsinhJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = math.log( x + math.sqrt(x*x + 1.0f));
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct AtanJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public AtanJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = math.atan(x);
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct AtanhJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public AtanhJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = 0.5f * math.log((1.0f + x)/(1.0f - x));
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct CosJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public CosJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = math.cos(x);
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct CoshJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public CoshJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = 0.5f * (math.exp(x) + math.exp(-x));
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct SinJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public SinJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = math.sin(x);
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct SinhJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public SinhJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = 0.5f * (math.exp(x) - math.exp(-x));
            Optr[i] = (half)v;
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct TanJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public TanJobHelper data;

        public void Execute(int i)
        {
            float x = Xptr[i];
            float v = math.tan(x);
            Optr[i] = (half)v;
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct HardSigmoidJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public HardSigmoidJobHelper data;

        public void Execute(int i)
        {
            Optr[i] = (half)(math.max(0.0f, math.min(1.0f, data.alpha * Xptr[i] + data.beta)));
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ClipJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public ClipJobHelper data;

        public void Execute(int i)
        {
            Optr[i] = (half)(math.clamp(Xptr[i], data.min, data.max));
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct PowJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public PowJobHelper data;

        public void Execute(int i)
        {
            Optr[i] = (half)(math.pow(Xptr[i], data.alpha));
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct ErfJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public ErfJobHelper data;

        public void Execute(int i)
        {
            float v = Xptr[i];

            // Abramowitz/Stegun approximations
            // erf(x) = -erf(-x)
            float x = math.abs(v);

            float p = 0.3275911f;
            float a1 = 0.254829592f; float a2 = -0.284496736f; float a3 = 1.421413741f;
            float a4 = -1.453152027f; float a5 = 1.061405429f;

            float t = 1.0f / (1.0f + p * x);
            float t2 = t * t;
            float t3 = t2 * t;
            float t4 = t3 * t;
            float t5 = t4 * t;

            Optr[i] = (half)(math.sign(v) * (1 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * math.exp(-x * x)));
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct EluJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public EluJobHelper data;

        public void Execute(int i)
        {
            // f(x) = alpha * (exp(x) - 1.) for x < 0, f(x) = x for x >= 0
            // "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)", DA Clevert, 2015
            // https://arxiv.org/abs/1511.07289
            float v = Xptr[i];
            if (v <= 0)
                v = data.alpha * (math.exp(v) - 1f);
            Optr[i] = (half)(v);
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct SeluJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public SeluJobHelper data;

        public void Execute(int i)
        {
            // f(x) = gamma * (alpha * e^x - alpha) for x <= 0, f(x) = gamma * x for x > 0
            float v = Xptr[i];
            if (v <= 0.0f)
                v = data.gamma * (data.alpha * math.exp(v) - data.alpha);
            else
                v = data.gamma * v;
            Optr[i] = (half)(v);
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct PReluJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadOnlyMemResource B { get; set; } half* Bptr => B.ptrhalf;//Always use activation type
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public PReluJobHelper data;

        const int unrollSize = 32;
        public void Execute(int i)
        {
            half* src   = Xptr + i * data.inOutChannels;
            half* dst   = Optr + i * data.inOutChannels;
            half* gamma = Bptr + i * data.inOutChannels * data.isGammaAVector;

            int j = 0;
            for (; j < data.inOutChannels - unrollSize + 1; j += unrollSize) // unroll of inOutChannels loop
                for (int q = 0; q < unrollSize; q++, src++, dst++, gamma+=data.isGammaAVector)
                    *dst = (half)(PRelu(*src, *gamma));
            for (; j < data.inOutChannels; j++, src++, dst++, gamma+=data.isGammaAVector) // remainder of inOutChannels loop
                *dst = (half)(PRelu(*src, *gamma));
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

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct SwishJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public SwishJobHelper data;

        public void Execute(int i)
        {
            // f(x) = sigmoid(x) * x = x / (1 + exp(-x))
            // "Searching for Activation Functions". P Ramachandran, 2017
            // https://arxiv.org/abs/1710.05941
            float v = Xptr[i];
            v = v / (1f + math.exp(-v));
            Optr[i] = (half)(v);
        }
    }

    #endregion
}
}
