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
    #region Other jobs declaration for mode: _Full_Float

    internal partial struct CopyJobHelper
    {
        public JobHandle ScheduleXO(Tensor X, Tensor O, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            var pinX = Pin(X);
            var pinO = Pin(O, uploadCache: false);
            bool AHalf = pinX.array.Type == DataType.Half;
            bool OHalf = pinO.array.Type == DataType.Half;
            UnityEngine.Assertions.Assert.AreEqual(AHalf, OHalf);
            if (AHalf)
            {
                var job = new CopyJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, fencingMode);
            }
            else
            {
                var job = new CopyJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct CopyJob_Full_Float : IJob, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public CopyJobHelper data;

        public void Execute()
        {
            UnsafeUtility.MemCpy(destination: Optr, source: Xptr, size: data.length * sizeof(float));
        }
    }

    internal partial struct CopyStrideJobHelper
    {
        public JobHandle ScheduleXO(BurstTensorData pinX, int offsetX, BurstTensorData pinO, int offsetY, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            bool AHalf = pinX.array.Type == DataType.Half;
            bool OHalf = pinO.array.Type == DataType.Half;
            UnityEngine.Assertions.Assert.AreEqual(AHalf, OHalf);
            if (AHalf)
            {
                var job = new CopyStrideJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, offsetX, pinO, offsetY, fencingMode);
            }
            else
            {
                var job = new CopyStrideJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, offsetX, pinO, offsetY, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct CopyStrideJob_Full_Float : IJob, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public CopyStrideJobHelper data;

        public void Execute()
        {
            UnsafeUtility.MemCpyStride(destination: Optr, destinationStride: data.OStride * sizeof(float),
                                       source: Xptr, sourceStride: data.XStride * sizeof(float),
                                       elementSize: data.length * sizeof(float), count: data.count);
        }
    }

    internal partial struct GenericSliceJobHelper
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
                var job = new GenericSliceJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new GenericSliceJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct GenericSliceJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public GenericSliceJobHelper data;

        public void Execute(int threadIndex)
        {
            int indexO = threadIndex * data.shapeO.channels;
            int s = 0, r = 0, n = 0, t = 0;
            int d = 0, h = 0, w = 0, c = 0;
            data.shapeO.GetPositionsFromIndex(indexO, ref s, ref r, ref n, ref t, ref d, ref h, ref w, ref c);
            s = data.startS + s * data.strideS;
            r = data.startR + r * data.strideR;
            n = data.startN + n * data.strideN;
            t = data.startT + t * data.strideT;
            d = data.startD + d * data.strideD;
            h = data.startH + h * data.strideH;
            w = data.startW + w * data.strideW;
            c = data.startC + c * data.strideC;
            int indexX = data.shapeX.Index(s, r, n, t, d, h, w, c);
            UnsafeUtility.MemCpy(destination: Optr+indexO, source: Xptr+indexX, size: data.shapeO.channels * sizeof(float));
        }
    }

    internal partial struct GenericStridedSliceJobHelper
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
                var job = new GenericStridedSliceJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new GenericStridedSliceJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct GenericStridedSliceJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public GenericStridedSliceJobHelper data;

        public void Execute(int i)
        {
            int s = 0, r = 0, n = 0, t = 0;
            int d = 0, h = 0, w = 0, c = 0;
            data.shapeO.GetPositionsFromIndex(i, ref s, ref r, ref n, ref t, ref d, ref h, ref w, ref c);
            s = data.startS + s * data.strideS;
            r = data.startR + r * data.strideR;
            n = data.startN + n * data.strideN;
            t = data.startT + t * data.strideT;
            d = data.startD + d * data.strideD;
            h = data.startH + h * data.strideH;
            w = data.startW + w * data.strideW;
            c = data.startC + c * data.strideC;
            Optr[i] = (float)(Xptr[data.shapeX.Index(s, r, n, t, d, h, w, c)]);
        }
    }

    internal partial struct Border2DJobHelper
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
                var job = new Border2DJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new Border2DJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct Border2DJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public Border2DJobHelper data;

        public void Execute(int i)
        {
            int n = 0, h = 0, w = 0, c = 0;
            data.shapeO.GetPositionsFromIndex(i, ref n, ref h, ref w, ref c);

            int readX = w - data.PadWidth;
            int readY = h - data.PadHeight;
            int readC = c - data.PadChannels;

            float v;
            if (readX < 0 || readX >= data.CroppedWidth ||
                readY < 0 || readY >= data.CroppedHeight ||
			    readC < 0 || readC >= data.CroppedChannels)
            {
                v = data.Beta;
            }
            else
            {
                v = Xptr[data.shapeX.Index(n, readY, readX, readC)];
            }

            Optr[i] = (float)(v);
        }
    }

    internal partial struct TransposeJobHelper
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
                var job = new TransposeJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new TransposeJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct TransposeJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public TransposeJobHelper data;

        public void Execute(int i)
        {
            int s = 0, r = 0, n = 0, t = 0, d = 0, h = 0, w = 0, c = 0;
            data.shapeX.GetPositionsFromIndex(i, ref s, ref r, ref n, ref t, ref d, ref h, ref w, ref c);

            int* index = stackalloc int[8];
            index[0] = s; index[1] = r; index[2] = n; index[3] = t; index[4] = d; index[5] = h; index[6] = w; index[7] = c;

            int indexO = data.shapeO.Index(index[data.permutations[0]],
                                           index[data.permutations[1]],
                                           index[data.permutations[2]],
                                           index[data.permutations[3]],
                                           index[data.permutations[4]],
                                           index[data.permutations[5]],
                                           index[data.permutations[6]],
                                           index[data.permutations[7]]);
            Optr[indexO] = (float)(Xptr[i]);
        }
    }

    internal partial struct Pad2DEdgeJobHelper
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
                var job = new Pad2DEdgeJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new Pad2DEdgeJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct Pad2DEdgeJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public Pad2DEdgeJobHelper data;

        public void Execute(int i)
        {
            int n = 0, h = 0, w = 0, c = 0;
            data.shapeO.GetPositionsFromIndex(i, ref n, ref h, ref w, ref c);

            int readX = w - data.PadWidth;
            int readY = h - data.PadHeight;
	        int readC = c - data.PadChannels;

            readX = math.max(readX, 0);
            readY = math.max(readY, 0);
            readC = math.max(readC, 0);
            readX = math.min(readX, data.shapeX.width - 1);
            readY = math.min(readY, data.shapeX.height - 1);
            readC = math.min(readC, data.shapeX.channels- 1);

            Optr[i] = (float)(Xptr[data.shapeX.Index(n, readY, readX, readC)]);
        }
    }

    internal partial struct Pad2DReflectJobHelper
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
                var job = new Pad2DReflectJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new Pad2DReflectJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct Pad2DReflectJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public Pad2DReflectJobHelper data;

        public void Execute(int i)
        {
            int n = 0, h = 0, w = 0, c = 0;
            data.shapeO.GetPositionsFromIndex(i, ref n, ref h, ref w, ref c);

            int readX = w - data.PadWidth;
            int readY = h - data.PadHeight;
	        int readC = c - data.PadChannels;

            int lastXIndex = data.shapeX.width - 1;
            int lastYIndex = data.shapeX.height - 1;
	        int lastCIndex = data.shapeX.channels - 1;

            //x reflect indexing
            if (readX < 0)
                readX = -readX;
            else if (readX > lastXIndex)
                readX = lastXIndex - (readX - lastXIndex);

            //y reflect indexing
            if (readY < 0)
                readY = -readY;
            else if (readY > lastYIndex)
                readY = lastYIndex - (readY - lastYIndex);

	        //c reflect indexing
	        if (readC < 0)
		        readC = -readC;
	        else if (readC > lastCIndex)
		        readC = lastCIndex - (readC - lastCIndex);

            readX = math.max(readX, 0);
            readY = math.max(readY, 0);
            readC = math.max(readC, 0);
            readX = math.min(readX, data.shapeX.width - 1);
            readY = math.min(readY, data.shapeX.height - 1);
            readC = math.min(readC, data.shapeX.channels- 1);

            Optr[i] = Xptr[data.shapeX.Index(n, readY, readX, readC)];
        }
    }

    internal partial struct Pad2DSymmetricJobHelper
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
                var job = new Pad2DSymmetricJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new Pad2DSymmetricJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct Pad2DSymmetricJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public Pad2DSymmetricJobHelper data;

        public void Execute(int i)
        {
            int n = 0, h = 0, w = 0, c = 0;
            data.shapeO.GetPositionsFromIndex(i, ref n, ref h, ref w, ref c);

            int readX = w - data.PadWidth;
            int readY = h - data.PadHeight;
	        int readC = c - data.PadChannels;

            int lastXIndex = data.shapeX.width - 1;
            int lastYIndex = data.shapeX.height - 1;
	        int lastCIndex = data.shapeX.channels - 1;

            //x symmetric indexing
            if (readX < 0)
                readX = -readX - 1;
            else if (readX > lastXIndex)
                readX = lastXIndex - (readX - lastXIndex) + 1;

            //y symmetric indexing
            if (readY < 0)
                readY = -readY - 1;
            else if (readY > lastYIndex)
                readY = lastYIndex - (readY - lastYIndex) + 1;

	        //c symmetric indexing
	        if (readC < 0)
		        readC = -readC - 1;
	        else if (readC > lastCIndex)
		        readC = lastCIndex - (readC - lastCIndex) + 1;

            readX = math.max(readX, 0);
            readY = math.max(readY, 0);
            readC = math.max(readC, 0);
            readX = math.min(readX, data.shapeX.width - 1);
            readY = math.min(readY, data.shapeX.height - 1);
            readC = math.min(readC, data.shapeX.channels- 1);

            Optr[i] = (float)(Xptr[data.shapeX.Index(n, readY, readX, readC)]);
        }
    }

    internal partial struct TileJobHelper
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
                var job = new TileJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new TileJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct TileJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public TileJobHelper data;

        public void Execute(int i)
        {
            int s = 0, r = 0, n = 0, t = 0, d = 0, h = 0, w = 0, c = 0;
            data.shapeO.GetPositionsFromIndex(i, ref s, ref r, ref n, ref t, ref d, ref h, ref w, ref c);

            s = s % data.shapeX[0];
            r = r % data.shapeX[1];
            n = n % data.shapeX[2];
            t = t % data.shapeX[3];
            d = d % data.shapeX[4];
            h = h % data.shapeX[5];
            w = w % data.shapeX[6];
            c = c % data.shapeX[7];

            float x = Xptr[data.shapeX.Index(s, r, n, t, d, h, w, c)];
            Optr[i] = (float)(x);
        }
    }

    internal partial struct GatherJobHelper
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
                var job = new GatherJob_Full_Half();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else //if (!AHalf)
            {
                var job = new GatherJob_Full_Float();
                job.data = this;
                return job.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct GatherJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadOnlyMemResource B { get; set; } float* Bptr => B.ptrfloat;//Always use activation type
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public GatherJobHelper data;

        public void Execute(int i)
        {
            int s = 0, r = 0, n = 0, t = 0, d = 0, h = 0, w = 0, c = 0;
            data.shapeO.GetPositionsFromIndex(i, ref s, ref r, ref n, ref t, ref d, ref h, ref w, ref c);

            int d0 = (data.axis == 0) ? (int) Bptr[s] : s;
            int d1 = (data.axis == 1) ? (int) Bptr[r] : r;
            int d2 = (data.axis == 2) ? (int) Bptr[n] : n;
            int d3 = (data.axis == 3) ? (int) Bptr[t] : t;
            int d4 = (data.axis == 4) ? (int) Bptr[d] : d;
            int d5 = (data.axis == 5) ? (int) Bptr[h] : h;
            int d6 = (data.axis == 6) ? (int) Bptr[w] : w;
            int d7 = (data.axis == 7) ? (int) Bptr[c] : c;

            Optr[i] = (float)(Xptr[data.shapeX.Index(d0, d1, d2, d3, d4, d5, d6, d7)]);
        }
    }

    internal partial struct OneHotJobHelper
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
                var job = new OneHotJob_Full_Half();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new OneHotJob_Full_Float();
                job.data = this;
                return job.ScheduleXO(pinX, pinO, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct OneHotJob_Full_Float : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } float* Xptr => X.ptrfloat;
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public OneHotJobHelper data;

        public void Execute(int idx)
        {
            // rank1: X = n,_,_,_
            // rank2: X = n,_,_,c
            // rank3: X = n,_,w,c

            if (data.inputRank == 1) // TensorShape(X.flatHeight, depth)
            {
                int j = idx % data.depth;
                int n = (idx / data.depth) % data.shapeX.flatHeight;

                int index = (int)Xptr[n];
                float v = (j == index) ? data.onValue: data.offValue;
                Optr[idx] = (float)(v);
            }
            else if (data.inputRank == 2) // TensorShape(X.flatHeight, 1, depth, X.channels));
            {
                int i = idx % data.shapeX.channels;
                int j = (idx / data.shapeX.channels) % data.depth;
                int n = ((idx / data.shapeX.channels) / data.depth) % data.shapeX.flatHeight;

                int index = (int)Xptr[data.shapeX.Index(n, i)];
                float v = (j == index) ? data.onValue: data.offValue;
                Optr[idx] = (float)(v);
            }
            else // TensorShape(X.batch, X.width, depth, X.channels))
            {
                int i = idx % data.shapeX.channels;
                int j = (idx / data.shapeX.channels) % data.depth;
                int k = ((idx / data.shapeX.channels) / data.depth) % data.shapeX.width;
                int n = (((idx / data.shapeX.channels) / data.depth) / data.shapeX.width) % data.shapeX.batch;

                int index = (int)Xptr[data.shapeX.Index(n, 0, k, i)];
                float v = (j == index) ? data.onValue: data.offValue;
                Optr[idx] = (float)(v);
            }
        }
    }

    internal partial struct RandomNormalJobHelper
    {
        public JobHandle ScheduleO(BurstTensorData pinO, int offset, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            bool OHalf = pinO.array.Type == DataType.Half;
            if (OHalf)
            {
                var job = new RandomNormalJob_Full_Half();
                job.data = this;
                return job.ScheduleO(pinO, offset, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new RandomNormalJob_Full_Float();
                job.data = this;
                return job.ScheduleO(pinO, offset, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct RandomNormalJob_Full_Float : IJobParallelFor, IJobResourceDeclarationO
    {
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public RandomNormalJobHelper data;

        float Gaussian(float mean, float stdDev)
        {
            float u, v, s;
            do {
                u = data.rng.NextFloat() * 2 - 1;
                v = data.rng.NextFloat() * 2 - 1;
                s = u * u + v * v;
            } while (s >= 1 || s == 0);
            float mul = Mathf.Sqrt(-2.0f * Mathf.Log(s) / s);
            return mean + stdDev * u * mul;
        }

        public void Execute(int i)
        {
            Optr[i] = (float)(Gaussian(data.mean, data.scale));
        }
    }

    internal partial struct RandomUniformJobHelper
    {
        public JobHandle ScheduleO(BurstTensorData pinO, int offset, int arrayLength, int innerBatchCount, FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            bool OHalf = pinO.array.Type == DataType.Half;
            if (OHalf)
            {
                var job = new RandomUniformJob_Full_Half();
                job.data = this;
                return job.ScheduleO(pinO, offset, arrayLength, innerBatchCount, fencingMode);
            }
            else
            {
                var job = new RandomUniformJob_Full_Float();
                job.data = this;
                return job.ScheduleO(pinO, offset, arrayLength, innerBatchCount, fencingMode);
            }
        }
    }
    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct RandomUniformJob_Full_Float : IJobParallelFor, IJobResourceDeclarationO
    {
        public ReadWriteMemResource O { get; set; } float* Optr => O.ptrfloat;
        public RandomUniformJobHelper data;

        public void Execute(int i)
        {
            float v = data.mean + data.scale * data.rng.NextFloat();
            Optr[i] = (float)(v);
        }
    }

    #endregion
    #region Other jobs declaration for mode: _ActAsFloat_WeightAsHalf















    #endregion
    #region Other jobs declaration for mode: _Full_Half

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct CopyJob_Full_Half : IJob, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public CopyJobHelper data;

        public void Execute()
        {
            UnsafeUtility.MemCpy(destination: Optr, source: Xptr, size: data.length * sizeof(half));
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct CopyStrideJob_Full_Half : IJob, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public CopyStrideJobHelper data;

        public void Execute()
        {
            UnsafeUtility.MemCpyStride(destination: Optr, destinationStride: data.OStride * sizeof(half),
                                       source: Xptr, sourceStride: data.XStride * sizeof(half),
                                       elementSize: data.length * sizeof(half), count: data.count);
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct GenericSliceJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public GenericSliceJobHelper data;

        public void Execute(int threadIndex)
        {
            int indexO = threadIndex * data.shapeO.channels;
            int s = 0, r = 0, n = 0, t = 0;
            int d = 0, h = 0, w = 0, c = 0;
            data.shapeO.GetPositionsFromIndex(indexO, ref s, ref r, ref n, ref t, ref d, ref h, ref w, ref c);
            s = data.startS + s * data.strideS;
            r = data.startR + r * data.strideR;
            n = data.startN + n * data.strideN;
            t = data.startT + t * data.strideT;
            d = data.startD + d * data.strideD;
            h = data.startH + h * data.strideH;
            w = data.startW + w * data.strideW;
            c = data.startC + c * data.strideC;
            int indexX = data.shapeX.Index(s, r, n, t, d, h, w, c);
            UnsafeUtility.MemCpy(destination: Optr+indexO, source: Xptr+indexX, size: data.shapeO.channels * sizeof(half));
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct GenericStridedSliceJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public GenericStridedSliceJobHelper data;

        public void Execute(int i)
        {
            int s = 0, r = 0, n = 0, t = 0;
            int d = 0, h = 0, w = 0, c = 0;
            data.shapeO.GetPositionsFromIndex(i, ref s, ref r, ref n, ref t, ref d, ref h, ref w, ref c);
            s = data.startS + s * data.strideS;
            r = data.startR + r * data.strideR;
            n = data.startN + n * data.strideN;
            t = data.startT + t * data.strideT;
            d = data.startD + d * data.strideD;
            h = data.startH + h * data.strideH;
            w = data.startW + w * data.strideW;
            c = data.startC + c * data.strideC;
            Optr[i] = (half)(Xptr[data.shapeX.Index(s, r, n, t, d, h, w, c)]);
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct Border2DJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public Border2DJobHelper data;

        public void Execute(int i)
        {
            int n = 0, h = 0, w = 0, c = 0;
            data.shapeO.GetPositionsFromIndex(i, ref n, ref h, ref w, ref c);

            int readX = w - data.PadWidth;
            int readY = h - data.PadHeight;
            int readC = c - data.PadChannels;

            float v;
            if (readX < 0 || readX >= data.CroppedWidth ||
                readY < 0 || readY >= data.CroppedHeight ||
			    readC < 0 || readC >= data.CroppedChannels)
            {
                v = data.Beta;
            }
            else
            {
                v = Xptr[data.shapeX.Index(n, readY, readX, readC)];
            }

            Optr[i] = (half)(v);
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct TransposeJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public TransposeJobHelper data;

        public void Execute(int i)
        {
            int s = 0, r = 0, n = 0, t = 0, d = 0, h = 0, w = 0, c = 0;
            data.shapeX.GetPositionsFromIndex(i, ref s, ref r, ref n, ref t, ref d, ref h, ref w, ref c);

            int* index = stackalloc int[8];
            index[0] = s; index[1] = r; index[2] = n; index[3] = t; index[4] = d; index[5] = h; index[6] = w; index[7] = c;

            int indexO = data.shapeO.Index(index[data.permutations[0]],
                                           index[data.permutations[1]],
                                           index[data.permutations[2]],
                                           index[data.permutations[3]],
                                           index[data.permutations[4]],
                                           index[data.permutations[5]],
                                           index[data.permutations[6]],
                                           index[data.permutations[7]]);
            Optr[indexO] = (half)(Xptr[i]);
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct Pad2DEdgeJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public Pad2DEdgeJobHelper data;

        public void Execute(int i)
        {
            int n = 0, h = 0, w = 0, c = 0;
            data.shapeO.GetPositionsFromIndex(i, ref n, ref h, ref w, ref c);

            int readX = w - data.PadWidth;
            int readY = h - data.PadHeight;
	        int readC = c - data.PadChannels;

            readX = math.max(readX, 0);
            readY = math.max(readY, 0);
            readC = math.max(readC, 0);
            readX = math.min(readX, data.shapeX.width - 1);
            readY = math.min(readY, data.shapeX.height - 1);
            readC = math.min(readC, data.shapeX.channels- 1);

            Optr[i] = (half)(Xptr[data.shapeX.Index(n, readY, readX, readC)]);
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct Pad2DReflectJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public Pad2DReflectJobHelper data;

        public void Execute(int i)
        {
            int n = 0, h = 0, w = 0, c = 0;
            data.shapeO.GetPositionsFromIndex(i, ref n, ref h, ref w, ref c);

            int readX = w - data.PadWidth;
            int readY = h - data.PadHeight;
	        int readC = c - data.PadChannels;

            int lastXIndex = data.shapeX.width - 1;
            int lastYIndex = data.shapeX.height - 1;
	        int lastCIndex = data.shapeX.channels - 1;

            //x reflect indexing
            if (readX < 0)
                readX = -readX;
            else if (readX > lastXIndex)
                readX = lastXIndex - (readX - lastXIndex);

            //y reflect indexing
            if (readY < 0)
                readY = -readY;
            else if (readY > lastYIndex)
                readY = lastYIndex - (readY - lastYIndex);

	        //c reflect indexing
	        if (readC < 0)
		        readC = -readC;
	        else if (readC > lastCIndex)
		        readC = lastCIndex - (readC - lastCIndex);

            readX = math.max(readX, 0);
            readY = math.max(readY, 0);
            readC = math.max(readC, 0);
            readX = math.min(readX, data.shapeX.width - 1);
            readY = math.min(readY, data.shapeX.height - 1);
            readC = math.min(readC, data.shapeX.channels- 1);

            Optr[i] = Xptr[data.shapeX.Index(n, readY, readX, readC)];
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct Pad2DSymmetricJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public Pad2DSymmetricJobHelper data;

        public void Execute(int i)
        {
            int n = 0, h = 0, w = 0, c = 0;
            data.shapeO.GetPositionsFromIndex(i, ref n, ref h, ref w, ref c);

            int readX = w - data.PadWidth;
            int readY = h - data.PadHeight;
	        int readC = c - data.PadChannels;

            int lastXIndex = data.shapeX.width - 1;
            int lastYIndex = data.shapeX.height - 1;
	        int lastCIndex = data.shapeX.channels - 1;

            //x symmetric indexing
            if (readX < 0)
                readX = -readX - 1;
            else if (readX > lastXIndex)
                readX = lastXIndex - (readX - lastXIndex) + 1;

            //y symmetric indexing
            if (readY < 0)
                readY = -readY - 1;
            else if (readY > lastYIndex)
                readY = lastYIndex - (readY - lastYIndex) + 1;

	        //c symmetric indexing
	        if (readC < 0)
		        readC = -readC - 1;
	        else if (readC > lastCIndex)
		        readC = lastCIndex - (readC - lastCIndex) + 1;

            readX = math.max(readX, 0);
            readY = math.max(readY, 0);
            readC = math.max(readC, 0);
            readX = math.min(readX, data.shapeX.width - 1);
            readY = math.min(readY, data.shapeX.height - 1);
            readC = math.min(readC, data.shapeX.channels- 1);

            Optr[i] = (half)(Xptr[data.shapeX.Index(n, readY, readX, readC)]);
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct TileJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public TileJobHelper data;

        public void Execute(int i)
        {
            int s = 0, r = 0, n = 0, t = 0, d = 0, h = 0, w = 0, c = 0;
            data.shapeO.GetPositionsFromIndex(i, ref s, ref r, ref n, ref t, ref d, ref h, ref w, ref c);

            s = s % data.shapeX[0];
            r = r % data.shapeX[1];
            n = n % data.shapeX[2];
            t = t % data.shapeX[3];
            d = d % data.shapeX[4];
            h = h % data.shapeX[5];
            w = w % data.shapeX[6];
            c = c % data.shapeX[7];

            float x = Xptr[data.shapeX.Index(s, r, n, t, d, h, w, c)];
            Optr[i] = (half)(x);
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct GatherJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXBO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadOnlyMemResource B { get; set; } half* Bptr => B.ptrhalf;//Always use activation type
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public GatherJobHelper data;

        public void Execute(int i)
        {
            int s = 0, r = 0, n = 0, t = 0, d = 0, h = 0, w = 0, c = 0;
            data.shapeO.GetPositionsFromIndex(i, ref s, ref r, ref n, ref t, ref d, ref h, ref w, ref c);

            int d0 = (data.axis == 0) ? (int) Bptr[s] : s;
            int d1 = (data.axis == 1) ? (int) Bptr[r] : r;
            int d2 = (data.axis == 2) ? (int) Bptr[n] : n;
            int d3 = (data.axis == 3) ? (int) Bptr[t] : t;
            int d4 = (data.axis == 4) ? (int) Bptr[d] : d;
            int d5 = (data.axis == 5) ? (int) Bptr[h] : h;
            int d6 = (data.axis == 6) ? (int) Bptr[w] : w;
            int d7 = (data.axis == 7) ? (int) Bptr[c] : c;

            Optr[i] = (half)(Xptr[data.shapeX.Index(d0, d1, d2, d3, d4, d5, d6, d7)]);
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct OneHotJob_Full_Half : IJobParallelFor, IJobResourceDeclarationXO
    {
        public ReadOnlyMemResource X { get; set; } half* Xptr => X.ptrhalf;
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public OneHotJobHelper data;

        public void Execute(int idx)
        {
            // rank1: X = n,_,_,_
            // rank2: X = n,_,_,c
            // rank3: X = n,_,w,c

            if (data.inputRank == 1) // TensorShape(X.flatHeight, depth)
            {
                int j = idx % data.depth;
                int n = (idx / data.depth) % data.shapeX.flatHeight;

                int index = (int)Xptr[n];
                float v = (j == index) ? data.onValue: data.offValue;
                Optr[idx] = (half)(v);
            }
            else if (data.inputRank == 2) // TensorShape(X.flatHeight, 1, depth, X.channels));
            {
                int i = idx % data.shapeX.channels;
                int j = (idx / data.shapeX.channels) % data.depth;
                int n = ((idx / data.shapeX.channels) / data.depth) % data.shapeX.flatHeight;

                int index = (int)Xptr[data.shapeX.Index(n, i)];
                float v = (j == index) ? data.onValue: data.offValue;
                Optr[idx] = (half)(v);
            }
            else // TensorShape(X.batch, X.width, depth, X.channels))
            {
                int i = idx % data.shapeX.channels;
                int j = (idx / data.shapeX.channels) % data.depth;
                int k = ((idx / data.shapeX.channels) / data.depth) % data.shapeX.width;
                int n = (((idx / data.shapeX.channels) / data.depth) / data.shapeX.width) % data.shapeX.batch;

                int index = (int)Xptr[data.shapeX.Index(n, 0, k, i)];
                float v = (j == index) ? data.onValue: data.offValue;
                Optr[idx] = (half)(v);
            }
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct RandomNormalJob_Full_Half : IJobParallelFor, IJobResourceDeclarationO
    {
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public RandomNormalJobHelper data;

        float Gaussian(float mean, float stdDev)
        {
            float u, v, s;
            do {
                u = data.rng.NextFloat() * 2 - 1;
                v = data.rng.NextFloat() * 2 - 1;
                s = u * u + v * v;
            } while (s >= 1 || s == 0);
            float mul = Mathf.Sqrt(-2.0f * Mathf.Log(s) / s);
            return mean + stdDev * u * mul;
        }

        public void Execute(int i)
        {
            Optr[i] = (half)(Gaussian(data.mean, data.scale));
        }
    }

    [BurstCompile(OptimizeFor = OptimizeFor.Performance, FloatMode = FloatMode.Default, FloatPrecision = FloatPrecision.Standard)]
    unsafe struct RandomUniformJob_Full_Half : IJobParallelFor, IJobResourceDeclarationO
    {
        public ReadWriteMemResource O { get; set; } half* Optr => O.ptrhalf;
        public RandomUniformJobHelper data;

        public void Execute(int i)
        {
            float v = data.mean + data.scale * data.rng.NextFloat();
            Optr[i] = (half)(v);
        }
    }

    #endregion
}
}
