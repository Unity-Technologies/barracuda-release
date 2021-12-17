using UnityEngine;
using UnityEngine.Assertions;
using System;
using Unity.Collections;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;
using Unity.Mathematics;

namespace Unity.Barracuda {

// BarracudaBurstCPU.Core.cs -- definition of class BurstCPUOps, Pin(), BurstTensorData
// BarracudaBurstCPU.Ops.cs  -- impl. IOps, job schedulers
// BarracudaBurstCPU.Jobs.cs -- impl. jobs

public partial class BurstCPUOps
{
    public enum BLAS
    {
        Disabled = 0,
        Native,
        Any
    }

    /// <summary>
    /// EXPERIMENTAL: Select BLAS preference
    /// Production code should stick to default (Native) for now.
    /// </summary>
    public static BLAS PreferBLAS { get; set; } = BLAS.Native;

    internal static JobHandle Dependencies(JobHandle job, JobHandle job2)
    {
        return JobHandle.CombineDependencies(job, job2);
    }
    internal static JobHandle Dependencies(JobHandle job, JobHandle job2, JobHandle job3)
    {
        return JobHandle.CombineDependencies(job, job2, job3);
    }
    internal static JobHandle Dependencies(JobHandle job, JobHandle job2, JobHandle job3, JobHandle job4)
    {
        return JobHandle.CombineDependencies(job, JobHandle.CombineDependencies(job2, job3, job4));
    }

    /// <inheritdoc/>
    public override Tensor MatMul(Tensor X, bool xTranspose, Tensor Y, bool yTranspose)
    {
        return MatMulHelper(X, xTranspose, Y, yTranspose, null, null, null, AllocScope.LayerOutput);
    }

    private Tensor MatMulHelper(Tensor X, bool xTranspose, Tensor Y, bool yTranspose,
        int? blockSizeM, int? blockSizeN, int? blockSizeK, AllocScope outputScope)
    {
        Assert.IsTrue(X.dimensions <= 2);
        Assert.IsTrue(Y.dimensions <= 2);

        int xw = X.flatWidth, xh = X.flatHeight;
        int yw = Y.flatWidth, yh = Y.flatHeight;

        if (xTranspose)
        {
            var tmp = xw; xw = xh; xh = tmp;
        }
        if (yTranspose)
        {
            var tmp = yw; yw = yh; yh = tmp;
        }

        Assert.AreEqual(xw, yh);
        var O = NewTensor(X.dataType, new TensorShape(xh, yw), outputScope, "");

        using (var ctx = new ForceFloatJobContext(X, Y, null, O))
        {
            { // O = broadcast(0)
                var job = new ZeroBroadcastJob();
                job.repeat = O.length;
                job.ScheduleO(ctx.o);
            }

            // O += X * K
            ScheduleSGEMM(
                ctx.x, X.flatHeight, X.flatWidth,
                ctx.w, Y.flatHeight, Y.flatWidth,
                ctx.o, O.flatHeight, O.flatWidth,
                blockSizeM: blockSizeM, blockSizeN: blockSizeN, blockSizeK: blockSizeK);
        }

        return O;
    }

    //O += X x K
    private unsafe void ScheduleSGEMM(
        IDependableMemoryResource pinX, int XM, int XN,
        IDependableMemoryResource pinK, int KM, int KN,
        IDependableMemoryResource pinO, int OM, int ON,
        bool transposeA = false, bool transposeB = false, int kernelOffset = 0,
        int? blockSizeM = null, int? blockSizeN = null, int? blockSizeK = null)
    {
        JobHandle dependOn = Dependencies(pinO.reuse, pinX.fence, pinK.fence);

        JobHandle jobFence = new JobHandle();
        float* ptrX = (float*)pinX.rawPtr;
        float* ptrK = (float*)pinK.rawPtr + kernelOffset;
        float* ptrO = (float*)pinO.rawPtr;

        if (PreferBLAS != BLAS.Disabled)
        {
            jobFence = blas.ScheduleSGEMM(dependOn,
                ptrX, XM, XN,
                ptrK, KM, KN,
                ptrO, OM, ON,
                16, transposeA, transposeB);
        }
        else if (Application.isMobilePlatform || Application.isConsolePlatform)
        {
            var job = new MatrixMultiplyLegacyJob();
            job.A = ptrX; job.AM = XM; job.AN = XN;
            job.B = ptrK; job.BM = KM; job.BN = KN;
            job.C = ptrO; job.CM = OM; job.CN = ON;
            job.transposeA = transposeA;
            job.transposeB = transposeB;

            jobFence = job.Schedule(dependOn);
        }
        else
        {
            var job = new MatrixMultiplyJob();
            job.A = ptrX; job.AM = XM; job.AN = XN;
            job.B = ptrK; job.BM = KM; job.BN = KN;
            job.C = ptrO; job.CM = OM; job.CN = ON;
            job.transposeA = transposeA;
            job.transposeB = transposeB;

            if (blockSizeM.HasValue)
                job.blockSizeM = blockSizeM.Value;

            if (blockSizeN.HasValue)
                job.blockSizeN = blockSizeN.Value;

            if (blockSizeK.HasValue)
                job.blockSizeK = blockSizeK.Value;

            jobFence = job.Schedule(dependOn);
        }

        pinO.fence = pinX.reuse = pinK.reuse = jobFence;
    }

    /// <inheritdoc/>
    public override Tensor MatMul(Tensor X, int rankX, Tensor Y, int rankY)
    {
        if (rankX == 2 && rankY == 2)
            return MatMul(X, false, Y, false);

        if (rankX == 3 && rankY == 2)
            return MatMul3x2(X,Y);
        else if (rankX == 4 && rankY == 4)
            return MatMul4x4(X,Y);
        else
            return base.MatMul(X, rankX, Y, rankY);
    }

    private Tensor MatMul3x2(Tensor X, Tensor Y)
    {
        int xb = X.batch,  xw = X.width, xh = X.channels;
        int yw = Y.channels, yh = Y.batch;

        Assert.AreEqual(xw, yh);
        var O = NewOutputTensor(X.dataType, new TensorShape(xb, 1, yw, xh));

        // O += X * K
        var job = new MatrixMultiply3x2Job();
        job.AM = xh;
        job.AN = xw;
        job.BM = yh;
        job.BN = yw;
        job.CM = xh;
        job.CN = yw;

        job.dispatchThreadX = ((xh + MatrixMultiply3x2Job.blockSize - 1) / MatrixMultiply3x2Job.blockSize);
        job.dispatchThreadY = ((yw + MatrixMultiply3x2Job.blockSize - 1) / MatrixMultiply3x2Job.blockSize);
        job.dispatchThreadZ = xb;

        using (var ctx = new ForceFloatJobContext(X, Y, null, O))
        {
            job.ScheduleXBO(ctx.x, ctx.w, ctx.o, job.dispatchThreadX * job.dispatchThreadY * job.dispatchThreadZ, 1);
        }

        return O;
    }

    private Tensor MatMul4x4(Tensor X, Tensor Y)
    {
        int xb0 = X.batch,  xh = X.height, xw = X.width, xb1 = X.channels;
        int yb0 = Y.batch,  yh = Y.height, yw = Y.width, yb1 = Y.channels;

        Assert.AreEqual(xw, yh);
        int ob0 = Mathf.Max(xb0, yb0); int ob1 = Mathf.Max(xb1, yb1);
        var O = NewOutputTensor(X.dataType, new TensorShape(ob0, xh, yw, ob1));

        // O += X * K
        var job = new MatrixMultiply4x4Job();
        job.AB0 = xb0;
        job.AB1 = xb1;
        job.AM = xh;
        job.AN = xw;
        job.BB0 = yb0;
        job.BB1 = yb1;
        job.BM = yh;
        job.BN = yw;
        job.CB1 = ob1;
        job.CM = xh;
        job.CN = yw;

        job.dispatchThreadX = ((xh + MatrixMultiply3x2Job.blockSize - 1) / MatrixMultiply3x2Job.blockSize);
        job.dispatchThreadY = ((yw + MatrixMultiply3x2Job.blockSize - 1) / MatrixMultiply3x2Job.blockSize);
        job.dispatchThreadZ = ob0*ob1;

        using (var ctx = new ForceFloatJobContext(X, Y, null, O))
        {
            job.ScheduleXBO(ctx.x, ctx.w, ctx.o, job.dispatchThreadX * job.dispatchThreadY * job.dispatchThreadZ, 1);
        }

        return O;
    }

    internal struct ForceFloatJobContext : IDisposable
    {
        private static Allocator memoryAllocator = Allocator.TempJob;

        //static to avoid GC. TODO try FencedMemoryAlloc as a struct
        private static FencedMemoryAlloc s_XFloat = new FencedMemoryAlloc();
        private static FencedMemoryAlloc s_WFloat = new FencedMemoryAlloc();
        private static FencedMemoryAlloc s_BFloat = new FencedMemoryAlloc();
        private static FencedMemoryAlloc s_OFloat = new FencedMemoryAlloc();

        public FencedMemoryAlloc xFloat;
        public FencedMemoryAlloc wFloat;
        public FencedMemoryAlloc bFloat;
        public FencedMemoryAlloc oFloat;
        private BurstTensorData pinO;

        public IDependableMemoryResource x;
        public IDependableMemoryResource w;
        public IDependableMemoryResource b;
        public IDependableMemoryResource o;

        public unsafe bool xConverted => xFloat.rawPtr != null;
        public unsafe bool wConverted => wFloat.rawPtr != null;
        public unsafe bool bConverted => bFloat.rawPtr != null;
        public unsafe bool oNeedConversion => oFloat.rawPtr != null;

        public ForceFloatJobContext(Tensor X, Tensor W, Tensor B, Tensor O)
        {
            // input & constants
            var pinX = Pin(X);
            var pinW = Pin(W);
            var pinB = (B!= null)? Pin(B) : null;
            // output
            pinO = Pin(O, uploadCache: false);

            xFloat = s_XFloat;
            wFloat = s_WFloat;
            bFloat = s_BFloat;
            oFloat = s_OFloat;

            ScheduleConversionToFloatIfNeeded(pinX, xFloat);
            ScheduleConversionToFloatIfNeeded(pinW, wFloat);
            ScheduleConversionToFloatIfNeeded(pinB, bFloat);
            AllocFencedMemoryIfNeeded(pinO, oFloat);

            unsafe
            {
                x = xFloat.rawPtr != null ? (IDependableMemoryResource)xFloat : pinX;
                w = wFloat.rawPtr != null ? (IDependableMemoryResource)wFloat : pinW;
                b = bFloat.rawPtr != null ? (IDependableMemoryResource)bFloat : pinB;
                o = oFloat.rawPtr != null ? (IDependableMemoryResource)oFloat : pinO;
            }

            if (B != null)
                Assert.AreEqual(wConverted, bConverted);
            Assert.AreEqual(xConverted, oNeedConversion);
        }

        public void Dispose()
        {
            //convert output as float to half
            if (oNeedConversion)
            {
                var convertFloatToHalfJob = new ConvertFloatToHalfJob();
                Assert.AreEqual(DataType.Float, oFloat.type);
                Assert.AreEqual(DataType.Half, pinO.dataType);
                Assert.AreEqual(oFloat.elementCount, pinO.count);
                convertFloatToHalfJob.ScheduleXO(oFloat, pinO, pinO.count, 1024);
            }

            // free activations buffers
            if (xConverted || oNeedConversion)
                unsafe {
                    var freeJob = new MemFreeJob();
                    freeJob.allocator = memoryAllocator;
                    freeJob.buffer0 = xFloat.rawPtr;
                    freeJob.buffer1 = oFloat.rawPtr;
                    freeJob.Schedule(pinO.fence);
                }

            // free weights buffers
            if (wConverted || bConverted)
                unsafe {
                    var freeJob = new MemFreeJob();
                    freeJob.allocator = memoryAllocator;
                    freeJob.buffer0 = wFloat.rawPtr;
                    freeJob.buffer1 = bFloat.rawPtr;
                    freeJob.Schedule(pinO.fence);
                }

            xFloat.ClearState();
            wFloat.ClearState();
            bFloat.ClearState();
            oFloat.ClearState();
        }

        private static bool AllocFencedMemoryIfNeeded(BurstTensorData pin, FencedMemoryAlloc fencedMem)
        {
            if (pin != null && pin.dataType == DataType.Half)
            {
                fencedMem.Allocate(pin.count, DataType.Float, JobsUtility.CacheLineSize, memoryAllocator);
                return true;
            }

            return false;
        }

        private static void ScheduleConversionToFloatIfNeeded(BurstTensorData pinnedTensor, FencedMemoryAlloc destination)
        {
            if (AllocFencedMemoryIfNeeded(pinnedTensor, destination))
            {
                var convertHalfToFloatJob = new ConvertHalfToFloatJob();
                Assert.AreEqual(DataType.Half, pinnedTensor.dataType);
                Assert.AreEqual(DataType.Float, destination.type);
                Assert.AreEqual(pinnedTensor.count, destination.elementCount);
                convertHalfToFloatJob.ScheduleXO(pinnedTensor, destination, pinnedTensor.count, 1024);
            }
        }
    }

    /// <inheritdoc/>
    public override Tensor Dense3(Tensor X, Tensor W, Tensor B)
    {
        int xb = X.batch,  xw = X.width, xh = X.channels;
        int yw = W.channels, yh = W.batch;

        Assert.AreEqual(xw, yh);
        var O = NewOutputTensor(X.dataType, new TensorShape(xb, 1, yw, xh));

        var job = new Dense3Job_Full_Float();
        job.data.AM = xh;
        job.data.AN = xw;
        job.data.BM = yh;
        job.data.BN = yw;
        job.data.SM = xh;
        job.data.SN = yw;

        job.data.dispatchThreadX = ((xh + Dense3Job_Full_Float.blockSize - 1) / Dense3Job_Full_Float.blockSize);
        job.data.dispatchThreadY = ((yw + Dense3Job_Full_Float.blockSize - 1) / Dense3Job_Full_Float.blockSize);
        job.data.dispatchThreadZ = xb;

        using (var ctx = new ForceFloatJobContext(X, W, B, O))
        {
            job.ScheduleXSBO(ctx.x, ctx.w, ctx.b, ctx.o, job.data.dispatchThreadX * job.data.dispatchThreadY * job.data.dispatchThreadZ, 1);
        }

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Dense(Tensor X, Tensor W, Tensor B, Layer.FusedActivation fusedActivation)
    {
        //D.Log(string.Format("X = {0}", X.shape));
        Assert.IsTrue(W.dimensions <= 2);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(B.flatWidth, W.flatWidth);
        Assert.AreEqual(X.flatWidth, W.flatHeight);
        var O = NewTensorForFusedActivation(X.dataType, new TensorShape(X.flatHeight, W.flatWidth), fusedActivation);

        using (var ctx = new ForceFloatJobContext(X, W, B, O))
        {
            { // O = broadcast(B)
                // @TODO: move broadcast B directly into MatrixMultiplyJob
                var job = new VectorBroadcastJob();
                job.channels = O.flatWidth;
                job.repeat = O.flatHeight;
                job.ScheduleXO(ctx.b, ctx.o);
            }

            ScheduleSGEMM(
                ctx.x, X.flatHeight, X.flatWidth,
                ctx.w, W.flatHeight, W.flatWidth,
                ctx.o, O.flatHeight, O.flatWidth);
        }

        return ApplyFusedActivation(O, fusedActivation);
    }

    /// <inheritdoc/>
    public override Tensor Conv2D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, Layer.FusedActivation fusedActivation)
    {
        return Conv2DUsingIm2ColSliced(X, K, B, stride, pad, fusedActivation);
    }

    Tensor Conv2DUsingIm2ColSliced(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, Layer.FusedActivation fusedActivation)
    {
        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(X.channels, K.kernelDepth);
        Assert.AreEqual(K.kernelCount, B.flatWidth);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);

        var kernelWidth = K.kernelWidth;
        var kernelHeight = K.kernelHeight;
        var inChannels = K.kernelDepth;
        var outChannels = K.kernelCount;
        var batch = X.batch;

        bool pointwiseConvolution = kernelWidth == 1 && kernelHeight == 1 &&                    // 1x1 kernel
                                    stride[0] == 1 && stride[1] == 1 &&                         // no strides
                                    pad[0] == 0 && pad[1] == 0 && pad[2] == 0 && pad[3] == 0;   // no padding

        var O = NewTensorForFusedActivation(X.dataType, X.shape.ApplyKernel(K.shape, stride, pad), fusedActivation);
        var T = pointwiseConvolution ? null:                       // pointwise convolution is just O=X*K, we can completely skip Im2Col()
                NewTempTensor(DataType.Float, new TensorShape(O.batch, O.height, O.width, inChannels), "Conv2DUsingIm2ColSliced/T"); // T holds slice of Im2Col(X)

        var outElements = O.batch * O.height * O.width;
        var inWidth = X.width;

        Assert.AreEqual(O.batch, batch);
        Assert.AreEqual(O.channels, B.flatWidth);
        Assert.AreEqual(O.channels, outChannels);

        using (var ctx = new ForceFloatJobContext(X, K, B, O))
        {
            // temporary slice
            var pinT  = pointwiseConvolution ? ctx.x  : Pin(T);
            if (T != null)
                Assert.AreEqual(DataType.Float, T.dataType);

            { // O = broadcast(B)
                // @TODO: move broadcast B directly into MatrixMultiplyJob
                var job = new VectorBroadcastJob();
                job.channels = outChannels;
                job.repeat = outElements;
                job.ScheduleXO(ctx.b, ctx.o);
            }

            // We can solve convolution by iteratively accumulating
            // matrix multiplication of X' and K' for each positon in kernel where:
            //  X' is input X repeatedly shifted according to kernel position,
            //  K' is slice of weights K according to kernel position.
            //
            // Pseudocode:
            //  X :: Input
            //  T :: Temporary
            //  K :: Kernel
            //  O :: Output
            //  foreach ky in kernelHeight:
            //      foreach kx in kernelWidth:
            //          Temporary = shift(Input, horizontal_shift = kx, vertical_shift = ky)
            //          Temporary = pad(Temporary)
            //          Temporary = stride(Temporary)
            //          Output += Temporary * Kernel[dy, dx, :, :]
            //
            // Note for functions above that:
            //  1) shift() can be implemented by copying data from n to T in a linear fashion.
            //  2) stride() can be implemented by copying data every Nth pixel in a linear fashion.
            //  3) pad() can be optimized for top and bottom of the tensor by writing 0s across the whole row.

            // O += conv(X, K)
            int kernelOffset = 0;
            for (int dy = 0; dy < kernelHeight; ++dy)
            for (int dx = 0; dx < kernelWidth; ++dx)
            {
                //T=im2col(X) else T=X
                if (!pointwiseConvolution)
                {
                    var offsetX = dx - pad[0];
                    var offsetY = dy - pad[1];

                    var strideX = stride[0];
                    var strideY = stride[1];

                            var firstPixel = 0 * strideX + offsetX;
                            var lastPixel = (T.width - 1) * strideX + offsetX;
                            int numberOfPixelsToPadLeft = SafeIntDivCeil(Math.Max(0, 0 - firstPixel), strideX); // count(x * stride[0] + offsetX < 0)
                            int numberOfPixelsToPadRight = SafeIntDivCeil(Math.Max(0, lastPixel - (inWidth - 1)), strideX); // count(x * stride[0] + offsetX >= inWidth)
                            int numberOfPixelsToSkipFromInputRow = (offsetX >= 0 || strideX == 0)
                                ? offsetX
                                : // strideX == 0 protects against div-by-zero
                                lastPixel % strideX; // first(x * stride[0] + offsetX >= 0) == (inWidth * stride[0] + offsetX) % stride[0]
                    int numberOfPixelsToCopyFromInputRow = T.width - numberOfPixelsToPadLeft - numberOfPixelsToPadRight;

                    if (UnityEngine.Debug.isDebugBuild) // only to Assert correctness of the values above
                    {
                        // validate above calculations with alternative approach
                        int assertNumberOfPixelsToPadLeft = 0;
                        int assertNumberOfPixelsToPadRight = 0;
                        int assertNumberOfPixelsToSkipFromInputRow = 0;
                        for (var x = 0; x < T.width; ++x)
                        {
                            var readX = x * strideX + offsetX;
                            if (readX < 0)
                                assertNumberOfPixelsToPadLeft++;
                            else
                            {
                                assertNumberOfPixelsToSkipFromInputRow = readX;
                                break;
                            }
                        }

                        for (var x = T.width - 1; x >= 0; --x)
                        {
                            var readX = x * strideX + offsetX;
                            if (readX >= inWidth)
                                assertNumberOfPixelsToPadRight++;
                            else
                                break;
                        }

                        int assertNumberOfPixelsToCopyFromInputRow = T.width - assertNumberOfPixelsToPadLeft - assertNumberOfPixelsToPadRight;

                                Assert.AreEqual(numberOfPixelsToPadLeft, assertNumberOfPixelsToPadLeft);
                                Assert.AreEqual(numberOfPixelsToPadRight, assertNumberOfPixelsToPadRight);
                                Assert.AreEqual(numberOfPixelsToSkipFromInputRow, assertNumberOfPixelsToSkipFromInputRow);
                                Assert.AreEqual(numberOfPixelsToCopyFromInputRow, assertNumberOfPixelsToCopyFromInputRow);
                    }

                    Assert.IsTrue(numberOfPixelsToPadLeft >= 0);
                    Assert.IsTrue(numberOfPixelsToPadRight >= 0);
                    Assert.IsTrue(numberOfPixelsToCopyFromInputRow >= 0);
                    Assert.IsTrue(numberOfPixelsToSkipFromInputRow >= 0);
                    Assert.IsTrue(numberOfPixelsToPadLeft + numberOfPixelsToPadRight <= T.width);
                    Assert.IsTrue(numberOfPixelsToSkipFromInputRow <= X.width);
                    Assert.IsTrue(numberOfPixelsToCopyFromInputRow <= X.width);
                    Assert.AreEqual(numberOfPixelsToPadLeft + numberOfPixelsToCopyFromInputRow + numberOfPixelsToPadRight, T.width);

                    // extra clamp for safety since we are in the unsafe code block
                            numberOfPixelsToPadLeft  = Math.Min(Math.Max(0, numberOfPixelsToPadLeft), T.width);
                            numberOfPixelsToPadRight = Math.Min(Math.Max(0, numberOfPixelsToPadRight), T.width - numberOfPixelsToPadLeft);
                    numberOfPixelsToSkipFromInputRow = Math.Min(Math.Max(0, numberOfPixelsToSkipFromInputRow), X.width);
                    numberOfPixelsToCopyFromInputRow = Math.Min(Math.Max(0, numberOfPixelsToCopyFromInputRow), X.width - numberOfPixelsToSkipFromInputRow);

                    var job = new Im2ColSliceJob();
                            job.inOutBatch = batch;
                            job.inOutChannels = inChannels;
                            job.inHeight = X.height;
                            job.inStrideN = X.height * X.width * X.channels;
                            job.inStrideH = X.width * X.channels;
                            job.inStrideW = X.channels;
                            job.outWidth = T.width;
                            job.outStrideN = T.height * T.width * T.channels;
                            job.outStrideH = T.width * T.channels;
                            job.strideX = strideX;
                            job.strideY = strideY;
                            job.offsetY = offsetY;
                            job.padLeft = numberOfPixelsToPadLeft;
                            job.padRight = numberOfPixelsToPadRight;
                    job.skipFromInputRow = numberOfPixelsToSkipFromInputRow;
                    job.copyFromInputRow = numberOfPixelsToCopyFromInputRow;

                    job.ScheduleXO(ctx.x, pinT, T.height, 16);
                }

                // O += slice(T) * slice(K)
                // With T=im2col(X) if pointwiseConvolution else T=X
                ScheduleSGEMM(
                    pinT, outElements, inChannels,
                                ctx.w, inChannels, outChannels,
                                ctx.o, outElements, outChannels, transposeA: false, transposeB: false, kernelOffset);

                kernelOffset += inChannels * outChannels;
            }
        }

        //Calling Dispose on BurstTensorData will sync the fences, so this is a performance VS memory peak tradeoff here.
        T?.Dispose();

        return ApplyFusedActivation(O, fusedActivation);
    }

    /// <inheritdoc/>
    public override Tensor MaxPool2D(Tensor X, int[] pool, int[] stride, int[] pad)
    {
        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(pool.Length, 2);
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);

        var O = NewOutputTensor(X.dataType,X.shape.ApplyPool(pool, stride, pad));

        var job = new MaxPool2DJobHelper();
        job.strideX = stride[0];
        job.strideY = stride[1];
        job.padX    = pad[0];
        job.padY    = pad[1];

        job.inHeight   = X.height;
        job.inWidth    = X.width;
        job.inChannels = X.channels;
        job.inStrideN  = X.height * X.width * X.channels;
        job.inStrideH  =            X.width * X.channels;
        job.inStrideW  =                      X.channels;

        job.kernelWidth   = pool[0];
        job.kernelHeight  = pool[1];

        job.outBatch   = O.batch;
        job.outWidth   = O.width;
        job.outStrideN = O.height * O.width * O.channels;
        job.outStrideH =            O.width * O.channels;
        job.outStrideW =                      O.channels;

        job.ScheduleXO(X, O, O.height, 4);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor AvgPool2D(Tensor X, int[] pool, int[] stride, int[] pad)
    {
        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(pool.Length, 2);
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);

        var O = NewOutputTensor(X.dataType,X.shape.ApplyPool(pool, stride, pad));

        var job = new AvgPool2DJobHelper();
        job.strideX = stride[0];
        job.strideY = stride[1];
        job.padX    = pad[0];
        job.padY    = pad[1];

        job.inHeight   = X.height;
        job.inWidth    = X.width;
        job.inChannels = X.channels;
        job.inStrideN  = X.height * X.width * X.channels;
        job.inStrideH  =            X.width * X.channels;
        job.inStrideW  =                      X.channels;

        job.kernelWidth   = pool[0];
        job.kernelHeight  = pool[1];

        job.outBatch   = O.batch;
        job.outWidth   = O.width;
        job.outStrideN = O.height * O.width * O.channels;
        job.outStrideH =            O.width * O.channels;
        job.outStrideW =                      O.channels;

        job.ScheduleXO(X, O, O.height, 4);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor GlobalMaxPool2D(Tensor X)
    {
        return MaxPool2D(X, new[] {X.width, X.height}, new[] {1, 1}, new[] {0, 0, 0, 0});
    }

    /// <inheritdoc/>
    public override Tensor GlobalAvgPool2D(Tensor X)
    {
        return AvgPool2D(X, new[] {X.width, X.height}, new[] {1, 1}, new[] {0, 0, 0, 0});
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

        var O = NewTensorForFusedActivation(X.dataType, X.shape.ApplyKernel(K.shape, stride, pad), fusedActivation);

        var job = new DepthwiseConv2DJobHelper();

        job.strideX = stride[0];
        job.strideY = stride[1];
        job.padX    = pad[0];
        job.padY    = pad[1];

        job.inHeight   = X.height;
        job.inWidth    = X.width;
        job.inChannels = X.channels;
        job.inStrideN  = X.height * X.width * X.channels;
        job.inStrideH  =            X.width * X.channels;
        job.inStrideW  =                      X.channels;

        job.kernelCount   = K.kernelCount;
        job.kernelHeight  = K.kernelHeight;
        job.kernelWidth   = K.kernelWidth;
        job.kernelStrideH = K.height * K.width * K.channels;
        job.kernelStrideW =            K.width * K.channels;

        job.outBatch   = O.batch;
        job.outWidth   = O.width;
        job.outStrideN = O.height * O.width * O.channels;
        job.outStrideH =            O.width * O.channels;
        job.outStrideW =                      O.channels;

        job.ScheduleXSBO(X, K, B, O, O.height, 4);

        return ApplyFusedActivation(O, fusedActivation);
    }

    /// <inheritdoc/>
    public override Tensor ScaleBias(Tensor X, Tensor S, Tensor B)
    {
        if (!X.shape.Is4D())
            base.ScaleBias(X, S, B);

        Assert.AreEqual(S.shape, B.shape);
        bool isScalarOp = (S.length == 1);
        bool isSaVector = (S.length == S.channels);
        bool isVectorOp = (X.channels == S.channels && isSaVector);
        bool isTensorOp = (X.shape == S.shape);
        Assert.IsTrue(isScalarOp || isVectorOp || isTensorOp);

        var O = NewTensorLike(X, AllocScope.LayerOutput);
        Assert.AreEqual(O.shape, X.shape);

        var jobData = new VectorBroadcastScaleBiasJobHelper();
        jobData.inOutChannels = O.channels;
        jobData.alpha = 1;
        jobData.ScheduleXSBO(X, S, B, O, O.length / O.channels, Math.Max(16, 1024 / O.channels));

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Relu(Tensor X)
    {
        var O = NewTensorLike(X, AllocScope.LayerOutput);
        Assert.AreEqual(O.length, X.length);

        var job = new ReluJobHelper();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Relu6(Tensor X)
    {
        var O = NewTensorLike(X, AllocScope.LayerOutput);
        Assert.AreEqual(O.length, X.length);

        var job = new Relu6JobHelper();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor LeakyRelu(Tensor X, float alpha)
    {
        var O = NewTensorLike(X, AllocScope.LayerOutput);
        Assert.AreEqual(O.length, X.length);

        var job = new LeakyReluJobHelper();
        job.alpha = alpha;
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Tanh(Tensor X)
    {
        var O = NewTensorLike(X, AllocScope.LayerOutput);
        Assert.AreEqual(O.length, X.length);

        var job = new TanhJobHelper();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

     /// <inheritdoc/>
    public override Tensor Softplus(Tensor X)
    {
        var O = NewTensorLike(X, AllocScope.LayerOutput);
        Assert.AreEqual(O.length, X.length);

        var job = new SoftplusJobHelper();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Sigmoid(Tensor X)
    {
        var O = NewTensorLike(X, AllocScope.LayerOutput);
        Assert.AreEqual(O.length, X.length);

        var job = new SigmoidJobHelper();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor HardSigmoid(Tensor X, float alpha, float beta)
    {
        var O = NewTensorLike(X, AllocScope.LayerOutput);
        Assert.AreEqual(O.length, X.length);

        var job = new HardSigmoidJobHelper();
        job.alpha = alpha;
        job.beta = beta;
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }


    /// <inheritdoc/>
    public override Tensor Elu(Tensor X, float alpha)
    {
        var O = NewTensorLike(X, AllocScope.LayerOutput);
        Assert.AreEqual(O.length, X.length);

        var job = new EluJobHelper();
        job.alpha = alpha;
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Selu(Tensor X, float alpha, float gamma)
    {
        var O = NewTensorLike(X, AllocScope.LayerOutput);
        Assert.AreEqual(O.length, X.length);

        var job = new SeluJobHelper();
        job.alpha = alpha;
        job.gamma = gamma;
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Swish(Tensor X)
    {
        var O = NewTensorLike(X, AllocScope.LayerOutput);
        Assert.AreEqual(O.length, X.length);

        var job = new SwishJobHelper();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor PRelu(Tensor X, Tensor S)
    {
        var O = NewTensorLike(X, AllocScope.LayerOutput);

        Assert.AreEqual(X.channels, O.channels);
        Assert.IsTrue((X.flatWidth == S.flatWidth) || (S.flatWidth == 1));

        var job = new PReluJobHelper();
        job.isGammaAVector = (S.flatWidth == 1) ? 0 : 1;
        job.inOutChannels = O.channels;
        job.ScheduleXBO(X, S, O, O.length / O.channels, Math.Max(16, 1024 / O.channels));

        return O;
    }

    internal static FencedMemoryAlloc s_maxValues = new FencedMemoryAlloc();
    internal static FencedMemoryAlloc s_expSums = new FencedMemoryAlloc();

    /// <inheritdoc/>
    public override Tensor Softmax(Tensor X, int axis)
    {
        var O = NewOutputTensor(X.dataType, X.shape);
        Assert.AreEqual(O.length, X.length);
        Assert.AreEqual(O.flatWidth, X.flatWidth);

        axis = X.shape.Axis(axis);

        var pinX = Pin(X);
        var pinO = Pin(O, uploadCache: false);

        //Allocate memory
        Allocator memoryAllocator = Allocator.TempJob;
        var reduceOpShape = X.shape.Reduce(axis);
        s_maxValues.Allocate(reduceOpShape.length, pinX.dataType, JobsUtility.CacheLineSize, memoryAllocator);
        s_expSums.Allocate(reduceOpShape.length, pinX.dataType, JobsUtility.CacheLineSize, memoryAllocator);

        int offsetReduce = 1;
        for (int i = 7; i >= axis; i--)
            offsetReduce *= reduceOpShape[i];

        // x_max = X.max(axis=1)
        {
            var job = new ReduceMaxJobHelper();
            job.offsetReduce = offsetReduce;
            job.reduceDim = X.shape[axis];
            job.ScheduleXO(pinX, s_maxValues, reduceOpShape.length, 1024);
        }
        // e_x_sum = Sum[exp(x[:,c] - x_max[:]), c]
        {
            var job = new ExpBiasReduceJobHelper();
            job.offsetReduce = offsetReduce;
            job.reduceDim = X.shape[axis];
            job.ScheduleXBO(pinX, s_maxValues, s_expSums, reduceOpShape.length, 1024);
        }
        // exp(x[n,c] - x_max[n]) / e_x_sum[n]
        {
            var job = new SoftmaxEndJobHelper();
            job.offsetReduce = offsetReduce;
            job.reduceDim = X.shape[axis];
            job.ScheduleXSBO(pinX, s_expSums, s_maxValues, pinO, O.length, 1024);
        }
        // free memory (in job)
        unsafe {
            var job = new MemFreeJob();
            job.allocator = memoryAllocator;
            job.buffer0 = s_maxValues.rawPtr;
            job.buffer1 = s_expSums.rawPtr;
            job.Schedule(pinO.fence);
        }

        s_maxValues.ClearState();
        s_expSums.ClearState();

        return O;
    }

    /// <inheritdoc/>
    public override Tensor LogSoftmax(Tensor X, int axis)
    {
        var O = NewOutputTensor(X.dataType, X.shape);
        Assert.AreEqual(O.length, X.length);
        Assert.AreEqual(O.flatWidth, X.flatWidth);

        axis = X.shape.Axis(axis);

        var pinX = Pin(X);
        var pinO = Pin(O, uploadCache: false);

        //Allocate memory
        Allocator memoryAllocator = Allocator.TempJob;
        var reduceOpShape = X.shape.Reduce(axis);
        s_maxValues.Allocate(reduceOpShape.length, pinX.dataType, JobsUtility.CacheLineSize, memoryAllocator);
        s_expSums.Allocate(reduceOpShape.length, pinX.dataType, JobsUtility.CacheLineSize, memoryAllocator);

        int offsetReduce = 1;
        for (int i = 7; i >= axis; i--)
            offsetReduce *= reduceOpShape[i];

        // x_max = X.max(axis=1)
        {
            var job = new ReduceMaxJobHelper();
            job.offsetReduce = offsetReduce;
            job.reduceDim = X.shape[axis];
            job.ScheduleXO(pinX, s_maxValues, reduceOpShape.length, 1024);
        }
        // e_x_sum = Sum[exp(x[:,c] - x_max[:]), c]
        {
            var job = new ExpBiasReduceJobHelper();
            job.offsetReduce = offsetReduce;
            job.reduceDim = X.shape[axis];
            job.ScheduleXBO(pinX, s_maxValues, s_expSums, reduceOpShape.length, 1024);
        }
        // (x[n,c] - x_max[n]) - log(e_x_sum[n])
        {
            var job = new LogSoftmaxEndJobHelper();
            job.offsetReduce = offsetReduce;
            job.reduceDim = X.shape[axis];
            job.ScheduleXSBO(pinX, s_expSums, s_maxValues, pinO, O.length, 1024);
        }
        // free memory (in job)
        unsafe {
            var job = new MemFreeJob();
            job.allocator = memoryAllocator;
            job.buffer0 = s_maxValues.rawPtr;
            job.buffer1 = s_expSums.rawPtr;
            job.Schedule(pinO.fence);
        }

        s_maxValues.ClearState();
        s_expSums.ClearState();

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Abs(Tensor X)
    {
        var O = NewTensorLike(X, AllocScope.LayerOutput);
        Assert.AreEqual(O.length, X.length);

        var job = new AbsJobHelper();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Neg(Tensor X)
    {
        var O = NewTensorLike(X, AllocScope.LayerOutput);
        Assert.AreEqual(O.length, X.length);

        var job = new NegJobHelper();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Ceil(Tensor X)
    {
        var O = NewTensorLike(X, AllocScope.LayerOutput);
        Assert.AreEqual(O.length, X.length);

        var job = new CeilJobHelper();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Clip(Tensor X, float min, float max)
    {
        var O = NewTensorLike(X, AllocScope.LayerOutput);
        Assert.AreEqual(O.length, X.length);

        var job = new ClipJobHelper();
        job.min = min;
        job.max = max;
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Floor(Tensor X)
    {
        var O = NewTensorLike(X, AllocScope.LayerOutput);
        Assert.AreEqual(O.length, X.length);

        var job = new FloorJobHelper();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Round(Tensor X)
    {
        var O = NewTensorLike(X, AllocScope.LayerOutput);
        Assert.AreEqual(O.length, X.length);

        var job = new RoundJobHelper();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Reciprocal(Tensor X)
    {
        var O = NewTensorLike(X, AllocScope.LayerOutput);
        Assert.AreEqual(O.length, X.length);

        var job = new ReciprocalJobHelper();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Pow(Tensor X, float alpha)
    {
        var O = NewTensorLike(X, AllocScope.LayerOutput);
        Assert.AreEqual(O.length, X.length);

        var job = new PowJobHelper();
        job.alpha = alpha;
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Exp(Tensor X)
    {
        var O = NewTensorLike(X, AllocScope.LayerOutput);
        Assert.AreEqual(O.length, X.length);

        var job = new ExpJobHelper();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Log(Tensor X)
    {
        var O = NewTensorLike(X, AllocScope.LayerOutput);
        Assert.AreEqual(O.length, X.length);

        var job = new LogJobHelper();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Sqrt(Tensor X)
    {
        var O = NewTensorLike(X, AllocScope.LayerOutput);
        Assert.AreEqual(O.length, X.length);

        var job = new SqrtJobHelper();
        job.ScheduleXO(X, O , O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Acos(Tensor X)
    {
        var O = NewTensorLike(X, AllocScope.LayerOutput);
        Assert.AreEqual(O.length, X.length);

        var job = new AcosJobHelper();
        job.ScheduleXO(X, O , O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Acosh(Tensor X)
    {
        var O = NewTensorLike(X, AllocScope.LayerOutput);
        Assert.AreEqual(O.length, X.length);

        var job = new AcoshJobHelper();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Asin(Tensor X)
    {
        var O = NewTensorLike(X, AllocScope.LayerOutput);
        Assert.AreEqual(O.length, X.length);

        var job = new AsinJobHelper();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Asinh(Tensor X)
    {
        var O = NewTensorLike(X, AllocScope.LayerOutput);
        Assert.AreEqual(O.length, X.length);

        var job = new AsinhJobHelper();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Atan(Tensor X)
    {
        var O = NewTensorLike(X, AllocScope.LayerOutput);
        Assert.AreEqual(O.length, X.length);

        var job = new AtanJobHelper();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Atanh(Tensor X)
    {
        var O = NewTensorLike(X, AllocScope.LayerOutput);
        Assert.AreEqual(O.length, X.length);

        var job = new AtanhJobHelper();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Cos(Tensor X)
    {
        var O = NewTensorLike(X, AllocScope.LayerOutput);
        Assert.AreEqual(O.length, X.length);

        var job = new CosJobHelper();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Cosh(Tensor X)
    {
        var O = NewTensorLike(X, AllocScope.LayerOutput);
        Assert.AreEqual(O.length, X.length);

        var job = new CoshJobHelper();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Sin(Tensor X)
    {
        var O = NewTensorLike(X, AllocScope.LayerOutput);
        Assert.AreEqual(O.length, X.length);

        var job = new SinJobHelper();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Sinh(Tensor X)
    {
        var O = NewTensorLike(X, AllocScope.LayerOutput);
        Assert.AreEqual(O.length, X.length);

        var job = new SinhJobHelper();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Tan(Tensor X)
    {
        var O = NewTensorLike(X, AllocScope.LayerOutput);
        Assert.AreEqual(O.length, X.length);

        var job = new TanJobHelper();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Erf(Tensor X)
    {
        var O = NewTensorLike(X, AllocScope.LayerOutput);
        Assert.AreEqual(O.length, X.length);

        var job = new ErfJobHelper();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    private unsafe void AssignTensorStrides8D(Tensor X, int* strides)
    {
        strides[0] = (X.sequenceLength == 1)     ? 0 : X.numberOfDirections * X.batch * X.extraDimension * X.depth * X.height * X.width * X.channels;
        strides[1] = (X.numberOfDirections == 1) ? 0 : X.batch * X.extraDimension * X.depth * X.height * X.width * X.channels;
        strides[2] = (X.batch == 1)              ? 0 : X.extraDimension * X.depth * X.height * X.width * X.channels;
        strides[3] = (X.extraDimension == 1)     ? 0 : X.depth * X.height * X.width * X.channels;
        strides[4] = (X.depth == 1)              ? 0 : X.height * X.width * X.channels;
        strides[5] = (X.height == 1)             ? 0 : X.width * X.channels;
        strides[6] = (X.width == 1)              ? 0 : X.channels;
        strides[7] = (X.channels == 1)           ? 0 : 1;
    }

    private void BroadcastAdd(ref Tensor O, Tensor X, Tensor Y, float alpha = 1f)
    {
        if(X.shape == O.shape && Y.length == 1)
        {
            var job = new ScalarBroadcastAddJobHelper();
            job.alpha = alpha;
            job.ScheduleXBO(X, Y, O, O.length, 1024);
        }
        else if (X.shape == O.shape && Y.shape == O.shape)
        {
            var job = new BroadcastAddJobHelper();
            job.alpha = alpha;
            job.ScheduleXBO(X, Y, O, O.length, 1024);
        }
        else
        {
            var job = new ElementwiseAddJobHelper();
            job.alpha = alpha;
            job.shapeO = O.shape;
            unsafe {
                AssignTensorStrides8D(X, job.stridesX);
                AssignTensorStrides8D(Y, job.stridesY);
            }
            job.ScheduleXBO(X, Y, O, O.length, 1024);
        }
    }

    private void BroadcastSub(ref Tensor O, Tensor X, Tensor Y)
    {
        BroadcastAdd(ref O, X, Y, -1f);
    }

    private void BroadcastMul(ref Tensor O, Tensor X, Tensor Y)
    {
        if(X.shape == O.shape && Y.length == 1)
        {
            var job = new ScalarBroadcastMulJobHelper();
            job.ScheduleXBO(X, Y, O, O.length, 1024);
        }
        else if (X.shape == O.shape && Y.shape == O.shape)
        {
            var job = new BroadcastMulJobHelper();
            job.ScheduleXBO(X, Y, O, O.length, 1024);
        }
        else
        {
            var job = new ElementwiseMulJobHelper();
            job.shapeO = O.shape;
            unsafe
            {
                AssignTensorStrides8D(X, job.stridesX);
                AssignTensorStrides8D(Y, job.stridesY);
            }
            job.ScheduleXBO(X, Y, O, O.length, 1024);
        }
    }

    private void BroadcastDiv(ref Tensor O, Tensor X, Tensor Y)
    {
        if(X.shape == O.shape && Y.length == 1)
        {
            var job = new ScalarBroadcastDivJobHelper();
            job.ScheduleXBO(X, Y, O, O.length, 1024);
        }
        else if (X.shape == O.shape && Y.shape == O.shape)
        {
            var job = new BroadcastDivJobHelper();
            job.ScheduleXBO(X, Y, O, O.length, 1024);
        }
        else
        {
            var job = new ElementwiseDivJobHelper();
            job.shapeO = O.shape;
            unsafe
            {
                AssignTensorStrides8D(X, job.stridesX);
                AssignTensorStrides8D(Y, job.stridesY);
            }
            job.ScheduleXBO(X, Y, O , O.length, 1024);
        }
    }

    private void BroadcastPow(ref Tensor O, Tensor X, Tensor Y)
    {
        if (X.shape == O.shape && Y.length == 1)
        {
            var job = new ScalarBroadcastPowJobHelper();
            job.ScheduleXBO(X, Y, O, O.length, 1024);
        }
        else if (X.shape == O.shape && Y.shape == O.shape)
        {
            var job = new BroadcastPowJobHelper();
            job.ScheduleXBO(X, Y, O, O.length, 1024);
        }
        else
        {
            var job = new ElementwisePowJobHelper();
            job.shapeO = O.shape;
            unsafe
            {
                AssignTensorStrides8D(X, job.stridesX);
                AssignTensorStrides8D(Y, job.stridesY);
            }
            job.ScheduleXBO(X, Y, O, O.length, 1024);        }
    }

    private void BroadcastMin(ref Tensor O, Tensor X, Tensor Y)
    {
        if(X.shape == O.shape && Y.length == 1)
        {
            var job = new ScalarBroadcastMinJobHelper();
            job.ScheduleXBO(X, Y, O, O.length, 1024);
        }
        else if (X.shape == O.shape && Y.shape == O.shape)
        {
            var job = new BroadcastMinJobHelper();
            job.ScheduleXBO(X, Y, O, O.length, 1024);
        }
        else
        {
            var job = new ElementwiseMinJobHelper();
            job.shapeO = O.shape;
            unsafe
            {
                AssignTensorStrides8D(X, job.stridesX);
                AssignTensorStrides8D(Y, job.stridesY);
            }
            job.ScheduleXBO(X, Y, O, O.length, 1024);
        }
    }

    private void BroadcastMax(ref Tensor O, Tensor X, Tensor Y)
    {
        if(X.shape == O.shape && Y.length == 1)
        {
            var job = new ScalarBroadcastMaxJobHelper();
            job.ScheduleXBO(X, Y, O, O.length, 1024);
        }
        else if (X.shape == O.shape && Y.shape == O.shape)
        {
            var job = new BroadcastMaxJobHelper();
            job.ScheduleXBO(X, Y, O, O.length, 1024);
        }
        else
        {
            var job = new ElementwiseMaxJobHelper();
            job.shapeO = O.shape;
            unsafe
            {
                AssignTensorStrides8D(X, job.stridesX);
                AssignTensorStrides8D(Y, job.stridesY);
            }
            job.ScheduleXBO(X, Y, O, O.length, 1024);
        }
    }

    private Tensor AddHelper(Tensor[] tensors, AllocScope outputScope)
    {
        if (!TensorExtensions.AreAllTensorsConvertibleTo4D(tensors))
            return base.Add(tensors);

        var O = NewTensorLike(tensors, outputScope);
        var X = tensors[0];

        for (int t = 1; t < tensors.Length; ++t)
        {
            BroadcastAdd(ref O, X, tensors[t]);
            X = O;
        }
        return O;
    }

    /// <inheritdoc/>
    // O = tensors[0] + tensors[1] + ... + tensors[N-1]
    public override Tensor Add(Tensor[] tensors)
    {
        return AddHelper(tensors, AllocScope.LayerOutput);
    }

    /// <inheritdoc/>
    // O = tensors[0] - tensors[1] - ... - tensors[N-1]
    public override Tensor Sub(Tensor[] tensors)
    {
        if (!TensorExtensions.AreAllTensorsConvertibleTo4D(tensors))
            return base.Sub(tensors);


        var O = NewTensorLike(tensors, AllocScope.LayerOutput);
        var X = tensors[0];

        for (int t = 1; t < tensors.Length; ++t)
        {
            BroadcastSub(ref O, X, tensors[t]);
            X = O;
        }
        return O;
    }

    /// <inheritdoc/>
    // O = tensors[0] * tensors[1] * ... * tensors[N-1]
    public override Tensor Mul(Tensor[] tensors)
    {
        if (!TensorExtensions.AreAllTensorsConvertibleTo4D(tensors))
            return base.Mul(tensors);


        var O = NewTensorLike(tensors, AllocScope.LayerOutput);
        var X = tensors[0];

        for (int t = 1; t < tensors.Length; ++t)
        {
            BroadcastMul(ref O, X, tensors[t]);
            X = O;
        }
        return O;
    }

    /// <inheritdoc/>
    // O = tensors[0] / tensors[1] / ... / tensors[N-1]
    public override Tensor Div(Tensor[] tensors)
    {
        if (!TensorExtensions.AreAllTensorsConvertibleTo4D(tensors))
            return base.Div(tensors);


        var O = NewTensorLike(tensors, AllocScope.LayerOutput);
        var X = tensors[0];

        for (int t = 1; t < tensors.Length; ++t)
        {
            BroadcastDiv(ref O, X, tensors[t]);
            X = O;
        }
        return O;
    }

    /// <inheritdoc/>
    // O = tensors[0] ^ tensors[1] ^ ... ^ tensors[N-1]
    public override Tensor Pow(Tensor[] tensors)
    {
        if (!TensorExtensions.AreAllTensorsConvertibleTo4D(tensors))
            return base.Pow(tensors);


        var O = NewTensorLike(tensors, AllocScope.LayerOutput);
        var X = tensors[0];

        for (int t = 1; t < tensors.Length; ++t)
        {
            BroadcastPow(ref O, X, tensors[t]);
            X = O;
        }
        return O;
    }

    /// <inheritdoc/>
    // O = min(tensors[0], tensors[1],  ... , tensors[N-1])
    public override Tensor Min(Tensor[] tensors)
    {
        if (!TensorExtensions.AreAllTensorsConvertibleTo4D(tensors))
            return base.Min(tensors);

        var O = NewTensorLike(tensors, AllocScope.LayerOutput);
        var X = tensors[0];

        for (int t = 1; t < tensors.Length; ++t)
        {
            BroadcastMin(ref O, X, tensors[t]);
            X = O;
        }
        return O;
    }

    /// <inheritdoc/>
    // O = max(tensors[0], tensors[1],  ... , tensors[N-1])
    public override Tensor Max(Tensor[] tensors)
    {
        if (!TensorExtensions.AreAllTensorsConvertibleTo4D(tensors))
            return base.Max(tensors);

        var O = NewTensorLike(tensors, AllocScope.LayerOutput);
        var X = tensors[0];

        for (int t = 1; t < tensors.Length; ++t)
        {
            BroadcastMax(ref O, X, tensors[t]);
            X = O;
        }
        return O;
    }

    // // O = (1/N) * (tensors[0] + tensors[1] + ... + tensors[N-1])
    // public override Tensor Mean(Tensor[] tensors)
    // {
    //    if (!TensorExtensions.AreAllTensorsConvertibleTo4D(tensors))
    //        base.Mean(tensors);

    //     // accumulate
    //     Func<float, float, float> op = (a, b) => a + b;
    //     var O = ApplyElementwiseWithBroadcast(tensors, op);

    //     // div by N
    //     var invN = 1.0f / tensors.Length;
    //     var end = O.length;
    //     for (int i = 0; i < O.length; ++i)
    //     {
    //         float v = O[i];
    //         v *= invN;
    //         O[i] = v;
    //     }
    //     return O;
    // }

    /// <inheritdoc/>
    protected override Tensor CopyAndReshape(Tensor X, TensorShape shape)
    {
        Assert.AreEqual(X.length, shape.length);
        var O = NewOutputTensor(X.dataType, shape);

        var job = new CopyJobHelper();
        job.length = O.length;
        job.ScheduleXO(X, O);

        return O;
    }

    public override Tensor Reshape(Tensor X, TensorShape newShape)
    {
        if (X.shape == newShape)
            return base.Reshape(X, newShape);

        return CopyAndReshape(X, newShape);
    }

    /// <inheritdoc/>
    public override Tensor Concat(Tensor[] tensors, int axis)
    {
        var concatShape = TensorExtensions.Concat(tensors, axis);
        var dataType = tensors.Length > 0 ? tensors[0].dataType : DataType.Float;
        var O = NewOutputTensor(dataType, concatShape);

        unsafe
        {
            // product of all tensor dimensions starting from axis
            var copyBlockLengths = stackalloc int[tensors.Length];
            var copyBlockLengthsAcum = stackalloc int[tensors.Length];
            int copyBlockLengthsSum = 0;
            for (int i = 0; i < tensors.Length; ++i)
            {
                copyBlockLengthsAcum[i] = copyBlockLengthsSum;
                copyBlockLengths[i] = (int)GetAggregatedDimLength(tensors[i].shape,  tensors[i].shape.Axis(axis), TensorShape.MaxRank);
                copyBlockLengthsSum += copyBlockLengths[i];
            }

            // copy tensor data interleaved into O
            int takes = (int)GetAggregatedDimLength(concatShape,  0, concatShape.Axis(axis));
            var pinO = Pin(O, uploadCache: false);
            using (var ctx = new ParallelJobsContext(pinO))
            {
                for (int i = 0; i < tensors.Length; ++i)
                {
                    var pinX = Pin(tensors[i]);
                    var job = new CopyStrideJobHelper();
                    job.OStride = copyBlockLengthsSum;
                    job.XStride = copyBlockLengths[i];
                    job.length = copyBlockLengths[i];
                    job.count = takes;
                    ctx.ScheduleXO(job, pinX, 0, pinO, copyBlockLengthsAcum[i]);
                }
            }
        }
        return O;
    }

    /// <inheritdoc/>
    public override Tensor StridedSlice(Tensor X, int[] starts4Dor8D, int[] ends4Dor8D, int[] strides4Dor8D)
    {
        return StridedSliceHelper(X, starts4Dor8D, ends4Dor8D, strides4Dor8D, AllocScope.LayerOutput);
    }

    private Tensor StridedSliceHelper(Tensor X, int[] starts4Dor8D, int[] ends4Dor8D, int[] strides4Dor8D, AllocScope outputScope)
    {
        unsafe
        {
            int* starts = stackalloc int[TensorShape.MaxRank];
            int* ends = stackalloc int[TensorShape.MaxRank];
            int* strides = stackalloc int[TensorShape.MaxRank];
            TensorExtensions.Get8DParametersNoAlloc(X.shape, starts4Dor8D, starts, 0);
            TensorExtensions.Get8DParametersNoAlloc(X.shape, ends4Dor8D, ends, 1);
            TensorExtensions.Get8DParametersNoAlloc(X.shape, strides4Dor8D, strides, 1);

            var O = NewTensor(X.dataType, X.shape.ApplyStridedSlice8DUnsafeNoAlloc(starts, ends, strides), outputScope);

            int* wrappedStartsIndices = ends; //reuse buffer to save a stack allocation.
            for (int i = 0; i < TensorShape.MaxRank; ++i)
                wrappedStartsIndices[i] = Math.Min(TensorExtensions.WrapIndex(starts[i], X.shape[i]), X.shape[i] - 1);

            Assert.AreEqual(8, TensorShape.MaxRank);

            //TODO/Idea for further optimisation: Add a version using UnsafeUtility.MemCpyStride when many strides are 1 (starting from C amd going upward).
            if (strides[TensorShape.C] == 1)
            {
                var job = new GenericSliceJobHelper();
                job.shapeX = X.shape;
                job.shapeO = O.shape;
                job.startS = wrappedStartsIndices[0];
                job.startR = wrappedStartsIndices[1];
                job.startN = wrappedStartsIndices[2];
                job.startT = wrappedStartsIndices[3];
                job.startD = wrappedStartsIndices[4];
                job.startH = wrappedStartsIndices[5];
                job.startW = wrappedStartsIndices[6];
                job.startC = wrappedStartsIndices[7];
                job.strideS = strides[0];
                job.strideR = strides[1];
                job.strideN = strides[2];
                job.strideT = strides[3];
                job.strideD = strides[4];
                job.strideH = strides[5];
                job.strideW = strides[6];
                job.strideC = strides[7];
                int numCopy = O.shape.length / O.shape.channels;
                job.ScheduleXO(X, O, numCopy, 64);
            }
            else
            {
                var job = new GenericStridedSliceJobHelper();
                job.shapeX = X.shape;
                job.shapeO = O.shape;
                job.startS = wrappedStartsIndices[0];
                job.startR = wrappedStartsIndices[1];
                job.startN = wrappedStartsIndices[2];
                job.startT = wrappedStartsIndices[3];
                job.startD = wrappedStartsIndices[4];
                job.startH = wrappedStartsIndices[5];
                job.startW = wrappedStartsIndices[6];
                job.startC = wrappedStartsIndices[7];
                job.strideS = strides[0];
                job.strideR = strides[1];
                job.strideN = strides[2];
                job.strideT = strides[3];
                job.strideD = strides[4];
                job.strideH = strides[5];
                job.strideW = strides[6];
                job.strideC = strides[7];
                job.ScheduleXO(X, O, O.length, 1024);
            }

            return O;
        }
    }

    /// <inheritdoc/>
    public override Tensor Border2D(Tensor X, int[] pad, float constant)
    {
        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(pad.Length, 6);

        var O = NewOutputTensor(X.dataType, X.shape.ApplyBorder(pad));

        int croppedWidth = X.width - Math.Max(0, -pad[3]);
        int croppedHeight = X.height - Math.Max(0, -pad[4]);
        int croppedChannels = X.channels - Math.Max(0, -pad[5]);

        var job = new Border2DJobHelper();

        job.shapeX = X.shape;
        job.shapeO = O.shape;

        job.PadWidth    = pad[0];
        job.PadHeight   = pad[1];
        job.PadChannels = pad[2];

        job.CroppedWidth    = croppedWidth;
        job.CroppedHeight   = croppedHeight;
        job.CroppedChannels = croppedChannels;

        job.Beta = constant;

        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Pad2DReflect(Tensor X, int[] pad)
    {
        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(pad.Length, 6);

        var O = NewOutputTensor(X.dataType, X.shape.ApplyBorder(pad));

        var job = new Pad2DReflectJobHelper();

        job.shapeX = X.shape;
        job.shapeO = O.shape;

        job.PadWidth    = pad[0];
        job.PadHeight   = pad[1];
        job.PadChannels = pad[2];

        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Pad2DSymmetric(Tensor X, int[] pad)
    {
        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(pad.Length, 6);

        var O = NewOutputTensor(X.dataType, X.shape.ApplyBorder(pad));

        var job = new Pad2DSymmetricJobHelper();

        job.shapeX = X.shape;
        job.shapeO = O.shape;

        job.PadWidth    = pad[0];
        job.PadHeight   = pad[1];
        job.PadChannels = pad[2];

        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Pad2DEdge(Tensor X, int[] pad)
    {
        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(pad.Length, 6);

        var O = NewOutputTensor(X.dataType, X.shape.ApplyBorder(pad));

        var job = new Pad2DEdgeJobHelper();

        job.shapeX = X.shape;
        job.shapeO = O.shape;

        job.PadWidth    = pad[0];
        job.PadHeight   = pad[1];
        job.PadChannels = pad[2];

        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Transpose(Tensor X, int[] permutations)
    {
        return TransposeHelper(X, permutations, AllocScope.LayerOutput);
    }

    private Tensor TransposeHelper(Tensor X, int[] permutations, AllocScope outputScope)
    {

        var outPermutations = TensorExtensions.Get8DPermutationsForNHWCPermutationsAndShape(
                                                X.shape, new NativeArray<int>(permutations, Allocator.Temp));
        var O = NewTensor(X.dataType, X.shape.Permute(outPermutations), outputScope);

        var job = new TransposeJobHelper();
        job.shapeX = X.shape;
        job.shapeO = O.shape;
        unsafe
        {
            job.permutations[0] = outPermutations[0];
            job.permutations[1] = outPermutations[1];
            job.permutations[2] = outPermutations[2];
            job.permutations[3] = outPermutations[3];
            job.permutations[4] = outPermutations[4];
            job.permutations[5] = outPermutations[5];
            job.permutations[6] = outPermutations[6];
            job.permutations[7] = outPermutations[7];
        }

        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor ReduceMean(Tensor X, int axis)
    {
        axis = X.shape.Axis(axis);
        var O = NewOutputTensor(X.dataType, X.shape.Reduce(axis));

        int offsetReduce = 1;
        for (int i = TensorShape.MaxRank - 1; i >= axis; i--)
            offsetReduce *= O.shape[i];

        var job = new ReduceMeanJobHelper();
        job.offsetReduce = offsetReduce;
        job.reduceDim = X.shape[axis];
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor ReduceSum(Tensor X, int axis)
    {
        axis = X.shape.Axis(axis);
        var O = NewOutputTensor(X.dataType, X.shape.Reduce(axis));

        int offsetReduce = 1;
        for (int i = TensorShape.MaxRank - 1; i >= axis; i--)
            offsetReduce *= O.shape[i];

        var job = new ReduceSumJobHelper();
        job.offsetReduce = offsetReduce;
        job.reduceDim = X.shape[axis];
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    public override Tensor ReduceMax(Tensor X, int axis)
    {
        axis = X.shape.Axis(axis);
        var O = NewOutputTensor(X.dataType, X.shape.Reduce(axis));

        int offsetReduce = 1;
        for (int i = TensorShape.MaxRank - 1; i >= axis; i--)
            offsetReduce *= O.shape[i];

        var job = new ReduceMaxJobHelper();
        job.offsetReduce = offsetReduce;
        job.reduceDim = X.shape[axis];
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Tile(Tensor X, int[] repeats)
    {
        Tensor O = NewOutputTensor(X.dataType, X.shape.Scale(repeats));

        var job = new TileJobHelper();
        job.shapeX = X.shape;
        job.shapeO = O.shape;
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Gather(Tensor[] tensors, int axis)
    {
        Tensor X = tensors[0];
        Tensor indices = tensors[1];

        var shape = X.shape;
        shape[axis] = indices.length;

        var O = NewOutputTensor(X.dataType, shape);

        Assert.AreEqual(TensorShape.MaxRank, 8);

        var job = new GatherJobHelper();
        job.axis = axis;
        job.shapeX = X.shape;
        job.shapeO = O.shape;
        job.ScheduleXBO(X, indices, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor OneHot(Tensor X, int depth, float onValue, float offValue, int inputRank=-1)
    {
        if (inputRank == -1)
            inputRank = X.dimensions;

        if (inputRank >= 4)
            throw new NotImplementedException();

        Tensor O;
        if (inputRank == 1)
            O = NewOutputTensor(X.dataType, new TensorShape(X.flatHeight, depth));
        else if (inputRank == 2)
            O = NewOutputTensor(X.dataType, new TensorShape(X.flatHeight, 1, depth, X.flatWidth));
        else
            O = NewOutputTensor(X.dataType, new TensorShape(X.batch, X.width, depth, X.channels));

        var job = new OneHotJobHelper();
        job.depth = depth;
        job.shapeX = X.shape;
        job.shapeO = O.shape;
        job.inputRank = inputRank;
        job.onValue = onValue;
        job.offValue = offValue;

        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    internal uint jobCountCall = 0;

    /// <inheritdoc/>
    public override Tensor RandomNormal(TensorShape s, float mean, float scale, int seed)
    {
        var O = NewOutputTensor(DataType.Float, s);
        //TODO fp16: RandomNormal should be able to select output type
        //see dtype here https://github.com/onnx/onnx/blob/master/docs/Operators.md#RandomNormal

        var pinO = Pin(O, uploadCache: false);

        var job = new RandomNormalJobHelper();
        // seed is combined with jobCountCall to keep rng persistent over frame
        var finalSeed = (uint) (seed ^ (++jobCountCall));
        job.rng = new Unity.Mathematics.Random(finalSeed != 0 ? finalSeed : 1);
        job.mean = mean;
        job.scale = scale;
        job.ScheduleO(pinO, 0, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor RandomUniform(TensorShape s, float mean, float scale, int seed)
    {
        var O = NewOutputTensor(DataType.Float, s);
        //TODO fp16: RandomNormal should be able to select output type
        //see dtype here https://github.com/onnx/onnx/blob/master/docs/Operators.md#RandomUniform

        var pinO = Pin(O, uploadCache: false);

        var job = new RandomUniformJobHelper();

        // seed is combined with jobCountCall to keep rng persistent over frame
        var finalSeed = (uint) (seed ^ (++jobCountCall));
        job.rng = new Unity.Mathematics.Random(finalSeed != 0 ? finalSeed : 1);
        job.mean = mean;
        job.scale = scale;
        job.ScheduleO(pinO, 0, O.length, 1024);

        return O;
    }

    Tensor LSTMDense3Helper(Tensor X, Tensor W, Tensor B)
    {
        int xb = X.batch, xh = X.width, xw = X.channels;
        int yh = W.batch, yw = W.channels;

        Assert.AreEqual(xw, yh);
        var Otemp = NewTempTensor(X.dataType, new TensorShape(xb, 1, xh, yw));

        var pinX = Pin(X);
        var pinW = Pin(W);
        var pinB = Pin(B);
        var pinO = Pin(Otemp, uploadCache: false);

        unsafe
        {
            float* ptrX = pinX.array.AddressAt(pinX.offset);
            float* ptrW = pinW.array.AddressAt(pinW.offset);
            float* ptrB = pinB.array.AddressAt(pinB.offset);
            float* ptrO = pinO.array.AddressAt(pinO.offset);
            {
                var job = new LSTMDense3Job();
                job.A = ptrX;
                job.AM = xh;
                job.AN = xw;
                job.B = ptrW;
                job.BM = yh;
                job.BN = yw;
                job.C = ptrB;
                job.CN = B.channels;
                job.S = ptrO;
                job.SM = xh;
                job.SN = yw;

                job.dispatchThreadX = ((xh + LSTMDense3Job.blockSize - 1) / LSTMDense3Job.blockSize);
                job.dispatchThreadY = ((yw + LSTMDense3Job.blockSize - 1) / LSTMDense3Job.blockSize);
                job.dispatchThreadZ = xb;

                pinO.fence = pinX.reuse = pinW.reuse = pinB.reuse =
                    job.Schedule(Dependencies(pinO.reuse, pinX.fence, pinW.fence, pinB.fence));
            }
        }

        return Otemp;
    }

    Tensor LSTMDenseHelper(Tensor X, Tensor W, Tensor B)
    {
        int xw = X.channels, xh = X.batch;
        int yw = W.channels, yh = W.batch;

        Assert.AreEqual(xw, yh);
        var Otemp = NewTempTensor(X.dataType, new TensorShape(xh, yw));

        var pinX = Pin(X);
        var pinW = Pin(W);
        var pinB = Pin(B);
        var pinO = Pin(Otemp, uploadCache: false);

        unsafe
        {
            float* ptrX = pinX.array.AddressAt(pinX.offset);
            float* ptrW = pinW.array.AddressAt(pinW.offset);
            float* ptrB = pinB.array.AddressAt(pinB.offset);
            float* ptrO = pinO.array.AddressAt(pinO.offset);
            {
                var job = new LSTMDenseJob();
                job.A = ptrX;
                job.AM = xh;
                job.AN = xw;
                job.B = ptrW;
                job.BM = yh;
                job.BN = yw;
                job.C = ptrB;
                job.CN = B.channels;
                job.S = ptrO;
                job.SM = xh;
                job.SN = yw;

                job.dispatchThreadX = ((xh + LSTMDenseJob.blockSize - 1) / LSTMDenseJob.blockSize);
                job.dispatchThreadY = ((yw + LSTMDenseJob.blockSize - 1) / LSTMDenseJob.blockSize);

                pinO.fence = pinX.reuse = pinW.reuse = pinB.reuse =
                    job.Schedule(Dependencies(pinO.reuse, pinX.fence, pinW.fence, pinB.fence));
            }
        }

        return Otemp;
    }

    public override Tensor[] LSTM(Tensor X, Tensor[] W, Tensor[] R, Tensor[] Wb, Tensor[] Rb, Tensor hidden, Tensor cell)
    {
        // Gate indices [iofj]
        const int g_i = 0, g_o = 1, g_f = 2, g_j = 3;

        TensorShape xShape = X.shape; // X shape is [seq_length, batch_size, input_size]
        int sequenceLength = xShape.batch;
        int batchSize = xShape.channels;
        int inputSize = xShape.width;
        int hiddenSize = cell.channels;

        Tensor O = NewOutputTensor(X.dataType, new TensorShape(sequenceLength, batchSize, hiddenSize, 1));
        var pinO = Pin(O, uploadCache: false);

        var cell_out = NewOutputTensor(X.dataType, new TensorShape(batchSize, hiddenSize));  //TODO this can create fragmentation in ping pong buffer
        var hidden_out = NewOutputTensor(X.dataType, new TensorShape(batchSize, hiddenSize));//TODO this can create fragmentation in ping pong buffer
        var pinCellOut = Pin(cell_out, uploadCache: false); var pinHiddenOut = Pin(hidden_out, uploadCache: false);

        Tensor i_mad_w_tmp = null;
        Tensor j_mad_w_tmp = null;
        Tensor f_mad_w_tmp = null;
        Tensor o_mad_w_tmp = null;
        Tensor i_mad_w = null;
        Tensor j_mad_w = null;
        Tensor f_mad_w = null;
        Tensor o_mad_w = null;

        // if platforms supports Blas, favor that path, this is faster than our Dense3 implem atm

        // transpose once for sequential Dense access
        Tensor Xt = TransposeHelper(X, new[] { 0, 1, 3, 2 }, AllocScope.InternalToLayer);
        var useBLAS = PreferBLAS != BLAS.Disabled;
        if (!useBLAS)
        {
            i_mad_w = LSTMDense3Helper(Xt, W[g_i], Wb[g_i]);
            j_mad_w = LSTMDense3Helper(Xt, W[g_j], Wb[g_j]);
            f_mad_w = LSTMDense3Helper(Xt, W[g_f], Wb[g_f]);
            o_mad_w = LSTMDense3Helper(Xt, W[g_o], Wb[g_o]);
        }

        JobHandle jobFence = new JobHandle();
        for (int s = 0; s < sequenceLength; s++)
        {
            Tensor X_sequence = null;
            if (useBLAS)
            {
                //Note/TODO: if Wb are not 4D tensors AddHelper will allocate via ping pong allocator leading to allocator fragmentation.
                X_sequence = StridedSliceHelper(Xt, new[] { s, 0, 0, 0 }, new[] { s + 1, int.MaxValue, int.MaxValue, int.MaxValue }, new[] { 1, 1, 1, 1 }, AllocScope.InternalToLayer);
                X_sequence = X_sequence.Reshape(new TensorShape(batchSize, inputSize));
                i_mad_w_tmp = MatMulHelper(X_sequence, false, W[g_i], false, null, null, null, AllocScope.InternalToLayer);
                j_mad_w_tmp = MatMulHelper(X_sequence, false, W[g_j], false, null, null, null, AllocScope.InternalToLayer);
                f_mad_w_tmp = MatMulHelper(X_sequence, false, W[g_f], false, null, null, null, AllocScope.InternalToLayer);
                o_mad_w_tmp = MatMulHelper(X_sequence, false, W[g_o], false, null, null, null, AllocScope.InternalToLayer);
                i_mad_w = AddHelper(new[]{i_mad_w_tmp, Wb[g_i]}, AllocScope.InternalToLayer);
                j_mad_w = AddHelper(new[]{j_mad_w_tmp, Wb[g_j]}, AllocScope.InternalToLayer);
                f_mad_w = AddHelper(new[]{f_mad_w_tmp, Wb[g_f]}, AllocScope.InternalToLayer);
                o_mad_w = AddHelper(new[]{o_mad_w_tmp, Wb[g_o]}, AllocScope.InternalToLayer);
            }

            var i_mad_r = LSTMDenseHelper(hidden, R[g_i], Rb[g_i]);
            var j_mad_r = LSTMDenseHelper(hidden, R[g_j], Rb[g_j]);
            var f_mad_r = LSTMDenseHelper(hidden, R[g_f], Rb[g_f]);
            var o_mad_r = LSTMDenseHelper(hidden, R[g_o], Rb[g_o]);

            var pinCell = Pin(cell); var pinHidden = Pin(hidden);
            var pinImadW = Pin(i_mad_w); var pinImadR = Pin(i_mad_r);
            var pinJmadW = Pin(j_mad_w); var pinJmadR = Pin(j_mad_r);
            var pinFmadW = Pin(f_mad_w); var pinFmadR = Pin(f_mad_r);
            var pinOmadW = Pin(o_mad_w); var pinOmadR = Pin(o_mad_r);

            unsafe
            {
                float* ptrCell = pinCell.array.AddressAt(pinCell.offset);
                float* ptrImadW = pinImadW.array.AddressAt(pinImadW.offset); float* ptrImadR = pinImadR.array.AddressAt(pinImadR.offset);
                float* ptrJmadW = pinJmadW.array.AddressAt(pinJmadW.offset); float* ptrJmadR = pinJmadR.array.AddressAt(pinJmadR.offset);
                float* ptrFmadW = pinFmadW.array.AddressAt(pinFmadW.offset); float* ptrFmadR = pinFmadR.array.AddressAt(pinFmadR.offset);
                float* ptrOmadW = pinOmadW.array.AddressAt(pinOmadW.offset); float* ptrOmadR = pinOmadR.array.AddressAt(pinOmadR.offset);
                float* ptrCellOut = pinCellOut.array.AddressAt(pinCellOut.offset); float* ptrHiddenOut = pinHiddenOut.array.AddressAt(pinHiddenOut.offset);
                float* ptrO = pinO.array.AddressAt(pinO.offset);
                {
                    var job = new LSTMEndJob();
                    job.cell_out = ptrCellOut;
                    job.hidden_out = ptrHiddenOut;
                    job.i_mad_w = ptrImadW;
                    job.j_mad_w = ptrJmadW;
                    job.f_mad_w = ptrFmadW;
                    job.o_mad_w = ptrOmadW;
                    job.i_mad_r = ptrImadR;
                    job.j_mad_r = ptrJmadR;
                    job.f_mad_r = ptrFmadR;
                    job.o_mad_r = ptrOmadR;
                    job.cell = ptrCell;
                    job.O = ptrO;
                    job.sequenceIndexO = s;
                    job.sequenceIndexI = useBLAS ? 0 : s;
                    job.batchSize = batchSize;
                    job.hiddenSize = hiddenSize;
                    job.batchSizeR = hidden.batch;

                    jobFence = pinCellOut.fence = pinHiddenOut.fence =
                    pinHidden.reuse = pinCell.reuse =
                    pinImadW.reuse = pinJmadW.reuse = pinFmadW.reuse = pinOmadW.reuse =
                    pinImadR.reuse = pinJmadR.reuse = pinFmadR.reuse = pinOmadR.reuse =
                        job.Schedule(batchSize*hiddenSize, 1024, JobHandle.CombineDependencies(pinO.reuse, pinCellOut.reuse, JobHandle.CombineDependencies(pinHiddenOut.reuse,
                                    pinImadW.fence, JobHandle.CombineDependencies(pinJmadW.fence, pinFmadW.fence, JobHandle.CombineDependencies(pinOmadW.fence,
                                    pinImadR.fence, JobHandle.CombineDependencies(pinJmadR.fence, pinFmadR.fence, JobHandle.CombineDependencies(pinOmadR.fence, pinCell.fence, pinHidden.fence)))))));
                }
            }

            hidden = hidden_out;
            cell = cell_out;

            i_mad_r.Dispose();
            j_mad_r.Dispose();
            f_mad_r.Dispose();
            o_mad_r.Dispose();

            if (useBLAS)
            {
                X_sequence.Dispose();
                i_mad_w_tmp.Dispose();
                j_mad_w_tmp.Dispose();
                f_mad_w_tmp.Dispose();
                o_mad_w_tmp.Dispose();
                i_mad_w.Dispose();
                j_mad_w.Dispose();
                f_mad_w.Dispose();
                o_mad_w.Dispose();
            }
        }

        pinO.fence = jobFence;

        Xt.Dispose();
        if (!useBLAS)
        {
            i_mad_w.Dispose();
            j_mad_w.Dispose();
            f_mad_w.Dispose();
            o_mad_w.Dispose();
        }

        return new[] { O, hidden, cell };
    }
}

} // namespace Barracuda
