using UnityEngine;
using UnityEngine.Assertions;
using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;

namespace Unity.Barracuda {

// BarracudaBurstCPU.Core.cs -- definition of class BurstCPUOps, Pin(), BurstTensorData
// BarracudaBurstCPU.Ops.cs  -- impl. IOps, job schedulers
// BarracudaBurstCPU.Jobs.cs -- impl. jobs

public partial class BurstCPUOps
{
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
        var O = NewTensor(xh, yw);

        var pinX = Pin(X);
        var pinY = Pin(Y);
        var pinO = Pin(O);

        { // O = broadcast(0)
            var job = new ZeroBroadcastJob();
            job.repeat = O.length;
            job.ScheduleO(pinO);
        }

        // O += X * K
        ScheduleSGEMM(
            pinX, X.flatHeight, X.flatWidth,
            pinY, Y.flatHeight, Y.flatWidth,
            pinO, O.flatHeight, O.flatWidth);

        return O;
    }

    //O += X x K
    private unsafe void ScheduleSGEMM(
        BurstTensorData pinX, int XN, int XM,
        BurstTensorData pinK, int KN, int KM,
        BurstTensorData pinO, int ON, int OM,
        bool transposeA = false, bool transposeB = false, int kernelOffset = 0)
    {
        JobHandle dependOn = Dependencies(pinO.reuse, pinX.fence, pinK.fence);

        JobHandle jobFence = new JobHandle();
        fixed (float*
            ptrX = &pinX.array[pinX.offset],
            ptrK = &pinK.array[pinK.offset+kernelOffset],
            ptrO = &pinO.array[pinO.offset])
        {
            if (m_UseBlas)
            {
                jobFence = blas.ScheduleSGEMM(dependOn,
                    ptrX, XN, XM,
                    ptrK, KN, KM,
                    ptrO, ON, OM,
                    16, transposeA, transposeB);
            }
            else
            {
                var job = new MatrixMultiplyJob();
                job.A = ptrX; job.AN = XN; job.AM = XM;
                job.B = ptrK; job.BN = KN; job.BM = KM;
                job.C = ptrO; job.CN = ON; job.CM = OM;
                job.transposeA = transposeA;
                job.transposeB = transposeB;
                jobFence = job.Schedule(dependOn);
            }
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
        var O = NewTensor(xb, 1, yw, xh);

        var pinX = Pin(X);
        var pinY = Pin(Y);
        var pinO = Pin(O);

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrY = &pinY.array[pinY.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                {   // O += X * K
                    var job = new MatrixMultiply3x2Job();
                    job.A = ptrX;
                    job.AB = xb;
                    job.AN = xh;
                    job.AM = xw;
                    job.B = ptrY;
                    job.BN = yh;
                    job.BM = yw;
                    job.C = ptrO;
                    job.CN = xh;
                    job.CM = yw;

                    job.dispatchThreadX = ((xh + MatrixMultiply3x2Job.blockSize - 1) / MatrixMultiply3x2Job.blockSize);
                    job.dispatchThreadY = ((yw + MatrixMultiply3x2Job.blockSize - 1) / MatrixMultiply3x2Job.blockSize);
                    job.dispatchThreadZ = xb;

                    pinO.fence = pinX.reuse = pinY.reuse =
                        job.Schedule(Dependencies(pinO.reuse, pinX.fence, pinY.fence));
                }
            }
        }

        return O;
    }

    private Tensor MatMul4x4(Tensor X, Tensor Y)
    {
        int xb0 = X.batch,  xh = X.height, xw = X.width, xb1 = X.channels;
        int yb0 = Y.batch,  yh = Y.height, yw = Y.width, yb1 = Y.channels;

        Assert.AreEqual(xw, yh);
        int ob0 = Mathf.Max(xb0, yb0); int ob1 = Mathf.Max(xb1, yb1);
        var O = NewTensor(ob0, xh, yw, ob1);

        var pinX = Pin(X);
        var pinY = Pin(Y);
        var pinO = Pin(O);

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrY = &pinY.array[pinY.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                {   // O += X * K
                    var job = new MatrixMultiply4x4Job();
                    job.A = ptrX;
                    job.AB0 = xb0;
                    job.AB1 = xb1;
                    job.AN = xh;
                    job.AM = xw;
                    job.B = ptrY;
                    job.BB0 = yb0;
                    job.BB1 = yb1;
                    job.BN = yh;
                    job.BM = yw;
                    job.C = ptrO;
                    job.CB0 = ob0;
                    job.CB1 = ob1;
                    job.CN = xh;
                    job.CM = yw;

                    job.dispatchThreadX = ((xh + MatrixMultiply3x2Job.blockSize - 1) / MatrixMultiply3x2Job.blockSize);
                    job.dispatchThreadY = ((yw + MatrixMultiply3x2Job.blockSize - 1) / MatrixMultiply3x2Job.blockSize);
                    job.dispatchThreadZ = ob0*ob1;

                    pinO.fence = pinX.reuse = pinY.reuse =
                        job.Schedule(Dependencies(pinO.reuse, pinX.fence, pinY.fence));
                }
            }
        }

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Dense3(Tensor X, Tensor W, Tensor B)
    {
        int xb = X.batch,  xw = X.width, xh = X.channels;
        int yw = W.channels, yh = W.batch;

        Assert.AreEqual(xw, yh);
        var O = NewTensor(xb, 1, yw, xh);

        var pinX = Pin(X);
        var pinW = Pin(W);
        var pinB = Pin(B);
        var pinO = Pin(O);

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrW = &pinW.array[pinW.offset],
                ptrB = &pinB.array[pinB.offset],
                ptrO = &pinO.array[pinO.offset])
                {
                    {   // O += X * K
                    var job = new Dense3Job();
                    job.A = ptrX;
                    job.AB = xb;
                    job.AN = xh;
                    job.AM = xw;
                    job.B = ptrW;
                    job.BN = yh;
                    job.BM = yw;
                    job.C = ptrB;
                    job.S = ptrO;
                    job.SN = xh;
                    job.SM = yw;

                    job.dispatchThreadX = ((xh + Dense3Job.blockSize - 1) / Dense3Job.blockSize);
                    job.dispatchThreadY = ((yw + Dense3Job.blockSize - 1) / Dense3Job.blockSize);
                    job.dispatchThreadZ = xb;

                    pinO.fence = pinX.reuse = pinW.reuse = pinB.reuse =
                        job.Schedule(Dependencies(pinO.reuse, pinX.fence, pinW.fence, pinB.fence));
                }
            }
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
        var O = NewTensor(X.flatHeight, W.flatWidth);

        var pinX = Pin(X);
        var pinW = Pin(W);
        var pinB = Pin(B);
        var pinO = Pin(O);

        { // O = broadcast(B)
            // @TODO: move broadcast B directly into MatrixMultiplyJob
            var job = new VectorBroadcastJob();
            job.channels = O.flatWidth;
            job.repeat = O.flatHeight;
            job.ScheduleXO(pinB, pinO);
        }

        // O += X * K
        ScheduleSGEMM(
            pinX, X.flatHeight, X.flatWidth,
            pinW, W.flatHeight, W.flatWidth,
            pinO, O.flatHeight, O.flatWidth);

        return ApplyFusedActivation(O, fusedActivation);
    }

    /// <inheritdoc/>
    public override Tensor Conv2D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, Layer.FusedActivation fusedActivation)
    {
        var O = Conv2DUsingIm2ColSliced(X, K, B, stride, pad);
        return ApplyFusedActivation(O, fusedActivation);
    }

    Tensor Conv2DUsingIm2ColSliced(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad)
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

        var O = NewTensor(X.shape.ApplyKernel(K.shape, stride, pad));
        var T = pointwiseConvolution ? null:                       // pointwise convolution is just O=X*K, we can completely skip Im2Col()
                NewTensor(O.batch, O.height, O.width, inChannels); // T holds slice of Im2Col(X)

        var outElements = O.batch * O.height * O.width;
        var inWidth = X.width;

        Assert.AreEqual(O.batch, batch);
        Assert.AreEqual(O.channels, B.flatWidth);
        Assert.AreEqual(O.channels, outChannels);

        // input & constants
        var pinX = Pin(X);
        var pinK = Pin(K);
        var pinB = Pin(B);

        // temporary slice
        var pinT  = pointwiseConvolution ? pinX  : Pin(T);

        // output
        var pinO = Pin(O);

        { // O = broadcast(B)
            // @TODO: move broadcast B directly into MatrixMultiplyJob
            var job = new VectorBroadcastJob();
            job.channels = outChannels;
            job.repeat = outElements;
            job.ScheduleXO(pinB, pinO);
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

                var firstPixel =             0 * strideX + offsetX;
                var lastPixel  = (T.width - 1) * strideX + offsetX;
                int numberOfPixelsToPadLeft  = SafeIntDivCeil(Math.Max(0, 0 - firstPixel                ), strideX);   // count(x * stride[0] + offsetX < 0)
                int numberOfPixelsToPadRight = SafeIntDivCeil(Math.Max(0,      lastPixel - (inWidth - 1)), strideX);   // count(x * stride[0] + offsetX >= inWidth)
                int numberOfPixelsToSkipFromInputRow = (offsetX >= 0 || strideX == 0) ? offsetX :                     // strideX == 0 protects against div-by-zero
                    lastPixel % strideX;                                                                              // first(x * stride[0] + offsetX >= 0) == (inWidth * stride[0] + offsetX) % stride[0]
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

                    Assert.AreEqual(numberOfPixelsToPadLeft,            assertNumberOfPixelsToPadLeft);
                    Assert.AreEqual(numberOfPixelsToPadRight,           assertNumberOfPixelsToPadRight);
                    Assert.AreEqual(numberOfPixelsToSkipFromInputRow,   assertNumberOfPixelsToSkipFromInputRow);
                    Assert.AreEqual(numberOfPixelsToCopyFromInputRow,   assertNumberOfPixelsToCopyFromInputRow);
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
                numberOfPixelsToPadLeft          = Math.Min(Math.Max(0, numberOfPixelsToPadLeft), T.width);
                numberOfPixelsToPadRight         = Math.Min(Math.Max(0, numberOfPixelsToPadRight), T.width - numberOfPixelsToPadLeft);
                numberOfPixelsToSkipFromInputRow = Math.Min(Math.Max(0, numberOfPixelsToSkipFromInputRow), X.width);
                numberOfPixelsToCopyFromInputRow = Math.Min(Math.Max(0, numberOfPixelsToCopyFromInputRow), X.width - numberOfPixelsToSkipFromInputRow);

                var job = new Im2ColSliceJob();
                job.inOutBatch       = batch;
                job.inOutChannels    = inChannels;
                job.inHeight         = X.height;
                job.inStrideN        = X.height * X.width * X.channels;
                job.inStrideH        =            X.width * X.channels;
                job.inStrideW        =                      X.channels;
                job.outWidth         = T.width;
                job.outStrideN       = T.height * T.width * T.channels;
                job.outStrideH       =            T.width * T.channels;
                job.strideX          = strideX;
                job.strideY          = strideY;
                job.offsetY          = offsetY;
                job.padLeft          = numberOfPixelsToPadLeft;
                job.padRight         = numberOfPixelsToPadRight;
                job.skipFromInputRow = numberOfPixelsToSkipFromInputRow;
                job.copyFromInputRow = numberOfPixelsToCopyFromInputRow;

                job.ScheduleXO(pinX, pinT, T.height, 16);
            }

            // O += slice(T) * slice(K)
            // With T=im2col(X) if pointwiseConvolution else T=X
            ScheduleSGEMM(
                pinT, outElements, inChannels,
                pinK, inChannels,  outChannels,
                pinO, outElements, outChannels, transposeA:false, transposeB:false, kernelOffset);

            kernelOffset += inChannels * outChannels;
        }

        //Calling Dispose on BurstTensorData will sync the fences, so this is a performance VS memory peak tradeoff here.
        T?.Dispose();

        return O;
    }

    /// <inheritdoc/>
    public override Tensor MaxPool2D(Tensor X, int[] pool, int[] stride, int[] pad)
    {
        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(pool.Length, 2);
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);

        var O = NewTensor(X.shape.ApplyPool(pool, stride, pad));

        var job = new MaxPool2DJob();
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

        var O = NewTensor(X.shape.ApplyPool(pool, stride, pad));

        var job = new AvgPool2DJob();
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

        var O = NewTensor(X.shape.ApplyKernel(K.shape, stride, pad));

        var job = new DepthwiseConv2DJob();

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

        var O = NewTensorLike(X);
        Assert.AreEqual(O.shape, X.shape);

        var job = new VectorBroadcastScaleBiasJob();
        job.inOutChannels = O.channels;
        job.alpha = 1f;
        job.ScheduleXSBO(X, S, B, O, O.length / O.channels, Math.Max(16, 1024 / O.channels));

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Relu(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var job = new ReluJob();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Relu6(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var job = new Relu6Job();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor LeakyRelu(Tensor X, float alpha)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var job = new LeakyReluJob();
        job.alpha = alpha;
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Tanh(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var job = new TanhJob();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

     /// <inheritdoc/>
    public override Tensor Softplus(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var job = new SoftplusJob();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Sigmoid(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var job = new SigmoidJob();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Elu(Tensor X, float alpha)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var job = new EluJob();
        job.alpha = alpha;
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Selu(Tensor X, float alpha, float gamma)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var job = new SeluJob();
        job.alpha = alpha;
        job.gamma = gamma;
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Swish(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var job = new SwishJob();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor PRelu(Tensor X, Tensor S)
    {
        var O = NewTensorLike(X);

        Assert.AreEqual(X.channels, O.channels);
        Assert.IsTrue((X.flatWidth == S.flatWidth) || (S.flatWidth == 1));

        var job = new PReluJob();
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
        if (X.shape.sequenceLength != 1 || X.shape.numberOfDirections != 1 || axis > X.shape.FirstNotIdentityFeatureDimensionIndex())
            return base.Softmax(X, axis);

        var O = NewTensor(X.shape);
        Assert.AreEqual(O.length, X.length);
        Assert.AreEqual(O.flatWidth, X.flatWidth);

        var pinX = Pin(X);
        var pinO = Pin(O);

        //Allocate memory
        Allocator memoryAllocator = Allocator.TempJob;
        var reduceOpMemSize = O.flatHeight * sizeof(float);
        s_maxValues.Malloc(reduceOpMemSize, JobsUtility.CacheLineSize, memoryAllocator);
        s_expSums.Malloc(reduceOpMemSize, JobsUtility.CacheLineSize, memoryAllocator);

        // x_max = X.max(axis=1)
        {
            var job = new ReduceMaxJob();
            job.offsetReduce = 1;
            job.reduceDim = O.flatWidth;
            job.ScheduleXO(pinX, s_maxValues, O.flatHeight, 1024);
        }
        // e_x_sum = Sum[exp(x[:,c] - x_max[:]), c]
        {
            var job = new ExpBiasReduceJob();
            job.inChannels = O.flatWidth;
            job.ScheduleXBO(pinX, s_maxValues, s_expSums, O.flatHeight, 1024);
        }
        // exp(x[n,c] - x_max[n]) / e_x_sum[n]
        {
            var job = new SoftmaxEndJob();
            job.inChannels = O.flatWidth;
            job.ScheduleXSBO(pinX, s_expSums, s_maxValues, pinO, O.length, 1024);
        }
        // free memory (in job)
        unsafe {
            var job = new MemFreeJob();
            job.allocator = memoryAllocator;
            job.buffer0 = s_maxValues.data;
            job.buffer1 = s_expSums.data;
            job.Schedule(pinO.fence);
        }

        s_maxValues.ClearState();
        s_expSums.ClearState();

        return O;
    }

    /// <inheritdoc/>
    public override Tensor LogSoftmax(Tensor X)
    {
        if (X.shape.sequenceLength != 1 || X.shape.numberOfDirections != 1)
            return base.LogSoftmax(X);

        var O = NewTensor(X.shape);
        Assert.AreEqual(O.length, X.length);
        Assert.AreEqual(O.flatWidth, X.flatWidth);

        var pinX = Pin(X);
        var pinO = Pin(O);

        //Allocate memory
        Allocator memoryAllocator = Allocator.TempJob;
        var reduceOpMemSize = O.flatHeight * sizeof(float);
        s_maxValues.Malloc(reduceOpMemSize, JobsUtility.CacheLineSize, memoryAllocator);
        s_expSums.Malloc(reduceOpMemSize, JobsUtility.CacheLineSize, memoryAllocator);

        // x_max = X.max(axis=1)
        {
            var job = new ReduceMaxJob();
            job.offsetReduce = 1;
            job.reduceDim = O.flatWidth;
            job.ScheduleXO(pinX, s_maxValues, O.flatHeight, 1024);
        }
        // e_x_sum = Sum[exp(x[:,c] - x_max[:]), c]
        {
            var job = new ExpBiasReduceJob();
            job.inChannels = O.flatWidth;
            job.ScheduleXBO(pinX, s_maxValues, s_expSums, O.flatHeight, 1024);
        }
        // exp(x[n,c] - x_max[n]) / e_x_sum[n]
        {
            var job = new LogSoftmaxEndJob();
            job.inChannels = O.flatWidth;
            job.ScheduleXSBO(pinX, s_expSums, s_maxValues, pinO, O.length, 1024);
        }
        // free memory (in job)
        unsafe {
            var job = new MemFreeJob();
            job.allocator = memoryAllocator;
            job.buffer0 = s_maxValues.data;
            job.buffer1 = s_expSums.data;
            job.Schedule(pinO.fence);
        }

        s_maxValues.ClearState();
        s_expSums.ClearState();

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Abs(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var job = new AbsJob();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Neg(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var job = new NegJob();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Ceil(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var job = new CeilJob();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Clip(Tensor X, float min, float max)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var job = new ClipJob();
        job.min = min;
        job.max = max;
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Floor(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var job = new FloorJob();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Round(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var job = new RoundJob();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Reciprocal(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var job = new ReciprocalJob();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Pow(Tensor X, float alpha)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var job = new PowJob();
        job.alpha = alpha;
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Exp(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var job = new ExpJob();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Log(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var job = new LogJob();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Sqrt(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var job = new SqrtJob();
        job.ScheduleXO(X, O , O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Acos(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var job = new AcosJob();
        job.ScheduleXO(X, O , O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Acosh(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var job = new AcoshJob();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Asin(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var job = new AsinJob();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Asinh(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var job = new AsinhJob();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Atan(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var job = new AtanJob();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Atanh(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var job = new AtanhJob();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Cos(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var job = new CosJob();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Cosh(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var job = new CoshJob();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Sin(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var job = new SinJob();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Sinh(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var job = new SinhJob();
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Tan(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var job = new TanJob();
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
            var job = new ScalarBroadcastAddJob();
            job.alpha = alpha;
            job.ScheduleXBO(X, Y, O, O.length, 1024);
        }
        else if (X.shape == O.shape && Y.shape == O.shape)
        {
            var job = new BroadcastAddJob();
            job.alpha = alpha;
            job.ScheduleXBO(X, Y, O, O.length, 1024);
        }
        else
        {
            var job = new ElementwiseAddJob();
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
            var job = new ScalarBroadcastMulJob();
            job.ScheduleXBO(X, Y, O, O.length, 1024);
        }
        else if (X.shape == O.shape && Y.shape == O.shape)
        {
            var job = new BroadcastMulJob();
            job.ScheduleXBO(X, Y, O, O.length, 1024);
        }
        else
        {
            var job = new ElementwiseMulJob();
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
            var job = new ScalarBroadcastDivJob();
            job.ScheduleXBO(X, Y, O, O.length, 1024);
        }
        else if (X.shape == O.shape && Y.shape == O.shape)
        {
            var job = new BroadcastDivJob();
            job.ScheduleXBO(X, Y, O, O.length, 1024);
        }
        else
        {
            var job = new ElementwiseDivJob();
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
            var job = new ScalarBroadcastPowJob();
            job.ScheduleXBO(X, Y, O, O.length, 1024);
        }
        else if (X.shape == O.shape && Y.shape == O.shape)
        {
            var job = new BroadcastPowJob();
            job.ScheduleXBO(X, Y, O, O.length, 1024);
        }
        else
        {
            var job = new ElementwisePowJob();
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
            var job = new ScalarBroadcastMinJob();
            job.ScheduleXBO(X, Y, O, O.length, 1024);
        }
        else if (X.shape == O.shape && Y.shape == O.shape)
        {
            var job = new BroadcastMinJob();
            job.ScheduleXBO(X, Y, O, O.length, 1024);
        }
        else
        {
            var job = new ElementwiseMinJob();
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
            var job = new ScalarBroadcastMaxJob();
            job.ScheduleXBO(X, Y, O, O.length, 1024);
        }
        else if (X.shape == O.shape && Y.shape == O.shape)
        {
            var job = new BroadcastMaxJob();
            job.ScheduleXBO(X, Y, O, O.length, 1024);
        }
        else
        {
            var job = new ElementwiseMaxJob();
            job.shapeO = O.shape;
            unsafe
            {
                AssignTensorStrides8D(X, job.stridesX);
                AssignTensorStrides8D(Y, job.stridesY);
            }
            job.ScheduleXBO(X, Y, O, O.length, 1024);
        }
    }

    /// <inheritdoc/>
    // O = tensors[0] + tensors[1] + ... + tensors[N-1]
    public override Tensor Add(Tensor[] tensors)
    {
        if (!TensorExtensions.AreAllTensorsConvertibleTo4D(tensors))
            return base.Add(tensors);

        var O = NewTensorLike(tensors);
        var X = tensors[0];

        for (int t = 1; t < tensors.Length; ++t)
        {
            BroadcastAdd(ref O, X, tensors[t]);
            X = O;
        }
        return O;
    }

    /// <inheritdoc/>
    // O = tensors[0] - tensors[1] - ... - tensors[N-1]
    public override Tensor Sub(Tensor[] tensors)
    {
        if (!TensorExtensions.AreAllTensorsConvertibleTo4D(tensors))
            return base.Sub(tensors);


        var O = NewTensorLike(tensors);
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


        var O = NewTensorLike(tensors);
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


        var O = NewTensorLike(tensors);
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


        var O = NewTensorLike(tensors);
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

        var O = NewTensorLike(tensors);
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

        var O = NewTensorLike(tensors);
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
        var O = NewTensor(shape);

        var job = new CopyJob();
        job.length = O.length;
        job.ScheduleXO(X, O);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Concat(Tensor[] tensors, int axis)
    {
        var concatShape = TensorExtensions.Concat(tensors, axis);
        var O = NewTensor(concatShape);

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
            var pinO = Pin(O);
            using (var ctx = new ParallelJobsContext(pinO))
            {
                for (int i = 0; i < tensors.Length; ++i)
                {
                    var pinX = Pin(tensors[i]);
                    var job = new CopyStrideJob();
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
        unsafe
        {
            int* starts = stackalloc int[TensorShape.MaxRank];
            int* ends = stackalloc int[TensorShape.MaxRank];
            int* strides = stackalloc int[TensorShape.MaxRank];
            TensorExtensions.Get8DParametersNoAlloc(X.shape, starts4Dor8D, starts, 0);
            TensorExtensions.Get8DParametersNoAlloc(X.shape, ends4Dor8D, ends, 1);
            TensorExtensions.Get8DParametersNoAlloc(X.shape, strides4Dor8D, strides, 1);

            var O = NewTensor(X.shape.ApplyStridedSlice8DUnsafeNoAlloc(starts, ends, strides));

            int* wrappedStartsIndices = ends; //reuse buffer to save a stack allocation.
            for (int i = 0; i < TensorShape.MaxRank; ++i)
                wrappedStartsIndices[i] = TensorExtensions.WrapIndex(starts[i], X.shape[i]);

            Assert.AreEqual(8, TensorShape.MaxRank);

            //TODO/Idea for further optimisation: Add a version using UnsafeUtility.MemCpyStride when many strides are 1 (starting from C amd going upward).
            if (strides[TensorShape.C] == 1)
            {
                var job = new GenericSliceJob();
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
                int numCopy = O.shape.length / O.shape.channels;
                job.ScheduleXO(X, O, numCopy, 64);
            }
            else
            {
                var job = new GenericStridedSliceJob();
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

    //TODO refactor when adding Burst support for other padding types (edge, reflect, symmetric)
    private Tensor ApplyBorderPadding(Tensor X, int[] pad, float constant)
    {
        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(pad.Length, 4);

        var O = NewTensor(X.shape.ApplyBorder(pad));

        int prePadX = Math.Max(0, pad[0]);
        int prePadY = Math.Max(0, pad[1]);
        int postPadX = Math.Max(0, pad[2]);
        int postPadY = Math.Max(0, pad[3]);

        // NOTE: negative "pad" variable will crop X tensor
        int preCropX  = Math.Max(0, -pad[0]);
        int preCropY  = Math.Max(0, -pad[1]);
        int postCropX = Math.Max(0, -pad[2]);
        int postCropY = Math.Max(0, -pad[3]);
        int croppedWidth = X.width - (preCropX + postCropX);
        int croppedHeight = X.height - (preCropY + postCropY);

        var pinX = Pin(X);
        var pinO = Pin(O);
        int numItemInARow = O.width * O.channels;
        int numItemInABatch = O.height * numItemInARow;

        using (var ctx = new ParallelJobsContext(pinO))
        {
            for (int b = 0; b < O.batch; ++b)
            {
                //PrePadY
                if (prePadY > 0)
                {
                    int numItemToPrepadInHeight = prePadY * O.width * O.channels;
                    int prepadOffset = numItemInABatch * b;
                    var jobPrePadY = new SetConstantPaddingJob();
                    jobPrePadY.constant = constant;
                    ctx.ScheduleO(jobPrePadY, pinO, prepadOffset, numItemToPrepadInHeight, 1024);
                }

                //PrePadX
                if (prePadX > 0)
                {
                    var jobPrePadX = new SetConstantPaddingWithStrideJob();
                    jobPrePadX.constant = constant;
                    jobPrePadX.length = prePadX * O.channels;
                    jobPrePadX.stride = O.width * O.channels;
                    ctx.ScheduleO(jobPrePadX, pinO, O.Index(b, prePadY, 0, 0), croppedHeight * prePadX * O.channels, 1024);
                }

                //Center X and Y
                {
                    int srcFloatOffset = X.Index(b, preCropY, preCropX, 0);
                    int dstFloatOffset = O.Index(b, prePadY, prePadX, 0);
                    int numFloatToCopy = O.channels * croppedWidth;
                    var jobCopy = new CopyStrideJob();
                    jobCopy.XStride = X.width * X.channels;
                    jobCopy.OStride = O.width * O.channels;
                    jobCopy.length = numFloatToCopy;
                    jobCopy.count = croppedHeight;
                    ctx.ScheduleXO(jobCopy, pinX, srcFloatOffset, pinO, dstFloatOffset);
                }

                //PostPadX
                if (postPadX > 0)
                {
                    var jobPostPadX = new SetConstantPaddingWithStrideJob();
                    jobPostPadX.constant = constant;
                    jobPostPadX.length = postPadX * O.channels;
                    jobPostPadX.stride = O.width * O.channels;
                    ctx.ScheduleO(jobPostPadX, pinO, O.Index(b, prePadY, O.width - postPadX, 0), croppedHeight * postPadX * O.channels, 1024);
                }

                //PostPadY
                if (postPadY > 0)
                {
                    int numItemToPostpadInHeight = postPadY * O.width * O.channels;
                    int postpadOffset = O.Index(b, O.height - postPadY, 0, 0);
                    var jobPostPadY = new SetConstantPaddingJob();
                    jobPostPadY.constant = constant;
                    ctx.ScheduleO(jobPostPadY, pinO, postpadOffset, numItemToPostpadInHeight, 1024);
                }
            }
        }
        return O;
    }

    /// <inheritdoc/>
    public override Tensor Border2D(Tensor X, int[] pad, float constant)
    {
        return ApplyBorderPadding(X, pad, constant);
    }

    /// <inheritdoc/>
    public override Tensor Transpose(Tensor X, int[] permutations)
    {
        permutations = TensorExtensions.Get8DPermutationsForNHWCPermutationsAndShape(X.shape, permutations);
        var O = NewTensor(X.shape.Permute(permutations));

        var job = new TransposeJob();
        job.shapeX = X.shape;
        job.shapeO = O.shape;
        unsafe {
            job.permutations[0] = permutations[0];
            job.permutations[1] = permutations[1];
            job.permutations[2] = permutations[2];
            job.permutations[3] = permutations[3];
            job.permutations[4] = permutations[4];
            job.permutations[5] = permutations[5];
            job.permutations[6] = permutations[6];
            job.permutations[7] = permutations[7];
        }
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor ReduceMean(Tensor X, int axis)
    {
        axis = X.shape.Axis(axis);
        var O = NewTensor(X.shape.Reduce(axis));

        int offsetReduce = 1;
        for (int i = 7; i >= axis; i--)
            offsetReduce *= O.shape[i];

        var job = new ReduceMeanJob();
        job.offsetReduce = offsetReduce;
        job.reduceDim = X.shape[axis];
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor ReduceSum(Tensor X, int axis)
    {
        axis = X.shape.Axis(axis);
        var O = NewTensor(X.shape.Reduce(axis));

        int offsetReduce = 1;
        for (int i = 7; i >= axis; i--)
            offsetReduce *= O.shape[i];

        var job = new ReduceSumJob();
        job.offsetReduce = offsetReduce;
        job.reduceDim = X.shape[axis];
        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor Tile(Tensor X, int[] repeats)
    {
        Tensor O = NewTensor(X.shape.Scale(repeats));

        var job = new TileJob();
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

        var O = NewTensor(shape);

        Assert.AreEqual(TensorShape.MaxRank, 8);

        var job = new GatherJob();
        job.axis = axis;
        job.shapeX = X.shape;
        job.shapeO = O.shape;
        job.ScheduleXBO(X, indices, O, O.length, 1024);

        return O;
    }

    /// <inheritdoc/>
    public override Tensor OneHot(Tensor X, int depth, float onValue, float offValue)
    {
        if (X.shape.sequenceLength != 1 || X.shape.numberOfDirections != 1)
            throw new NotImplementedException();

        bool isInput1D = (X.flatWidth == 1);

        Tensor O;
        if (isInput1D)
            O = NewTensor(X.flatHeight, depth);
        else
            O = NewTensor(X.flatHeight, 1, depth, X.flatWidth);


        var job = new OneHotJob();
        job.depth = depth;
        job.shapeX = X.shape;
        job.shapeO = O.shape;
        job.isInput1D = isInput1D;
        job.onValue = onValue;
        job.offValue = offValue;

        job.ScheduleXO(X, O, O.length, 1024);

        return O;
    }
}

} // namespace Barracuda
