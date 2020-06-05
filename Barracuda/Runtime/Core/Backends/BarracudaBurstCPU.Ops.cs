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
    JobHandle Dependencies(JobHandle job, JobHandle job2)
    {
        return JobHandle.CombineDependencies(job, job2);
    }
    JobHandle Dependencies(JobHandle job, JobHandle job2, JobHandle job3)
    {
        return JobHandle.CombineDependencies(job, job2, job3);
    }

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

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrY = &pinY.array[pinY.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                { // O = broadcast(0)
                    var job = new ZeroBroadcastJob();
                    job.O = ptrO;
                    job.repeat = O.length;
                    pinO.fence = job.Schedule(pinO.reuse);
                }

                // O += X * K
                if (m_UseBlas)
                {
                    pinO.fence = pinX.reuse = pinY.reuse =
                        blas.ScheduleSGEMM(
                            Dependencies(pinO.fence, pinX.fence, pinY.fence),
                            ptrX, X.flatHeight, X.flatWidth,
                            ptrY, Y.flatHeight, Y.flatWidth,
                            ptrO, O.flatHeight, O.flatWidth,
                            16, xTranspose, yTranspose);
                }
                else
                {
                    var job = new MatrixMultiplyJob();
                    job.A = ptrX;
                    job.AN = X.flatHeight;
                    job.AM = X.flatWidth;
                    job.B = ptrY;
                    job.BN = Y.flatHeight;
                    job.BM = Y.flatWidth;
                    job.C = ptrO;
                    job.CN = O.flatHeight;
                    job.CM = O.flatWidth;
                    job.transposeA = xTranspose;
                    job.transposeB = yTranspose;

                    pinO.fence = pinX.reuse = pinY.reuse =
                        job.Schedule(Dependencies(pinO.fence, pinX.fence, pinY.fence));
                }
            }
        }

        return O;
    }

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

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrW = &pinW.array[pinW.offset],
                ptrB = &pinB.array[pinB.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                { // O = broadcast(B)
                    // @TODO: move broadcast B directly into MatrixMultiplyJob
                    var job = new VectorBroadcastJob();
                    job.X = ptrB;
                    job.O = ptrO;
                    job.channels = O.flatWidth;
                    job.repeat = O.flatHeight;
                    pinO.fence = job.Schedule(pinO.reuse);
                }

                // O += X * K
                if (m_UseBlas)
                {
                    pinO.fence = pinX.reuse =
                        blas.ScheduleSGEMM(
                            Dependencies(pinO.fence, pinX.fence),
                            ptrX, X.flatHeight, X.flatWidth,
                            ptrW, W.flatHeight, W.flatWidth,
                            ptrO, O.flatHeight, O.flatWidth,
                            16);
                }
                else
                {
                    var job = new MatrixMultiplyJob();
                    job.A = ptrX;
                    job.AN = X.flatHeight;
                    job.AM = X.flatWidth;
                    job.B = ptrW;
                    job.BN = W.flatHeight;
                    job.BM = W.flatWidth;
                    job.C = ptrO;
                    job.CN = O.flatHeight;
                    job.CM = O.flatWidth;
                    job.transposeA = false;
                    job.transposeB = false;

                    pinO.fence = pinX.reuse = job.Schedule(Dependencies(pinO.fence, pinX.fence));
                }
            }
        }
        return ApplyFusedActivation(O, fusedActivation);
    }

    public override Tensor Conv2D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, Layer.FusedActivation fusedActivation)
    {
        var O = Conv2DUsingIm2ColSliced(X, K, B, stride, pad);
        return ApplyFusedActivation(O, fusedActivation);
    }

    Tensor Conv2DUsingIm2ColSliced(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad)
    {
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

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrT = &pinT.array[pinT.offset],
                ptrK = &pinK.array[pinK.offset],
                ptrB = &pinB.array[pinB.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                { // O = broadcast(B)
                    // @TODO: move broadcast B directly into MatrixMultiplyJob
                    var job = new VectorBroadcastJob();
                    job.X = ptrB;
                    job.O = ptrO;
                    job.channels = outChannels;
                    job.repeat = outElements;
                    pinO.fence = job.Schedule(pinO.reuse);
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
                float* ptrW = ptrK;
                for (int dy = 0; dy < kernelHeight; ++dy)
                    for (int dx = 0; dx < kernelWidth; ++dx)
                    {
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
                            job.X                = ptrX;
                            job.O                = ptrT;
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

                            pinX.reuse = pinT.fence = job.Schedule(T.height, 16,
                                Dependencies(pinT.reuse, pinX.fence, pinO.fence));  // NOTE: need to fence on O here
                                                                                    // due to T being shared between multiple iterations
                                                                                    // and its use for previous iteration on O has to complete before we can start filling T again
                        }

                        // O += slice(im2col(X)) * slice(K)
                        if (m_UseBlas)
                        {
                            pinO.fence = pinT.reuse =
                                blas.ScheduleSGEMM(
                                    Dependencies(pinO.fence, pinT.fence),
                                        ptrT, outElements, inChannels,
                                        ptrW, inChannels,  outChannels,
                                        ptrO, outElements, outChannels,
                                        16);
                        }
                        else
                        {
                            var job = new MatrixMultiplyJob();
                            job.A  = ptrT;
                            job.AN = outElements;
                            job.AM = inChannels;
                            job.B  = ptrW;
                            job.BN = inChannels;
                            job.BM = outChannels;
                            job.C  = ptrO;
                            job.CN = outElements;
                            job.CM = outChannels;
                            job.transposeA = false;
                            job.transposeB = false;

                            pinO.fence = pinT.reuse = job.Schedule(Dependencies(pinO.fence, pinT.fence));
                        }

                        ptrW += inChannels * outChannels;
                    }
            }
        }

        T?.Dispose();

        return O;
    }

    public override Tensor MaxPool2D(Tensor X, int[] pool, int[] stride, int[] pad)
    {
        Assert.AreEqual(pool.Length, 2);
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);

        var O = NewTensor(X.shape.ApplyPool(pool, stride, pad));

        var pinX = Pin(X);
        var pinO = Pin(O);

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                var job = new MaxPool2DJob();
                job.X = ptrX;
                job.O = ptrO;

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

                pinO.fence = pinX.reuse = job.Schedule(O.height, 4, Dependencies(pinO.reuse, pinX.fence));
            }
        }
        return O;
    }

    public override Tensor AvgPool2D(Tensor X, int[] pool, int[] stride, int[] pad)
    {
        Assert.AreEqual(pool.Length, 2);
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);

        var O = NewTensor(X.shape.ApplyPool(pool, stride, pad));

        var pinX = Pin(X);
        var pinO = Pin(O);

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                var job = new AvgPool2DJob();
                job.X = ptrX;
                job.O = ptrO;

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

                pinO.fence = pinX.reuse = job.Schedule(O.height, 4, Dependencies(pinO.reuse, pinX.fence));
            }
        }
        return O;
    }

    public override Tensor GlobalMaxPool2D(Tensor X)
    {
        return MaxPool2D(X, new[] {X.width, X.height}, new[] {1, 1}, new[] {0, 0, 0, 0});
    }

    public override Tensor GlobalAvgPool2D(Tensor X)
    {
        return AvgPool2D(X, new[] {X.width, X.height}, new[] {1, 1}, new[] {0, 0, 0, 0});
    }

    public override Tensor DepthwiseConv2D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, Layer.FusedActivation fusedActivation)
    {
        if (K.kernelDepth != 1)
            throw new NotImplementedException();

        Assert.AreEqual(K.kernelDepth, 1);
        Assert.AreEqual(K.kernelCount, X.channels);
        Assert.AreEqual(K.kernelCount, B.flatWidth);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);

        var O = NewTensor(X.shape.ApplyKernel(K.shape, stride, pad));

        var pinX = Pin(X);
        var pinK = Pin(K);
        var pinB = Pin(B);
        var pinO = Pin(O);

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrK = &pinK.array[pinK.offset],
                ptrB = &pinB.array[pinB.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                var job = new DepthwiseConv2DJob();
                job.X = ptrX;
                job.K = ptrK;
                job.B = ptrB;
                job.O = ptrO;

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

                pinO.fence = pinX.reuse = job.Schedule(O.height, 4, Dependencies(pinO.reuse, pinX.fence));
            }
        }
        return ApplyFusedActivation(O, fusedActivation);
    }

    public override Tensor ScaleBias(Tensor X, Tensor S, Tensor B)
    {
        Assert.AreEqual(S.shape, B.shape);
        bool isScalarOp = (S.length == 1);
        bool isSaVector = (S.length == S.channels);
        bool isVectorOp = (X.channels == S.channels && isSaVector);
        bool isTensorOp = (X.shape == S.shape);
        Assert.IsTrue(isScalarOp || isVectorOp || isTensorOp);

        var O = NewTensorLike(X);
        Assert.AreEqual(O.shape, X.shape);

        var pinX = Pin(X);
        var pinS = Pin(S);
        var pinB = Pin(B);
        var pinO = Pin(O);

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrS = &pinS.array[pinS.offset],
                ptrB = &pinB.array[pinB.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                var job = new VectorBroadcastScaleBiasJob();
                job.X = ptrX;
                job.optionalS = ptrS;
                job.optionalB = ptrB;
                job.O = ptrO;
                job.inOutChannels = O.channels;
                job.alpha = 1f;
                pinO.fence = pinX.reuse = job.Schedule(
                    O.length / O.channels, Math.Max(16, 1024 / O.channels), Dependencies(pinO.reuse, pinX.fence));
            }
        }
        return O;
    }

    public override Tensor Relu(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var pinX = Pin(X);
        var pinO = Pin(O);

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                var job = new ReluJob();
                job.X = ptrX;
                job.O = ptrO;
                pinO.fence = pinX.reuse = job.Schedule(O.length, 1024, Dependencies(pinO.reuse, pinX.fence));
            }
        }
        return O;
    }

    public override Tensor Relu6(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var pinX = Pin(X);
        var pinO = Pin(O);

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                var job = new Relu6Job();
                job.X = ptrX;
                job.O = ptrO;
                pinO.fence = pinX.reuse = job.Schedule(O.length, 1024, Dependencies(pinO.reuse, pinX.fence));
            }
        }
        return O;
    }

    public override Tensor LeakyRelu(Tensor X, float alpha)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var pinX = Pin(X);
        var pinO = Pin(O);

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                var job = new LeakyReluJob();
                job.X = ptrX;
                job.O = ptrO;
                job.alpha = alpha;
                pinO.fence = pinX.reuse = job.Schedule(O.length, 1024, Dependencies(pinO.reuse, pinX.fence));
            }
        }
        return O;
    }

    public override Tensor Tanh(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var pinX = Pin(X);
        var pinO = Pin(O);

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                var job = new TanhJob();
                job.X = ptrX;
                job.O = ptrO;
                pinO.fence = pinX.reuse = job.Schedule(O.length, 1024, Dependencies(pinO.reuse, pinX.fence));
            }
        }
        return O;
    }

    public override Tensor Sigmoid(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var pinX = Pin(X);
        var pinO = Pin(O);

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                var job = new SigmoidJob();
                job.X = ptrX;
                job.O = ptrO;
                pinO.fence = pinX.reuse = job.Schedule(O.length, 1024, Dependencies(pinO.reuse, pinX.fence));
            }
        }
        return O;
    }

    public override Tensor Elu(Tensor X, float alpha)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var pinX = Pin(X);
        var pinO = Pin(O);

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                var job = new EluJob();
                job.X = ptrX;
                job.O = ptrO;
                job.alpha = alpha;
                pinO.fence = pinX.reuse = job.Schedule(O.length, 1024, Dependencies(pinO.reuse, pinX.fence));
            }
        }
        return O;
    }

    public override Tensor Selu(Tensor X, float alpha, float gamma)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var pinX = Pin(X);
        var pinO = Pin(O);

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                var job = new SeluJob();
                job.X = ptrX;
                job.O = ptrO;
                job.alpha = alpha;
                job.gamma = gamma;
                pinO.fence = pinX.reuse = job.Schedule(O.length, 1024, Dependencies(pinO.reuse, pinX.fence));
            }
        }
        return O;
    }

    public override Tensor Swish(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var pinX = Pin(X);
        var pinO = Pin(O);

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                var job = new SwishJob();
                job.X = ptrX;
                job.O = ptrO;
                pinO.fence = pinX.reuse = job.Schedule(O.length, 1024, Dependencies(pinO.reuse, pinX.fence));
            }
        }
        return O;
    }

    public override Tensor PRelu(Tensor X, Tensor S)
    {
        var O = NewTensorLike(X);

        Assert.AreEqual(X.channels, O.channels);
        Assert.IsTrue((X.flatWidth == S.flatWidth) || (S.flatWidth == 1));

        var pinX = Pin(X);
        var pinS = Pin(S);
        var pinO = Pin(O);

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrS = &pinS.array[pinS.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                if (S.flatWidth == 1)
                {
                    var job = new LeakyReluJob();
                    job.X = ptrX;
                    job.O = ptrO;
                    job.alpha = *ptrS;
                    pinO.fence = pinX.reuse = job.Schedule(O.length, 1024, Dependencies(pinO.reuse, pinX.fence));
                }
                else
                {
                    var job = new PReluJob();
                    job.X = ptrX;
                    job.S = ptrS;
                    job.O = ptrO;
                    job.inOutChannels = O.channels;
                    pinO.fence = pinX.reuse = job.Schedule(O.length / O.channels, Math.Max(16, 1024 / O.channels), Dependencies(pinO.reuse, pinX.fence));
                }
            }
        }
        return O;
    }

    // public override Tensor Softmax(Tensor X)
    // {
    //     var O = NewTensor(X.shape.Flatten());
    //     Assert.AreEqual(O.length, X.length);
    //     Assert.AreEqual(O.flatWidth, X.flatWidth);

    //     var pinX = Pin(X);
    //     var pinO = Pin(O);

    //     unsafe
    //     {
    //         fixed (float*
    //             ptrX = &pinX.array[pinX.offset],
    //             ptrO = &pinO.array[pinO.offset])
    //         {
    //             var job = new SoftmaxJob();
    //             job.X = ptrX;
    //             job.O = ptrO;
    //             job.flatWidth = O.flatWidth;
    //             job.logistic = false;
    //             pinO.fence = pinX.reuse = job.Schedule(O.flatHeight, 1, Dependencies(pinO.reuse, pinX.fence));
    //         }
    //     }
    //     return O;
    // }

    public override Tensor Softmax(Tensor X)
    {
        var O = NewTensor(X.shape.Flatten());
        Assert.AreEqual(O.length, X.length);
        Assert.AreEqual(O.flatWidth, X.flatWidth);

        var pinX = Pin(X);
        var pinO = Pin(O);

        unsafe
        {
            var reduceOpMemSize = O.flatHeight * sizeof(float);
            float* maxValues = (float*)UnsafeUtility.Malloc(reduceOpMemSize, JobsUtility.CacheLineSize, Allocator.TempJob);
            float* expSums   = (float*)UnsafeUtility.Malloc(reduceOpMemSize, JobsUtility.CacheLineSize, Allocator.TempJob);

            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                // numpy implementation:
                //  x_max = X.max(axis=1)
                //  e_x = np.exp(X - x_max)
                //  X = e_x / e_x.sum(axis=1)

                JobHandle fence;
                { // x_max = X.max(axis=1)
                    var job = new ChannelReduceMaxJob();
                    job.X = ptrX;
                    job.O = maxValues;
                    job.inChannels = O.flatWidth;
                    fence = job.Schedule(O.flatHeight, 1, Dependencies(pinO.reuse, pinX.fence));
                }

                { //  e_x = np.exp(X - x_max)
                    var combinedFence = new JobHandle();
                    var job = new ScalarBroadcastBiasedExpJob();
                    job.X = ptrX;
                    job.B = maxValues;
                    job.O = ptrO;
                    job.alpha = -1f;
                    for (var n = 0; n < O.flatHeight; n++,
                        job.X += O.flatWidth, job.O += O.flatWidth, job.B++)
                    {
                        combinedFence = Dependencies(
                            job.Schedule(O.flatWidth, 64 * O.flatHeight, fence),
                            combinedFence);
                    }
                    fence = combinedFence;
                }
                pinX.reuse = fence;

                { // e_x_sum = e_x.sum(axis=1)
                    var job = new ChannelReduceSumJob();
                    job.X = ptrO;
                    job.O = expSums;
                    job.inChannels = O.flatWidth;
                    fence = job.Schedule(O.flatHeight, 1, fence);
                }

                { // O = e_x / e_x_sum
                    var combinedFence = new JobHandle();
                    var job = new ScalarBroadcastDivJob();
                    job.X = ptrO;
                    job.D = expSums;
                    job.O = ptrO;
                    for (var n = 0; n < O.flatHeight; n++,
                        job.X += O.flatWidth, job.O += O.flatWidth, job.D++)
                    {
                        combinedFence = Dependencies(
                            job.Schedule(O.flatWidth, 512 * O.flatHeight, fence),
                            combinedFence);
                    }
                    fence = combinedFence;
                }
                pinO.fence = fence;
            }

            UnsafeUtility.Free(expSums, Allocator.TempJob);
            UnsafeUtility.Free(maxValues, Allocator.TempJob);
        }

        return O;
    }

    public override Tensor LogSoftmax(Tensor X)
    {
        var O = NewTensor(X.shape.Flatten());
        Assert.AreEqual(O.length, X.length);
        Assert.AreEqual(O.flatWidth, X.flatWidth);

        var pinX = Pin(X);
        var pinO = Pin(O);

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                var job = new SoftmaxJob();
                job.X = ptrX;
                job.O = ptrO;
                job.flatWidth = O.flatWidth;
                job.logistic = true;
                pinO.fence = pinX.reuse = job.Schedule(O.flatHeight, 1, Dependencies(pinO.reuse, pinX.fence));
            }
        }
        return O;
    }

    public override Tensor Abs(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var pinX = Pin(X);
        var pinO = Pin(O);

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                var job = new AbsJob();
                job.X = ptrX;
                job.O = ptrO;
                pinO.fence = pinX.reuse = job.Schedule(O.length, 1024, Dependencies(pinO.reuse, pinX.fence));
            }
        }
        return O;
    }

    public override Tensor Neg(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var pinX = Pin(X);
        var pinO = Pin(O);

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                var job = new NegJob();
                job.X = ptrX;
                job.O = ptrO;
                pinO.fence = pinX.reuse = job.Schedule(O.length, 1024, Dependencies(pinO.reuse, pinX.fence));
            }
        }
        return O;
    }

    public override Tensor Ceil(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var pinX = Pin(X);
        var pinO = Pin(O);

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                var job = new CeilJob();
                job.X = ptrX;
                job.O = ptrO;
                pinO.fence = pinX.reuse = job.Schedule(O.length, 1024, Dependencies(pinO.reuse, pinX.fence));
            }
        }
        return O;
    }

    public override Tensor Clip(Tensor X, float min, float max)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var pinX = Pin(X);
        var pinO = Pin(O);

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                var job = new ClipJob();
                job.X = ptrX;
                job.O = ptrO;
                job.min = min;
                job.max = max;
                pinO.fence = pinX.reuse = job.Schedule(O.length, 1024, Dependencies(pinO.reuse, pinX.fence));
            }
        }
        return O;
    }

    public override Tensor Floor(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var pinX = Pin(X);
        var pinO = Pin(O);

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                var job = new FloorJob();
                job.X = ptrX;
                job.O = ptrO;
                pinO.fence = pinX.reuse = job.Schedule(O.length, 1024, Dependencies(pinO.reuse, pinX.fence));
            }
        }
        return O;
    }

    public override Tensor Reciprocal(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var pinX = Pin(X);
        var pinO = Pin(O);

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                var job = new ReciprocalJob();
                job.X = ptrX;
                job.O = ptrO;
                pinO.fence = pinX.reuse = job.Schedule(O.length, 1024, Dependencies(pinO.reuse, pinX.fence));
            }
        }
        return O;
    }

    public override Tensor Pow(Tensor X, float alpha)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var pinX = Pin(X);
        var pinO = Pin(O);

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                var job = new PowJob();
                job.X = ptrX;
                job.O = ptrO;
                job.alpha = alpha;
                pinO.fence = pinX.reuse = job.Schedule(O.length, 1024, Dependencies(pinO.reuse, pinX.fence));
            }
        }
        return O;
    }

    public override Tensor Exp(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var pinX = Pin(X);
        var pinO = Pin(O);

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                var job = new ExpJob();
                job.X = ptrX;
                job.O = ptrO;
                pinO.fence = pinX.reuse = job.Schedule(O.length, 1024, Dependencies(pinO.reuse, pinX.fence));
            }
        }
        return O;
    }

    public override Tensor Log(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var pinX = Pin(X);
        var pinO = Pin(O);

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                var job = new LogJob();
                job.X = ptrX;
                job.O = ptrO;
                pinO.fence = pinX.reuse = job.Schedule(O.length, 1024, Dependencies(pinO.reuse, pinX.fence));
            }
        }
        return O;
    }

    public override Tensor Sqrt(Tensor X)
    {
        var O = NewTensorLike(X);
        Assert.AreEqual(O.length, X.length);

        var pinX = Pin(X);
        var pinO = Pin(O);

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                var job = new SqrtJob();
                job.X = ptrX;
                job.O = ptrO;
                pinO.fence = pinX.reuse = job.Schedule(O.length, 1024, Dependencies(pinO.reuse, pinX.fence));
            }
        }
        return O;
    }

    protected virtual Tensor GenericBroadcast(Tensor X, TensorShape broadcastShape)
    {
        var O = NewTensor(broadcastShape);
        var pinX = Pin(X);
        var pinO = Pin(O);

        bool noRemainder = (O.length % X.channels == 0);
        bool isXaVector = (X.channels == X.length) && noRemainder;
        bool isXaScalar = (X.length == 1);

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                JobHandle fence;
                var dependsOn = Dependencies(pinO.reuse, pinX.fence);
                if (isXaScalar || isXaVector)
                {
                    var job = new VectorBroadcastJob();
                    job.X = ptrX;
                    job.O = ptrO;
                    job.channels = X.channels;
                    job.repeat = O.length / X.channels;
                    fence = job.Schedule(dependsOn);
                }
                else // slow generic broadcast
                {
                    var job = new GenericBroadcastJob();
                    job.X = ptrX;
                    job.O = ptrO;
                    job.shapeO = O.shape;
                    job.strideBatchX = (X.batch == 1) ? 0 : X.height * X.width * X.channels;
                    job.strideHeightX = (X.height == 1) ? 0 : X.width * X.channels;
                    job.strideWidthX = (X.width == 1) ? 0 : X.channels;
                    job.strideChannelsX = (X.channels == 1) ? 0 : 1;
                    fence = job.Schedule(O.length, 1024, dependsOn);
                }
                pinO.fence = pinX.reuse = fence;
            }
        }
        return O;
    }

    private void BroadcastAdd(ref Tensor O, Tensor X, Tensor Y, float alpha = 1f)
    {
        var pinX = Pin(X);
        var pinY = Pin(Y);
        var pinO = Pin(O);

        bool inOutShapesMatch = O.shape == X.shape;
        bool allShapesMatch = X.shape == Y.shape && inOutShapesMatch;
        bool isVectorOp = X.channels == Y.channels && Y.channels == Y.length && inOutShapesMatch;
        bool isScalarOp = Y.length == 1 && inOutShapesMatch;

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrY = &pinY.array[pinY.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                JobHandle fence;
                var dependsOn = Dependencies(pinO.reuse, pinX.fence, pinY.fence);
                if (allShapesMatch)
                {
                    var job = new ElementwiseAddJob();
                    job.X = ptrX;
                    job.B = ptrY;
                    job.O = ptrO;
                    job.alpha = alpha;
                    fence = job.Schedule(O.length, 1024, dependsOn);
                }
                else if (isScalarOp)
                {
                    var job = new ScalarBroadcastAddJob();
                    job.X = ptrX;
                    job.B = ptrY;
                    job.O = ptrO;
                    job.alpha = alpha;
                    fence = job.Schedule(O.length, 1024, dependsOn);
                }
                else if (isVectorOp)
                {
                    var job = new VectorBroadcastScaleBiasJob();
                    job.X = ptrX;
                    job.optionalS = null;
                    job.optionalB = ptrY;
                    job.O = ptrO;
                    job.inOutChannels = O.channels;
                    job.alpha = alpha;
                    fence = job.Schedule(O.length / O.channels, Math.Max(16, 1024 / O.channels), dependsOn);
                }
                else // generic broadcast
                {
                    var job = new GenericBroadcastAddJob();
                    job.X = ptrX;
                    job.B = ptrY;
                    job.O = ptrO;
                    job.shapeO = O.shape;
                    job.strideBatchX = (X.batch == 1) ? 0 : X.height * X.width * X.channels;
                    job.strideHeightX = (X.height == 1) ? 0 : X.width * X.channels;
                    job.strideWidthX = (X.width == 1) ? 0 : X.channels;
                    job.strideChannelsX = (X.channels == 1) ? 0 : 1;
                    job.strideBatchB = (Y.batch == 1) ? 0 : Y.height * Y.width * Y.channels;
                    job.strideHeightB = (Y.height == 1) ? 0 : Y.width * Y.channels;
                    job.strideWidthB = (Y.width == 1) ? 0 : Y.channels;
                    job.strideChannelsB = (Y.channels == 1) ? 0 : 1;
                    job.alpha = alpha;
                    fence = job.Schedule(O.length, 1024, dependsOn);
                }

                pinO.fence = pinX.reuse = pinY.reuse = fence;
            }
        }
    }

    private void BroadcastSub(ref Tensor O, Tensor X, Tensor Y)
    {
        BroadcastAdd(ref O, X, Y, -1f);
    }

    private void BroadcastMul(ref Tensor O, Tensor X, Tensor Y)
    {
        var pinX = Pin(X);
        var pinY = Pin(Y);
        var pinO = Pin(O);

        bool inOutShapesMatch = O.shape == X.shape;
        bool allShapesMatch = X.shape == Y.shape && inOutShapesMatch;
        bool isVectorOp = X.channels == Y.channels && Y.channels == Y.length && inOutShapesMatch;
        bool isScalarOp = Y.length == 1 && inOutShapesMatch;

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrY = &pinY.array[pinY.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                JobHandle fence;
                var dependsOn = Dependencies(pinO.reuse, pinX.fence, pinY.fence);
                if (allShapesMatch)
                {
                    var job = new ElementwiseMulJob();
                    job.X = ptrX;
                    job.S = ptrY;
                    job.O = ptrO;
                    fence = job.Schedule(O.length, 1024, dependsOn);
                }
                else if (isScalarOp)
                {
                    var job = new ScalarBroadcastMulJob();
                    job.X = ptrX;
                    job.S = ptrY;
                    job.O = ptrO;
                    fence = job.Schedule(O.length, 1024, dependsOn);
                }
                else if (isVectorOp)
                {
                    var job = new VectorBroadcastScaleBiasJob();
                    job.X = ptrX;
                    job.optionalS = ptrY;
                    job.optionalB = null;
                    job.O = ptrO;
                    job.inOutChannels = O.channels;
                    fence = job.Schedule(O.length / O.channels, Math.Max(16, 1024 / O.channels), dependsOn);
                }
                else // generic broadcast
                {
                    var job = new GenericBroadcastMulJob();
                    job.X = ptrX;
                    job.S = ptrY;
                    job.O = ptrO;
                    job.shapeO = O.shape;
                    job.strideBatchX = (X.batch == 1) ? 0 : X.height * X.width * X.channels;
                    job.strideHeightX = (X.height == 1) ? 0 : X.width * X.channels;
                    job.strideWidthX = (X.width == 1) ? 0 : X.channels;
                    job.strideChannelsX = (X.channels == 1) ? 0 : 1;
                    job.strideBatchS = (Y.batch == 1) ? 0 : Y.height * Y.width * Y.channels;
                    job.strideHeightS = (Y.height == 1) ? 0 : Y.width * Y.channels;
                    job.strideWidthS = (Y.width == 1) ? 0 : Y.channels;
                    job.strideChannelsS = (Y.channels == 1) ? 0 : 1;
                    fence = job.Schedule(O.length, 1024, dependsOn);
                }

                pinO.fence = pinX.reuse = pinY.reuse = fence;
            }
        }
    }

    private void BroadcastDiv(ref Tensor O, Tensor X, Tensor Y)
    {
        var T = (X.shape != O.shape) ? GenericBroadcast(X, O.shape): X;
        var U = (Y.shape != O.shape) ? GenericBroadcast(Y, O.shape): Y;

        var pinX = Pin(T);
        var pinY = Pin(U);
        var pinO = Pin(O);

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrY = &pinY.array[pinY.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                var job = new ElementwiseDivJob();
                job.X = ptrX;
                job.D = ptrY;
                job.O = ptrO;
                pinO.fence = pinX.reuse = pinY.reuse =
                    job.Schedule(O.length, 1024, Dependencies(pinO.reuse, pinX.fence, pinY.fence));
            }
        }

        if (T != X) T.Dispose();
        if (U != Y) U.Dispose();
    }

    private void BroadcastPow(ref Tensor O, Tensor X, Tensor Y)
    {
        var T = (X.shape != O.shape) ? GenericBroadcast(X, O.shape): X;
        var U = (Y.shape != O.shape) ? GenericBroadcast(Y, O.shape): Y;

        var pinX = Pin(T);
        var pinY = Pin(U);
        var pinO = Pin(O);

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrY = &pinY.array[pinY.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                var job = new ElementwisePowJob();
                job.X = ptrX;
                job.E = ptrY;
                job.O = ptrO;
                pinO.fence = pinX.reuse = pinY.reuse =
                    job.Schedule(O.length, 1024, Dependencies(pinO.reuse, pinX.fence, pinY.fence));
            }
        }

        if (T != X) T.Dispose();
        if (U != Y) U.Dispose();
    }

    private void BroadcastMin(ref Tensor O, Tensor X, Tensor Y)
    {
        var T = (X.shape != O.shape) ? GenericBroadcast(X, O.shape): X;
        var U = (Y.shape != O.shape) ? GenericBroadcast(Y, O.shape): Y;

        var pinX = Pin(T);
        var pinY = Pin(U);
        var pinO = Pin(O);

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrY = &pinY.array[pinY.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                var job = new ElementwiseMinJob();
                job.X = ptrX;
                job.Y = ptrY;
                job.O = ptrO;
                pinO.fence = pinX.reuse = pinY.reuse =
                    job.Schedule(O.length, 1024, Dependencies(pinO.reuse, pinX.fence, pinY.fence));
            }
        }

        if (T != X) T.Dispose();
        if (U != Y) U.Dispose();
    }

    private void BroadcastMax(ref Tensor O, Tensor X, Tensor Y)
    {
        var T = (X.shape != O.shape) ? GenericBroadcast(X, O.shape): X;
        var U = (Y.shape != O.shape) ? GenericBroadcast(Y, O.shape): Y;

        var pinX = Pin(T);
        var pinY = Pin(U);
        var pinO = Pin(O);

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrY = &pinY.array[pinY.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                var job = new ElementwiseMaxJob();
                job.X = ptrX;
                job.Y = ptrY;
                job.O = ptrO;
                pinO.fence = pinX.reuse = pinY.reuse =
                    job.Schedule(O.length, 1024, Dependencies(pinO.reuse, pinX.fence, pinY.fence));
            }
        }

        if (T != X) T.Dispose();
        if (U != Y) U.Dispose();
    }

    // O = tensors[0] + tensors[1] + ... + tensors[N-1]
    public override Tensor Add(Tensor[] tensors)
    {
        var O = NewTensorLike(tensors);
        var X = tensors[0];

        for (int t = 1; t < tensors.Length; ++t)
        {
            BroadcastAdd(ref O, X, tensors[t]);
            X = O;
        }
        return O;
    }

    // O = tensors[0] - tensors[1] - ... - tensors[N-1]
    public override Tensor Sub(Tensor[] tensors)
    {
        var O = NewTensorLike(tensors);
        var X = tensors[0];

        for (int t = 1; t < tensors.Length; ++t)
        {
            BroadcastSub(ref O, X, tensors[t]);
            X = O;
        }
        return O;
    }

    // O = tensors[0] * tensors[1] * ... * tensors[N-1]
    public override Tensor Mul(Tensor[] tensors)
    {
        var O = NewTensorLike(tensors);
        var X = tensors[0];

        for (int t = 1; t < tensors.Length; ++t)
        {
            BroadcastMul(ref O, X, tensors[t]);
            X = O;
        }
        return O;
    }

    // O = tensors[0] / tensors[1] / ... / tensors[N-1]
    public override Tensor Div(Tensor[] tensors)
    {
        var O = NewTensorLike(tensors);
        var X = tensors[0];

        for (int t = 1; t < tensors.Length; ++t)
        {
            BroadcastDiv(ref O, X, tensors[t]);
            X = O;
        }
        return O;
    }

    // O = tensors[0] ^ tensors[1] ^ ... ^ tensors[N-1]
    public override Tensor Pow(Tensor[] tensors)
    {
        var O = NewTensorLike(tensors);
        var X = tensors[0];

        for (int t = 1; t < tensors.Length; ++t)
        {
            BroadcastPow(ref O, X, tensors[t]);
            X = O;
        }
        return O;
    }

    // O = min(tensors[0], tensors[1],  ... , tensors[N-1])
    public override Tensor Min(Tensor[] tensors)
    {
        var O = NewTensorLike(tensors);
        var X = tensors[0];

        for (int t = 1; t < tensors.Length; ++t)
        {
            BroadcastMin(ref O, X, tensors[t]);
            X = O;
        }
        return O;
    }

    // O = max(tensors[0], tensors[1],  ... , tensors[N-1])
    public override Tensor Max(Tensor[] tensors)
    {
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

    protected override Tensor CopyAndReshape(Tensor X, TensorShape shape)
    {
        Assert.AreEqual(X.length, shape.length);
        var O = NewTensor(shape);

        var pinX = Pin(X);
        var pinO = Pin(O);

        unsafe
        {
            fixed (float*
                ptrX = &pinX.array[pinX.offset],
                ptrO = &pinO.array[pinO.offset])
            {
                var job = new CopyJob();
                job.X = ptrX;
                job.O = ptrO;
                job.length = O.length;
                pinO.fence = pinX.reuse = job.Schedule(Dependencies(pinO.reuse, pinX.fence));
            }
        }
        return O;
    }
}

} // namespace Barracuda
