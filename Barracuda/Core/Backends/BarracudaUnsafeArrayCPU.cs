using UnityEngine;
using UnityEngine.Assertions;
using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using UnityEngine.Profiling;

namespace Barracuda {

public class UnsafeArrayTensorData : SharedArrayTensorData
{
    protected bool m_Readonly = false;

    // creates new array
    public UnsafeArrayTensorData(int count) : base(new float[count])
    {
    }

    // creates new array
    public UnsafeArrayTensorData(TensorShape shape) : this(shape.length)
    {
    }

    // uses shared array
    public UnsafeArrayTensorData(ArrayTensorData sharedArray) : base(sharedArray.array, 0, -1)
    {
    }

    // uses shared array
    public UnsafeArrayTensorData(SharedArrayTensorData sharedArray) : base(sharedArray.array, sharedArray.offset, sharedArray.count)
    {
        m_Readonly = true;
    }

    ~UnsafeArrayTensorData()
    {
        Dispose();
    }

    public override void Dispose()
    {
        m_Array = null;
        m_Offset = m_Count = 0;
    }

    public override void Reserve(int count)
    {
        if (m_Readonly)
        {
            base.Reserve(count);
            return;
        }

        if (count > m_Array.Length)
            m_Array = new float[count];

        m_Offset = 0;
        m_Count = count;
    }

    public override void Upload(float[] data, int offset = 0, int count = -1)
    {
        if (m_Readonly)
        {
            base.Upload(data, offset, count);
            return;
        }

        Assert.IsTrue(offset >= 0);
        if (count < 0)
            count = data.Length - offset;

        if (m_Array == data && m_Offset == offset && m_Count == count)
            return;

        Reserve(count);

        Array.Copy(data, offset, m_Array, m_Offset, m_Count);
    }

    public override float[] Download(int count)
    {
        //;;D.logStackTraceEnabled = true;
        //;;D.Log("Download UnsafeArrayTensorData " + count + " from " + m_Count + " @ " + ToString());
        //;;D.logStackTraceEnabled = false;

        if (!m_Readonly && count <= m_Array.Length && m_Offset == 0)
            return m_Array;

        return base.Download(count);
    }

    public override string ToString()
    {
        return string.Format("(CPU unsafe: {0} length: {1} offset: {2} uploaded: {3})",
            GetHashCode(), m_Array.Length, m_Offset, m_Count);
    }
}


public class UnsafeArrayCPUOps : ReferenceCPUOps
{
    BLASPlugin blas = null;

    const int BlockSize = 32;

    internal InnerLoop m_InnerLoop = new InnerLoop();


    public UnsafeArrayCPUOps(ITensorAllocator allocator = null)
    : base(allocator)
    {
        blas = BLASPluginFactory.CreateBLASPlugin();
    }

    public static UnsafeArrayTensorData Pin(Tensor X)
    {
        X.FlushCache();

        var onDevice = X.tensorOnDevice as UnsafeArrayTensorData;
        if (onDevice == null)
        {
            // try to adopt CPU arrays
            var asSharedArray = X.tensorOnDevice as SharedArrayTensorData;
            var asArray = X.tensorOnDevice as ArrayTensorData;
            if (asSharedArray != null) X.CastOnDevice(new UnsafeArrayTensorData(asSharedArray)); // adopt unsafe array without copy
            else if (asArray != null) X.CastOnDevice(new UnsafeArrayTensorData(asArray)); // adopt unsafe array without copy
            else
                X.PinToDeviceAndUploadToIt(new UnsafeArrayTensorData(X.shape)); // device is uncompatible, create new array and upload
        }

        return X.tensorOnDevice as UnsafeArrayTensorData;
    }

    // ---------------------------------------------------------------------------------

    // NOTE: Parallel.For with small number of work items results in varying and often worse performance
    // As a workaround we will fallback to 'for' loop when number of work items is below heuristically determined threshold
    private static void Parallel_For(long begin, long end, Action<long> body)
    {
        if (end - begin > 2048) // threshold determined heuristically. If work items < threshold, then for loop is faster than Parallel.For()
            Parallel.For(begin, end, body);
        else
            for(var n = begin; n < end; n++)
                body(n);
    }

    public override Tensor Neg(Tensor X)
    {
        // f(x) = -x
        var O = NewTensorLike(X);
        var end = X.length;
        const int unrollSize = 4;

        unsafe
        {
            fixed (float*
                xPtr = &Pin(X).array[Pin(X).offset],
                oPtr = &Pin(O).array[Pin(O).offset])
            {
                NegInnerLoop(end, unrollSize, xPtr, oPtr);

                // Remainder
                for (int i = (end / unrollSize) * unrollSize; i < end; ++i)
                {
                    oPtr[i] = -xPtr[i];
                }
            }
        }

        return O;
    }

    private unsafe void NegInnerLoop(int length, int unrollSize, float* xPtr, float* oPtr)
    {
        Assert.AreEqual(unrollSize, 4);

        m_InnerLoop.SetState(unrollSize, xPtr, oPtr);

        Parallel_For(0L, length / unrollSize, m_InnerLoop.m_negInnerLoopDelegate);
    }

    public override Tensor Relu(Tensor X)
    {
        // f(x) = max(x,0.0)
        var O = NewTensorLike(X);
        var end = X.length;
        const int unrollSize = 4;

        unsafe
        {
            fixed (float*
                xPtr = &Pin(X).array[Pin(X).offset],
                oPtr = &Pin(O).array[Pin(O).offset])
            {
                ReluInnerLoop(end, unrollSize, xPtr, oPtr);

                // Remainder
                for (int i = (end / unrollSize) * unrollSize; i < end; ++i)
                {
                    float v = xPtr[i];
                    v = Mathf.Max(v, 0.0f);
                    oPtr[i] = v;
                }
            }
        }

        return O;
    }

    private unsafe void ReluInnerLoop(int length, int unrollSize, float* xPtr, float* oPtr)
    {
        Assert.AreEqual(unrollSize, 4);

        m_InnerLoop.SetState(unrollSize, xPtr, oPtr);

        Parallel_For(0L, length / unrollSize, m_InnerLoop.m_reluInnerLoopDelegate);
    }

    public override Tensor Elu(Tensor X, float alpha)
    {
        // f(x) = alpha * (exp(x) - 1.) for x < 0, f(x) = x for x >= 0
        // "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)", DA Clevert, 2015
        // https://arxiv.org/abs/1511.07289
        var O = NewTensorLike(X);
        var end = X.length;
        const int unrollSize = 4;

        unsafe
        {
            fixed (float*
                xPtr = &Pin(X).array[Pin(X).offset],
                oPtr = &Pin(O).array[Pin(O).offset])
            {
                EluInnerLoop(end, unrollSize, xPtr, oPtr, alpha);

                // Remainder
                for (int i = (end / unrollSize) * unrollSize; i < end; ++i)
                {
                    float v = xPtr[i];
                    if (v <= 0)
                        v = alpha * (Mathf.Exp(v) - 1f);
                    oPtr[i] = v;
                }
            }
        }

        return O;
    }

    private unsafe void EluInnerLoop(int length, int unrollSize, float* xPtr, float* oPtr, float alpha)
    {
        Assert.AreEqual(unrollSize, 4);

        m_InnerLoop.SetState(unrollSize, xPtr, oPtr, alpha);

        Parallel_For(0L, length / unrollSize, m_InnerLoop.m_eluInnerLoopDelegate);
    }


    public override Tensor PRelu(Tensor X, Tensor S)
    {
        Assert.IsTrue((X.flatWidth == S.flatWidth) || (S.flatWidth == 1));

        // f(x) = x for x >= 0, f(x) = slope*x for x <= 0
        var O = NewTensorLike(X);
        var end = X.length;
        const int unrollSize = 4;

        unsafe
        {
            fixed (float*
                xPtr = &Pin(X).array[Pin(X).offset],
                oPtr = &Pin(O).array[Pin(O).offset],
                wPtr = &Pin(S).array[Pin(S).offset])
            {
                PReluInnerLoop(end, unrollSize, xPtr, X.length, oPtr, wPtr, S.length);

                // Remainder
                for (int i = (end / unrollSize) * unrollSize; i < end; ++i)
                {
                    float v = xPtr[i];
                    float slope = wPtr[i % S.length];
	                v = Mathf.Max(0.0f, v) + slope * Mathf.Min(0.0f, v);
                    oPtr[i] = v;
                }
            }
        }

        return O;
    }

    private unsafe void PReluInnerLoop(int length, int unrollSize, float* xPtr, int xLen, float* oPtr, float* wPtr, int wLen)
    {
        Assert.AreEqual(unrollSize, 4);

        m_InnerLoop.SetState(unrollSize, oPtr, xPtr, xLen, wPtr, wLen);

        Parallel_For(0L, length / unrollSize, m_InnerLoop.m_preluInnerLoopDelegate);
    }

    public override Tensor Sigmoid(Tensor X)
    {
        // f(x) = 1 / (1 + exp(-x))
        var O = NewTensorLike(X);
        var end = X.length;
        const int unrollSize = 4;

        unsafe
        {
            fixed (float*
                xPtr = &Pin(X).array[Pin(X).offset],
                oPtr = &Pin(O).array[Pin(O).offset])
            {
                SigmoidInnerLoop(end, unrollSize, xPtr, oPtr);

                // Remainder
                for (int i = (end / unrollSize) * unrollSize; i < end; ++i)
                {
                    float v = xPtr[i];
                    v = 1f / (1f + Mathf.Exp(-v));
                    oPtr[i] = v;
                }
            }
        }

        return O;
    }

    private unsafe void SigmoidInnerLoop(int length, int unrollSize, float* xPtr, float* oPtr)
    {
        Assert.AreEqual(unrollSize, 4);

        m_InnerLoop.SetState(unrollSize, xPtr, oPtr);

        Parallel_For(0L, length / unrollSize, m_InnerLoop.m_sigmoidInnerLoopDelegate);
    }

    public override Tensor Swish(Tensor X)
    {
        // f(x) = sigmoid(x) * x = x / (1 + exp(-x))
        // "Searching for Activation Functions". P Ramachandran, 2017
        // https://arxiv.org/abs/1710.05941

        var O = NewTensorLike(X);
        var end = X.length;
        const int unrollSize = 4;

        unsafe
        {
            fixed (float*
                xPtr = &Pin(X).array[Pin(X).offset],
                oPtr = &Pin(O).array[Pin(O).offset])
            {
                SwishInnerLoop(end, unrollSize, xPtr, oPtr);

                // Remainder
                for (int i = (end / unrollSize) * unrollSize; i < end; ++i)
                {
                    float v = xPtr[i];
                    v = v / (1f + Mathf.Exp(-v));
                    oPtr[i] = v;
                }
            }
        }

        return O;
    }

    private unsafe void SwishInnerLoop(int length, int unrollSize, float* xPtr, float* oPtr)
    {
        Assert.AreEqual(unrollSize, 4);

        m_InnerLoop.SetState(unrollSize, xPtr, oPtr);

        Parallel_For(0L, length / unrollSize, m_InnerLoop.m_swishInnerLoopDelegate);
    }

    public override Tensor Exp(Tensor X)
    {
        var O = NewTensorLike(X);
        var end = X.length;
        const int unrollSize = 4;

        unsafe
        {
            fixed (float*
                xPtr = &Pin(X).array[Pin(X).offset],
                oPtr = &Pin(O).array[Pin(O).offset])
            {
                ExpInnerLoop(end, unrollSize, xPtr, oPtr);

                // Remainder
                for (int i = (end / unrollSize) * unrollSize; i < end; ++i)
                {
                    float v = xPtr[i];
                    v = Mathf.Exp(v);
                    oPtr[i] = v;
                }
            }
        }

        return O;
    }

    private unsafe void ExpInnerLoop(int length, int unrollSize, float* xPtr, float* oPtr)
    {
        Assert.AreEqual(unrollSize, 4);

        m_InnerLoop.SetState(unrollSize, xPtr, oPtr);

        Parallel_For(0L, length / unrollSize, m_InnerLoop.m_expInnerLoopDelegate);
    }

    public override Tensor Sqrt(Tensor X)
    {
        var O = NewTensorLike(X);
        var end = X.length;
        const int unrollSize = 4;

        unsafe
        {
            fixed (float*
                xPtr = &Pin(X).array[Pin(X).offset],
                oPtr = &Pin(O).array[Pin(O).offset])
            {
                SqrtInnerLoop(end, unrollSize, xPtr, oPtr);

                // Remainder
                for (int i = (end / unrollSize) * unrollSize; i < end; ++i)
                {
                    float v = xPtr[i];
                    v = Mathf.Sqrt(v);
                    oPtr[i] = v;
                }
            }
        }

        return O;
    }

    private unsafe void SqrtInnerLoop(int length, int unrollSize, float* xPtr, float* oPtr)
    {
        Assert.AreEqual(unrollSize, 4);

        m_InnerLoop.SetState(unrollSize, xPtr, oPtr);

        Parallel_For(0L, length / unrollSize, m_InnerLoop.m_sqrtInnerLoopDelegate);
    }

    public override Tensor Tanh(Tensor X)
    {
        var O = NewTensorLike(X);
        var end = X.length;
        const int unrollSize = 4;

        unsafe
        {
            fixed (float*
                xPtr = &Pin(X).array[Pin(X).offset],
                oPtr = &Pin(O).array[Pin(O).offset])
            {
                TanhInnerLoop(end, unrollSize, xPtr, oPtr);

                // Remainder
                for (int i = (end / unrollSize) * unrollSize; i < end; ++i)
                {
                    float v = xPtr[i];
                    v = MathfEx.tanh(v);
                    oPtr[i] = v;
                }
            }
        }

        return O;
    }

    private unsafe void TanhInnerLoop(int length, int unrollSize, float* xPtr, float* oPtr)
    {
        Assert.AreEqual(unrollSize, 4);

        m_InnerLoop.SetState(unrollSize, xPtr, oPtr);

        Parallel_For(0L, length / unrollSize, m_InnerLoop.m_tanhInnerLoopDelegate);
    }

    private Tensor GetOutputTensorFromBroadcast(Tensor[] tensors)
    {
        Assert.IsTrue(tensors.Length > 0);

        var O = NewTensor(TensorExtensions.MaxShape(tensors));
        foreach (var t in tensors)
        {
            Assert.IsTrue((t.batch    == 1) || (t.batch    == O.batch));
            Assert.IsTrue((t.height   == 1) || (t.height   == O.height));
            Assert.IsTrue((t.width    == 1) || (t.width    == O.width));
            Assert.IsTrue((t.channels == 1) || (t.channels == O.channels));
        }

        return O;
    }

    private bool CanUseModuloForBroadcasting(TensorShape o, TensorShape a)
    {
        if (o == a)
           return true;

        int firstDimensionMismatchInMemoryOrder = -1;
        for (int i=3; i > 0; --i)
        {
            if (o[i] != a[i])
            {
                firstDimensionMismatchInMemoryOrder = i;
                break;
            }
        }

        for (int i = firstDimensionMismatchInMemoryOrder; i > 0; --i)
        {
            if (a[i] != 1)
                return false;
        }

        return true;
    }

    private bool CanUseModuloForBroadcasting(TensorShape o, TensorShape a, TensorShape b)
    {
        return CanUseModuloForBroadcasting(o,a) && CanUseModuloForBroadcasting(o,b);
    }

    private Tensor ApplyElementwiseWithBroadcast(Tensor[] tensors, Func<float,float,float> opRemainder, Action<long> opInnerLoop, Action<long> opInnerLoopNoBroadcast)
    {
        var O = GetOutputTensorFromBroadcast(tensors);
        var A = tensors[0];

        unsafe
        {
            fixed (float*
                t0Ptr = &Pin(A).array[Pin(A).offset],
                oPtr = &Pin(O).array[Pin(O).offset])
            {
                float* aPtr = t0Ptr;
                var aShape = A.shape;

                for (int t = 1; t < tensors.Length; ++t)
                {
                    var B = tensors[t];
                    fixed (float* bPtr = &Pin(B).array[Pin(B).offset])
                    {
                        //Inner loop
                        const int unrollSize = 4;
                        m_InnerLoop.SetState(unrollSize, oPtr, aPtr, bPtr, O.shape, aShape, B.shape);
                        if (CanUseModuloForBroadcasting(O.shape, aShape, B.shape))
                            Parallel_For(0L, O.length / unrollSize, opInnerLoopNoBroadcast);
                        else
                            Parallel_For(0L, O.length / unrollSize, opInnerLoop);


                        // Remainder
                        for (int i = (O.length / unrollSize) * unrollSize; i < O.length; ++i)
                        {
                            int b0 = 0, h0 = 0, w0 = 0, ch0 = 0;
                            O.shape.GetPositionsFromIndex(i, ref b0, ref h0, ref w0, ref ch0);
                            oPtr[i] = opRemainder(aPtr[A.shape.IndexWithBroadcast(b0, h0, w0, ch0)], bPtr[B.shape.IndexWithBroadcast(b0, h0, w0, ch0)]);
                        }
                    }

                    aPtr = oPtr;
                    aShape = O.shape;
                }
            }
        }

        return O;
    }

    public override Tensor Add(Tensor[] tensors)
    {
        return ApplyElementwiseWithBroadcast(tensors, m_InnerLoop.m_addOpDelegate, m_InnerLoop.m_addInnerLoopDelegate, m_InnerLoop.m_addInnerLoopDelegateNoBroadcast);
    }

    public override Tensor Sub(Tensor[] tensors)
    {
        return ApplyElementwiseWithBroadcast(tensors, m_InnerLoop.m_subOpDelegate, m_InnerLoop.m_subInnerLoopDelegate, m_InnerLoop.m_subInnerLoopDelegateNoBroadcast);
    }

    public override Tensor Mul(Tensor[] tensors)
    {
        return ApplyElementwiseWithBroadcast(tensors, m_InnerLoop.m_mulOpDelegate, m_InnerLoop.m_mulInnerLoopDelegate, m_InnerLoop.m_mulInnerLoopDelegateNoBroadcast);
    }

    public override Tensor Div(Tensor[] tensors)
    {
        return ApplyElementwiseWithBroadcast(tensors, m_InnerLoop.m_divOpDelegate, m_InnerLoop.m_divInnerLoopDelegate, m_InnerLoop.m_divInnerLoopDelegateNoBroadcast);
    }

    public override Tensor Min(Tensor[] tensors)
    {
        return ApplyElementwiseWithBroadcast(tensors, m_InnerLoop.m_minOpDelegate, m_InnerLoop.m_minInnerLoopDelegate, m_InnerLoop.m_minInnerLoopDelegateNoBroadcast);
    }

    public override Tensor Max(Tensor[] tensors)
    {
        return ApplyElementwiseWithBroadcast(tensors, m_InnerLoop.m_maxOpDelegate, m_InnerLoop.m_maxInnerLoopDelegate, m_InnerLoop.m_maxInnerLoopDelegateNoBroadcast);
    }

    public override Tensor Greater(Tensor A, Tensor B)
    {
        return ApplyLogicalOperator(A, B, m_InnerLoop.m_greaterOpDelegate, m_InnerLoop.m_greaterInnerLoopDelegate, m_InnerLoop.m_greaterInnerLoopDelegateNoBroadcast);
    }
    public override Tensor GreaterEqual(Tensor A, Tensor B)
    {
        return ApplyLogicalOperator(A, B, m_InnerLoop.m_greaterEqualOpDelegate, m_InnerLoop.m_greaterEqualInnerLoopDelegate, m_InnerLoop.m_greaterEqualInnerLoopDelegateNoBroadcast);
    }
    public override Tensor Less(Tensor A, Tensor B)
    {
        return ApplyLogicalOperator(A, B, m_InnerLoop.m_lessOpDelegate, m_InnerLoop.m_lessInnerLoopDelegate, m_InnerLoop.m_lessInnerLoopDelegateNoBroadcast);
    }
    public override Tensor LessEqual(Tensor A, Tensor B)
    {
        return ApplyLogicalOperator(A, B, m_InnerLoop.m_lessEqualOpDelegate, m_InnerLoop.m_lessEqualInnerLoopDelegate, m_InnerLoop.m_lessEqualInnerLoopDelegateNoBroadcast);
    }
    public override Tensor Equal(Tensor A, Tensor B)
    {
        return ApplyLogicalOperator(A, B, m_InnerLoop.m_equalOpDelegate, m_InnerLoop.m_equalInnerLoopDelegate, m_InnerLoop.m_equalInnerLoopDelegateNoBroadcast);
    }
    public override Tensor LogicalOr(Tensor A, Tensor B)
    {
        return ApplyLogicalOperator(A, B, m_InnerLoop.m_logicalOrOpDelegate, m_InnerLoop.m_logicalOrInnerLoopDelegate, m_InnerLoop.m_logicalOrInnerLoopDelegateNoBroadcast);
    }
    public override Tensor LogicalAnd(Tensor A, Tensor B)
    {
        return ApplyLogicalOperator(A, B, m_InnerLoop.m_logicalAndOpDelegate, m_InnerLoop.m_logicalAndInnerLoopDelegate, m_InnerLoop.m_logicalAndInnerLoopDelegateNoBroadcast);
    }
    public override Tensor LogicalXor(Tensor A, Tensor B)
    {
        return ApplyLogicalOperator(A, B, m_InnerLoop.m_logicalXorOpDelegate, m_InnerLoop.m_logicalXorInnerLoopDelegate, m_InnerLoop.m_logicalXorInnerLoopDelegateNoBroadcast);
    }

    public override Tensor LogicalNot(Tensor X)
    {
        var O = NewTensorLike(X);

        unsafe
        {
            fixed (float*
            xPtr = &Pin(X).array[Pin(X).offset],
            oPtr = &Pin(O).array[Pin(O).offset])
            {
                const int unrollSize = 4;
                m_InnerLoop.SetState(unrollSize, xPtr, oPtr);
                Parallel_For(0L, O.length / unrollSize, m_InnerLoop.m_logicaNotInnerLoopDelegate);

                // Remainder
                for (int i = (O.length / unrollSize) * unrollSize; i < O.length; ++i)
                    oPtr[i] = Convert.ToSingle( !Convert.ToBoolean(xPtr[i]) );
            }
        }
        return O;
    }

    private Tensor ApplyLogicalOperator(Tensor A, Tensor B, Func<float,float,float> logicalOpRemainder, Action<long> logicalOpInnerLoop, Action<long> logicalOpInnerLoopNoBroadcast)
    {
        var O = GetOutputTensorFromBroadcast(new Tensor[] { A, B });

        unsafe
        {
            fixed (float*
                aPtr = &Pin(A).array[Pin(A).offset],
                bPtr = &Pin(B).array[Pin(B).offset],
                oPtr = &Pin(O).array[Pin(O).offset])
            {
                const int unrollSize = 4;
                m_InnerLoop.SetState(unrollSize, oPtr, aPtr, bPtr, O.shape, A.shape, B.shape);
                if ((O.shape == A.shape) && (O.shape == B.shape))
                    Parallel_For(0L, O.length / unrollSize, logicalOpInnerLoopNoBroadcast);
                else
                    Parallel_For(0L, O.length / unrollSize, logicalOpInnerLoop);

                // Remainder
                for (int i = (O.length / unrollSize) * unrollSize; i < O.length; ++i)
                {
                    int b0 = 0, h0 = 0, w0 = 0, ch0 = 0;
                    O.shape.GetPositionsFromIndex(i, ref b0, ref h0, ref w0, ref ch0);
                    oPtr[i] = logicalOpRemainder(aPtr[A.shape.IndexWithBroadcast(b0, h0, w0, ch0)], bPtr[B.shape.IndexWithBroadcast(b0, h0, w0, ch0)]);
                }
            }
        }

        return O;
    }

    public override Tensor MatMul(Tensor X, bool xTranspose, Tensor Y, bool yTranspose)
    {
        //var Z = base.MatMul(X, xTranspose, Y, yTranspose);
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

        unsafe
        {
            fixed (float*
                xPtr = &Pin(X).array[Pin(X).offset],
                yPtr = &Pin(Y).array[Pin(Y).offset],
                oPtr = &Pin(O).array[Pin(O).offset])
            {
                // NOTE: assumes newly created Tensor data is initialized with 0

                //D.Log(string.Format("===> X.b[{0}] x Y.w[{1}] * Y.h[{2}] x Y.w[{3}] = O.w[{4}] x O.h[{5}]", X.flatHeight, X.flatWidth, Y.flatHeight, Y.flatWidth, O.batch, O.width));
                blas.SGEMM(xPtr, X.flatHeight, X.flatWidth,
                    yPtr, Y.flatHeight, Y.flatWidth,
                    oPtr, O.flatHeight, O.flatWidth, 32, xTranspose, yTranspose);
            }
        }

        Profiler.EndSample ();

        //O.PrintDataPart(32, "O");
        //Z.PrintDataPart(32, "Z");
        //CompareOps.CheckSame(O, Z, "MatMul");

        return O;
    }

    public override Tensor Dense(Tensor X, Tensor W, Tensor B)
    {
        //D.Log(string.Format("X = {0}", X.shape));
        Assert.IsTrue(W.dimensions <= 2);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(X.flatWidth, W.flatHeight);
        var O = NewTensor(X.flatHeight, W.flatWidth);

        unsafe
        {
            fixed (float*
            xPtr = &Pin(X).array[Pin(X).offset],
            wPtr = &Pin(W).array[Pin(W).offset],
            bPtr = &Pin(B).array[Pin(B).offset],
            oPtr = &Pin(O).array[Pin(O).offset])
            {
                    var BOffset = Pin(O).offset;
                    var BCount = Pin(B).count;
                    var Barray = Pin(O).array;

                    for (int i = 0; i < O.batch; i++)
                    {
                        Marshal.Copy((IntPtr)bPtr, Barray, BOffset + i * BCount, BCount);
                    }

                    //X.Print(); W.Print();
                    blas.SGEMM(xPtr, X.flatHeight, X.flatWidth, wPtr, W.flatHeight, W.flatWidth, oPtr, O.batch, O.channels, 16);
            }
        }

        return O;
    }

    public override Tensor MaxPool2D(Tensor X, int[] pool, int[] stride, int[] pad)
    {
        Assert.AreEqual(pool.Length, 2);
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);

        var O = NewTensor(X.shape.ApplyPool(pool, stride, pad));

        int xnMult = X.height * X.width * X.channels;
        int xyMult = X.width * X.channels;
        int xxMult = X.channels;

        int onMult = O.height * O.width * O.channels;
        int oyMult = O.width * O.channels;
        int oxMult = O.channels;

        int oBatch = O.batch;
        int oHeight = O.height;
        int oWidth = O.width;
        int oChannels = O.channels;
        int xHeight = X.height;
        int xWidth = X.width;

        unsafe
        {
            fixed (float*
                xPtr = &Pin(X).array[Pin(X).offset],
                oPtr = &Pin(O).array[Pin(O).offset])
            {
                MaxPool2DInnerLoop(pool, stride, pad,
                    xHeight, xWidth, xPtr, xnMult, xyMult, xxMult,
                    oBatch, oHeight, oWidth, oChannels, oPtr, onMult, oyMult, oxMult);
            }
        }

        return O;
    }

    private static unsafe void MaxPool2DInnerLoop(int[] pool, int[] stride, int[] pad,
        int xHeight, int xWidth, float* xPtr, int xnMult, int xyMult, int xxMult,
        int oBatch, int oHeight, int oWidth, int oChannels, float* oPtr, int onMult, int oyMult, int oxMult)
    {
        Parallel.For(0, oBatch, n =>
        {
            for (var y = 0; y < oHeight; ++y)
            for (var x = 0; x < oWidth; ++x)
            for (var c = 0; c < oChannels; ++c)
            {
                float maxVal = float.MinValue;
                for (int dy = 0; dy < pool[1]; ++dy)
                    for (int dx = 0; dx < pool[0]; ++dx)
                    {
                        int oy = y * stride[1] + dy - pad[1];
                        int ox = x * stride[0] + dx - pad[0];

                        if (oy < 0) continue;
                        if (oy >= xHeight) continue;
                        if (ox < 0) continue;
                        if (ox >= xWidth) continue;

                        float v = xPtr[n * xnMult + oy * xyMult + ox * xxMult + c];
                        maxVal = Mathf.Max(v, maxVal);
                    }
                oPtr[n * onMult + y * oyMult + x * oxMult + c] = maxVal;
            }
        });
    }

    public override Tensor Conv2D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad)
    {
        Assert.AreEqual(X.channels, K.kernelDepth);
        Assert.AreEqual(K.kernelCount, B.flatWidth);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);

        var O = NewTensor(X.shape.ApplyKernel(K.shape, stride, pad));

        int xnMult = X.height * X.width * X.channels;
        int xyMult = X.width * X.channels;
        int xxMult = X.channels;

        int kyMult = K.height * K.width * K.channels;
        int kxMult = K.width * K.channels;
        int kcMult = K.channels;

        int onMult = O.height * O.width * O.channels;
        int oyMult = O.width * O.channels;
        int oxMult = O.channels;

        int oBatch = O.batch;
        int oHeight = O.height;
        int oWidth = O.width;
        int kKernelCount = K.kernelCount;
        int kKernelHeight = K.kernelHeight;
        int kKernelWidth = K.kernelWidth;
        int xHeight = X.height;
        int xWidth = X.width;
        int xChannels = X.channels;

        unsafe
        {
            fixed (float*
                xPtr = &Pin(X).array[Pin(X).offset],
                kPtr = &Pin(K).array[Pin(K).offset],
                bPtr = &Pin(B).array[Pin(B).offset],
                oPtr = &Pin(O).array[Pin(O).offset])
            {
                Conv2DInnerLoop(stride, pad, oBatch, oHeight, oWidth, kKernelCount, bPtr, kKernelHeight, kKernelWidth,
                    xHeight, xWidth, xChannels, xPtr, xnMult, xyMult, xxMult, kPtr, kyMult, kxMult, kcMult, oPtr, onMult, oyMult, oxMult);
            }
        }

        return O;
    }

    private static unsafe void Conv2DInnerLoop(int[] stride, int[] pad, int oBatch, int oHeight, int oWidth, int kKernelCount,
        float* bPtr, int kKernelHeight, int kKernelWidth, int xHeight, int xWidth, int xChannels, float* xPtr,
        int xnMult, int xyMult, int xxMult, float* kPtr, int kyMult, int kxMult, int kcMult, float* oPtr, int onMult,
        int oyMult, int oxMult)
    {
        Parallel.For(0, oBatch, n =>
        {
            for (var y = 0; y < oHeight; ++y)
            for (var x = 0; x < oWidth; ++x)
            for (var k = 0; k < kKernelCount; ++k)
            {
                float v = bPtr[k];
                for (int dy = 0; dy < kKernelHeight; ++dy)
                {
                    for (int dx = 0; dx < kKernelWidth; ++dx)
                    {
                        int oy = y * stride[1] + dy - pad[1];
                        int ox = x * stride[0] + dx - pad[0];

                        if (oy < 0) continue;
                        if (oy >= xHeight) continue;
                        if (ox < 0) continue;
                        if (ox >= xWidth) continue;

                        for (var c = 0; c < xChannels; ++c)
                        {
                            float xv = xPtr[n * xnMult + oy * xyMult + ox * xxMult + c];
                            float kv = kPtr[dy * kyMult + dx * kxMult + c * kcMult + k];

                            v += xv * kv;
                        }
                    }
                }

                oPtr[n * onMult + y * oyMult + x * oxMult + k] = v;
            }
        });
    }

    private Tensor ApplyPadding(Tensor X, int[] pad, float constant, Action<long> paddingOp)
    {
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

        unsafe
        {
            fixed (float*
                xPtr = &Pin(X).array[Pin(X).offset],
                oPtr = &Pin(O).array[Pin(O).offset])
            {
                m_InnerLoop.SetState(oPtr, xPtr, O.shape, X.shape, constant, prePadX, prePadY);

                long numItemInARow = O.width * O.channels;
                long numItemInABatch = O.height * numItemInARow;

                for (int b = 0; b < O.batch; ++b)
                {
                    //PrePadY
                    long prepadOffset = numItemInABatch * b;
                    long numItemToPrepadInHeight = prePadY * O.width * O.channels;
                    Parallel_For(prepadOffset, prepadOffset + numItemToPrepadInHeight, paddingOp);

                    //CenterY
                    Parallel.For(prePadY, croppedHeight + prePadY, y =>
                    {
                        long offset = numItemInABatch * b + numItemInARow * y;
                        //PrePadX
                        long numItemToPrepadInWidth = prePadX * O.channels;
                        for (long n = offset; n < (offset + numItemToPrepadInWidth); ++n)
                            paddingOp(n);
                        offset += numItemToPrepadInWidth;

                        //CenterX
                        int srcFloatOffset = X.Index(b, (int)y - prePadY, preCropX, 0) + Pin(X).offset;
                        int dstFloatOffset = O.Index(b, (int)y, prePadX, 0) + Pin(O).offset;
                        int numFloatToCopy = O.channels * croppedWidth;
                        Buffer.BlockCopy(Pin(X).array, srcFloatOffset * sizeof(float), Pin(O).array, dstFloatOffset * sizeof(float), numFloatToCopy * sizeof(float));
                        offset += numFloatToCopy;

                        //PostPadX
                        long numItemToPostInWidth = postPadX * O.channels;
                        for (long n = offset; n < (offset + numItemToPostInWidth); ++n)
                            paddingOp(n);
                    });

                    //PostPadY
                    long postpadOffset = prepadOffset + numItemToPrepadInHeight + numItemInARow * croppedHeight;
                    long numItemToPostpadInHeight = postPadY * O.width * O.channels;
                    Parallel_For(postpadOffset, postpadOffset + numItemToPostpadInHeight, paddingOp);
                }
            }
        }
        return O;
    }

    public override Tensor Border2D(Tensor X, int[] pad, float constant)
    {
        return ApplyPadding(X, pad, constant, m_InnerLoop.m_border2DInnerLoopDelegate);
    }

    public override Tensor Pad2DEdge(Tensor X, int[] pad)
    {
        return ApplyPadding(X, pad, 0.0f, m_InnerLoop.m_pad2DEdgeInnerLoopDelegate);
    }

    public override Tensor Pad2DReflect(Tensor X, int[] pad)
    {
        return ApplyPadding(X, pad, 0.0f, m_InnerLoop.m_pad2DReflectInnerLoopDelegate);
    }

    public override Tensor Pad2DSymmetric(Tensor X, int[] pad)
    {
        return ApplyPadding(X, pad, 0.0f, m_InnerLoop.m_pad2DSymmetricInnerLoopDelegate);
    }

    public override Tensor Prepare(Tensor X)
    {
        Pin(X);
        return X;
    }
}

    internal unsafe class InnerLoop
    {
        private int unrollSize;
        private float* oPtr;
        private float* xPtr;
        private int xLen;
        private float* bPtr;
        private int bLen;
        private float alpha;
        private int prePadX;
        private int prePadY;

        private TensorShape oShape;
        private TensorShape xShape;
        private TensorShape bShape;

        public Action<long> m_tanhInnerLoopDelegate;
        public Action<long> m_expInnerLoopDelegate;
        public Action<long> m_sqrtInnerLoopDelegate;
        public Action<long> m_swishInnerLoopDelegate;
        public Action<long> m_sigmoidInnerLoopDelegate;
        public Action<long> m_negInnerLoopDelegate;
        public Action<long> m_eluInnerLoopDelegate;
        public Action<long> m_reluInnerLoopDelegate;
        public Action<long> m_preluInnerLoopDelegate;
        public Action<long> m_maxInnerLoopDelegate;
        public Action<long> m_minInnerLoopDelegate;
        public Action<long> m_divInnerLoopDelegate;
        public Action<long> m_mulInnerLoopDelegate;
        public Action<long> m_subInnerLoopDelegate;
        public Action<long> m_addInnerLoopDelegate;
        public Action<long> m_greaterInnerLoopDelegate;
        public Action<long> m_greaterEqualInnerLoopDelegate;
        public Action<long> m_lessInnerLoopDelegate;
        public Action<long> m_lessEqualInnerLoopDelegate;
        public Action<long> m_equalInnerLoopDelegate;
        public Action<long> m_logicalAndInnerLoopDelegate;
        public Action<long> m_logicalOrInnerLoopDelegate;
        public Action<long> m_logicalXorInnerLoopDelegate;
        public Action<long> m_logicaNotInnerLoopDelegate;
        public Action<long> m_maxInnerLoopDelegateNoBroadcast;
        public Action<long> m_minInnerLoopDelegateNoBroadcast;
        public Action<long> m_divInnerLoopDelegateNoBroadcast;
        public Action<long> m_mulInnerLoopDelegateNoBroadcast;
        public Action<long> m_subInnerLoopDelegateNoBroadcast;
        public Action<long> m_addInnerLoopDelegateNoBroadcast;
        public Action<long> m_greaterInnerLoopDelegateNoBroadcast;
        public Action<long> m_greaterEqualInnerLoopDelegateNoBroadcast;
        public Action<long> m_lessInnerLoopDelegateNoBroadcast;
        public Action<long> m_lessEqualInnerLoopDelegateNoBroadcast;
        public Action<long> m_equalInnerLoopDelegateNoBroadcast;
        public Action<long> m_logicalAndInnerLoopDelegateNoBroadcast;
        public Action<long> m_logicalOrInnerLoopDelegateNoBroadcast;
        public Action<long> m_logicalXorInnerLoopDelegateNoBroadcast;
        public Action<long> m_border2DInnerLoopDelegate;
        public Action<long> m_pad2DReflectInnerLoopDelegate;
        public Action<long> m_pad2DSymmetricInnerLoopDelegate;
        public Action<long> m_pad2DEdgeInnerLoopDelegate;

        public Func<float,float,float> m_maxOpDelegate;
        public Func<float,float,float> m_minOpDelegate;
        public Func<float,float,float> m_divOpDelegate;
        public Func<float,float,float> m_mulOpDelegate;
        public Func<float,float,float> m_subOpDelegate;
        public Func<float,float,float> m_addOpDelegate;
        public Func<float,float,float> m_greaterOpDelegate;
        public Func<float,float,float> m_greaterEqualOpDelegate;
        public Func<float,float,float> m_lessOpDelegate;
        public Func<float,float,float> m_lessEqualOpDelegate;
        public Func<float,float,float> m_equalOpDelegate;
        public Func<float,float,float> m_logicalAndOpDelegate;
        public Func<float,float,float> m_logicalOrOpDelegate;
        public Func<float,float,float> m_logicalXorOpDelegate;
        public Func<float,float>       m_logicaNotOpDelegate;

        public InnerLoop()
        {
            //Store delegates to avoid GC allocation because of repeated cast from functions to delegate at runtime
            m_tanhInnerLoopDelegate = TanhInnerLoop;
            m_expInnerLoopDelegate = ExpInnerLoop;
            m_sqrtInnerLoopDelegate = SqrtInnerLoop;
            m_swishInnerLoopDelegate = SwishInnerLoop;
            m_sigmoidInnerLoopDelegate = SigmoidInnerLoop;
            m_negInnerLoopDelegate = NegInnerLoop;
            m_eluInnerLoopDelegate = EluInnerLoop;
            m_reluInnerLoopDelegate = ReluInnerLoop;
            m_preluInnerLoopDelegate = PReluInnerLoop;
            m_maxInnerLoopDelegate = MaxInnerLoop;
            m_minInnerLoopDelegate = MinInnerLoop;
            m_divInnerLoopDelegate = DivInnerLoop;
            m_mulInnerLoopDelegate = MulInnerLoop;
            m_subInnerLoopDelegate = SubInnerLoop;
            m_addInnerLoopDelegate = AddInnerLoop;
            m_greaterInnerLoopDelegate = GreaterInnerLoop;
            m_greaterEqualInnerLoopDelegate = GreaterEqualInnerLoop;
            m_lessInnerLoopDelegate = LessInnerLoop;
            m_lessEqualInnerLoopDelegate = LessEqualInnerLoop;
            m_equalInnerLoopDelegate = EqualInnerLoop;
            m_logicalAndInnerLoopDelegate = LogicalAndInnerLoop;
            m_logicalOrInnerLoopDelegate = LogicalOrInnerLoop;
            m_logicalXorInnerLoopDelegate = LogicalXorInnerLoop;
            m_logicaNotInnerLoopDelegate = LogicalNotInnerLoop;
            m_maxInnerLoopDelegateNoBroadcast = MaxInnerLoopNoBroadcast;
            m_minInnerLoopDelegateNoBroadcast = MinInnerLoopNoBroadcast;
            m_divInnerLoopDelegateNoBroadcast = DivInnerLoopNoBroadcast;
            m_mulInnerLoopDelegateNoBroadcast = MulInnerLoopNoBroadcast;
            m_subInnerLoopDelegateNoBroadcast = SubInnerLoopNoBroadcast;
            m_addInnerLoopDelegateNoBroadcast = AddInnerLoopNoBroadcast;
            m_greaterInnerLoopDelegateNoBroadcast = GreaterInnerLoopNoBroadcast;
            m_greaterEqualInnerLoopDelegateNoBroadcast = GreaterEqualInnerLoopNoBroadcast;
            m_lessInnerLoopDelegateNoBroadcast = LessInnerLoopNoBroadcast;
            m_lessEqualInnerLoopDelegateNoBroadcast = LessEqualInnerLoopNoBroadcast;
            m_equalInnerLoopDelegateNoBroadcast = EqualInnerLoopNoBroadcast;
            m_logicalAndInnerLoopDelegateNoBroadcast = LogicalAndInnerLoopNoBroadcast;
            m_logicalOrInnerLoopDelegateNoBroadcast = LogicalOrInnerLoopNoBroadcast;
            m_logicalXorInnerLoopDelegateNoBroadcast = LogicalXorInnerLoopNoBroadcast;
            m_border2DInnerLoopDelegate = Border2DInnerLoop;
            m_pad2DEdgeInnerLoopDelegate = Pad2DEdgeInnerLoop;
            m_pad2DReflectInnerLoopDelegate = Pad2DReflectInnerLoop;
            m_pad2DSymmetricInnerLoopDelegate = Pad2DSymmetricInnerLoop;
            m_maxOpDelegate = Max;
            m_minOpDelegate = Min;
            m_divOpDelegate = Div;
            m_mulOpDelegate = Mul;
            m_subOpDelegate = Sub;
            m_addOpDelegate = Add;
            m_greaterOpDelegate = Greater;
            m_greaterEqualOpDelegate = GreaterEqual;
            m_lessOpDelegate = Less;
            m_lessEqualOpDelegate = LessEqual;
            m_equalOpDelegate = Equal;
            m_logicalAndOpDelegate = LogicalAnd;
            m_logicalOrOpDelegate = LogicalOr;
            m_logicalXorOpDelegate = LogicalXor;
            m_logicaNotOpDelegate = LogicalNot;
        }

        public void SetState(int unrollSize, float* oPtr, float* xPtr, float* bPtr, TensorShape oShape, TensorShape xShape, TensorShape bShape)
        {
            this.unrollSize = unrollSize;
            this.oPtr = oPtr;
            this.oShape = oShape;
            this.xPtr = xPtr;
            this.xShape = xShape;
            this.xLen = xShape.length;
            this.bPtr = bPtr;
            this.bShape = bShape;
            this.bLen = bShape.length;
        }

        public void SetState(int unrollSize, float* oPtr, float* xPtr, int xLen, float* bPtr, int bLen)
        {
            this.unrollSize = unrollSize;
            this.oPtr = oPtr;
            this.xPtr = xPtr;
            this.xLen = xLen;
            this.bPtr = bPtr;
            this.bLen = bLen;
        }

        public void SetState(int unrollSize, float* xPtr, float* oPtr)
        {
            this.unrollSize = unrollSize;
            this.oPtr = oPtr;
            this.xPtr = xPtr;
        }

        public void SetState(int unrollSize, float* xPtr, float* oPtr, float* bPtr)
        {
            this.unrollSize = unrollSize;
            this.oPtr = oPtr;
            this.xPtr = xPtr;
            this.bPtr = bPtr;
        }

        public void SetState(int unrollSize, float* xPtr, float* oPtr, float alpha)
        {
            this.unrollSize = unrollSize;
            this.oPtr = oPtr;
            this.xPtr = xPtr;
            this.alpha = alpha;
        }

        public void SetState(float* oPtr, float* xPtr, TensorShape oShape, TensorShape xShape, float constant, int prePadX, int prePadY)
        {
            this.oPtr = oPtr;
            this.xPtr = xPtr;
            this.oShape = oShape;
            this.xShape = xShape;
            this.alpha = constant;
            this.prePadX = prePadX;
            this.prePadY = prePadY;
        }

        private void NegInnerLoop(long n)
        {
            float* baseXPtr = xPtr + n * unrollSize;
            float* baseOPtr = oPtr + n * unrollSize;
            float v0 = baseXPtr[0];
            float v1 = baseXPtr[1];
            float v2 = baseXPtr[2];
            float v3 = baseXPtr[3];

            v0 = -v0;
            v1 = -v1;
            v2 = -v2;
            v3 = -v3;

            baseOPtr[0] = v0;
            baseOPtr[1] = v1;
            baseOPtr[2] = v2;
            baseOPtr[3] = v3;
        }

        private void ReluInnerLoop(long n)
        {
            float* baseXPtr = xPtr + n * unrollSize;
            float* baseOPtr = oPtr + n * unrollSize;
            float v0 = baseXPtr[0];
            float v1 = baseXPtr[1];
            float v2 = baseXPtr[2];
            float v3 = baseXPtr[3];

            v0 = 0.5f * (v0 + Mathf.Abs(v0));
            v1 = 0.5f * (v1 + Mathf.Abs(v1));
            v2 = 0.5f * (v2 + Mathf.Abs(v2));
            v3 = 0.5f * (v3 + Mathf.Abs(v3));

            baseOPtr[0] = v0;
            baseOPtr[1] = v1;
            baseOPtr[2] = v2;
            baseOPtr[3] = v3;
        }

        private void EluInnerLoop(long n)
        {
            float* baseXPtr = xPtr + n * unrollSize;
            float* baseOPtr = oPtr + n * unrollSize;
            float v0 = baseXPtr[0];
            float v1 = baseXPtr[1];
            float v2 = baseXPtr[2];
            float v3 = baseXPtr[3];

            if (v0 <= 0)
                v0 = alpha * (Mathf.Exp(v0) - 1f);
            if (v1 <= 0)
                v1 = alpha * (Mathf.Exp(v1) - 1f);
            if (v2 <= 0)
                v2 = alpha * (Mathf.Exp(v2) - 1f);
            if (v3 <= 0)
                v3 = alpha * (Mathf.Exp(v3) - 1f);

            baseOPtr[0] = v0;
            baseOPtr[1] = v1;
            baseOPtr[2] = v2;
            baseOPtr[3] = v3;
        }

        private void PReluInnerLoop(long n)
        {
            float* baseXPtr = xPtr + n * unrollSize;
            float* baseOPtr = oPtr + n * unrollSize;
            float* baseBPtr = bPtr + (n * unrollSize) % bLen;
            float v0 = baseXPtr[0];
            float v1 = baseXPtr[1];
            float v2 = baseXPtr[2];
            float v3 = baseXPtr[3];

            float s0 = baseBPtr[0 % bLen];
            float s1 = baseBPtr[1 % bLen];
            float s2 = baseBPtr[2 % bLen];
            float s3 = baseBPtr[3 % bLen];

            if (v0 <= 0)
                v0 = s0 * v0;
            if (v1 <= 0)
                v1 = s1 * v1;
            if (v2 <= 0)
                v2 = s2 * v2;
            if (v3 <= 0)
                v3 = s3 * v3;

            baseOPtr[0] = v0;
            baseOPtr[1] = v1;
            baseOPtr[2] = v2;
            baseOPtr[3] = v3;
        }

        private void SigmoidInnerLoop(long n)
        {
            float* baseXPtr = xPtr + n * unrollSize;
            float* baseOPtr = oPtr + n * unrollSize;
            float v0 = baseXPtr[0];
            float v1 = baseXPtr[1];
            float v2 = baseXPtr[2];
            float v3 = baseXPtr[3];

            v0 = 1f / (1f + Mathf.Exp(-v0));
            v1 = 1f / (1f + Mathf.Exp(-v1));
            v2 = 1f / (1f + Mathf.Exp(-v2));
            v3 = 1f / (1f + Mathf.Exp(-v3));

            baseOPtr[0] = v0;
            baseOPtr[1] = v1;
            baseOPtr[2] = v2;
            baseOPtr[3] = v3;
        }

        private void SwishInnerLoop(long n)
        {
            float* baseXPtr = xPtr + n * unrollSize;
            float* baseOPtr = oPtr + n * unrollSize;
            float v0 = baseXPtr[0];
            float v1 = baseXPtr[1];
            float v2 = baseXPtr[2];
            float v3 = baseXPtr[3];

            v0 = v0 / (1f + Mathf.Exp(-v0));
            v1 = v1 / (1f + Mathf.Exp(-v1));
            v2 = v2 / (1f + Mathf.Exp(-v2));
            v3 = v3 / (1f + Mathf.Exp(-v3));

            baseOPtr[0] = v0;
            baseOPtr[1] = v1;
            baseOPtr[2] = v2;
            baseOPtr[3] = v3;
        }

        private void ExpInnerLoop(long n)
        {
            float* baseXPtr = xPtr + n * unrollSize;
            float* baseOPtr = oPtr + n * unrollSize;
            float v0 = baseXPtr[0];
            float v1 = baseXPtr[1];
            float v2 = baseXPtr[2];
            float v3 = baseXPtr[3];

            v0 = Mathf.Exp(v0);
            v1 = Mathf.Exp(v1);
            v2 = Mathf.Exp(v2);
            v3 = Mathf.Exp(v3);

            baseOPtr[0] = v0;
            baseOPtr[1] = v1;
            baseOPtr[2] = v2;
            baseOPtr[3] = v3;
        }

        private void SqrtInnerLoop(long n)
        {
            float* baseXPtr = xPtr + n * unrollSize;
            float* baseOPtr = oPtr + n * unrollSize;
            float v0 = baseXPtr[0];
            float v1 = baseXPtr[1];
            float v2 = baseXPtr[2];
            float v3 = baseXPtr[3];

            v0 = Mathf.Sqrt(v0);
            v1 = Mathf.Sqrt(v1);
            v2 = Mathf.Sqrt(v2);
            v3 = Mathf.Sqrt(v3);

            baseOPtr[0] = v0;
            baseOPtr[1] = v1;
            baseOPtr[2] = v2;
            baseOPtr[3] = v3;
        }

        private void TanhInnerLoop(long n)
        {
            float* baseXPtr = xPtr + n * unrollSize;
            float* baseOPtr = oPtr + n * unrollSize;
            float v0 = baseXPtr[0];
            float v1 = baseXPtr[1];
            float v2 = baseXPtr[2];
            float v3 = baseXPtr[3];

            v0 = MathfEx.tanh(v0);
            v1 = MathfEx.tanh(v1);
            v2 = MathfEx.tanh(v2);
            v3 = MathfEx.tanh(v3);

            baseOPtr[0] = v0;
            baseOPtr[1] = v1;
            baseOPtr[2] = v2;
            baseOPtr[3] = v3;
        }

        private void AddInnerLoop(long n)
        {
            int i = (int)n * unrollSize;

            int b0 = 0, h0 = 0, w0 = 0, ch0 = 0;
            int b1 = 0, h1 = 0, w1 = 0, ch1 = 0;
            int b2 = 0, h2 = 0, w2 = 0, ch2 = 0;
            int b3 = 0, h3 = 0, w3 = 0, ch3 = 0;
            oShape.GetPositionsFromIndex(i + 0, ref b0, ref h0, ref w0, ref ch0);
            oShape.GetPositionsFromIndex(i + 1, ref b1, ref h1, ref w1, ref ch1);
            oShape.GetPositionsFromIndex(i + 2, ref b2, ref h2, ref w2, ref ch2);
            oShape.GetPositionsFromIndex(i + 3, ref b3, ref h3, ref w3, ref ch3);

            oPtr[i + 0] = xPtr[xShape.IndexWithBroadcast(b0, h0, w0, ch0)] + bPtr[bShape.IndexWithBroadcast(b0, h0, w0, ch0)];
            oPtr[i + 1] = xPtr[xShape.IndexWithBroadcast(b1, h1, w1, ch1)] + bPtr[bShape.IndexWithBroadcast(b1, h1, w1, ch1)];
            oPtr[i + 2] = xPtr[xShape.IndexWithBroadcast(b2, h2, w2, ch2)] + bPtr[bShape.IndexWithBroadcast(b2, h2, w2, ch2)];
            oPtr[i + 3] = xPtr[xShape.IndexWithBroadcast(b3, h3, w3, ch3)] + bPtr[bShape.IndexWithBroadcast(b3, h3, w3, ch3)];
        }

        private void SubInnerLoop(long n)
        {
            int i = (int)n * unrollSize;

            int b0 = 0, h0 = 0, w0 = 0, ch0 = 0;
            int b1 = 0, h1 = 0, w1 = 0, ch1 = 0;
            int b2 = 0, h2 = 0, w2 = 0, ch2 = 0;
            int b3 = 0, h3 = 0, w3 = 0, ch3 = 0;
            oShape.GetPositionsFromIndex(i + 0, ref b0, ref h0, ref w0, ref ch0);
            oShape.GetPositionsFromIndex(i + 1, ref b1, ref h1, ref w1, ref ch1);
            oShape.GetPositionsFromIndex(i + 2, ref b2, ref h2, ref w2, ref ch2);
            oShape.GetPositionsFromIndex(i + 3, ref b3, ref h3, ref w3, ref ch3);

            oPtr[i + 0] = xPtr[xShape.IndexWithBroadcast(b0, h0, w0, ch0)] - bPtr[bShape.IndexWithBroadcast(b0, h0, w0, ch0)];
            oPtr[i + 1] = xPtr[xShape.IndexWithBroadcast(b1, h1, w1, ch1)] - bPtr[bShape.IndexWithBroadcast(b1, h1, w1, ch1)];
            oPtr[i + 2] = xPtr[xShape.IndexWithBroadcast(b2, h2, w2, ch2)] - bPtr[bShape.IndexWithBroadcast(b2, h2, w2, ch2)];
            oPtr[i + 3] = xPtr[xShape.IndexWithBroadcast(b3, h3, w3, ch3)] - bPtr[bShape.IndexWithBroadcast(b3, h3, w3, ch3)];
        }

        private void MulInnerLoop(long n)
        {
            int i = (int)n * unrollSize;

            int b0 = 0, h0 = 0, w0 = 0, ch0 = 0;
            int b1 = 0, h1 = 0, w1 = 0, ch1 = 0;
            int b2 = 0, h2 = 0, w2 = 0, ch2 = 0;
            int b3 = 0, h3 = 0, w3 = 0, ch3 = 0;
            oShape.GetPositionsFromIndex(i + 0, ref b0, ref h0, ref w0, ref ch0);
            oShape.GetPositionsFromIndex(i + 1, ref b1, ref h1, ref w1, ref ch1);
            oShape.GetPositionsFromIndex(i + 2, ref b2, ref h2, ref w2, ref ch2);
            oShape.GetPositionsFromIndex(i + 3, ref b3, ref h3, ref w3, ref ch3);

            oPtr[i + 0] = xPtr[xShape.IndexWithBroadcast(b0, h0, w0, ch0)] * bPtr[bShape.IndexWithBroadcast(b0, h0, w0, ch0)];
            oPtr[i + 1] = xPtr[xShape.IndexWithBroadcast(b1, h1, w1, ch1)] * bPtr[bShape.IndexWithBroadcast(b1, h1, w1, ch1)];
            oPtr[i + 2] = xPtr[xShape.IndexWithBroadcast(b2, h2, w2, ch2)] * bPtr[bShape.IndexWithBroadcast(b2, h2, w2, ch2)];
            oPtr[i + 3] = xPtr[xShape.IndexWithBroadcast(b3, h3, w3, ch3)] * bPtr[bShape.IndexWithBroadcast(b3, h3, w3, ch3)];
        }

        private void DivInnerLoop(long n)
        {
            int i = (int)n * unrollSize;

            int b0 = 0, h0 = 0, w0 = 0, ch0 = 0;
            int b1 = 0, h1 = 0, w1 = 0, ch1 = 0;
            int b2 = 0, h2 = 0, w2 = 0, ch2 = 0;
            int b3 = 0, h3 = 0, w3 = 0, ch3 = 0;
            oShape.GetPositionsFromIndex(i + 0, ref b0, ref h0, ref w0, ref ch0);
            oShape.GetPositionsFromIndex(i + 1, ref b1, ref h1, ref w1, ref ch1);
            oShape.GetPositionsFromIndex(i + 2, ref b2, ref h2, ref w2, ref ch2);
            oShape.GetPositionsFromIndex(i + 3, ref b3, ref h3, ref w3, ref ch3);

            oPtr[i + 0] = xPtr[xShape.IndexWithBroadcast(b0, h0, w0, ch0)] / bPtr[bShape.IndexWithBroadcast(b0, h0, w0, ch0)];
            oPtr[i + 1] = xPtr[xShape.IndexWithBroadcast(b1, h1, w1, ch1)] / bPtr[bShape.IndexWithBroadcast(b1, h1, w1, ch1)];
            oPtr[i + 2] = xPtr[xShape.IndexWithBroadcast(b2, h2, w2, ch2)] / bPtr[bShape.IndexWithBroadcast(b2, h2, w2, ch2)];
            oPtr[i + 3] = xPtr[xShape.IndexWithBroadcast(b3, h3, w3, ch3)] / bPtr[bShape.IndexWithBroadcast(b3, h3, w3, ch3)];
        }

        private void MinInnerLoop(long n)
        {
            int i = (int)n * unrollSize;

            int b0 = 0, h0 = 0, w0 = 0, ch0 = 0;
            int b1 = 0, h1 = 0, w1 = 0, ch1 = 0;
            int b2 = 0, h2 = 0, w2 = 0, ch2 = 0;
            int b3 = 0, h3 = 0, w3 = 0, ch3 = 0;
            oShape.GetPositionsFromIndex(i + 0, ref b0, ref h0, ref w0, ref ch0);
            oShape.GetPositionsFromIndex(i + 1, ref b1, ref h1, ref w1, ref ch1);
            oShape.GetPositionsFromIndex(i + 2, ref b2, ref h2, ref w2, ref ch2);
            oShape.GetPositionsFromIndex(i + 3, ref b3, ref h3, ref w3, ref ch3);

            oPtr[i + 0] = Mathf.Min( xPtr[xShape.IndexWithBroadcast(b0, h0, w0, ch0)] , bPtr[bShape.IndexWithBroadcast(b0, h0, w0, ch0)] );
            oPtr[i + 1] = Mathf.Min( xPtr[xShape.IndexWithBroadcast(b1, h1, w1, ch1)] , bPtr[bShape.IndexWithBroadcast(b1, h1, w1, ch1)] );
            oPtr[i + 2] = Mathf.Min( xPtr[xShape.IndexWithBroadcast(b2, h2, w2, ch2)] , bPtr[bShape.IndexWithBroadcast(b2, h2, w2, ch2)] );
            oPtr[i + 3] = Mathf.Min( xPtr[xShape.IndexWithBroadcast(b3, h3, w3, ch3)] , bPtr[bShape.IndexWithBroadcast(b3, h3, w3, ch3)] );
        }

        private void MaxInnerLoop(long n)
        {
            int i = (int)n * unrollSize;

            int b0 = 0, h0 = 0, w0 = 0, ch0 = 0;
            int b1 = 0, h1 = 0, w1 = 0, ch1 = 0;
            int b2 = 0, h2 = 0, w2 = 0, ch2 = 0;
            int b3 = 0, h3 = 0, w3 = 0, ch3 = 0;
            oShape.GetPositionsFromIndex(i + 0, ref b0, ref h0, ref w0, ref ch0);
            oShape.GetPositionsFromIndex(i + 1, ref b1, ref h1, ref w1, ref ch1);
            oShape.GetPositionsFromIndex(i + 2, ref b2, ref h2, ref w2, ref ch2);
            oShape.GetPositionsFromIndex(i + 3, ref b3, ref h3, ref w3, ref ch3);

            oPtr[i + 0] = Mathf.Max(xPtr[xShape.IndexWithBroadcast(b0, h0, w0, ch0)], bPtr[bShape.IndexWithBroadcast(b0, h0, w0, ch0)]);
            oPtr[i + 1] = Mathf.Max(xPtr[xShape.IndexWithBroadcast(b1, h1, w1, ch1)], bPtr[bShape.IndexWithBroadcast(b1, h1, w1, ch1)]);
            oPtr[i + 2] = Mathf.Max(xPtr[xShape.IndexWithBroadcast(b2, h2, w2, ch2)], bPtr[bShape.IndexWithBroadcast(b2, h2, w2, ch2)]);
            oPtr[i + 3] = Mathf.Max(xPtr[xShape.IndexWithBroadcast(b3, h3, w3, ch3)], bPtr[bShape.IndexWithBroadcast(b3, h3, w3, ch3)]);
        }

        private void GreaterInnerLoop(long n)
        {
            int i = (int)n * unrollSize;

            int b0 = 0, h0 = 0, w0 = 0, ch0 = 0;
            int b1 = 0, h1 = 0, w1 = 0, ch1 = 0;
            int b2 = 0, h2 = 0, w2 = 0, ch2 = 0;
            int b3 = 0, h3 = 0, w3 = 0, ch3 = 0;
            oShape.GetPositionsFromIndex(i + 0, ref b0, ref h0, ref w0, ref ch0);
            oShape.GetPositionsFromIndex(i + 1, ref b1, ref h1, ref w1, ref ch1);
            oShape.GetPositionsFromIndex(i + 2, ref b2, ref h2, ref w2, ref ch2);
            oShape.GetPositionsFromIndex(i + 3, ref b3, ref h3, ref w3, ref ch3);

            oPtr[i + 0] = (xPtr[xShape.IndexWithBroadcast(b0, h0, w0, ch0)] > bPtr[bShape.IndexWithBroadcast(b0, h0, w0, ch0)]) ? 1.0f : 0.0f;
            oPtr[i + 1] = (xPtr[xShape.IndexWithBroadcast(b1, h1, w1, ch1)] > bPtr[bShape.IndexWithBroadcast(b1, h1, w1, ch1)]) ? 1.0f : 0.0f;
            oPtr[i + 2] = (xPtr[xShape.IndexWithBroadcast(b2, h2, w2, ch2)] > bPtr[bShape.IndexWithBroadcast(b2, h2, w2, ch2)]) ? 1.0f : 0.0f;
            oPtr[i + 3] = (xPtr[xShape.IndexWithBroadcast(b3, h3, w3, ch3)] > bPtr[bShape.IndexWithBroadcast(b3, h3, w3, ch3)]) ? 1.0f : 0.0f;
        }

        private void GreaterEqualInnerLoop(long n)
        {
            int i = (int)n * unrollSize;

            int b0 = 0, h0 = 0, w0 = 0, ch0 = 0;
            int b1 = 0, h1 = 0, w1 = 0, ch1 = 0;
            int b2 = 0, h2 = 0, w2 = 0, ch2 = 0;
            int b3 = 0, h3 = 0, w3 = 0, ch3 = 0;
            oShape.GetPositionsFromIndex(i + 0, ref b0, ref h0, ref w0, ref ch0);
            oShape.GetPositionsFromIndex(i + 1, ref b1, ref h1, ref w1, ref ch1);
            oShape.GetPositionsFromIndex(i + 2, ref b2, ref h2, ref w2, ref ch2);
            oShape.GetPositionsFromIndex(i + 3, ref b3, ref h3, ref w3, ref ch3);

            oPtr[i + 0] = (xPtr[xShape.IndexWithBroadcast(b0, h0, w0, ch0)] >= bPtr[bShape.IndexWithBroadcast(b0, h0, w0, ch0)]) ? 1.0f : 0.0f;
            oPtr[i + 1] = (xPtr[xShape.IndexWithBroadcast(b1, h1, w1, ch1)] >= bPtr[bShape.IndexWithBroadcast(b1, h1, w1, ch1)]) ? 1.0f : 0.0f;
            oPtr[i + 2] = (xPtr[xShape.IndexWithBroadcast(b2, h2, w2, ch2)] >= bPtr[bShape.IndexWithBroadcast(b2, h2, w2, ch2)]) ? 1.0f : 0.0f;
            oPtr[i + 3] = (xPtr[xShape.IndexWithBroadcast(b3, h3, w3, ch3)] >= bPtr[bShape.IndexWithBroadcast(b3, h3, w3, ch3)]) ? 1.0f : 0.0f;
        }

        private void LessInnerLoop(long n)
        {
            int i = (int)n * unrollSize;

            int b0 = 0, h0 = 0, w0 = 0, ch0 = 0;
            int b1 = 0, h1 = 0, w1 = 0, ch1 = 0;
            int b2 = 0, h2 = 0, w2 = 0, ch2 = 0;
            int b3 = 0, h3 = 0, w3 = 0, ch3 = 0;
            oShape.GetPositionsFromIndex(i + 0, ref b0, ref h0, ref w0, ref ch0);
            oShape.GetPositionsFromIndex(i + 1, ref b1, ref h1, ref w1, ref ch1);
            oShape.GetPositionsFromIndex(i + 2, ref b2, ref h2, ref w2, ref ch2);
            oShape.GetPositionsFromIndex(i + 3, ref b3, ref h3, ref w3, ref ch3);

            oPtr[i + 0] = (xPtr[xShape.IndexWithBroadcast(b0, h0, w0, ch0)] < bPtr[bShape.IndexWithBroadcast(b0, h0, w0, ch0)]) ? 1.0f : 0.0f;
            oPtr[i + 1] = (xPtr[xShape.IndexWithBroadcast(b1, h1, w1, ch1)] < bPtr[bShape.IndexWithBroadcast(b1, h1, w1, ch1)]) ? 1.0f : 0.0f;
            oPtr[i + 2] = (xPtr[xShape.IndexWithBroadcast(b2, h2, w2, ch2)] < bPtr[bShape.IndexWithBroadcast(b2, h2, w2, ch2)]) ? 1.0f : 0.0f;
            oPtr[i + 3] = (xPtr[xShape.IndexWithBroadcast(b3, h3, w3, ch3)] < bPtr[bShape.IndexWithBroadcast(b3, h3, w3, ch3)]) ? 1.0f : 0.0f;
        }

        private void LessEqualInnerLoop(long n)
        {
            int i = (int)n * unrollSize;

            int b0 = 0, h0 = 0, w0 = 0, ch0 = 0;
            int b1 = 0, h1 = 0, w1 = 0, ch1 = 0;
            int b2 = 0, h2 = 0, w2 = 0, ch2 = 0;
            int b3 = 0, h3 = 0, w3 = 0, ch3 = 0;
            oShape.GetPositionsFromIndex(i + 0, ref b0, ref h0, ref w0, ref ch0);
            oShape.GetPositionsFromIndex(i + 1, ref b1, ref h1, ref w1, ref ch1);
            oShape.GetPositionsFromIndex(i + 2, ref b2, ref h2, ref w2, ref ch2);
            oShape.GetPositionsFromIndex(i + 3, ref b3, ref h3, ref w3, ref ch3);

            oPtr[i + 0] = (xPtr[xShape.IndexWithBroadcast(b0, h0, w0, ch0)] <= bPtr[bShape.IndexWithBroadcast(b0, h0, w0, ch0)]) ? 1.0f : 0.0f;
            oPtr[i + 1] = (xPtr[xShape.IndexWithBroadcast(b1, h1, w1, ch1)] <= bPtr[bShape.IndexWithBroadcast(b1, h1, w1, ch1)]) ? 1.0f : 0.0f;
            oPtr[i + 2] = (xPtr[xShape.IndexWithBroadcast(b2, h2, w2, ch2)] <= bPtr[bShape.IndexWithBroadcast(b2, h2, w2, ch2)]) ? 1.0f : 0.0f;
            oPtr[i + 3] = (xPtr[xShape.IndexWithBroadcast(b3, h3, w3, ch3)] <= bPtr[bShape.IndexWithBroadcast(b3, h3, w3, ch3)]) ? 1.0f : 0.0f;
        }

        private void EqualInnerLoop(long n)
        {
            int i = (int)n * unrollSize;

            int b0 = 0, h0 = 0, w0 = 0, ch0 = 0;
            int b1 = 0, h1 = 0, w1 = 0, ch1 = 0;
            int b2 = 0, h2 = 0, w2 = 0, ch2 = 0;
            int b3 = 0, h3 = 0, w3 = 0, ch3 = 0;
            oShape.GetPositionsFromIndex(i + 0, ref b0, ref h0, ref w0, ref ch0);
            oShape.GetPositionsFromIndex(i + 1, ref b1, ref h1, ref w1, ref ch1);
            oShape.GetPositionsFromIndex(i + 2, ref b2, ref h2, ref w2, ref ch2);
            oShape.GetPositionsFromIndex(i + 3, ref b3, ref h3, ref w3, ref ch3);

            oPtr[i + 0] = (xPtr[xShape.IndexWithBroadcast(b0, h0, w0, ch0)] == bPtr[bShape.IndexWithBroadcast(b0, h0, w0, ch0)]) ? 1.0f : 0.0f;
            oPtr[i + 1] = (xPtr[xShape.IndexWithBroadcast(b1, h1, w1, ch1)] == bPtr[bShape.IndexWithBroadcast(b1, h1, w1, ch1)]) ? 1.0f : 0.0f;
            oPtr[i + 2] = (xPtr[xShape.IndexWithBroadcast(b2, h2, w2, ch2)] == bPtr[bShape.IndexWithBroadcast(b2, h2, w2, ch2)]) ? 1.0f : 0.0f;
            oPtr[i + 3] = (xPtr[xShape.IndexWithBroadcast(b3, h3, w3, ch3)] == bPtr[bShape.IndexWithBroadcast(b3, h3, w3, ch3)]) ? 1.0f : 0.0f;
        }

        private void LogicalOrInnerLoop(long n)
        {
            int i = (int)n * unrollSize;

            int b0 = 0, h0 = 0, w0 = 0, ch0 = 0;
            int b1 = 0, h1 = 0, w1 = 0, ch1 = 0;
            int b2 = 0, h2 = 0, w2 = 0, ch2 = 0;
            int b3 = 0, h3 = 0, w3 = 0, ch3 = 0;
            oShape.GetPositionsFromIndex(i + 0, ref b0, ref h0, ref w0, ref ch0);
            oShape.GetPositionsFromIndex(i + 1, ref b1, ref h1, ref w1, ref ch1);
            oShape.GetPositionsFromIndex(i + 2, ref b2, ref h2, ref w2, ref ch2);
            oShape.GetPositionsFromIndex(i + 3, ref b3, ref h3, ref w3, ref ch3);

            oPtr[i + 0] = (Convert.ToBoolean(xPtr[xShape.IndexWithBroadcast(b0, h0, w0, ch0)]) || Convert.ToBoolean(bPtr[bShape.IndexWithBroadcast(b0, h0, w0, ch0)])) ? 1.0f : 0.0f;
            oPtr[i + 1] = (Convert.ToBoolean(xPtr[xShape.IndexWithBroadcast(b1, h1, w1, ch1)]) || Convert.ToBoolean(bPtr[bShape.IndexWithBroadcast(b1, h1, w1, ch1)])) ? 1.0f : 0.0f;
            oPtr[i + 2] = (Convert.ToBoolean(xPtr[xShape.IndexWithBroadcast(b2, h2, w2, ch2)]) || Convert.ToBoolean(bPtr[bShape.IndexWithBroadcast(b2, h2, w2, ch2)])) ? 1.0f : 0.0f;
            oPtr[i + 3] = (Convert.ToBoolean(xPtr[xShape.IndexWithBroadcast(b3, h3, w3, ch3)]) || Convert.ToBoolean(bPtr[bShape.IndexWithBroadcast(b3, h3, w3, ch3)])) ? 1.0f : 0.0f;
        }

        private void LogicalAndInnerLoop(long n)
        {
            int i = (int)n * unrollSize;

            int b0 = 0, h0 = 0, w0 = 0, ch0 = 0;
            int b1 = 0, h1 = 0, w1 = 0, ch1 = 0;
            int b2 = 0, h2 = 0, w2 = 0, ch2 = 0;
            int b3 = 0, h3 = 0, w3 = 0, ch3 = 0;
            oShape.GetPositionsFromIndex(i + 0, ref b0, ref h0, ref w0, ref ch0);
            oShape.GetPositionsFromIndex(i + 1, ref b1, ref h1, ref w1, ref ch1);
            oShape.GetPositionsFromIndex(i + 2, ref b2, ref h2, ref w2, ref ch2);
            oShape.GetPositionsFromIndex(i + 3, ref b3, ref h3, ref w3, ref ch3);

            oPtr[i + 0] = (Convert.ToBoolean(xPtr[xShape.IndexWithBroadcast(b0, h0, w0, ch0)]) && Convert.ToBoolean(bPtr[bShape.IndexWithBroadcast(b0, h0, w0, ch0)])) ? 1.0f : 0.0f;
            oPtr[i + 1] = (Convert.ToBoolean(xPtr[xShape.IndexWithBroadcast(b1, h1, w1, ch1)]) && Convert.ToBoolean(bPtr[bShape.IndexWithBroadcast(b1, h1, w1, ch1)])) ? 1.0f : 0.0f;
            oPtr[i + 2] = (Convert.ToBoolean(xPtr[xShape.IndexWithBroadcast(b2, h2, w2, ch2)]) && Convert.ToBoolean(bPtr[bShape.IndexWithBroadcast(b2, h2, w2, ch2)])) ? 1.0f : 0.0f;
            oPtr[i + 3] = (Convert.ToBoolean(xPtr[xShape.IndexWithBroadcast(b3, h3, w3, ch3)]) && Convert.ToBoolean(bPtr[bShape.IndexWithBroadcast(b3, h3, w3, ch3)])) ? 1.0f : 0.0f;
        }

        private void LogicalXorInnerLoop(long n)
        {
            int i = (int)n * unrollSize;

            int b0 = 0, h0 = 0, w0 = 0, ch0 = 0;
            int b1 = 0, h1 = 0, w1 = 0, ch1 = 0;
            int b2 = 0, h2 = 0, w2 = 0, ch2 = 0;
            int b3 = 0, h3 = 0, w3 = 0, ch3 = 0;
            oShape.GetPositionsFromIndex(i + 0, ref b0, ref h0, ref w0, ref ch0);
            oShape.GetPositionsFromIndex(i + 1, ref b1, ref h1, ref w1, ref ch1);
            oShape.GetPositionsFromIndex(i + 2, ref b2, ref h2, ref w2, ref ch2);
            oShape.GetPositionsFromIndex(i + 3, ref b3, ref h3, ref w3, ref ch3);

            oPtr[i + 0] = (Convert.ToBoolean(xPtr[xShape.IndexWithBroadcast(b0, h0, w0, ch0)]) ^ Convert.ToBoolean(bPtr[bShape.IndexWithBroadcast(b0, h0, w0, ch0)])) ? 1.0f : 0.0f;
            oPtr[i + 1] = (Convert.ToBoolean(xPtr[xShape.IndexWithBroadcast(b1, h1, w1, ch1)]) ^ Convert.ToBoolean(bPtr[bShape.IndexWithBroadcast(b1, h1, w1, ch1)])) ? 1.0f : 0.0f;
            oPtr[i + 2] = (Convert.ToBoolean(xPtr[xShape.IndexWithBroadcast(b2, h2, w2, ch2)]) ^ Convert.ToBoolean(bPtr[bShape.IndexWithBroadcast(b2, h2, w2, ch2)])) ? 1.0f : 0.0f;
            oPtr[i + 3] = (Convert.ToBoolean(xPtr[xShape.IndexWithBroadcast(b3, h3, w3, ch3)]) ^ Convert.ToBoolean(bPtr[bShape.IndexWithBroadcast(b3, h3, w3, ch3)])) ? 1.0f : 0.0f;
        }

        private void AddInnerLoopNoBroadcast(long n)
        {
            int i = (int)n * unrollSize;

            oPtr[i + 0] = xPtr[(i + 0) % xLen] + bPtr[(i + 0) % bLen];
            oPtr[i + 1] = xPtr[(i + 1) % xLen] + bPtr[(i + 1) % bLen];
            oPtr[i + 2] = xPtr[(i + 2) % xLen] + bPtr[(i + 2) % bLen];
            oPtr[i + 3] = xPtr[(i + 3) % xLen] + bPtr[(i + 3) % bLen];
        }

        private void SubInnerLoopNoBroadcast(long n)
        {
            int i = (int)n * unrollSize;

            oPtr[i + 0] = xPtr[(i + 0) % xLen] - bPtr[(i + 0) % bLen];
            oPtr[i + 1] = xPtr[(i + 1) % xLen] - bPtr[(i + 1) % bLen];
            oPtr[i + 2] = xPtr[(i + 2) % xLen] - bPtr[(i + 2) % bLen];
            oPtr[i + 3] = xPtr[(i + 3) % xLen] - bPtr[(i + 3) % bLen];
        }

        private void MulInnerLoopNoBroadcast(long n)
        {
            int i = (int)n * unrollSize;

            oPtr[i + 0] = xPtr[(i + 0) % xLen] * bPtr[(i + 0) % bLen];
            oPtr[i + 1] = xPtr[(i + 1) % xLen] * bPtr[(i + 1) % bLen];
            oPtr[i + 2] = xPtr[(i + 2) % xLen] * bPtr[(i + 2) % bLen];
            oPtr[i + 3] = xPtr[(i + 3) % xLen] * bPtr[(i + 3) % bLen];
        }

        private void DivInnerLoopNoBroadcast(long n)
        {
            int i = (int)n * unrollSize;

            oPtr[i + 0] = xPtr[(i + 0) % xLen] / bPtr[(i + 0) % bLen];
            oPtr[i + 1] = xPtr[(i + 1) % xLen] / bPtr[(i + 1) % bLen];
            oPtr[i + 2] = xPtr[(i + 2) % xLen] / bPtr[(i + 2) % bLen];
            oPtr[i + 3] = xPtr[(i + 3) % xLen] / bPtr[(i + 3) % bLen];
        }

        private void MinInnerLoopNoBroadcast(long n)
        {
            int i = (int)n * unrollSize;

            oPtr[i + 0] = Mathf.Min(xPtr[(i + 0) % xLen], bPtr[(i + 0) % bLen]);
            oPtr[i + 1] = Mathf.Min(xPtr[(i + 1) % xLen], bPtr[(i + 1) % bLen]);
            oPtr[i + 2] = Mathf.Min(xPtr[(i + 2) % xLen], bPtr[(i + 2) % bLen]);
            oPtr[i + 3] = Mathf.Min(xPtr[(i + 3) % xLen], bPtr[(i + 3) % bLen]);
        }

        private void MaxInnerLoopNoBroadcast(long n)
        {
            int i = (int)n * unrollSize;

            oPtr[i + 0] = Mathf.Max(xPtr[(i + 0) % xLen], bPtr[(i + 0) % bLen]);
            oPtr[i + 1] = Mathf.Max(xPtr[(i + 1) % xLen], bPtr[(i + 1) % bLen]);
            oPtr[i + 2] = Mathf.Max(xPtr[(i + 2) % xLen], bPtr[(i + 2) % bLen]);
            oPtr[i + 3] = Mathf.Max(xPtr[(i + 3) % xLen], bPtr[(i + 3) % bLen]);
        }

        private void GreaterInnerLoopNoBroadcast(long n)
        {
            int i = (int)n * unrollSize;

            oPtr[i + 0] = (xPtr[(i + 0) % xLen] > bPtr[(i + 0) % bLen]) ? 1.0f : 0.0f;
            oPtr[i + 1] = (xPtr[(i + 1) % xLen] > bPtr[(i + 1) % bLen]) ? 1.0f : 0.0f;
            oPtr[i + 2] = (xPtr[(i + 2) % xLen] > bPtr[(i + 2) % bLen]) ? 1.0f : 0.0f;
            oPtr[i + 3] = (xPtr[(i + 3) % xLen] > bPtr[(i + 3) % bLen]) ? 1.0f : 0.0f;
        }

        private void GreaterEqualInnerLoopNoBroadcast(long n)
        {
            int i = (int)n * unrollSize;

            oPtr[i + 0] = (xPtr[(i + 0) % xLen] >= bPtr[(i + 0) % bLen]) ? 1.0f : 0.0f;
            oPtr[i + 1] = (xPtr[(i + 1) % xLen] >= bPtr[(i + 1) % bLen]) ? 1.0f : 0.0f;
            oPtr[i + 2] = (xPtr[(i + 2) % xLen] >= bPtr[(i + 2) % bLen]) ? 1.0f : 0.0f;
            oPtr[i + 3] = (xPtr[(i + 3) % xLen] >= bPtr[(i + 3) % bLen]) ? 1.0f : 0.0f;
        }

        private void LessInnerLoopNoBroadcast(long n)
        {
            int i = (int)n * unrollSize;

            oPtr[i + 0] = (xPtr[(i + 0) % xLen] < bPtr[(i + 0) % bLen]) ? 1.0f : 0.0f;
            oPtr[i + 1] = (xPtr[(i + 1) % xLen] < bPtr[(i + 1) % bLen]) ? 1.0f : 0.0f;
            oPtr[i + 2] = (xPtr[(i + 2) % xLen] < bPtr[(i + 2) % bLen]) ? 1.0f : 0.0f;
            oPtr[i + 3] = (xPtr[(i + 3) % xLen] < bPtr[(i + 3) % bLen]) ? 1.0f : 0.0f;
        }

        private void LessEqualInnerLoopNoBroadcast(long n)
        {
            int i = (int)n * unrollSize;

            oPtr[i + 0] = (xPtr[(i + 0) % xLen] <= bPtr[(i + 0) % bLen]) ? 1.0f : 0.0f;
            oPtr[i + 1] = (xPtr[(i + 1) % xLen] <= bPtr[(i + 1) % bLen]) ? 1.0f : 0.0f;
            oPtr[i + 2] = (xPtr[(i + 2) % xLen] <= bPtr[(i + 2) % bLen]) ? 1.0f : 0.0f;
            oPtr[i + 3] = (xPtr[(i + 3) % xLen] <= bPtr[(i + 3) % bLen]) ? 1.0f : 0.0f;
        }

        private void EqualInnerLoopNoBroadcast(long n)
        {
            int i = (int)n * unrollSize;

            oPtr[i + 0] = (xPtr[(i + 0) % xLen] == bPtr[(i + 0) % bLen]) ? 1.0f : 0.0f;
            oPtr[i + 1] = (xPtr[(i + 1) % xLen] == bPtr[(i + 1) % bLen]) ? 1.0f : 0.0f;
            oPtr[i + 2] = (xPtr[(i + 2) % xLen] == bPtr[(i + 2) % bLen]) ? 1.0f : 0.0f;
            oPtr[i + 3] = (xPtr[(i + 3) % xLen] == bPtr[(i + 3) % bLen]) ? 1.0f : 0.0f;
        }

        private void LogicalOrInnerLoopNoBroadcast(long n)
        {
            int i = (int)n * unrollSize;

            oPtr[i + 0] = (Convert.ToBoolean(xPtr[(i + 0) % xLen]) || Convert.ToBoolean(bPtr[(i + 0) % bLen])) ? 1.0f : 0.0f;
            oPtr[i + 1] = (Convert.ToBoolean(xPtr[(i + 1) % xLen]) || Convert.ToBoolean(bPtr[(i + 1) % bLen])) ? 1.0f : 0.0f;
            oPtr[i + 2] = (Convert.ToBoolean(xPtr[(i + 2) % xLen]) || Convert.ToBoolean(bPtr[(i + 2) % bLen])) ? 1.0f : 0.0f;
            oPtr[i + 3] = (Convert.ToBoolean(xPtr[(i + 3) % xLen]) || Convert.ToBoolean(bPtr[(i + 3) % bLen])) ? 1.0f : 0.0f;
        }

        private void LogicalAndInnerLoopNoBroadcast(long n)
        {
            int i = (int)n * unrollSize;

            oPtr[i + 0] = (Convert.ToBoolean(xPtr[(i + 0) % xLen]) && Convert.ToBoolean(bPtr[(i + 0) % bLen])) ? 1.0f : 0.0f;
            oPtr[i + 1] = (Convert.ToBoolean(xPtr[(i + 1) % xLen]) && Convert.ToBoolean(bPtr[(i + 1) % bLen])) ? 1.0f : 0.0f;
            oPtr[i + 2] = (Convert.ToBoolean(xPtr[(i + 2) % xLen]) && Convert.ToBoolean(bPtr[(i + 2) % bLen])) ? 1.0f : 0.0f;
            oPtr[i + 3] = (Convert.ToBoolean(xPtr[(i + 3) % xLen]) && Convert.ToBoolean(bPtr[(i + 3) % bLen])) ? 1.0f : 0.0f;
        }

        private void LogicalXorInnerLoopNoBroadcast(long n)
        {
            int i = (int)n * unrollSize;

            oPtr[i + 0] = (Convert.ToBoolean(xPtr[(i + 0) % xLen]) ^ Convert.ToBoolean(bPtr[(i + 0) % bLen])) ? 1.0f : 0.0f;
            oPtr[i + 1] = (Convert.ToBoolean(xPtr[(i + 1) % xLen]) ^ Convert.ToBoolean(bPtr[(i + 1) % bLen])) ? 1.0f : 0.0f;
            oPtr[i + 2] = (Convert.ToBoolean(xPtr[(i + 2) % xLen]) ^ Convert.ToBoolean(bPtr[(i + 2) % bLen])) ? 1.0f : 0.0f;
            oPtr[i + 3] = (Convert.ToBoolean(xPtr[(i + 3) % xLen]) ^ Convert.ToBoolean(bPtr[(i + 3) % bLen])) ? 1.0f : 0.0f;
        }

        private void LogicalNotInnerLoop(long n)
        {
            int i = (int)n * unrollSize;

            oPtr[i + 0] = Convert.ToBoolean(xPtr[i + 0]) ? 0.0f : 1.0f;
            oPtr[i + 1] = Convert.ToBoolean(xPtr[i + 1]) ? 0.0f : 1.0f;
            oPtr[i + 2] = Convert.ToBoolean(xPtr[i + 2]) ? 0.0f : 1.0f;
            oPtr[i + 3] = Convert.ToBoolean(xPtr[i + 3]) ? 0.0f : 1.0f;
        }

        private static void ClampHWToTensorShape(TensorShape shape, ref int height, ref int width)
        {
            width = Math.Max(width, 0);
            height = Math.Max(height, 0);
            width = Math.Min(width, shape.width - 1);
            height = Math.Min(height, shape.height - 1);
        }
        private void Border2DInnerLoop(long n)
        {
            int i = (int)n;
            oPtr[i]  = alpha;
        }
        private void Pad2DEdgeInnerLoop(long n)
        {
            int i = (int)n;
            int b0 = 0, h0 = 0, w0 = 0, ch0 = 0;
            oShape.GetPositionsFromIndex(i, ref b0, ref h0, ref w0, ref ch0);
            h0 -= prePadY;
            w0 -= prePadX;

            ClampHWToTensorShape(xShape, ref h0, ref w0);

            oPtr[i] = xPtr[xShape.Index(b0, h0, w0, ch0)];
        }

        private void Pad2DReflectInnerLoop(long n)
        {
            int i = (int)n;
            int b0 = 0, h0 = 0, w0 = 0, ch0 = 0;
            oShape.GetPositionsFromIndex(i, ref b0, ref h0, ref w0, ref ch0);
            h0 -= prePadY;
            w0 -= prePadX;

            int lastXIndex = xShape.width - 1;
            int lastYIndex = xShape.height - 1;

            if (w0 < 0)
                w0 = -w0;
            else if (w0 > lastXIndex)
                w0 = lastXIndex - (w0 - lastXIndex);

            if (h0 < 0)
                h0 = -h0;
            else if (h0 > lastYIndex)
                h0 = lastYIndex - (h0 - lastYIndex);

            ClampHWToTensorShape(xShape, ref h0, ref w0);

            oPtr[i] = xPtr[xShape.Index(b0, h0, w0, ch0)];
        }

        private void Pad2DSymmetricInnerLoop(long n)
        {
            int i = (int)n;
            int b0 = 0, h0 = 0, w0 = 0, ch0 = 0;
            oShape.GetPositionsFromIndex(i, ref b0, ref h0, ref w0, ref ch0);
            h0 -= prePadY;
            w0 -= prePadX;

            int lastXIndex = xShape.width - 1;
            int lastYIndex = xShape.height - 1;

            if (w0 < 0)
                w0 = -w0 - 1;
            else if (w0 > lastXIndex)
                w0 = lastXIndex - (w0 - lastXIndex) + 1;

            if (h0 < 0)
                h0 = -h0 - 1;
            else if (h0 > lastYIndex)
                h0 = lastYIndex - (h0 - lastYIndex) + 1;

            ClampHWToTensorShape(xShape, ref h0, ref w0);

            oPtr[i] = xPtr[xShape.Index(b0, h0, w0, ch0)];
        }

        private float Add(float a, float b)
        {
            return a + b;
        }
        private float Sub(float a, float b)
        {
            return a - b;
        }
        private float Mul(float a, float b)
        {
            return a * b;
        }
        private float Div(float a, float b)
        {
            return a / b;
        }
        private float Min(float a, float b)
        {
            return Mathf.Min(a, b);
        }
        private float Max(float a, float b)
        {
            return Mathf.Max(a, b);
        }
        private float Greater(float a, float b)
        {
            return Convert.ToSingle(a > b);
        }
        private float GreaterEqual(float a, float b)
        {
            return Convert.ToSingle(a >= b);
        }
        private float Less(float a, float b)
        {
            return Convert.ToSingle(a < b);
        }
        private float LessEqual(float a, float b)
        {
            return Convert.ToSingle(a <= b);
        }
        private float Equal(float a, float b)
        {
            return Convert.ToSingle(a == b);
        }
        private float LogicalOr(float a, float b)
        {
            return Convert.ToSingle(Convert.ToBoolean(a) || Convert.ToBoolean(b));
        }
        private float LogicalAnd(float a, float b)
        {
            return Convert.ToSingle(Convert.ToBoolean(a) && Convert.ToBoolean(b));
        }
        private float LogicalXor(float a, float b)
        {
            return Convert.ToSingle(Convert.ToBoolean(a) ^ Convert.ToBoolean(b));
        }
        private float LogicalNot(float a)
        {
            return Convert.ToSingle(!Convert.ToBoolean(a));
        }
    }


} // namespace Barracuda
