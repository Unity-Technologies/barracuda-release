using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using UnityEngine.Assertions;
using Random = UnityEngine.Random;

namespace Unity.Barracuda {

/// <summary>
/// Internal `Tensor` data backed by managed array
/// </summary>
public class ArrayTensorData : UniqueResourceId, ITensorData
{
    internal BarracudaArray m_Array;

    /// <summary>
    /// Data storage array
    /// </summary>
    public BarracudaArray array { get { return m_Array; } }

    /// <summary>
    /// Create `ArrayTensorData` and allocate storage for `count` elements
    /// </summary>
    /// <param name="count">number of elements to pre-allocate</param>
    public ArrayTensorData(int count, DataType dataType = DataType.Float)
    {
        m_Array = new BarracudaArray(count, dataType);
    }

    /// <summary>
    /// Create `ArrayTensorData` and allocate storage for `Tensor` described by `shape`
    /// </summary>
    /// <param name="shape">shape</param>
    public ArrayTensorData(TensorShape shape, DataType dataType = DataType.Float) : this(shape.length, dataType)
    {
    }

    /// <summary>
    /// Finalizer
    /// </summary>
    ~ArrayTensorData()
    {
        Dispose();
    }

    /// <summary>
    /// Dispose storage
    /// </summary>
    public virtual void Dispose()
    {
        m_Array = null;
    }

    /// <inheritdoc/>
    public virtual void Reserve(int count)
    {
        if (count > m_Array.Length)
            m_Array = new BarracudaArray(count, m_Array.Type);
    }

    /// <inheritdoc/>
    public virtual void Upload(float[] data, TensorShape shape, int managedBufferStartIndex = 0)
    {
        var numItemToCopy = shape.length;
        var numItemAvailableInData = data.Length - managedBufferStartIndex;

        Assert.IsTrue(managedBufferStartIndex >= 0);
        Assert.IsTrue(numItemToCopy <= numItemAvailableInData);

        Reserve(numItemToCopy);
        BarracudaArray.Copy(data, managedBufferStartIndex, m_Array, 0, numItemToCopy);
    }

    /// <inheritdoc/>
    public virtual bool ScheduleAsyncDownload(int count)
    {
        return true;
    }

    /// <inheritdoc/>
    public virtual float[] Download(TensorShape shape)
    {
        //;;D.logStackTraceEnabled = true;
        //;;D.Log("Download ArrayTensorData " + count + " from " + m_Array.Length + " @ " + ToString());
        //;;D.logStackTraceEnabled = false;

        var count = shape.length;
        Assert.IsTrue(m_Array.Length >= count);

        var dest = new float[count];
        BarracudaArray.Copy(m_Array, 0, dest, 0, count);
        return dest;
    }

    /// <inheritdoc/>
    public virtual BarracudaArray SharedAccess(out int offset)
    {
        offset = 0;
        return m_Array;
    }

    /// <inheritdoc/>
    public virtual int maxCapacity { get
    {
        return m_Array.Length;
    } }

    /// <inheritdoc/>
    public virtual DataType dataType { get
    {
        return m_Array.Type;
    } }

    /// <inheritdoc/>
    public virtual bool inUse { get
    {
        return true;
    } }

    /// <inheritdoc/>
    public virtual bool isGPUMem { get
    {
        return false;
    } }

    /// <summary>
    /// Storage summary as string
    /// </summary>
    /// <returns>storage summary as string</returns>
    public override string ToString()
    {
        return string.Format("(CPU array: {0} max: {1})",
            GetHashCode(), m_Array?.Length);
    }
}

/// <summary>
/// Base class to track unique resource by an id.
/// </summary>
public class UniqueResourceId: IUniqueResource
{
    class UniqueResourceHelper {
        public int lastIdRequested;
    }
    static UniqueResourceHelper SpinLock = new UniqueResourceHelper();

    /// <inheritdoc/>
    public int uniqueId { get; internal set; }

    public UniqueResourceId()
    {
        uniqueId = GetUniqueId();
    }

    public static int GetUniqueId()
    {
        lock(SpinLock)
        {
            return SpinLock.lastIdRequested++;
        }
    }
}

/// <summary>
/// Internal `Tensor` data backed by managed array that is shared between multiple tensors
/// </summary>
public class SharedArrayTensorData : UniqueResourceId, ITensorData
{
    internal BarracudaArray m_Array;
    internal int m_Offset;
    internal int m_Count;

    /// <summary>
    /// Data storage array
    /// </summary>
    public BarracudaArray array { get { return m_Array; } }

    /// <summary>
    /// Offset in storage array
    /// </summary>
    public int offset { get { return m_Offset; } }

    /// <summary>
    /// Data element count
    /// </summary>
    public int count { get { return m_Count; } }

    /// <summary>
    /// Create `SharedArrayTensorData` with supplied shared `data`
    /// </summary>
    /// <param name="data">shared array</param>
    public SharedArrayTensorData(float[] data) : this(new BarracudaArrayFromManagedArray(data), 0, data.Length)
    {
    }

    /// <summary>
    /// Create `SharedArrayTensorData` with supplied shared `data`
    /// </summary>
    /// <param name="data">shared array</param>
    public SharedArrayTensorData(BarracudaArray data) : this(data, 0, data.Length)
    {
    }

    internal SharedArrayTensorData(BarracudaArray data, TensorShape shape, int offset) : this(data, offset, shape.length)
    {
    }

    internal SharedArrayTensorData(float[] data, int offset, int count) : this(new BarracudaArrayFromManagedArray(data), offset, count)
    {
    }

    internal SharedArrayTensorData(BarracudaArray data, int offset, int count)
    {
        Assert.IsTrue(offset >= 0);
        m_Array = data;
        m_Offset = offset;
        Assert.IsTrue(count >= 0);
        Assert.IsTrue(offset + count <= m_Array.Length);
        m_Count = count;
    }

    /// <summary>
    /// Finalize
    /// </summary>
    ~SharedArrayTensorData()
    {
        Dispose();
    }

    /// <summary>
    /// Dispose storage
    /// </summary>
    public virtual void Dispose()
    {
    }

    /// <inheritdoc/>
    public virtual void Reserve(int count)
    {
        // currently always readonly
        throw new InvalidOperationException("SharedArrayTensorData is readonly!");
    }

    /// <inheritdoc/>
    public virtual void Upload(float[] data, TensorShape shape, int managedBufferStartIndex = 0)
    {
        // currently always readonly
        throw new InvalidOperationException("SharedArrayTensorData is readonly!");
    }

    /// <inheritdoc/>
    public virtual bool ScheduleAsyncDownload(int count)
    {
        return true;
    }

    /// <inheritdoc/>
    public virtual float[] Download(TensorShape shape)
    {
        //;;D.logStackTraceEnabled = true;
        //;;D.Log("Download SharedArrayTensorData " + count + " from " + m_Count + " @ " + ToString());
        //;;D.logStackTraceEnabled = false;

        var count = shape.length;
        Assert.IsTrue(m_Count >= count);

        var dest = new float[count];
        BarracudaArray.Copy(m_Array, m_Offset, dest, 0, count);
        return dest;
    }

    /// <inheritdoc/>
    public virtual BarracudaArray SharedAccess(out int offset)
    {
        offset = m_Offset;
        return m_Array;
    }

    /// <inheritdoc/>
    public virtual int maxCapacity { get
    {
        return m_Count;
    } }

    /// <inheritdoc/>
    public virtual DataType dataType { get
    {
        return m_Array.Type;
    } }

    /// <inheritdoc/>
    public virtual bool inUse { get
    {
        return true;
    } }

    /// <inheritdoc/>
    public virtual bool isGPUMem { get
    {
        return false;
    } }


    /// <summary>
    /// Storage summary as string
    /// </summary>
    /// <returns>storage summary as string</returns>
    public override string ToString()
    {
        return string.Format("(CPU shared: {0} max: {1} offset: {2} count: {3})",
            GetHashCode(), m_Array.Length, m_Offset, m_Count);
    }
}

/// <summary>
/// Reference CPU implementation of `IOps`
/// </summary>
public class ReferenceCPUOps : IOps
{
    private IModelExecutionsReporter m_ModelExecutionsReporter;
    private ITensorAllocator m_Allocator;
    private StringCache m_StringCache = new StringCache();

    /// <inheritdoc/>
    public virtual void PostLayerCleanup()
    {
        m_Allocator.PostLayerCleanup();
    }

    /// <summary>
    /// Create `ReferenceCPUOps`
    /// </summary>
    /// <param name="allocator">allocator</param>
    public ReferenceCPUOps(ITensorAllocator allocator = null)
    {
        if (allocator == null)
            allocator = new TensorCachingAllocator();
        m_Allocator = allocator;
    }

    #region Tensor creation helpers (for reference implementation only)
    /// <summary>
    /// Allocate new `Tensor` via allocator using LayerOutput allocation scope.
    /// Should only be used on reference backend, production backends should use explicit
    /// allocation scope for better peak mem usage.
    /// </summary>
    /// <param name="dataType">data type</param>
    /// <param name="s">shape</param>
    /// <param name="scope">tensor lifetime scope</param>
    /// <param name="name">name</param>
    /// <returns>new `Tensor`</returns>
    private Tensor NewTensor(DataType dataType, TensorShape s)
    {
        return NewTensor(dataType, s, AllocScope.LayerOutput);
    }

    /// <summary>
    /// Allocate new `Tensor` via allocator using LayerOutput allocation scope.
    /// Should only be used on reference backend, production backends should use explicit
    /// allocation scope for better peak mem usage.
    /// </summary>
    /// <param name="t">`Tensor`</param>
    /// <returns>new `Tensor`</returns>
    private Tensor NewTensorLike(Tensor t)
    {
        return NewTensorLike(t, AllocScope.LayerOutput);
    }

    /// <summary>
    /// Allocate new `Tensor` via allocator using LayerOutput allocation scope.
    /// Should only be used on reference backend, production backends should use explicit
    /// allocation scope for better peak mem usage.
    /// </summary>
    /// <param name="dataType">data type</param>
    /// <param name="b">batch</param>
    /// <param name="ch">channels</param>
    /// <param name="name">name</param>
    /// <returns>new `Tensor`</returns>
    private Tensor NewTensor(DataType dataType, int b, int ch, string name = "")
    {
        return NewTensor(dataType, new TensorShape(b, ch), AllocScope.LayerOutput, name);
    }

    /// <summary>
    /// Allocate new `Tensor` via allocator using LayerOutput allocation scope.
    /// Should only be used on reference backend, production backends should use explicit
    /// allocation scope for better peak mem usage.
    /// </summary>
    /// <param name="dataType">data type</param>
    /// <param name="b">batch</param>
    /// <param name="h">height</param>
    /// <param name="w">width</param>
    /// <param name="ch">channels</param>
    /// <param name="name">name</param>
    /// <returns>new `Tensor`</returns>
    private Tensor NewTensor(DataType dataType, int b, int h, int w, int ch, string name = "")
    {
        return NewTensor(dataType, new TensorShape(b, h, w, ch), AllocScope.LayerOutput, name);
    }

    #endregion

    /// <summary>
    /// Allocate new `Tensor` via allocator
    /// </summary>
    /// <param name="dataType">data type</param>
    /// <param name="s">shape</param>
    /// <param name="scope">tensor lifetime scope</param>
    /// <param name="name">name</param>
    /// <returns>new `Tensor`</returns>
    protected Tensor NewTensor(DataType dataType, TensorShape s, AllocScope scope, string name = "")
    {
        if (name == "")
            name = (scope == AllocScope.LayerOutput ? "LayerOutput" : "InternalToLayer");

        var tensor = m_Allocator.Alloc(s, scope, dataType);
        tensor.name = name;

        return tensor;
    }

    /// <summary>
    /// Allocate new `Tensor` similar to specified `Tensor` `t`
    /// </summary>
    /// <param name="t">`Tensor`</param>
    /// <param name="scope">tensor lifetime scope</param>
    /// <returns>new `Tensor`</returns>
    protected Tensor NewTensorLike(Tensor t, AllocScope scope)
    {
        return NewTensor(t.dataType, t.shape, scope);
    }

    /// <summary>
    /// Allocate new `Tensor` corresponding to max shape of specified `tensors`
    /// </summary>
    /// <param name="tensors">tensors</param>
    /// <param name="scope">tensor lifetime scope</param>
    /// <param name="validateType">should this method validate that all tensors are the same type</param>
    /// <returns>new `Tensor`</returns>
    protected Tensor NewTensorLike(Tensor[] tensors, AllocScope scope, bool validateType = true)
    {
        Assert.IsTrue(tensors.Length > 0);

        var O = NewTensor(tensors[0].dataType, TensorExtensions.MaxShape(tensors), scope);
        foreach (var t in tensors)
        {
            if (validateType)
                Assert.AreEqual(O.dataType, t.dataType);
            for (int i = 0; i < TensorShape.MaxRank; ++i)
            {
                Assert.IsTrue((t.shape[i] == 1) || (t.shape[i] == O.shape[i]));
            }
        }

        return O;
    }

    /// <summary>
    /// Check if `fusedActivation` is supported in-place
    /// </summary>
    /// <param name="fusedActivation">fused activation type</param>
    /// <returns>`true` if supported in-place</returns>
    protected virtual bool IsFusedActivationSupported(Layer.FusedActivation fusedActivation)
    {
        switch (fusedActivation)
        {
            case Layer.FusedActivation.None:
                return true;
            default:
                return false;
        }
    }

    /// <summary>
    /// Allocate new `Tensor` via allocator
    /// tensor lifetime will be OutputLayer if activation is supported in place, InternalToLayer otherwise.
    /// </summary>
    /// <param name="dataType">data type</param>
    /// <param name="shape">shape of the tensor to be created</param>
    /// <param name="fusedActivation">fused activation type</param>
    /// <returns>new `Tensor`</returns>
    protected Tensor NewTensorForFusedActivation(DataType dataType, TensorShape shape, Layer.FusedActivation fusedActivation)
    {
        if (IsFusedActivationSupported(fusedActivation))
            return NewOutputTensor(dataType, shape);
        else
            return NewTempTensor(dataType, shape);
    }

    /// <summary>
    /// Allocate new `Tensor` via allocator using AllocScope.LayerOutput scope
    /// </summary>
    /// <param name="type">data type</param>
    /// <param name="s">shape of the tensor to be created</param>
    /// <param name="name">tensor name</param>
    /// <returns>new `Tensor`</returns>
    protected Tensor NewOutputTensor(DataType type, TensorShape s, string name = "")
    {
        return NewTensor(type, s, AllocScope.LayerOutput, name);
    }

    /// <summary>
    /// Allocate new `Tensor` via allocator using AllocScope.InternalToLayer scope
    /// </summary>
    /// <param name="type">data type</param>
    /// <param name="s">shape of the tensor to be created</param>
    /// <param name="name">tensor name</param>
    /// <returns>new `Tensor`</returns>
    protected Tensor NewTempTensor(DataType type, TensorShape s, string name = "")
    {
        return NewTensor(type, s, AllocScope.InternalToLayer, name);
    }

#if ENABLE_BARRACUDA_STATS
    /// <inheritdoc/>
    public virtual IEnumerable<TempMemoryStatistics> GetTempMemoryStatistics()
    {
        return Enumerable.Empty<TempMemoryStatistics>();
    }
#endif //ENABLE_BARRACUDA_STATS

    /// <inheritdoc/>
    public virtual void ResetAllocator(bool keepCachedMemory = true)
    {
        m_Allocator.Reset(keepCachedMemory);
    }

    /// <inheritdoc/>
    public void SetModelExecutionsReporter(IModelExecutionsReporter executionsReporter)
    {
        m_ModelExecutionsReporter = executionsReporter;
    }

    /// <inheritdoc/>
    public IModelExecutionsReporter GetModelExecutionsReporter()
    {
        return m_ModelExecutionsReporter;
    }

    private float ApplyFusedActivation(float v, Layer.FusedActivation fusedActivation)
    {
        switch (fusedActivation)
        {
            case Layer.FusedActivation.None:
                break;
            case Layer.FusedActivation.Relu:
                v = Mathf.Max(v, 0.0f);
                break;
            case Layer.FusedActivation.Tanh:
                v = MathfEx.Tanh(v);
                break;
            case Layer.FusedActivation.Softplus:
                v = Mathf.Log(Mathf.Exp(v) + 1f);
                break;
            case Layer.FusedActivation.Sigmoid:
                v = 1f / (1f + Mathf.Exp(-v));
                break;
            case Layer.FusedActivation.Relu6:
                v = Mathf.Min(Mathf.Max(0f, v), 6f);
                break;
            case Layer.FusedActivation.Swish:
                v = v / (1f + Mathf.Exp(-v));
                break;
            case Layer.FusedActivation.Neg:
                v = -v;
                break;
            case Layer.FusedActivation.Sqrt:
                v = Mathf.Sqrt(v);
                break;
            case Layer.FusedActivation.Exp:
                v = Mathf.Exp(v);
                break;
            case Layer.FusedActivation.Log:
                v = Mathf.Log(v);
                break;
            case Layer.FusedActivation.Acos:
                v = Mathf.Acos(v);
                break;
            case Layer.FusedActivation.Acosh:
                v = Mathf.Log(v + Mathf.Sqrt(v * v - 1.0f));
                break;
            case Layer.FusedActivation.Asin:
                v = Mathf.Asin(v);
                break;
            case Layer.FusedActivation.Asinh:
                v = Mathf.Log(v + Mathf.Sqrt(v * v + 1.0f));
                break;
            case Layer.FusedActivation.Atan:
                v = Mathf.Atan(v);
                break;
            case Layer.FusedActivation.Atanh:
                v = 0.5f * Mathf.Log((1.0f + v) / (1.0f - v));
                break;
            case Layer.FusedActivation.Cos:
                v = Mathf.Cos(v);
                break;
            case Layer.FusedActivation.Cosh:
                v = 0.5f * (Mathf.Exp(v) + Mathf.Exp(-v));
                break;
            case Layer.FusedActivation.Sin:
                v = Mathf.Sin(v);
                break;
            case Layer.FusedActivation.Sinh:
                v = 0.5f * (Mathf.Exp(v) - Mathf.Exp(-v));
                break;
            case Layer.FusedActivation.Tan:
                v = Mathf.Tan(v);
                break;
            case Layer.FusedActivation.Erf:
            {
                // Abramowitz/Stegun approximations
                // erf(x) = -erf(-x)
                float x = Mathf.Abs(v);

                float p = 0.3275911f;
                float a1 = 0.254829592f; float a2 = -0.284496736f; float a3 = 1.421413741f;
                float a4 = -1.453152027f; float a5 = 1.061405429f;

                float t = 1 / (1 + p * x);
                float t2 = t * t;
                float t3 = t2 * t;
                float t4 = t3 * t;
                float t5 = t4 * t;

                v = Mathf.Sign(v)*(1 - (a1*t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5)*Mathf.Exp(-x * x));
                break;
            }
            default:
                throw new NotImplementedException();
        }
        return v;
    }

    /// <inheritdoc/>
    public virtual Tensor Dense3(Tensor X, Tensor W, Tensor B)
    {
        return Add(new[] { MatMul(X, 3, W, 2), Reshape(B, new TensorShape(1, 1, B.length, 1)) });
    }

    // ---------------------------------------------------------------------------------
    /// <inheritdoc/>
    public virtual Tensor MatMul(Tensor X, int rankX, Tensor Y, int rankY)
    {
        // Barracuda Tensor layout is not broadcast friendly:
        // rank4: NHWC
        // rank3: N_WC
        // rank2: N__C
        // rank1: N___
        // on top of things, ONNX does not transpose layout like it does for conv.
        // => so to get broadcast correctly we need to convert our Barracuda Tensor to an ONNX-broadcastable layout
        // rank4: NCHW
        // rank3: _NCW
        // rank2: __NC
        // rank1: ___N
        // and then perform the broadcast MatMul
        // the input tensor ranks are computed at import time and stored in the layer (TODO: keep track of it in the Tensor itself)

        // support for legacy case where rank needs to be inferred at runtime
        if (rankX < 0 || rankY < 0)
            ModelAnalyzer.LegacyGetXYRanks(X.shape, Y.shape, out rankX, out rankY);

        var onnxXshape = Compiler.IRShapeInferenceHelper.ShapeInference.BarracudaShapeToOnnxLayout(X.shape, rankX);
        var onnxYshape = Compiler.IRShapeInferenceHelper.ShapeInference.BarracudaShapeToOnnxLayout(Y.shape, rankY);

        int rankO = Math.Max(rankX, rankY);

        if (rankO <= 2)
            return MatMul(X, false, Y, false);

        // pad 1 on front of shape to both be rankO shape
        for (int i = rankX; i < rankO; i++)
            onnxXshape.Insert(0, 1);

        for (int i = rankY; i < rankO; i++)
            onnxYshape.Insert(0, 1);

        // Max values for X, Y from ONNX shape (needed for modulo later)
        int xN = 1;
        int yN = 1;
        int xC = 1;
        int yC = 1;

        int matN = 1;
        int matC = 1;
        int matH = 1;
        int matW = 1;
        Tensor O;
        if (rankO == 3)
        {
            xC = onnxXshape[0];
            yC = onnxYshape[0];
            matC = Math.Max(xC, yC);
            matH = onnxXshape[1];
            matW = onnxYshape[2];
            O = NewTensor(X.dataType, new TensorShape(matC, 1, matW, matH));
        }
        else
        {
            xN = onnxXshape[0];
            yN = onnxYshape[0];

            xC = onnxXshape[1];
            yC = onnxYshape[1];

            matN = Math.Max(xN, yN);
            matC = Math.Max(xC, yC);
            matH = onnxXshape[2];
            matW = onnxYshape[3];
            O = NewTensor(X.dataType, new TensorShape(matN, matH, matW, matC));
        }

        var Xt = Transpose(X, new[] { 0, 3, 1, 2 });
        var Yt = Transpose(Y, new[] { 0, 3, 1, 2 });
        if(rankX == 2)
            Xt = Reshape(Xt, new TensorShape(1, 1, Xt.batch, Xt.height));
        else if (rankX == 3)
            Xt = Reshape(Xt, new TensorShape(1, Xt.batch, Xt.height, Xt.channels));
        if (rankY == 2)
            Yt = Reshape(Yt, new TensorShape(1, 1, Yt.batch, Yt.height));
        else if (rankY == 3)
            Yt = Reshape(Yt, new TensorShape(1, Yt.batch, Yt.height, Yt.channels));

        var startsX = new[] { 0, 0, 0, 0 };
        var startsY = new[] { 0, 0, 0, 0 };

        var endsX = new[] { 1, 1, Xt.width, Xt.channels};
        var endsY = new[] { 1, 1, Yt.width, Yt.channels};
        var strides = new[] { 1, 1, 1, 1 };

        for (int b = 0; b < matN; b++)
        {
            Tensor Ob = NewTensorLike(O);

            if (rankX == 4)
            {
                startsX[0] = b % xN;
                endsX[0] = b % xN + 1;
            }
            if (rankY == 4)
            {
                startsY[0] = b % yN;
                endsY[0] = b % yN + 1;
            }

            for (int c = 0; c < matC; c++)
            {
                if (rankX >= 3)
                {
                    startsX[1] = c % xC;
                    endsX[1] = c % xC + 1;
                }
                if (rankY >= 3)
                {
                    startsY[1] = c % yC;
                    endsY[1] = c % yC + 1;
                }

                // __NC -> N__C
                Tensor Xs = StridedSlice(Xt, startsX, endsX, strides); Xs = Reshape(Xs, new TensorShape(Xt.width, Xt.channels));
                Tensor Ys = StridedSlice(Yt, startsY, endsY, strides); Ys = Reshape(Ys, new TensorShape(Yt.width, Yt.channels));
                Tensor Oc = MatMul(Xs, false, Ys, false);
                if(rankO == 2)
                {
                    Ob = Oc;
                }
                if (rankO == 3)
                {
                    Oc = Transpose(Oc, new[] { 1, 2, 3, 0 }); // N__C -> _1,C,N
                    if (c == 0)
                        Ob = Oc;
                    else
                        Ob = Concat(new[] { Ob, Oc }, TensorShape.DataBatch);
                }
                else if (rankO == 4)
                {
                    Oc = Reshape(Oc, new TensorShape(1, Oc.batch, Oc.channels, 1)); // N__C -> _,N,C,_
                    if (c == 0)
                        Ob = Oc;
                    else
                        Ob = Concat(new[] { Ob, Oc }, TensorShape.C);
                }
            }
            if (b == 0)
                O = Ob;
            else
                O = Concat(new[] { O, Ob }, TensorShape.DataBatch);
        }
        return O;
    }

    /// <summary>
    /// Simple 2D matrix multiplication O = `X` ⨯ `Y`
    /// </summary>
    /// <param name="X">left Tensor</param>
    /// <param name="xTranspose">`X` transposed data flag</param>
    /// <param name="Y">right Tensor</param>
    /// <param name="yTranspose">`Y` transposed data flag</param>
    /// <returns>output Tensor</returns>
    public virtual Tensor MatMul(Tensor X, bool xTranspose, Tensor Y, bool yTranspose)
    {
        Assert.IsTrue(X.dimensions <= 2);
        Assert.IsTrue(Y.dimensions <= 2);
        X = Flatten(X);
        Y = Flatten(Y);

        if (xTranspose)
            X = Transpose(X);
        if (yTranspose)
            Y = Transpose(Y);

        Assert.AreEqual(X.flatWidth, Y.flatHeight);
        var O = NewTensor(X.dataType, X.flatHeight, Y.flatWidth);

        for (int y = 0; y < O.flatHeight; ++y)
            for (int x = 0; x < O.flatWidth; ++x)
            {
                float v = 0;
                for (int i = 0; i < X.flatWidth; ++i)
                {
                    v += X[y, i] * Y[i, x];
                }
                O[y, x] = v;
            }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Dense(Tensor X, Tensor W, Tensor B, Layer.FusedActivation fusedActivation)
    {
        Assert.IsTrue(W.dimensions <= 2);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(X.flatWidth, W.flatHeight);

        var O = NewTensor(X.dataType, X.flatHeight, W.flatWidth);

        for (int y = 0; y < O.flatHeight; ++y)
            for (int x = 0; x < O.flatWidth; ++x)
            {
                float v = B[x];
                for (int i = 0; i < X.flatWidth; ++i)
                {
                    v += X[y, i] * W[i, x];
                }
                O[y, x] = ApplyFusedActivation(v, fusedActivation);
            }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Conv2D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, Layer.FusedActivation fusedActivation)
    {
        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(X.channels, K.kernelDepth);
        Assert.AreEqual(K.kernelCount, B.flatWidth);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);

        var O = NewTensor(X.dataType, X.shape.ApplyKernel(K.shape, stride, pad));

        for (var n = 0; n < O.batch; ++n)
            for (var y = 0; y < O.height; ++y)
                for (var x = 0; x < O.width; ++x)
                    for (var k = 0; k < K.kernelCount; ++k)
                    {
                        float v = B[k];
                        for (int dy = 0; dy < K.kernelHeight; ++dy)
                        {
                            for (int dx = 0; dx < K.kernelWidth; ++dx)
                            {
                                int oy = y * stride[1] + dy - pad[1];
                                int ox = x * stride[0] + dx - pad[0];

                                if (oy < 0) continue;
                                if (oy >= X.height) continue;
                                if (ox < 0) continue;
                                if (ox >= X.width) continue;

                                for (var c = 0; c < X.channels; ++c)
                                {
                                    float xv = X[n, oy, ox, c];
                                    float kv = K[dy, dx, c, k];

                                    v += xv * kv;
                                }
                            }
                        }
                        O[n, y, x, k] = ApplyFusedActivation(v, fusedActivation);
                    }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Conv3D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, Layer.FusedActivation fusedActivation)
    {
        Assert.IsTrue(X.shape.IsNDHWC());
        Assert.AreEqual(X.channels, K.kernelDepth);
        Assert.AreEqual(K.kernelCount, B.flatWidth);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(stride.Length, 3);//WHD
        Assert.AreEqual(pad.Length, 6);

        var O = NewTensor(X.dataType, X.shape.ApplyKernel(K.shape, stride, pad));

        for (var n = 0; n < O.batch; ++n)
        for (var d = 0; d < O.depth; ++d)
        for (var y = 0; y < O.height; ++y)
        for (var x = 0; x < O.width; ++x)
        for (var k = 0; k < K.kernelCount; ++k)
        {
            float v = B[k];
            for (int dd = 0; dd < K.kernelSpatialDepth; ++dd)
            {
                for (int dy = 0; dy < K.kernelHeight; ++dy)
                {
                    for (int dx = 0; dx < K.kernelWidth; ++dx)
                    {
                        int od = d * stride[2] + dd - pad[2];
                        int oy = y * stride[1] + dy - pad[1];
                        int ox = x * stride[0] + dx - pad[0];

                        if (od < 0) continue;
                        if (od >= X.depth) continue;
                        if (oy < 0) continue;
                        if (oy >= X.height) continue;
                        if (ox < 0) continue;
                        if (ox >= X.width) continue;

                        for (var c = 0; c < X.channels; ++c)
                        {
                            float xv = X[ n, od, oy, ox, c];
                            float kv = K[ 0, dd, dy, 0, 0, dx, c, k];
                            v += xv * kv;
                        }
                    }
                }
            }
            O[ n, d, y, x, k] = ApplyFusedActivation(v, fusedActivation);
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor DepthwiseConv2D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, Layer.FusedActivation fusedActivation)
    {
        if (K.kernelDepth != 1)
            throw new NotImplementedException("DepthwiseConv2D only support number of groups == number of input channels at the moment.");

        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(K.kernelDepth, 1);
        Assert.AreEqual(K.kernelCount, X.channels);
        Assert.AreEqual(K.kernelCount, B.flatWidth);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(stride.Length, 2);//WH
        Assert.AreEqual(pad.Length, 4);

        // ONNX: (M x C/group x kH x kW)
        // TF: [H, W, in_channels, channel_multiplier]

        // TF pseudocode:
        // output[b, i, j, k * channel_multiplier + q] =
        // sum_{di, dj}
        //      input [b, i + di, j + dj, k] *
        //      filter[di, dj, k, q] *

        var O = NewTensor(X.dataType, X.shape.ApplyKernel(K.shape, stride, pad));

        for (var n = 0; n < O.batch; ++n)
            for (var y = 0; y < O.height; ++y)
                for (var x = 0; x < O.width; ++x)
                    for (var k = 0; k < K.kernelCount; ++k)
                    {
                        float v = B[k];
                        for (int dy = 0; dy < K.kernelHeight; ++dy)
                            for (int dx = 0; dx < K.kernelWidth; ++dx)
                            {
                                int oy = y * stride[1] + dy - pad[1];
                                int ox = x * stride[0] + dx - pad[0];

                                if (oy < 0) continue;
                                if (oy >= X.height) continue;
                                if (ox < 0) continue;
                                if (ox >= X.width) continue;

                                float xv = X[n, oy, ox, k];
                                float kv = K[dy, dx, 0, k];
                                v += xv * kv;
                            }
                        O[n, y, x, k] =  ApplyFusedActivation(v, fusedActivation);
                    }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Conv2DTrans(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, int[] outputAdjustment, Layer.FusedActivation fusedActivation)
    {
        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(X.channels, K.kernelDepth);
        Assert.AreEqual(K.kernelCount, B.flatWidth);
        Assert.AreEqual(B.flatWidth, B.length);
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);
        Assert.AreEqual(pad[0],pad[2]);
        Assert.AreEqual(pad[1],pad[3]);

        var O = NewTensor(X.dataType, X.shape.ApplyKernelInverse(K.shape, stride, pad, outputAdjustment));
        int prePadW = K.kernelWidth  - pad[0] - 1;
        int prePadH = K.kernelHeight - pad[1] - 1;
        int strideH = 1;
        int strideW = 1;

        for (var n = 0; n < O.batch; ++n)
            for (var y = 0; y < O.height; ++y)
                for (var x = 0; x < O.width; ++x)
                    for (var k = 0; k < K.kernelCount; ++k)
                    {
                        float v = B[k];
                        for (int dy = 0; dy < K.kernelHeight; dy += strideH)
                            for (int dx = 0; dx < K.kernelWidth; dx += strideW)
                            {
                                int readX = (x + dx - prePadW) / stride[0];
                                int readY = (y + dy - prePadH) / stride[1];

                                if ((x + dx - prePadW) % stride[0] != 0) continue;
                                if ((y + dy - prePadH) % stride[0] != 0) continue;
                                if (readX < 0) continue;
                                if (readX >= X.width) continue;
                                if (readY < 0) continue;
                                if (readY >= X.height) continue;

                                for (var c = 0; c < X.channels; ++c)
                                {
                                    float xv = X[n, readY, readX, c];
                                    float kv = K[K.kernelHeight - 1 - dy,
                                                 K.kernelWidth  - 1 - dx, c, k];
                                    v += xv * kv;
                                }
                            }

                        O[n, y, x, k] = ApplyFusedActivation(v, fusedActivation);
                    }
        return O;
    }

    private static float BilinearInterpolation(float fracSrcPosX, float fracSrcPosY, float p00, float p01, float p10, float p11)
    {
        float v =
            p00 * (1-fracSrcPosX) * (1-fracSrcPosY) +
            p01 * (1-fracSrcPosX) *    fracSrcPosY  +
            p10 *    fracSrcPosX  * (1-fracSrcPosY) +
            p11 *    fracSrcPosX  *    fracSrcPosY;
        return v;
    }

    /// <inheritdoc/>
    public virtual Tensor Upsample3D(Tensor X, int[] scale, bool trilinear)
    {
        Assert.IsTrue(X.shape.IsNDHWC());
        Assert.AreEqual(scale.Length, 3);
        float scaleX = (float)scale[0];
        float scaleY = (float)scale[1];
        float scaleD = (float)scale[2];

        var O = NewTensor(X.dataType, new TensorShape(1, 1,X.batch, 1, X.depth*scale[2], X.height*scale[1], X.width*scale[0], X.channels));

        for (int b = 0; b < O.batch; ++b)
            for (int d = 0; d < O.depth; ++d)
                for (int y = 0; y < O.height; ++y)
                    for (int x = 0; x < O.width; ++x)
                        for (int c = 0; c < O.channels; ++c)
                        {
                            if (trilinear)
                            {
                                float srcPosD = (d + 0.5f) / scaleD - 0.5f;
                                float srcPosX = (x + 0.5f) / scaleX - 0.5f;
                                float srcPosY = (y + 0.5f) / scaleY - 0.5f;
                                float floorSrcPosD = Mathf.Floor(srcPosD);
                                float floorSrcPosX = Mathf.Floor(srcPosX);
                                float floorSrcPosY = Mathf.Floor(srcPosY);
                                float fracSrcPosD = srcPosD - floorSrcPosD;
                                float fracSrcPosX = srcPosX - floorSrcPosX;
                                float fracSrcPosY = srcPosY - floorSrcPosY;

                                //from https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/interpolation/trilinear-interpolation
                                float p000 = X[X.IndexWithClamp(b, (int)floorSrcPosD + 0, (int)floorSrcPosY + 0, (int)floorSrcPosX + 0, c)];
                                float p100 = X[X.IndexWithClamp(b, (int)floorSrcPosD + 1, (int)floorSrcPosY + 0, (int)floorSrcPosX + 0, c)];
                                float p010 = X[X.IndexWithClamp(b, (int)floorSrcPosD + 0, (int)floorSrcPosY + 1, (int)floorSrcPosX + 0, c)];
                                float p110 = X[X.IndexWithClamp(b, (int)floorSrcPosD + 1, (int)floorSrcPosY + 1, (int)floorSrcPosX + 0, c)];
                                float p001 = X[X.IndexWithClamp(b, (int)floorSrcPosD + 0, (int)floorSrcPosY + 0, (int)floorSrcPosX + 1, c)];
                                float p101 = X[X.IndexWithClamp(b, (int)floorSrcPosD + 1, (int)floorSrcPosY + 0, (int)floorSrcPosX + 1, c)];
                                float p011 = X[X.IndexWithClamp(b, (int)floorSrcPosD + 0, (int)floorSrcPosY + 1, (int)floorSrcPosX + 1, c)];
                                float p111 = X[X.IndexWithClamp(b, (int)floorSrcPosD + 1, (int)floorSrcPosY + 1, (int)floorSrcPosX + 1, c)];
                                float e = BilinearInterpolation(fracSrcPosX, fracSrcPosY, p000, p100, p010, p110);
                                float f = BilinearInterpolation(fracSrcPosX, fracSrcPosY, p001, p101, p011, p111);
                                float v = e * ( 1 - fracSrcPosD) + f * fracSrcPosD;
                                O[b, d, y, x, c] = v;
                            }
                            else
                            {
                                int od = d / scale[2];
                                int oy = y / scale[1];
                                int ox = x / scale[0];
                                O[b, d, y, x, c] = X[b, od, oy, ox, c];
                            }
                        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Upsample2D(Tensor X, int[] scale, bool bilinear)
    {
        Assert.AreEqual(scale.Length, 2);
        float scaleX = (float)scale[0];
        float scaleY = (float)scale[1];

        Assert.IsTrue(X.shape.Is4D());
        var O = NewTensor(X.dataType, X.batch, X.height*scale[1], X.width*scale[0], X.channels);

        for (int b = 0; b < O.batch; ++b)
            for (int y = 0; y < O.height; ++y)
                for (int x = 0; x < O.width; ++x)
                    for (int c = 0; c < O.channels; ++c)
                    {
                        if (bilinear)
                        {
                            float srcPosX = (x + 0.5f) / scaleX - 0.5f;
                            float srcPosY = (y + 0.5f) / scaleY - 0.5f;
                            float floorSrcPosX = Mathf.Floor(srcPosX);
                            float floorSrcPosY = Mathf.Floor(srcPosY);
                            float fracSrcPosX = srcPosX - floorSrcPosX;
                            float fracSrcPosY = srcPosY - floorSrcPosY;

                            float p00 = X[X.IndexWithClamp(b, (int)floorSrcPosY + 0, (int)floorSrcPosX + 0, c)];
                            float p01 = X[X.IndexWithClamp(b, (int)floorSrcPosY + 1, (int)floorSrcPosX + 0, c)];
                            float p10 = X[X.IndexWithClamp(b, (int)floorSrcPosY + 0, (int)floorSrcPosX + 1, c)];
                            float p11 = X[X.IndexWithClamp(b, (int)floorSrcPosY + 1, (int)floorSrcPosX + 1, c)];
                            O[b, y, x, c] = BilinearInterpolation(fracSrcPosX, fracSrcPosY, p00, p01, p10, p11);
                        }
                        else
                        {
                            int oy = y / scale[1];
                            int ox = x / scale[0];
                            O[b, y, x, c] = X[b, oy, ox, c];
                        }

                    }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Resample2D(Tensor X, int[] size, bool bilinear)
    {
        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(size.Length, 2);
        var O = NewTensor(X.dataType, X.batch, size[1], size[0], X.channels);

        float scaleX = O.width / (float) X.width;
        float scaleY = O.height / (float) X.height;

        for (int b = 0; b < O.batch; ++b)
            for (int y = 0; y < O.height; ++y)
                for (int x = 0; x < O.width; ++x)
                    for (int c = 0; c < O.channels; ++c)
                    {
                        if (bilinear)
                        {
                            float srcPosX = (x + 0.5f) / scaleX - 0.5f;
                            float srcPosY = (y + 0.5f) / scaleY - 0.5f;
                            float floorSrcPosX = Mathf.Floor(srcPosX);
                            float floorSrcPosY = Mathf.Floor(srcPosY);
                            float fracSrcPosX = srcPosX - floorSrcPosX;
                            float fracSrcPosY = srcPosY - floorSrcPosY;

                            float p00 = X[X.IndexWithClamp(b, (int)floorSrcPosY + 0, (int)floorSrcPosX + 0, c)];
                            float p01 = X[X.IndexWithClamp(b, (int)floorSrcPosY + 1, (int)floorSrcPosX + 0, c)];
                            float p10 = X[X.IndexWithClamp(b, (int)floorSrcPosY + 0, (int)floorSrcPosX + 1, c)];
                            float p11 = X[X.IndexWithClamp(b, (int)floorSrcPosY + 1, (int)floorSrcPosX + 1, c)];
                            float v =
                                p00 * (1 - fracSrcPosX) * (1 - fracSrcPosY) +
                                p01 * (1 - fracSrcPosX) * fracSrcPosY +
                                p10 * fracSrcPosX * (1 - fracSrcPosY) +
                                p11 * fracSrcPosX * fracSrcPosY;
                            O[b, y, x, c] = v;
                        }
                        else
                        {
                            var srcY = Mathf.FloorToInt(y / scaleY);
                            var srcX = Mathf.FloorToInt(x / scaleX);
                            O[b, y, x, c] = X[X.IndexWithClamp(b, srcY, srcX, c)];
                        }
                    }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor DepthToSpace(Tensor X, int[] blocksize, Layer.DepthToSpaceMode mode)
    {
        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(blocksize.Length, 2);
        int bsX = blocksize[0];
        int bsY = blocksize[1];

        Assert.AreEqual(X.channels % (bsX * bsY), 0);

        var O = NewTensor(X.dataType, X.batch, X.height * bsY, X.width * bsX, X.channels / (bsX * bsY));

        for (int b = 0; b < O.batch; ++b)
            for (int y = 0; y < O.height; ++y)
                for (int x = 0; x < O.width; ++x)
                    for (int c = 0; c < O.channels; ++c)
                    {
                        int iy = y / bsY;
                        int by = y % bsY;
                        int ix = x / bsX;
                        int bx = x % bsX;
                        switch (mode)
                        {
                            case Layer.DepthToSpaceMode.CRD:
                                O[b, y, x, c] = X[b, iy, ix, (c * bsX * bsY) + (by * bsX) + bx];
                                break;
                            case Layer.DepthToSpaceMode.DCR:
                                O[b, y, x, c] = X[b, iy, ix, (by * bsX * O.channels) + (bx * O.channels) + c];
                                break;
                        }
                    }

        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor SpaceToDepth(Tensor X, int[] blocksize)
    {
        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(blocksize.Length, 2);
        int bsX = blocksize[0];
        int bsY = blocksize[1];

        Assert.AreEqual(X.height % bsY, 0);
        Assert.AreEqual(X.width % bsX, 0);

        var O = NewTensor(X.dataType, X.batch, X.height / bsY, X.width / bsX, X.channels * (bsX * bsY));

        for (int b = 0; b < O.batch; ++b)
        for (int y = 0; y < O.height; ++y)
        for (int x = 0; x < O.width; ++x)
        for (int c = 0; c < O.channels; ++c)
        {
            int ic = c % X.channels;
            int bx = c / X.channels % bsX;
            int by = c / X.channels / bsX;
            int ix = x * bsX + bx;
            int iy = y * bsY + by;

            O[b, y, x, c] = X[b, iy, ix, ic];
        }

        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor MaxPool2D(Tensor X, int[] pool, int[] stride, int[] pad)
    {
        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(pool.Length, 2);
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);

        var O = NewTensor(X.dataType, X.shape.ApplyPool(pool, stride, pad));

        for (int b = 0; b < O.batch; ++b)
            for (int y = 0; y < O.height; ++y)
                for (int x = 0; x < O.width; ++x)
                    for (int c = 0; c < O.channels; ++c)
                    {
                        float maxVal = float.MinValue;
                        for (int dy = 0; dy < pool[1]; ++dy)
                            for (int dx = 0; dx < pool[0]; ++dx)
                            {
                                int oy = y * stride[1] + dy - pad[1];
                                int ox = x * stride[0] + dx - pad[0];

                                if (oy < 0) continue;
                                if (oy >= X.height) continue;
                                if (ox < 0) continue;
                                if (ox >= X.width) continue;

                                float v = X[b, oy, ox, c
                                    //b  * X.height * X.width * X.channels +
                                    //oy * X.width * X.channels +
                                    //ox * X.channels +
                                    //c +
                                    //X.offset
                                ];
                                maxVal = Mathf.Max(v, maxVal);
                            }

                        O[b, y, x, c
                            //b * O.height * O.width * O.channels +
                            //y * O.width * O.channels +
                            //x * O.channels +
                            //c +
                            //O.offset
                        ] = maxVal;
                    }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor AvgPool2D(Tensor X, int[] pool, int[] stride, int[] pad)
    {
        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(pool.Length, 2);
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);

        var O = NewTensor(X.dataType, X.shape.ApplyPool(pool, stride, pad));

        for (int b = 0; b < O.batch; ++b)
            for (int y = 0; y < O.height; ++y)
                for (int x = 0; x < O.width; ++x)
                    for (int c = 0; c < O.channels; ++c)
                    {
                        float accum = 0.0f;
                        float counter = 0.0f;
                        for (int dy = 0; dy < pool[1]; ++dy)
                            for (int dx = 0; dx < pool[0]; ++dx)
                            {
                                int oy = y * stride[1] + dy - pad[1];
                                int ox = x * stride[0] + dx - pad[0];

                                if (oy < 0) continue;
                                if (oy >= X.height) continue;
                                if (ox < 0) continue;
                                if (ox >= X.width) continue;

                                float v = X[b, oy, ox, c
                                    //b  * X.height * X.width * X.channels +
                                    //oy * X.width * X.channels +
                                    //ox * X.channels +
                                    //c +
                                    //X.offset
                                ];
                                accum += v;
                                ++counter;
                            }

                        O[b, y, x, c
                            //b * O.height * O.width * O.channels +
                            //y * O.width * O.channels +
                            //x * O.channels +
                            //c +
                            //O.offset
                        ] = accum / counter;
                    }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor GlobalMaxPool2D(Tensor X)
    {
        Assert.IsTrue(X.shape.Is4D());
        var O = NewTensor(X.dataType, X.batch, 1, 1, X.channels);

        for (int b = 0; b < X.batch; ++b)
            for (int c = 0; c < X.channels; ++c)
            {
                float maxVal = float.MinValue;
                for (int y = 0; y < X.height; ++y)
                    for (int x = 0; x < X.width; ++x)
                    {
                        float v = X[b, y, x, c
                            //b * X.height * X.width * X.channels +
                            //y * X.width * X.channels +
                            //x * X.channels +
                            //c +
                            //X.offset
                        ];
                        maxVal = Mathf.Max(v, maxVal);
                    }

                O[b, 0, 0, c
                    //b * O.channels +
                    //c +
                    //O.offset
                ] = maxVal;
            }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor GlobalAvgPool2D(Tensor X)
    {
        var O = NewTensor(X.dataType, X.batch, 1, 1, X.channels);

        for (int b = 0; b < X.batch; ++b)
            for (int c = 0; c < X.channels; ++c)
            {
                float accum = 0.0f;
                for (int y = 0; y < X.height; ++y)
                    for (int x = 0; x < X.width; ++x)
                    {
                        float v = X[b, y, x, c
                            //b * X.height * X.width * X.channels +
                            //y * X.width * X.channels +
                            //x * X.channels +
                            //c +
                            //X.offset
                        ];
                        accum += v;
                    }

                O[b, 0, 0, c
                    //b * O.channels +
                    //c +
                    //O.offset
                ] = accum / (X.width * X.height);
            }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor GlobalAvgVariancePool2D(Tensor X)
    {
        Assert.IsTrue(X.shape.Is4D());
        var O = NewTensor(X.dataType, X.batch, 2, 1, X.channels);

        for (int b = 0; b < X.batch; ++b)
            for (int c = 0; c < X.channels; ++c)
            {
                float mean = 0.0f;
                float mean2 = 0.0f;
                for (int y = 0; y < X.height; ++y)
                    for (int x = 0; x < X.width; ++x)
                    {
                        float v = X[b, y, x, c
                            //b * X.height * X.width * X.channels +
                            //y * X.width * X.channels +
                            //x * X.channels +
                            //c +
                            //X.offset
                        ];
                        mean  += v;
                        mean2 += v*v;
                    }

                mean  /= (X.width * X.height);
                mean2 /= (X.width * X.height);

                O[b, 0, 0, c
                    //b * O.channels +
                    //c +
                    //O.offset
                ] = mean;

                O[b, 1, 0, c
                    //b * O.channels +
                    //c +
                    //O.offset
                ] = mean2 - mean * mean;
            }
        return O;
    }

    private Tensor ApplyPadding(Tensor X, int[] pad, Func<Tensor, int, int, int, int, int, float> paddingOp)
    {
        Assert.IsTrue(X.shape.IsNDHWC());
        Assert.IsTrue(pad.Length == 6 || pad.Length == 8);

        var O = NewTensor(X.dataType, X.shape.ApplyBorder(pad));

        int prePadW  = pad[0];
        int prePadH  = pad[1];
        int prePadD  = pad.Length == 6 ? 0      : pad[2];
        int prePadC  = pad.Length == 6 ? pad[2] : pad[3];

        int postPadW = pad.Length == 6 ? pad[3] : pad[4];
        int postPadH = pad.Length == 6 ? pad[4] : pad[5];
        int postPadD = pad.Length == 6 ? 0      : pad[6];
        int postPadC = pad.Length == 6 ? pad[5] : pad[7];

        // NOTE: negative "pad" variable will crop X tensor
        int croppedWidth  = X.width  - Math.Max(0, -postPadW);
        int croppedHeight = X.height - Math.Max(0, -postPadH);
        int croppedDepth  = X.depth  - Math.Max(0, -postPadD);
        int croppedChannels = X.channels - Math.Max(0, -postPadC);

        for (int b = 0; b < O.batch; ++b)
            for (int d = 0; d < O.depth; ++d)
                for (int h = 0; h < O.height; ++h)
                    for (int w = 0; w < O.width; ++w)
                        for (int c = 0; c < O.channels; ++c)
                        {
                            int readW = w - prePadW;
                            int readH = h - prePadH;
                            int readD = d - prePadD;
                            int readC = c - prePadC;

                            if (readW < 0 || readW >= croppedWidth ||
                                readH < 0 || readH >= croppedHeight ||
                                readD < 0 || readD >= croppedDepth ||
                                readC < 0 || readC >= croppedChannels)
                            {
                                O[b, d, h, w, c] = paddingOp(X, b, readD, readH, readW, readC);
                            }
                            else
                            {
                                O[b, d, h, w, c] = X[b, readD, readH, readW, readC];
                            }
                        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Border2D(Tensor X, int[] pad, float value)
    {
        Func<Tensor, int, int, int, int, int, float> padOp = (tensor, b, d, h, w, c) => value;
        return ApplyPadding(X, pad, padOp);
    }

    /// <inheritdoc/>
    public virtual Tensor Border3D(Tensor X, int[] pad, float value)
    {
        Func<Tensor, int, int, int, int, int, float> padOp = (tensor, b, d, h, w, c) => value;
        return ApplyPadding(X, pad, padOp);
    }

    private static void ClampHWCToTensorShape(TensorShape shape, ref int height, ref int width, ref int channels)
    {
        width = Math.Max(width, 0);
        height = Math.Max(height, 0);
        channels = Math.Max(channels, 0);
        width = Math.Min(width, shape.width - 1);
        height = Math.Min(height, shape.height - 1);
        channels = Math.Min(channels, shape.channels - 1);
    }

    /// <inheritdoc/>
    public virtual Tensor Pad2DReflect(Tensor X, int[] pad)
    {
        float GetReflectPadding(Tensor tensorX, int b, int readD, int readY, int readX, int readC)
        {
            //TODO when implementing Pad3DReflect change to function and support depth
            int lastXIndex = tensorX.shape.width - 1;
            int lastYIndex = tensorX.shape.height - 1;
            int lastCIndex = tensorX.shape.channels - 1;

            if (readX < 0)
                readX = -readX;
            else if (readX > lastXIndex)
                readX = lastXIndex - (readX - lastXIndex);

            if (readY < 0)
                readY = -readY;
            else if (readY > lastYIndex)
                readY = lastYIndex - (readY - lastYIndex);

            if (readC < 0)
                readC = -readC;
            else if (readC > lastCIndex)
                readC = lastCIndex - (readC - lastCIndex);

            ClampHWCToTensorShape(tensorX.shape, ref readY, ref readX, ref readC);
            return tensorX[b, readY, readX, readC];
        }

        return ApplyPadding(X, pad, GetReflectPadding);
    }

    /// <inheritdoc/>
    public virtual Tensor Pad2DSymmetric(Tensor X, int[] pad)
    {
        float GetSymmetricPadding(Tensor tensorX, int b, int readD, int readY, int readX, int readC)
        {
            //TODO when implementing Pad3DSymmetric change to function and support depth
            int lastXIndex = tensorX.shape.width - 1;
            int lastYIndex = tensorX.shape.height - 1;
            int lastCIndex = tensorX.shape.channels - 1;

            if (readX < 0)
                readX = -readX - 1;
            else if (readX > lastXIndex)
                readX = lastXIndex - (readX - lastXIndex) + 1;

            if (readY < 0)
                readY = -readY - 1;
            else if (readY > lastYIndex)
                readY = lastYIndex - (readY - lastYIndex) + 1;

            if (readC < 0)
                readC = -readC - 1;
            else if (readC > lastCIndex)
                readC = lastCIndex - (readC - lastCIndex) + 1;

            ClampHWCToTensorShape(tensorX.shape, ref readY, ref readX, ref readC);
            return tensorX[b, readY, readX, readC];
        }

        return ApplyPadding(X, pad, GetSymmetricPadding);
    }

    /// <inheritdoc/>
    public virtual Tensor Pad2DEdge(Tensor X, int[] pad)
    {
        float GetEdgePadding(Tensor tensorX, int b, int readD, int readY, int readX, int readC)
        {
            //TODO when implementing Pad3DEdge change to function and support depth
            ClampHWCToTensorShape(tensorX.shape, ref readY, ref readX, ref readC);
            return tensorX[b, readY, readX, readC];
        }

        return ApplyPadding(X, pad, GetEdgePadding);
    }

    /// <inheritdoc/>
    public virtual Tensor ScaleBias(Tensor X, Tensor S, Tensor B)
    {
        Assert.AreEqual(X.channels, B.channels); Assert.AreEqual(X.channels, S.channels);
        Assert.AreEqual(B.length, B.channels); Assert.AreEqual(S.length, S.channels);

        var O = NewTensorLike(X);

        for (var it = new TensorIterator(O); it.IsValid(); it.Next())
        {
            float beta = B[0, 0, 0, it.d7];//.array[c + B.offset];
            float gamma = S[0, 0, 0, it.d7];//S.array[c + S.offset];

            //var i = X.IndexWithOffset(b, y, x, c);
            float v = X[it.index];//.array[i];
            O[it.index] = v * gamma + beta;
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor LRN(Tensor X, float alpha, float beta, float bias, int size)
    {
        // https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
        // However divide the sum by size to follow onnx and pytorch implementation
        // ONNX https://github.com/onnx/onnx/blob/master/docs/Operators.md#LRN
        // PYTORCH https://github.com/pytorch/pytorch/blob/1465970a343e61f2f2b104859ca7f5d7e03f5d02/torch/nn/functional.py#L2069
        // Tensorflow don't and follow the paper to the letter https://github.com/tensorflow/tensorflow/blob/e6faa845c51bb69465146d93646947fd2ba53efa/tensorflow/python/kernel_tests/lrn_op_test.py#L53
        // However they bake the division to alpha when exporting to ONNX https://github.com/onnx/tensorflow-onnx/blob/7c37ccb97e0fd478ce093910c4a1411b18e44fd7/tf2onnx/onnx_opset/math.py
        var O = NewTensorLike(X);
        float sizef = size;

        for (var it = new TensorIterator(O); it.IsValid(); it.Next())
        {
            int c = it.d7;
            float regionCenter = (sizef - 1.0f) / 2.0f;
            int regionStart = Math.Max(0, c - (int)Mathf.Floor(regionCenter));
            int regionEnd = Math.Min(X.channels, c + (int)Mathf.Ceil(regionCenter)+1);
            float sumOfSquared = 0.0f;
            for (int ci = regionStart; ci < regionEnd; ++ci)
            {
                float regionValue = X[it.d0, it.d1, it.d2, it.d3, it.d4, it.d5, it.d6 ,ci];
                sumOfSquared += regionValue * regionValue;
            }

            O[it.index] = X[it.index] / Mathf.Pow(bias + alpha * sumOfSquared / sizef, beta);
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Normalization(Tensor X, Tensor S, Tensor B, int pool, int axis, float epsilon, Layer.FusedActivation fusedActivation)
    {
        if (!X.shape.Is4D())
            throw new NotImplementedException();

        Assert.AreEqual(X.channels, B.channels); Assert.AreEqual(X.channels, S.channels);

        if (axis != TensorShape.C && axis != -1)
            throw new NotImplementedException();

        // Special cases of Normalization:
        // 1) Instance Normalization, if pool == 1
        // 2) Batch Normalization, if pool <= 0
        if (pool <= 0)
            pool = X.batch;

        var O = NewTensorLike(X);

        var channels = X.channels;
        var width = X.width;
        var height = X.height;

        for (int subBatch = 0; subBatch < O.batch; subBatch += pool)
            for (int c = 0; c < channels; ++c)
            {
                int bBegin = subBatch;
                int bEnd = Math.Min(subBatch + pool, O.batch);

                float gamma = S[0, 0, 0, c];//.array[c + S.offset];
                float beta = B[0, 0, 0, c];//B.array[c + B.offset];

                // calc mean
                double sum = 0;
                for (int b = bBegin; b < bEnd; ++b)
                    for (int y = 0; y < height; ++y)
                        for (int x = 0; x < width; ++x)
                        {
                            double v = X[b, y, x, c];
                            sum += v;
                        }
                double mean = sum / (width * height);

                // calc variance
                sum = 0;
                for (int b = bBegin; b < bEnd; ++b)
                    for (int y = 0; y < height; ++y)
                        for (int x = 0; x < width; ++x)
                        {
                            double v = X[b, y, x, c];
                            sum += (v - mean) * (v - mean);
                        }
                double var = sum / (width * height);

                // apply normalization
                for (int b = bBegin; b < bEnd; ++b)
                    for (int y = 0; y < height; ++y)
                        for (int x = 0; x < width; ++x)
                        {
                            float v = X[b, y, x, c];
                            v = (float)(gamma * (v - mean) / Math.Sqrt(var + epsilon) + beta);
                            O[b, y, x, c] = ApplyFusedActivation(v, fusedActivation);
                        }
            }
        return O;
    }

    /// <summary>
    /// Bernoulli distribution
    /// </summary>
    /// <param name="p">p</param>
    /// <returns>random value</returns>
    protected float Bernoulli(float p)
    {
        return (Random.value <= p) ? 1f: 0f;
    }

    /// <summary>
    /// Gaussian distribution
    /// </summary>
    /// <param name="mean">mean</param>
    /// <param name="stdDev">standard deviation</param>
    /// <returns>random value</returns>
    protected float Gaussian(float mean, float stdDev)
    {
        float u, v, s;
        do {
            u = Random.value * 2 - 1;
            v = Random.value * 2 - 1;
            s = u * u + v * v;
        } while (s >= 1 || s == 0);
        float mul = Mathf.Sqrt(-2.0f * Mathf.Log(s) / s);
        return mean + stdDev * u * mul;
    }

    internal class Seed : IDisposable
    {
        Random.State[] m_SeedStorage;
        Random.State m_EngineSeed;
        public Seed(ref Random.State[] storage, int initialSeed)
        {
            m_EngineSeed = Random.state;
            if (storage == null)
            {
                storage = new Random.State[1];
                Random.InitState(initialSeed);
                storage[0] = Random.state;
            }
            else
                Random.state = storage[0];
            m_SeedStorage = storage;
        }

        public virtual void Dispose()
        {
            m_SeedStorage[0] = Random.state;
            Random.state = m_EngineSeed;
        }
    }

    internal Random.State[] m_DropoutSeed;
    /// <inheritdoc/>
    public virtual Tensor Dropout(Tensor X, float alpha)
    {
        Assert.IsTrue(alpha >= 0f && alpha <= 1f);
        var O = NewTensorLike(X);

        // Based on PyTorch Dropout implementation
        // See: https://github.com/pytorch/pytorch/blob/master/torch/nn/_functions/dropout.py

        using (var seedOverride = new Seed(ref m_DropoutSeed, 1337))
        {
            var end = X.length;
            for (int i = 0; i < end; ++i)
            {
                float v = X[i];
                v *= Bernoulli(1f - alpha) / (1f - alpha);
                O[i] = v;
            }
        }
        return O;
    }

    private Random.State[] m_RandomNormalSeed;
    /// <inheritdoc/>
    public virtual Tensor RandomNormal(TensorShape s, float mean, float scale, int seed)
    {
        var O = NewTensor(DataType.Float, s);
        //TODO fp16: RandomNormal should be able to select output type
        //see dtype here https://github.com/onnx/onnx/blob/master/docs/Operators.md#RandomNormal

        using (var seedOverride = new Seed(ref m_RandomNormalSeed, seed))
        {
            var end = O.length;
            for (int i = 0; i < end; ++i)
                O[i] = Gaussian(mean, scale);
        }

        return O;
    }

    private Random.State[] m_RandomUniformSeed;
    /// <inheritdoc/>
    public virtual Tensor RandomUniform(TensorShape s, float mean, float scale, int seed)
    {
        var O = NewTensor(DataType.Float, s);
        //TODO fp16: RandomNormal should be able to select output type
        //see dtype here https://github.com/onnx/onnx/blob/master/docs/Operators.md#RandomUniform

        using (var seedOverride = new Seed(ref m_RandomUniformSeed, seed))
        {
            var end = O.length;
            for (int i = 0; i < end; ++i)
                O[i] = mean + scale * Random.value;
        }

        return O;
    }

    private Random.State[] m_MultinomialSeed;
    /// <inheritdoc/>
    public virtual Tensor Multinomial(Tensor X, int count, int seed)
    {
        if (X.shape.sequenceLength != 1 || X.shape.numberOfDirections != 1)
            throw new NotImplementedException();

        var O = NewTensor(X.dataType, X.flatHeight, count);

        // Tensorflow Multinomial for reference
        // See: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/multinomial_op.cc

        using (var seedOverride = new Seed(ref m_MultinomialSeed, seed))
        {
            for (int n = 0; n < X.flatHeight; ++n)
            {
                var maxLogP = Mathf.NegativeInfinity;
                for (int i = 0; i < X.flatWidth; ++i)
                    maxLogP = Mathf.Max(X[n, i], maxLogP);

                float sumOfProbabilities = 0f;
                for (int i = 0; i < X.flatWidth; ++i)
                    sumOfProbabilities += Mathf.Exp(X[n, i] - maxLogP); // NOTE: X contains log-probabilities

                for (int sample = 0; sample < count; ++sample)
                {
                    float p = Random.value * sumOfProbabilities;

                    int i = 0;
                    float cumulativeP = 0f;
                    while (i < X.flatWidth && p > cumulativeP)
                    {
                        cumulativeP += Mathf.Exp(X[n, i] - maxLogP);
                        i++;
                    }
                    Assert.IsTrue(i > 0);
                    O[n, sample] = (float)(i - 1);
                }
            }
        }

        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor OneHot(Tensor X, int depth, float onValue, float offValue, int inputRank=-1)
    {
        if (inputRank == -1)
            inputRank = X.dimensions;

        if (inputRank >= 4)
            throw new NotImplementedException();

        Tensor O;
        if (inputRank == 1)
            O = NewOutputTensor(X.dataType, new TensorShape(X.flatHeight, depth));
        else if (inputRank == 2)
            O = NewOutputTensor(X.dataType, new TensorShape(X.flatHeight, 1, depth, X.channels));
        else
            O = NewOutputTensor(X.dataType, new TensorShape(X.batch, X.width, depth, X.channels));

        // rank1: X = n,_,_,_
        // rank2: X = n,_,_,c
        // rank3: X = n,_,w,c

        for (int n = 0; n < X.batch; ++n)
        {
            for (int j = 0; j < depth; ++j)
            {
                for (int k = 0; k < X.width; ++k)
                {
                    for (int i = 0; i < X.channels; ++i)
                    {
                        if (inputRank == 1)
                        {
                            int index = (int)X[n];
                            float v = (j == index) ? onValue: offValue;
                            O[n, j] = v;
                        }
                        else if (inputRank == 2)
                        {
                            int index = (int)X[n, i];
                            float v = (j == index) ? onValue: offValue;
                            O[n, 0, j, i] = v;
                        }
                        else
                        {
                            int index = (int)X[n, 0, k, i];
                            float v = (j == index) ? onValue: offValue;
                            O[n, k, j, i] = v;
                        }
                    }
                }
            }
        }
        return O;
    }

    private float NearestNeighbourBilinearInterpolation(Tensor X, int n, float y, float x, int c, bool snapToBorder = false)
    {
        if (snapToBorder)
        {
            y = Mathf.Clamp(y, 0, X.height - 1);
            x = Mathf.Clamp(x, 0, X.width - 1);
        }

        int y_low = (int)Mathf.Floor(y);
        int x_low = (int)Mathf.Floor(x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        float wy_h = y - y_low;
        float wx_h = x - x_low;
        float wy_l = 1.0f - wy_h;
        float wx_l = 1.0f - wx_h;

        float v = 0.0f;
        if(y_low >= 0 && y_low < X.height && x_low >= 0 && x_low < X.width)
            v += wx_l * wy_l * X[n, y_low, x_low, c];
        if (y_low >= 0 && y_low < X.height && x_high >= 0 && x_high < X.width)
            v += wx_h * wy_l * X[n, y_low, x_high, c];
        if (y_high >= 0 && y_high < X.height && x_low >= 0 && x_low < X.width)
            v += wx_l * wy_h * X[n, y_high, x_low, c];
        if (y_high >= 0 && y_high < X.height && x_high >= 0 && x_high < X.width)
            v += wx_h * wy_h * X[n, y_high, x_high, c];

        return v;
    }

    /// <inheritdoc/>

    public virtual Tensor RoiAlign(Tensor X, Tensor Rois, Tensor Indices, int outputHeight, int outputWidth, int samplingRatio, float spatialScale)
    {
        // https://arxiv.org/abs/1703.06870
        // https://github.com/pytorch/vision/blob/cdb6fba52f461b276d9b4d0a817b62e69344021c/test/test_ops.py
        Assert.IsTrue(X.shape.Is4D());
        Assert.AreEqual(Rois.flatHeight, Indices.batch);
        Assert.AreEqual(Rois.flatWidth, 4);

        Tensor O = NewTensor(X.dataType, Rois.flatHeight, outputHeight, outputWidth, X.channels);

        bool aligned = false;
        float offset = aligned ? 0.5f : 0.0f;

        for (int n = 0; n < Rois.flatHeight; n++)
        {
            float j_begin = Rois[n, 0] * spatialScale - offset;
            float i_begin = Rois[n, 1] * spatialScale - offset;
            float j_end = Rois[n, 2] * spatialScale - offset;
            float i_end = Rois[n, 3] * spatialScale - offset;

            float roi_h = i_end - i_begin;
            float roi_w = j_end - j_begin;
            float bin_h = roi_h / ((float)outputHeight);
            float bin_w = roi_w / ((float)outputWidth);

            int batchIdx = (int)Indices[n];

            for (int i = 0; i < outputHeight; i++)
                for (int j = 0; j < outputWidth; j++)
                {
                    float start_h = i_begin + i * bin_h;
                    float grid_h = samplingRatio > 0 ? samplingRatio : Mathf.Ceil(bin_h);
                    float start_w = j_begin + j * bin_w;
                    float grid_w = samplingRatio > 0 ? samplingRatio : Mathf.Ceil(bin_w);

                    for (int c = 0; c < X.channels; c++)
                    {
                        float v = 0.0f;
                        for (int iy = 0; iy < (int)grid_h; iy++)
                            for (int ix = 0; ix < (int)grid_w; ix++)
                            {
                                float y = start_h + (iy + 0.5f) * bin_h / grid_h;
                                float x = start_w + (ix + 0.5f) * bin_w / grid_w;

                                if(x >= X.width || x < 0 || y >= X.height || y < 0)
                                    v += 0.0f;
                                else
                                    v += NearestNeighbourBilinearInterpolation(X, batchIdx, y, x, c, true);
                            }

                            v /= grid_h * grid_w;


                        O[n, i, j, c] = v;
                    }
                }
        }

        return O;
    }


    // TODO: Revisit flattened approach (see previous attempt in source history), which had two of the four axis cases working
    //    but couldn't get the strides just right for the outer loop, so opted for this straightforward approach
    // NOTE: If `sorted` is false, then the output is undefined, so it's only necessary to implement something explicitly
    //    if there is a benefit in terms of performance
    /// <inheritdoc/>
    public virtual Tensor TopKIndices(Tensor X, int k, int axis, bool largest, bool sorted)
    {
        if (!X.shape.Is4D())
            throw new NotImplementedException();

        TensorShape xShape = X.shape;
        int[] inputShape = xShape.ToArray();

        int[] outputShape = xShape.ToArray();
        outputShape[axis] = Mathf.Min(k, outputShape[axis]); // Can't have more elements then there are in the original input tensor
        var O = NewTensor(X.dataType, new TensorShape(outputShape));
        TensorShape oShape = O.shape;

        // Determine the iteration order, so that the selected axis is the final loop; Everything else is shifted accordingly
        int[] iterators = new int[4];         // initialized to all 0s
        int[] iteratorAxes = new int[4];      // initialized below
        int[] iteratorAxes8D = new int[4];    // initialized below

        // Since we are assuming rank 4 convert axis to appropriate index (from rank 8)
        axis = TensorExtensions.Convert8DAxisTo4D(axis);
        int axisIndex = axis;
        for (int i = iteratorAxes.Length - 1; i >= 0; i--)
        {
            iteratorAxes[i] = axisIndex % iteratorAxes.Length;
            iteratorAxes8D[i] = TensorExtensions.Convert4DTo8DAxis(iteratorAxes[i]);
            axisIndex++;
        }

        var topK = new SortedList<float, int>();
        int[] coords = new int[4];
        for (iterators[0] = 0; iterators[0] < inputShape[iteratorAxes8D[0]]; iterators[0]++)
        {
            for (iterators[1] = 0; iterators[1] < inputShape[iteratorAxes8D[1]]; iterators[1]++)
            {
                for (iterators[2] = 0; iterators[2] < inputShape[iteratorAxes8D[2]]; iterators[2]++)
                {
                    for (iterators[3] = 0; iterators[3] < inputShape[iteratorAxes8D[3]]; iterators[3]++)
                    {
                        coords[iteratorAxes[0]] = iterators[0];
                        coords[iteratorAxes[1]] = iterators[1];
                        coords[iteratorAxes[2]] = iterators[2];
                        coords[iteratorAxes[3]] = iterators[3];
                        int n = coords[0];
                        int h = coords[1];
                        int w = coords[2];
                        int c = coords[3];
                        int index = xShape.Index(n, h, w, c);
                        float value = X[index];
                        if (topK.TryGetValue(value, out int existingIndex))
                            index = Mathf.Min(index, existingIndex); // Per ONNX choose the lower index

                        topK[value] = index;
                    }

                    IEnumerable<KeyValuePair<float, int>> elements = largest ? topK.Reverse().Take(k) : topK.Take(k);

                    int e = 0;
                    foreach (KeyValuePair<float, int> element in elements)
                    {
                        int index = element.Value;
                        xShape.GetPositionsFromIndex(index, ref coords[0], ref coords[1], ref coords[2], ref coords[3]);
                        int n = coords[0];
                        int h = coords[1];
                        int w = coords[2];
                        int c = coords[3];
                        var outputCoords = new [] { n, h, w, c };
                        outputCoords[axis] = e;

                        int outputIndex = oShape.Index(outputCoords[0], outputCoords[1], outputCoords[2], outputCoords[3]);
                        O[outputIndex] = coords[axis];
                        e++;
                    }

                    topK.Clear();
                }
            }
        }

        return O;
    }

    /// <inheritdoc/>
    public Tensor NonZero(Tensor X)
    {
        //https://github.com/onnx/onnx/blob/master/docs/Operators.md#NonZero
        //https://numpy.org/doc/stable/reference/generated/numpy.nonzero.html
        //Return the indices of the elements that are non-zero.

        //The values are supposed to be return in row-major, C-style order. In order to match ONNX
        //result we need to iterate tensor as if it was channel first.
        List<int[]> nonZeroIndices = new List<int[]>();
        for (var d0 = 0; d0 < X.shape[0]; ++d0) //s
        for (var d1 = 0; d1 < X.shape[1]; ++d1) //r
        for (var d2 = 0; d2 < X.shape[2]; ++d2) //n
        for (var d7 = 0; d7 < X.shape[7]; ++d7) //c <--channel first
        for (var d3 = 0; d3 < X.shape[3]; ++d3) //t
        for (var d4 = 0; d4 < X.shape[4]; ++d4) //d
        for (var d5 = 0; d5 < X.shape[5]; ++d5) //h
        for (var d6 = 0; d6 < X.shape[6]; ++d6) //w
        {
            if (Math.Abs(X[d0,d1,d2,d3,d4,d5,d6,d7]) > Single.Epsilon)
            {
                nonZeroIndices.Add(new int[] {d0,d1,d2,d3,d4,d5,d6,d7});
            }
        }

        var O = NewTensor(X.dataType, new TensorShape(X.dimensions,nonZeroIndices.Count));
        for(int i = 0; i < nonZeroIndices.Count; ++i)
        {
            int destinationTensorDim = 0;
            for (int d = 0; d < TensorShape.MaxRank; ++d)
            {
                //TODO: This won't match ONNX output size for tensor with one or many dimension of size 1.
                //We need the notion of rank in Barracuda to handle this according to ONNX spec.
                if (X.shape[d] > 1)
                {
                    O[destinationTensorDim, i] = nonZeroIndices[i][d];
                    ++destinationTensorDim;
                }
            }
        }

        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor TopKValues(Tensor X, Tensor I, int axis)
    {
        if (!X.shape.Is4D())
            throw new NotImplementedException();

        TensorShape xShape = X.shape;
        TensorShape iShape = I.shape;
        int[] indicesShape = iShape.ToArray();

        var O = NewTensor(X.dataType, iShape);
        // Determine the iteration order, so that the selected axis is the final loop; Everything else is shifted accordingly
        int[] iterators = new int[4];         // initialized to all 0s
        int[] iteratorAxes = new int[4];      // initialized below
        int[] iteratorAxes8D = new int[4];    // initialized below

        // Since we are assuming rank 4 convert axis to appropriate index (from rank 8)
        axis = TensorExtensions.Convert8DAxisTo4D(axis);
        int axisIndex = axis;
        for (int i = iteratorAxes.Length - 1; i >= 0; i--)
        {
            iteratorAxes[i] = axisIndex % iteratorAxes.Length;
            iteratorAxes8D[i] = TensorExtensions.Convert4DTo8DAxis(iteratorAxes[i]);
            axisIndex++;
        }


        int[] coords = new int[4];
        for (iterators[0] = 0; iterators[0] < indicesShape[iteratorAxes8D[0]]; iterators[0]++)
        {
            for (iterators[1] = 0; iterators[1] < indicesShape[iteratorAxes8D[1]]; iterators[1]++)
            {
                for (iterators[2] = 0; iterators[2] < indicesShape[iteratorAxes8D[2]]; iterators[2]++)
                {
                    for (iterators[3] = 0; iterators[3] < indicesShape[iteratorAxes8D[3]]; iterators[3]++)
                    {
                        coords[iteratorAxes[0]] = iterators[0];
                        coords[iteratorAxes[1]] = iterators[1];
                        coords[iteratorAxes[2]] = iterators[2];
                        coords[iteratorAxes[3]] = iterators[3];
                        int n = coords[0];
                        int h = coords[1];
                        int w = coords[2];
                        int c = coords[3];
                        // Even though storage format is NHWC use NCHW indexing to match ONNX iteration
                        int index = iShape.Index(n, h, w, c);

                        // Get the computed index (axis-relative) value for this element
                        int topKAxisIndex = (int)I[index];
                        coords[iteratorAxes[3]] = topKAxisIndex; // Replace original coordinate lookup
                        n = coords[0];
                        h = coords[1];
                        w = coords[2];
                        c = coords[3];
                        int topKIndex = xShape.Index(n, h, w, c);

                        O[index] = X[topKIndex];
                    }
                }
            }
        }

        return O;
    }


    /// <inheritdoc/>
    public virtual Tensor Relu(Tensor X)
    {
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = Mathf.Max(v, 0.0f);
            O[i] = v;
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor PRelu(Tensor X, Tensor S)
    {
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            float slope = S[i % S.length];

            v = Mathf.Max(0.0f, v) + slope * Mathf.Min(0.0f, v);
            O[i] = v;
        }

        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Softmax(Tensor X, int axis)
    {
        TensorShape xShape = X.shape;
        axis = xShape.Axis(axis); // Adjust for negative axis values
        var O = NewTensor(X.dataType, xShape);
        Assert.AreEqual(O.flatWidth, X.flatWidth);

        int height = 1;
        int axis8D = axis;
        for (var i = 0; i < axis8D; i++)
        {
            height *= xShape[i];
        }

        int width = 1;
        for (var i = axis8D + 1; i < TensorShape.MaxRank; i++)
        {
            width *= xShape[i];
        }

        int reducedDim = xShape[axis8D];

        //e_x = np.exp(X - X.max(axis=1, keepdims=True))
        //X = e_x / e_x.sum(axis=1, keepdims=True)
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                float maxV = Mathf.NegativeInfinity;
                for (int r = 0; r < reducedDim; ++r)
                {
                    float v = X[y * width * reducedDim + r * width + x];

                    if (v > maxV)
                        maxV = v;
                }

                float sum = 0.0f;
                for (int r = 0; r < reducedDim; ++r)
                {
                    float v = X[y * width * reducedDim + r * width + x];
                    sum += Mathf.Exp(v - maxV);
                }

                for (int r = 0; r < reducedDim; ++r)
                {
                    float v = X[y * width * reducedDim + r * width + x];
                    v = Mathf.Exp(v - maxV) / sum;
                    O[y * width * reducedDim + r * width + x] = v;
                }
            }
        }

        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor LogSoftmax(Tensor X, int axis)
    {
        TensorShape xShape = X.shape;
        axis = xShape.Axis(axis); // Adjust for negative axis values
        var O = NewTensor(X.dataType, xShape);
        Assert.AreEqual(O.flatWidth, X.flatWidth);

        int height = 1;
        int axis8D = axis;
        for (var i = 0; i < axis8D; i++)
        {
            height *= xShape[i];
        }

        int width = 1;
        for (var i = axis8D + 1; i < TensorShape.MaxRank; i++)
        {
            width *= xShape[i];
        }

        int reducedDim = xShape[axis8D];

        //e_x = np.exp(X - X.max(axis=1, keepdims=True))
        //X = log(e_x / e_x.sum(axis=1, keepdims=True))
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                float maxV = Mathf.NegativeInfinity;
                for (int r = 0; r < reducedDim; ++r)
                {
                    float v = X[y * width * reducedDim + r * width + x];

                    if (v > maxV)
                        maxV = v;
                }

                float sum = 0.0f;
                for (int r = 0; r < reducedDim; ++r)
                {
                    float v = X[y * width * reducedDim + r * width + x];
                    sum += Mathf.Exp(v - maxV);
                }

                for (int r = 0; r < reducedDim; ++r)
                {
                    float v = X[y * width * reducedDim + r * width + x];
                    v = (v - maxV) - Mathf.Log(sum);
                    O[y * width * reducedDim + r * width + x] = v;
                }
            }
        }

        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Tanh(Tensor X)
    {
        // f(x) = tanh(x) = sinh(x) / cosh(x) = (exp(2*x) - 1) / (exp(2*x) + 1)
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            O[i] = MathfEx.Tanh(X[i]);
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Softplus(Tensor X)
    {
        // f(x) = ln(exp(x) + 1)
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = Mathf.Log(Mathf.Exp(v) + 1f);
            O[i] = v;
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Sigmoid(Tensor X)
    {
        // f(x) = 1 / (1 + exp(-x))
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = 1f / (1f + Mathf.Exp(-v));
            O[i] = v;
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor HardSigmoid(Tensor X, float alpha, float beta)
    {
        // https://pytorch.org/docs/stable/generated/torch.nn.Hardsigmoid.html
        // https://github.com/onnx/onnx/blob/master/docs/Operators.md#HardSigmoid
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = Mathf.Max(0.0f, Mathf.Min(1.0f, alpha*v + beta));
            O[i] = v;
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Relu6(Tensor X)
    {
        // f(x) = min(max(x, 0), 6)
        // "Convolutional Deep Belief Networks on CIFAR-10", A Krizhevsky, 2010
        // http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = Mathf.Min(Mathf.Max(0f, v), 6f);
            O[i] = v;
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Elu(Tensor X, float alpha)
    {
        // f(x) = alpha * (exp(x) - 1.) for x < 0, f(x) = x for x >= 0
        // "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)", DA Clevert, 2015
        // https://arxiv.org/abs/1511.07289
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            if (v <= 0)
                v = alpha * (Mathf.Exp(v) - 1f);
            O[i] = v;
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor LeakyRelu(Tensor X, float alpha)
    {
        // f(x) = alpha * x for x < 0, f(x) = x for x >= 0.
        // "Rectifier Nonlinearities Improve Neural Network Acoustic Models". AL Maas, 2013
        // http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf
        Assert.IsTrue(alpha <= 1);
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = Mathf.Max(v, alpha * v);
            // @TODO: doublecheck the following code
            // from Theano impl
            // https://github.com/Theano/theano/blob/d395439aec5a6ddde8ef5c266fd976412a5c5695/theano/tensor/nnet/nnet.py#L2209-L2251
            //float f1 = 0.5f * (1f + alpha)
            //float f2 = 0.5f * (1f - alpha)
            //v = f1 * v + f2 * Mathf.Abs(v);
            O[i] = v;
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Selu(Tensor X, float alpha, float gamma)
    {
        // f(x) = gamma * (alpha * e^x - alpha) for x <= 0, f(x) = gamma * x for x > 0
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            if (v <= 0)
                v = gamma * (alpha * Mathf.Exp(v) - alpha);
            else
                v = gamma * v;
            O[i] = v;
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Swish(Tensor X)
    {
        // f(x) = sigmoid(x) * x = x / (1 + exp(-x))
        // "Searching for Activation Functions". P Ramachandran, 2017
        // https://arxiv.org/abs/1710.05941
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = v / (1f + Mathf.Exp(-v));
            O[i] = v;
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Abs(Tensor X)
    {
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = Mathf.Abs(v);
            O[i] = v;
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Neg(Tensor X)
    {
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = -v;
            O[i] = v;
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Ceil(Tensor X)
    {
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = Mathf.Ceil(v);
            O[i] = v;
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Clip(Tensor X, float min, float max)
    {
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = Mathf.Clamp(v, min, max);

            O[i] = v;
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Floor(Tensor X)
    {
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = Mathf.Floor(v);
            O[i] = v;
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Round(Tensor X)
    {
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = Mathf.Round(v);
            O[i] = v;
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Reciprocal(Tensor X)
    {
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = 1.0f / v;
            O[i] = v;
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Pow(Tensor X, float alpha)
    {
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = Mathf.Pow(v, alpha);
            O[i] = v;
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Exp(Tensor X)
    {
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = Mathf.Exp(v);
            O[i] = v;
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Log(Tensor X)
    {
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = Mathf.Log(v);
            O[i] = v;
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Sqrt(Tensor X)
    {
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = Mathf.Sqrt(v);
            O[i] = v;
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Acos(Tensor X)
    {
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = Mathf.Acos(v);
            O[i] = v;
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Acosh(Tensor X)
    {
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = Mathf.Log(v + Mathf.Sqrt(v*v - 1.0f));
            O[i] = v;
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Asin(Tensor X)
    {
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = Mathf.Asin(v);
            O[i] = v;
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Asinh(Tensor X)
    {
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = Mathf.Log(v + Mathf.Sqrt(v*v + 1.0f));
            O[i] = v;
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Atan(Tensor X)
    {
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = Mathf.Atan(v);
            O[i] = v;
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Atanh(Tensor X)
    {
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = 0.5f * Mathf.Log((1.0f + v)/(1.0f - v));
            O[i] = v;
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Cos(Tensor X)
    {
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = Mathf.Cos(v);
            O[i] = v;
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Cosh(Tensor X)
    {
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = 0.5f * (Mathf.Exp(v) + Mathf.Exp(-v));
            O[i] = v;
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Sin(Tensor X)
    {
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = Mathf.Sin(v);
            O[i] = v;
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Sinh(Tensor X)
    {
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = 0.5f * (Mathf.Exp(v) - Mathf.Exp(-v));
            O[i] = v;
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Tan(Tensor X)
    {
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            v = Mathf.Tan(v);
            O[i] = v;
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Erf(Tensor X)
    {
        var O = NewTensorLike(X);

        var end = X.length;
        for (int i = 0; i < end; ++i)
        {
            float v = X[i];
            // Abramowitz/Stegun approximations
            // erf(x) = -erf(-x)
            float x = Mathf.Abs(v);

            float p = 0.3275911f;
            float a1 = 0.254829592f; float a2 = -0.284496736f; float a3 = 1.421413741f;
            float a4 = -1.453152027f; float a5 = 1.061405429f;

            float t = 1.0f / (1.0f + p * x);
            float t2 = t * t;
            float t3 = t2 * t;
            float t4 = t3 * t;
            float t5 = t4 * t;

            v = Mathf.Sign(v) * (1 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * Mathf.Exp(-x * x));
            O[i] = v;
        }
        return O;
    }

    internal long GetAggregatedDimLength(TensorShape shape, int startDim, int endDim)
    {
        long aggregatedLength = 1L;
        for (var d = startDim; d < endDim; ++d)
            aggregatedLength *= shape[d];
        return aggregatedLength;
    }

    /// <inheritdoc/>
    public virtual Tensor Concat(Tensor[] tensors, int axis)
    {
        var concatShape = TensorExtensions.Concat(tensors, axis);
        var dataType = tensors.Length > 0 ? tensors[0].dataType : DataType.Float;
        var O = NewTensor(dataType, concatShape);

        unsafe
        {
            var srcIndices = stackalloc long[tensors.Length];
            UnsafeUtility.MemClear(srcIndices, tensors.Length * Marshal.SizeOf<long>());
            // NOTE: once we have Tensor.ToReadOnlyArray(ref arrayOffset),
            // will need to initialize srcIndices[i] = arrayOffset;

            // product of all tensor dimensions starting from axis
            var copyBlockLengths = stackalloc long[tensors.Length];
            for (int i = 0; i < tensors.Length; ++i)
                copyBlockLengths[i] = GetAggregatedDimLength(tensors[i].shape,  tensors[i].shape.Axis(axis), TensorShape.MaxRank);

            // copy tensor data interleaved into O
            int intDstIndex = 0;
            var dstArray = new float[concatShape.length];
            long dstIndex = intDstIndex;
            long takes = GetAggregatedDimLength(concatShape,  0, concatShape.Axis(axis));
            for (int take = 0; take < takes; ++take)
            for (int i = 0; i < tensors.Length; ++i)
            {
                var copyLength = copyBlockLengths[i];

                Array.Copy(tensors[i].ToReadOnlyArray(), srcIndices[i], // from
                    dstArray, dstIndex, copyLength);                // to

                srcIndices[i] += copyLength;
                dstIndex += copyLength;
            }

            O.data.Upload(dstArray, concatShape, 0);
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor StridedSlice(Tensor X, int[] starts4Dor8D, int[] ends4Dor8D, int[] strides4Dor8D)
    {

        unsafe
        {
            int* starts = stackalloc int[TensorShape.MaxRank];
            int* ends = stackalloc int[TensorShape.MaxRank];
            int* strides = stackalloc int[TensorShape.MaxRank];
            TensorExtensions.Get8DParametersNoAlloc(X.shape, starts4Dor8D, starts, 0);
            TensorExtensions.Get8DParametersNoAlloc(X.shape, ends4Dor8D, ends, 1);
            TensorExtensions.Get8DParametersNoAlloc(X.shape, strides4Dor8D, strides, 1);

            var O = NewTensor(X.dataType, X.shape.ApplyStridedSlice8DUnsafeNoAlloc(starts, ends, strides));

            int* wrappedStartsIndices = ends;//reuse buffer to save a stack allocation.
            for (int i = 0; i < TensorShape.MaxRank; ++i)
                wrappedStartsIndices[i] = Math.Min(TensorExtensions.WrapIndex(starts[i], X.shape[i]), X.shape[i] - 1);

            Assert.AreEqual(8, TensorShape.MaxRank);
            for (var it = new TensorIterator(O); it.IsValid(); it.Next())
            {
                // sample either from dim or index 0 in case of expansion
                O[it.index] = X[
                    wrappedStartsIndices[0] + it.d0 * strides[0],
                    wrappedStartsIndices[1] + it.d1 * strides[1],
                    wrappedStartsIndices[2] + it.d2 * strides[2],
                    wrappedStartsIndices[3] + it.d3 * strides[3],
                    wrappedStartsIndices[4] + it.d4 * strides[4],
                    wrappedStartsIndices[5] + it.d5 * strides[5],
                    wrappedStartsIndices[6] + it.d6 * strides[6],
                    wrappedStartsIndices[7] + it.d7 * strides[7]];
            }

            return O;
        }
    }

    /// <inheritdoc/>
    public virtual Tensor Tile(Tensor X, int[] repeats)
    {
        Tensor O = NewTensor(X.dataType, X.shape.Scale(repeats));

        for (var it = new TensorIterator(O); it.IsValid(); it.Next())
        {
            // sample either from dim or index 0 in case of expansion
            O[it.index] = X[it.d0 % X.shape[0],
                            it.d1 % X.shape[1],
                            it.d2 % X.shape[2],
                            it.d3 % X.shape[3],
                            it.d4 % X.shape[4],
                            it.d5 % X.shape[5],
                            it.d6 % X.shape[6],
                            it.d7 % X.shape[7]];
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor ConstantOfShape(TensorShape X, DataType type, float value = 0.0f)
    {
        Tensor O = NewTensor(type, X);
        for (int i = 0; i < O.length; ++i)
            O[i] = value;

        return O;
    }

    /// <inheritdoc/>
    public Tensor Shape(Tensor X, int axis = -1)
    {
        int[] shape = X.shape.ToArray();

        int shapeRank = axis > 0 ? 1 : shape.Length;
        var O = NewTensor(X.dataType, new TensorShape(shapeRank, 1, 1, 1));
        if (axis > 0)
        {
            O[0] = shape[axis];
        }
        else
        {
            for (var i = 0; i < shape.Length; i++)
            {
                O[i] = shape[i];
            }
        }

        return O;
    }

    private Tensor ApplyElementwiseWithBroadcast(Tensor[] tensors, Func<float, float, float> operation)
    {
        var O = NewTensorLike(tensors, AllocScope.LayerOutput, false);
        var A = tensors[0];
        for (int t = 1; t < tensors.Length; ++t)
        {
            var B = tensors[t];
            for (var itO = new TensorIterator(O.shape); itO.IsValid(); itO.Next())
            {
                var valueA = A[A.IndexWithBroadcast(itO.d0, itO.d1, itO.d2, itO.d3, itO.d4, itO.d5, itO.d6, itO.d7)];
                var valueB = B[B.IndexWithBroadcast(itO.d0, itO.d1, itO.d2, itO.d3, itO.d4, itO.d5, itO.d6, itO.d7)];
                O[itO.index] = operation(valueA, valueB);
            }

            A = O;
        }
        return O;
    }

    /// <inheritdoc/>
    // O = tensors[0] + tensors[1] + ... + tensors[N-1]
    public virtual Tensor Add(Tensor[] tensors)
    {
        Func<float, float, float> op = (a, b) => a + b;
        return ApplyElementwiseWithBroadcast(tensors, op);
    }

    /// <inheritdoc/>
    // O = tensors[0] - tensors[1] - ... - tensors[N-1]
    public virtual Tensor Sub(Tensor[] tensors)
    {
        Func<float, float, float> op = (a, b) => a - b;
        return ApplyElementwiseWithBroadcast(tensors, op);
    }

    /// <inheritdoc/>
    // O = tensors[0] * tensors[1] * ... * tensors[N-1]
    public virtual Tensor Mul(Tensor[] tensors)
    {
        Func<float, float, float> op = (a, b) => a * b;
        return ApplyElementwiseWithBroadcast(tensors, op);
    }

    /// <inheritdoc/>
    // O = tensors[0] / tensors[1] / ... / tensors[N-1]
    public virtual Tensor Div(Tensor[] tensors)
    {
        Func<float, float, float> op = (a, b) => a / b;
        return ApplyElementwiseWithBroadcast(tensors, op);
    }

    /// <inheritdoc/>
    // O = tensors[0] ^ tensors[1] ^ ... ^ tensors[N-1]
    public virtual Tensor Pow(Tensor[] tensors)
    {
        Func<float, float, float> op = (a, b) => Mathf.Pow(a, b);
        return ApplyElementwiseWithBroadcast(tensors, op);
    }

    /// <inheritdoc/>
    // O = min(tensors[0], tensors[1],  ... , tensors[N-1])
    public virtual Tensor Min(Tensor[] tensors)
    {
        Func<float, float, float> op = (a, b) => Mathf.Min(a, b);
        return ApplyElementwiseWithBroadcast(tensors, op);
    }

    /// <inheritdoc/>
    // O = max(tensors[0], tensors[1],  ... , tensors[N-1])
    public virtual Tensor Max(Tensor[] tensors)
    {
        Func<float, float, float> op = (a, b) => Mathf.Max(a, b);
        return ApplyElementwiseWithBroadcast(tensors, op);
    }

    /// <inheritdoc/>
    // O = (1/N) * (tensors[0] + tensors[1] + ... + tensors[N-1])
    public virtual Tensor Mean(Tensor[] tensors)
    {
        // accumulate
        Func<float, float, float> op = (a, b) => a + b;
        var O = ApplyElementwiseWithBroadcast(tensors, op);

        // div by N
        var invN = 1.0f / tensors.Length;
        var end = O.length;
        for (int i = 0; i < end; ++i)
        {
            float v = O[i];
            v *= invN;
            O[i] = v;
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor ReduceMin(Tensor X, int axis)
    {
        var O = NewTensor(X.dataType, X.shape.Reduce(axis));

        for (var itO = new TensorIterator(O.shape); itO.IsValid(); itO.Next())
        {
            O[itO.index] = float.MaxValue;
        }
        for (var itX = new TensorIterator(X.shape); itX.IsValid(); itX.Next())
        {
            int iO = itX.IndexInReducedShape(O.shape);
            O[iO] = Mathf.Min(O[iO], X[itX.index]);
        }

        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor ReduceMax(Tensor X, int axis)
    {
        var O = NewTensor(X.dataType, X.shape.Reduce(axis));

        for (var itO = new TensorIterator(O.shape); itO.IsValid(); itO.Next())
        {
            O[itO.index] = float.MinValue;
        }
        for (var itX = new TensorIterator(X.shape); itX.IsValid(); itX.Next())
        {
            int iO = itX.IndexInReducedShape(O.shape);
            O[iO] = Mathf.Max(O[iO], X[itX.index]);
        }

        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor ArgMax(Tensor X, int axis)
    {
        var O = NewTensor(X.dataType, X.shape.Reduce(axis));

        for (var itO = new TensorIterator(O.shape); itO.IsValid(); itO.Next())
        {
            O[itO.index] = 0;
        }

        for (var itX = new TensorIterator(X.shape); itX.IsValid(); itX.Next())
        {
            int iO = itX.IndexInReducedShape(O.shape);
            int xBestValueIndex = itX.IndexWithReplacedAxis(axis, (int) O[iO]);
            if (X[itX.index] > X[xBestValueIndex])
                O[iO] = itX[axis];
        }

        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor ArgMin(Tensor X, int axis)
    {
        var O = NewTensor(X.dataType, X.shape.Reduce(axis));

        for (var itO = new TensorIterator(O.shape); itO.IsValid(); itO.Next())
        {
            O[itO.index] = 0;
        }

        for (var itX = new TensorIterator(X.shape); itX.IsValid(); itX.Next())
        {
            int iO = itX.IndexInReducedShape(O.shape);
            int xBestValueIndex = itX.IndexWithReplacedAxis(axis, (int) O[iO]);
            if (X[itX.index] <  X[xBestValueIndex])
                O[iO] = itX[axis];
        }

        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor ReduceSum(Tensor X, int axis)
    {
        var O = NewTensor(X.dataType, X.shape.Reduce(axis));

        for (var itO = new TensorIterator(O.shape); itO.IsValid(); itO.Next())
        {
            O[itO.index] = 0.0f;
        }
        for (var itX = new TensorIterator(X.shape); itX.IsValid(); itX.Next())
        {
            O[itX.IndexInReducedShape(O.shape)] += X[itX.index];
        }

        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor ReduceMean(Tensor X, int axis)
    {
        var O = NewTensor(X.dataType, X.shape.Reduce(axis));

        for (var itO = new TensorIterator(O.shape); itO.IsValid(); itO.Next())
        {
            O[itO.index] = 0.0f;
        }
        for (var itX = new TensorIterator(X.shape); itX.IsValid(); itX.Next())
        {
            O[itX.IndexInReducedShape(O.shape)] += X[itX.index];
        }
        for (var itO = new TensorIterator(O.shape); itO.IsValid(); itO.Next())
        {
            O[itO.index] /= X.shape[axis];
        }

        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor ReduceProd(Tensor X, int axis)
    {
        var O = NewTensor(X.dataType, X.shape.Reduce(axis));

        for (var itO = new TensorIterator(O.shape); itO.IsValid(); itO.Next())
        {
            O[itO.index] = 1.0f;
        }
        for (var itX = new TensorIterator(X.shape); itX.IsValid(); itX.Next())
        {
            O[itX.IndexInReducedShape(O.shape)] *= X[itX.index];
        }

        return O;
    }

    /// <inheritdoc/>
    private Tensor ApplyLogicalOperator(Tensor tensorA, Tensor tensorB, Func<float, float, float> logicOp)
    {
        var O = NewTensorLike(new Tensor[] { tensorA, tensorB }, AllocScope.LayerOutput, false);
        for (var itO = new TensorIterator(O.shape); itO.IsValid(); itO.Next())
        {
            var A = tensorA[tensorA.IndexWithBroadcast(itO.d0, itO.d1, itO.d2, itO.d3, itO.d4, itO.d5, itO.d6, itO.d7)];
            var B = tensorB[tensorB.IndexWithBroadcast(itO.d0, itO.d1, itO.d2, itO.d3, itO.d4, itO.d5, itO.d6, itO.d7)];
            O[itO.index] = logicOp(A,B);
        }

        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Greater(Tensor A, Tensor B)
    {
        Func<float, float, float> logicOp = (a, b) => Convert.ToSingle(a > b);
        return ApplyLogicalOperator(A, B, logicOp);
    }

    /// <inheritdoc/>
    public virtual Tensor GreaterEqual(Tensor A, Tensor B)
    {
        Func<float, float, float> logicOp = (a, b) => Convert.ToSingle(a >= b);
        return ApplyLogicalOperator(A, B, logicOp);
    }

    /// <inheritdoc/>
    public virtual Tensor Less(Tensor A, Tensor B)
    {
        Func<float, float, float> logicOp = (a, b) => Convert.ToSingle(a < b);
        return ApplyLogicalOperator(A, B, logicOp);
    }

    /// <inheritdoc/>
    public virtual Tensor LessEqual(Tensor A, Tensor B)
    {
        Func<float, float, float> logicOp = (a, b) => Convert.ToSingle(a <= b);
        return ApplyLogicalOperator(A, B, logicOp);
    }

    /// <inheritdoc/>
    public virtual Tensor Equal(Tensor A, Tensor B)
    {
        Func<float, float, float> logicOp = (a, b) => Convert.ToSingle(a == b);
        return ApplyLogicalOperator(A, B, logicOp);
    }

    /// <inheritdoc/>
    public virtual Tensor LogicalOr(Tensor A, Tensor B)
    {
        Func<float, float, float> logicOp = (a, b) => Convert.ToSingle( Convert.ToBoolean(a) || Convert.ToBoolean(b) );
        return ApplyLogicalOperator(A, B, logicOp);
    }

    /// <inheritdoc/>
    public virtual Tensor LogicalAnd(Tensor A, Tensor B)
    {
        Func<float, float, float> logicOp = (a, b) => Convert.ToSingle( Convert.ToBoolean(a) && Convert.ToBoolean(b) );
        return ApplyLogicalOperator(A, B, logicOp);
    }

    /// <inheritdoc/>
    public virtual Tensor LogicalXor(Tensor A, Tensor B)
    {
        Func<float, float, float> logicOp = (a, b) => Convert.ToSingle( Convert.ToBoolean(a) ^ Convert.ToBoolean(b) );
        return ApplyLogicalOperator(A, B, logicOp);
    }

    /// <inheritdoc/>
    public virtual Tensor LogicalNot(Tensor X)
    {
        var O = NewTensorLike(X);
        var end = O.length;
        for (int i = 0; i < end; ++i)
            O[i] = Convert.ToSingle( !Convert.ToBoolean(X[i]) );
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Sign(Tensor X)
    {
        var O = NewTensorLike(X);
        var end = O.length;
        for (int i = 0; i < end; ++i)
            O[i] = (X[i] > 0) ? 1.0f : ((X[i] < 0) ? -1.0f : 0.0f);
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Where(Tensor C, Tensor A, Tensor B)
    {
        var O = NewTensorLike(new [] { C, A, B }, AllocScope.LayerOutput, false);
        for (var itO = new TensorIterator(O.shape); itO.IsValid(); itO.Next())
        {
            var x = A[A.IndexWithBroadcast(itO.d0, itO.d1, itO.d2, itO.d3, itO.d4, itO.d5, itO.d6, itO.d7)];
            var y = B[B.IndexWithBroadcast(itO.d0, itO.d1, itO.d2, itO.d3, itO.d4, itO.d5, itO.d6, itO.d7)];
            var c = C[C.IndexWithBroadcast(itO.d0, itO.d1, itO.d2, itO.d3, itO.d4, itO.d5, itO.d6, itO.d7)];
            O[itO.index] = Convert.ToBoolean(c) ? x : y;
        }

        return O;
    }

    /// <summary>
    /// Copy and reshape `Tensor`
    /// </summary>
    /// <param name="X">input</param>
    /// <param name="shape">shape</param>
    /// <returns>output `Tensor`</returns>
    protected virtual Tensor CopyAndReshape(Tensor X, TensorShape shape)
    {
        Assert.AreEqual(X.length, shape.length);
        var O = NewTensor(X.dataType, shape);
        for (int i = 0; i < X.length; ++i)
            O[i] = X[i];
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Copy(Tensor X)
    {
        // make shallow copy and patch the shape, if already managed by allocator
        if (X.allocator != null)
            return X.ShallowCopy(m_StringCache.Lookup("ShallowCopy of", X.name));

        return CopyAndReshape(X, X.shape);
    }

    /// <inheritdoc/>
    public virtual Tensor Flatten(Tensor X)
    {
        // make shallow copy and patch the shape, if already managed by allocator
        if (X.allocator != null)
            return X.Flatten(m_StringCache.Lookup("Flatten of", X.name));

        // otherwise deep copy
        var newShape = X.shape.Flatten();
        return CopyAndReshape(X, newShape);
    }

    /// <inheritdoc/>
    public virtual Tensor Reshape(Tensor X, TensorShape newShape)
    {
        // if already managed by allocator, can do a shallow copy
        bool canDoShallowCopy = X.allocator != null;

        // in most case layer needing storage should use there own
        // allocator to avoid memory fragmentation in the long run.
        // Here we disallow shallow copy in that case here to help.
        // Would be better to verify if target and source allocator
        // are the same but storage/reshape-to-storage is an uncommon case.
        var varsWithReuse = m_Allocator as GenericVarsWithReuse;
        canDoShallowCopy &= varsWithReuse != null &&
                            !varsWithReuse.layerRequiresStorage;

        // however if tensor is on GPU and in channel first memory layout we can't (reshape is actually a transpose in that case)
        var onDeviceComputeTensorData = X.tensorOnDevice as ComputeTensorData;
        canDoShallowCopy &= onDeviceComputeTensorData == null ||
                            onDeviceComputeTensorData.channelsOrder == ComputeInfo.ChannelsOrder.NHWC;

        if (canDoShallowCopy)
            return X.Reshape(newShape, m_StringCache.Lookup("Reshape of", X.name));

        // otherwise deep copy
        return CopyAndReshape(X, newShape);
    }

    /// <inheritdoc/>
    public virtual Tensor Expand(Tensor X, TensorShape newShape)
    {
        // scale is either 1 or 0 in case of expansion
        int[] s = new int[TensorShape.MaxRank];
        for(int i = 0; i < TensorShape.MaxRank; ++i)
            s[i] = X.shape[i] / newShape[i];

        for (int i = 0; i < TensorShape.MaxRank; ++i)
        {
            Assert.IsTrue(newShape[i] == X.shape[i] || X.shape[i] == 1);
            Assert.IsTrue(s[i] == 0 || s[i] == 1);
        }

        var O = NewTensor(X.dataType, newShape);
        Assert.AreEqual(8, TensorShape.MaxRank);
        for (var it = new TensorIterator(newShape); it.IsValid(); it.Next())
        {
            // sample either from dim or index 0 in case of expansion
            O[it.index] = X[s[0]*it.d0, s[1]*it.d1, s[2]*it.d2, s[3]*it.d3, s[4]*it.d4, s[5]*it.d5, s[6]*it.d6, s[7]*it.d7];
        }

        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Gather(Tensor[] tensors, int axis)
    {
        Tensor X = tensors[0];
        Tensor indices = tensors[1];

        var shape = X.shape;
        shape[axis] = indices.length;

        var O = NewTensor(X.dataType, shape);

        Assert.AreEqual(TensorShape.MaxRank, 8);
        for (var it = new TensorIterator(shape); it.IsValid(); it.Next())
        {
            int d0 = (axis == 0) ? (int) indices[it.d0] : it.d0;
            int d1 = (axis == 1) ? (int) indices[it.d1] : it.d1;
            int d2 = (axis == 2) ? (int) indices[it.d2] : it.d2;
            int d3 = (axis == 3) ? (int) indices[it.d3] : it.d3;
            int d4 = (axis == 4) ? (int) indices[it.d4] : it.d4;
            int d5 = (axis == 5) ? (int) indices[it.d5] : it.d5;
            int d6 = (axis == 6) ? (int) indices[it.d6] : it.d6;
            int d7 = (axis == 7) ? (int) indices[it.d7] : it.d7;
            O[it.index] = X[d0, d1, d2, d3, d4, d5, d6, d7];
        }
        return O;
    }

    public virtual Tensor ScatterND(Tensor X, Tensor indices, Tensor updates, Layer.ScatterNDReductionMode reduction)
    {
        // only support for scattering on C for now
        Assert.IsTrue(indices.batch == X.batch);
        Assert.IsTrue(updates.width == X.width && updates.height == X.height);
        var outputShape = X.shape;

        var O = NewTensor(X.dataType, outputShape);

        for (var n = 0; n < O.batch; ++n)
        for (var h = 0; h < O.height; ++h)
        for (var w = 0; w < O.width; ++w)
        for (var c = 0; c < O.channels; ++c)
        {
            float v = X[n, h, w, c];
            O[n, h, w, c] = v;

            for (int idx = 0; idx < indices.flatWidth; idx++)
            {
                int indexRemap = (int)(indices[idx]);
                if (c != indexRemap)
                    continue;

                float vw = updates[n % updates.batch, h % updates.height, w % updates.width, idx % updates.channels];

                int indexWrite = O.Index(n, h, w, indexRemap);
                if (reduction == Layer.ScatterNDReductionMode.None)
                {
                    O[indexWrite] = vw;
                }
                else if (reduction == Layer.ScatterNDReductionMode.Add)
                {
                    O[indexWrite] += vw;
                }
                else if (reduction == Layer.ScatterNDReductionMode.Mul)
                {
                    O[indexWrite] *= vw;
                }
            }
        }

        return O;
    }

    /// <inheritdoc/>
    public Tensor NonMaxSuppression(Tensor[] tensors, int maxOutputBoxesPerClass, float iouThreshold, float scoreThreshold, int centerPointBox)
    {
        // ONNX: https://github.com/onnx/onnx/blob/master/docs/Operators.md#NonMaxSuppression
        // ORT reference: https://github.com/microsoft/onnxruntime/blob/464bbd27a939ebc73bfd7fe3eea0eeb93a76e56b/onnxruntime/core/providers/cpu/object_detection/non_max_suppression.cc
        // PyTorch: https://pytorch.org/docs/stable/_modules/torchvision/ops/boxes.html#nms
        var boxes = tensors[0];
        var scores = tensors[1];

        Assert.IsTrue(boxes.shape.Is4D());//should be rank 3
        Assert.IsTrue(scores.shape.Is4D());//should be rank 3

        int boxCount = Mathf.Min(boxes.channels, scores.width); // Box spatial dimension (C) / Score spatial dimension (W)
        var boxIndices = new List<int>(boxCount);
        var selectedIndices = new List<(int, int, int)>(); // batch index, class index, box index
        var classSelectedIndices = new List<(int, int, int)>(); // batch index, class index, box index
        var S = new List<float>();

        for (int n = 0; n < scores.batch; n++)
        {
            // Iterate over each class
            for (int c = 0; c < scores.channels; c++)
            {
                classSelectedIndices.Clear();

                boxIndices.Clear();
                S.Clear();
                for (int b = 0; b < boxCount; b++)
                {
                    float score = scores[n, 0, b, c];
                    if (score > scoreThreshold)
                    {
                        S.Add(score);
                        boxIndices.Add(b);
                    }
                }

                while (boxIndices.Any() && classSelectedIndices.Count < maxOutputBoxesPerClass)
                {
                    float maxScore = float.MinValue;
                    int relativeIndex = 0;
                    for (int i = 0; i < S.Count; i++)
                    {
                        float score = S[i];
                        if (score > maxScore)
                        {
                            maxScore = score;
                            relativeIndex = i;
                        }
                    }

                    int m = boxIndices[relativeIndex]; // Get absolute index from relative index since the working sets change
                    Rect M = centerPointBox == 0 ? GetRect(boxes, n, m) : GetRectFromCenter(boxes, n, m);

                    boxIndices.RemoveAt(relativeIndex);
                    S.RemoveAt(relativeIndex);

                    // Suppress this box if IOU with another box exceeds threshold
                    var selected = true;
                    foreach (var (_, _, otherIndex) in classSelectedIndices)
                    {
                        Rect b = centerPointBox == 0 ? GetRect(boxes, n, otherIndex) : GetRectFromCenter(boxes, n, otherIndex);
                        if (M.Overlaps(b) && GetIntersectionOverUnionArea(M, b) > iouThreshold)
                        {
                            selected = false;
                            break;
                        }
                    }

                    if (selected)
                        classSelectedIndices.Add((n, c, m));
                }

                // Collect what was selected for this class
                selectedIndices.AddRange(classSelectedIndices);
            }
        }

        var O = NewTensor(boxes.dataType, new TensorShape(new [] {selectedIndices.Count, 1, 1, 3}));
        if (selectedIndices.Count > 0)
        {
            for (var i = 0; i < selectedIndices.Count; i++)
            {
                (int batchIndex, int classIndex, int boxIndex) = selectedIndices[i];
                O[i, 0] = batchIndex;
                O[i, 1] = classIndex;
                O[i, 2] = boxIndex;
            }
        }
        else
        {
            // TODO: Remove this when empty tensors are supported
            // See https://github.com/Unity-Technologies/barracuda-release/issues/173#issuecomment-837352917
            O.Fill(-1f);
        }

        return O;

        float GetIntersectionOverUnionArea(Rect a, Rect b)
        {
            var intersectionArea = GetIntersectionArea(a, b);
            return intersectionArea / (a.width * a.height + b.width * b.height - intersectionArea);
        }

        float GetIntersectionArea(Rect a, Rect b)
        {
            float xMin = Mathf.Max(a.xMin, b.xMin);
            float yMin = Mathf.Max(a.yMin, b.yMin);
            float xMax = Mathf.Min(a.xMax, b.xMax);
            float yMax = Mathf.Min(a.yMax, b.yMax);

            var rect = Rect.MinMaxRect(xMin, yMin, xMax, yMax);
            return Math.Max(rect.width, 0) * Math.Max(rect.height, 0); // Non-overlapping rects will have negative width / height
        }

        Rect GetRect(Tensor t, int batch, int index)
        {
            TensorShape tShape = t.shape;
            float x1 = t[tShape.Index(batch, 0, 1, index)];
            float y1 = t[tShape.Index(batch, 0, 0, index)];
            float x2 = t[tShape.Index(batch, 0, 3, index)];
            float y2 = t[tShape.Index(batch, 0, 2, index)];

            // Correct flipped coordinates
            if (x1 > x2)
            {
                float temp = x1;
                x1 = x2;
                x2 = temp;
            }

            if (y1 > y2)
            {
                float temp = y1;
                y1 = y2;
                y2 = temp;
            }

            return Rect.MinMaxRect(x1, y1, x2, y2);
        }

        Rect GetRectFromCenter(Tensor t, int batch, int index)
        {
            TensorShape tShape = t.shape;
            float xCenter = t[tShape.Index(batch, 0, 0, index)];
            float yCenter = t[tShape.Index(batch, 0, 1, index)];
            float width =   t[tShape.Index(batch, 0, 2, index)];
            float height =  t[tShape.Index(batch, 0, 3, index)];

            float halfWidth = width * 0.5f;
            float halfHeight = height * 0.5f;

            return new Rect(xCenter - halfWidth, yCenter - halfHeight, width, height);
        }
    }

    /// <inheritdoc/>
    public virtual Tensor[] LSTM(Tensor X, Tensor[] W, Tensor[] R, Tensor[] Wb, Tensor[] Rb, Tensor hidden, Tensor cell)
    {
        // Gate indices [iofj]
        const int g_i = 0, g_o = 1, g_f = 2, g_j = 3;

        TensorShape xShape = X.shape;
        int sequenceLength = xShape.batch; // X shape is [seq_length, batch_size, input_size]

        Tensor O = null;

        for (int s = 0; s < sequenceLength; s++)
        {
            using (var td = new TensorScope()) // This will dispose every sequence iteration
            {
                TensorScope.F _ = td._; // Shorthand
                Tensor X_sequence = _(StridedSlice(X, new[] { s, 0, 0, 0 }, new[] { s + 1, int.MaxValue, int.MaxValue, int.MaxValue }, new[] { 1, 1, 1, 1 }));

                // Convert to [batch_size, input_size], dropping sequence axis
                X_sequence = _(Transpose(X_sequence, new[] { 3, 0, 1, 2 }));

                var i_mad_w = _(Add(new[] { _(MatMul(X_sequence, false, W[g_i], false)), Wb[g_i] }));
                var i_mad_r = _(Add(new[] { _(MatMul(hidden, false, R[g_i], false)), Rb[g_i] }));
                var i_mad = _(Add(new[] { i_mad_w, i_mad_r }));

                var j_mad_w = _(Add(new[] { _(MatMul(X_sequence, false, W[g_j], false)), Wb[g_j] }));
                var j_mad_r = _(Add(new[] { _(MatMul(hidden, false, R[g_j], false)), Rb[g_j] }));
                var j_mad = _(Add(new[] { j_mad_w, j_mad_r }));

                var f_mad_w = _(Add(new[] { _(MatMul(X_sequence, false, W[g_f], false)), Wb[g_f] }));
                var f_mad_r = _(Add(new[] { _(MatMul(hidden, false, R[g_f], false)), Rb[g_f] }));
                var f_mad = _(Add(new[] { f_mad_w, f_mad_r }));

                var o_mad_w = _(Add(new[] { _(MatMul(X_sequence, false, W[g_o], false)), Wb[g_o] }));
                var o_mad_r = _(Add(new[] { _(MatMul(hidden, false, R[g_o], false)), Rb[g_o] }));
                var o_mad = _(Add(new[] { o_mad_w, o_mad_r }));

                var i = _(Sigmoid(i_mad));
                var j = _(Tanh(j_mad));
                var f = _(Sigmoid(f_mad));
                var o = _(Sigmoid(o_mad));

                var state_c_mul = _(Mul(new[] { cell, f }));
                var i_j_mul = _(Mul(new[] { i, j }));
                var state_c = Add(new[] { state_c_mul, i_j_mul });  // Not disposed automatically
                var state_c_tanh = _(Tanh(state_c));
                var state_h = Mul(new[] { o, state_c_tanh });       // Not disposed automatically

                // Must be in the shape [num_directions=1, batch_size, hidden_size]
                Tensor reshaped_state_h = Reshape(state_h, new TensorShape(1, state_h.batch, state_h.channels, 1));
                if (O == null)
                    O = reshaped_state_h;
                else
                    O = Concat(new[] { _(O), _(reshaped_state_h) }, TensorShape.DataBatch);

                // Collect previous memories before assigning new ones.
                // Don't dispose the original hidden / cell memories since those were input tensors
                if (s != 0)
                {
                    _(hidden);
                    _(cell);
                }

                hidden = state_h;
                cell = state_c;
            }
        }

        return new[] { O, hidden, cell };
    }

    /// <inheritdoc/>
    public virtual Tensor Transpose(Tensor X)
    {
        // TODO: reshape when possible
        Assert.IsTrue(X.dimensions <= 2);
        X = Flatten(X);

        var O = NewTensor(X.dataType, X.flatWidth, X.flatHeight);

        for (int y = 0; y < O.flatHeight; ++y)
            for (int x = 0; x < O.flatWidth; ++x)
                O[y, x] = X[x, y];

        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Transpose(Tensor X, int[] permutations)
    {
        permutations = TensorExtensions.Get8DPermutationsForNHWCPermutationsAndShape(X.shape, permutations);
        var O = NewTensor(X.dataType, X.shape.Permute(permutations));

        Assert.AreEqual(TensorShape.MaxRank, 8);
        for (var it = new TensorIterator(X); it.IsValid(); it.Next())
        {
            O[  it[permutations[0]], it[permutations[1]],
                it[permutations[2]], it[permutations[3]],
                it[permutations[4]], it[permutations[5]],
                it[permutations[6]], it[permutations[7]]] = X[it.index];
        }
        return O;
    }

    /// <inheritdoc/>
    public virtual Tensor Prepare(Tensor X)
    {
        X.PrepareCacheForAccess();
        return X;
    }

    /// <inheritdoc/>
    public virtual Tensor PrepareNoAlloc(Tensor X)
    {
        // reference op 0-initalize tensors
        X.PrepareCacheForAccess();
        return X;
    }
}

internal class MathfEx
{
    internal static float Tanh(float x)
    {
        // tanh = (exp(2*x) - 1) / (exp(2*x) + 1)

        // Constant taken from http://llvm.org/svn/llvm-project/libclc/trunk/generic/lib/math/tanh.cl
        // const float large_threshold = 0x1.0a2b24p+3f;
        const float LargeThreshold = 8.317766f;

        // See also: https://stackoverflow.com/questions/34835641/tanh-returning-nan-for-large-input

        // Handle edge-cases to prevent NaNs creeping in
        if (x >= LargeThreshold || x <= -LargeThreshold)
            return Mathf.Sign(x);

        float exp2 = Mathf.Exp(2f * x);
        return (exp2 - 1f) / (exp2 + 1f);
    }
}

} // namespace Unity.Barracuda
