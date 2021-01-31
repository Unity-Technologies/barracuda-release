using System;
using UnityEngine;
using System.Collections.Generic;

namespace Unity.Barracuda {

/// <summary>
/// Proxy `IOps` implementation for tracking computational expenses for specific model
/// </summary>
public class StatsOps : IOps, IModelCompiler
{
    class Transcendental
    {

        // Table of approximate alu operation costs
        //  mul       1
        //  rcp/mad   2
        //  div/sqrt  10
        //  log/exp   100
        //  pow       200
        // see: https://www.sciencedirect.com/topics/computer-science/division-operation
        // see: https://colfaxresearch.com/arithmetics-on-intels-sandy-bridge-and-westmere-cpus-not-all-flops-are-created-equal/

        public const long Reciprocal = 2L;
        public const long Div = 10L;
        public const long Root = 10L;
        public const long Exponent = 100L;
        public const long Pow = 200L;
        public const long Trigonometric = 200L;
    }

    private IOps m_Ops;
    private long m_Alu;
    private long m_Mem;

    /// <summary>
    /// Create `StatsOps`
    /// </summary>
    /// <param name="ops">target ops</param>
    public StatsOps(IOps ops)
    {
        m_Ops = ops;
        m_Alu = 0L;
        m_Mem = 0L;
    }

    /// <inheritdoc/>
    public virtual void PrepareModel(Model model, IDictionary<string, TensorShape> inputShapes)
    {
        if (m_Ops is IModelCompiler)
            ((IModelCompiler)m_Ops).PrepareModel(model, inputShapes);
    }

    /// <inheritdoc/>
    public virtual void PreExecuteLayer(Layer layer, Tensor[] inputs)
    {
        if (m_Ops is IModelCompiler)
            ((IModelCompiler)m_Ops).PreExecuteLayer(layer, inputs);
    }

    /// <inheritdoc/>
    Tensor IOps.MatMul(Tensor X, int rankX, Tensor Y, int rankY)
    {
        var O = m_Ops.MatMul(X, rankX, Y, rankY);

        m_Alu += (long)X.height * (long)X.width * (long)Y.width * 2L * (long)X.batch * (long)X.channels;
        m_Mem += (long)X.length + (long)Y.length + (long)O.length;

        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.MatMul(Tensor X, bool xTranspose, Tensor Y, bool yTranspose)
    {
        var O = m_Ops.MatMul(X, xTranspose, Y, yTranspose);
        m_Alu += (long)X.flatHeight * (long)X.flatWidth * (long)Y.flatWidth * 2L;
        m_Mem += (long)X.length + (long)Y.length + (long)O.length;
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Dense(Tensor X, Tensor W, Tensor B, Layer.FusedActivation fusedActivation)
    {
        var O = m_Ops.Dense(X, W, B, fusedActivation);
        m_Alu += (long)X.flatHeight * (long)X.flatWidth * (long)W.flatWidth * 2L;
        m_Mem += (long)X.length + (long)W.length + (long)B.length + (long)O.length;
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Conv2D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, Layer.FusedActivation fusedActivation)
    {
        var O = m_Ops.Conv2D(X, K, B, stride, pad, fusedActivation);
        long m = (long)O.batch * (long)O.width * (long)O.height;
        long n = (long)X.channels;
        long k = (long)K.kernelWidth * (long)K.kernelHeight * (long)K.channels;
        m_Alu += m * n * k * 2L;
        m_Mem += (long)X.length + (long)K.length + (long)B.length + (long)O.length;
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Conv3D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, Layer.FusedActivation fusedActivation)
    {
        var O = m_Ops.Conv3D(X, K, B, stride, pad, fusedActivation);
        long m = (long)O.batch * (long)O.width * (long)O.height * O.depth;
        long n = (long)X.channels;
        long k = (long)K.kernelSpatialDepth * K.kernelWidth * (long)K.kernelHeight * (long)K.channels;
        m_Alu += m * n * k * 2L;
        m_Mem += (long)X.length + (long)K.length + (long)B.length + (long)O.length;
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.DepthwiseConv2D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, Layer.FusedActivation fusedActivation)
    {
        var O = m_Ops.DepthwiseConv2D(X, K, B, stride, pad, fusedActivation);
        long m = (long)O.batch * (long)O.width * (long)O.height;
        long n = (long)X.channels;
        long k = (long)K.kernelWidth * (long)K.kernelHeight;
        m_Alu += m * n * k * 2L;
        m_Mem += (long)X.length + (long)K.length + (long)B.length + (long)O.length;
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Conv2DTrans(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, int[] outputAdjustment, Layer.FusedActivation fusedActivation)
    {
        var O = m_Ops.Conv2DTrans(X, K, B, stride, pad, outputAdjustment, fusedActivation);
        long m = (long)O.batch * (long)O.width * (long)O.height;
        long n = (long)X.channels;
        long k = (long)(K.kernelWidth/stride[1]) * (long)(K.kernelHeight/stride[0]) * (long)K.channels;
        m_Alu += m * n * k * 2L;
        m_Mem += (long)X.length + (long)K.length + (long)B.length + (long)O.length;
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Upsample2D(Tensor X, int[] scale, bool bilinear)
    {
        var O = m_Ops.Upsample2D(X, scale, bilinear);
        m_Alu += (long)O.length * (bilinear ? 8 : 1);
        m_Mem += (long)X.length * (bilinear ? 4 : 1) + (long)O.length;
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Upsample3D(Tensor X, int[] scale, bool trilinear)
    {
        var O = m_Ops.Upsample3D(X, scale, trilinear);
        m_Alu += (long)O.length * (trilinear ? 18 : 1);
        m_Mem += (long)X.length * (trilinear ? 8 : 1) + (long)O.length;
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Resample2D(Tensor X, int[] size, bool bilinear)
    {
        var O = m_Ops.Resample2D(X, size, bilinear);
        m_Alu += (long)O.length * (bilinear ? 8 : 1);
        m_Mem += (long)X.length * (bilinear ? 4 : 1) + (long)O.length;
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.DepthToSpace(Tensor X, int[] scale, Layer.DepthToSpaceMode mode)
    {
        var O = m_Ops.DepthToSpace(X, scale, mode);
        m_Mem += (long)X.length + (long)O.length;
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.SpaceToDepth(Tensor X, int[] scale)
    {
        var O = m_Ops.SpaceToDepth(X, scale);
        m_Mem += (long)X.length + (long)O.length;
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.MaxPool2D(Tensor X, int[] pool, int[] stride, int[] pad)
    {
        var O = m_Ops.MaxPool2D(X, pool, stride, pad);
        Reduce(X, O);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.AvgPool2D(Tensor X, int[] pool, int[] stride, int[] pad)
    {
        var O = m_Ops.AvgPool2D(X, pool, stride, pad);
        Reduce(X, O);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.GlobalMaxPool2D(Tensor X)
    {
        var O = m_Ops.GlobalMaxPool2D(X);
        Reduce(X, O);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.GlobalAvgPool2D(Tensor X)
    {
        var O = m_Ops.GlobalAvgPool2D(X);
        Reduce(X, O);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.GlobalAvgVariancePool2D(Tensor X)
    {
        var O = m_Ops.GlobalAvgVariancePool2D(X);
        m_Alu += (long)X.length * 2L + (long)O.length;
        m_Mem += (long)X.length + (long)O.length;
        return O;
    }
    /// <inheritdoc/>
    Tensor IOps.Border2D(Tensor X, int[] pad, float value)
    {
        var O = m_Ops.Border2D(X, pad, value);
        m_Mem += (long)X.length + (long)O.length;
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Border3D(Tensor X, int[] pad, float value)
    {
        var O = m_Ops.Border3D(X, pad, value);
        m_Mem += (long)X.length + (long)O.length;
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Pad2DReflect(Tensor X, int[] pad)
    {
        var O = m_Ops.Pad2DReflect(X, pad);
        m_Mem += (long)X.length + (long)O.length;
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Pad2DSymmetric(Tensor X, int[] pad)
    {
        var O = m_Ops.Pad2DSymmetric(X, pad);
        m_Mem += (long)X.length + (long)O.length;
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Pad2DEdge(Tensor X, int[] pad)
    {
        var O = m_Ops.Pad2DEdge(X, pad);
        m_Mem += (long)X.length + (long)O.length;
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.ScaleBias(Tensor X, Tensor S, Tensor B)
    {
        Elementwise(X, 2L);
        return m_Ops.ScaleBias(X, S, B);
    }

    /// <inheritdoc/>
    Tensor IOps.Normalization(Tensor X, Tensor S, Tensor B, int pool, int axis, float epsilon, Layer.FusedActivation fusedActivation)
    {
        var O = m_Ops.Normalization(X, S, B, pool, axis, epsilon, fusedActivation);
        m_Alu += (long)X.length * 4L + (long)O.length * 2L;
        m_Mem += (long)X.length + (long)O.length;
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.LRN(Tensor X, float alpha, float beta, float bias, int size)
    {
        var O = m_Ops.LRN(X, alpha, beta, bias, size);
        //A bit over conservative. Number of read/alu is lower than `size` when normalisation windows is too large for data at current index.
        long sizeL = size;
        m_Alu += (long)X.length * (5L + sizeL * 2L);
        m_Mem += (long)X.length * (sizeL + 2L);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Dropout(Tensor X, float alpha)
    {
        Elementwise(X);
        return m_Ops.Dropout(X, alpha);
    }

    /// <inheritdoc/>
    Tensor IOps.RandomNormal(TensorShape s, float mean, float scale, int seed)
    {
        var O = m_Ops.RandomNormal(s, mean, scale, seed);
        // @TODO: not implemented
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.RandomUniform(TensorShape s, float mean, float scale, int seed)
    {
        var O = m_Ops.RandomUniform(s, mean, scale, seed);
        // @TODO: not implemented
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Multinomial(Tensor X, int count, int seed)
    {
        var O = m_Ops.Multinomial(X, count, seed);
        // @TODO: not implemented
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.OneHot(Tensor X, int depth, float onValue, float offValue)
    {
        var O = m_Ops.OneHot(X, depth, onValue, offValue);
        // @TODO: not implemented
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.TopKIndices(Tensor X, int k, int axis, bool largest, bool sorted)
    {
        var O = m_Ops.TopKIndices(X, k, axis, largest, sorted);
        // @TODO: not implemented
        return O;
    }

    /// <inheritdoc/>
    public Tensor TopKValues(Tensor X, Tensor I, int axis)
    {
        var O = m_Ops.TopKValues(X, I, axis);
        // @TODO: not implemented
        return O;
    }

    /// <inheritdoc/>
    public Tensor NonZero(Tensor X)
    {
        var O = m_Ops.NonZero(X);
        // @TODO: not implemented
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Relu(Tensor X)
    {
        Elementwise(X);
        return m_Ops.Relu(X);
    }

    /// <inheritdoc/>
    Tensor IOps.Softmax(Tensor X, int axis)
    {
        Elementwise(X, Transcendental.Exponent);
        return m_Ops.Softmax(X, axis);
    }

    /// <inheritdoc/>
    Tensor IOps.LogSoftmax(Tensor X)
    {
        Elementwise(X, Transcendental.Exponent);
        return m_Ops.LogSoftmax(X);
    }

    /// <inheritdoc/>
    Tensor IOps.Tanh(Tensor X)
    {
        Elementwise(X, Transcendental.Trigonometric);
        return m_Ops.Tanh(X);
    }

    /// <inheritdoc/>
    Tensor IOps.Softplus(Tensor X)
    {
        Elementwise(X, Transcendental.Trigonometric);
        return m_Ops.Softplus(X);
    }

    /// <inheritdoc/>
    Tensor IOps.Sigmoid(Tensor X)
    {
        Elementwise(X, Transcendental.Trigonometric);
        return m_Ops.Sigmoid(X);
    }

    /// <inheritdoc/>
    Tensor IOps.Relu6(Tensor X)
    {
        Elementwise(X, 4L);
        return m_Ops.Relu6(X);
    }

    /// <inheritdoc/>
    Tensor IOps.Elu(Tensor X, float alpha)
    {
        Elementwise(X, Transcendental.Exponent);
        return m_Ops.Elu(X, alpha);
    }

    /// <inheritdoc/>
    Tensor IOps.LeakyRelu(Tensor X, float alpha)
    {
        Elementwise(X, 4L);
        return m_Ops.LeakyRelu(X, alpha);
    }

    /// <inheritdoc/>
    Tensor IOps.Selu(Tensor X, float alpha, float gamma)
    {
        Elementwise(X, Transcendental.Exponent);
        return m_Ops.Selu(X, alpha, gamma);
    }

    /// <inheritdoc/>
    Tensor IOps.PRelu(Tensor X, Tensor S)
    {
        Elementwise(X, 4L);
        return m_Ops.PRelu(X, S);
    }

    /// <inheritdoc/>
    Tensor IOps.Swish(Tensor X)
    {
        Elementwise(X, Transcendental.Trigonometric);
        return m_Ops.Swish(X);
    }

    /// <inheritdoc/>
    Tensor IOps.Abs(Tensor X)
    {
        Elementwise(X);
        return m_Ops.Abs(X);
    }

    /// <inheritdoc/>
    Tensor IOps.Neg(Tensor X)
    {
        Elementwise(X);
        return m_Ops.Neg(X);
    }

    /// <inheritdoc/>
    Tensor IOps.Ceil(Tensor X)
    {
        Elementwise(X);
        return m_Ops.Ceil(X);
    }

    /// <inheritdoc/>
    Tensor IOps.Clip(Tensor X, float min, float max)
    {
        Elementwise(X, 2L);
        return m_Ops.Clip(X, min, max);
    }

    /// <inheritdoc/>
    Tensor IOps.Floor(Tensor X)
    {
        Elementwise(X);
        return m_Ops.Floor(X);
    }

    /// <inheritdoc/>
    Tensor IOps.Reciprocal(Tensor X)
    {
        Elementwise(X, Transcendental.Reciprocal);
        return m_Ops.Reciprocal(X);
    }

    /// <inheritdoc/>
    Tensor IOps.Pow(Tensor X, float alpha)
    {
        Elementwise(X, Transcendental.Pow);
        return m_Ops.Pow(X, alpha);
    }

    /// <inheritdoc/>
    Tensor IOps.Exp(Tensor X)
    {
        Elementwise(X, Transcendental.Exponent);
        return m_Ops.Exp(X);
    }

    /// <inheritdoc/>
    Tensor IOps.Log(Tensor X)
    {
        Elementwise(X, Transcendental.Exponent);
        return m_Ops.Log(X);
    }

    /// <inheritdoc/>
    Tensor IOps.Sqrt(Tensor X)
    {
        Elementwise(X, Transcendental.Root);
        return m_Ops.Sqrt(X);
    }

    /// <inheritdoc/>
    Tensor IOps.Acos(Tensor X)
    {
        Elementwise(X, Transcendental.Trigonometric);
        return m_Ops.Acos(X);
    }

    /// <inheritdoc/>
    Tensor IOps.Acosh(Tensor X)
    {
        Elementwise(X, Transcendental.Exponent + 1 + Transcendental.Root + 3);
        return m_Ops.Acosh(X);
    }

    /// <inheritdoc/>
    Tensor IOps.Asin(Tensor X)
    {
        Elementwise(X, Transcendental.Trigonometric);
        return m_Ops.Asin(X);
    }

    /// <inheritdoc/>
    Tensor IOps.Asinh(Tensor X)
    {
        Elementwise(X, Transcendental.Exponent + 1 + Transcendental.Root + 3);
        return m_Ops.Asinh(X);
    }

    /// <inheritdoc/>
    Tensor IOps.Atan(Tensor X)
    {
        Elementwise(X, Transcendental.Trigonometric);
        return m_Ops.Atan(X);
    }

    /// <inheritdoc/>
    Tensor IOps.Atanh(Tensor X)
    {
        Elementwise(X, 1 + Transcendental.Exponent + 2 + Transcendental.Div);
        return m_Ops.Atanh(X);
    }

    /// <inheritdoc/>
    Tensor IOps.Cos(Tensor X)
    {
        Elementwise(X, Transcendental.Trigonometric);
        return m_Ops.Cos(X);
    }

    /// <inheritdoc/>
    Tensor IOps.Cosh(Tensor X)
    {
        Elementwise(X, 2 + 2*Transcendental.Exponent);
        return m_Ops.Cosh(X);
    }

    /// <inheritdoc/>
    Tensor IOps.Sin(Tensor X)
    {
        Elementwise(X, Transcendental.Trigonometric);
        return m_Ops.Sin(X);
    }

    /// <inheritdoc/>
    Tensor IOps.Sinh(Tensor X)
    {
        Elementwise(X, 2 + 2*Transcendental.Exponent);
        return m_Ops.Sinh(X);
    }

    /// <inheritdoc/>
    Tensor IOps.Tan(Tensor X)
    {
        Elementwise(X, Transcendental.Trigonometric);
        return m_Ops.Tan(X);
    }

    /// <inheritdoc/>
    Tensor IOps.Add(Tensor[] tensors)
    {
        var O = m_Ops.Add(tensors);
        ElementwiseBroadcast(tensors, O);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Sub(Tensor[] tensors)
    {
        var O = m_Ops.Sub(tensors);
        ElementwiseBroadcast(tensors, O);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Mul(Tensor[] tensors)
    {
        var O = m_Ops.Mul(tensors);
        ElementwiseBroadcast(tensors, O);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Div(Tensor[] tensors)
    {
        var O = m_Ops.Div(tensors);
        ElementwiseBroadcast(tensors, O, Transcendental.Div);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Pow(Tensor[] tensors)
    {
        var O = m_Ops.Pow(tensors);
        ElementwiseBroadcast(tensors, O, Transcendental.Pow);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Min(Tensor[] tensors)
    {
        var O = m_Ops.Min(tensors);
        ElementwiseBroadcast(tensors, O);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Max(Tensor[] tensors)
    {
        var O = m_Ops.Max(tensors);
        ElementwiseBroadcast(tensors, O);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Mean(Tensor[] tensors)
    {
        var O = m_Ops.Mean(tensors);
        ElementwiseBroadcast(tensors, O);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.ArgMax(Tensor X, int axis)
    {
        var O = m_Ops.ArgMax(X, axis);
        Reduce(X, O);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.ArgMin(Tensor X, int axis)
    {
        var O = m_Ops.ArgMin(X, axis);
        Reduce(X, O);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.ReduceMax(Tensor X, int axis)
    {
        var O = m_Ops.ReduceMax(X, axis);
        Reduce(X, O);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.ReduceMean(Tensor X, int axis)
    {
        var O = m_Ops.ReduceMean(X, axis);
        Reduce(X, O);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.ReduceMin(Tensor X, int axis)
    {
        var O = m_Ops.ReduceMin(X, axis);
        Reduce(X, O);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.ReduceProd(Tensor X, int axis)
    {
        var O = m_Ops.ReduceProd(X, axis);
        Reduce(X, O);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.ReduceSum(Tensor X, int axis)
    {
        var O = m_Ops.ReduceSum(X, axis);
        Reduce(X, O);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Greater(Tensor a, Tensor b)
    {
        var O = m_Ops.Greater(a, b);
        Elementwise(O);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.GreaterEqual(Tensor a, Tensor b)
    {
        var O = m_Ops.GreaterEqual(a, b);
        Elementwise(O);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Less(Tensor a, Tensor b)
    {
        var O = m_Ops.Less(a, b);
        Elementwise(O);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.LessEqual(Tensor a, Tensor b)
    {
        var O = m_Ops.LessEqual(a, b);
        Elementwise(O);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Equal(Tensor a, Tensor b)
    {
        var O = m_Ops.Equal(a, b);
        Elementwise(O);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.LogicalOr(Tensor a, Tensor b)
    {
        var O = m_Ops.LogicalOr(a, b);
        Elementwise(O);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.LogicalAnd(Tensor a, Tensor b)
    {
        var O = m_Ops.LogicalAnd(a, b);
        Elementwise(O);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.LogicalXor(Tensor a, Tensor b)
    {
        var O = m_Ops.LogicalXor(a, b);
        Elementwise(O);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.LogicalNot(Tensor x)
    {
        var O = m_Ops.LogicalNot(x);
        Elementwise(O);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Where(Tensor c, Tensor a, Tensor b)
    {
        var O = m_Ops.Where(c, a, b);
        Elementwise(O);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Flatten(Tensor X)
    {
        return m_Ops.Flatten(X);
    }

    /// <inheritdoc/>
    Tensor IOps.Reshape(Tensor X, TensorShape shape)
    {
        return m_Ops.Reshape(X, shape);
    }

    /// <inheritdoc/>
    Tensor IOps.Expand(Tensor X, TensorShape shape)
    {
        var O = m_Ops.Expand(X, shape);
        m_Mem += (long)X.length + (long)O.length;
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Transpose(Tensor X)
    {
        Elementwise(X);
        return m_Ops.Transpose(X);
    }

    /// <inheritdoc/>
    Tensor IOps.Transpose(Tensor X, int[] permutations)
    {
        Elementwise(X);
        return m_Ops.Transpose(X, permutations);
    }

    /// <inheritdoc/>
    Tensor IOps.Gather(Tensor[] tensors, int axis)
    {
        var O =  m_Ops.Gather(tensors, axis);
        Elementwise(O);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.NonMaxSuppression(Tensor[] tensors, int maxOutputBoxesPerClass, float iouThreshold, float scoreThreshold, int centerPointBox)
    {
        var O = m_Ops.NonMaxSuppression(tensors, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, centerPointBox);
        // @TODO: not implemented
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Concat(Tensor[] tensors, int axis)
    {
        var O = m_Ops.Concat(tensors, axis);
        Elementwise(O);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.StridedSlice(Tensor X, int[] starts, int[] ends, int[] strides)
    {
        var O = m_Ops.StridedSlice(X, starts, ends, strides);
        Elementwise(O);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Tile(Tensor X, int[] repeats)
    {
        var O = m_Ops.Tile(X, repeats);
        Elementwise(O);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Shape(Tensor X, int axis)
    {
        var O = m_Ops.Shape(X, axis);
        Elementwise(O);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Copy(Tensor x)
    {
        var O = m_Ops.Copy(x);
        Elementwise(O);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Prepare(Tensor X)
    {
        return m_Ops.Prepare(X);
    }

    /// <inheritdoc/>
    void IOps.ResetAllocator(bool keepCachedMemory)
    {
        m_Ops.ResetAllocator(keepCachedMemory);
        m_Alu = 0;
        m_Mem = 0;
    }

    /// <summary>
    /// Build execution summary
    /// </summary>
    /// <returns>execution summary</returns>
    public override string ToString()
    {
        string alu = m_Alu.ToString();
        if (m_Alu > 1e12)
            alu = $"{(double)m_Alu / (1e12):###.0}T";
        else if (m_Alu > 1e9)
            alu = $"{(double)m_Alu / (1e9):###.0}G";
        else if (m_Alu > 1e6)
            alu = $"{(double)m_Alu / (1e6):###.0}M";

        var mem4 = m_Mem * 4L;
        string mem = mem4.ToString();
        if (mem4 > 1024*1024*1024)
            mem = $"{(double)mem4 / (1024*1024*1024):###.0}Gb";
        else if (mem4 > 1024*1024)
            mem = $"{(double)mem4 / (1024*1024):###.0}Mb";
        return $"ALU operations: {alu} bytes accessed: {mem}";
    }

    // -----
    internal void Elementwise(Tensor X, long aluOperationsPerElement = 1L)
    {
        m_Alu += (long)X.length * aluOperationsPerElement;
        m_Mem += (long)X.length * 2L;
    }

    internal void ElementwiseBroadcast(Tensor[] tensors, Tensor X, long aluOperationsPerElement = 1L)
    {
        m_Alu += (long)X.length * aluOperationsPerElement;
        m_Mem += (long)X.length;
        foreach (var t in tensors)
            m_Mem += (long)t.length;
    }

    internal void Reduce(Tensor X, Tensor O, long aluOperationsPerElement = 1L)
    {
        m_Alu += (long)X.length * aluOperationsPerElement;
        m_Mem += (long)X.length + (long)O.length;
    }
}


} // namespace Unity.Barracuda
