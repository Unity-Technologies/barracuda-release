using System.Collections.Generic;

namespace Unity.Barracuda {

/// <summary>
/// Compares output of two different implementations of `IOps`. Useful for debugging purposes
/// </summary>
public class CompareOps : IOps, IModelCompiler
{
    private readonly IOps m_Ops1;
    private readonly IOps m_Ops2;
    private readonly CompareOpsUtils.LogLevel m_DifferenceLogLevel;
    private readonly float m_Epsilon;

    /// <summary>
    /// Create `CompareOps`
    /// </summary>
    /// <param name="ops1">first `IOps` implementation</param>
    /// <param name="ops2">second `IOps` implementation</param>
    /// <param name="differenceLogLevel">difference log level</param>
    /// <param name="epsilon">error threshold</param>
    public CompareOps(IOps ops1, IOps ops2, CompareOpsUtils.LogLevel differenceLogLevel, float epsilon)
    {
        m_Ops1 = ops1;
        m_Ops2 = ops2;
        m_DifferenceLogLevel = differenceLogLevel;
        m_Epsilon = epsilon;
    }

#if ENABLE_BARRACUDA_STATS
    public IEnumerable<TempMemoryStatistics> GetTempMemoryStatistics()
    {
        return m_Ops1.GetTempMemoryStatistics();
    }
#endif //ENABLE_BARRACUDA_STATS

    /// <inheritdoc/>
    public virtual void PostLayerCleanup()
    {
        m_Ops1.PostLayerCleanup();
        m_Ops2.PostLayerCleanup();
    }

    /// <inheritdoc/>
    public virtual void PrepareModel(Model model, IDictionary<string, TensorShape> inputShapes, IVars vars)
    {
        if (m_Ops1 is IModelCompiler)
            ((IModelCompiler)m_Ops1).PrepareModel(model, inputShapes, vars);

        if (m_Ops2 is IModelCompiler)
            ((IModelCompiler)m_Ops2).PrepareModel(model, inputShapes, vars);
    }

    /// <inheritdoc/>
    public virtual void PreExecuteLayer(Layer layer, Tensor[] inputs)
    {
        if (m_Ops1 is IModelCompiler)
            ((IModelCompiler)m_Ops1).PreExecuteLayer(layer, inputs);

        if (m_Ops2 is IModelCompiler)
            ((IModelCompiler)m_Ops1).PreExecuteLayer(layer, inputs);
    }

    /// <inheritdoc/>
    Tensor IOps.MatMul(Tensor X, int rankX, Tensor Y, int rankY)
    {
        var A = m_Ops1.MatMul(X, rankX, Y, rankY);
        var B = m_Ops2.MatMul(X, rankX, Y, rankY);
        CheckSame(A, B, Layer.Type.MatMul);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.MatMul(Tensor X, bool xTranspose, Tensor Y, bool yTranspose)
    {
        var A = m_Ops1.MatMul(X, xTranspose, Y, yTranspose);
        var B = m_Ops2.MatMul(X, xTranspose, Y, yTranspose);
        CheckSame(A, B, Layer.Type.MatMul);
        return A;
    }

    /// <inheritdoc/>
    Tensor IOps.Dense(Tensor X, Tensor W, Tensor B, Layer.FusedActivation fusedActivation)
    {
        var Y = m_Ops1.Dense(X, W, B, fusedActivation);
        var Z = m_Ops2.Dense(X, W, B, fusedActivation);
        CheckSame(Y, Z, Layer.Type.Dense);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Dense3(Tensor X, Tensor W, Tensor B)
    {
        var Y = m_Ops1.Dense3(X, W, B);
        var Z = m_Ops2.Dense3(X, W, B);
        CheckSame(Y, Z, Layer.Type.Dense3);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Conv2D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, Layer.FusedActivation fusedActivation)
    {
        var Y = m_Ops1.Conv2D(X, K, B, stride, pad, fusedActivation);
        var Z = m_Ops2.Conv2D(X, K, B, stride, pad, fusedActivation);
        CheckSame(Y, Z, Layer.Type.Conv2D);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Conv3D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, Layer.FusedActivation fusedActivation)
    {
        var Y = m_Ops1.Conv3D(X, K, B, stride, pad, fusedActivation);
        var Z = m_Ops2.Conv3D(X, K, B, stride, pad, fusedActivation);
        CheckSame(Y, Z, Layer.Type.Conv3D);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.DepthwiseConv2D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, Layer.FusedActivation fusedActivation)
    {
        var Y = m_Ops1.DepthwiseConv2D(X, K, B, stride, pad, fusedActivation);
        var Z = m_Ops2.DepthwiseConv2D(X, K, B, stride, pad, fusedActivation);
        CheckSame(Y, Z, Layer.Type.DepthwiseConv2D);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Conv2DTrans(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, int[] outputAdjustment, Layer.FusedActivation fusedActivation)
    {
        var Y = m_Ops1.Conv2DTrans(X, K, B, stride, pad, outputAdjustment, fusedActivation);
        var Z = m_Ops2.Conv2DTrans(X, K, B, stride, pad, outputAdjustment, fusedActivation);
        CheckSame(Y, Z, Layer.Type.Conv2DTrans);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Upsample2D(Tensor X, int[] scale, bool bilinear)
    {
        var Y = m_Ops1.Upsample2D(X, scale, bilinear);
        var Z = m_Ops2.Upsample2D(X, scale, bilinear);
        CheckSame(Y, Z, Layer.Type.Upsample2D);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Upsample3D(Tensor X, int[] scale, bool trilinear)
    {
        var Y = m_Ops1.Upsample3D(X, scale, trilinear);
        var Z = m_Ops2.Upsample3D(X, scale, trilinear);
        CheckSame(Y, Z, Layer.Type.Upsample3D);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Resample2D(Tensor X, int[] size, bool bilinear)
    {
        var Y = m_Ops1.Resample2D(X, size, bilinear);
        var Z = m_Ops2.Resample2D(X, size, bilinear);
        CheckSame(Y, Z, Layer.Type.Resample2D);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.DepthToSpace(Tensor X, int[] scale, Layer.DepthToSpaceMode mode)
    {
        var Y = m_Ops1.DepthToSpace(X, scale, mode);
        var Z = m_Ops2.DepthToSpace(X, scale, mode);
        CheckSame(Y, Z, Layer.Type.DepthToSpace);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.SpaceToDepth(Tensor X, int[] scale)
    {
        var Y = m_Ops1.SpaceToDepth(X, scale);
        var Z = m_Ops2.SpaceToDepth(X, scale);
        CheckSame(Y, Z, Layer.Type.SpaceToDepth);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.MaxPool2D(Tensor X, int[] pool, int[] stride, int[] pad)
    {
        var Y = m_Ops1.MaxPool2D(X, pool, stride, pad);
        var Z = m_Ops2.MaxPool2D(X, pool, stride, pad);
        CheckSame(Y, Z, Layer.Type.MaxPool2D);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.AvgPool2D(Tensor X, int[] pool, int[] stride, int[] pad)
    {
        var Y = m_Ops1.AvgPool2D(X, pool, stride, pad);
        var Z = m_Ops2.AvgPool2D(X, pool, stride, pad);
        CheckSame(Y, Z, Layer.Type.AvgPool2D);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.GlobalMaxPool2D(Tensor X)
    {
        var Y = m_Ops1.GlobalMaxPool2D(X);
        var Z = m_Ops2.GlobalMaxPool2D(X);
        CheckSame(Y, Z, Layer.Type.GlobalMaxPool2D);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.GlobalAvgPool2D(Tensor X)
    {
        var Y = m_Ops1.GlobalAvgPool2D(X);
        var Z = m_Ops2.GlobalAvgPool2D(X);
        CheckSame(Y, Z, Layer.Type.GlobalAvgPool2D);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.GlobalAvgVariancePool2D(Tensor X)
    {
        var Y = m_Ops1.GlobalAvgVariancePool2D(X);
        var Z = m_Ops2.GlobalAvgVariancePool2D(X);
        CheckSame(Y, Z, Layer.Type.GlobalAvgPool2D);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Border2D(Tensor x, int[] pad, float value)
    {
        var Y = m_Ops1.Border2D(x, pad, value);
        var Z = m_Ops2.Border2D(x, pad, value);
        CheckSame(Y, Z, Layer.Type.Border2D);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Border3D(Tensor x, int[] pad, float value)
    {
        var Y = m_Ops1.Border3D(x, pad, value);
        var Z = m_Ops2.Border3D(x, pad, value);
        CheckSame(Y, Z, Layer.Type.Border3D);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Pad2DReflect(Tensor x, int[] pad)
    {
        var Y = m_Ops1.Pad2DReflect(x, pad);
        var Z = m_Ops2.Pad2DReflect(x, pad);
        CheckSame(Y, Z, Layer.Type.Pad2DReflect);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Pad2DSymmetric(Tensor x, int[] pad)
    {
        var Y = m_Ops1.Pad2DSymmetric(x, pad);
        var Z = m_Ops2.Pad2DSymmetric(x, pad);
        CheckSame(Y, Z, Layer.Type.Pad2DSymmetric);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Pad2DEdge(Tensor x, int[] pad)
    {
        var Y = m_Ops1.Pad2DEdge(x, pad);
        var Z = m_Ops2.Pad2DEdge(x, pad);
        CheckSame(Y, Z, Layer.Type.Pad2DEdge);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.ScaleBias(Tensor X, Tensor S, Tensor B)
    {
        var Y = m_Ops1.ScaleBias(X, S, B);
        var Z = m_Ops2.ScaleBias(X, S, B);
        CheckSame(Y, Z, Layer.Type.ScaleBias);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Normalization(Tensor X, Tensor S, Tensor B, int pool, int axis, float epsilon, Layer.FusedActivation fusedActivation)
    {
        var Y = m_Ops1.Normalization(X, S, B, pool, axis, epsilon, fusedActivation);
        var Z = m_Ops2.Normalization(X, S, B, pool, axis, epsilon, fusedActivation);
        CheckSame(Y, Z, Layer.Type.Normalization);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.LRN(Tensor X, float alpha, float beta, float bias, int size)
    {
        var Y = m_Ops1.LRN(X, alpha, beta, bias, size);
        var Z = m_Ops2.LRN(X, alpha, beta, bias, size);
        CheckSame(Y, Z, Layer.Type.LRN);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Dropout(Tensor X, float alpha)
    {
        var Y = m_Ops1.Dropout(X, alpha);
        var Z = m_Ops2.Dropout(X, alpha);
        CheckSame(Y, Z, Layer.Type.Dropout);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.RandomNormal(TensorShape s, float mean, float scale, int seed)
    {
        var Y = m_Ops1.RandomNormal(s, mean, scale, seed);
        var Z = m_Ops2.RandomNormal(s, mean, scale, seed);
        CheckSame(Y, Z, Layer.Type.RandomNormal);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.RandomUniform(TensorShape s, float mean, float scale, int seed)
    {
        var Y = m_Ops1.RandomUniform(s, mean, scale, seed);
        var Z = m_Ops2.RandomUniform(s, mean, scale, seed);
        CheckSame(Y, Z, Layer.Type.RandomUniform);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Multinomial(Tensor X, int count, int seed)
    {
        var Y = m_Ops1.Multinomial(X, count, seed);
        var Z = m_Ops2.Multinomial(X, count, seed);
        CheckSame(Y, Z, Layer.Type.Multinomial);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.OneHot(Tensor X, int depth, float onValue, float offValue, int inputRank)
    {
        var Y = m_Ops1.OneHot(X, depth, onValue, offValue, inputRank);
        var Z = m_Ops2.OneHot(X, depth, onValue, offValue, inputRank);
        CheckSame(Y, Z, Layer.Type.OneHot);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.RoiAlign(Tensor X, Tensor rois, Tensor indices, int outputHeight, int outputWidth, int samplingRatio, float spatialScale)
    {
        var Y = m_Ops1.RoiAlign(X, rois, indices, outputHeight, outputWidth, samplingRatio, spatialScale);
        var Z = m_Ops2.RoiAlign(X, rois, indices, outputHeight, outputWidth, samplingRatio, spatialScale);
        CheckSame(Y, Z, Layer.Type.RoiAlign);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.TopKIndices(Tensor X, int k, int axis, bool largest, bool sorted)
    {
        var Y = m_Ops1.TopKIndices(X, k, axis, largest, sorted);
        var Z = m_Ops2.TopKIndices(X, k, axis, largest, sorted);
        CheckSame(Y, Z, Layer.Type.TopKIndices);
        return Y;
    }

    /// <inheritdoc/>
    public Tensor TopKValues(Tensor X, Tensor I, int axis)
    {
        var Y = m_Ops1.TopKValues(X, I, axis);
        var Z = m_Ops2.TopKValues(X, I, axis);
        CheckSame(Y, Z, Layer.Type.TopKValues);
        return Y;
    }

    /// <inheritdoc/>
    public Tensor NonZero(Tensor X)
    {
        var Y = m_Ops1.NonZero(X);
        var Z = m_Ops2.NonZero(X);
        CheckSame(Y, Z, Layer.Type.NonZero);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Relu(Tensor X)
    {
        var Y = m_Ops1.Relu(X);
        var Z = m_Ops2.Relu(X);
        CheckSame(Y, Z, Layer.Type.Activation + " " + Layer.Activation.Relu);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Softmax(Tensor X, int axis)
    {
        var Y = m_Ops1.Softmax(X, axis);
        var Z = m_Ops2.Softmax(X, axis);
        CheckSame(Y, Z, Layer.Type.Activation + " " + Layer.Activation.Softmax);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.LogSoftmax(Tensor X, int axis)
    {
        var Y = m_Ops1.LogSoftmax(X, axis);
        var Z = m_Ops2.LogSoftmax(X, axis);
        CheckSame(Y, Z, Layer.Type.Activation + " " + Layer.Activation.LogSoftmax);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Tanh(Tensor X)
    {
        var Y = m_Ops1.Tanh(X);
        var Z = m_Ops2.Tanh(X);
        CheckSame(Y, Z, Layer.Type.Activation + " " + Layer.Activation.Tanh);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Softplus(Tensor X)
    {
        var Y = m_Ops1.Softplus(X);
        var Z = m_Ops2.Softplus(X);
        CheckSame(Y, Z, Layer.Type.Activation + " " + Layer.Activation.Softplus);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Sigmoid(Tensor X)
    {
        var Y = m_Ops1.Sigmoid(X);
        var Z = m_Ops2.Sigmoid(X);
        CheckSame(Y, Z, Layer.Type.Activation + " " + Layer.Activation.Sigmoid);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.HardSigmoid(Tensor X, float alpha, float beta)
    {
        var Y = m_Ops1.HardSigmoid(X, alpha, beta);
        var Z = m_Ops2.HardSigmoid(X, alpha, beta);
        CheckSame(Y, Z, Layer.Type.Activation + " " + Layer.Activation.HardSigmoid);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Elu(Tensor X, float alpha)
    {
        var Y = m_Ops1.Elu(X, alpha);
        var Z = m_Ops2.Elu(X, alpha);
        CheckSame(Y, Z, Layer.Type.Activation + " " + Layer.Activation.Elu);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Relu6(Tensor X)
    {
        var Y = m_Ops1.Relu6(X);
        var Z = m_Ops2.Relu6(X);
        CheckSame(Y, Z, Layer.Type.Activation + " " + Layer.Activation.Relu6);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.LeakyRelu(Tensor X, float alpha)
    {
        var Y = m_Ops1.LeakyRelu(X, alpha);
        var Z = m_Ops2.LeakyRelu(X, alpha);
        CheckSame(Y, Z, Layer.Type.Activation + " " + Layer.Activation.LeakyRelu);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Selu(Tensor X, float alpha, float gamma)
    {
        var Y = m_Ops1.Selu(X, alpha, gamma);
        var Z = m_Ops2.Selu(X, alpha, gamma);
        CheckSame(Y, Z, Layer.Type.Activation + " " + Layer.Activation.Selu);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.PRelu(Tensor X, Tensor S)
    {
        var Y = m_Ops1.PRelu(X, S);
        var Z = m_Ops2.PRelu(X, S);
        CheckSame(Y, Z, Layer.Type.Activation + " " + Layer.Activation.PRelu);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Swish(Tensor X)
    {
        var Y = m_Ops1.Swish(X);
        var Z = m_Ops2.Swish(X);
        CheckSame(Y, Z, Layer.Type.Activation + " " + Layer.Activation.Swish);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Abs(Tensor X)
    {
        var Y = m_Ops1.Abs(X);
        var Z = m_Ops2.Abs(X);
        CheckSame(Y, Z, Layer.Type.Activation + " " + Layer.Activation.Abs);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Neg(Tensor X)
    {
        var Y = m_Ops1.Neg(X);
        var Z = m_Ops2.Neg(X);
        CheckSame(Y, Z, Layer.Type.Activation + " " + Layer.Activation.Neg);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Ceil(Tensor X)
    {
        var Y = m_Ops1.Ceil(X);
        var Z = m_Ops2.Ceil(X);
        CheckSame(Y, Z, Layer.Type.Activation + " " + Layer.Activation.Ceil);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Clip(Tensor X, float min, float max)
    {
        var Y = m_Ops1.Clip(X, min, max);
        var Z = m_Ops2.Clip(X, min, max);
        CheckSame(Y, Z, Layer.Type.Activation + " " + Layer.Activation.Clip);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Floor(Tensor X)
    {
        var Y = m_Ops1.Floor(X);
        var Z = m_Ops2.Floor(X);
        CheckSame(Y, Z, Layer.Type.Activation + " " + Layer.Activation.Floor);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Round(Tensor X)
    {
        var Y = m_Ops1.Round(X);
        var Z = m_Ops2.Round(X);
        CheckSame(Y, Z, Layer.Type.Activation + " " + Layer.Activation.Round);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Reciprocal(Tensor X)
    {
        var Y = m_Ops1.Reciprocal(X);
        var Z = m_Ops2.Reciprocal(X);
        CheckSame(Y, Z, Layer.Type.Activation + " " + Layer.Activation.Reciprocal);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Pow(Tensor X, float alpha)
    {
        var Y = m_Ops1.Pow(X, alpha);
        var Z = m_Ops2.Pow(X, alpha);
        CheckSame(Y, Z, Layer.Type.Activation + " " + Layer.Activation.Pow);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Exp(Tensor X)
    {
        var Y = m_Ops1.Exp(X);
        var Z = m_Ops2.Exp(X);
        CheckSame(Y, Z, Layer.Type.Activation + " " + Layer.Activation.Exp);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Log(Tensor X)
    {
        var Y = m_Ops1.Log(X);
        var Z = m_Ops2.Log(X);
        CheckSame(Y, Z, Layer.Type.Activation + " " + Layer.Activation.Log);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Sqrt(Tensor X)
    {
        var Y = m_Ops1.Sqrt(X);
        var Z = m_Ops2.Sqrt(X);
        CheckSame(Y, Z, Layer.Type.Activation + " " + Layer.Activation.Sqrt);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Acos(Tensor X)
    {
        var Y = m_Ops1.Acos(X);
        var Z = m_Ops2.Acos(X);
        CheckSame(Y, Z, Layer.Type.Activation + " " + Layer.Activation.Acos);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Acosh(Tensor X)
    {
        var Y = m_Ops1.Acosh(X);
        var Z = m_Ops2.Acosh(X);
        CheckSame(Y, Z, Layer.Type.Activation + " " + Layer.Activation.Acosh);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Asin(Tensor X)
    {
        var Y = m_Ops1.Asin(X);
        var Z = m_Ops2.Asin(X);
        CheckSame(Y, Z, Layer.Type.Activation + " " + Layer.Activation.Asin);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Asinh(Tensor X)
    {
        var Y = m_Ops1.Asinh(X);
        var Z = m_Ops2.Asinh(X);
        CheckSame(Y, Z, Layer.Type.Activation + " " + Layer.Activation.Asinh);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Atan(Tensor X)
    {
        var Y = m_Ops1.Atan(X);
        var Z = m_Ops2.Atan(X);
        CheckSame(Y, Z, Layer.Type.Activation + " " + Layer.Activation.Atan);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Atanh(Tensor X)
    {
        var Y = m_Ops1.Atanh(X);
        var Z = m_Ops2.Atanh(X);
        CheckSame(Y, Z, Layer.Type.Activation + " " + Layer.Activation.Atanh);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Cos(Tensor X)
    {
        var Y = m_Ops1.Cos(X);
        var Z = m_Ops2.Cos(X);
        CheckSame(Y, Z, Layer.Type.Activation + " " + Layer.Activation.Cos);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Cosh(Tensor X)
    {
        var Y = m_Ops1.Cosh(X);
        var Z = m_Ops2.Cosh(X);
        CheckSame(Y, Z, Layer.Type.Activation + " " + Layer.Activation.Cosh);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Sin(Tensor X)
    {
        var Y = m_Ops1.Sin(X);
        var Z = m_Ops2.Sin(X);
        CheckSame(Y, Z, Layer.Type.Activation + " " + Layer.Activation.Sin);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Sinh(Tensor X)
    {
        var Y = m_Ops1.Sinh(X);
        var Z = m_Ops2.Sinh(X);
        CheckSame(Y, Z, Layer.Type.Activation + " " + Layer.Activation.Sinh);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Tan(Tensor X)
    {
        var Y = m_Ops1.Tan(X);
        var Z = m_Ops2.Tan(X);
        CheckSame(Y, Z, Layer.Type.Activation + " " + Layer.Activation.Tan);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Erf(Tensor X)
    {
        var Y = m_Ops1.Erf(X);
        var Z = m_Ops2.Erf(X);
        CheckSame(Y, Z, Layer.Type.Activation + " " + Layer.Activation.Erf);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Add(Tensor[] tensors)
    {
        var Y = m_Ops1.Add(tensors);
        var Z = m_Ops2.Add(tensors);
        CheckSame(Y, Z, Layer.Type.Add);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Sub(Tensor[] tensors)
    {
        var Y = m_Ops1.Sub(tensors);
        var Z = m_Ops2.Sub(tensors);
        CheckSame(Y, Z, Layer.Type.Sub);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Mul(Tensor[] tensors)
    {
        var Y = m_Ops1.Mul(tensors);
        var Z = m_Ops2.Mul(tensors);
        CheckSame(Y, Z, Layer.Type.Mul, tensors);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Div(Tensor[] tensors)
    {
        var Y = m_Ops1.Div(tensors);
        var Z = m_Ops2.Div(tensors);
        CheckSame(Y, Z, Layer.Type.Div);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Pow(Tensor[] tensors)
    {
        var Y = m_Ops1.Pow(tensors);
        var Z = m_Ops2.Pow(tensors);
        CheckSame(Y, Z, Layer.Type.Pow);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Min(Tensor[] tensors)
    {
        var Y = m_Ops1.Min(tensors);
        var Z = m_Ops2.Min(tensors);
        CheckSame(Y, Z, Layer.Type.Min);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Max(Tensor[] tensors)
    {
        var Y = m_Ops1.Max(tensors);
        var Z = m_Ops2.Max(tensors);
        CheckSame(Y, Z, Layer.Type.Max);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Mean(Tensor[] tensors)
    {
        var Y = m_Ops1.Mean(tensors);
        var Z = m_Ops2.Mean(tensors);
        CheckSame(Y, Z, Layer.Type.Mean);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.ArgMax(Tensor X, int axis)
    {
        var Y = m_Ops1.ArgMax(X, axis);
        var Z = m_Ops2.ArgMax(X, axis);
        CheckSame(Y, Z, Layer.Type.ArgMax);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.ArgMin(Tensor X, int axis)
    {
        var Y = m_Ops1.ArgMin(X, axis);
        var Z = m_Ops2.ArgMin(X, axis);
        CheckSame(Y, Z, Layer.Type.ArgMin);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.ReduceMax(Tensor X, int axis)
    {
        var Y = m_Ops1.ReduceMax(X, axis);
        var Z = m_Ops2.ReduceMax(X, axis);
        CheckSame(Y, Z, Layer.Type.ReduceMax);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.ReduceMean(Tensor X, int axis)
    {
        var Y = m_Ops1.ReduceMean(X, axis);
        var Z = m_Ops2.ReduceMean(X, axis);
        CheckSame(Y, Z, Layer.Type.ReduceMean);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.ReduceMin(Tensor X, int axis)
    {
        var Y = m_Ops1.ReduceMin(X, axis);
        var Z = m_Ops2.ReduceMin(X, axis);
        CheckSame(Y, Z, Layer.Type.ReduceMin);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.ReduceProd(Tensor X, int axis)
    {
        var Y = m_Ops1.ReduceProd(X, axis);
        var Z = m_Ops2.ReduceProd(X, axis);
        CheckSame(Y, Z, Layer.Type.ReduceProd);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.ReduceSum(Tensor X, int axis)
    {
        var Y = m_Ops1.ReduceSum(X, axis);
        var Z = m_Ops2.ReduceSum(X, axis);
        CheckSame(Y, Z, Layer.Type.ReduceSum);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Greater(Tensor a, Tensor b)
    {
        var Y = m_Ops1.Greater(a, b);
        var Z = m_Ops2.Greater(a, b);
        CheckSame(Y, Z, Layer.Type.Greater);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.GreaterEqual(Tensor a, Tensor b)
    {
        var Y = m_Ops1.GreaterEqual(a, b);
        var Z = m_Ops2.GreaterEqual(a, b);
        CheckSame(Y, Z, Layer.Type.GreaterEqual);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Less(Tensor a, Tensor b)
    {
        var Y = m_Ops1.Less(a, b);
        var Z = m_Ops2.Less(a, b);
        CheckSame(Y, Z, Layer.Type.Less);
        return Y;

    }

    /// <inheritdoc/>
    Tensor IOps.LessEqual(Tensor a, Tensor b)
    {
        var Y = m_Ops1.LessEqual(a, b);
        var Z = m_Ops2.LessEqual(a, b);
        CheckSame(Y, Z, Layer.Type.LessEqual);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Equal(Tensor a, Tensor b)
    {
        var Y = m_Ops1.Equal(a, b);
        var Z = m_Ops2.Equal(a, b);
        CheckSame(Y, Z, Layer.Type.Equal);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.LogicalOr(Tensor a, Tensor b)
    {
        var Y = m_Ops1.LogicalOr(a, b);
        var Z = m_Ops2.LogicalOr(a, b);
        CheckSame(Y, Z, Layer.Type.LogicalOr);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.LogicalAnd(Tensor a, Tensor b)
    {
        var Y = m_Ops1.LogicalAnd(a, b);
        var Z = m_Ops2.LogicalAnd(a, b);
        CheckSame(Y, Z, Layer.Type.LogicalAnd);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.LogicalXor(Tensor a, Tensor b)
    {
        var Y = m_Ops1.LogicalXor(a, b);
        var Z = m_Ops2.LogicalXor(a, b);
        CheckSame(Y, Z, Layer.Type.LogicalXor);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.LogicalNot(Tensor x)
    {
        var Y = m_Ops1.LogicalNot(x);
        var Z = m_Ops2.LogicalNot(x);
        CheckSame(Y, Z, Layer.Type.LogicalNot);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Sign(Tensor x)
    {
        var Y = m_Ops1.Sign(x);
        var Z = m_Ops2.Sign(x);
        CheckSame(Y, Z, Layer.Type.Sign);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Where(Tensor c, Tensor a, Tensor b)
    {
        var Y = m_Ops1.Where(c, a, b);
        var Z = m_Ops2.Where(c, a, b);
        CheckSame(Y, Z, Layer.Type.Where);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Flatten(Tensor X)
    {
        var Y = m_Ops1.Flatten(X);
        var Z = m_Ops2.Flatten(X);
        CheckSame(Y, Z, Layer.Type.Flatten);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Reshape(Tensor X, TensorShape shape)
    {
        var Y = m_Ops1.Reshape(X, shape);
        var Z = m_Ops2.Reshape(X, shape);
        CheckSame(Y, Z, Layer.Type.Reshape);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Expand(Tensor X, TensorShape shape)
    {
        var Y = m_Ops1.Expand(X, shape);
        var Z = m_Ops2.Expand(X, shape);
        CheckSame(Y, Z, Layer.Type.Expand);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Transpose(Tensor X)
    {
        var Y = m_Ops1.Transpose(X);
        var Z = m_Ops2.Transpose(X);
        CheckSame(Y, Z, Layer.Type.Transpose);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Transpose(Tensor X, int[] permutations)
    {
        var Y = m_Ops1.Transpose(X, permutations);
        var Z = m_Ops2.Transpose(X, permutations);
        CheckSame(Y, Z, Layer.Type.Transpose);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Gather(Tensor[] tensors, int axis)
    {
        var Y = m_Ops1.Gather(tensors, axis);
        var Z = m_Ops2.Gather(tensors, axis);
        CheckSame(Y, Z, Layer.Type.Gather);
        return Y;
    }

    // <inheritdoc/>
    Tensor IOps.ScatterND(Tensor X, Tensor indices, Tensor updates, Layer.ScatterNDReductionMode reduction)
    {
        var Y = m_Ops1.ScatterND(X, indices, updates, reduction);
        var Z = m_Ops2.ScatterND(X, indices, updates, reduction);
        CheckSame(Y, Z, Layer.Type.ScatterND);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.NonMaxSuppression(Tensor[] tensors, int maxOutputBoxesPerClass, float iouThreshold, float scoreThreshold, int centerPointBox)
    {
        var Y = m_Ops1.NonMaxSuppression(tensors, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, centerPointBox);
        var Z = m_Ops2.NonMaxSuppression(tensors, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, centerPointBox);
        CheckSame(Y, Z, Layer.Type.NonMaxSuppression);
        return Y;
    }

    /// <inheritdoc/>
    public Tensor[] LSTM(Tensor X, Tensor[] W, Tensor[] R, Tensor[] Wb, Tensor[] Rb, Tensor hidden, Tensor cell)
    {
        var Y = m_Ops1.LSTM(X, W, R, Wb, Rb, hidden, cell);
        var Z = m_Ops2.LSTM(X, W, R, Wb, Rb, hidden, cell);
        for (int i = 0; i < Y.Length; i++)
        {
            CheckSame(Y[i], Z[i], Layer.Type.LSTM);
        }

        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Concat(Tensor[] tensors, int axis)
    {
        var Y = m_Ops1.Concat(tensors, axis);
        var Z = m_Ops2.Concat(tensors, axis);
        CheckSame(Y, Z, Layer.Type.Concat);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.StridedSlice(Tensor X, int[] starts, int[] ends, int[] strides)
    {
        var Y = m_Ops1.StridedSlice(X, starts, ends, strides);
        var Z = m_Ops2.StridedSlice(X, starts, ends, strides);
        CheckSame(Y, Z, Layer.Type.StridedSlice);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Tile(Tensor X, int[] repeats)
    {
        var Y = m_Ops1.Tile(X, repeats);
        var Z = m_Ops2.Tile(X, repeats);
        CheckSame(Y, Z, Layer.Type.Tile);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Shape(Tensor X, int axis)
    {
        var Y = m_Ops1.Shape(X, axis);
        var Z = m_Ops2.Shape(X, axis);
        CheckSame(Y, Z, Layer.Type.Shape);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.ConstantOfShape(TensorShape X, DataType type, float value)
    {
        var Y = m_Ops1.ConstantOfShape(X, type, value);
        var Z = m_Ops2.ConstantOfShape(X, type, value);
        CheckSame(Y, Z, Layer.Type.ConstantOfShape);
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Copy(Tensor x)
    {
        var Y = m_Ops1.Copy(x);
        var Z = m_Ops2.Copy(x);
        CheckSame(Y, Z, "Copy");
        return Y;
    }

    /// <inheritdoc/>
    Tensor IOps.Prepare(Tensor X)
    {
        var Y = m_Ops1.Prepare(X);
        var Z = m_Ops2.Prepare(X);
        CheckSame(Y, Z, "Prepare");
        return Y;
    }


    /// <inheritdoc/>

    Tensor IOps.PrepareNoAlloc(Tensor X)
    {
        var Y = m_Ops1.PrepareNoAlloc(X);
        var Z = m_Ops2.PrepareNoAlloc(X);
        CheckSame(Y, Z, "PrepareNoAlloc");
        return Y;
    }

    /// <inheritdoc/>
    void IOps.ResetAllocator(bool keepCachedMemory)
    {
        m_Ops1.ResetAllocator(keepCachedMemory);
        m_Ops2.ResetAllocator(keepCachedMemory);
    }

    /// <inheritdoc/>
    void IOps.SetModelExecutionsReporter(IModelExecutionsReporter executionsReporter)
    {
        m_Ops1.SetModelExecutionsReporter(executionsReporter);
        m_Ops2.SetModelExecutionsReporter(null);
    }

    /// <inheritdoc/>
    IModelExecutionsReporter IOps.GetModelExecutionsReporter()
    {
        return m_Ops1.GetModelExecutionsReporter();
    }

    private void CheckSame(Tensor X, Tensor Y, Layer.Type layerType, params Tensor[] inputs)
    {
        CompareOpsUtils.CheckSame(X, Y, layerType, m_DifferenceLogLevel, m_Epsilon, inputs);
    }

    private void CheckSame(Tensor X, Tensor Y, string opName, params Tensor[] inputs)
    {
        CompareOpsUtils.CheckSame(X, Y, opName, m_DifferenceLogLevel, m_Epsilon, inputs);
    }
}


} // namespace Unity.Barracuda
