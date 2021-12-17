using System.Collections.Generic;
using System.Linq;

namespace Unity.Barracuda {

    /// <summary>
    /// Verbose proxy to other `IOps` implementation
    /// </summary>
public class VerboseOps : IOps, IModelCompiler
{
    private bool m_UseUnityLogFile;
    private IOps m_Ops;
    private const string Prefix = "After ";

    /// <summary>
    /// Create `VerboseOps` for target `ops`
    /// </summary>
    /// <param name="ops">target `IOps` instance</param>
    /// <param name="ops">produce log in Unity standard log file, model execution reporter from IOps will always be used if it exist.</param>
    public VerboseOps(IOps ops, bool useUnityLogFile = true)
    {
        m_Ops = ops;
        m_UseUnityLogFile = useUnityLogFile;
    }

#if ENABLE_BARRACUDA_STATS
    /// <inheritdoc/>
    public IEnumerable<TempMemoryStatistics> GetTempMemoryStatistics()
    {
        return m_Ops.GetTempMemoryStatistics();
    }
#endif //ENABLE_BARRACUDA_STATS

    /// <inheritdoc/>
    public virtual void PrepareModel(Model model, IDictionary<string, TensorShape> inputShapes, IVars vars)
    {
        if (m_Ops is IModelCompiler)
            ((IModelCompiler)m_Ops).PrepareModel(model, inputShapes, vars);
    }

    /// <inheritdoc/>
    public virtual void PostLayerCleanup()
    {
        m_Ops.PostLayerCleanup();
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
        LogLayerSummary(rankX + ":(" + X.batch * X.channels + "," + X.height + "," + X.width + ")" +
            " *" + rankY + ":(" + Y.batch * Y.channels + "," + Y.height + "," + Y.width + ")");
        var O = m_Ops.MatMul(X, rankX, Y, rankY);
        LogOutputTensorSummary(O, Prefix + "MatMul");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.MatMul(Tensor X, bool xTranspose, Tensor Y, bool yTranspose)
    {

        LogLayerSummary("(" + X.flatHeight + "," + X.flatWidth + ")" + (xTranspose ? ".T" : "") +
            " * (" + Y.flatHeight + "," + Y.flatWidth + ")" + (yTranspose ? ".T" : ""));
        var O = m_Ops.MatMul(X, xTranspose, Y, yTranspose);
        LogOutputTensorSummary(O, Prefix + "MatMul");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Dense(Tensor X, Tensor W, Tensor B, Layer.FusedActivation fusedActivation)
    {
        LogLayerSummary(X.shape + " * (" + W.flatHeight + "," + W.flatWidth + ") + (" + B.flatWidth + ")");
        var O = m_Ops.Dense(X, W, B, fusedActivation);
        LogOutputTensorSummary(O, Prefix + "Dense");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Dense3(Tensor X, Tensor W, Tensor B)
    {
        LogLayerSummary(X.shape + " * (" + W.flatHeight + "," + W.flatWidth + ") + (" + B.flatWidth + ")");
        var O = m_Ops.Dense3(X, W, B);
        LogOutputTensorSummary(O, Prefix + "Dense3");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Conv2D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, Layer.FusedActivation fusedActivation)
    {
        LogLayerSummary(X.shape + " # " + K.shape + " + (" + B.flatWidth + ")");
        var O = m_Ops.Conv2D(X, K, B, stride, pad, fusedActivation);
        LogOutputTensorSummary(O, Prefix + "Conv2D");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Conv3D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, Layer.FusedActivation fusedActivation)
    {
        LogLayerSummary(X.shape + " # " + K.shape + " + (" + B.flatWidth + ")");
        var O = m_Ops.Conv3D(X, K, B, stride, pad, fusedActivation);
        LogOutputTensorSummary(O, Prefix + "Conv3D");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.DepthwiseConv2D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, Layer.FusedActivation fusedActivation)
    {
        LogLayerSummary(X.shape + " ∆ " + K.shape + " + (" + B.flatWidth + ")");
        var O = m_Ops.DepthwiseConv2D(X, K, B, stride, pad, fusedActivation);
        LogOutputTensorSummary(O, Prefix + "DepthwiseConv2D");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Conv2DTrans(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, int[] outputAdjustment, Layer.FusedActivation fusedActivation)
    {
        LogLayerSummary(X.shape + " @ " + K.shape + " + (" + B.flatWidth + ")");
        var O = m_Ops.Conv2DTrans(X, K, B, stride, pad, outputAdjustment, fusedActivation);
        LogOutputTensorSummary(O, Prefix + "Conv2DTrans");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Upsample2D(Tensor X, int[] scale, bool bilinear)
    {
        var O = m_Ops.Upsample2D(X, scale, bilinear);
        LogLayerSummary(X.shape + " ^ " + (bilinear ? "bilinear" : "") + O.shape);
        LogOutputTensorSummary(O, Prefix + "Upsample2D");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Upsample3D(Tensor X, int[] scale, bool trilinear)
    {
        var O = m_Ops.Upsample3D(X, scale, trilinear);
        LogLayerSummary(X.shape + " ^ " + (trilinear ? "trilinear" : "") + O.shape);
        LogOutputTensorSummary(O, Prefix + "Upsample3D");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Resample2D(Tensor X, int[] size, bool bilinear)
    {
        var O = m_Ops.Resample2D(X, size, bilinear);
        LogLayerSummary(X.shape + " ^ " + (bilinear ? "bilinear" : "") + O.shape);
        LogOutputTensorSummary(O, Prefix + "Resample2D");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.DepthToSpace(Tensor X, int[] scale, Layer.DepthToSpaceMode mode)
    {
        var O = m_Ops.DepthToSpace(X, scale, mode);
        LogLayerSummary(X.shape + " ^ " + mode + O.shape);
        LogOutputTensorSummary(O, Prefix + "DepthToSpace");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.SpaceToDepth(Tensor X, int[] scale)
    {
        var O = m_Ops.SpaceToDepth(X, scale);
        LogLayerSummary(X.shape + " ^ " + O.shape);
        LogOutputTensorSummary(O, Prefix + "SpaceToDepth");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.MaxPool2D(Tensor X, int[] pool, int[] stride, int[] pad)
    {
        var O = m_Ops.MaxPool2D(X, pool, stride, pad);
        LogLayerSummary(X.shape + " > " + O.shape);
        LogOutputTensorSummary(O, Prefix + "MaxPool2D");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.AvgPool2D(Tensor X, int[] pool, int[] stride, int[] pad)
    {
        var O = m_Ops.AvgPool2D(X, pool, stride, pad);
        LogLayerSummary(X.shape + " ≥ " + O.shape);
        LogOutputTensorSummary(O, Prefix + "AvgPool2D");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.GlobalMaxPool2D(Tensor X)
    {
        var O = m_Ops.GlobalMaxPool2D(X);
        LogLayerSummary(X.shape + " >> " + O.shape);
        LogOutputTensorSummary(O, Prefix + "GlobalMaxPool2D");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.GlobalAvgPool2D(Tensor X)
    {
        var O = m_Ops.GlobalAvgPool2D(X);
        LogLayerSummary(X.shape + " ≥≥ " + O.shape);
        LogOutputTensorSummary(O, Prefix + "GlobalAvgPool2D");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.GlobalAvgVariancePool2D(Tensor X)
    {
        var O = m_Ops.GlobalAvgVariancePool2D(X);
        LogLayerSummary(X.shape + " ≥≥ " + O.shape);
        LogOutputTensorSummary(O, Prefix + "GlobalAvgVariancePool2D");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Border2D(Tensor X, int[] pad, float value)
    {
        LogLayerSummary($"{X.shape} ¶(border) value={value} pad=[{pad[0]},{pad[1]},{pad[2]},{pad[3]})");
        var O = m_Ops.Border2D(X, pad, value);
        LogOutputTensorSummary(O, Prefix + "Border2D");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Border3D(Tensor X, int[] pad, float value)
    {
        LogLayerSummary($"{X.shape} ¶(border3d) value={value} pad=[{pad[0]},{pad[1]},{pad[2]},{pad[3]},{pad[4]},{pad[5]})");
        var O = m_Ops.Border3D(X, pad, value);
        LogOutputTensorSummary(O, Prefix + "Border3D");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Pad2DReflect(Tensor X, int[] pad)
    {
        LogLayerSummary($"{X.shape} ¶(reflect) pad=[{pad[0]},{pad[1]},{pad[2]},{pad[3]})");
        var O = m_Ops.Pad2DReflect(X, pad);
        LogOutputTensorSummary(O, Prefix + "Pad2DReflect");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Pad2DSymmetric(Tensor X, int[] pad)
    {
        LogLayerSummary($"{X.shape} ¶(symmetric) pad=[{pad[0]},{pad[1]},{pad[2]},{pad[3]})");
        var O = m_Ops.Pad2DSymmetric(X, pad);
        LogOutputTensorSummary(O, Prefix + "Pad2DSymmetric");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Pad2DEdge(Tensor X, int[] pad)
    {
        LogLayerSummary($"{X.shape} ¶(edge) pad=[{pad[0]},{pad[1]},{pad[2]},{pad[3]})");
        var O = m_Ops.Pad2DEdge(X, pad);
        LogOutputTensorSummary(O, Prefix + "Pad2DEdge");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.ScaleBias(Tensor X, Tensor S, Tensor B)
    {
        LogLayerSummary(X.shape + " * (" + S.channels + ") + (" + B.channels + ")");
        var O = m_Ops.ScaleBias(X, S, B);
        LogOutputTensorSummary(O, Prefix + "ScaleBias");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Normalization(Tensor X, Tensor S, Tensor B, int pool, int axis, float epsilon, Layer.FusedActivation fusedActivation)
    {
        LogLayerSummary(X.shape + " ! " + (pool==1 ? "instance": "batch") + " axis=" + axis);
        var O = m_Ops.Normalization(X, S, B, pool, axis, epsilon, fusedActivation);
        LogOutputTensorSummary(O, Prefix + "Normalization");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.LRN(Tensor X, float alpha, float beta, float bias, int size)
    {
        LogLayerSummary(X.shape + " LRN n=" + size + " a=" + alpha + " b=" + beta + " bias=" + bias);
        var O = m_Ops.LRN(X, alpha, beta, bias, size);
        LogOutputTensorSummary(O, Prefix + "LRN");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Dropout(Tensor X, float alpha)
    {
        LogLayerSummary(X.shape + "  a=" + alpha);
        var O = m_Ops.Dropout(X, alpha);
        LogOutputTensorSummary(O, Prefix + "Dropout");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.RandomNormal(TensorShape s, float mean, float scale, int seed)
    {
        LogLayerSummary(s + " N m=" + mean + " s=" + scale + " s=" + seed);
        var O = m_Ops.RandomNormal(s, mean, scale, seed);
        LogOutputTensorSummary(O, Prefix + "RandomNormal");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.RandomUniform(TensorShape s, float mean, float scale, int seed)
    {
        LogLayerSummary(s + " U m=" + mean + " s=" + scale + " s=" + seed);
        var O = m_Ops.RandomUniform(s, mean, scale, seed);
        LogOutputTensorSummary(O, Prefix + "RandomUniform");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Multinomial(Tensor X, int count, int seed)
    {
        LogLayerSummary(X.shape + " M n=" + count + " s=" + seed);
        var O = m_Ops.Multinomial(X, count, seed);
        LogOutputTensorSummary(O, Prefix + "Multinomial");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.OneHot(Tensor X, int depth, float onValue, float offValue, int inputRank)
    {
        LogLayerSummary(X.shape + " Ω n=" + depth + " 1=" + onValue + " 0=" + offValue);
        var O = m_Ops.OneHot(X, depth, onValue, offValue, inputRank);
        LogOutputTensorSummary(O, Prefix + "OneHot");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.RoiAlign(Tensor X, Tensor rois, Tensor indices, int outputHeight, int outputWidth, int samplingRatio, float spatialScale)
    {
        LogLayerSummary(X.shape + " # " + rois.shape + "-> (" + outputHeight + "," + outputWidth + "," + samplingRatio + "," + spatialScale + ")");
        var O = m_Ops.RoiAlign(X, rois, indices, outputHeight, outputWidth, samplingRatio, spatialScale);
        LogOutputTensorSummary(O, Prefix + "RoiAlign");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.TopKIndices(Tensor X, int k, int axis, bool largest, bool sorted)
    {
        LogLayerSummary($"{X.shape} Ω k={k} a={axis} l={largest} s={sorted}");
        var O = m_Ops.TopKIndices(X, k, axis, largest, sorted);
        LogOutputTensorSummary(O, Prefix + "TopKIndices");
        return O;
    }

    /// <inheritdoc/>
    public Tensor TopKValues(Tensor X, Tensor I, int axis)
    {
        LogLayerSummary($"{X.shape} {I.shape} Ω a={axis}");
        var O = m_Ops.TopKValues(X, I, axis);
        LogOutputTensorSummary(O, Prefix + "TopKValues");
        return O;
    }

    /// <inheritdoc/>
    public Tensor NonZero(Tensor X)
    {
        LogLayerSummary($"{X.shape} NonZero");
        var O = m_Ops.NonZero(X);
        LogOutputTensorSummary(O, Prefix + "NonZero");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Relu(Tensor X)
    {
        LogLayerSummary(X.shape + " ()");
        var O = m_Ops.Relu(X);
        LogOutputTensorSummary(O, Prefix + "Relu");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Softmax(Tensor X, int axis)
    {
        LogLayerSummary(X.shape + " ()");
        var O = m_Ops.Softmax(X, axis);
        LogOutputTensorSummary(O, Prefix + "Softmax");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.LogSoftmax(Tensor X, int axis)
    {
        LogLayerSummary(X.shape + " ()");
        var O = m_Ops.LogSoftmax(X, axis);
        LogOutputTensorSummary(O, Prefix + "LogSoftmax");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Tanh(Tensor X)
    {
        LogLayerSummary(X.shape + " ()");
        var O = m_Ops.Tanh(X);
        LogOutputTensorSummary(O, Prefix + "Tanh");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Softplus(Tensor X)
    {
        LogLayerSummary(X.shape + " ()");
        var O = m_Ops.Softplus(X);
        LogOutputTensorSummary(O, Prefix + "Softplus");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Sigmoid(Tensor X)
    {
        LogLayerSummary(X.shape + " ()");
        var O = m_Ops.Sigmoid(X);
        LogOutputTensorSummary(O, Prefix + "Sigmoid");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.HardSigmoid(Tensor X, float alpha, float beta)
    {
        LogLayerSummary(X.shape + " ()");
        var O = m_Ops.HardSigmoid(X, alpha, beta);
        LogOutputTensorSummary(O, Prefix + "HardSigmoid");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Relu6(Tensor X)
    {
        LogLayerSummary(X.shape + " ()");
        var O = m_Ops.Relu6(X);
        LogOutputTensorSummary(O, Prefix + "Relu6");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Elu(Tensor X, float alpha)
    {
        LogLayerSummary(X.shape + " () a=" + alpha);
        var O = m_Ops.Elu(X, alpha);
        LogOutputTensorSummary(O, Prefix + "Elu");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.LeakyRelu(Tensor X, float alpha)
    {
        LogLayerSummary(X.shape + " () a=" + alpha);
        var O = m_Ops.LeakyRelu(X, alpha);
        LogOutputTensorSummary(O, Prefix + "LeakyRelu");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Selu(Tensor X, float alpha, float gamma)
    {
        LogLayerSummary(X.shape + " () a=" + alpha + " g=" + gamma);
        var O = m_Ops.Selu(X, alpha, gamma);
        LogOutputTensorSummary(O, Prefix + "Selu");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.PRelu(Tensor X, Tensor S)
    {
        LogLayerSummary(X.shape + " * (" + S.channels + ")");
        var O = m_Ops.PRelu(X, S);
        LogOutputTensorSummary(O, Prefix + "PRelu");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Swish(Tensor X)
    {
        LogLayerSummary(X.shape + " ()");
        var O = m_Ops.Swish(X);
        LogOutputTensorSummary(O, Prefix + "Swish");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Abs(Tensor X)
    {
        LogLayerSummary(X.shape + " ()");
        var O = m_Ops.Abs(X);
        LogOutputTensorSummary(O, Prefix + "Abs");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Neg(Tensor X)
    {
        LogLayerSummary(X.shape + " ()");
        var O = m_Ops.Neg(X);
        LogOutputTensorSummary(O, Prefix + "Neg");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Ceil(Tensor X)
    {
        LogLayerSummary(X.shape + " ()");
        var O = m_Ops.Ceil(X);
        LogOutputTensorSummary(O, Prefix + "Ceil");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Clip(Tensor X, float min, float max)
    {
        LogLayerSummary(X.shape + " () min=" + min + " max=" + max);
        var O = m_Ops.Clip(X, min, max);
        LogOutputTensorSummary(O, Prefix + "Clip");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Floor(Tensor X)
    {
        LogLayerSummary(X.shape + " ()");
        var O = m_Ops.Floor(X);
        LogOutputTensorSummary(O, Prefix + "Floor");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Round(Tensor X)
    {
        LogLayerSummary(X.shape + " ()");
        var O = m_Ops.Round(X);
        LogOutputTensorSummary(O, Prefix + "Round");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Reciprocal(Tensor X)
    {
        LogLayerSummary(X.shape + " ()");
        var O = m_Ops.Reciprocal(X);
        LogOutputTensorSummary(O, Prefix + "Reciprocal");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Pow(Tensor X, float alpha)
    {
        LogLayerSummary(X.shape + " () a=" + alpha);
        var O = m_Ops.Pow(X, alpha);
        LogOutputTensorSummary(O, Prefix + "Pow");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Exp(Tensor X)
    {
        LogLayerSummary(X.shape + " ()");
        var O = m_Ops.Exp(X);
        LogOutputTensorSummary(O, Prefix + "Exp");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Log(Tensor X)
    {
        LogLayerSummary(X.shape + " ()");
        var O = m_Ops.Log(X);
        LogOutputTensorSummary(O, Prefix + "Log");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Sqrt(Tensor X)
    {
        LogLayerSummary(X.shape + " ()");
        var O = m_Ops.Sqrt(X);
        LogOutputTensorSummary(O, Prefix + "Sqrt");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Acos(Tensor X)
    {
        LogLayerSummary(X.shape + " ()");
        var O = m_Ops.Acos(X);
        LogOutputTensorSummary(O, Prefix + "Acos");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Acosh(Tensor X)
    {
        LogLayerSummary(X.shape + " ()");
        var O = m_Ops.Acosh(X);
        LogOutputTensorSummary(O, Prefix + "Acosh");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Asin(Tensor X)
    {
        LogLayerSummary(X.shape + " ()");
        var O = m_Ops.Asin(X);
        LogOutputTensorSummary(O, Prefix + "Asin");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Asinh(Tensor X)
    {
        LogLayerSummary(X.shape + " ()");
        var O = m_Ops.Asinh(X);
        LogOutputTensorSummary(O, Prefix + "Asinh");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Atan(Tensor X)
    {
        LogLayerSummary(X.shape + " ()");
        var O = m_Ops.Atan(X);
        LogOutputTensorSummary(O, Prefix + "Atan");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Atanh(Tensor X)
    {
        LogLayerSummary(X.shape + " ()");
        var O = m_Ops.Atanh(X);
        LogOutputTensorSummary(O, Prefix + "Atanh");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Cos(Tensor X)
    {
        LogLayerSummary(X.shape + " ()");
        var O = m_Ops.Cos(X);
        LogOutputTensorSummary(O, Prefix + "Cos");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Cosh(Tensor X)
    {
        LogLayerSummary(X.shape + " ()");
        var O = m_Ops.Cosh(X);
        LogOutputTensorSummary(O, Prefix + "Cosh");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Sin(Tensor X)
    {
        LogLayerSummary(X.shape + " ()");
        var O = m_Ops.Sin(X);
        LogOutputTensorSummary(O, Prefix + "Sin");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Sinh(Tensor X)
    {
        LogLayerSummary(X.shape + " ()");
        var O = m_Ops.Sinh(X);
        LogOutputTensorSummary(O, Prefix + "Sinh");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Tan(Tensor X)
    {
        LogLayerSummary(X.shape + " ()");
        var O = m_Ops.Tan(X);
        LogOutputTensorSummary(O, Prefix + "Tan");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Erf(Tensor X)
    {
        LogLayerSummary(X.shape + " ()");
        var O = m_Ops.Erf(X);
        LogOutputTensorSummary(O, Prefix + "Erf");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Add(Tensor[] tensors)
    {
        var O = m_Ops.Add(tensors);
        LogLayerSummary("{" + tensors.Length + "} + " + O.shape); // @TODO: print input dimensions
        LogOutputTensorSummary(O, Prefix + "Add");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Sub(Tensor[] tensors)
    {
        var O = m_Ops.Sub(tensors);
        LogLayerSummary("{" + tensors.Length + "} - " + O.shape); // @TODO: print input dimensions
        LogOutputTensorSummary(O, Prefix + "Sub");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Mul(Tensor[] tensors)
    {
        var O = m_Ops.Mul(tensors);
        LogLayerSummary("{" + tensors.Length + "} * " + O.shape); // @TODO: print input dimensions
        LogOutputTensorSummary(O, Prefix + "Mul");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Div(Tensor[] tensors)
    {
        var O = m_Ops.Div(tensors);
        LogLayerSummary("{" + tensors.Length + "} / " + O.shape); // @TODO: print input dimensions
        LogOutputTensorSummary(O, Prefix + "Div");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Pow(Tensor[] tensors)
    {
        var O = m_Ops.Pow(tensors);
        LogLayerSummary("{" + tensors.Length + "} ^ " + O.shape); // @TODO: print input dimensions
        LogOutputTensorSummary(O, Prefix + "Pow");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Min(Tensor[] tensors)
    {
        var O = m_Ops.Min(tensors);
        LogLayerSummary("{" + tensors.Length + "} < " + O.shape); // @TODO: print input dimensions
        LogOutputTensorSummary(O, Prefix + "Min");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Max(Tensor[] tensors)
    {
        var O = m_Ops.Max(tensors);
        LogLayerSummary("{" + tensors.Length + "} > " + O.shape); // @TODO: print input dimensions
        LogOutputTensorSummary(O, Prefix + "Max");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Mean(Tensor[] tensors)
    {
        var O = m_Ops.Mean(tensors);
        LogLayerSummary("{" + tensors.Length + "} ∑ " + O.shape); // @TODO: print input dimensions
        LogOutputTensorSummary(O, Prefix + "Mean");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.ReduceMax(Tensor X, int axis)
    {
        var O = m_Ops.ReduceMax(X, axis);
        LogLayerSummary(X.shape + " .> " + O.shape);
        LogOutputTensorSummary(O, Prefix + "ReduceMax");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.ReduceMean(Tensor X, int axis)
    {
        var O = m_Ops.ReduceMean(X, axis);
        LogLayerSummary(X.shape + " .∑ " + O.shape);
        LogOutputTensorSummary(O, Prefix + "ReduceMean");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.ReduceMin(Tensor X, int axis)
    {
        var O = m_Ops.ReduceMin(X, axis);
        LogLayerSummary(X.shape + " .< " + O.shape);
        LogOutputTensorSummary(O, Prefix + "ReduceMin");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.ReduceProd(Tensor X, int axis)
    {
        var O = m_Ops.ReduceProd(X, axis);
        LogLayerSummary(X.shape + " .* " + O.shape);
        LogOutputTensorSummary(O, Prefix + "ReduceProd");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.ReduceSum(Tensor X, int axis)
    {
        var O = m_Ops.ReduceSum(X, axis);
        LogLayerSummary(X.shape + " .+ " + O.shape);
        LogOutputTensorSummary(O, Prefix + "ReduceSum");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.ArgMax(Tensor X, int axis)
    {
        var O = m_Ops.ArgMax(X, axis);
        LogLayerSummary(X.shape + " .+ " + O.shape);
        LogOutputTensorSummary(O, Prefix + "ArgMax");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.ArgMin(Tensor X, int axis)
    {
        var O = m_Ops.ArgMin(X, axis);
        LogLayerSummary(X.shape + " .+ " + O.shape);
        LogOutputTensorSummary(O, Prefix + "ArgMin");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Greater(Tensor a, Tensor b)
    {
        var O = m_Ops.Greater(a, b);
        LogLayerSummary(a.shape + " > " + b.shape + " = " + O.shape);
        LogOutputTensorSummary(O, Prefix + "Greater");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.GreaterEqual(Tensor a, Tensor b)
    {
        var O = m_Ops.GreaterEqual(a, b);
        LogLayerSummary(a.shape + " >= " + b.shape + " = " + O.shape);
        LogOutputTensorSummary(O, Prefix + "GreaterEqual");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Less(Tensor a, Tensor b)
    {
        var O = m_Ops.Less(a, b);
        LogLayerSummary(a.shape + " < " + b.shape + " = " + O.shape);
        LogOutputTensorSummary(O, Prefix + "Less");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.LessEqual(Tensor a, Tensor b)
    {
        var O = m_Ops.LessEqual(a, b);
        LogLayerSummary(a.shape + " <= " + b.shape + " = " + O.shape);
        LogOutputTensorSummary(O, Prefix + "LessEqual");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Equal(Tensor a, Tensor b)
    {
        var O = m_Ops.Equal(a, b);
        LogLayerSummary(a.shape + " == " + b.shape + " = " + O.shape);
        LogOutputTensorSummary(O, Prefix + "Equal");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.LogicalOr(Tensor a, Tensor b)
    {
        var O = m_Ops.LogicalOr(a, b);
        LogLayerSummary(a.shape + " || " + b.shape + " = " + O.shape);
        LogOutputTensorSummary(O, Prefix + "LogicalOr");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.LogicalAnd(Tensor a, Tensor b)
    {
        var O = m_Ops.LogicalAnd(a, b);
        LogLayerSummary(a.shape + " && " + b.shape + " = " + O.shape);
        LogOutputTensorSummary(O, Prefix + "LogicalAnd");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.LogicalXor(Tensor a, Tensor b)
    {
        var O = m_Ops.LogicalXor(a, b);
        LogLayerSummary(a.shape + " ^ " + b.shape + " = " + O.shape);
        LogOutputTensorSummary(O, Prefix + "LogicalXor");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.LogicalNot(Tensor x)
    {
        var O = m_Ops.LogicalNot(x);
        LogLayerSummary("!(" + x.shape +" )");
        LogOutputTensorSummary(O, Prefix + "LogicalNot");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Sign(Tensor x)
    {
        var O = m_Ops.Sign(x);
        LogLayerSummary("!(" + x.shape +" )");
        LogOutputTensorSummary(O, Prefix + "Sign");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Where(Tensor c, Tensor a, Tensor b)
    {
        var O = m_Ops.Where(c, a, b);
        LogLayerSummary(c.shape + " ? " + a.shape + ":" + b.shape + " = " + O.shape);
        LogOutputTensorSummary(O, Prefix + "Where");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Flatten(Tensor X)
    {
        var O = m_Ops.Flatten(X);
        LogLayerSummary(X.shape + " = " + O.shape);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Reshape(Tensor X, TensorShape shape)
    {
        var O = m_Ops.Reshape(X, shape);
        LogLayerSummary(X.shape + " $ " + O.shape);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Expand(Tensor X, TensorShape shape)
    {
        var O = m_Ops.Expand(X, shape);
        LogLayerSummary(X.shape + " $ " + O.shape);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Transpose(Tensor X)
    {
        var O = m_Ops.Transpose(X);
        LogLayerSummary(X.shape + " T " + O.shape);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Transpose(Tensor X, int[] permutations)
    {
        var O = m_Ops.Transpose(X, permutations);
        LogLayerSummary(X.shape + " T " + O.shape);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Gather(Tensor[] tensors, int axis)
    {
        var O = m_Ops.Gather(tensors,axis);
        LogLayerSummary("{" + tensors[0].shape + "," + tensors[1].shape + "," + axis + "} # " + O.shape);
        LogOutputTensorSummary(O, Prefix + "Gather");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.ScatterND(Tensor X, Tensor indices, Tensor updates, Layer.ScatterNDReductionMode reduction)
    {
        var O = m_Ops.ScatterND(X, indices, updates, reduction);
        LogLayerSummary("{" + X.shape + "," + indices.shape + "," + updates.shape + "," + reduction + "} # " + O.shape);
        LogOutputTensorSummary(O, Prefix + "Gather");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.NonMaxSuppression(Tensor[] tensors, int maxOutputBoxesPerClass, float iouThreshold, float scoreThreshold, int centerPointBox)
    {
        var O = m_Ops.NonMaxSuppression(tensors, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, centerPointBox);
        LogLayerSummary($"{string.Join(",", Enumerable.Select(tensors, t => t.shape.ToString()))} centerPointBox: {centerPointBox} # {O.shape}");
        LogOutputTensorSummary(O, Prefix + nameof(IOps.NonMaxSuppression));
        return O;
    }

    /// <inheritdoc/>
    public Tensor[] LSTM(Tensor X, Tensor[] W, Tensor[] R, Tensor[] Wb, Tensor[] Rb, Tensor hidden, Tensor cell)
    {
        var O = m_Ops.LSTM(X, W, R, Wb, Rb, hidden, cell);
        LogLayerSummary($"X: {X.shape} hidden: {hidden.shape} cell: {cell.shape}");
        LogOutputTensorSummary(O[0], Prefix + nameof(IOps.LSTM));
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Concat(Tensor[] tensors, int axis)
    {
        var O = m_Ops.Concat(tensors, axis);
        LogLayerSummary("{" + tensors.Length + "} # " + O.shape); // @TODO: print input dimensions
        LogOutputTensorSummary(O, Prefix + "Concat");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.StridedSlice(Tensor X, int[] starts, int[] ends, int[] strides)
    {
        var O = m_Ops.StridedSlice(X, starts, ends, strides);
        LogLayerSummary(X.shape + " | " + O.shape);
        LogOutputTensorSummary(O, Prefix + "StridedSlice");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Tile(Tensor X, int[] repeats)
    {
        var O = m_Ops.Tile(X, repeats);
        LogLayerSummary(X.shape + " % " + O.shape);
        LogOutputTensorSummary(O, Prefix + "Tile");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Shape(Tensor X, int axis)
    {
        LogLayerSummary($"{X.shape}");
        var O = m_Ops.Shape(X, axis);
        LogOutputTensorSummary(O, Prefix + nameof(IOps.Shape));
        return O;
    }


    /// <inheritdoc/>
    Tensor IOps.ConstantOfShape(TensorShape X, DataType type, float value)
    {
        LogLayerSummary($"ConstantOfShape {value}");
        var O = m_Ops.ConstantOfShape(X, type, value);
        LogOutputTensorSummary(O, Prefix + nameof(IOps.ConstantOfShape));
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Copy(Tensor x)
    {
        var O = m_Ops.Copy(x);
        LogLayerSummary("!(" + x.shape +" )");
        LogOutputTensorSummary(O, "Copy");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Prepare(Tensor X)
    {
        if (m_UseUnityLogFile)
        D.Log("!" + X.shape);
        return m_Ops.Prepare(X);
    }

    /// <inheritdoc/>
    Tensor IOps.PrepareNoAlloc(Tensor X)
    {
        D.Log("!" + X.shape);
        return m_Ops.PrepareNoAlloc(X);
    }

    /// <inheritdoc/>
    void IOps.ResetAllocator(bool keepCachedMemory)
    {
        m_Ops.ResetAllocator(keepCachedMemory);
    }

    /// <inheritdoc/>
    void IOps.SetModelExecutionsReporter(IModelExecutionsReporter executionsReporter)
    {
        m_Ops.SetModelExecutionsReporter(executionsReporter);
    }

    /// <inheritdoc/>
    IModelExecutionsReporter IOps.GetModelExecutionsReporter()
    {
        return m_Ops.GetModelExecutionsReporter();
    }

    private void LogLayerSummary(string summary)
    {
        if (m_UseUnityLogFile)
            D.Log(summary);
#if ENABLE_BARRACUDA_STATS
        m_Ops.GetModelExecutionsReporter()?.SetLayerSummary(summary);
#endif //ENABLE_BARRACUDA_STATS
    }

    private void LogOutputTensorSummary(Tensor O, string messagePrefix, int size = 32)
    {
        if (m_UseUnityLogFile)
            O.PrintDataPart(size, messagePrefix);
    }
}


} // namespace Unity.Barracuda
