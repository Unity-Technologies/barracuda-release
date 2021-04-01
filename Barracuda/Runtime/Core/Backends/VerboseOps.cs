using System.Collections.Generic;
using System.Linq;

namespace Unity.Barracuda {

    /// <summary>
    /// Verbose proxy to other `IOps` implementation
    /// </summary>
public class VerboseOps : IOps, IModelCompiler
{
    private IOps m_Ops;
    private const string Prefix = "After ";

    /// <summary>
    /// Create `VerboseOps` for target `ops`
    /// </summary>
    /// <param name="ops">target `IOps` instance</param>
    public VerboseOps(IOps ops)
    {
        m_Ops = ops;
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
        D.Log(rankX + ":(" + X.batch * X.channels + "," + X.height + "," + X.width + ")" +
            " *" + rankY + ":(" + Y.batch * Y.channels + "," + Y.height + "," + Y.width + ")");
        var O = m_Ops.MatMul(X, rankX, Y, rankY);
        O.PrintDataPart(32, Prefix + "MatMul");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.MatMul(Tensor X, bool xTranspose, Tensor Y, bool yTranspose)
    {

        D.Log("(" + X.flatHeight + "," + X.flatWidth + ")" + (xTranspose ? ".T" : "") +
            " * (" + Y.flatHeight + "," + Y.flatWidth + ")" + (yTranspose ? ".T" : ""));
        var O = m_Ops.MatMul(X, xTranspose, Y, yTranspose);
        O.PrintDataPart(32, Prefix + "MatMul");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Dense(Tensor X, Tensor W, Tensor B, Layer.FusedActivation fusedActivation)
    {
        D.Log(X.shape + " * (" + W.flatHeight + "," + W.flatWidth + ") + (" + B.flatWidth + ")");
        var O = m_Ops.Dense(X, W, B, fusedActivation);
        O.PrintDataPart(32, Prefix + "Dense");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Dense3(Tensor X, Tensor W, Tensor B)
    {
        D.Log(X.shape + " * (" + W.flatHeight + "," + W.flatWidth + ") + (" + B.flatWidth + ")");
        var O = m_Ops.Dense3(X, W, B);
        O.PrintDataPart(32, Prefix + "Dense3");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Conv2D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, Layer.FusedActivation fusedActivation)
    {
        D.Log(X.shape + " # " + K.shape + " + (" + B.flatWidth + ")");
        var O = m_Ops.Conv2D(X, K, B, stride, pad, fusedActivation);
        O.PrintDataPart(32, Prefix + "Conv2D");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Conv3D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, Layer.FusedActivation fusedActivation)
    {
        D.Log(X.shape + " # " + K.shape + " + (" + B.flatWidth + ")");
        var O = m_Ops.Conv3D(X, K, B, stride, pad, fusedActivation);
        O.PrintDataPart(32, Prefix + "Conv3D");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.DepthwiseConv2D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, Layer.FusedActivation fusedActivation)
    {
        D.Log(X.shape + " ∆ " + K.shape + " + (" + B.flatWidth + ")");
        var O = m_Ops.DepthwiseConv2D(X, K, B, stride, pad, fusedActivation);
        O.PrintDataPart(32, Prefix + "DepthwiseConv2D");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Conv2DTrans(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, int[] outputAdjustment, Layer.FusedActivation fusedActivation)
    {
        D.Log(X.shape + " @ " + K.shape + " + (" + B.flatWidth + ")");
        var O = m_Ops.Conv2DTrans(X, K, B, stride, pad, outputAdjustment, fusedActivation);
        O.PrintDataPart(32, Prefix + "Conv2DTrans");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Upsample2D(Tensor X, int[] scale, bool bilinear)
    {
        var O = m_Ops.Upsample2D(X, scale, bilinear);
        D.Log(X.shape + " ^ " + (bilinear ? "bilinear" : "") + O.shape);
        O.PrintDataPart(32, Prefix + "Upsample2D");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Upsample3D(Tensor X, int[] scale, bool trilinear)
    {
        var O = m_Ops.Upsample3D(X, scale, trilinear);
        D.Log(X.shape + " ^ " + (trilinear ? "trilinear" : "") + O.shape);
        O.PrintDataPart(32, Prefix + "Upsample3D");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Resample2D(Tensor X, int[] size, bool bilinear)
    {
        var O = m_Ops.Resample2D(X, size, bilinear);
        D.Log(X.shape + " ^ " + (bilinear ? "bilinear" : "") + O.shape);
        O.PrintDataPart(32, Prefix + "Resample2D");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.DepthToSpace(Tensor X, int[] scale, Layer.DepthToSpaceMode mode)
    {
        var O = m_Ops.DepthToSpace(X, scale, mode);
        D.Log(X.shape + " ^ " + mode + O.shape);
        O.PrintDataPart(32, Prefix + "DepthToSpace");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.SpaceToDepth(Tensor X, int[] scale)
    {
        var O = m_Ops.SpaceToDepth(X, scale);
        D.Log(X.shape + " ^ " + O.shape);
        O.PrintDataPart(32, Prefix + "SpaceToDepth");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.MaxPool2D(Tensor X, int[] pool, int[] stride, int[] pad)
    {
        var O = m_Ops.MaxPool2D(X, pool, stride, pad);
        D.Log(X.shape + " > " + O.shape);
        O.PrintDataPart(32, Prefix + "MaxPool2D");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.AvgPool2D(Tensor X, int[] pool, int[] stride, int[] pad)
    {
        var O = m_Ops.AvgPool2D(X, pool, stride, pad);
        D.Log(X.shape + " ≥ " + O.shape);
        O.PrintDataPart(32, Prefix + "AvgPool2D");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.GlobalMaxPool2D(Tensor X)
    {
        var O = m_Ops.GlobalMaxPool2D(X);
        D.Log(X.shape + " >> " + O.shape);
        O.PrintDataPart(32, Prefix + "GlobalMaxPool2D");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.GlobalAvgPool2D(Tensor X)
    {
        var O = m_Ops.GlobalAvgPool2D(X);
        D.Log(X.shape + " ≥≥ " + O.shape);
        O.PrintDataPart(32, Prefix + "GlobalAvgPool2D");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.GlobalAvgVariancePool2D(Tensor X)
    {
        var O = m_Ops.GlobalAvgVariancePool2D(X);
        D.Log(X.shape + " ≥≥ " + O.shape);
        O.PrintDataPart(32, Prefix + "GlobalAvgVariancePool2D");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Border2D(Tensor X, int[] pad, float value)
    {
        D.Log($"{X.shape} ¶(border) value={value} pad=[{pad[0]},{pad[1]},{pad[2]},{pad[3]})");
        var O = m_Ops.Border2D(X, pad, value);
        O.PrintDataPart(32, Prefix + "Border2D");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Border3D(Tensor X, int[] pad, float value)
    {
        D.Log($"{X.shape} ¶(border3d) value={value} pad=[{pad[0]},{pad[1]},{pad[2]},{pad[3]},{pad[4]},{pad[5]})");
        var O = m_Ops.Border3D(X, pad, value);
        O.PrintDataPart(32, Prefix + "Border3D");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Pad2DReflect(Tensor X, int[] pad)
    {
        D.Log($"{X.shape} ¶(reflect) pad=[{pad[0]},{pad[1]},{pad[2]},{pad[3]})");
        var O = m_Ops.Pad2DReflect(X, pad);
        O.PrintDataPart(32, Prefix + "Pad2DReflect");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Pad2DSymmetric(Tensor X, int[] pad)
    {
        D.Log($"{X.shape} ¶(symmetric) pad=[{pad[0]},{pad[1]},{pad[2]},{pad[3]})");
        var O = m_Ops.Pad2DSymmetric(X, pad);
        O.PrintDataPart(32, Prefix + "Pad2DSymmetric");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Pad2DEdge(Tensor X, int[] pad)
    {
        D.Log($"{X.shape} ¶(edge) pad=[{pad[0]},{pad[1]},{pad[2]},{pad[3]})");
        var O = m_Ops.Pad2DEdge(X, pad);
        O.PrintDataPart(32, Prefix + "Pad2DEdge");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.ScaleBias(Tensor X, Tensor S, Tensor B)
    {
        D.Log(X.shape + " * (" + S.channels + ") + (" + B.channels + ")");
        var O = m_Ops.ScaleBias(X, S, B);
        O.PrintDataPart(32, Prefix + "ScaleBias");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Normalization(Tensor X, Tensor S, Tensor B, int pool, int axis, float epsilon, Layer.FusedActivation fusedActivation)
    {
        D.Log(X.shape + " ! " + (pool==1 ? "instance": "batch") + " axis=" + axis);
        var O = m_Ops.Normalization(X, S, B, pool, axis, epsilon, fusedActivation);
        O.PrintDataPart(32, Prefix + "Normalization");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.LRN(Tensor X, float alpha, float beta, float bias, int size)
    {
        D.Log(X.shape + " LRN n=" + size + " a=" + alpha + " b=" + beta + " bias=" + bias);
        var O = m_Ops.LRN(X, alpha, beta, bias, size);
        O.PrintDataPart(32, Prefix + "LRN");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Dropout(Tensor X, float alpha)
    {
        D.Log(X.shape + "  a=" + alpha);
        var O = m_Ops.Dropout(X, alpha);
        O.PrintDataPart(32, Prefix + "Dropout");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.RandomNormal(TensorShape s, float mean, float scale, int seed)
    {
        D.Log(s + " N m=" + mean + " s=" + scale + " s=" + seed);
        var O = m_Ops.RandomNormal(s, mean, scale, seed);
        O.PrintDataPart(32, Prefix + "RandomNormal");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.RandomUniform(TensorShape s, float mean, float scale, int seed)
    {
        D.Log(s + " U m=" + mean + " s=" + scale + " s=" + seed);
        var O = m_Ops.RandomUniform(s, mean, scale, seed);
        O.PrintDataPart(32, Prefix + "RandomUniform");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Multinomial(Tensor X, int count, int seed)
    {
        D.Log(X.shape + " M n=" + count + " s=" + seed);
        var O = m_Ops.Multinomial(X, count, seed);
        O.PrintDataPart(32, Prefix + "Multinomial");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.OneHot(Tensor X, int depth, float onValue, float offValue)
    {
        Debug.Log(X.shape + " Ω n=" + depth + " 1=" + onValue + " 0=" + offValue);
        var O = m_Ops.OneHot(X, depth, onValue, offValue);
        O.PrintDataPart(32, Prefix + "OneHot");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.TopKIndices(Tensor X, int k, int axis, bool largest, bool sorted)
    {
        Debug.Log($"{X.shape} Ω k={k} a={axis} l={largest} s={sorted}");
        var O = m_Ops.TopKIndices(X, k, axis, largest, sorted);
        O.PrintDataPart(32, Prefix + "TopKIndices");
        return O;
    }

    /// <inheritdoc/>
    public Tensor TopKValues(Tensor X, Tensor I, int axis)
    {
        Debug.Log($"{X.shape} {I.shape} Ω a={axis}");
        var O = m_Ops.TopKValues(X, I, axis);
        O.PrintDataPart(32, Prefix + "TopKValues");
        return O;
    }

    /// <inheritdoc/>
    public Tensor NonZero(Tensor X)
    {
        Debug.Log($"{X.shape} NonZero");
        var O = m_Ops.NonZero(X);
        O.PrintDataPart(32, Prefix + "NonZero");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Relu(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Relu(X);
        O.PrintDataPart(32, Prefix + "Relu");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Softmax(Tensor X, int axis)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Softmax(X, axis);
        O.PrintDataPart(32, Prefix + "Softmax");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.LogSoftmax(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.LogSoftmax(X);
        O.PrintDataPart(32, Prefix + "LogSoftmax");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Tanh(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Tanh(X);
        O.PrintDataPart(32, Prefix + "Tanh");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Softplus(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Softplus(X);
        O.PrintDataPart(32, Prefix + "Softplus");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Sigmoid(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Sigmoid(X);
        O.PrintDataPart(32, Prefix + "Sigmoid");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Relu6(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Relu6(X);
        O.PrintDataPart(32, Prefix + "Relu6");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Elu(Tensor X, float alpha)
    {
        D.Log(X.shape + " () a=" + alpha);
        var O = m_Ops.Elu(X, alpha);
        O.PrintDataPart(32, Prefix + "Elu");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.LeakyRelu(Tensor X, float alpha)
    {
        D.Log(X.shape + " () a=" + alpha);
        var O = m_Ops.LeakyRelu(X, alpha);
        O.PrintDataPart(32, Prefix + "LeakyRelu");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Selu(Tensor X, float alpha, float gamma)
    {
        D.Log(X.shape + " () a=" + alpha + " g=" + gamma);
        var O = m_Ops.Selu(X, alpha, gamma);
        O.PrintDataPart(32, Prefix + "Selu");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.PRelu(Tensor X, Tensor S)
    {
        D.Log(X.shape + " * (" + S.channels + ")");
        var O = m_Ops.PRelu(X, S);
        O.PrintDataPart(32, Prefix + "PRelu");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Swish(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Swish(X);
        O.PrintDataPart(32, Prefix + "Swish");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Abs(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Abs(X);
        O.PrintDataPart(32, Prefix + "Abs");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Neg(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Neg(X);
        O.PrintDataPart(32, Prefix + "Neg");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Ceil(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Ceil(X);
        O.PrintDataPart(32, Prefix + "Ceil");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Clip(Tensor X, float min, float max)
    {
        D.Log(X.shape + " () min=" + min + " max=" + max);
        var O = m_Ops.Clip(X, min, max);
        O.PrintDataPart(32, Prefix + "Clip");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Floor(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Floor(X);
        O.PrintDataPart(32, Prefix + "Floor");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Round(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Round(X);
        O.PrintDataPart(32, Prefix + "Round");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Reciprocal(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Reciprocal(X);
        O.PrintDataPart(32, Prefix + "Reciprocal");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Pow(Tensor X, float alpha)
    {
        D.Log(X.shape + " () a=" + alpha);
        var O = m_Ops.Pow(X, alpha);
        O.PrintDataPart(32, Prefix + "Pow");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Exp(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Exp(X);
        O.PrintDataPart(32, Prefix + "Exp");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Log(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Log(X);
        O.PrintDataPart(32, Prefix + "Log");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Sqrt(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Sqrt(X);
        O.PrintDataPart(32, Prefix + "Sqrt");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Acos(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Reciprocal(X);
        O.PrintDataPart(32, Prefix + "Acos");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Acosh(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Reciprocal(X);
        O.PrintDataPart(32, Prefix + "Acos");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Asin(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Reciprocal(X);
        O.PrintDataPart(32, Prefix + "Asin");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Asinh(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Reciprocal(X);
        O.PrintDataPart(32, Prefix + "Asin");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Atan(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Reciprocal(X);
        O.PrintDataPart(32, Prefix + "Atan");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Atanh(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Reciprocal(X);
        O.PrintDataPart(32, Prefix + "Atan");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Cos(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Reciprocal(X);
        O.PrintDataPart(32, Prefix + "(");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Cosh(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Reciprocal(X);
        O.PrintDataPart(32, Prefix + "Cosh");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Sin(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Reciprocal(X);
        O.PrintDataPart(32, Prefix + "(");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Sinh(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Reciprocal(X);
        O.PrintDataPart(32, Prefix + "Sinh");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Tan(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Reciprocal(X);
        O.PrintDataPart(32, Prefix + "(");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Add(Tensor[] tensors)
    {
        var O = m_Ops.Add(tensors);
        D.Log("{" + tensors.Length + "} + " + O.shape); // @TODO: print input dimensions
        O.PrintDataPart(32, Prefix + "Add");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Sub(Tensor[] tensors)
    {
        var O = m_Ops.Sub(tensors);
        D.Log("{" + tensors.Length + "} - " + O.shape); // @TODO: print input dimensions
        O.PrintDataPart(32, Prefix + "Sub");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Mul(Tensor[] tensors)
    {
        var O = m_Ops.Mul(tensors);
        D.Log("{" + tensors.Length + "} * " + O.shape); // @TODO: print input dimensions
        O.PrintDataPart(32, Prefix + "Mul");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Div(Tensor[] tensors)
    {
        var O = m_Ops.Div(tensors);
        D.Log("{" + tensors.Length + "} / " + O.shape); // @TODO: print input dimensions
        O.PrintDataPart(32, Prefix + "Div");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Pow(Tensor[] tensors)
    {
        var O = m_Ops.Pow(tensors);
        D.Log("{" + tensors.Length + "} ^ " + O.shape); // @TODO: print input dimensions
        O.PrintDataPart(32, Prefix + "Pow");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Min(Tensor[] tensors)
    {
        var O = m_Ops.Min(tensors);
        D.Log("{" + tensors.Length + "} < " + O.shape); // @TODO: print input dimensions
        O.PrintDataPart(32, Prefix + "Min");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Max(Tensor[] tensors)
    {
        var O = m_Ops.Max(tensors);
        D.Log("{" + tensors.Length + "} > " + O.shape); // @TODO: print input dimensions
        O.PrintDataPart(32, Prefix + "Max");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Mean(Tensor[] tensors)
    {
        var O = m_Ops.Mean(tensors);
        D.Log("{" + tensors.Length + "} ∑ " + O.shape); // @TODO: print input dimensions
        O.PrintDataPart(32, Prefix + "Mean");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.ReduceMax(Tensor X, int axis)
    {
        var O = m_Ops.ReduceMax(X, axis);
        D.Log(X.shape + " .> " + O.shape);
        O.PrintDataPart(32, Prefix + "ReduceMax");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.ReduceMean(Tensor X, int axis)
    {
        var O = m_Ops.ReduceMean(X, axis);
        D.Log(X.shape + " .∑ " + O.shape);
        O.PrintDataPart(32, Prefix + "ReduceMean");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.ReduceMin(Tensor X, int axis)
    {
        var O = m_Ops.ReduceMin(X, axis);
        D.Log(X.shape + " .< " + O.shape);
        O.PrintDataPart(32, Prefix + "ReduceMin");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.ReduceProd(Tensor X, int axis)
    {
        var O = m_Ops.ReduceProd(X, axis);
        D.Log(X.shape + " .* " + O.shape);
        O.PrintDataPart(32, Prefix + "ReduceProd");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.ReduceSum(Tensor X, int axis)
    {
        var O = m_Ops.ReduceSum(X, axis);
        D.Log(X.shape + " .+ " + O.shape);
        O.PrintDataPart(32, Prefix + "ReduceSum");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.ArgMax(Tensor X, int axis)
    {
        var O = m_Ops.ArgMax(X, axis);
        D.Log(X.shape + " .+ " + O.shape);
        O.PrintDataPart(32, Prefix + "ArgMax");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.ArgMin(Tensor X, int axis)
    {
        var O = m_Ops.ArgMin(X, axis);
        D.Log(X.shape + " .+ " + O.shape);
        O.PrintDataPart(32, Prefix + "ArgMin");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Greater(Tensor a, Tensor b)
    {
        var O = m_Ops.Greater(a, b);
        D.Log(a.shape + " > " + b.shape + " = " + O.shape);
        O.PrintDataPart(32, Prefix + "Greater");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.GreaterEqual(Tensor a, Tensor b)
    {
        var O = m_Ops.GreaterEqual(a, b);
        D.Log(a.shape + " >= " + b.shape + " = " + O.shape);
        O.PrintDataPart(32, Prefix + "GreaterEqual");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Less(Tensor a, Tensor b)
    {
        var O = m_Ops.Less(a, b);
        D.Log(a.shape + " < " + b.shape + " = " + O.shape);
        O.PrintDataPart(32, Prefix + "Less");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.LessEqual(Tensor a, Tensor b)
    {
        var O = m_Ops.LessEqual(a, b);
        D.Log(a.shape + " <= " + b.shape + " = " + O.shape);
        O.PrintDataPart(32, Prefix + "LessEqual");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Equal(Tensor a, Tensor b)
    {
        var O = m_Ops.Equal(a, b);
        D.Log(a.shape + " == " + b.shape + " = " + O.shape);
        O.PrintDataPart(32, Prefix + "Equal");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.LogicalOr(Tensor a, Tensor b)
    {
        var O = m_Ops.LogicalOr(a, b);
        D.Log(a.shape + " || " + b.shape + " = " + O.shape);
        O.PrintDataPart(32, Prefix + "LogicalOr");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.LogicalAnd(Tensor a, Tensor b)
    {
        var O = m_Ops.LogicalAnd(a, b);
        D.Log(a.shape + " && " + b.shape + " = " + O.shape);
        O.PrintDataPart(32, Prefix + "LogicalAnd");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.LogicalXor(Tensor a, Tensor b)
    {
        var O = m_Ops.LogicalXor(a, b);
        D.Log(a.shape + " ^ " + b.shape + " = " + O.shape);
        O.PrintDataPart(32, Prefix + "LogicalXor");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.LogicalNot(Tensor x)
    {
        var O = m_Ops.LogicalNot(x);
        D.Log("!(" + x.shape +" )");
        O.PrintDataPart(32, Prefix + "LogicalNot");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Sign(Tensor x)
    {
        var O = m_Ops.Sign(x);
        D.Log("!(" + x.shape +" )");
        O.PrintDataPart(32, Prefix + "Sign");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Where(Tensor c, Tensor a, Tensor b)
    {
        var O = m_Ops.Where(c, a, b);
        D.Log(c.shape + " ? " + a.shape + ":" + b.shape + " = " + O.shape);
        O.PrintDataPart(32, Prefix + "Where");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Flatten(Tensor X)
    {
        var O = m_Ops.Flatten(X);
        D.Log(X.shape + " = " + O.shape);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Reshape(Tensor X, TensorShape shape)
    {
        var O = m_Ops.Reshape(X, shape);
        D.Log(X.shape + " $ " + O.shape);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Expand(Tensor X, TensorShape shape)
    {
        var O = m_Ops.Expand(X, shape);
        D.Log(X.shape + " $ " + O.shape);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Transpose(Tensor X)
    {
        var O = m_Ops.Transpose(X);
        D.Log(X.shape + " T " + O.shape);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Transpose(Tensor X, int[] permutations)
    {
        var O = m_Ops.Transpose(X, permutations);
        D.Log(X.shape + " T " + O.shape);
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Gather(Tensor[] tensors, int axis)
    {
        var O = m_Ops.Gather(tensors,axis);
        D.Log("{" + tensors[0].shape + "," + tensors[1].shape + "," + axis + "} # " + O.shape);
        O.PrintDataPart(32, Prefix + "Gather");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.NonMaxSuppression(Tensor[] tensors, int maxOutputBoxesPerClass, float iouThreshold, float scoreThreshold, int centerPointBox)
    {
        var O = m_Ops.NonMaxSuppression(tensors, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, centerPointBox);
        D.Log($"{string.Join(",", tensors.Select(t => t.shape.ToString()))} centerPointBox: {centerPointBox} # {O.shape}");
        O.PrintDataPart(32, Prefix + nameof(IOps.NonMaxSuppression));
        return O;
    }

    /// <inheritdoc/>
    public Tensor[] LSTM(Tensor X, Tensor[] W, Tensor[] R, Tensor[] Wb, Tensor[] Rb, Tensor hidden, Tensor cell)
    {
        var O = m_Ops.LSTM(X, W, R, Wb, Rb, hidden, cell);
        D.Log($"X: {X.shape} hidden: {hidden.shape} cell: {cell.shape}");
        O[0].PrintDataPart(32, Prefix + nameof(IOps.LSTM));
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Concat(Tensor[] tensors, int axis)
    {
        var O = m_Ops.Concat(tensors, axis);
        D.Log("{" + tensors.Length + "} # " + O.shape); // @TODO: print input dimensions
        O.PrintDataPart(32, Prefix + "Concat");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.StridedSlice(Tensor X, int[] starts, int[] ends, int[] strides)
    {
        var O = m_Ops.StridedSlice(X, starts, ends, strides);
        D.Log(X.shape + " | " + O.shape);
        O.PrintDataPart(32, Prefix + "StridedSlice");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Tile(Tensor X, int[] repeats)
    {
        var O = m_Ops.Tile(X, repeats);
        D.Log(X.shape + " % " + O.shape);
        O.PrintDataPart(32, Prefix + "Tile");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Shape(Tensor X, int axis)
    {
        Debug.Log($"{X.shape}");
        var O = m_Ops.Shape(X, axis);
        O.PrintDataPart(32, Prefix + nameof(IOps.Shape));
        return O;
    }

        
    /// <inheritdoc/>
    Tensor IOps.ConstantOfShape(TensorShape X, float value)
    {
        Debug.Log($"ConstantOfShape {value}");
        var O = m_Ops.ConstantOfShape(X, value);
        O.PrintDataPart(32, Prefix + nameof(IOps.ConstantOfShape));
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Copy(Tensor x)
    {
        var O = m_Ops.Copy(x);
        D.Log("!(" + x.shape +" )");
        O.PrintDataPart(32, "Copy");
        return O;
    }

    /// <inheritdoc/>
    Tensor IOps.Prepare(Tensor X)
    {
        D.Log("!" + X.shape);
        return m_Ops.Prepare(X);
    }

    /// <inheritdoc/>
    void IOps.ResetAllocator(bool keepCachedMemory)
    {
        m_Ops.ResetAllocator(keepCachedMemory);
    }
}


} // namespace Unity.Barracuda
