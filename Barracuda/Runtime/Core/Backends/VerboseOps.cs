using System;
using UnityEngine;
using System.Collections.Generic;

namespace Unity.Barracuda {


public class VerboseOps : IOps, IModelCompiler
{
    private IOps m_Ops;
    private const string Prefix = "After ";

    public VerboseOps(IOps ops)
    {
        m_Ops = ops;
    }

    public virtual void PrepareModel(Model model, IDictionary<string, TensorShape> inputShapes)
    {
        if (m_Ops is IModelCompiler)
            ((IModelCompiler)m_Ops).PrepareModel(model, inputShapes);
    }

    public virtual void PreExecuteLayer(Layer layer, Tensor[] inputs)
    {
        if (m_Ops is IModelCompiler)
            ((IModelCompiler)m_Ops).PreExecuteLayer(layer, inputs);
    }

    public virtual void WaitForCompletion(Tensor x)
    {
        m_Ops.WaitForCompletion(x);
    }

    Tensor IOps.MatMul(Tensor X, bool xTranspose, Tensor Y, bool yTranspose)
    {
        D.Log("(" + X.flatHeight + "," + X.flatWidth + ")" + (xTranspose?".T":"") +
            " * (" + Y.flatHeight + "," + Y.flatWidth + ")"+ (yTranspose?".T":""));
        var O = m_Ops.MatMul(X, xTranspose, Y, yTranspose);
        O.PrintDataPart(32, Prefix + "MatMul");
        return O;
    }
    Tensor IOps.Dense(Tensor X, Tensor W, Tensor B, Layer.FusedActivation fusedActivation)
    {
        D.Log(X.shape + " * (" + W.flatHeight + "," + W.flatWidth + ") + (" + B.flatWidth + ")");
        var O = m_Ops.Dense(X, W, B, fusedActivation);
        O.PrintDataPart(32, Prefix + "Dense");
        return O;
    }
    Tensor IOps.Conv2D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, Layer.FusedActivation fusedActivation)
    {
        D.Log(X.shape + " # " + K.shape + " + (" + B.flatWidth + ")");
        var O = m_Ops.Conv2D(X, K, B, stride, pad, fusedActivation);
        O.PrintDataPart(32, Prefix + "Conv2D");
        return O;
    }
    Tensor IOps.DepthwiseConv2D(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad)
    {
        D.Log(X.shape + " ∆ " + K.shape + " + (" + B.flatWidth + ")");
        var O = m_Ops.DepthwiseConv2D(X, K, B, stride, pad);
        O.PrintDataPart(32, Prefix + "DepthwiseConv2D");
        return O;
    }
    Tensor IOps.Conv2DTrans(Tensor X, Tensor K, Tensor B, int[] stride, int[] pad, int[] outputAdjustment)
    {
        D.Log(X.shape + " @ " + K.shape + " + (" + B.flatWidth + ")");
        var O = m_Ops.Conv2DTrans(X, K, B, stride, pad, outputAdjustment);
        O.PrintDataPart(32, Prefix + "Conv2DTrans");
        return O;
    }
    Tensor IOps.Upsample2D(Tensor X, int[] scale, bool bilinear)
    {
        var O = m_Ops.Upsample2D(X, scale, bilinear);
        D.Log(X.shape + " ^ " + (bilinear ? "bilinear" : "") + O.shape);
        O.PrintDataPart(32, Prefix + "Upsample2D");
        return O;
    }
    Tensor IOps.MaxPool2D(Tensor X, int[] pool, int[] stride, int[] pad)
    {
        var O = m_Ops.MaxPool2D(X, pool, stride, pad);
        D.Log(X.shape + " > " + O.shape);
        O.PrintDataPart(32, Prefix + "MaxPool2D");
        return O;
    }
    Tensor IOps.AvgPool2D(Tensor X, int[] pool, int[] stride, int[] pad)
    {
        var O = m_Ops.AvgPool2D(X, pool, stride, pad);
        D.Log(X.shape + " ≥ " + O.shape);
        O.PrintDataPart(32, Prefix + "AvgPool2D");
        return O;
    }
    Tensor IOps.GlobalMaxPool2D(Tensor X)
    {
        var O = m_Ops.GlobalMaxPool2D(X);
        D.Log(X.shape + " >> " + O.shape);
        O.PrintDataPart(32, Prefix + "GlobalMaxPool2D");
        return O;
    }
    Tensor IOps.GlobalAvgPool2D(Tensor X)
    {
        var O = m_Ops.GlobalAvgPool2D(X);
        D.Log(X.shape + " ≥≥ " + O.shape);
        O.PrintDataPart(32, Prefix + "GlobalAvgPool2D");
        return O;
    }
    Tensor IOps.GlobalAvgVariancePool2D(Tensor X)
    {
        var O = m_Ops.GlobalAvgVariancePool2D(X);
        D.Log(X.shape + " ≥≥ " + O.shape);
        O.PrintDataPart(32, Prefix + "GlobalAvgVariancePool2D");
        return O;
    }
    Tensor IOps.Border2D(Tensor X, int[] pad, float value)
    {
        D.Log($"{X.shape} ¶(border) value={value} pad=[{pad[0]},{pad[1]},{pad[2]},{pad[3]})");
        var O = m_Ops.Border2D(X, pad, value);
        O.PrintDataPart(32, Prefix + "Border2D");
        return O;
    }
    Tensor IOps.Pad2DReflect(Tensor X, int[] pad)
    {
        D.Log($"{X.shape} ¶(reflect) pad=[{pad[0]},{pad[1]},{pad[2]},{pad[3]})");
        var O = m_Ops.Pad2DReflect(X, pad);
        O.PrintDataPart(32, Prefix + "Pad2DReflect");
        return O;
    }
    Tensor IOps.Pad2DSymmetric(Tensor X, int[] pad)
    {
        D.Log($"{X.shape} ¶(symmetric) pad=[{pad[0]},{pad[1]},{pad[2]},{pad[3]})");
        var O = m_Ops.Pad2DSymmetric(X, pad);
        O.PrintDataPart(32, Prefix + "Pad2DSymmetric");
        return O;
    }
    Tensor IOps.Pad2DEdge(Tensor X, int[] pad)
    {
        D.Log($"{X.shape} ¶(edge) pad=[{pad[0]},{pad[1]},{pad[2]},{pad[3]})");
        var O = m_Ops.Pad2DEdge(X, pad);
        O.PrintDataPart(32, Prefix + "Pad2DEdge");
        return O;
    }

    Tensor IOps.ScaleBias(Tensor X, Tensor S, Tensor B)
    {
        D.Log(X.shape + " * (" + S.channels + ") + (" + B.channels + ")");
        var O = m_Ops.ScaleBias(X, S, B);
        O.PrintDataPart(32, Prefix + "ScaleBias");
        return O;
    }
    Tensor IOps.Normalization(Tensor X, Tensor S, Tensor B, int pool, int axis, float epsilon, Layer.FusedActivation fusedActivation)
    {
        D.Log(X.shape + " ! " + (pool==1 ? "instance": "batch") + " axis=" + axis);
        var O = m_Ops.Normalization(X, S, B, pool, axis, epsilon, fusedActivation);
        O.PrintDataPart(32, Prefix + "Normalization");
        return O;
    }
    Tensor IOps.LRN(Tensor X, float alpha, float beta, float bias, int size)
    {
        D.Log(X.shape + " LRN n=" + size + " a=" + alpha + " b=" + beta + " bias=" + bias);
        var O = m_Ops.LRN(X, alpha, beta, bias, size);
        O.PrintDataPart(32, Prefix + "LRN");
        return O;
    }
    Tensor IOps.Dropout(Tensor X, float alpha)
    {
        D.Log(X.shape + "  a=" + alpha);
        var O = m_Ops.Dropout(X, alpha);
        O.PrintDataPart(32, Prefix + "Dropout");
        return O;
    }
    Tensor IOps.RandomNormal(TensorShape s, float mean, float scale, int seed)
    {
        D.Log(s + " N m=" + mean + " s=" + scale + " s=" + seed);
        var O = m_Ops.RandomNormal(s, mean, scale, seed);
        O.PrintDataPart(32, Prefix + "RandomNormal");
        return O;
    }
    Tensor IOps.RandomUniform(TensorShape s, float mean, float scale, int seed)
    {
        D.Log(s + " U m=" + mean + " s=" + scale + " s=" + seed);
        var O = m_Ops.RandomUniform(s, mean, scale, seed);
        O.PrintDataPart(32, Prefix + "RandomUniform");
        return O;
    }
    Tensor IOps.Multinomial(Tensor X, int count, int seed)
    {
        D.Log(X.shape + " M n=" + count + " s=" + seed);
        var O = m_Ops.Multinomial(X, count, seed);
        O.PrintDataPart(32, Prefix + "Multinomial");
        return O;
    }
    Tensor IOps.OneHot(Tensor X, int depth, float onValue, float offValue)
    {
        Debug.Log(X.shape + " Ω n=" + depth + " 1=" + onValue + " 0=" + offValue);
        var O = m_Ops.OneHot(X, depth, onValue, offValue);
        O.PrintDataPart(32, Prefix + "OneHot");
        return O;
    }

    Tensor IOps.Relu(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Relu(X);
        O.PrintDataPart(32, Prefix + "Relu");
        return O;
    }
    Tensor IOps.Softmax(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Softmax(X);
        O.PrintDataPart(32, Prefix + "Softmax");
        return O;
    }
    Tensor IOps.LogSoftmax(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.LogSoftmax(X);
        O.PrintDataPart(32, Prefix + "LogSoftmax");
        return O;
    }
    Tensor IOps.Tanh(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Tanh(X);
        O.PrintDataPart(32, Prefix + "Tanh");
        return O;
    }
    Tensor IOps.Sigmoid(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Sigmoid(X);
        O.PrintDataPart(32, Prefix + "Sigmoid");
        return O;
    }
    Tensor IOps.Relu6(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Relu6(X);
        O.PrintDataPart(32, Prefix + "Relu6");
        return O;
    }
    Tensor IOps.Elu(Tensor X, float alpha)
    {
        D.Log(X.shape + " () a=" + alpha);
        var O = m_Ops.Elu(X, alpha);
        O.PrintDataPart(32, Prefix + "Elu");
        return O;
    }
    Tensor IOps.LeakyRelu(Tensor X, float alpha)
    {
        D.Log(X.shape + " () a=" + alpha);
        var O = m_Ops.LeakyRelu(X, alpha);
        O.PrintDataPart(32, Prefix + "LeakyRelu");
        return O;
    }
    Tensor IOps.Selu(Tensor X, float alpha, float gamma)
    {
        D.Log(X.shape + " () a=" + alpha + " g=" + gamma);
        var O = m_Ops.Selu(X, alpha, gamma);
        O.PrintDataPart(32, Prefix + "Selu");
        return O;
    }
    Tensor IOps.PRelu(Tensor X, Tensor S)
    {
        D.Log(X.shape + " * (" + S.channels + ")");
        var O = m_Ops.PRelu(X, S);
        O.PrintDataPart(32, Prefix + "PRelu");
        return O;
    }
    Tensor IOps.Swish(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Swish(X);
        O.PrintDataPart(32, Prefix + "Swish");
        return O;
    }
    Tensor IOps.Abs(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Abs(X);
        O.PrintDataPart(32, Prefix + "Abs");
        return O;
    }
    Tensor IOps.Neg(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Neg(X);
        O.PrintDataPart(32, Prefix + "Neg");
        return O;
    }
    Tensor IOps.Ceil(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Ceil(X);
        O.PrintDataPart(32, Prefix + "Ceil");
        return O;
    }
    Tensor IOps.Clip(Tensor X, float min, float max)
    {
        D.Log(X.shape + " () min=" + min + " max=" + max);
        var O = m_Ops.Clip(X, min, max);
        O.PrintDataPart(32, Prefix + "Clip");
        return O;
    }
    Tensor IOps.Floor(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Floor(X);
        O.PrintDataPart(32, Prefix + "Floor");
        return O;
    }

    Tensor IOps.Reciprocal(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Reciprocal(X);
        O.PrintDataPart(32, Prefix + "Reciprocal");
        return O;
    }
    Tensor IOps.Pow(Tensor X, float alpha)
    {
        D.Log(X.shape + " () a=" + alpha);
        var O = m_Ops.Pow(X, alpha);
        O.PrintDataPart(32, Prefix + "Pow");
        return O;
    }
    Tensor IOps.Exp(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Exp(X);
        O.PrintDataPart(32, Prefix + "Exp");
        return O;
    }
    Tensor IOps.Log(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Log(X);
        O.PrintDataPart(32, Prefix + "Log");
        return O;
    }
    Tensor IOps.Sqrt(Tensor X)
    {
        D.Log(X.shape + " ()");
        var O = m_Ops.Sqrt(X);
        O.PrintDataPart(32, Prefix + "Sqrt");
        return O;
    }

    Tensor IOps.Add(Tensor[] tensors)
    {
        var O = m_Ops.Add(tensors);
        D.Log("{" + tensors.Length + "} + " + O.shape); // @TODO: print input dimensions
        O.PrintDataPart(32, Prefix + "Add");
        return O;
    }
    Tensor IOps.Sub(Tensor[] tensors)
    {
        var O = m_Ops.Sub(tensors);
        D.Log("{" + tensors.Length + "} - " + O.shape); // @TODO: print input dimensions
        O.PrintDataPart(32, Prefix + "Sub");
        return O;
    }
    Tensor IOps.Mul(Tensor[] tensors)
    {
        var O = m_Ops.Mul(tensors);
        D.Log("{" + tensors.Length + "} * " + O.shape); // @TODO: print input dimensions
        O.PrintDataPart(32, Prefix + "Mul");
        return O;
    }
    Tensor IOps.Div(Tensor[] tensors)
    {
        var O = m_Ops.Div(tensors);
        D.Log("{" + tensors.Length + "} / " + O.shape); // @TODO: print input dimensions
        O.PrintDataPart(32, Prefix + "Div");
        return O;
    }
    Tensor IOps.Pow(Tensor[] tensors)
    {
        var O = m_Ops.Pow(tensors);
        D.Log("{" + tensors.Length + "} ^ " + O.shape); // @TODO: print input dimensions
        O.PrintDataPart(32, Prefix + "Pow");
        return O;
    }
    Tensor IOps.Min(Tensor[] tensors)
    {
        var O = m_Ops.Min(tensors);
        D.Log("{" + tensors.Length + "} < " + O.shape); // @TODO: print input dimensions
        O.PrintDataPart(32, Prefix + "Min");
        return O;
    }
    Tensor IOps.Max(Tensor[] tensors)
    {
        var O = m_Ops.Max(tensors);
        D.Log("{" + tensors.Length + "} > " + O.shape); // @TODO: print input dimensions
        O.PrintDataPart(32, Prefix + "Max");
        return O;
    }
    Tensor IOps.Mean(Tensor[] tensors)
    {
        var O = m_Ops.Mean(tensors);
        D.Log("{" + tensors.Length + "} ∑ " + O.shape); // @TODO: print input dimensions
        O.PrintDataPart(32, Prefix + "Mean");
        return O;
    }

    Tensor IOps.ReduceMax(Tensor X, int axis)
    {
        var O = m_Ops.ReduceMax(X, axis);
        D.Log(X.shape + " .> " + O.shape);
        O.PrintDataPart(32, Prefix + "ReduceMax");
        return O;
    }
    Tensor IOps.ReduceMean(Tensor X, int axis)
    {
        var O = m_Ops.ReduceMean(X, axis);
        D.Log(X.shape + " .∑ " + O.shape);
        O.PrintDataPart(32, Prefix + "ReduceMean");
        return O;
    }
    Tensor IOps.ReduceMin(Tensor X, int axis)
    {
        var O = m_Ops.ReduceMin(X, axis);
        D.Log(X.shape + " .< " + O.shape);
        O.PrintDataPart(32, Prefix + "ReduceMin");
        return O;
    }
    Tensor IOps.ReduceProd(Tensor X, int axis)
    {
        var O = m_Ops.ReduceProd(X, axis);
        D.Log(X.shape + " .* " + O.shape);
        O.PrintDataPart(32, Prefix + "ReduceProd");
        return O;
    }
    Tensor IOps.ReduceSum(Tensor X, int axis)
    {
        var O = m_Ops.ReduceSum(X, axis);
        D.Log(X.shape + " .+ " + O.shape);
        O.PrintDataPart(32, Prefix + "ReduceSum");
        return O;
    }
    Tensor IOps.Greater(Tensor a, Tensor b)
    {
        var O = m_Ops.Greater(a, b);
        D.Log(a.shape + " > " + b.shape + " = " + O.shape);
        O.PrintDataPart(32, Prefix + "Greater");
        return O;
    }
    Tensor IOps.GreaterEqual(Tensor a, Tensor b)
    {
        var O = m_Ops.GreaterEqual(a, b);
        D.Log(a.shape + " >= " + b.shape + " = " + O.shape);
        O.PrintDataPart(32, Prefix + "GreaterEqual");
        return O;
    }
    Tensor IOps.Less(Tensor a, Tensor b)
    {
        var O = m_Ops.Less(a, b);
        D.Log(a.shape + " < " + b.shape + " = " + O.shape);
        O.PrintDataPart(32, Prefix + "Less");
        return O;
    }
    Tensor IOps.LessEqual(Tensor a, Tensor b)
    {
        var O = m_Ops.LessEqual(a, b);
        D.Log(a.shape + " <= " + b.shape + " = " + O.shape);
        O.PrintDataPart(32, Prefix + "LessEqual");
        return O;
    }
    Tensor IOps.Equal(Tensor a, Tensor b)
    {
        var O = m_Ops.Equal(a, b);
        D.Log(a.shape + " == " + b.shape + " = " + O.shape);
        O.PrintDataPart(32, Prefix + "Equal");
        return O;
    }
    Tensor IOps.LogicalOr(Tensor a, Tensor b)
    {
        var O = m_Ops.LogicalOr(a, b);
        D.Log(a.shape + " || " + b.shape + " = " + O.shape);
        O.PrintDataPart(32, Prefix + "LogicalOr");
        return O;
    }
    Tensor IOps.LogicalAnd(Tensor a, Tensor b)
    {
        var O = m_Ops.LogicalAnd(a, b);
        D.Log(a.shape + " && " + b.shape + " = " + O.shape);
        O.PrintDataPart(32, Prefix + "LogicalAnd");
        return O;
    }
    Tensor IOps.LogicalXor(Tensor a, Tensor b)
    {
        var O = m_Ops.LogicalXor(a, b);
        D.Log(a.shape + " ^ " + b.shape + " = " + O.shape);
        O.PrintDataPart(32, Prefix + "LogicalXor");
        return O;
    }
    Tensor IOps.LogicalNot(Tensor x)
    {
        var O = m_Ops.LogicalNot(x);
        D.Log("!(" + x.shape +" )");
        O.PrintDataPart(32, Prefix + "LogicalNot");
        return O;
    }

    Tensor IOps.Flatten(Tensor X)
    {
        var O = m_Ops.Flatten(X);
        D.Log(X.shape + " = " + O.shape);
        return O;
    }
    Tensor IOps.Reshape(Tensor X, TensorShape shape)
    {
        var O = m_Ops.Reshape(X, shape);
        D.Log(X.shape + " $ " + O.shape);
        return O;
    }
    Tensor IOps.Transpose(Tensor X)
    {
        var O = m_Ops.Transpose(X);
        D.Log(X.shape + " T " + O.shape);
        return O;
    }
    Tensor IOps.Gather(Tensor[] tensors, int axis)
    {
        var O = m_Ops.Gather(tensors,axis);
        D.Log("{" + tensors[0].shape + "," + tensors[1].shape + "," + axis + "} # " + O.shape);
        O.PrintDataPart(32, Prefix + "Gather");
        return O;
    }
    Tensor IOps.Concat(Tensor[] tensors, int axis)
    {
        var O = m_Ops.Concat(tensors, axis);
        D.Log("{" + tensors.Length + "} # " + O.shape); // @TODO: print input dimensions
        O.PrintDataPart(32, Prefix + "Concat");
        return O;
    }
    Tensor IOps.StridedSlice(Tensor X, int[] starts, int[] ends, int[] strides)
    {
        var O = m_Ops.StridedSlice(X, starts, ends, strides);
        D.Log(X.shape + " | " + O.shape);
        O.PrintDataPart(32, Prefix + "StridedSlice");
        return O;
    }
    Tensor IOps.Tile(Tensor X, int[] repeats)
    {
        var O = m_Ops.Tile(X, repeats);
        D.Log(X.shape + " % " + O.shape);
        O.PrintDataPart(32, Prefix + "Tile");
        return O;
    }
    Tensor IOps.Copy(Tensor x)
    {
        var O = m_Ops.Copy(x);
        D.Log("!(" + x.shape +" )");
        O.PrintDataPart(32, "Copy");
        return O;
    }

    Tensor IOps.Prepare(Tensor X)
    {
        D.Log("!" + X.shape);
        return m_Ops.Prepare(X);
    }

    void IOps.ResetAllocator(bool keepCachedMemory)
    {
        m_Ops.ResetAllocator(keepCachedMemory);
    }
}


} // namespace Unity.Barracuda
