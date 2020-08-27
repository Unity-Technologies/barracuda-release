using System;
using System.Collections.Generic;

namespace Unity.Barracuda {

/// <summary>
/// Interfaces for backend implementers
/// see ModelBuilder.cs for detail on layers.
/// </summary>
public interface IOps
{
    Tensor MatMul(Tensor x, bool xTranspose, Tensor y, bool yTranspose);// @TODO: consider MatMulAdd instead
    Tensor Dense(Tensor x, Tensor w, Tensor b, Layer.FusedActivation fusedActivation);
    Tensor Conv2D(Tensor x, Tensor k, Tensor b, int[] stride, int[] pad, Layer.FusedActivation fusedActivation);
    Tensor DepthwiseConv2D(Tensor x, Tensor k, Tensor b, int[] stride, int[] pad, Layer.FusedActivation fusedActivation);
    Tensor Conv2DTrans(Tensor x, Tensor k, Tensor b, int[] stride, int[] pad, int[] outputAdjustment, Layer.FusedActivation fusedActivation);
    Tensor Upsample2D(Tensor x, int[] scale, bool bilinear);
    Tensor Resample2D(Tensor x, int[] size, bool bilinear);
    Tensor DepthToSpace(Tensor x, int[] scale, Layer.DepthToSpaceMode mode);
    Tensor SpaceToDepth(Tensor x, int[] scale);
    Tensor MaxPool2D(Tensor x, int[] pool, int[] stride, int[] pad);
    Tensor AvgPool2D(Tensor x, int[] pool, int[] stride, int[] pad);
    Tensor GlobalMaxPool2D(Tensor x); // @TODO: consider, if it should be just a special case of MaxPool2D with {pool=X.width/height, stride=1}
    Tensor GlobalAvgPool2D(Tensor x);
    Tensor GlobalAvgVariancePool2D(Tensor x);
    Tensor Border2D(Tensor x, int[] pad, float borderValue);
    Tensor Pad2DReflect(Tensor x, int[] pad);
    Tensor Pad2DSymmetric(Tensor x, int[] pad);
    Tensor Pad2DEdge(Tensor x, int[] pad);

    Tensor ScaleBias(Tensor x, Tensor s, Tensor b);
    Tensor Normalization(Tensor x, Tensor s, Tensor b, int pool, int axis, float epsilon, Layer.FusedActivation fusedActivation);
    Tensor LRN(Tensor x, float alpha, float beta, float bias, int size);
    Tensor Dropout(Tensor x, float alpha);
    Tensor RandomNormal(TensorShape s, float mean, float scale, int seed);
    Tensor RandomUniform(TensorShape s, float mean, float scale, int seed);
    Tensor Multinomial(Tensor x, int count, int seed);
    Tensor OneHot(Tensor x, int depth, float onValue, float offValue);
    Tensor TopKIndices(Tensor x, int k, int axis, bool largest, bool sorted);
    Tensor TopKValues(Tensor X, Tensor I, int axis);

    Tensor Relu(Tensor x);
    Tensor Softmax(Tensor x);
    Tensor LogSoftmax(Tensor x);
    Tensor Tanh(Tensor x);
    Tensor Sigmoid(Tensor x);
    Tensor Elu(Tensor x, float alpha);
    Tensor Relu6(Tensor x);
    Tensor LeakyRelu(Tensor x, float alpha);
    Tensor Selu(Tensor x, float alpha, float gamma);
    Tensor PRelu(Tensor x, Tensor alpha);
    Tensor Swish(Tensor x);
    Tensor Abs(Tensor x);
    Tensor Neg(Tensor x);
    Tensor Ceil(Tensor x);
    Tensor Clip(Tensor x, float min, float max);
    Tensor Floor(Tensor x);

    Tensor Reciprocal(Tensor x);
    Tensor Pow(Tensor x, float alpha);
    Tensor Exp(Tensor x);
    Tensor Log(Tensor x);
    Tensor Sqrt(Tensor x);
    Tensor Acos(Tensor x);
    Tensor Acosh(Tensor x);
    Tensor Asin(Tensor x);
    Tensor Asinh(Tensor x);
    Tensor Atan(Tensor x);
    Tensor Atanh(Tensor x);
    Tensor Cos(Tensor x);
    Tensor Cosh(Tensor x);
    Tensor Sin(Tensor x);
    Tensor Sinh(Tensor x);
    Tensor Tan(Tensor x);

    Tensor Add(Tensor[] tensors);
    Tensor Sub(Tensor[] tensors);
    Tensor Mul(Tensor[] tensors);
    Tensor Div(Tensor[] tensors);
    Tensor Pow(Tensor[] tensors);
    Tensor Min(Tensor[] tensors);
    Tensor Max(Tensor[] tensors);
    Tensor Mean(Tensor[] tensors);

    Tensor ReduceMax(Tensor x, int axis);
    Tensor ReduceMean(Tensor x, int axis);
    Tensor ReduceMin(Tensor x, int axis);
    Tensor ReduceProd(Tensor x, int axis);
    Tensor ReduceSum(Tensor x, int axis);

    Tensor Greater(Tensor a, Tensor b);
    Tensor GreaterEqual(Tensor a, Tensor b);
    Tensor Less(Tensor a, Tensor b);
    Tensor LessEqual(Tensor a, Tensor b);
    Tensor Equal(Tensor a, Tensor b);
    Tensor LogicalOr(Tensor a, Tensor b);
    Tensor LogicalAnd(Tensor a, Tensor b);
    Tensor LogicalXor(Tensor a, Tensor b);
    Tensor LogicalNot(Tensor x);

    Tensor Flatten(Tensor x);
    Tensor Reshape(Tensor x, TensorShape shape);
    Tensor Expand(Tensor x, TensorShape shape);
    Tensor Transpose(Tensor x);
    Tensor Transpose(Tensor x, int[] permutations);
    Tensor Concat(Tensor[] tensors, int axis);
    Tensor StridedSlice(Tensor x, int[] starts, int[] ends, int[] stride);
    Tensor Tile(Tensor x, int[] repeats);
    Tensor Gather(Tensor[] tensors, int axis);

    Tensor Copy(Tensor x);

    /// <summary>
    /// Prepares tensor for use
    /// </summary>
    Tensor Prepare(Tensor x);

    /// <summary>
    /// Reset internal allocator
    /// </summary>
    void ResetAllocator(bool keepCachedMemory = true);
}

/// <summary>
/// Interfaces for model compiler
/// </summary>
internal interface IModelCompiler
{
    void PrepareModel(Model model, IDictionary<string, TensorShape> inputShapes);
    void PreExecuteLayer(Layer layer, Tensor[] inputs);
}

/// <summary>
/// Interfaces for variables
/// </summary>
public interface IVars : IDisposable
{
    void SetInput(string name, Tensor x);
    void PrepareStorage(Model model, IOps optionalOpsToPrepareTensors = null, IDictionary<string, TensorShape> optionalInputShapes = null);
    Tensor[] GatherInputs(Layer forLayer);
    void PrepareStorage(Layer forLayer);
    void Store(Layer fromLayer, Tensor result);
    Tensor PeekOutput(string name);
    Tensor[] PeekConstants(string layerName);

    ITensorAllocator GetAllocator();
}

/// <summary>
/// Interfaces for tensor allocator
/// </summary>
public interface ITensorAllocator : IDisposable
{
    Tensor Alloc(TensorShape shape);
    Tensor Alloc(TensorShape shape, ITensorData buffer);

    // MoveToDevice() callback is called from the following Tensor methods:
    // UploadToDevice(), AttachToDevice() and DetachFromDevice()
    void MoveToDevice(Tensor x, ITensorData newBuffer, ITensorData oldBuffer, bool disposeDetachedBufferHint);

    // NOTE: Release() should be ready to handle edge-case situation when
    //  externally created new Tensor instance is passed with
    //  ITensorData (tensorOnDevice) that is already owned by the allocator
    void Release(Tensor x, bool calledFromTensorDispose);

    void WaiveOwnership(Tensor x);
    void Reset(bool keepCachedMemory); // end-of-frame
}

} // namespace Unity.Barracuda
