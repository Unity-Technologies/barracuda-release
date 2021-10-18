using System;
using System.Collections.Generic;
using System.Linq;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Assertions;

namespace Unity.Barracuda {

/// <summary>
/// Tensor extension methods
/// </summary>
public static class TensorExtensions
{
    static internal void TestInit(this Tensor X, int n = -1, int modulus = -1)
    {
        if (n < 0)
            n = X.length;
        n = Math.Min(n, X.length);
        for (int i = 0; i < n; ++i)
        {
            if (modulus > 1)
                X[i] = i % modulus;
            else
                X[i] = i;
        }
    }

    static internal void TestInitCos(this Tensor X, int n = -1, float offset = 0.0f)
    {
        if (n < 0)
            n = X.length;
        n = Math.Min(n, X.length);
        for (int i = 0; i < n; ++i)
            X[i] = Mathf.Cos(i + offset);
    }

    static internal void TestInitRandom(this Tensor X, int n = -1)
    {
        if (n < 0)
            n = X.length;
        n = Math.Min(n, X.length);
        for (int i = 0; i < n; ++i)
            X[i] = UnityEngine.Random.value;
    }

    static internal void TestInitValue(this Tensor X, float value=0.1f, int n = -1)
    {
        if (n < 0)
            n = X.length;
        n = Math.Min(n, X.length);
        for (int i = 0; i < n; ++i)
            X[i] = value;
    }

    /// <summary>
    /// Return Tensor data as float array, this will create a blocking read operation
    /// </summary>
    /// <param name="x">Tensor</param>
    /// <returns>Tensor data as float array</returns>
    static public float[] AsFloats(this Tensor x)
    {
        return x.ToReadOnlyArray();
    }

    /// <summary>
    /// Return Tensor data as int array (slow operation), this will create a blocking read operation
    /// </summary>
    /// <param name="x">Tensor</param>
    /// <returns>Tensor data as int array</returns>
    static public int[] AsInts(this Tensor x)
    {
        return Array.ConvertAll(x.ToReadOnlyArray(), v => v <= (float)int.MinValue ? int.MinValue : v >= (float)int.MaxValue ? int.MaxValue : (int)v);
    }

    /// <summary>
    /// Return Tensor data as string, limits number of elements to `size`
    /// </summary>
    /// <param name="X">Tensor</param>
    /// <param name="size">element number limit</param>
    /// <returns>Returns Tensor data as string</returns>
    static public string DataToString(this Tensor X, int size = 32)
    {
        var str = "";
        for (int i = 0; i < X.length && i < size; ++i)
        {
            str += X[i];
            str += " ";
        }
        if (X.length > size)
            str += "...";
        return str;
    }

    /// <summary>
    /// Print Tensor metadata to console
    /// </summary>
    /// <param name="X">Tensor</param>
    /// <param name="msg">message prefix</param>
    static public void Print(this Tensor X, string msg = "")
    {
        if (msg.Length > 0)
            msg += " ";
        D.Log($"{msg}{X.name} {X.shape}");
    }

    /// <summary>
    /// Print Tensor data to console
    /// </summary>
    /// <param name="X">Tensor</param>
    /// <param name="size">element number limit</param>
    /// <param name="msg">message prefix</param>
    static public void PrintDataPart(this Tensor X, int size, string msg = "")
    {
        if (msg.Length > 0)
            msg += " ";
        D.Log($"{msg}{X.DataToString(size)}");
    }

    /// <summary>
    /// Compare Tensor contents
    /// </summary>
    /// <param name="X">left Tensor</param>
    /// <param name="Y">right Tensor</param>
    /// <returns>`true` if shape and data content matches</returns>
    static public bool Equals(this Tensor X, Tensor Y)
    {
        if (X.shape != Y.shape)
            return false;

        if (X.length != Y.length)
            return false;

        for (int i = 0; i < X.length; ++i)
        {
            if (X[i] != Y[i])
                return false;
        }

        return true;
    }

    /// <summary>
    /// Compare Tensor contents approximately
    /// </summary>
    /// <param name="X">left Tensor</param>
    /// <param name="Y">right Tensor</param>
    /// <param name="epsilon">comparison threshold</param>
    /// <param name="count">limit number of elements to compare</param>
    /// <returns>`true` if shape match and while data content matches approximately</returns>
    static public bool Approximately(this Tensor X, Tensor Y, float epsilon = 1e-4f, int count = -1)
    {
        if (X.shape != Y.shape)
            return false;

        if (X.length != Y.length)
            return false;

        if (count < 0)
            count = X.length;
        for (int i = 0; i < count; ++i)
        {
            // If one of the values is NaN, the comparison against epislon will return false.
            // But if tensor has NaN and the other doesn't, they shouldn't be considered "close".
            if (Mathf.Abs(X[i] - Y[i]) > epsilon || float.IsNaN(X[i]) != float.IsNaN(Y[i]))
            {
                // @TODO: move logging into dedicated function
                D.Log("First mismatch @ [" + i + "]: " + X[i] + " != " + Y[i]);
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Calculate max difference between two tensors
    /// </summary>
    /// <param name="X">first Tensor</param>
    /// <param name="Y">second Tensor</param>
    /// <returns></returns>
    static public float MaxDifference(this Tensor X, Tensor Y)
    {
        float maxD = 0f;
        for (int i = 0; i < X.length; ++i)
            maxD = Mathf.Max(Mathf.Abs(X[i] - Y[i]), maxD);
        return maxD;
    }

    /// <summary>
    /// Reshape Tensor
    /// </summary>
    /// <param name="X">Tensor</param>
    /// <param name="size">new shape as array of int (expected as size 4 for NHWC or size 8 for SRNTDHWC)</param>
    /// <returns>reshaped Tensor</returns>
    static public Tensor Reshape(this Tensor X, int[] size)
    {
        var newShape = X.shape.Reshape(size);
        return X.Reshape(newShape);
    }

    /// <summary>
    /// Calculate max value index
    /// </summary>
    /// <param name="X">Tensor</param>
    /// <returns>max value index</returns>
    static public int[] ArgMax(this Tensor X)
    {
        Assert.AreEqual(TensorShape.DataChannel, TensorShape.MaxRank - 1); // expects channels last layout
        Assert.IsTrue(X.channels != 0);
        Assert.AreEqual(X.length % X.channels, 0);

        // reduce over the last dimension - channels
        var innerLength = X.channels;
        var outterLength = X.length / innerLength;

        int[] result = new int[outterLength];
        for (var n = 0; n < outterLength; ++n)
        {
            float maxV = Mathf.NegativeInfinity;
            for (int c = 0; c < innerLength; ++c)
            {
                var v = X[n * innerLength + c];
                if (maxV >= v)
                    continue;
                maxV = v;
                result[n] = c;
            }
        }
        return result;
    }

    /// <summary>
    /// Return indices in order that would produce sorted Tensor values
    /// </summary>
    /// <param name="X">Tensor</param>
    /// <returns>indices in order that would produce sorted Tensor values</returns>
    static public int[][] ArgSort(this Tensor X)
    {
        Assert.AreEqual(TensorShape.DataChannel, TensorShape.MaxRank - 1); // expects channels last layout
        Assert.IsTrue(X.channels != 0);
        Assert.AreEqual(X.length % X.channels, 0);

        // reduce over the last dimension - channels
        var innerLength = X.channels;
        var outterLength = X.length / innerLength;

        var result = new List<int[]>();
        for (var n = 0; n < outterLength; ++n)
        {
            int[] indices = Enumerable.Range(0, innerLength).ToArray<int>();

            var sliceOffset = n * innerLength;
            Array.Sort<int>(indices, (a, b) => X[sliceOffset + a].CompareTo(X[sliceOffset + b]));
            result.Add(indices);
        }
        return result.ToArray();
    }

    /// <summary>
    /// Fill Tensor with `value`
    /// </summary>
    /// <param name="X">Tensor</param>
    /// <param name="value">value</param>
    public static void Fill(this Tensor X, float value)
    {
        for (int i = 0; i < X.length; ++i)
            X[i] = value;
    }

    /// <summary>
    /// Calculate output shape for Gather operation
    /// </summary>
    /// <param name="shapes">input shapes</param>
    /// <param name="axis">axis</param>
    /// <returns>output shape</returns>
    static public TensorShape Gather(TensorShape[] shapes, int axis)
    {
        TensorShape shape = shapes[0];
        if (!shape.hasNamedDimensions)
            shape = shape.AsNamed();

        TensorShape indices = shapes[1];
        if (!indices.hasNamedDimensions)
            indices = indices.AsNamed();

        shape[axis] = indices.length;

        return shape;
    }

    /// <summary>
    /// Concatenate `Tensor` array along `axis` and calculate output shape
    /// </summary>
    /// <param name="tensors">Tensor array</param>
    /// <param name="axis">axis</param>
    /// <returns>new `TensorShape`</returns>
    /// <exception cref="ArgumentException">Off-axis dimension mismatch</exception>
    static public TensorShape Concat(Tensor[] tensors, int axis)
    {
        if (tensors.Length == 0)
            return new TensorShape();

        var a = tensors[0].shape;
        if (!a.hasNamedDimensions)
            a = a.AsNamed();
        var aAxis = a.Axis(axis);

        // validate that off axis dimensions are equal
        for (var i = 1; i < tensors.Length; ++i)
        {
            var b = tensors[i].shape;
            if (!b.hasNamedDimensions)
                b = b.AsNamed();

            var bAxis = b.Axis(axis);
            a[aAxis] = 0; b[bAxis] = 0;
            if (a != b)
            {
                foreach (var s in tensors)
                    D.Log(s.shape);
                throw new ArgumentException("Off-axis dimensions must match");
            }
        }

        var shape = tensors[0].shape;
        if (!shape.hasNamedDimensions)
            shape = shape.AsNamed();

        var dstAxis = tensors[0].shape.Axis(axis);
        for (var i = 1; i < tensors.Length; ++i)
        {
            var otherShape = tensors[i].shape;
            if (!otherShape.hasNamedDimensions)
                otherShape = otherShape.AsNamed();

            shape[dstAxis] += otherShape[axis];
        }

        return shape;
    }

    /// <summary>
    /// Calculate concatenation output shape
    /// </summary>
    /// <param name="shapes">input shapes</param>
    /// <param name="axis">concatenation axis</param>
    /// <returns>output shape</returns>
    /// <exception cref="ArgumentException">Off-axis dimension mismatch</exception>
    static public TensorShape Concat(TensorShape[] shapes, int axis)
    {
        if (shapes.Length == 0)
            return new TensorShape();

        var a = shapes[0];
        if (!a.hasNamedDimensions)
            a = a.AsNamed();
        var aAxis = a.Axis(axis);

        // validate that off axis dimensions are equal
        for (var i = 1; i < shapes.Length; ++i)
        {

            var b = shapes[i];
            if (!b.hasNamedDimensions)
                b = b.AsNamed();

            var bAxis = b.Axis(axis);
            a[aAxis] = 0; b[bAxis] = 0;
            if (a != b)
            {
                foreach (var s in shapes)
                    D.Log(s);
                throw new ArgumentException("Off-axis dimensions must match");
            }
        }

        var shape = shapes[0];
        if (!shape.hasNamedDimensions)
            shape = shape.AsNamed();

        var dstAxis = shape.Axis(axis);
        for (var i = 1; i < shapes.Length; ++i)
        {
            var otherShape = shapes[i];
            if (!otherShape.hasNamedDimensions)
                otherShape = otherShape.AsNamed();

            shape[dstAxis] += otherShape[axis];
        }

        return shape;
    }

    /// <summary>
    /// Calculate maximum shape that would cover all input shapes
    /// </summary>
    /// <param name="shapes">input shapes</param>
    /// <returns>output shape</returns>
    static public TensorShape Max(TensorShape[] shapes)
    {
        Assert.IsTrue(shapes.Length > 0);

        var shape = shapes[0];

        if (!shape.hasNamedDimensions)
            shape = shape.AsNamed();

        for (var i = 1; i < shapes.Length; ++i)
        {
            var otherShape = shapes[i];
            if (!otherShape.hasNamedDimensions)
                otherShape = otherShape.AsNamed();

            for (var axis = 0; axis < TensorShape.MaxRank; axis++)
            {
                shape[axis] = Math.Max(shape[axis], otherShape[axis]);
            }
        }

        return shape;
    }

    /// <summary>
    /// Calculate maximum shape that would cover all input tensors
    /// </summary>
    /// <param name="tensors">input tensors</param>
    /// <returns>output shape</returns>
    static public TensorShape MaxShape(Tensor[] tensors)
    {
        Assert.IsTrue(tensors.Length > 0);
        var shape = tensors[0].shape;

        if (!shape.hasNamedDimensions)
            shape = shape.AsNamed();

        for (var i = 1; i < tensors.Length; ++i)
        {
            for (var axis = 0; axis < TensorShape.MaxRank; axis++)
            {
                var otherShape = tensors[i].shape;
                if (!otherShape.hasNamedDimensions)
                    otherShape = otherShape.AsNamed();

                shape[axis] = Math.Max(shape[axis], otherShape[axis]);
            }
        }

        return shape;
    }

    /// <summary>
    /// Scale TensorShape by the `scale` factor
    /// </summary>
    /// <param name="shape">TensorShape</param>
    /// <param name="scale">scale</param>
    /// <returns>output shape</returns>
    static public TensorShape Scale(this TensorShape shape, TensorShape scale)
    {
        if (!shape.hasNamedDimensions)
            shape = shape.AsNamed();

        var newShape = shape;
        for (var axis = 0; axis < TensorShape.MaxRank; axis++)
            newShape[axis] *= scale[axis];
        return newShape;
    }

    /// <summary>
    /// Scale TensorShape by the `scale` factor
    /// </summary>
    /// <param name="shape">TensorShape</param>
    /// <param name="scale">scale</param>
    /// <returns>output shape</returns>
    static public TensorShape Scale(this TensorShape shape, int[] scale)
    {
        if (!shape.hasNamedDimensions)
            shape = shape.AsNamed();

        if (scale.Length == TensorShape.MaxRank)
        {
            for (var axis = 0; axis < TensorShape.MaxRank; axis++)
                shape[axis] *= scale[axis];
        }
        else
        {
            Assert.AreEqual(4, scale.Length);
            shape[TensorShape.DataBatch] *= scale[0];
            shape[5] *= scale[1];
            shape[6] *= scale[2];
            shape[7] *= scale[3];
        }
        return shape;
    }

    /// <summary>
    /// Reduce TensorShape across specified `axis`
    /// </summary>
    /// <param name="shape">TensorShape</param>
    /// <param name="axis">axis</param>
    /// <returns>output shape</returns>
    static public TensorShape Reduce(this TensorShape shape, int axis)
    {
        if (!shape.hasNamedDimensions)
            shape = shape.AsNamed();

        axis = shape.Axis(axis);
        var newShapeArray = shape;
        newShapeArray[axis] = 1;
        return newShapeArray;
    }

    /// <summary>
    /// Reshape TensorShape into new shape specified by `size`. At most one dimension of the new shape can be -1.
    /// See: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reshape
    /// </summary>
    /// <param name="shape">TensorShape</param>
    /// <param name="size4Dor8D">new shape</param>
    /// <returns>output shape</returns>
    /// <exception cref="ArgumentException">more than one dimension is unspecified</exception>
    static public TensorShape Reshape(this TensorShape shape, int[] size4Dor8D)
    {
        if (!shape.hasNamedDimensions)
            shape = shape.AsNamed();

        unsafe
        {
            int* size = stackalloc int[TensorShape.MaxRank];
            int* newShapeArray = stackalloc int[TensorShape.MaxRank];

            Get8DParametersNoAlloc(shape, size4Dor8D, size, 1);
            for (int d = 0; d < TensorShape.MaxRank; ++d)
                newShapeArray[d] = shape[d];

            // From: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reshape
            //
            // At most one dimension of the new shape can be -1.
            // In this case, the value is inferred from the size of the tensor and the remaining dimensions.
            //
            // A dimension could also be 0,
            // in which case the actual dimension value is unchanged (i.e. taken from the input tensor).

            var multipleOf = 1;
            var unknownIndex = -1;
            for (int q = 0; q < TensorShape.MaxRank; ++q)
            {
                if (size[q] > 0)
                {
                    multipleOf *= size[q];
                    newShapeArray[q] = size[q];
                }
                else if (size[q] == 0)
                    multipleOf *= newShapeArray[q];
                else if (unknownIndex == -1)
                    unknownIndex = q;
                else
                    throw new ArgumentException("Can only specify one unknown dimension");
            }

            if (unknownIndex == -1)
            {
                // all dimensions are given
                var newShape = new TensorShape(newShapeArray[0], newShapeArray[1], newShapeArray[2], newShapeArray[3],
                                               newShapeArray[4], newShapeArray[5], newShapeArray[6], newShapeArray[7]);
                if (shape.length != newShape.length)
                    throw new ArgumentException("Cannot reshape array of size " + shape.length +
                                                " into shape " + newShape);
                return newShape;
            }

            var solveForIndex = shape.length / multipleOf;
            bool remainderLeft = shape.length % multipleOf != 0;

            if (remainderLeft)
                throw new ArgumentException("Cannot reshape array of size " + shape.length +
                                            " into shape with multiple of " + multipleOf + " elements");

            newShapeArray[unknownIndex] = solveForIndex;
            return new TensorShape(newShapeArray[0], newShapeArray[1], newShapeArray[2], newShapeArray[3],
                                   newShapeArray[4], newShapeArray[5], newShapeArray[6], newShapeArray[7]);
        }
    }

    /// <summary>
    /// Calculate new shape after applying border to current TensorShape
    /// </summary>
    /// <param name="shape">TensorShape</param>
    /// <param name="border">border</param>
    /// <returns>new TensorShape</returns>
    static public TensorShape ApplyBorder(this TensorShape shape, int[] border)
    {
        if (!shape.hasNamedDimensions)
            shape = shape.AsNamed();

        Assert.IsTrue(border.Length == 6 || border.Length == 8);
        if(border.Length == 6)
        {
            shape[TensorShape.H] += border[1] + border[4];
            shape[TensorShape.W] += border[0] + border[3];
            shape[TensorShape.C] += border[2] + border[5];
        }
        else if (border.Length == 8)
        {
            shape[TensorShape.D] += border[2] + border[6];
            shape[TensorShape.H] += border[1] + border[5];
            shape[TensorShape.W] += border[0] + border[4];
            shape[TensorShape.C] += border[3] + border[7];
        }

        return shape;
    }

    static internal int[] AdjustPadToKernel(this Tensor tensor, Tensor kernel, int[] stride, int[] pad)
    {
        return AdjustPadToKernel(tensor.shape, kernel.shape, stride, pad);
    }

    static internal int[] AdjustPadToKernel(this TensorShape shape, TensorShape kernel, int[] stride, int[] pad)
    {
        if (!shape.hasNamedDimensions)
            shape = shape.AsNamed();

        Assert.IsTrue(stride.Length==2 || stride.Length==3);
        unsafe
        {
            int* kernelDims = stackalloc int[stride.Length == 2 ? 2 : 3];
            kernelDims[0] = kernel.kernelWidth;
            kernelDims[1] = kernel.kernelHeight;

            if (stride.Length > 2)
                kernelDims[2] = kernel.kernelSpatialDepth;

            return AdjustPadToPool(shape, kernelDims, stride, pad);
        }
    }

    static internal int[] AdjustPadToPool(this Tensor tensor, int[] pool, int[] stride, int[] pad)
    {
        return AdjustPadToPool(tensor.shape, pool, stride, pad);
    }

    static internal unsafe int[] AdjustPadToPool(this Tensor tensor, int* pool, int[] stride, int[] pad)
    {
        return AdjustPadToPool(tensor.shape, pool, stride, pad);
    }

    static internal int[] AdjustPadToPool(this TensorShape shape, int[] pool, int[] stride, int[] pad)
    {
        if (!shape.hasNamedDimensions)
            shape = shape.AsNamed();

        unsafe
        {
            fixed (int* pPool = pool)
            {
                return AdjustPadToPool(shape, pPool, stride, pad);
            }
        }
    }

    static internal unsafe int[] AdjustPadToPool(this TensorShape shape, int* pool, int[] stride, int[] pad)
    {
        if (!shape.hasNamedDimensions)
            shape = shape.AsNamed();

        Assert.IsTrue(stride.Length > 0);
        int featureCount = stride.Length;
        Assert.IsTrue(featureCount <= TensorShape.DataFeatures.Length);

        // negative pad values mean auto_pad type is used
        if (pad[0] >= 0)
            return pad;

        var type = (Layer.AutoPad)pad[0];
        if (type == Layer.AutoPad.SameUpper || type == Layer.AutoPad.SameLower)
        {
            // Based on ONNX (AveragePool & MaxPool)
            //        https://github.com/onnx/onnx/blob/master/docs/Operators.md
            // and TensorFlow docs:
            //         https://www.tensorflow.org/api_guides/python/nn#Notes_on_SAME_Convolution_Padding
            var adjustedPad = new int [featureCount*2];
            for (var i = 0; i < featureCount; ++i)
            {
                var featureModStride = shape.width % stride[i];
                if (featureModStride == 0)
                    featureModStride = stride[i];

                var padAlongFeature = Math.Max(pool[i] - featureModStride, 0);
                // Code above (based on TensorFlow docs) is equivalent to (based on ONNX docs):
                // padAlongWidth = (Mathf.Ceil(shape.width/stride[0]) - 1) * stride[0] + pool[0] - shape.width;
                // padAlongHeight = (Mathf.Ceil(shape.height/stride[1]) - 1) * stride[1] + pool[1] - shape.height;
                var featureSmall = padAlongFeature / 2;
                var featureLarge = padAlongFeature - featureSmall;
                if (type == Layer.AutoPad.SameUpper) {
                    adjustedPad[i] = featureSmall;
                    adjustedPad[i+featureCount] = featureLarge;
                } else {
                    adjustedPad[i] = featureLarge;
                    adjustedPad[i+featureCount] = featureSmall;
                }
            }
            return adjustedPad;
        }
        else
            throw new NotImplementedException("This padding type is not implemented yet!");
    }

    static internal TensorShape ApplyPool(this TensorShape shape, int[] pool, int[] stride, int[] pad,
        bool ceilMode = false)
    {
        if (!shape.hasNamedDimensions)
            shape = shape.AsNamed();

        Assert.IsTrue(stride.Length == pool.Length);
        unsafe
        {
            fixed (int* pPool = pool)
            {
                return ApplyPool(shape, pPool, stride, pad, ceilMode);
            }
        }
    }

    static internal unsafe TensorShape ApplyPool(this TensorShape shape, int* pool, int[] stride, int[] pad, bool ceilMode = false)
    {
        if (!shape.hasNamedDimensions)
            shape = shape.AsNamed();

        Assert.IsTrue(stride.Length > 0);

        Assert.IsTrue(stride.Length*2 == pad.Length);
        int featureCount = stride.Length;
        Assert.IsTrue(featureCount <= TensorShape.DataFeatures.Length);

        // Based on ONNX (AveragePool & MaxPool)
        //        https://github.com/onnx/onnx/blob/master/docs/Operators.md
        // Theano "Convolution arithmetic tutorial"
        //        http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html#quick-reference
        // and TensorFlow docs:
        //         https://www.tensorflow.org/api_guides/python/nn#Convolution
        //         https://www.tensorflow.org/api_guides/python/nn#Notes_on_SAME_Convolution_Padding
        //
        //   output_size = (input_size + pad_left + pad_right - kernel_size) / stride + 1
        var newShape = shape;
        for (var i = 0; i < featureCount; ++i)
        {
            // C# automatically rounds down
            // https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/operators/arithmetic-operators
            if (ceilMode)
                newShape[TensorShape.DataFeatures[i]] = (shape[TensorShape.DataFeatures[i]] + (pad[i]+pad[i+featureCount]) - pool[i] + stride[i] - 1) / stride[i] + 1;
            else
                newShape[TensorShape.DataFeatures[i]] = (shape[TensorShape.DataFeatures[i]] + (pad[i]+pad[i+featureCount]) - pool[i]) / stride[i] + 1;
        }
        return newShape;
    }

    static internal TensorShape ApplyKernel(this TensorShape shape, TensorShape kernel, int[] stride, int[] pad)
    {
        if (!shape.hasNamedDimensions)
            shape = shape.AsNamed();

        unsafe
        {
            Assert.IsTrue(stride.Length==2 || stride.Length==3);
            int* kernelDims = stackalloc int[stride.Length == 2 ? 2 : 3];
            kernelDims[0] = kernel.kernelWidth;
            kernelDims[1] = kernel.kernelHeight;
            if (stride.Length > 2)
                kernelDims[2] = kernel.kernelSpatialDepth;

            var outShape = ApplyPool(shape, kernelDims, stride, pad);
            outShape[7] = kernel.kernelCount;
            return outShape;
        }
    }

    static internal TensorShape ApplyKernelInverse(this TensorShape shape, TensorShape kernel, int[] stride, int[] pad, int[] outputAdjustment)
    {
        if (!shape.hasNamedDimensions)
            shape = shape.AsNamed();

        Assert.IsTrue(stride.Length > 0);
        Assert.IsTrue(stride.Length * 2 == pad.Length);
        Assert.IsTrue(stride.Length <= TensorShape.KernelSpatials.Length);
        Assert.IsTrue(stride.Length <= TensorShape.DataFeatures.Length);

        // Based on ONNX (ConvTranspose)
        //        https://github.com/onnx/onnx/blob/master/docs/Operators.md
        // and Theano "Convolution arithmetic tutorial"
        //        http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html#transposed-convolution-arithmetic
        //
        // Inverse of:
        //   output_size = (input_size + pad_left + pad_right - kernel_size) / stride + 1
        // Resulting in:
        //   output_size = (input_size - 1 ) * stride - (pad_left + pad_right) + kernel_size + output_adj
        //   output_adj = (input_size + (pad_left + pad_right) - kernel_size) % stride
        //
        if (outputAdjustment == null || outputAdjustment.Length == 0)
        {
            outputAdjustment = new int[stride.Length];
            for (var i = 0; i < stride.Length; ++i)
            {
                var featureAxis = TensorShape.DataFeatures[i];
                var kernelAxis = TensorShape.KernelSpatials[i];
                var padding = pad[i] + pad[stride.Length+i];
                outputAdjustment[i] = (shape[featureAxis] + padding - kernel[kernelAxis]) % stride[i];
            }
        }

        var newShape = shape;
        for (var i = 0; i < stride.Length; ++i)
        {
            var featureAxis = TensorShape.DataFeatures[i];
            var kernelAxis = TensorShape.KernelSpatials[i];
            var padding = pad[i] + pad[stride.Length+i];
            newShape[featureAxis] = (shape[featureAxis] - 1) * stride[i] - padding + kernel[kernelAxis] + outputAdjustment[i];
        }

        newShape[TensorShape.KernelOutChannel] = kernel.kernelCount;
        return newShape;
    }

    /// <summary>
    /// Wrap index (emulate Python array index behavior)
    /// </summary>
    /// <param name="i">index</param>
    /// <param name="length">array length</param>
    /// <returns>wrapped around index</returns>
    static public int WrapIndex(int i, int length)
    {
        // allow index to be equal to length
        // in order to enable iteration over [i,end) range
        if (i >= length)
            return length;

        // in C# modulo of negative is negative
        // to emulate Python array behavior, we use: https://stackoverflow.com/questions/1082917/mod-of-negative-number-is-melting-my-brain/1082938
        var v = i % length;
        return v < 0 ? (v + length): v;
    }

    static internal bool IsNDHWC(this TensorShape shape)
    {
        if (!shape.hasNamedDimensions)
            shape = shape.AsNamed();

        return shape.sequenceLength == 1 &&
               shape.numberOfDirections == 1 &&
               shape.extraDimension == 1;
    }

    static internal bool Is4D(this TensorShape shape)
    {
        if (!shape.hasNamedDimensions)
            shape = shape.AsNamed();

        return shape.sequenceLength == 1 &&
               shape.numberOfDirections == 1 &&
               shape.extraDimension == 1 &&
               shape.depth == 1;
    }

    // Works for NCHW or NHWC
    static internal int Convert4DTo8DAxis(int axis)
    {
        Assert.IsTrue(axis < 4);
        Assert.IsTrue(axis > -4);
        if (axis < 0) //backward indexing
        {
            return axis;
        }
        else if (axis == 0) //batch
            return TensorShape.DataBatch;
        else //H,W,C
            return axis + TensorShape.D;
    }

    static internal int FirstNotIdentityFeatureDimensionIndex(this TensorShape shape)
    {
        if (!shape.hasNamedDimensions)
            shape = shape.AsNamed();

        for (int dimIndex = TensorShape.DataFeature3; dimIndex < TensorShape.MaxRank; ++dimIndex)
        {
            if (shape[dimIndex] > 1)
                return dimIndex;
        }

        return TensorShape.MaxRank;
    }

    static internal bool Is8DAxisConvertibleTo4D(int axis)
    {
        Assert.IsTrue(axis > -4);
        Assert.IsTrue(axis < TensorShape.MaxRank);
        return axis < 0 || axis == TensorShape.DataBatch || axis > TensorShape.D;
    }

    /// <summary>
    /// Check if all tensors are convertible to 4D tensors
    /// </summary>
    /// <param name="tensors">tensors</param>
    /// <returns>`true` if all tensors are 4D (or less)</returns>
    static public bool AreAllTensorsConvertibleTo4D(Tensor[] tensors)
    {
        for (int i = 0; i < tensors.Length; ++i)
        {
            if (!tensors[i].shape.Is4D())
                return false;
        }

        return true;
    }

    static internal int Convert8DAxisTo4D(int axis)
    {
        Assert.IsTrue(Is8DAxisConvertibleTo4D(axis));
        if (axis < 0) //backward indexing
        {
            return axis;
        }
        else if (axis == TensorShape.DataBatch) //batch
            return 0;
        else //H,W,C
            return axis - TensorShape.D;
    }

    static internal unsafe void Get8DParametersNoAlloc(this TensorShape shape, int[] parameters, int* parameters8D, int defaultValue)
    {
        if (!shape.hasNamedDimensions)
            shape = shape.AsNamed();

        if (parameters.Length == TensorShape.MaxRank)
        {
            for (int i = 0; i < TensorShape.MaxRank; ++i)
                parameters8D[i] = parameters[i];
        }
        else
        {
            Assert.AreEqual(4, parameters.Length);
            if (!shape.Is4D()) Assert.IsTrue(false, $"4D Parameters {parameters} can't be used with a tensor of shape {shape} as it contains other dimensions, please use 8D parameters for this shape.");
            parameters8D[0] = defaultValue;
            parameters8D[1] = defaultValue;
            parameters8D[2] = parameters[0];
            parameters8D[3] = defaultValue;
            parameters8D[4] = defaultValue;
            parameters8D[5] = parameters[1];
            parameters8D[6] = parameters[2];
            parameters8D[7] = parameters[3];
        }
    }

    /// <summary>
    /// Calculate 8D permutations from 4D
    /// </summary>
    /// <param name="shape">shape</param>
    /// <param name="permutations">permutations</param>
    /// <returns>8D permutations</returns>
    static public int[] Get8DPermutationsForNHWCPermutationsAndShape(this TensorShape shape, int[] permutations)
    {
        if (!shape.hasNamedDimensions)
            shape = shape.AsNamed();

        if (permutations.Length == TensorShape.MaxRank)
            return permutations;

        Assert.AreEqual(4, permutations.Length);
        if (!shape.Is4D()) Assert.IsTrue(false, $"4D Permutation {permutations} can't be used with a tensor of shape {shape} as it contains other dimensions, please use an 8D permutation for this shape.");
        int batchOldAxis = Convert4DTo8DAxis(permutations[0]);
        int heighOldAxis = Convert4DTo8DAxis(permutations[1]);
        int widthOldIndex = Convert4DTo8DAxis(permutations[2]);
        int channeOldIndex = Convert4DTo8DAxis(permutations[3]);
        return new int[] {0, 1, batchOldAxis, 3, 4, heighOldAxis, widthOldIndex, channeOldIndex };
    }

    static internal NativeArray<int> Get8DPermutationsForNHWCPermutationsAndShape(this TensorShape shape, NativeArray<int> inPermutations)
    {
        if (!shape.hasNamedDimensions)
            shape = shape.AsNamed();

        if (inPermutations.Length == TensorShape.MaxRank)
            return inPermutations;

        Assert.AreEqual(4, inPermutations.Length);
        if (!shape.Is4D()) Assert.IsTrue(false, $"4D Permutation {inPermutations.ToString()} can't be used with a tensor of shape {shape} as it contains other dimensions, please use an 8D permutation for this shape.");
        int batchOldAxis = Convert4DTo8DAxis(inPermutations[0]);
        int heighOldAxis = Convert4DTo8DAxis(inPermutations[1]);
        int widthOldIndex = Convert4DTo8DAxis(inPermutations[2]);
        int channeOldIndex = Convert4DTo8DAxis(inPermutations[3]);

        // Valid only for single frame
        NativeArray<int> outPermutations = new NativeArray<int>(8, Allocator.Temp);
        outPermutations[0] = 0;
        outPermutations[1] = 1;
        outPermutations[2] = batchOldAxis;
        outPermutations[3] = 3;
        outPermutations[4] = 4;
        outPermutations[5] = heighOldAxis;
        outPermutations[6] = widthOldIndex;
        outPermutations[7] = channeOldIndex;

        return outPermutations;
    }

    static internal int[] Get8DPermutationsForNCHWPermutationsAndShape(this TensorShape shape, int[] permutations)
    {
        if (!shape.hasNamedDimensions)
            shape = shape.AsNamed();

        if (permutations.Length == TensorShape.MaxRank)
            return permutations;

        Assert.AreEqual(4, permutations.Length);
        if (!shape.Is4D()) Assert.IsTrue(false, $"4D Permutation {permutations} can't be used with a tensor of shape {shape} as it contains other dimensions, please use an 8D permutation for this shape.");
        int batchOldAxis = Convert4DTo8DAxis(permutations[0]);
        int channelOldIndex = Convert4DTo8DAxis(permutations[1]);
        int heightOldIndex = Convert4DTo8DAxis(permutations[2]);
        int widthOldIndex = Convert4DTo8DAxis(permutations[3]);
        return new int[] {0, 1, batchOldAxis, 3, 4, channelOldIndex, heightOldIndex, widthOldIndex };
    }

    static internal NativeArray<int> Get8DPermutationsForNCHWPermutationsAndShape(this TensorShape shape, NativeArray<int> inPermutations)
    {
        if (!shape.hasNamedDimensions)
            shape = shape.AsNamed();

        if (inPermutations.Length == TensorShape.MaxRank)
            return inPermutations;

        Assert.AreEqual(4, inPermutations.Length);
        if (!shape.Is4D()) Assert.IsTrue(false, $"4D Permutation {inPermutations.ToString()} can't be used with a tensor of shape {shape} as it contains other dimensions, please use an 8D permutation for this shape.");
        int batchOldAxis = Convert4DTo8DAxis(inPermutations[0]);
        int channelOldIndex = Convert4DTo8DAxis(inPermutations[1]);
        int heightOldIndex = Convert4DTo8DAxis(inPermutations[2]);
        int widthOldIndex = Convert4DTo8DAxis(inPermutations[3]);

        // Valid only for single frame
        NativeArray<int> outPermutations = new NativeArray<int>(8, Allocator.Temp);
        outPermutations[0] = 0;
        outPermutations[1] = 1;
        outPermutations[2] = batchOldAxis;
        outPermutations[3] = 3;
        outPermutations[4] = 4;
        outPermutations[5] = channelOldIndex;
        outPermutations[6] = heightOldIndex;
        outPermutations[7] = widthOldIndex;

        return outPermutations;
    }

    static internal unsafe TensorShape ApplyStridedSlice8DUnsafeNoAlloc(this TensorShape shape, int* starts, int* ends,
        int* stride)
    {
        if (!shape.hasNamedDimensions)
            shape = shape.AsNamed();

        TensorShape sliced = shape;

        for (int i = 0; i < shape.rank; ++i)
        {
            // NOTE: begin=0, end=0, stride=1  <=  full range from the existing axis
            //       begin=0, end=X, stride=1  <=  full range from the existing axis, if X==last element on this axis
            //       begin=0, end=0, stride=0  <=  new axis OR shrink axis to a single 1st element
            //       begin=N, end=N, stride=0  <=              shrink axis to a single Nth element

            // take + 1 is si > shape[i]
            int ei = TensorExtensions.WrapIndex(ends[i], shape[i]);
            int si = TensorExtensions.WrapIndex(starts[i], shape[i]);


            // Barracuda convetion (non ONNX), t[0:0] => t[:]
            if (si == 0 && ei == 0)
                ei = shape[i];

            if (stride[i] > 0)
                sliced[i] = (int)Math.Round((double)(Math.Min(ei, shape[i]) - Math.Min(si, shape[i] - 1)) / (double)(Mathf.Abs(stride[i])), MidpointRounding.AwayFromZero);
            else if (stride[i] < 0)
            {
                bool inclusive = ends[i] < -shape[i]; // edge case when ends is negative and bigger than nchwShape
                sliced[i] = (int)Math.Round((double)(Math.Min(si, shape[i] - 1) - Math.Min(ei, shape[i]) + (inclusive ? 1 : 0)) / (double)(Mathf.Abs(stride[i])), MidpointRounding.AwayFromZero);
            }
            else
            {
                // Assert.IsTrue(stride[i] != 0); // 0 strides not allowed
                // breaks legacy implementations
                D.LogWarning("StridedSlice with 0 strides, not supported! Slicing to 1D dimension");
                sliced[i] = 1;
            }
        }

        return sliced;
    }

    static internal TensorShape ApplyStridedSlice(this TensorShape shape, int[] starts, int[] ends, int[] stride)
    {
        if (!shape.hasNamedDimensions)
            shape = shape.AsNamed();

        unsafe
        {
            int* starts8Dbuffer = stackalloc int[TensorShape.MaxRank];
            int* ends8Dbuffer = stackalloc int[TensorShape.MaxRank];
            int* stride8Dbuffer = stackalloc int[TensorShape.MaxRank];
            Get8DParametersNoAlloc(shape, starts, starts8Dbuffer, 0);
            Get8DParametersNoAlloc(shape, ends, ends8Dbuffer, 1);
            Get8DParametersNoAlloc(shape, stride, stride8Dbuffer, 1);

            return shape.ApplyStridedSlice8DUnsafeNoAlloc(starts8Dbuffer, ends8Dbuffer, stride8Dbuffer);
        }
    }


    /// <summary>
    /// Calculate shape after applying permutations
    /// </summary>
    /// <param name="shape">shape</param>
    /// <param name="permutations">permutations</param>
    /// <returns>new shape</returns>
    static public int[] Permute(int[] shape, int[] permutations)
    {
        Assert.AreEqual(shape.Length, permutations.Length);
        var output = new int[shape.Length];
        for (var i = 0; i < permutations.Length; ++i)
            output[i] = permutations[i] >= 0 ? shape[permutations[i]] : 1;
        return output;
    }

    /// <summary>
    /// Calculate TensorShape after applying permutations
    /// </summary>
    /// <param name="shape">shape</param>
    /// <param name="permutations">permutations</param>
    /// <returns>new TensorShape</returns>
    static public TensorShape Permute(this TensorShape shape, int[] permutations)
    {
        if (!shape.hasNamedDimensions)
            shape = shape.AsNamed();

        if (permutations.Length == 4)
            permutations = Get8DPermutationsForNHWCPermutationsAndShape(shape, permutations);

        var permutedShape = new int[TensorShape.MaxRank];
        for (var i = 0; i < permutations.Length; ++i)
            permutedShape[i] = permutations[i] >= 0 ? shape[permutations[i]] : 1;

        var output = new TensorShape(permutedShape);
        return output;
    }

    static internal TensorShape Permute(this TensorShape shape, NativeArray<int> permutations)
    {
        if (!shape.hasNamedDimensions)
            shape = shape.AsNamed();

        if (permutations.Length == 4)
            permutations = Get8DPermutationsForNHWCPermutationsAndShape(shape, permutations);

        var permutedShape = new int[TensorShape.MaxRank];
        for (var i = 0; i < permutations.Length; ++i)
            permutedShape[i] = permutations[i] >= 0 ? shape[permutations[i]] : 1;

        var output = new TensorShape(permutedShape);
        return output;
    }

    /// <summary>
    /// Create ITensorData from Texture
    /// </summary>
    /// <param name="tex">Texture</param>
    /// <param name="shape">shape</param>
    /// <returns>created ITensorData</returns>
    /// <exception cref="NotImplementedException">thrown if unsupported texture type is supplied</exception>
    static public ITensorData CreateFromTexture(Texture tex, TensorShape shape)
    {
        Assert.AreEqual(tex.width, shape.width);
        Assert.AreEqual(tex.height, shape.height);
        Assert.IsTrue(shape.channels < 4);

        // @TODO: implement proper GPU storage
        var data = new ArrayTensorData(shape);
        if (tex is Texture2D)
        {
            Texture2D tex2d = tex as Texture2D;
            var pixels = tex2d.GetPixels();
            for (int i = 0; i < data.array.Length && i < pixels.Length * shape.channels; ++i)
                data.array[i] = pixels[i / shape.channels][i % shape.channels];
        }
        else
            throw new NotImplementedException();

        return data;
    }
}

} // namespace Unity.Barracuda
