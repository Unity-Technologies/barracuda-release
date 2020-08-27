using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq; // Enumerable.Range(), Enumerable.SequenceEqual()

using UnityEngine;
using UnityEngine.Assertions;

namespace Unity.Barracuda {


public static class TensorExtensions
{
    static public void TestInit(this Tensor X, int n = -1)
    {
        if (n < 0)
            n = X.length;
        n = Math.Min(n, X.length);
        for (int i = 0; i < n; ++i)
            X[i] = i;
    }

    static public void TestInitCos(this Tensor X, int n = -1, float offset = 0.0f)
    {
        if (n < 0)
            n = X.length;
        n = Math.Min(n, X.length);
        for (int i = 0; i < n; ++i)
            X[i] = Mathf.Cos(i + offset);
    }

    static public void TestInitValue(this Tensor X, float value=0.1f, int n = -1)
    {
        if (n < 0)
            n = X.length;
        n = Math.Min(n, X.length);
        for (int i = 0; i < n; ++i)
            X[i] = value;
    }

    static public float[] AsFloats(this Tensor x)
    {
        return x.ToReadOnlyArray();
    }

    static public int[] AsInts(this Tensor x)
    {
        return Array.ConvertAll(x.ToReadOnlyArray(), (v => (int)v));
    }

    static public long[] AsLongs(this Tensor x)
    {
        return Array.ConvertAll(x.ToReadOnlyArray(), (v => (long)v));
    }

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

    static public void Print(this Tensor X, string msg = "")
    {
        if (msg.Length > 0)
            msg += " ";
        D.Log($"{msg}{X.name} {X.shape}");
    }

    static public void PrintDataPart(this Tensor X, int size, string msg = "")
    {
        if (msg.Length > 0)
            msg += " ";
        D.Log($"{msg}{X.DataToString(size)}");
    }

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
            if (Mathf.Abs(X[i] - Y[i]) > epsilon)
            {
                // @TODO: move logging into dedicated function
                D.Log("First mismatch @ [" + i + "]: " + X[i] + " != " + Y[i]);
                return false;
            }
        }

        return true;
    }

    static public float MaxDifference(this Tensor X, Tensor Y)
    {
        float maxD = 0f;
        for (int i = 0; i < X.length; ++i)
            maxD = Mathf.Max(Mathf.Abs(X[i] - Y[i]), maxD);
        return maxD;
    }

    static public Tensor Reshape(this Tensor X, int[] size)
    {
        var newShape = X.shape.Reshape(size);
        return X.Reshape(newShape);
    }

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

    static public TensorShape Gather(TensorShape[] shapes, int axis)
    {
        TensorShape shape = shapes[0];
        TensorShape indices = shapes[1];
        shape[axis] = indices.length;

        return shape;
    }

    static public TensorShape Concat(Tensor[] tensors, int axis)
    {
        if (tensors.Length == 0)
            return new TensorShape();

        // validate that off axis dimensions are equal
        for (var i = 1; i < tensors.Length; ++i)
        {
            var a = tensors[0].shape;
            var b = tensors[i].shape;
            var aAxis = tensors[0].shape.Axis(axis);
            var bAxis = tensors[i].shape.Axis(axis);
            a[aAxis] = 0; b[bAxis] = 0;
            if (a != b)
            {
                foreach (var s in tensors)
                    D.Log(s.shape);
                throw new ArgumentException("Off-axis dimensions must match");
            }
        }

        var shape = tensors[0].shape;
        var dstAxis = tensors[0].shape.Axis(axis);
        for (var i = 1; i < tensors.Length; ++i)
            shape[dstAxis] += tensors[i].shape[axis];
        return shape;
    }

    static public TensorShape Concat(TensorShape[] shapes, int axis)
    {
        if (shapes.Length == 0)
            return new TensorShape();

        // validate that off axis dimensions are equal
        for (var i = 1; i < shapes.Length; ++i)
        {
            var a = shapes[0];
            var b = shapes[i];
            var aAxis = shapes[0].Axis(axis);
            var bAxis = shapes[i].Axis(axis);
            a[aAxis] = 0; b[bAxis] = 0;
            if (a != b)
            {
                foreach (var s in shapes)
                    D.Log(s);
                throw new ArgumentException("Off-axis dimensions must match");
            }
        }

        var shape = shapes[0];
        var dstAxis = shapes[0].Axis(axis);
        for (var i = 1; i < shapes.Length; ++i)
            shape[dstAxis] += shapes[i][axis];
        return shape;
    }

    static public TensorShape Max(TensorShape[] shapes)
    {
        Assert.IsTrue(shapes.Length > 0);

        var shape = shapes[0];
        for (var i = 1; i < shapes.Length; ++i)
            for (var axis = 0; axis < TensorShape.MaxRank; axis++)
                shape[axis] = Math.Max(shape[axis], shapes[i][axis]);

        return shape;
    }

    static public TensorShape MaxShape(Tensor[] tensors)
    {
        Assert.IsTrue(tensors.Length > 0);
        var shape = tensors[0].shape;
        for (var i = 1; i < tensors.Length; ++i)
            for (var axis = 0; axis < TensorShape.MaxRank; axis++)
                shape[axis] = Math.Max(shape[axis], tensors[i].shape[axis]);
        return shape;
    }

    static public TensorShape Scale(this TensorShape shape, TensorShape scale)
    {
        var newShape = shape;
        for (var axis = 0; axis < TensorShape.MaxRank; axis++)
            newShape[axis] *= scale[axis];
        return newShape;
    }

    static public TensorShape Scale(this TensorShape shape, int[] scale)
    {
        scale = Get8DParametersFromNHWCParametersAndShape(shape, scale, 1);

        for (var axis = 0; axis < TensorShape.MaxRank; axis++)
            shape[axis] *= scale[axis];
        return shape;
    }

    static public TensorShape Reduce(this TensorShape shape, int axis)
    {
        axis = shape.Axis(axis);
        var newShapeArray = shape;
        newShapeArray[axis] = 1;
        return newShapeArray;
    }

    static public TensorShape Reshape(this TensorShape shape, int[] size)
    {
        size = Get8DParametersFromNHWCParametersAndShape(shape, size, 1);
        var newShapeArray = shape.ToArray();

        // From: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reshape
        //
        // At most one dimension of the new shape can be -1.
        // In this case, the value is inferred from the size of the tensor and the remaining dimensions.
        //
        // A dimension could also be 0,
        // in which case the actual dimension value is unchanged (i.e. taken from the input tensor).

        var multipleOf = 1;
        var unknownIndex = -1;
        for (int q = 0; q < size.Length; ++q)
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
            var newShape = new TensorShape(newShapeArray);
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
        return new TensorShape(newShapeArray);
    }

    static public TensorShape ApplyBorder(this TensorShape shape, int[] border)
    {
        Assert.IsTrue(border.Length > 0);
        Assert.IsTrue(border.Length % 2 == 0);
        int featureCount = border.Length / 2;
        Assert.IsTrue(featureCount <= TensorShape.DataFeatures.Length);
        for (var i = 0; i < featureCount; ++i)
        {
            shape[TensorShape.DataFeatures[i]] += border[i               ];
            shape[TensorShape.DataFeatures[i]] += border[i + featureCount];
        }
        return shape;
    }

    static public int[] AdjustPadToKernel(this Tensor tensor, Tensor kernel, int[] stride, int[] pad)
    {
        return AdjustPadToKernel(tensor.shape, kernel.shape, stride, pad);
    }

    static public int[] AdjustPadToKernel(this TensorShape shape, TensorShape kernel, int[] stride, int[] pad)
    {
        return AdjustPadToPool(shape, (kernel.kernelWidth, kernel.kernelHeight), stride, pad);
    }

    static public int[] AdjustPadToPool(this Tensor tensor, int[] pool, int[] stride, int[] pad)
    {
        return AdjustPadToPool(tensor.shape, pool, stride, pad);
    }

    static public int[] AdjustPadToPool(this TensorShape shape, int[] pool, int[] stride, int[] pad)
    {
        return AdjustPadToPool(shape, (pool[0], pool[1]), stride, pad);
    }

    static public int[] AdjustPadToPool(this Tensor tensor, ValueTuple<int,int> pool, int[] stride, int[] pad)
    {
        return AdjustPadToPool(tensor.shape, pool, stride, pad);
    }

    // @TODO: implement 3D, ND pool suppport
    static public int[] AdjustPadToPool(this TensorShape shape, ValueTuple<int,int> pool, int[] stride, int[] pad)
    {
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

            var widthModStride = shape.width % stride[0];
            var heightModStride = shape.height % stride[1];

            if (widthModStride == 0)
                widthModStride = stride[0];
            if (heightModStride == 0)
                heightModStride = stride[1];

            var padAlongWidth = Math.Max(pool.Item1 - widthModStride, 0);
            var padAlongHeight = Math.Max(pool.Item2 - heightModStride, 0);
            // Code above (based on TensorFlow docs) is equivalent to (based on ONNX docs):
            // padAlongWidth = (Mathf.Ceil(shape.width/stride[0]) - 1) * stride[0] + pool[0] - shape.width;
            // padAlongHeight = (Mathf.Ceil(shape.height/stride[1]) - 1) * stride[1] + pool[1] - shape.height;

            var widthSmall = padAlongWidth / 2;
            var widthLarge = padAlongWidth - widthSmall;
            var heightSmall = padAlongHeight / 2;
            var heightLarge = padAlongHeight - heightSmall;

            // In case of odd number add the extra padding
            // at the end for SAME_UPPER and at the beginning for SAME_LOWER
            if (type == Layer.AutoPad.SameUpper)
                return new [] { widthSmall, heightSmall, widthLarge, heightLarge };
            else
                return new [] { widthLarge, heightLarge, widthSmall, heightSmall };
        }
        else
            throw new NotImplementedException("This padding type is not implemented yet!");
    }

    static public TensorShape ApplyPool(this TensorShape shape, int[] pool, int[] stride, int[] pad, bool ceilMode = false)
    {
         return ApplyPool(shape, (pool[0], pool[1]), stride, pad, ceilMode);
    }

    // @TODO: implement 3D, ND pool suppport
    // @SEE: ApplyBorder() for generic impl
    static public TensorShape ApplyPool(this TensorShape shape, ValueTuple<int,int> pool, int[] stride, int[] pad, bool ceilMode = false)
    {
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);

        // Based on ONNX (AveragePool & MaxPool)
        //        https://github.com/onnx/onnx/blob/master/docs/Operators.md
        // Theano "Convolution arithmetic tutorial"
        //        http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html#quick-reference
        // and TensorFlow docs:
        //         https://www.tensorflow.org/api_guides/python/nn#Convolution
        //         https://www.tensorflow.org/api_guides/python/nn#Notes_on_SAME_Convolution_Padding
        //
        //   output_size = (input_size + pad_left + pad_right - kernel_size) / stride + 1
        //
        var newShape = shape;
        if (ceilMode)
        {
            newShape[TensorShape.H] = (shape.height + (pad[1]+pad[3]) - pool.Item2 + stride[1] - 1) / stride[1] + 1;
            newShape[TensorShape.W] = (shape.width  + (pad[0]+pad[2]) - pool.Item1 + stride[1] - 1) / stride[0] + 1;
            return newShape;
        }
        // C# automatically rounds down
        // https://docs.microsoft.com/en-us/dotnet/csharp/language-reference/operators/arithmetic-operators
        newShape[TensorShape.H] = (shape.height + (pad[1]+pad[3]) - pool.Item2) / stride[1] + 1;
        newShape[TensorShape.W] = (shape.width  + (pad[0]+pad[2]) - pool.Item1) / stride[0] + 1;
        return newShape;
    }

    static public TensorShape ApplyKernel(this TensorShape shape, TensorShape kernel, int[] stride, int[] pad)
    {
        int[] shapeArray = ApplyPool(shape, (kernel.kernelWidth, kernel.kernelHeight), stride, pad).ToArray();
        shapeArray[7] = kernel.kernelCount;
        return new TensorShape(shapeArray);
    }

    static public TensorShape ApplyKernelInverse(this TensorShape shape, TensorShape kernel, int[] stride, int[] pad, int[] outputAdjustment)
    {
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

    static internal bool IsNHWC(this TensorShape shape)
    {
        return shape.sequenceLength == 1 &&
               shape.numberOfDirections == 1 &&
               shape.extraDimension == 1 &&
               shape.depth == 1;
    }

    static internal int NHWCTo8DAxis(int nhwcAxis)
    {
        Assert.IsTrue(nhwcAxis < 4);
        Assert.IsTrue(nhwcAxis > -4);
        if (nhwcAxis < 0) //backward indexing
        {
            return nhwcAxis;
        }
        else if (nhwcAxis == 0) //batch
            return TensorShape.DataBatch;
        else //H,W,C
            return nhwcAxis + TensorShape.D;
    }

    static internal bool Is8DAxisConvertibleToNHWC(int axis)
    {
        Assert.IsTrue(axis > -4);
        Assert.IsTrue(axis < TensorShape.MaxRank);
        return axis < 0 || axis == TensorShape.DataBatch || axis > TensorShape.D;
    }

    static public bool AreAllTensorsConvertibleToNCHW(Tensor[] tensors)
    {
        for (int i = 0; i < tensors.Length; ++i)
        {
            if (!tensors[i].shape.IsNHWC())
                return false;
        }

        return true;
    }

    static internal int Convert8DAxisToNHWC(int axis)
    {
        Assert.IsTrue(Is8DAxisConvertibleToNHWC(axis));
        if (axis < 0) //backward indexing
        {
            return axis;
        }
        else if (axis == TensorShape.DataBatch) //batch
            return 0;
        else //H,W,C
            return axis - TensorShape.D;
    }

    static public int[] GetNHWCParametersFrom8DParameterAndShape(TensorShape shape, int[] parameters)
    {
        if (parameters.Length == 4)
            return parameters;

        Assert.IsTrue(shape.IsNHWC(), $"Parameters {parameters} can't be converted to NCHW with a tensor of shape {shape} as it contains other dimensions.");
        Assert.AreEqual(parameters.Length, TensorShape.MaxRank);
        return new int[] {parameters[TensorShape.DataBatch], parameters[TensorShape.H], parameters[TensorShape.W], parameters[TensorShape.C] };
    }

    static public int[] Get8DParametersFromNHWCParametersAndShape(TensorShape shape, int[] parameters, int defaultValue)
    {
        if (parameters.Length == TensorShape.MaxRank)
            return parameters;

        Assert.AreEqual(4, parameters.Length);
        Assert.IsTrue(shape.IsNHWC(), $"4D NCHW Parameters {parameters} can't be used with a tensor of shape {shape} as it contains other dimensions, please use 8D parameters for this shape.");
        return new int[] {defaultValue, defaultValue, parameters[0], defaultValue, defaultValue, parameters[1], parameters[2], parameters[3] };
    }

    static public int[] Get8DPermutationsForNHWCPermutationsAndShape(TensorShape shape, int[] permutations)
    {
        if (permutations.Length == TensorShape.MaxRank)
            return permutations;

        Assert.AreEqual(4, permutations.Length);
        Assert.IsTrue( shape.IsNHWC(), $"4D NCHW Permutation {permutations} can't be used with a tensor of shape {shape} as it contains other dimensions, please use an 8D permutation for this shape.");
        int batchOldAxis = NHWCTo8DAxis(permutations[0]);
        int heighOldAxis = NHWCTo8DAxis(permutations[1]);
        int widthOldIndex = NHWCTo8DAxis(permutations[2]);
        int channeOldIndex = NHWCTo8DAxis(permutations[3]);
        return new int[] {0, 1, batchOldAxis, 3, 4, heighOldAxis, widthOldIndex, channeOldIndex };
    }

    // TODO: implement negative strides
    static public TensorShape ApplyStridedSlice(this TensorShape shape, int[] starts, int[] ends, int[] stride)
    {
        starts = Get8DParametersFromNHWCParametersAndShape(shape, starts, 0);
        ends = Get8DParametersFromNHWCParametersAndShape(shape, ends, 1);
        stride = Get8DParametersFromNHWCParametersAndShape(shape, stride, 1);

        TensorShape counts = shape;
        TensorShape sliced = shape;

        for (int i = 0; i < shape.rank; ++i)
        {
            // NOTE: begin=0, end=0, stride=1  <=  full range from the existing axis
            //       begin=0, end=X, stride=1  <=  full range from the existing axis, if X==last element on this axis
            //       begin=0, end=0, stride=0  <=  new axis OR shrink axis to a single 1st element
            //       begin=N, end=N, stride=0  <=              shrink axis to a single Nth element

            Assert.IsTrue(starts[i] < counts[i]);
            if (starts[i] != ends[i])
                sliced[i] = WrapIndex(ends[i], counts[i]) - WrapIndex(starts[i], counts[i]);
            else
                sliced[i] = counts[i];
            if (stride[i] != 0 && stride[i] < counts[i])
                sliced[i] /= stride[i];
            else
                sliced[i] = 1;

            if (sliced[i] < 0)
                sliced[i] = counts[i] + sliced[i];

            if (sliced[i] < 0)
                sliced[i] = 0;
        }

        return sliced;
    }

    static public TensorShape Permute(this TensorShape shape, int[] permutations)
    {
        permutations = Get8DPermutationsForNHWCPermutationsAndShape(shape, permutations);

        var output = new TensorShape();
        for (var i = 0; i < permutations.Length; ++i)
            output[i] = permutations[i] >= 0 ? shape[permutations[i]] : 1;
        return output;
    }

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
