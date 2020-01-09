using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq; // Enumerable.Range(), Enumerable.SequenceEqual()

using UnityEngine;
using UnityEngine.Assertions;

namespace Barracuda {


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

    static public void TestInit2(this Tensor X, int n = -1)
    {
        if (n < 0)
            n = X.length;
        n = Math.Min(n, X.length);
        for (int i = 0; i < n; ++i)
            X[i] = 0.1f;
    }

    static public void TestInitCos(this Tensor X, int n = -1)
    {
        if (n < 0)
            n = X.length;
        n = Math.Min(n, X.length);
        for (int i = 0; i < n; ++i)
            X[i] = Mathf.Cos(i);
    }

    static public void Print(this Tensor X, string msg = "")
    {
        D.Log(msg + X.name + " " + X.shape);
    }

    static public void PrintDataPart(this Tensor X, int size, string msg = "")
    {
        if (msg.Length > 0)
            msg += " ";
        for (int i = 0; i < X.length && i < size; ++i)
        {
            msg += X[i];
            msg += " ";
        }
        D.Log(msg);
    }

    static public bool Equals(this Tensor X, Tensor Y)
    {
        if (X.batch != Y.batch || X.height != Y.height || X.width != Y.width || X.channels != Y.channels)
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
        if (X.batch != Y.batch || X.height != Y.height || X.width != Y.width || X.channels != Y.channels)
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

    static public int[] ArgMax(this Tensor X)
    {
        int[] result = new int[X.batch];
        for (int b = 0; b < X.batch; ++b)
        {
            float maxV = Mathf.NegativeInfinity;
            var i = 0;
            for (int y = 0; y < X.height; ++y)
                for (int x = 0; x < X.width; ++x)
                    for (int c = 0; c < X.channels; ++c, ++i)
                    {
                        var v = X[b, y, x, c];
                        if (maxV >= v)
                            continue;
                        maxV = v;
                        result[b] = i;
                    }
        }
        return result;
    }

    static public Tensor Reshape(this Tensor X, int[] size)
    {
        var newShape = X.shape.Reshape(size);
        return X.Reshape(newShape);
    }

    static public int[][] ArgSort(this Tensor X)
    {
        var count = X.height * X.width * X.channels;
        var result = new List<int[]>();

        for (int n = 0; n < X.batch; ++n)
        {
            int[] indices = Enumerable.Range(0, count).ToArray<int>();

            var sliceOffset = n * count;
            Array.Sort<int>(indices, (a, b) => X[sliceOffset + a].CompareTo(X[sliceOffset + b]));
            result.Add(indices);
        }
        return result.ToArray();
    }

    static public TensorShape Concat(TensorShape[] shapes, int axis)
    {
        if (shapes.Length == 0)
            return new TensorShape();

        // validate that off axis dimensions are equal
        for (var i = 1; i < shapes.Length; ++i)
        {
            var a = shapes[0].ToArray();
            var b = shapes[i].ToArray();
            var aAxis = shapes[0].Axis(axis);
            var bAxis = shapes[i].Axis(axis);
            a[aAxis] = 0; b[bAxis] = 0;
            if (!Enumerable.SequenceEqual(a, b))
            {
                foreach (var s in shapes)
                    D.Log(s);
                throw new ArgumentException("Off-axis dimensions must match");
            }
        }

        var shape = shapes[0].ToArray();
        var dstAxis = shapes[0].Axis(axis);
        for (var i = 1; i < shapes.Length; ++i)
            shape[dstAxis] += shapes[i][axis];
        return new TensorShape(shape);
    }

    static public TensorShape ConcatShapes(Tensor[] tensors, int axis)
    {
        return Concat(tensors.Select(t => t.shape).ToArray(), axis);
    }

    static public TensorShape Max(TensorShape[] shapes)
    {
        Assert.IsTrue(shapes.Length > 0);
        int batch = 0, height = 0, width = 0, channels = 0;
        foreach (var s in shapes)
        {
            batch =    Math.Max(s.batch, batch);
            height =   Math.Max(s.height, height);
            width =    Math.Max(s.width, width);
            channels = Math.Max(s.channels, channels);
        }
        return new TensorShape(batch, height, width, channels);
    }

    static public TensorShape MaxShape(Tensor[] tensors)
    {
        Assert.IsTrue(tensors.Length > 0);
        int batch = 0, height = 0, width = 0, channels = 0;
        foreach (var t in tensors)
        {
            batch =    Math.Max(t.batch, batch);
            height =   Math.Max(t.height, height);
            width =    Math.Max(t.width, width);
            channels = Math.Max(t.channels, channels);
        }
        return new TensorShape(batch, height, width, channels);
    }

    static public TensorShape Scale(this TensorShape shape, TensorShape scale)
    {
        return new TensorShape(
            shape.batch * scale.batch,
            shape.height * scale.height,
            shape.width * scale.width,
            shape.channels * scale.channels);
    }

    static public TensorShape Scale(this TensorShape shape, int[] scale)
    {
        Assert.AreEqual(scale.Length, 4);
        return Scale(shape, new TensorShape(scale));
    }

    static public TensorShape Reduce(this TensorShape shape, int axis)
    {
        axis = shape.Axis(axis);
        var newShapeArray = shape.ToArray();
        newShapeArray[axis] = 1;
        return new TensorShape(newShapeArray);
    }

    static public TensorShape Reshape(this TensorShape shape, int[] size)
    {
        Assert.AreEqual(size.Length, 4);
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
        return new TensorShape(
            shape.batch,
            (shape.height + (border[1]+border[3])),
            (shape.width  + (border[0]+border[2])),
            shape.channels);
    }

    static public int[] AdjustPadToKernel(this Tensor tensor, Tensor kernel, int[] stride, int[] pad)
    {
        return AdjustPadToKernel(tensor.shape, kernel.shape, stride, pad);
    }

    static public int[] AdjustPadToKernel(this TensorShape shape, TensorShape kernel, int[] stride, int[] pad)
    {
        return AdjustPadToPool(shape, new int[] { kernel.kernelWidth, kernel.kernelHeight }, stride, pad);
    }

    static public int[] AdjustPadToPool(this Tensor tensor, int[] pool, int[] stride, int[] pad)
    {
        return AdjustPadToPool(tensor.shape, pool, stride, pad);
    }

    static public int[] AdjustPadToPool(this TensorShape shape, int[] pool, int[] stride, int[] pad)
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

            var padAlongWidth = Math.Max(pool[0] - widthModStride, 0);
            var padAlongHeight = Math.Max(pool[1] - heightModStride, 0);
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

    static public TensorShape ApplyPool(this TensorShape shape, int[] pool, int[] stride, int[] pad)
    {
        Assert.AreEqual(pool.Length, 2);
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

        return new TensorShape(
            shape.batch,
            (shape.height + (pad[1]+pad[3]) - pool[1]) / stride[1] + 1,
            (shape.width  + (pad[0]+pad[2]) - pool[0]) / stride[0] + 1,
            shape.channels);
    }

    static public TensorShape ApplyKernel(this TensorShape shape, TensorShape kernel, int[] stride, int[] pad)
    {
        shape = ApplyPool(shape, new int[] { kernel.kernelWidth, kernel.kernelHeight }, stride, pad);
        return new TensorShape(shape.batch, shape.height, shape.width, kernel.kernelCount);
    }

    static public TensorShape ApplyKernelInverse(this TensorShape shape, TensorShape kernel, int[] stride, int[] pad, int[] outputAdjustment)
    {
        Assert.AreEqual(stride.Length, 2);
        Assert.AreEqual(pad.Length, 4);

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
            outputAdjustment = new int[] {
                (shape.width + (pad[0]+pad[2]) - kernel.kernelWidth) % stride[0],
                (shape.height + (pad[1]+pad[3]) - kernel.kernelHeight) % stride[1]
            };
        }
        return new TensorShape(
            shape.batch,
            (shape.height - 1) * stride[1] - (pad[1]+pad[3]) + kernel.kernelHeight + outputAdjustment[1],
            (shape.width  - 1) * stride[0] - (pad[0]+pad[2]) + kernel.kernelWidth + outputAdjustment[0],
            kernel.kernelCount);
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

    // TODO: implement negative strides
    static public TensorShape ApplyStridedSlice(this TensorShape shape, int[] starts, int[] ends, int[] stride)
    {
        Assert.AreEqual(starts.Length, shape.rank);
        Assert.AreEqual(ends.Length, shape.rank);
        Assert.AreEqual(stride.Length, shape.rank);

        int[] counts = shape.ToArray();
        int[] sliced = shape.ToArray();
        Assert.AreEqual(counts.Length, shape.rank);
        for (int i = 0; i < counts.Length; ++i)
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

        return new TensorShape(sliced);
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

} // namespace Barracuda
