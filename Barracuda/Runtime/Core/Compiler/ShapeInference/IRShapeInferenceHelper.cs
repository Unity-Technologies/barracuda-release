using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;

using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Profiling;

namespace Unity.Barracuda.Compiler.IRShapeInferenceHelper
{
    internal class ShapeInference
    {
        static public int[] BarracudaLayoutToTensorShapeLayout(int[] size)
        {
            const int _ = 1;
            if (size.Length == 0)
                return new[] { _, _, 1, _, _, 1, 1, 1 };
            else if (size.Length == 1)
                return new[] { _, _, size[0], _, _, 1, 1, 1 };
            else if (size.Length == 2)
                return new[] { _, _, size[0], _, _, 1, 1, size[1] };
            else if (size.Length == 3)
                return new[] { _, _, size[0], _, _, 1, size[1], size[2] };
            else if (size.Length == 4)
                return new[] { _, _, size[0], _, _, size[1], size[2], size[3] };
            else if (size.Length == 5)
                return new[] { _, _, size[0], _, size[1], size[2], size[3], size[4] };
            else if (size.Length == 6)
                return new[] { _, _, size[0], size[1], size[2], size[3], size[4], size[5] };
            else
                return new[] { size[0], size[1], size[2], size[3], size[4], size[5], size[6], size[7] };
        }
        static public List<int> BarracudaShapeToOnnxLayout(TensorShape X, int rank)
        {
            if (rank == 0)
                return new List<int> { 1 };
            else if (rank == 1)
                return new List<int> { X.batch };
            else if (rank == 2)
                return new List<int> { X.batch, X.channels };
            else if (rank == 3)
                return new List<int> { X.batch, X.channels, X.width };
            else if (rank == 4)
                return new List<int> { X.batch, X.channels, X.height, X.width };
            else if (rank == 5)
                return new List<int> { X.batch, X.channels, X.depth, X.height, X.width };
            else if (rank == 6)
                return new List<int> { X.batch, X.channels, X.depth, X.extraDimension, X.height, X.width };
            else
                return new List<int> { X.sequenceLength, X.numberOfDirections, X.batch, X.extraDimension, X.channels, X.depth, X.height, X.width }; // TODO not sure
        }
        static public List<int> ShapeToOnnxLayout(TensorShape X, int rank)
        {
            if (rank == 0)
                return new List<int> { 1 };
            else if (rank == 1)
                return new List<int> { X.batch };
            else if (rank == 2)
                return new List<int> { X.batch, X.height };
            else if (rank == 3)
                return new List<int> { X.batch, X.height, X.width };
            else if (rank == 4)
                return new List<int> { X.batch, X.height, X.width, X.channels };
            else if (rank == 5)
                return new List<int> { X.batch, X.depth, X.height, X.width, X.channels };
            else if (rank == 6)
                return new List<int> { X.batch, X.depth, X.extraDimension, X.height, X.width, X.channels };
            else
                return new List<int> { X.sequenceLength, X.numberOfDirections, X.batch, X.extraDimension, X.depth, X.height, X.width, X.channels };
        }

        static public int[] OnnxLayoutToTensorShapeLayout(int[] size) // needed to keep -1 and 0 in shape
        {
            const int _ = 1;
            if (size.Length == 0)
                return new[] { _, _, 1, _, _, 1, 1, 1 };
            else if (size.Length == 1)
                return new[] { _, _, size[0], _, _, 1, 1, 1 };
            else if (size.Length == 2)
                return new[] { _, _, size[0], _, _, size[1], 1, 1 };
            else if (size.Length == 3)
                return new[] { _, _, size[0], _, _, size[1], size[2], 1 };
            else if (size.Length == 4)
                return new[] { _, _, size[0], _, _, size[1], size[2], size[3] };
            else if (size.Length == 5)
                return new[] { _, _, size[0], _, size[1], size[2], size[3], size[4] };
            else if (size.Length == 6)
                return new[] { _, _, size[0], size[1], size[2], size[3], size[4], size[5] };
            else
                return new[] { size[0], size[1], size[2], size[3], size[4], size[5], size[6], size[7] };
        }

        static public TensorShape OnnxLayoutToTensorShape(int[] size)
        {
            if (size.Length == 0)
                return new TensorShape(1, 1, 1, 1);
            else if (size.Length == 1)
                return new TensorShape(size[0], 1, 1, 1);
            else if (size.Length == 2)
                return new TensorShape(size[0], size[1], 1, 1);
            else if (size.Length == 3)
                return new TensorShape(size[0], size[1], size[2], 1);
            else if (size.Length == 4)
                return new TensorShape(size[0], size[1], size[2], size[3]);
            else if (size.Length == 5)
                return new TensorShape(size[0], size[1], size[2], size[3], size[4]);
            else if (size.Length == 6)
                return new TensorShape(1, 1, size[0], size[1], size[2], size[3], size[4], size[5]);
            else
                return new TensorShape(size[0], size[1], size[2], size[3], size[4], size[5], size[6], size[7]);
        }

        static public TensorShape? InferOutputShapeNCHW(Layer layer, int[] inputRanks, TensorShape[] inputShapes)
        {
            switch (layer.type)
            {
                case Layer.Type.Conv3D:
                {
                    TensorShape X = inputShapes[0];
                    // N C D H W, constructor is N D H W C
                    // => N = N C = D, D = H, H = W, W = C
                    // TODO helper function for that
                    X = new TensorShape(X.batch, X.height, X.width, X.channels, X.depth);
                    var K = layer.datasets[0].shape;

                    Assert.IsNotNull(layer.stride);
                    Assert.IsNotNull(layer.pad);
                    var pad = X.AdjustPadToKernel(K, layer.stride, layer.pad);

                    var O = X.ApplyKernel(K, layer.stride, pad);
                    return new TensorShape(O.batch, O.channels, O.depth, O.height, O.width);
                }
                case Layer.Type.Conv2D:
                case Layer.Type.DepthwiseConv2D:
                {
                    TensorShape X = inputShapes[0];
                    // N C H W, constructor is N H W C
                    // => N = N C = H, H = W, H = C
                    // TODO helper function for that
                    X = new TensorShape(X.batch, X.width, X.channels, X.height);
                    var K = layer.datasets[0].shape;

                    Assert.IsNotNull(layer.stride);
                    Assert.IsNotNull(layer.pad);
                    var pad = X.AdjustPadToKernel(K, layer.stride, layer.pad);

                    var O = X.ApplyKernel(K, layer.stride, pad);
                    return new TensorShape(O.batch, O.channels, O.height, O.width);
                }
                case Layer.Type.Conv2DTrans:
                {
                    TensorShape X = inputShapes[0];
                    // N C H W, constructor is N H W C
                    // => N = N C = H, H = W, H = C
                    // TODO helper function for that
                    X = new TensorShape(X.batch, X.width, X.channels, X.height);
                    var K = layer.datasets[0].shape;

                    Assert.IsNotNull(layer.stride);
                    Assert.IsNotNull(layer.pad);
                    // pool size is treated as output_adjustment aka output_padding here
                    var outputAdjustment = layer.pool;
                    var pad = X.AdjustPadToKernel(K, layer.stride, layer.pad);
                    var O = X.ApplyKernelInverse(K, layer.stride, pad, outputAdjustment);
                    return new TensorShape(O.batch, O.channels, O.height, O.width);
                }
                case Layer.Type.GlobalMaxPool2D:
                case Layer.Type.GlobalAvgPool2D:
                {
                    TensorShape X = inputShapes[0];
                    int rankX = inputRanks[0];
                    List<int> xShape = ShapeToOnnxLayout(X, rankX);

                    for (int i = 2; i < xShape.Count; i++)
                        xShape[i] = 1;
                    return OnnxLayoutToTensorShape(xShape.ToArray());
                }
                case Layer.Type.Dense:
                {
                    TensorShape X = inputShapes[0];
                    X = new TensorShape(X.batch, X.width, X.channels, X.height);
                    Assert.IsNotNull(layer.datasets);
                    var W = layer.datasets[0].shape;
                    var O = new TensorShape(X.flatHeight, W.flatWidth);
                    return new TensorShape(O.batch, O.channels, O.height, O.width);
                }
                case Layer.Type.MatMul:
                {
                    TensorShape X = inputShapes[0];
                    int rankX = inputRanks[0];
                    List<int> xShape = ShapeToOnnxLayout(X, rankX);

                    TensorShape Y = inputShapes[1];
                    int rankY = inputRanks[1];
                    List<int> yShape = ShapeToOnnxLayout(Y, rankY);

                    int rankO = Mathf.Max(rankX, rankY);
                    for (int i = 0; i < rankO - rankX; i++)
                        xShape.Insert(0, 1);
                    for (int i = 0; i < rankO - rankY; i++)
                        yShape.Insert(0, 1);

                    List<int> oShape = new List<int>();

                    for (int i = 0; i < rankO - 2; i++)
                        oShape.Add(Mathf.Max(xShape[i], yShape[i]));

                    oShape.Add(xShape[rankO - 2]);
                    oShape.Add(yShape[rankO - 1]);

                    return OnnxLayoutToTensorShape(oShape.ToArray());
                }
                case Layer.Type.Border3D:
                {
                    TensorShape X = inputShapes[0];
                    X = new TensorShape(X.batch, X.height, X.width, X.channels, X.depth);
                    Assert.IsNotNull(layer.pad);
                    var O = X.ApplyBorder(layer.pad);
                    return new TensorShape(O.batch, O.channels, O.depth, O.height, O.width);
                }
                case Layer.Type.Border2D:
                case Layer.Type.Pad2DReflect:
                case Layer.Type.Pad2DSymmetric:
                case Layer.Type.Pad2DEdge:
                {
                    TensorShape X = inputShapes[0];
                    X = new TensorShape(X.batch, X.width, X.channels, X.height);
                    Assert.IsNotNull(layer.pad);
                    var O = X.ApplyBorder(layer.pad);
                    return new TensorShape(O.batch, O.channels, O.height, O.width);
                }
                case Layer.Type.Upsample2D:
                {
                    TensorShape X = inputShapes[0];
                    if (inputShapes.Length > 1)
                    {
                        return null;
                    }
                    else
                    {
                        // pool size is treated as upsample coefficient here
                        Assert.IsNotNull(layer.pool);
                        Assert.AreEqual(layer.pool.Length, 4);
                        return new TensorShape(X.batch * layer.pool[0], X.height * layer.pool[1], X.width * layer.pool[2], X.channels * layer.pool[3]);
                    }
                }
                case Layer.Type.Upsample3D:
                {
                    TensorShape X = inputShapes[0];
                    if (inputShapes.Length > 1)
                    {
                        return null;
                    }
                    else
                    {
                        // pool size is treated as upsample coefficient here
                        Assert.IsNotNull(layer.pool);
                        Assert.AreEqual(layer.pool.Length, 5);
                        return new TensorShape(X.batch * layer.pool[0], X.depth * layer.pool[1], X.height * layer.pool[2], X.width * layer.pool[3], X.channels * layer.pool[4]);
                    }
                }
                case Layer.Type.Resample2D:
                {
                    TensorShape X = inputShapes[0];
                    if (inputShapes.Length > 1)
                    {
                        return null;
                    }
                    else
                    {
                        // pool is treated as resample size here
                        var size = layer.pool;
                        Assert.IsNotNull(size);
                        Assert.AreEqual(size.Length, 4);
                        return new TensorShape(size[0], size[1], size[2], size[3]);
                    }
                }
                case Layer.Type.TopKIndices:
                case Layer.Type.TopKValues:
                {
                    // Calculated at runtime: same shape as input 0 with k elements in the dimension specified by axis
                    return null;
                }
                case Layer.Type.NonMaxSuppression:
                {
                    int maxOutputBoxesPerClass = 0;

                    if (layer.pool.Length > 0)
                        maxOutputBoxesPerClass = layer.pool[0];

                    if (maxOutputBoxesPerClass <= 0)
                        return null;

                    return new TensorShape(maxOutputBoxesPerClass, 3);
                }
                case Layer.Type.NonZero:
                {
                    // Calculated at runtime
                    return null;
                }
                case Layer.Type.Add:
                case Layer.Type.Sub:
                case Layer.Type.Mul:
                case Layer.Type.Div:
                case Layer.Type.Pow:
                case Layer.Type.Min:
                case Layer.Type.Max:
                case Layer.Type.Mean:
                case Layer.Type.Greater:
                case Layer.Type.GreaterEqual:
                case Layer.Type.Less:
                case Layer.Type.LessEqual:
                case Layer.Type.Equal:
                case Layer.Type.LogicalOr:
                case Layer.Type.LogicalAnd:
                case Layer.Type.LogicalXor:
                {
                    int rankO = 0;
                    for (int i = 0; i < inputRanks.Length; i++)
                        rankO = Mathf.Max(inputRanks[i], rankO);
                    var O = new List<int>();
                    for (int i = 0; i < rankO; i++)
                        O.Add(1);
                    for (int i = 0; i < inputShapes.Length; i++)
                    {
                        TensorShape X = inputShapes[i];
                        int rankX = inputRanks[i];
                        List<int> xShape = ShapeToOnnxLayout(X, rankX);

                        for (int k = 0; k < rankO - rankX; k++)
                            xShape.Insert(0, 1);

                        for (int k = 0; k < rankO; k++)
                            O[k] = Math.Max(O[k], xShape[k]);
                    }

                    return OnnxLayoutToTensorShape(O.ToArray());
                }
                case Layer.Type.ReduceL1:
                case Layer.Type.ReduceL2:
                case Layer.Type.ReduceLogSum:
                case Layer.Type.ReduceLogSumExp:
                case Layer.Type.ReduceMax:
                case Layer.Type.ReduceMean:
                case Layer.Type.ReduceMin:
                case Layer.Type.ReduceProd:
                case Layer.Type.ReduceSum:
                case Layer.Type.ReduceSumSquare:
                case Layer.Type.ArgMax:
                case Layer.Type.ArgMin:
                {
                    TensorShape X = inputShapes[0];
                    int rank = inputRanks[0];
                    var xShape = ShapeToOnnxLayout(X, rank);

                    var axis = layer.axis;
                    if (axis < 0)
                        axis = rank + axis;

                    xShape[axis] = 1;
                    if (layer.alpha != 1.0f)  // keepdim == 0
                        xShape.RemoveAt(axis);

                    return OnnxLayoutToTensorShape(xShape.ToArray());
                }
                case Layer.Type.Transpose:
                {
                    TensorShape X = inputShapes[0];
                    var permutations = layer.pool;
                    if (permutations == null)
                        return new TensorShape(X.batch, X.width);
                    else
                    {
                        int rank = inputRanks[0];
                        List<int> xShape = ShapeToOnnxLayout(X, rank);

                        var oShape = TensorExtensions.Permute(xShape.ToArray(), permutations);
                        return OnnxLayoutToTensorShape(oShape);
                    }
                }
                case Layer.Type.MaxPool2D:
                case Layer.Type.AvgPool2D:
                {
                    TensorShape X = inputShapes[0];
                    X = new TensorShape(X.batch, X.width, X.channels, X.height);
                    Assert.IsNotNull(layer.pool);
                    Assert.IsNotNull(layer.stride);
                    Assert.IsNotNull(layer.pad);
                    var pad = X.AdjustPadToPool(layer.pool, layer.stride, layer.pad);
                    var O = X.ApplyPool(layer.pool, layer.stride, pad);
                    return new TensorShape(O.batch, O.channels, O.height, O.width);
                }
                case Layer.Type.Load:
                {
                    return layer.datasets[0].shape;
                }
                case Layer.Type.DepthToSpace:
                {
                    TensorShape X = inputShapes[0];
                    X = new TensorShape(X.batch, X.width, X.channels, X.height);
                    // pool size is treated as blocksize here
                    Assert.IsNotNull(layer.pool);
                    Assert.AreEqual(layer.pool.Length, 2);
                    Assert.AreEqual(X.channels % (layer.pool[0] * layer.pool[1]), 0);
                    var O = new TensorShape(X.batch, X.height * layer.pool[1], X.width * layer.pool[0], X.channels / (layer.pool[0] * layer.pool[1]));
                    return new TensorShape(O.batch, O.channels, O.height, O.width);
                }
                case Layer.Type.SpaceToDepth:
                {
                    TensorShape X = inputShapes[0];
                    X = new TensorShape(X.batch, X.width, X.channels, X.height);
                    // pool size is treated as blocksize here
                    Assert.IsNotNull(layer.pool);
                    Assert.AreEqual(layer.pool.Length, 2);
                    var O = new TensorShape(X.batch, X.height / layer.pool[1], X.width / layer.pool[0], X.channels * (layer.pool[0] * layer.pool[1]));
                    return new TensorShape(O.batch, O.channels, O.height, O.width);
                }
                case Layer.Type.RandomNormal:
                case Layer.Type.RandomUniform:
                {
                    Assert.IsNotNull(layer.pool);
                    // pool size is treated as shape constant, if not empty
                    // otherwise shape of the previous tensor is used
                    if (layer.pool.Length > 0)
                        return new TensorShape(layer.pool);
                    else
                        return inputShapes[0];
                }
                case Layer.Type.Multinomial:
                {
                    TensorShape X = inputShapes[0];
                    Assert.IsNotNull(layer.pool);
                    Assert.AreEqual(layer.pool.Length, 1);
                    return new TensorShape(X.batch, layer.pool[0]);
                }
                case Layer.Type.OneHot:
                {
                    TensorShape X = inputShapes[0];
                    int rank = inputRanks[0];
                    var nchwShape = ShapeToOnnxLayout(X, rank);
                    int depth = layer.pool[0];
                    nchwShape.Add(depth);

                    for (int i = 0; i < 4 - rank - 1; i++)
                        nchwShape.Add(1);

                    return new TensorShape(nchwShape[0], nchwShape[1], nchwShape[2], nchwShape[3]);
                }
                case Layer.Type.LSTM:
                {
                    TensorShape X = inputShapes[0];
                    var nchwShape = new List<int> { X.batch, X.height, X.width, X.channels };
                    int hiddenSize = layer.pool[0];

                    // The first output, Y, is rank 4; Other outputs are handled as identity layers
                    return new TensorShape(nchwShape[0], 1, nchwShape[1], hiddenSize);
                }
                case Layer.Type.Flatten:
                {
                    TensorShape X = inputShapes[0];
                    return X.Flatten();
                }
                case Layer.Type.Tile:
                {
                    TensorShape X = inputShapes[0];

                    if (inputShapes.Length > 1)
                        return null;

                    // pool size is treated as tiling coefficient here
                    Assert.IsNotNull(layer.pool);
                    Assert.AreEqual(layer.pool.Length, 4);
                    var scale = layer.pool;
                    return X.Scale(scale);
                }
                case Layer.Type.ConstantOfShape:
                {
                    if (inputShapes.Length == 1)
                        return inputShapes[0];
                    else
                        return OnnxLayoutToTensorShape(layer.pool);
                }
                case Layer.Type.Reshape:
                {
                    if (inputShapes.Length > 1)
                        return null;

                    // TODO shape to onnx shape given rank
                    TensorShape X = inputShapes[0];
                    int rank = inputRanks[0];
                    var nchwShape = ShapeToOnnxLayout(X, rank);

                    var unknownIndex = -1;
                    var multipleOf = 1;
                    var size = layer.pool.ToArray();
                    for (var i = 0; i < size.Length; ++i)
                    {
                        if (size[i] == 0)
                            size[i] = nchwShape[i];

                        if (size[i] < 0)
                            unknownIndex = i;
                        else
                            multipleOf *= size[i];
                    }

                    if (unknownIndex != -1)
                        size[unknownIndex] = X.length / multipleOf;

                    return OnnxLayoutToTensorShape(size);
                }
                case Layer.Type.Expand:
                {
                    if (inputShapes.Length > 1)
                        return null;
                    // pool size is treated as new shape
                    var size = layer.pool;

                    return OnnxLayoutToTensorShape(size);
                }
                case Layer.Type.Concat:
                {
                    var shape = ShapeToOnnxLayout(inputShapes[0], inputRanks[0]);
                    var axis = layer.axis;
                    if (axis < 0)
                        axis += inputRanks[0];

                    for (int i = 1; i < inputShapes.Length; i++)
                    {
                        var shapei = ShapeToOnnxLayout(inputShapes[i], inputRanks[i]);
                        shape[axis] += shapei[axis];
                    }

                    return OnnxLayoutToTensorShape(shape.ToArray());
                }
                case Layer.Type.Gather:
                {
                    var input0Shape = inputShapes[0];
                    var input1Shape = inputShapes[1];


                    int rank0 = inputRanks[0];
                    var shape = ShapeToOnnxLayout(input0Shape, rank0);
                    var axis = layer.axis;
                    if (axis < 0)
                        axis += rank0;

                    shape[axis] = input1Shape.length;

                    return OnnxLayoutToTensorShape(shape.ToArray());
                }
                // elementwise operations
                case Layer.Type.Nop:
                case Layer.Type.ScaleBias:
                case Layer.Type.Normalization:
                case Layer.Type.LRN:
                case Layer.Type.Dropout:
                case Layer.Type.LogicalNot:
                case Layer.Type.Where:
                {
                    // works in place, keeps the same shape size
                    return inputShapes[0];
                }
                case Layer.Type.Activation:
                {
                    TensorShape X = inputShapes[0];

                    // LSTMs have multiple outputs, so deal with those separately
                    if (layer.activation == Layer.Activation.None && layer.pad.Length > 0
                        && layer.name.IndexOf("lstm", StringComparison.OrdinalIgnoreCase) >= 0)
                    {
                        int rank = layer.pad[0];
                        switch (rank)
                        {
                            case 4:
                                // Y
                                return X;

                            case 3:
                                // Y_h, Y_c: seq_length is stripped off
                                return new TensorShape(X[1], X[2], X[3]);
                        }
                    }

                    // works in place, keeps the same shape size
                    return X;
                }
                case Layer.Type.Shape:
                {
                    TensorShape X = inputShapes[0];
                    int rank = inputRanks[0];
                    return new TensorShape(rank);
                }
                case Layer.Type.Squeeze:
                {
                    TensorShape X = inputShapes[0];
                    int rank = inputRanks[0];

                    var nchwShape = ShapeToOnnxLayout(X, rank);

                    var squeezedShape = new List<int>();
                    for (int i = 0; i < nchwShape.Count; i++)
                    {
                        if (!layer.pool.Contains(i))
                            squeezedShape.Add(nchwShape[i]);
                    }

                    return OnnxLayoutToTensorShape(squeezedShape.ToArray());
                }
                case Layer.Type.Unsqueeze:
                {
                    TensorShape X = inputShapes[0];
                    int rank = inputRanks[0];

                    if (rank < 0)
                        return null;

                    var nchwShape = ShapeToOnnxLayout(X, rank);

                    if (rank == 0)
                        return new TensorShape(new int[] { 1, 1, 1, 1 });

                    for (int a = 0; a < layer.pool.Length; a++)
                    {
                        var axis = layer.pool[a];
                        if (axis < 0)
                            axis += rank;

                        nchwShape.Insert(axis, 1);
                        rank++;
                    }

                    return OnnxLayoutToTensorShape(nchwShape.ToArray());
                }
                case Layer.Type.StridedSlice:
                {
                    TensorShape X = inputShapes[0];
                    int rank = inputRanks[0];
                    var nchwShape = ShapeToOnnxLayout(X, rank);

                    var starts = layer.pad;
                    var ends = layer.pool;
                    var steps = layer.stride;
                    var axes = layer.axes;

                    var onnxStarts = Enumerable.Repeat(0, rank).ToArray();
                    var onnxEnds = Enumerable.Repeat(int.MaxValue, rank).ToArray(); // by default copy the whole axis till the end
                    var onnxSteps = Enumerable.Repeat(1, rank).ToArray();

                    // NOTE: begin=0, end=0, stride=1  <=  full range from existing axis
                    //       begin=0, end=inf,stride=1 <=  full range from existing axis
                    //       begin=0, end=X, stride=1  <=  full range from existing axis, if X==last element on this axis
                    //       begin=0, end=0, stride=0  <=  new axis OR shrink axis to single 1st element
                    //       begin=N, end=N, stride=0  <=              shrink axis to single Nth element
                    // These notes are copied from TensorExtensions.ApplyStridedSlice(...)

                    for (int i = 0; i < axes.Length; ++i)
                    {
                        var axis = axes[i];
                        if (axis < 0)
                            axis += rank;
                        axis = Math.Min(Math.Max(axis, 0), rank);

                        onnxStarts[axis] = starts[i];
                        onnxEnds[axis] = ends[i];
                        onnxSteps[axis] = steps[i];
                    }


                    var counts = new int[rank];
                    var sliced = new int[rank];
                    for (int i = 0; i < rank; ++i)
                    {
                        counts[i] = nchwShape[i];
                        sliced[i] = nchwShape[i];
                    }


                    for (int i = 0; i < rank; ++i)
                    {
                        // NOTE: begin=0, end=0, stride=1  <=  full range from the existing axis
                        //       begin=0, end=X, stride=1  <=  full range from the existing axis, if X==last element on this axis
                        //       begin=0, end=0, stride=0  <=  new axis OR shrink axis to a single 1st element
                        //       begin=N, end=N, stride=0  <=              shrink axis to a single Nth element

                        Assert.IsTrue(onnxStarts[i] < counts[i]);
                        if (onnxStarts[i] != onnxEnds[i])
                            sliced[i] = TensorExtensions.WrapIndex(onnxEnds[i], counts[i]) - TensorExtensions.WrapIndex(onnxStarts[i], counts[i]);
                        else
                            sliced[i] = counts[i];
                        if (onnxSteps[i] != 0 && onnxSteps[i] < counts[i])
                            sliced[i] /= onnxSteps[i];
                        else
                            sliced[i] = 1;

                        if (sliced[i] < 0)
                            sliced[i] = counts[i] + sliced[i];

                        if (sliced[i] < 0)
                            sliced[i] = 0;
                    }


                    return OnnxLayoutToTensorShape(sliced.ToArray());
                }
                default:
                    throw new NotImplementedException("InferOutputShapeNCHW: Unhandled layer: " + layer.ToString());
            }
        }

        // TODO merge that with NHWC : flank by transpose shape and call InferOutputShapeNHWC
        public static void UpdateKnownTensorShapesNCHW(Model model, IDictionary<string, int?> ranksByName, ref IDictionary<string, TensorShape?> shapesByName)
        {
            foreach (var l in model.layers)
            {
                if (shapesByName.ContainsKey(l.name) && shapesByName[l.name] != null)
                    continue;

                TensorShape[] layerInputShapes = new TensorShape[l.inputs.Length];
                int[] layerInputShapeRanks = new int[l.inputs.Length];

                bool allshapesAreKnown = true;
                for (int i = 0; i < l.inputs.Length; i++)
                {
                    shapesByName.TryGetValue(l.inputs[i], out TensorShape? ishape);

                    if (ishape == null)
                    {
                        allshapesAreKnown = false;
                        break;
                    }

                    layerInputShapes[i] = ishape.Value;
                    layerInputShapeRanks[i] = ranksByName[l.inputs[i]].Value;
                }
                TensorShape? outputShape = allshapesAreKnown ? InferOutputShapeNCHW(l, layerInputShapeRanks, layerInputShapes) : null;
                shapesByName[l.name] = outputShape;
            }
        }
        public static TensorShape?[] ListTemporaryTensorShapesNCHW(Model model, IDictionary<string, TensorShape> inputShapes, IDictionary<string, int?> ranksByName,
            out IDictionary<string, TensorShape?> shapesByName)
        {
            Profiler.BeginSample("Barracuda.ListTemporaryTensorShapesNCHW");
            var shapes = new List<TensorShape?>();
            shapesByName = new Dictionary<string, TensorShape?>();
            foreach (var i in inputShapes)
                shapesByName.Add(i.Key, i.Value);

            foreach (var l in model.layers)
            {
                TensorShape[] layerInputShapes = new TensorShape[l.inputs.Length];
                int[] layerInputShapeRanks = new int[l.inputs.Length];

                bool allshapesAreKnown = true;
                for (int i = 0; i < l.inputs.Length; i++)
                {
                    shapesByName.TryGetValue(l.inputs[i], out TensorShape? ishape);

                    if (ishape == null)
                    {
                        allshapesAreKnown = false;
                        break;
                    }

                    layerInputShapes[i] = ishape.Value;
                    layerInputShapeRanks[i] = ranksByName[l.inputs[i]].Value;
                }
                TensorShape? outputShape = allshapesAreKnown ? InferOutputShapeNCHW(l, layerInputShapeRanks, layerInputShapes) : null;

                shapes.Add(outputShape);
                shapesByName.Add(l.name, outputShape);
            }

            Profiler.EndSample();
            return shapes.ToArray();
        }
    }
}
