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
        static public TensorShape OnnxLayoutToBarracudaTensorShape(int[] size)
        {
            if (size.Length == 0)
                return new TensorShape(1, 1, 1, 1);
            else if (size.Length == 1)
                return new TensorShape(size[0], 1, 1, 1);
            else if (size.Length == 2)
                return new TensorShape(size[0], 1, 1, size[1]);
            else if (size.Length == 3)
                return new TensorShape(size[0], 1, size[2], size[1]);
            else if (size.Length == 4)
                return new TensorShape(size[0], size[2], size[3], size[1]);
            else if (size.Length == 5)
                return new TensorShape(size[0], size[2], size[3], size[4], size[1]);
            else if (size.Length == 6)
                return new TensorShape(1, 1, size[0], size[2], size[3], size[4], size[5], size[1]);
            else
                return new TensorShape(size[0], size[1], size[2], size[4], size[5], size[6], size[7], size[3]);
        }

        static public List<int> BarracudaShapeToList(TensorShape X, int rank)
        {
            if (rank == 0)
                return new List<int> { 1 };
            else if (rank == 1)
                return new List<int> { X.batch };
            else if (rank == 2)
                return new List<int> { X.batch, X.channels };
            else if (rank == 3)
                return new List<int> { X.batch, X.width, X.channels };
            else if (rank == 4)
                return new List<int> { X.batch, X.height, X.width, X.channels };
            else if (rank == 5)
                return new List<int> { X.batch, X.depth, X.height, X.width, X.channels };
            else if (rank == 6)
                return new List<int> { X.batch, X.depth, X.extraDimension, X.height, X.width, X.channels };
            else
                return new List<int> { X.sequenceLength, X.numberOfDirections, X.batch, X.extraDimension, X.depth, X.height, X.width, X.channels };
        }

        static public int BarracudaAxisToTensor(int axis, int rank)
        {
            if (rank == 0)
                return 0;
            else if (rank == 1)
                return 0;
            else if (rank == 2)
                return axis == TensorShape.DataBatch ? 0 : 1;
            else if (rank == 3)
                return axis == TensorShape.DataBatch ? 0 : axis - TensorShape.W + 1;
            else if (rank == 4)
                return axis == TensorShape.DataBatch ? 0 : axis - TensorShape.H + 1;
            else if (rank == 5)
                return axis == TensorShape.DataBatch ? 0 : axis - TensorShape.D + 1;
            else if (rank == 6)
                return axis == TensorShape.DataBatch ? 0 : axis - TensorShape.DataFeature3 + 1;
            else
                return axis;
        }

        static public TensorShape? InferOutputShapeNCHW(Layer layer, int?[] inputRanks, TensorShape?[] inputShapes)
        {
            switch (layer.type)
            {
                case Layer.Type.Conv3D:
                {
                    if(inputShapes[0] == null)
                        return null;

                    TensorShape X = inputShapes[0].Value;
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
                    if(inputShapes[0] == null)
                        return null;

                    TensorShape X = inputShapes[0].Value;
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
                    if(inputShapes[0] == null)
                        return null;

                    TensorShape X = inputShapes[0].Value;
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
                    if(inputShapes[0] == null)
                        return null;

                    TensorShape X = inputShapes[0].Value;
                    int rankX = inputRanks[0].Value;
                    List<int> xShape = ShapeToOnnxLayout(X, rankX);

                    for (int i = 2; i < xShape.Count; i++)
                        xShape[i] = 1;
                    return OnnxLayoutToTensorShape(xShape.ToArray());
                }
                case Layer.Type.Dense:
                {
                    if(inputShapes[0] == null)
                        return null;

                    TensorShape X = inputShapes[0].Value;
                    X = new TensorShape(X.batch, X.width, X.channels, X.height);
                    Assert.IsNotNull(layer.datasets);
                    var W = layer.datasets[0].shape;
                    var O = new TensorShape(X.flatHeight, W.flatWidth);
                    return new TensorShape(O.batch, O.channels, O.height, O.width);
                }
                case Layer.Type.MatMul:
                {
                    if(inputShapes[0] == null || inputShapes[1] == null)
                        return null;

                    TensorShape X = inputShapes[0].Value;
                    int rankX = inputRanks[0].Value;
                    List<int> xShape = ShapeToOnnxLayout(X, rankX);

                    TensorShape Y = inputShapes[1].Value;
                    int rankY = inputRanks[1].Value;
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
                case Layer.Type.Pad:
                {
                    if (inputShapes.Length > 1)
                        return null;

                    if(inputShapes[0] == null)
                        return null;

                    TensorShape X = inputShapes[0].Value;
                    int rankX = inputRanks[0].Value;
                    List<int> xShape = ShapeToOnnxLayout(X, rankX);


                    for (int i = 0; i < xShape.Count; i++)
                    {
                        xShape[i] += layer.pad[i] + layer.pad[rankX + i];
                    }

                    return OnnxLayoutToTensorShape(xShape.ToArray());
                }
                case Layer.Type.Upsample2D:
                {
                    if (inputShapes.Length > 1)
                        return null;

                    if(inputShapes[0] == null)
                        return null;

                    TensorShape X = inputShapes[0].Value;

                    // pool size is treated as upsample coefficient here
                    Assert.IsNotNull(layer.pool);
                    Assert.AreEqual(layer.pool.Length, 4);
                    return new TensorShape(X.batch * layer.pool[0], X.height * layer.pool[1], X.width * layer.pool[2], X.channels * layer.pool[3]);
                }
                case Layer.Type.Upsample3D:
                {
                    if (inputShapes.Length > 1)
                        return null;

                    if(inputShapes[0] == null)
                        return null;

                    TensorShape X = inputShapes[0].Value;

                    // pool size is treated as upsample coefficient here
                    Assert.IsNotNull(layer.pool);
                    Assert.AreEqual(layer.pool.Length, 5);
                    return new TensorShape(X.batch * layer.pool[0], X.depth * layer.pool[1], X.height * layer.pool[2], X.width * layer.pool[3], X.channels * layer.pool[4]);
                }
                case Layer.Type.Resample2D:
                {
                    if (inputShapes.Length > 1)
                        return null;

                    if(inputShapes[0] == null)
                        return null;

                    TensorShape X = inputShapes[0].Value;

                    // pool is treated as resample size here
                    var size = layer.pool;
                    Assert.IsNotNull(size);
                    Assert.AreEqual(size.Length, 4);
                    return new TensorShape(size[0], size[1], size[2], size[3]);
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
                    if(inputShapes.Any(x => x == null))
                        return null;


                    int rankO = inputRanks.Max().Value;

                    var O = new List<int>();
                    for (int i = 0; i < rankO; i++)
                        O.Add(1);
                    for (int i = 0; i < inputShapes.Length; i++)
                    {
                        TensorShape X = inputShapes[i].Value;
                        int rankX = inputRanks[i].Value;
                        List<int> xShape = ShapeToOnnxLayout(X, rankX);

                        for (int k = 0; k < rankO - rankX; k++)
                            xShape.Insert(0, 1);

                        for (int k = 0; k < rankO; k++)
                            O[k] = Math.Max(O[k], xShape[k]);
                    }

                    return OnnxLayoutToTensorShape(O.ToArray());
                }
                case Layer.Type.Range:
                {
                    return null; // only const support
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
                    if(inputShapes[0] == null)
                        return null;

                    TensorShape X = inputShapes[0].Value;

                    int rank = inputRanks[0].Value;
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
                    if(inputShapes[0] == null)
                        return null;

                    TensorShape X = inputShapes[0].Value;
                    var permutations = layer.pool;
                    if (permutations == null)
                        return new TensorShape(X.batch, X.width);
                    else
                    {
                        int rank = inputRanks[0].Value;
                        List<int> xShape = ShapeToOnnxLayout(X, rank);

                        // Permutations may already be in padded form for op purposes, so strip down to match rank
                        permutations = permutations.Take(rank).ToArray();

                        var oShape = TensorExtensions.Permute(xShape.ToArray(), permutations);
                        return OnnxLayoutToTensorShape(oShape);
                    }
                }
                case Layer.Type.MaxPool2D:
                case Layer.Type.AvgPool2D:
                {
                    if(inputShapes[0] == null)
                        return null;

                    TensorShape X = inputShapes[0].Value;
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
                    if(inputShapes[0] == null)
                        return null;

                    TensorShape X = inputShapes[0].Value;
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
                    if(inputShapes[0] == null)
                        return null;

                    TensorShape X = inputShapes[0].Value;
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
                    if(inputShapes[0] == null)
                        return null;

                    TensorShape X = inputShapes[0].Value;
                    Assert.IsNotNull(layer.pool);
                    Assert.AreEqual(layer.pool.Length, 1);
                    return new TensorShape(X.batch, layer.pool[0]);
                }
                case Layer.Type.OneHot:
                {
                    if(inputShapes[0] == null)
                        return null;

                    TensorShape X = inputShapes[0].Value;
                    int rank = inputRanks[0].Value;
                    var nchwShape = ShapeToOnnxLayout(X, rank);
                    int depth = layer.pool[0];
                    nchwShape.Add(depth);

                    return OnnxLayoutToTensorShape(nchwShape.ToArray());
                }
                case Layer.Type.RoiAlign:
                {
                    if(inputShapes[0] == null)
                        return null;

                    TensorShape X = inputShapes[0].Value;
                    TensorShape rois = inputShapes[0].Value;

                    return new TensorShape(rois.batch, X.height, layer.pool[0], layer.pool[1]);
                }
                case Layer.Type.LSTM:
                {
                    if(inputShapes[0] == null)
                        return null;

                    TensorShape X = inputShapes[0].Value;
                    var nchwShape = new List<int> { X.batch, X.height, X.width, X.channels };
                    int hiddenSize = layer.pool[0];

                    // The first output, Y, is rank 4; Other outputs are handled as identity layers
                    return new TensorShape(nchwShape[0], 1, nchwShape[1], hiddenSize);
                }
                case Layer.Type.Flatten:
                {
                    if(inputShapes[0] == null)
                        return null;

                    TensorShape X = inputShapes[0].Value;
                    return X.Flatten();
                }
                case Layer.Type.Tile:
                {
                    if (inputShapes.Length > 1)
                        return null;

                    if(inputShapes[0] == null)
                        return null;

                    var inputShape = ShapeToOnnxLayout(inputShapes[0].Value, inputRanks[0].Value);
                    var scale = layer.pool.ToArray();
                    Assert.IsNotNull(scale);
                    Assert.AreEqual(scale.Length, inputShape.Count);

                    for (int i = 0; i < scale.Length; i++)
                        scale[i] *= inputShape[i];

                    return OnnxLayoutToTensorShape(scale);
                }
                case Layer.Type.ConstantOfShape:
                {
                    if(layer.axis == 1)
                        return inputShapes[0];

                    if (inputShapes.Length == 1)
                        return null;
                    else
                        return OnnxLayoutToTensorShape(layer.pool);
                }
                case Layer.Type.Reshape:
                {
                    if (inputShapes.Length > 1)
                        return null;

                    if (inputShapes[0] == null)
                        return null;

                    // TODO shape to onnx shape given rank
                    TensorShape X = inputShapes[0].Value;
                    int rank = inputRanks[0].Value;
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

                    if(inputShapes[0] == null)
                        return null;

                    var size = layer.pool.ToList();
                    var inputShape = ShapeToOnnxLayout(inputShapes[0].Value, inputRanks[0].Value);

                    int rankO = Math.Max(size.Count, inputShape.Count);
                    for (int i = 0; i < rankO - size.Count; i++)
                        size.Insert(0, 1);
                    for (int i = 0; i < rankO - inputShape.Count; i++)
                        inputShape.Insert(0, 1);

                    var tiledShape = new int[rankO];
                    for (int i = 0; i < rankO; i++)
                        tiledShape[i] = Mathf.Max(size[i], inputShape[i]);

                    return OnnxLayoutToTensorShape(tiledShape);
                }
                case Layer.Type.Concat:
                {
                    if(inputShapes.Any(x => x == null))
                        return null;

                    int maxRank = inputRanks.Max().Value;

                    var shape = ShapeToOnnxLayout(inputShapes[0].Value, maxRank);
                    var axis = layer.axis;
                    if (axis < 0)
                        axis += maxRank;

                    for (int i = 1; i < inputShapes.Length; i++)
                    {
                        var shapei = ShapeToOnnxLayout(inputShapes[i].Value, maxRank);
                        shape[axis] += shapei[axis];
                    }

                    return OnnxLayoutToTensorShape(shape.ToArray());
                }
                case Layer.Type.Gather:
                {
                    if(inputShapes[0] == null || inputShapes[1] == null)
                        return null;

                    var input0Shape = inputShapes[0].Value;
                    var input1Shape = inputShapes[1].Value;


                    int rank0 = inputRanks[0].Value;
                    int rank1 = inputRanks[1].Value;
                    var shape = ShapeToOnnxLayout(input0Shape, rank0);
                    var indicies = ShapeToOnnxLayout(input1Shape, rank1);

                    var axis = layer.axis;
                    if (axis < 0)
                        axis += rank0;

                    shape.InsertRange(axis, indicies);
                    shape.RemoveAt(axis + indicies.Count);

                    return OnnxLayoutToTensorShape(shape.ToArray());
                }
                case Layer.Type.ScatterND:
                    return inputShapes[0];
                // elementwise operations
                case Layer.Type.Nop:
                case Layer.Type.ScaleBias:
                case Layer.Type.Normalization:
                case Layer.Type.LRN:
                case Layer.Type.Dropout:
                case Layer.Type.LogicalNot:
                case Layer.Type.Sign:
                case Layer.Type.Where:
                {
                    // works in place, keeps the same shape size
                    return inputShapes[0];
                }
                case Layer.Type.Activation:
                {
                    // LSTMs have multiple outputs, so deal with those separately
                    if (layer.activation == Layer.Activation.None && layer.pad.Length > 0
                        && layer.name.IndexOf("lstm", StringComparison.OrdinalIgnoreCase) >= 0)
                    {
                        int rank = layer.pad[0];
                        switch (rank)
                        {
                            case 4:
                                // Y
                                return inputShapes[0];

                            case 3:
                            {
                                if (inputShapes[0] == null)
                                    return null;

                                TensorShape X = inputShapes[0].Value;
                                // Y_h, Y_c: seq_length is stripped off
                                return new TensorShape(X[1], X[2], X[3]);
                            }
                        }
                    }

                    // works in place, keeps the same shape size
                    return inputShapes[0];
                }
                case Layer.Type.Shape:
                {
                    if(inputRanks[0] == null)
                        return null;

                    int rank = inputRanks[0].Value;
                    return new TensorShape(rank);
                }
                case Layer.Type.Squeeze:
                {
                    if(inputShapes[0] == null)
                        return null;

                    TensorShape X = inputShapes[0].Value;
                    int rank = inputRanks[0].Value;

                    if (inputShapes.Length > 1)
                        return null;

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
                    if(inputShapes[0] == null)
                        return null;

                    TensorShape X = inputShapes[0].Value;
                    int rank = inputRanks[0].Value;

                    if (inputShapes.Length > 1)
                        return null;

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
                    if (inputShapes.Length > 1)
                        return null;

                    if(inputShapes[0] == null)
                        return null;

                    TensorShape X = inputShapes[0].Value;
                    int rank = inputRanks[0].Value;
                    var nchwShape = ShapeToOnnxLayout(X, rank);

                    var starts = layer.pad.ToArray();
                    var ends = layer.pool.ToArray();
                    var steps = layer.stride.ToArray();
                    var axes = layer.axes.ToArray();

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

                    var sliced = new int[rank];
                    for (int i = 0; i < rank; ++i)
                    {
                        // NOTE: begin=0, end=0, stride=1  <=  full range from the existing axis
                        //       begin=0, end=X, stride=1  <=  full range from the existing axis, if X==last element on this axis
                        //       begin=0, end=0, stride=0  <=  new axis OR shrink axis to a single 1st element
                        //       begin=N, end=N, stride=0  <=              shrink axis to a single Nth element
                        int ei = TensorExtensions.WrapIndex(onnxEnds[i], nchwShape[i]);
                        int si = TensorExtensions.WrapIndex(onnxStarts[i], nchwShape[i]);

                        if (onnxSteps[i] > 0)
                            sliced[i] = (int)Mathf.Round((float)(Math.Min(ei, nchwShape[i]) - Math.Min(si, nchwShape[i] - 1)) / (float)(Mathf.Abs(onnxSteps[i])));
                        else
                        {
                            bool inclusive = onnxEnds[i] < -nchwShape[i]; // edge case when ends is negative and bigger than nchwShape
                            sliced[i] = (int)Mathf.Round((float)(Math.Min(si, nchwShape[i] - 1) - Math.Min(ei, nchwShape[i]) + (inclusive ? 1 : 0)) / (float)(Mathf.Abs(onnxSteps[i])));
                        }
                    }

                        return OnnxLayoutToTensorShape(sliced.ToArray());
                }
                default:
                    throw new NotImplementedException("InferOutputShapeNCHW: Unhandled layer: " + layer.ToString());
            }
        }

        // TODO merge that with NHWC : flank by transpose shape and call InferOutputShapeNHWC
        public static void UpdateKnownTensorShapesNCHW(Model model, ref IDictionary<string, int?> ranksByName, ref IDictionary<string, TensorShape?> shapesByName)
        {
            foreach (var l in model.layers)
            {
                TensorShape?[] layerInputShapes = new TensorShape?[l.inputs.Length];
                int?[] layerInputShapeRanks = new int?[l.inputs.Length];

                for (int i = 0; i < l.inputs.Length; i++)
                {
                    shapesByName.TryGetValue(l.inputs[i], out TensorShape? ishape);
                    ranksByName.TryGetValue(l.inputs[i], out int? irank);

                    layerInputShapes[i] = ishape;
                    layerInputShapeRanks[i] = irank;
                }

                // knowing rank might imply knowing shape:
                // + compute rank first
                // + compute shape
                // knowing shape might imply knowing rank:
                // + compute rank
                int? outputRank = RankInference.InferOutputRank(l, layerInputShapeRanks, layerInputShapes);
                ranksByName[l.name] = outputRank;
                TensorShape? outputShape = InferOutputShapeNCHW(l, layerInputShapeRanks, layerInputShapes);
                outputRank = RankInference.InferOutputRank(l, layerInputShapeRanks, layerInputShapes);
                ranksByName[l.name]  = outputRank;
                shapesByName[l.name] = outputShape;
            }
        }
        public static TensorShape?[] ListTemporaryTensorShapesNCHW(Model model, IDictionary<string, TensorShape> inputShapes, ref IDictionary<string, int?> ranksByName,
            out IDictionary<string, TensorShape?> shapesByName)
        {
            Profiler.BeginSample("Barracuda.ListTemporaryTensorShapesNCHW");
            var shapes = new List<TensorShape?>();
            shapesByName = new Dictionary<string, TensorShape?>();
            foreach (var i in inputShapes)
                shapesByName.Add(i.Key, i.Value);

            foreach (var l in model.layers)
            {
                TensorShape?[] layerInputShapes = new TensorShape?[l.inputs.Length];
                int?[] layerInputShapeRanks = new int?[l.inputs.Length];

                for (int i = 0; i < l.inputs.Length; i++)
                {
                    shapesByName.TryGetValue(l.inputs[i], out TensorShape? ishape);
                    ranksByName.TryGetValue(l.inputs[i], out int? irank);

                    layerInputShapes[i] = ishape;
                    layerInputShapeRanks[i] = irank;
                }


                int? outputRank = RankInference.InferOutputRank(l, layerInputShapeRanks, layerInputShapes);
                ranksByName[l.name] = outputRank;
                TensorShape? outputShape = InferOutputShapeNCHW(l, layerInputShapeRanks, layerInputShapes);
                outputRank = RankInference.InferOutputRank(l, layerInputShapeRanks, layerInputShapes);
                ranksByName[l.name] = outputRank;

                shapes.Add(outputShape);
                shapesByName.Add(l.name, outputShape);
            }

            Profiler.EndSample();
            return shapes.ToArray();
        }
    }
}
