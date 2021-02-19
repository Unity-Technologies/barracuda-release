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
    internal class RankInference
    {
        static public int? InferOutputRank(Layer layer, int[] inputRanks)
        {
            switch (layer.type)
            {
                case Layer.Type.Dense:
                {
                    Assert.AreEqual(inputRanks.Length, 1, "InferOutputRank.Dense inputRanks.Length"); Assert.AreEqual(inputRanks[0], 2, "InferOutputRank.Dense inputRanks");
                    return 2;
                }
                case Layer.Type.MatMul:
                {
                    int maxRank = inputRanks[0];
                    for (int i = 1; i < inputRanks.Length; i++)
                        maxRank = Mathf.Max(maxRank, inputRanks[i]);
                    return maxRank;
                }
                case Layer.Type.Conv3D:
                {
                    Assert.AreEqual(inputRanks.Length, 1, "InferOutputRank.*Conv3D* inputRanks.Length"); Assert.IsTrue(inputRanks[0] >= 2 && inputRanks[0] <= 5, "InferOutputRank.*Conv3D* inputRanks");
                    return inputRanks[0];
                }
                case Layer.Type.Conv2D:
                case Layer.Type.DepthwiseConv2D:
                case Layer.Type.Conv2DTrans:
                {
                    Assert.AreEqual(inputRanks.Length, 1, "InferOutputRank.*Conv2D* inputRanks.Length"); Assert.IsTrue(inputRanks[0] >= 2 && inputRanks[0] <= 4, "InferOutputRank.*Conv2D* inputRanks"); // conv1D/2D are done via conv2D
                    return inputRanks[0];
                }
                case Layer.Type.DepthToSpace:
                case Layer.Type.SpaceToDepth:
                {
                    Assert.AreEqual(inputRanks.Length, 1, "InferOutputRank.ToDepth/Space inputRanks.Length"); Assert.AreEqual(inputRanks[0], 4, "InferOutputRank.ToDepth/Space inputRanks");
                    return 4;
                }
                case Layer.Type.Upsample3D:
                {
                    Assert.AreEqual(inputRanks[0], 5, "InferOutputRank.*Upsample3D inputRanks");
                    return 5;
                }
                case Layer.Type.Upsample2D:
                case Layer.Type.Resample2D:
                {
                    Assert.AreEqual(inputRanks[0], 4, "InferOutputRank.*Upsample2D inputRanks");
                    return 4;
                }
                case Layer.Type.MaxPool2D:
                case Layer.Type.AvgPool2D:
                {
                    // TODO 3D case?
                    Assert.AreEqual(inputRanks.Length, 1, "InferOutputRank.AvgPool2D inputRanks.Length"); Assert.IsTrue(inputRanks[0] == 4 || inputRanks[0] == 3, "InferOutputRank.AvgPool2D inputRanks");
                    return inputRanks[0];
                }
                case Layer.Type.GlobalMaxPool2D:
                case Layer.Type.GlobalAvgPool2D:
                {
                    Assert.AreEqual(inputRanks.Length, 1, "InferOutputRank.Global*Pool2D inputRanks.Length"); Assert.IsTrue(inputRanks[0] == 4 || inputRanks[0] == 3, "InferOutputRank.Global*Pool2D inputRanks");
                    return inputRanks[0];
                }
                case Layer.Type.Border3D:
                {
                    Assert.AreEqual(inputRanks.Length, 1, "InferOutputRank.*Pad inputRanks.Length"); Assert.AreEqual(inputRanks[0], 5, "InferOutputRank.*Pad inputRanks");
                    return 5;
                }
                case Layer.Type.Border2D:
                case Layer.Type.Pad2DReflect:
                case Layer.Type.Pad2DSymmetric:
                case Layer.Type.Pad2DEdge:
                {
                    Assert.AreEqual(inputRanks.Length, 1, "InferOutputRank.*Pad inputRanks.Length"); Assert.AreEqual(inputRanks[0], 4, "InferOutputRank.*Pad inputRanks");
                    return 4;
                }
                case Layer.Type.RandomNormal:
                case Layer.Type.RandomUniform:
                {
                    if (layer.pool.Length > 0)
                        return layer.pool.Length;
                    else
                    {
                        Assert.AreEqual(inputRanks.Length, 1, "InferOutputRank.*Random inputRanks.Length");
                        return inputRanks[0];
                    }
                }
                case Layer.Type.Multinomial:
                        return 2;
                case Layer.Type.OneHot:
                {
                    Assert.AreEqual(inputRanks.Length, 1, "InferOutputRank.OneHot inputRanks.Length");
                    return inputRanks[0] + 1;
                }
                case Layer.Type.LSTM:
                        return 4;
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
                    int maxRank = inputRanks[0];
                    for (int i = 1; i < inputRanks.Length; i++)
                        maxRank = Mathf.Max(maxRank, inputRanks[i]);
                    return maxRank;
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
                    if (layer.alpha != 1.0f)
                        return inputRanks[0] - 1;
                    else
                        return inputRanks[0];
                }
                case Layer.Type.Flatten:
                    return 2;
                case Layer.Type.ConstantOfShape:
                {
                    if (inputRanks.Length == 1)
                        return inputRanks[0];
                    else
                        return layer.pool.Length;
                }
                case Layer.Type.Reshape:
                {
                    if (inputRanks.Length > 1)
                        return null;

                    if (layer.pad.Length > 0)
                        return layer.pad[0]; // original rank stored here

                    return layer.pool.Length;
                }
                case Layer.Type.Expand:
                {
                    if (inputRanks.Length > 1)
                        return null;

                    return layer.pool.Length;
                }
                case Layer.Type.Transpose:
                    return inputRanks[0];
                case Layer.Type.Gather:
                {
                    // Gather can implicilty do a squeeze in inputs are single int
                    // we don't but instead append a squeeze op after Gather if that is the case
                    return inputRanks[0] + Mathf.Max(inputRanks[1], 1) - 1;
                }
                case Layer.Type.TopKIndices:
                case Layer.Type.TopKValues:
                    return inputRanks[0];
                case Layer.Type.NonMaxSuppression:
                    return 2;
                case Layer.Type.NonZero:
                    return 2;
                case Layer.Type.Squeeze:
                    return inputRanks[0] - layer.pool.Length;
                case Layer.Type.Unsqueeze:
                    return inputRanks[0] + layer.pool.Length;
                case Layer.Type.Concat:
                {
                    int maxRank = inputRanks[0];
                    return inputRanks.Max();
                }
                case Layer.Type.StridedSlice:
                    // TODO : figure out if slice can produce lower rank output
                    return inputRanks[0];
                case Layer.Type.Tile:
                    return inputRanks[0];
                case Layer.Type.Load:
                {
                    if (layer.datasets[0].length == 1 && layer.axis == 1)
                        return 0; // TODO const float vs [float] maybe override rank in ONNXTensor
                    return layer.axis;
                }
                case Layer.Type.Nop:
                case Layer.Type.ScaleBias:
                case Layer.Type.Normalization:
                case Layer.Type.LRN:
                case Layer.Type.Dropout:
                case Layer.Type.LogicalNot:
                case Layer.Type.Where:
                {
                    return inputRanks[0];
                }
                case Layer.Type.Activation:
                {
                    // LSTMs have multiple outputs, so deal with those separately
                    if (layer.activation == Layer.Activation.None && layer.pad.Length > 0
                && layer.name.IndexOf("lstm", StringComparison.OrdinalIgnoreCase) >= 0)
                    {
                        return layer.pad[0];
                    }

                    return inputRanks[0];
                }
                case Layer.Type.Shape:
                    return 1;
                default:
                    return null;
            }
        }

        // TODO merge List&Update***
        public static void UpdateKnownTensorRanks(Model model, IDictionary<string, int?> ranksByName)
        {
            foreach (var l in model.layers)
            {
                if (ranksByName.ContainsKey(l.name) && ranksByName[l.name] != null)
                    continue;

                int[] inputRanks = new int[l.inputs.Length];

                bool allshapesAreKnown = true;
                for (int i = 0; i < l.inputs.Length; i++)
                {
                    ranksByName.TryGetValue(l.inputs[i], out int? irank);

                    if (irank == null)
                    {
                        allshapesAreKnown = false;
                        break;
                    }

                    inputRanks[i] = irank.Value;
                }

                int? outputRank = allshapesAreKnown ? InferOutputRank(l, inputRanks) : null;
                ranksByName[l.name] = outputRank;
            }
        }

        public static int?[] ListTemporaryTensorRanks(Model model,
            out IDictionary<string, int?> ranksByName)
        {
            Profiler.BeginSample("Barracuda.ListTemporaryTensorRanks");
            var ranks = new List<int?>();
            ranksByName = new Dictionary<string, int?>();
            foreach (var i in model.inputs)
                ranksByName.Add(i.name, i.rank);

            foreach (var l in model.layers)
            {
                int[] inputRanks = new int[l.inputs.Length];
                bool allshapesAreKnown = true;
                for (int i = 0; i < l.inputs.Length; i++)
                {
                    ranksByName.TryGetValue(l.inputs[i], out int? irank);

                    if (irank == null)
                    {
                        allshapesAreKnown = false;
                        break;
                    }

                    inputRanks[i] = irank.Value;
                }

                int? outputRank = allshapesAreKnown ? InferOutputRank(l, inputRanks) : null;

                ranks.Add(outputRank);
                ranksByName.Add(l.name, outputRank);
            }

            Profiler.EndSample();
            return ranks.ToArray();
        }
    }
}
