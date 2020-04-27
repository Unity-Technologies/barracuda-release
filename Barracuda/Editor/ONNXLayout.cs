using Onnx;
using UnityEngine;
using System;
using System.Linq;
using System.Runtime.CompilerServices;

[assembly: InternalsVisibleToAttribute("Barracuda.EditorTests")]

namespace Unity.Barracuda
{
    // ONNX specification mandates "channels first" layout of the tensors, while Barracuda uses "channels last" layout just like Tensorflow.
    // Moreover Barracuda uses "named dimensions" and expects particular dimension in specific position of the tensor.
    // The code below handles conversion between different layouts and mapping to particular "name".
    //
    // Tensor dimension names:
    //  N       - batch
    //  C       - channels
    //  H       - height
    //  W       - width
    //  K or M  - feature maps aka output channels
    //  ?       - unknown layout
    //
    // NOTE: "_" stands for dimension that is not present in the specific ONNX tensor. It will make respected dimension of size 1 ("empty") in Barracuda tensor.
    public class ONNXLayout
    {
        public static int[] AxisPermutationsForMappingONNXLayoutToBarracuda(int onnxRank, string onnxLayout="NCHW")
        {
            // R dimensions is currently unused and is coming from `sequence` dimension in recurrent networks
            // Input tensors:           NCHW -> NHWC, NCW -> N1WC, NC -> N11C, C -> 111C
            // Convolution kernels:     KCHW -> HWCK, KCW -> 1WCK
            // Transpose convolutions:  CKHW -> HWCK, CKW -> 1WCK
            // LSTM weights:            RCK  -> C11K
            // LSTM weights:            RKC  -> C11K
            // LSTM biases:             RC   -> 111C
            // GemmTransposeB, MatMul:  CK   -> C11K
            // Gemm weights             KC   -> C11K

            const int _ = -1;

            if (onnxRank == 0)
                return new[] {_, _, _, _};

            if (onnxRank > 4)
                throw new OnnxLayerImportException($"Only tensors of rank 4 or less are supported, but got rank {onnxRank}");

            else if (onnxLayout == "NCHW") // -> NHWC
                switch (onnxRank)
                {
                    case 4:
                        return new int[] {0, 2, 3, 1};
                    case 3:
                        return new int[] {0, _, 2, 1};
                    case 2:
                        return new int[] {0, _, _, 1};
                    case 1:
                        return new int[] {_, _, _, 0};
                }
            else if (onnxLayout == "CONST") // -> NHWC
                switch (onnxRank)
                {
                    case 4:
                        return new int[] {0, 2, 3, 1}; // assume NCHW
                    case 3:
                        return new int[] {_, 2, 1, 0}; // assume  CHW
                    case 2:
                        return new int[] {_, _, 1, 0}; // assume   CW
                    case 1:
                        return new int[] {_, _, _, 0}; // assume    C
                }
            else if (onnxLayout == "MCHW" || onnxLayout == "KCHW") // -> HWCK
                switch (onnxRank)
                {
                    case 4:
                        return new int[] {2, 3, 1, 0};
                    case 3:
                        return new int[] {_, 2, 1, 0};
                    default:
                        throw new OnnxLayerImportException($"MCHW layout requires kernel weight tensor of rank 3 or higher, but got {onnxRank}");
                }
            else if (onnxLayout == "CMHW" || onnxLayout == "CKHW") // -> HWCK
                switch (onnxRank)
                {
                    case 4:
                        return new int[] {2, 3, 0, 1};
                    case 3:
                        return new int[] {_, 2, 0, 1};
                    default:
                        throw new OnnxLayerImportException($"CMHW layout requires kernel weight tensor of rank 3 or higher, but got {onnxRank}");
                }
            else if (onnxLayout == "CHWM" || onnxLayout == "CHWK") // -> HWCK
                switch (onnxRank)
                {
                    case 4:
                        return new int[] {1, 2, 0, 3};
                    case 3:
                        return new int[] {_, 1, 0, 2};
                    default:
                        throw new OnnxLayerImportException($"CHWM layout requires kernel weight tensor of rank 3 or higher, but got {onnxRank}");
                }
            else if (onnxLayout == "CM" || onnxLayout == "CK" || onnxLayout == "RCK") // -> C__K
                switch (onnxRank)
                {
                    case 2:
                        return new int[] {0, _, _, 1};
                    case 3:
                        return new int[] {1, _, _, 2};
                    default:
                        throw new OnnxLayerImportException($"CM layout requires weight tensor of rank 2 or 3(LSTM), but got {onnxRank}");
                }
            else if (onnxLayout == "MC" || onnxLayout == "KC" || onnxLayout == "RKC") // -> C__K
                switch (onnxRank)
                {
                    case 2:
                        return new int[] {1, _, _, 0};
                    case 3:
                        return new int[] {2, _, _, 1};
                    default:
                        throw new OnnxLayerImportException($"MC layout requires weight tensor of rank 2 or 3(LSTM), but got {onnxRank}");
                }
            else if (onnxLayout == "RC") // -> ___C
                switch (onnxRank)
                {
                    case 2:
                        return new int[] {_, _, _, 1};
                    default:
                        throw new OnnxLayerImportException($"RC layout requires tensor of rank 2, but got {onnxRank}");
                }
            else if (onnxLayout == "C") // -> ___C
                switch (onnxRank)
                {
                    case 1:
                        return new int[] {_, _, _, 0};
                    default:
                        throw new OnnxLayerImportException($"C layout requires tensor of rank 1, but got {onnxRank}");
                }

            else if (onnxLayout == "?")
                switch (onnxRank)
                {
                    case 4:
                        return new int[] {0, 1, 2, 3};
                    case 3:
                        return new int[] {0, 1, 2, _};
                    case 2:
                        return new int[] {0, 1, _, _};
                    case 1:
                        return new int[] {0, _, _, _};
                }
            else
                throw new OnnxLayerImportException($"Unknown tensor layout {onnxLayout}");

            throw new OnnxLayerImportException($"Unsupported combination of tensor layout {onnxLayout} and tensor rank {onnxRank}");
        }

        public static int[] PermuteToBarracuda(long[] shape, string onnxLayout)
        {
            var onnxRank = shape.Length;
            var permutations = AxisPermutationsForMappingONNXLayoutToBarracuda(onnxRank, onnxLayout);
            Debug.Assert(shape.Length <= permutations.Length);
            Debug.Assert(shape.Length == permutations.Count(v => v >= 0));
            var output = new int[permutations.Length];
            for (var i = 0; i < permutations.Length; ++i)
                output[i] = permutations[i] >= 0 ? (int)shape[permutations[i]] : 1;
            return output;
        }

        public static int[] Permute(int[] shape, int[] permutations)
        {
            Debug.Assert(shape.Length <= permutations.Length);
            Debug.Assert(shape.Count(v => v > 1) <= permutations.Count(v => v >= 0));
            var output = new int[permutations.Length];
            for (var i = 0; i < permutations.Length; ++i)
                output[i] = permutations[i] >= 0 ? shape[permutations[i]] : 1;
            return output;
        }

        public static long[] Permute(long[] shape, int[] permutations)
        {
            Debug.Assert(shape.Length <= permutations.Length);
            Debug.Assert(shape.Count(v => v > 1) <= permutations.Count(v => v >= 0));
            var output = new long[permutations.Length];
            for (var i = 0; i < permutations.Length; ++i)
                output[i] = permutations[i] >= 0 ? shape[permutations[i]] : 1;
            return output;
        }

        public static int[] InversePermute(int[] permutations)
        {
            // {0, 2, 3, 1} => {0, 3, 1, 2}
            // {2, 3, 1, 0} => {3, 2, 0, 1}
            //              => {find_index(0), find_index(1), find_index(2), find_index(3)}
            var reversePermute = new int[permutations.Length];
            for (var i = 0; i < permutations.Length; ++i)
                reversePermute[i] = Array.IndexOf(permutations, i);
            return reversePermute;
        }

        public static int ConvertAxisToBarracuda(int axis, int onnxRank, string onnxLayout)
        {
            var permutations = AxisPermutationsForMappingONNXLayoutToBarracuda(onnxRank, onnxLayout);
            if (axis < 0)
                axis = onnxRank + axis;
            return Array.IndexOf(permutations, axis);
        }

        public static TensorShape ConvertShapeToBarracuda(long[] onnxShape, string onnxLayout)
        {
            var shape = ConvertSymbolicShapeToBarracuda(onnxShape, onnxLayout);
            if (shape.Any(s => s < 0))
                throw new OnnxLayerImportException($"Expected ONNX shape with all dimensions known, instead got {string.Join(", ",shape)}");
            return new TensorShape(shape);
        }

        public static int[] ConvertSymbolicShapeToBarracuda(TensorShapeProto shape, string onnxLayout)
        {
            // TODO: use dimension denotation from TensorShapeProto to figure, if this particular tensor has specific data layout
            // https://github.com/onnx/onnx/blob/master/docs/DimensionDenotation.md
            return ConvertSymbolicShapeToBarracuda(shape.Dim.Select(d => d.DimValue).ToArray(), onnxLayout);
        }

        public static int[] ConvertSymbolicShapeToBarracuda(long[] onnxShape, string onnxLayout)
        {
            var permutedShape = PermuteToBarracuda(onnxShape, onnxLayout);
            Debug.Assert(permutedShape.Length == 4);
            return Enumerable.Repeat(1, 4 - permutedShape.Length).Concat(permutedShape).ToArray();
        }

        internal static bool CanSymbolicShapeBeUsedWithReshapeLike(long[] onnxShape, int featureCount)
        {
            // If symbolic shape matches [-1, featureCount, -1, ... -1] OR [featureCount]
            // original tensor can be used as a source for ReshapeLike() layer

            var channelsDimension = (onnxShape.Length == 1) ? 0: 1; // C dimension in ONNX layout

            var expectedPattern = Enumerable.Repeat(-1L, onnxShape.Length).ToArray();
            expectedPattern[channelsDimension] = featureCount;

            return onnxShape.SequenceEqual(expectedPattern);
        }
    }
}
