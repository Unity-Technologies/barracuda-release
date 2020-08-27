using Onnx;
using UnityEngine;
using System;
using System.Linq;
using System.Runtime.CompilerServices;

[assembly: InternalsVisibleToAttribute("Barracuda.EditorTests")]
[assembly: InternalsVisibleToAttribute("Unity.Barracuda.Editor")]

namespace Unity.Barracuda.ONNX
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
            // Input tensors:           NCHW -> __N__HWC, NCW -> __N___WC, NC -> __N____C, C -> _______C
            // Convolution kernels:     KCHW -> __H__WCK, KCW -> __H__WCK
            // Transpose convolutions:  CKHW -> __H__WCK, CKW -> __H__WCK
            // LSTM weights:            RCK  -> __C____K
            // LSTM weights:            RKC  -> __C____K
            // LSTM biases:             RC   -> _______C
            // GemmTransposeB, MatMul:  CK   -> __C____K
            // Gemm weights             KC   -> __C____K

            const int _ = -1;

            if (onnxRank == 0)
                return new[] {_, _, _, _, _, _, _, _};

            if (onnxRank > 4)
                throw new OnnxLayerImportException($"Only tensors of rank 4 or less are supported, but got rank {onnxRank}");

            else if (onnxLayout == "NCHW") // -> __N__HWC
                switch (onnxRank)
                {
                    case 4:
                        return new int[] { _, _, 0, _, _, 2, 3, 1};
                    case 3:
                        return new int[] { _, _, 0, _, _, _, 2, 1};
                    case 2:
                        return new int[] { _, _, 0, _, _, _, _, 1};
                    case 1:
                        return new int[] { _, _, _, _, _, _, _, 0};
                }
            else if (onnxLayout == "CONST") // -> __N__HWC
                switch (onnxRank)
                {
                    case 4:
                        return new int[] { _, _, 0, _, _, 2, 3, 1}; // assume NCHW
                    case 3:
                        return new int[] { _, _, _, _, _, 2, 1, 0}; // assume  CHW
                    case 2:
                        return new int[] { _, _, _, _, _, _, 1, 0}; // assume   CW
                    case 1:
                        return new int[] { _, _, _, _, _, _, _, 0}; // assume    C
                }
            else if (onnxLayout == "MCHW" || onnxLayout == "KCHW") // -> __H__WCK
                switch (onnxRank)
                {
                    case 4:
                        return new int[] { _, _, 2, _, _, 3, 1, 0};
                    case 3:
                        return new int[] { _, _, _, _, _, 2, 1, 0};
                    default:
                        throw new OnnxLayerImportException($"MCHW layout requires kernel weight tensor of rank 3 or higher, but got {onnxRank}");
                }
            else if (onnxLayout == "CMHW" || onnxLayout == "CKHW") // -> __H__WCK
                switch (onnxRank)
                {
                    case 4:
                        return new int[] { _, _, 2, _, _, 3, 0, 1};
                    case 3:
                        return new int[] { _, _, _, _, _, 2, 0, 1};
                    default:
                        throw new OnnxLayerImportException($"CMHW layout requires kernel weight tensor of rank 3 or higher, but got {onnxRank}");
                }
            else if (onnxLayout == "CHWM" || onnxLayout == "CHWK") // -> __H__WCK
                switch (onnxRank)
                {
                    case 4:
                        return new int[] { _, _, 1, _, _, 2, 0, 3};
                    case 3:
                        return new int[] { _, _, _, _, _, 1, 0, 2};
                    default:
                        throw new OnnxLayerImportException($"CHWM layout requires kernel weight tensor of rank 3 or higher, but got {onnxRank}");
                }
            else if (onnxLayout == "CM" || onnxLayout == "CK" || onnxLayout == "RCK") // -> __C____K
                switch (onnxRank)
                {
                    case 2:
                        return new int[] { _, _, 0, _, _, _, _, 1};
                    case 3:
                        return new int[] { _, _, 1, _, _, _, _, 2};
                    default:
                        throw new OnnxLayerImportException($"CM layout requires weight tensor of rank 2 or 3(LSTM), but got {onnxRank}");
                }
            else if (onnxLayout == "MC" || onnxLayout == "KC" || onnxLayout == "RKC") // -> __C____K
                switch (onnxRank)
                {
                    case 2:
                        return new int[] { _, _, 1, _, _, _, _, 0};
                    case 3:
                        return new int[] { _, _, 2, _, _, _, _, 1};
                    default:
                        throw new OnnxLayerImportException($"MC layout requires weight tensor of rank 2 or 3(LSTM), but got {onnxRank}");
                }
            else if (onnxLayout == "RC") // -> _______C
                switch (onnxRank)
                {
                    case 2:
                        return new int[] {_ ,_ ,_ ,_ ,_ , _, _, 1};;
                    default:
                        throw new OnnxLayerImportException($"RC layout requires tensor of rank 2, but got {onnxRank}");
                }
            else if (onnxLayout == "C") // -> _______C
                switch (onnxRank)
                {
                    case 1:
                        return new int[] {_ ,_ ,_ ,_ ,_ , _, _, 0};
                    default:
                        throw new OnnxLayerImportException($"C layout requires tensor of rank 1, but got {onnxRank}");
                }

            else if (onnxLayout == "?")
                switch (onnxRank)
                {
                    case 4:
                        return new int[] {0, 1, 2, 3, _, _, _, _};
                    case 3:
                        return new int[] {0, 1, 2, _, _, _, _, _};
                    case 2:
                        return new int[] {0, 1, _, _, _, _, _, _};
                    case 1:
                        return new int[] {0, _, _, _, _, _, _, _};
                }
            else
                throw new OnnxLayerImportException($"Unknown tensor layout {onnxLayout}");

            throw new OnnxLayerImportException($"Unsupported combination of tensor layout {onnxLayout} and tensor rank {onnxRank}");
        }

        public static int[] PermuteToBarracuda(long[] shape, string onnxLayout, int defaultValue = 1)
        {
            var onnxRank = shape.Length;
            var permutations = AxisPermutationsForMappingONNXLayoutToBarracuda(onnxRank, onnxLayout);
            Debug.Assert(shape.Length <= permutations.Length);
            Debug.Assert(shape.Length == permutations.Count(v => v >= 0));
            var output = new int[permutations.Length];
            for (var i = 0; i < permutations.Length; ++i)
                output[i] = permutations[i] >= 0 ? (int)shape[permutations[i]] : defaultValue;
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
            Debug.Assert(permutedShape.Length == 8);
            return Enumerable.Repeat(1, 8 - permutedShape.Length).Concat(permutedShape).ToArray();
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

        public static int[] SqueezeAxisPermutationForMappingONNXLayoutToBarracuda(int onnxRank, int onnxAxis, string onnxLayout = "NCHW")
        {
            if (onnxRank > 4)
                throw new OnnxLayerImportException($"Only tensors of rank 4 or less are supported, but got rank {onnxRank}");

            if (onnxLayout != "NCHW")
                throw new OnnxLayerImportException($"Only NCHW tensor layout supported {onnxLayout}");

            var identity = new[] { 0, 1, 2, 3 };

            if (onnxRank == 4)
            {
                //            axis:   0       1      2      3
                // ONNX:      NCHW    CHW     NHW    NCW    NCH   
                // Barracuda: NHWC    CW_H    NW_H   NW_C   NH_C
                if (onnxAxis == 0)
                    return new[] { 3, 2, 0, 1 };
                else if (onnxAxis == 1)
                    return new[] { 0, 2, 3, 1 };
                else if (onnxAxis == 2)
                    return new[] { 0, 2, 1, 3 };
                else
                    return identity;
            }
            if (onnxRank == 3)
            {
                //            axis:   0       1      2
                // ONNX:      NCH     CH      NH     NC
                // Barracuda: NH_C    C__H    N__H   N__C
                if (onnxAxis == 0)
                    return new[] { 3, 0, 2, 1 };
                else if (onnxAxis == 1)
                    return new[] { 0, 2, 3, 1 };
                else
                    return identity;
            }
            else if (onnxRank == 2)
            {
                //            axis:   0       1
                // ONNX:      NC      C       N
                // Barracuda: N__C    C___    N___
                if (onnxAxis == 0)
                    return new[] { 3, 0, 1, 2 };
                else
                    return identity;
            }
            else
            {
                return identity;
            }
        }

        public static int[] UnSqueezeAxisPermutationForMappingONNXLayoutToBarracuda(int onnxRank, int onnxAxis, string onnxLayout = "NCHW")
        {
            if (onnxRank > 4)
                throw new OnnxLayerImportException($"Only tensors of rank 4 or less are supported, but got rank {onnxRank}");

            if (onnxLayout != "NCHW")
                throw new OnnxLayerImportException($"Only NCHW tensor layout supported {onnxLayout}");

            var identity = new[] { 0, 1, 2, 3 };

            if (onnxRank == 3)
            {
                //            axis:   0       1      2      3
                // ONNX:      NCH     1NCH    N1CH   NC1H   NCH1
                // Barracuda: NH_C    1CHN    NCH1   N1HC   NH1C
                if (onnxAxis == 0)
                    return new[] { 2, 3, 1, 0 };
                else if (onnxAxis == 1)
                    return new[] { 0, 3, 1, 2 };
                else if (onnxAxis == 2)
                    return new[] { 0, 2, 1, 3 };
                else
                    return identity;
            }
            else if (onnxRank == 2)
            {
                //            axis:   0       1      2   
                // ONNX:      NC      1NC     N1C    NC1
                // Barracuda: N__C    1C_N    NC_1   N1_C
                if (onnxAxis == 0)
                    return new[] { 1, 3, 2, 0 };
                else if (onnxAxis == 1)
                    return new[] { 0, 3, 1, 2 };
                else
                    return identity;
            }
            else if (onnxRank == 1)
            {
                //            axis:   0       1   
                // ONNX:      N       1N      N1 
                // Barracuda: N___    1__N    N__1
                if (onnxAxis == 0)
                    return new[] { 1, 2, 3, 0 };
                else
                    return identity;
            }
            else 
            {
                throw new OnnxLayerImportException($"Unsqeeze leading to tensor of rank >= 4, Not supported");
            }
        }

    }
}
