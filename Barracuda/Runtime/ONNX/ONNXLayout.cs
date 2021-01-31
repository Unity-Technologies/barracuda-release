using Onnx;
using System;
using System.Linq;
using System.Runtime.CompilerServices;
using UnityEngine.Assertions;

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
    internal class ONNXLayout
    {
        public static int[] AxisPermutationsForMappingONNXLayoutToBarracuda(int onnxRank, string onnxLayout="NCHW")
        {
            // R dimensions is currently unused and is coming from `sequence` dimension in recurrent networks
            // 8D Input tensors:        NCTDHW -> SRNTDHWC, SRNCDHW -> SRN_DHWC, SRNC__HW -> SRN__HWC
            // 4D Input tensors:        NCHW -> __N__HWC, NCW -> __N___WC, NC -> __N____C, C -> _______C
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

            int maxRank = 6;
            if (onnxRank > maxRank)
                throw new OnnxLayerImportException($"Only tensors of rank {maxRank} or less are supported for layout {onnxLayout}, but got rank {onnxRank}");

            else if (onnxLayout == "NC0C1HW") // NC0C1HW -> __N_HWC0C1
                switch (onnxRank)
                {
                    case 5:
                        return new int[] { _, _, 0, _, 3, 4, 1, 2};
                    default:
                        throw new OnnxLayerImportException($"NC0C1HW layout requires weight tensor of rank 5, but got {onnxRank}");
                }
            else if (onnxLayout == "NC0C1C2HW") // NC0C1C2HW -> __NHWC0C1C2
                switch (onnxRank)
                {
                    case 6:
                        return new int[] { _, _, 0, 4, 5, 1, 2, 3};
                    default:
                        throw new OnnxLayerImportException($"NC0C1C2HW layout requires weight tensor of rank 6, but got {onnxRank}");
                }
            else if (onnxLayout == "NCTDHW" || onnxLayout == "NCHW") // NCTDHW -> __NTDHWC, NCHW -> __N__HWC
                switch (onnxRank)
                {
                    case 6:
                        return new int[] { _, _, 0, 2, 3, 4, 5, 1};
                    case 5:
                        return new int[] { _, _, 0, _, 2, 3, 4, 1};
                    case 4:
                        return new int[] { _, _, 0, _, _, 2, 3, 1};
                    case 3:
                        return new int[] { _, _, 0, _, _, _, 2, 1};
                    case 2:
                        return new int[] { _, _, 0, _, _, _, _, 1};
                    case 1:
                        return new int[] { _, _, 0, _, _, _, _, _};
                }
            else if (onnxLayout == "CONST") // -> __N__HWC
                switch (onnxRank)
                {
                    case 4:
                        return new int[] { _, _, 0, _, _, 2, 3, 1}; // assume NCHW
                    case 3:
                        return new int[] { _, _, _, _, _, 1, 2, 0}; // assume  CHW
                    case 2:
                        return new int[] { _, _, _, _, _, _, 1, 0}; // assume   CW
                    case 1:
                        return new int[] { _, _, _, _, _, _, _, 0}; // assume    C
                }
            else if (onnxLayout == "MCDHW" || onnxLayout == "MCHW" || onnxLayout == "KCHW") // -> __H__WCK
                switch (onnxRank)
                {
                    case 5:
                        return new int[] { _, 2, 3, _, _, 4, 1, 0};
                    case 4:
                        return new int[] { _, _, 2, _, _, 3, 1, 0};
                    case 3:
                        return new int[] { _, _, _, _, _, 2, 1, 0};
                    default:
                        throw new OnnxLayerImportException($"MCDHW layout requires kernel weight tensor of rank 3 or higher, but got {onnxRank}");
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
                        return new int[] {_ ,_ ,_ ,_ ,_ , _, _, 1};
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
            else if (onnxLayout == "ONNX") // Keep ONNX format
                switch (onnxRank)
                {
                    case 6:
                        return new int[] { _, _, 0, 1, 2, 3, 4, 5};
                    case 5:
                        return new int[] { _, _, 0, _, 1, 2, 3, 4};
                    case 4:
                        return new int[] { _, _, 0, _, _, 1, 2, 3};
                    case 3:
                        return new int[] { _, _, 0, _, _, 1, 2, _};
                    case 2:
                        return new int[] { _, _, 0, _, _, 1, _, _};
                    case 1:
                        return new int[] { _, _, 0, _, _, _, _, _};
                }
            else if (onnxLayout == "?")
                switch (onnxRank)
                {
                    case 8:
                        return new int[] {0, 1, 2, 3, 4, 5, 6, 7};
                    case 7:
                        return new int[] {0, 1, 2, 3, 4, 5, 6, _};
                    case 6:
                        return new int[] {0, 1, 2, 3, 4, 5, _, _};
                    case 5:
                        return new int[] {0, 1, 2, 3, 4, _, _, _};
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

        public static int[] PermuteToBarracuda(int[] shape, string onnxLayout, int defaultValue = 1)
        {
            var onnxRank = shape.Length;
            var permutations = AxisPermutationsForMappingONNXLayoutToBarracuda(onnxRank, onnxLayout);
            Assert.IsTrue(shape.Length <= permutations.Length);
            Assert.IsTrue(shape.Length == permutations.Count(v => v >= 0));
            var output = new int[permutations.Length];
            for (var i = 0; i < permutations.Length; ++i)
                output[i] = permutations[i] >= 0 ? (int)shape[permutations[i]] : defaultValue;
            return output;
        }

        public static int[] Permute(int[] shape, int[] permutations)
        {
            Assert.IsTrue(shape.Length <= permutations.Length);
            Assert.IsTrue(shape.Count(v => v > 1) <= permutations.Count(v => v >= 0));
            var output = new int[permutations.Length];
            for (var i = 0; i < permutations.Length; ++i)
                output[i] = permutations[i] >= 0 ? shape[permutations[i]] : 1;
            return output;
        }

        public static long[] Permute(long[] shape, int[] permutations)
        {
            Assert.IsTrue(shape.Length <= permutations.Length);
            Assert.IsTrue(shape.Count(v => v > 1) <= permutations.Count(v => v >= 0));
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

        private static int Adjust6DAxisForPaddingInChannelFirst(int axis, int padding)
        {
            //if `axis` is past channels rank, axis index need to be increased by the amount of padding
            //to is gonna be added between channels and other features.
            return (axis >= 2) ? axis + padding : axis;
        }

        public static int[] ExpandONNXPermutationToNCTDHW(int[] onnxPermutation, out int centerPadding)
        {
            var permutationsNCTDHW = new[] { 0, 1, 2, 3, 4, 5 };
            centerPadding = permutationsNCTDHW.Length - onnxPermutation.Length;
            if (onnxPermutation.Length > 0) permutationsNCTDHW[0] = Adjust6DAxisForPaddingInChannelFirst(onnxPermutation[0], centerPadding);//batch
            if (onnxPermutation.Length > 1) permutationsNCTDHW[1] = Adjust6DAxisForPaddingInChannelFirst(onnxPermutation[1], centerPadding);//channels
            for (int i = 2; i < onnxPermutation.Length; ++i)
                permutationsNCTDHW[i + centerPadding] = Adjust6DAxisForPaddingInChannelFirst(onnxPermutation[i], centerPadding);

            return permutationsNCTDHW;
        }

        public static int[] ConvertPermutationToLayout(int[] sourcePermutations, string sourceLayout, string targetLayout)
        {
            //Given a permutation in `sourceLayout` format, this function return the semantically equivalent permutation in `targetLayout`.
            //For example if `sourceLayout` is NCHW, `sourcePermutations` is 0132 (swapping H and W), and targetLayout is `NHWC`
            //it will return 0213 (swapping of H and W in NHWC layout).
            Assert.IsTrue(sourceLayout.Length == sourcePermutations.Length);
            Assert.IsTrue(sourceLayout.Length == targetLayout.Length);

            var targetPermutation = new int[sourcePermutations.Length];

            //For each target dimension
            for(int idTarget = 0; idTarget<targetPermutation.Length; ++idTarget)
            {
                //Find target semantic.
                char destinationSemantic = targetLayout[idTarget];
                //Find semantic index in `sourceLayout`
                int sourceDestinationSemanticIndex = sourceLayout.IndexOf(destinationSemantic);
                Assert.IsTrue(sourceDestinationSemanticIndex != -1);
                //Find permutation in `sourceLayout` space.
                int sourcePermutationSemanticIndex = sourcePermutations[sourceDestinationSemanticIndex];
                //Find permutation semantic
                char permutationSemantic = sourceLayout[sourcePermutationSemanticIndex];
                //Find permutation semantic index in `targetLayout`.
                int targetPermutationSemanticIndex = targetLayout.IndexOf(permutationSemantic);
                Assert.IsTrue(targetPermutationSemanticIndex != -1);
                //Done store it
                targetPermutation[idTarget] = targetPermutationSemanticIndex;
            }
            return targetPermutation;
        }

        public static TensorShape ConvertShapeToBarracuda(int[] onnxShape, string onnxLayout)
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
            var onnxShape = shape.AsInts();
            return ConvertSymbolicShapeToBarracuda(onnxShape, onnxLayout);
        }

        public static int[] ConvertReshapeToBarracuda(int[] onnxShape, int inputRank, out int numDimensionContainingChannelsInformationAfterReshape)
        {
            //sufflenet and super_resolution_cnn are splitting channels into two dimensions
            //care need to be taken as C is channelLast in Barracuda and channelFirst in ONNX:
            //An example from shufflenet:
            //ONNX    => NCHW 1,112,56,56 -> NC1C2HW 1,4,28,56,56 should map to
            //Barruda => NHWC 1,56,56,112 -> NHWC1C2 1,56,56,4,28 (and not 1,4,56,56,28)
            //Another example from sub_pixel_cnn
            //ONNX    => NCHW 1,9,224,224 -> NC1C2HW 1,3,3,224,224 should map to
            //Barruda => NHWC 1,224,224,9 -> NHWC1C2 1,3,3,224,224 (and not 1,3,224,224,3)
            //However we don't support multidimensional features. Thus Barracuda will instead have:
            //shufflenet    -> NTDHWC with C=45,W=4,H=56,D=56,T=1,N=1
            //sub_pixel_cnn -> NTDHWC with C=224,W=224,H=3,D=3,T=1,N=1
            //further more we need to keep this information for Transpose layer that follow in those architectures.
            //indeed convertion from transpose parameters in channelFirst vs channelLast is dependant of
            //the number of dimensions channels are represented by.
            var outputRank = onnxShape.Length;
            if (inputRank == 4 && outputRank == 5)
            {
                numDimensionContainingChannelsInformationAfterReshape = 2;
                return ConvertSymbolicShapeToBarracuda(onnxShape, "NC0C1HW");
            }
            if (inputRank == 4 && outputRank == 6)
            {
                numDimensionContainingChannelsInformationAfterReshape = 3;
                return ConvertSymbolicShapeToBarracuda(onnxShape, "NC0C1C2HW");
            }

            numDimensionContainingChannelsInformationAfterReshape = 1;
            return ConvertSymbolicShapeToBarracuda(onnxShape, "NCTDHW");
        }

        public static int[] ConvertSymbolicShapeToBarracuda(int[] onnxShape, string onnxLayout)
        {
            var permutedShape = PermuteToBarracuda(onnxShape, onnxLayout);
            Assert.IsTrue(permutedShape.Length == 8);
            return Enumerable.Repeat(1, 8 - permutedShape.Length).Concat(permutedShape).ToArray();
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
                // Barracuda: NHWC    C_WH    N_WH   N_WC   N_HC
                if (onnxAxis == 0)
                    return new[] { 3, 0, 2, 1 };
                else if (onnxAxis == 1)
                    return new[] { 0, 3, 2, 1 };
                else if (onnxAxis == 2)
                    return identity;
                else
                    return new[] { 0, 2, 1, 3 };
            }
            else if (onnxRank == 3)
            {
                //            axis:   0       1      2
                // ONNX:      NCH     CH      NH     NC
                // Barracuda: N_HC    C__H    N__H   N__C
                if (onnxAxis == 0)
                    return new[] { 3, 0, 1, 2 };
                else if (onnxAxis == 1)
                    return new[] { 0, 1, 3, 2 };
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
                // Barracuda: N_HC    1CHN    NCH1   N1HC   NH1C
                if (onnxAxis == 0)
                    return new[] { 1, 3, 2, 0 };
                else if (onnxAxis == 1)
                    return new[] { 0, 3, 2, 1 };
                else if (onnxAxis == 2)
                    return identity;
                else
                    return new[] { 0, 2, 1, 3 };
            }
            else if (onnxRank == 2)
            {
                //            axis:   0       1      2
                // ONNX:      NC      1NC     N1C    NC1
                // Barracuda: N__C    1_CN    N_C1   N_1C
                if (onnxAxis == 0)
                    return new[] { 1, 2, 3, 0 };
                else if (onnxAxis == 1)
                    return new[] { 0, 1, 3, 2 };
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
            else if (onnxRank == 0)
            {
                return identity;
            }
            else
            {
                throw new OnnxLayerImportException($"Unsqueeze leading to tensor of rank >= 4, Not supported");
            }
        }

    }
}
