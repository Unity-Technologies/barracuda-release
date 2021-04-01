using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Profiling;

namespace Unity.Barracuda
{

    internal enum ComputeShaderContext
    {
        Reference,
        Optimized
    }

    /// <summary>
    /// Stores compute kernel cache for GPU compute backends
    /// </summary>
    public sealed class ComputeShaderSingleton
    {
        /// <summary>
        /// Enable kernel usage tracking
        /// </summary>
        public bool EnableDebug = false;

        private static readonly ComputeShaderSingleton instance = new ComputeShaderSingleton ();

        // Maps kernel name -> shader name
        private Dictionary<string, string> mKernelToShaderName = new Dictionary<string, string>();

        // Maps shader name -> ComputeShader
        private Dictionary<string, ComputeShader> mShaderNameToComputeShader = new Dictionary<string, ComputeShader>();

        private HashSet<string> mUsedOptimizedKernels = new HashSet<string>();
        private HashSet<string> mUsedReferenceKernels = new HashSet<string>();

        private ComputeShaderSingleton()
        {
            RegisterKernels("Barracuda/TextureUtils",
                new[] {"TextureToTensor", "TensorToTextureNoLUT", "TensorToTexture3DLUT"});

            RegisterKernels("Barracuda/ActivationA",
                new[]
                {
                    "Relu_Flat", "Relu_FlatStrict", "Relu_Loop", "Relu6_Flat", "Relu6_FlatStrict", "Relu6_Loop",
                    "Tanh_Flat", "Tanh_FlatStrict", "Tanh_Loop", "Swish_Flat", "Swish_FlatStrict", "Swish_Loop",
                    "Sigmoid_Flat", "Sigmoid_FlatStrict", "Sigmoid_Loop", "LeakyRelu_Flat", "LeakyRelu_FlatStrict",
                    "LeakyRelu_Loop", "Clip_Flat", "Clip_FlatStrict", "Clip_Loop", "PRelu_Flat", "PRelu_Loop"
                });

            RegisterKernels("Barracuda/ActivationB",
                new[]
                {
                    "Reciprocal_Flat", "Reciprocal_FlatStrict", "Reciprocal_Loop", "Sqrt_Flat", "Sqrt_FlatStrict",
                    "Sqrt_Loop"
                });

            RegisterKernels("Barracuda/ActivationBase",
                new string[]
                {
                    "Abs_Flat", "Abs_FlatStrict", "Abs_Loop", "Neg_Flat", "Neg_FlatStrict", "Neg_Loop", "Ceil_Flat",
                    "Ceil_FlatStrict", "Ceil_Loop", "Floor_Flat", "Floor_FlatStrict", "Floor_Loop",
                    "Round_Flat", "Round_FlatStrict", "Round_Loop", "Selu_Flat",
                    "Selu_FlatStrict", "Selu_Loop", "Softplus_Flat", "Softplus_FlatStrict", "Softplus_Loop", "Elu_Flat",
                    "Elu_FlatStrict", "Elu_Loop", "Exp_Flat", "Exp_FlatStrict", "Exp_Loop", "Log_Flat",
                    "Log_FlatStrict", "Log_Loop", "Pow_Flat", "Pow_FlatStrict", "Pow_Loop", "LogicalNot_Flat",
                    "LogicalNot_FlatStrict", "LogicalNot_Loop",  "Sign_Flat", "Sign_FlatStrict", "Sign_Loop",
                    "Acos_Flat", "Acos_FlatStrict", "Acos_Loop",
                    "Acosh_Flat", "Acosh_FlatStrict", "Acosh_Loop", "Asin_Flat", "Asin_FlatStrict", "Asin_Loop",
                    "Asinh_Flat", "Asinh_FlatStrict", "Asinh_Loop", "Atan_Flat", "Atan_FlatStrict", "Atan_Loop",
                    "Atanh_Flat", "Atanh_FlatStrict", "Atanh_Loop", "Cos_Flat", "Cos_FlatStrict", "Cos_Loop",
                    "Cosh_Flat", "Cosh_FlatStrict", "Cosh_Loop", "Sin_Flat", "Sin_FlatStrict", "Sin_Loop", "Sinh_Flat",
                    "Sinh_FlatStrict", "Sinh_Loop", "Tan_Flat", "Tan_FlatStrict", "Tan_Loop", "Relu_NHWC", "Relu_NCHW",
                    "Relu_CNyx_NHWC", "Relu_Nyxc_NHWC", "Relu6_NHWC", "Relu6_NCHW", "Relu6_CNyx_NHWC",
                    "Relu6_Nyxc_NHWC", "PRelu_NHWC", "PRelu_NCHW", "PRelu_CNyx2_NHWC", "Selu_NHWC", "Selu_NCHW",
                    "Selu_CNyx_NHWC", "Selu_Nyxc_NHWC", "Tanh_NHWC", "Tanh_NCHW", "Tanh_CNyx_NHWC", "Tanh_Nyxc_NHWC",
                    "Swish_NHWC", "Swish_NCHW", "Swish_CNyx_NHWC", "Swish_Nyxc_NHWC", "Softplus_NHWC", "Softplus_NCHW",
                    "Softplus_CNyx_NHWC", "Softplus_Nyxc_NHWC", "Sigmoid_NHWC", "Sigmoid_NCHW", "Sigmoid_CNyx_NHWC",
                    "Sigmoid_Nyxc_NHWC", "Elu_NHWC", "Elu_NCHW", "Elu_CNyx_NHWC", "Elu_Nyxc_NHWC", "LeakyRelu_NHWC",
                    "LeakyRelu_NCHW", "LeakyRelu_CNyx_NHWC", "LeakyRelu_Nyxc_NHWC", "Exp_NHWC", "Exp_NCHW",
                    "Exp_CNyx_NHWC", "Exp_Nyxc_NHWC", "Log_NHWC", "Log_NCHW", "Log_CNyx_NHWC", "Log_Nyxc_NHWC",
                    "Sqrt_NHWC", "Sqrt_NCHW", "Sqrt_CNyx_NHWC", "Sqrt_Nyxc_NHWC", "Pow_NHWC", "Pow_NCHW",
                    "Pow_CNyx_NHWC", "Pow_Nyxc_NHWC", "Softmax_NHWC", "Softmax_NCHW", "LogSoftmax_NHWC",
                    "LogSoftmax_NCHW", "Clip_NHWC", "Clip_NCHW", "Clip_CNyx_NHWC", "Clip_Nyxc_NHWC", "Acos_NHWC",
                    "Acos_NCHW", "Acos_CNyx_NHWC", "Acos_Nyxc_NHWC", "Acosh_NHWC", "Acosh_NCHW", "Acosh_CNyx_NHWC",
                    "Acosh_Nyxc_NHWC", "Asin_NHWC", "Asin_NCHW", "Asin_CNyx_NHWC", "Asin_Nyxc_NHWC", "Asinh_NHWC",
                    "Asinh_NCHW", "Asinh_CNyx_NHWC", "Asinh_Nyxc_NHWC", "Atan_NHWC", "Atan_NCHW", "Atan_CNyx_NHWC",
                    "Atan_Nyxc_NHWC", "Atanh_NHWC", "Atanh_NCHW", "Atanh_CNyx_NHWC", "Atanh_Nyxc_NHWC", "Cos_NHWC",
                    "Cos_NCHW", "Cos_CNyx_NHWC", "Cos_Nyxc_NHWC", "Cosh_NHWC", "Cosh_NCHW", "Cosh_CNyx_NHWC",
                    "Cosh_Nyxc_NHWC", "Sin_NHWC", "Sin_NCHW", "Sin_CNyx_NHWC", "Sin_Nyxc_NHWC", "Sinh_NHWC",
                    "Sinh_NCHW", "Sinh_CNyx_NHWC", "Sinh_Nyxc_NHWC", "Tan_NHWC", "Tan_NCHW", "Tan_CNyx_NHWC",
                    "Tan_Nyxc_NHWC"
                });

            RegisterKernels("Barracuda/Broadcast_NHWC",
                new[]
                {
                    "BroadcastAdd_NHWC", "BroadcastSub_NHWC", "BroadcastMul_NHWC", "BroadcastDiv_NHWC",
                    "BroadcastPow_NHWC", "BroadcastMin_NHWC", "BroadcastMax_NHWC", "BroadcastMean_NHWC",
                    "BroadcastGreater_NHWC", "BroadcastGreaterEqual_NHWC", "BroadcastLess_NHWC",
                    "BroadcastLessEqual_NHWC", "BroadcastEqual_NHWC", "BroadcastLogicalOr_NHWC",
                    "BroadcastLogicalAnd_NHWC", "BroadcastLogicalXor_NHWC", "BroadcastWhere_NHWC"
                });

            RegisterKernels("Barracuda/Broadcast_NCHW",
                new[]
                {
                    "BroadcastAdd_NCHW", "BroadcastSub_NCHW", "BroadcastMul_NCHW", "BroadcastDiv_NCHW",
                    "BroadcastPow_NCHW", "BroadcastMin_NCHW", "BroadcastMax_NCHW", "BroadcastMean_NCHW",
                    "BroadcastGreater_NCHW", "BroadcastGreaterEqual_NCHW", "BroadcastLess_NCHW",
                    "BroadcastLessEqual_NCHW", "BroadcastEqual_NCHW", "BroadcastLogicalOr_NCHW",
                    "BroadcastLogicalAnd_NCHW", "BroadcastLogicalXor_NCHW", "BroadcastWhere_NCHW"
                });

            RegisterKernels("Barracuda/Conv2dA_NHWC",
                new[]
                {
                    "Conv2D_NHWC", "Conv2D_RegisterBlock4x2_NHWC", "DepthwiseConv2D_NHWC",
                    "Conv2DKernelKxK_StrictC16K64_T16x16_R4x4_NHWC", "Conv2DKernelKxK_T16x16_R4x4_NHWC",
                    "Conv2DKernel1x1_StrictC16K64_T16x16_R4x4_NHWC"
                });

            RegisterKernels("Barracuda/Conv2dA_NCHW",
                new[]
                {
                    "Conv2D_NCHW", "Conv2D_RegisterBlock4x2_NCHW", "DepthwiseConv2D_NCHW",
                    "Conv2DKernelKxK_StrictC16K64_T16x16_R4x4_NCHW", "Conv2DKernelKxK_T16x16_R4x4_NCHW",
                    "Conv2DKernel1x1_StrictC16K64_T16x16_R4x4_NCHW"
                });

            RegisterKernels("Barracuda/Conv2dBase",
                new[]
                {
                    "Conv2DKernelKxK_StrictC16StrictK64_T8x8_R8x8_NHWC",
                    "Conv2DKernelKxK_StrictC16StrictK64_T8x8_R8x8_NCHW",
                    "Conv2DKernelKxK_StrictC16LaxK64_T8x8_R8x8_NHWC", "Conv2DKernelKxK_StrictC16LaxK64_T8x8_R8x8_NCHW",
                    "Conv2DKernelKxK_StrictC4StrictK16_T2x32_R8x8_NHWC",
                    "Conv2DKernelKxK_StrictC4StrictK16_T2x32_R8x8_NCHW",
                    "Conv2DKernelKxK_LaxC4StrictK16_T2x32_R8x8_NHWC", "Conv2DKernelKxK_LaxC4StrictK16_T2x32_R8x8_NCHW",
                    "Conv2DKernelKxK_StrictC4LaxK16_T2x32_R8x8_NHWC", "Conv2DKernelKxK_StrictC4LaxK16_T2x32_R8x8_NCHW",
                    "Conv2DTrans_NHWC", "Conv2DTrans_NCHW", "Conv2DTrans_KernelCached_K5x5_T16x16_NHWC",
                    "Conv2DTrans_KernelCached_K5x5_T16x16_NCHW", "Conv2DTransFlipKernel", "Conv2DTransPadFill_NHWC",
                    "Conv2DTransPadFill_NCHW", "KernelWinograd_3x3",
                    "Conv2DWinograd_2x2_Kernel3x3_StrictC8StrictK16_T16x16_R4x4_NCHW",
                    "Conv2DWinograd_2x2_Kernel3x3_StrictC8LaxK16_T16x16_R4x4_NCHW"
                });

            RegisterKernels("Barracuda/Conv3d",
                new[]
                {
                    "Conv3D_NHWC", "Conv3D_NCHW", "Conv3DKernelKxK_LaxC8LaxK32_T8x16_R4x4_NHWC",
                    "Conv3DKernelKxK_LaxC8LaxK32_T8x16_R4x4_NCHW", "Conv3DKernelKxK_StrictC8LaxK32_T8x16_R4x4_NHWC",
                    "Conv3DKernelKxK_StrictC8LaxK32_T8x16_R4x4_NCHW",
                    "Conv3DKernelKxK_StrictC8StrictK32_T8x16_R4x4_NHWC",
                    "Conv3DKernelKxK_StrictC8StrictK32_T8x16_R4x4_NCHW"
                });

            RegisterKernels("Barracuda/Dense",
                new[]
                {
                    "Dense_L1Cached64", "DenseTiled16x16", "DenseTiled32x32", "DenseTiled64x64", "Dense_T8x8_R4x4",
                    "Dense_T16x16_R4x4", "Dense_Tilled2x2_Cached", "Dense_Tilled4x4_Cached", "MatMulPackB0Bias",
                    "Dense_V_L1Cached64"
                });

            RegisterKernels("Barracuda/MatMul",
                new[]
                {
                    "MultidimMatMul_T16x16_R4x4_AR3_BR2_NHWC", "MultidimMatMul_T16x16_R4x4_AR3_BR2_NCHW",
                    "MultidimMatMul_T8x8_R8x8_AR3_BR2_NHWC", "MultidimMatMul_T8x8_R8x8_AR3_BR2_NCHW",
                    "MultidimMatMul_L1Cached64_AR3_BR2_NHWC", "MultidimMatMul_L1Cached64_AR3_BR2_NCHW"
                });

            RegisterKernels("Barracuda/Dense3",
                new[]
                {
                    "Dense3_T8x8_R8x8_NHWC", "Dense3_T8x8_R8x8_NCHW",
                    "Dense3_T8x16_R4x4_NHWC", "Dense3_T8x16_R4x4_NCHW",
                    "Dense3_L1Cached64_NHWC", "Dense3_L1Cached64_NCHW"
                });

            RegisterKernels("Barracuda/Generic",
                new[]
                {
                    "ScaleBias_NHWC", "ScaleBias_NCHW", "ScaleBias_CNyx_NHWC", "ScaleBias_CNyx2_NHWC",
                    "ScaleBias_Flat_NHWC", "ScaleBias_Flat_NCHW", "ScaleBias_Loop_NHWC", "ScaleBias_Loop_NCHW",
                    "InstanceNormTail_CNyx2_NHWC", "InstanceNormTail_Flat_NHWC", "InstanceNormTail_Flat_NCHW",
                    "InstanceNormTail_Loop_NHWC", "InstanceNormTail_Loop_NCHW", "Upsample2D_NHWC", "Upsample2D_NCHW",
                    "UpsampleBilinear2D_NHWC", "UpsampleBilinear2D_NCHW", "UpsampleBilinear2D_2x2_NHWC",
                    "UpsampleBilinear2D_2x2_NCHW", "Copy_NHWC", "Copy_NCHW", "ReshapeFromNHWCModel_Flat_NCHW",
                    "ReshapeFromNHWCModel_Loop_NCHW", "TransposeToChannelFirst"
                });

            RegisterKernels("Barracuda/Pad",
                new[]
                {
                    "Border2D_NHWC", "Border2D_NCHW", "Pad2DEdge_NHWC", "Pad2DEdge_NCHW", "Pad2DReflect_NHWC",
                    "Pad2DReflect_NCHW", "Pad2DSymmetric_NHWC", "Pad2DSymmetric_NCHW"
                });

            RegisterKernels("Barracuda/Transpose",
                new[]
                {
                    "Transpose2D_NHWC","Transpose2D_NCHW","Transpose_NHWC","Transpose_NCHW","Transpose8D"
                });

            RegisterKernels("Barracuda/Pool_NHWC",
                new[]
                {
                    "AvgPool2D_NHWC", "MaxPool2D_NHWC", "AvgPool2DReduce_NHWC", "MaxPool2DReduce_NHWC",
                    "GlobalAvgPool2D_NHWC", "GlobalMaxPool2D_NHWC", "AvgVariancePool2DReduce_NHWC",
                    "GlobalAvgVariancePool2D_NHWC"
                });

            RegisterKernels("Barracuda/Pool_NCHW",
                new[]
                {
                    "AvgPool2D_NCHW", "MaxPool2D_NCHW", "AvgPool2DReduce_NCHW", "MaxPool2DReduce_NCHW",
                    "GlobalAvgPool2D_NCHW", "GlobalMaxPool2D_NCHW", "AvgVariancePool2DReduce_NCHW",
                    "GlobalAvgVariancePool2D_NCHW"
                });

            RegisterKernels("Barracuda/Reduce",
                new[]
                {
                    "PartialReduceMin", "PartialReduceMin_Loop",
                    "GlobalReduceMin", "GlobalReduceMin_Loop",

                    "PartialReduceMax", "PartialReduceMax_Loop",
                    "GlobalReduceMax", "GlobalReduceMax_Loop",

                    "PartialReduceSum", "PartialReduceSum_Loop",
                    "GlobalReduceSum", "GlobalReduceSum_Loop",

                    "PartialReduceMean", "PartialReduceMean_Loop",
                    "GlobalReduceMean", "GlobalReduceMean_Loop",

                    "PartialReduceProd", "PartialReduceProd_Loop",
                    "GlobalReduceProd", "GlobalReduceProd_Loop"
                });
            RegisterKernels("Barracuda/ReduceSlow",
                new[]
                {
                     "ArgMax_NHWC", "ArgMax_NCHW", "ArgMin_NHWC", "ArgMin_NCHW"
                });
        }

        private void RegisterKernels(string shaderName, string[] kernels)
        {
            foreach (var kernel in kernels)
            {
                mKernelToShaderName[kernel] = shaderName;
            }
        }

        internal ComputeShader FindComputeShader(ComputeShaderContext ctx, string kernelName)
        {
            if (ctx == ComputeShaderContext.Optimized)
                return FindOptimizedComputeShader(kernelName);

            return FindReferenceComputeShader(kernelName);
        }

        private ComputeShader FindReferenceComputeShader(string kernelName)
        {
            if (EnableDebug) mUsedReferenceKernels.Add(kernelName);

            return FindComputeShader("Barracuda/BarracudaReferenceImpl");
        }

        private ComputeShader FindOptimizedComputeShader(string kernelName)
        {
            string shaderName = null;
            mKernelToShaderName.TryGetValue(kernelName, out shaderName);

            // Kernel not found
            if (shaderName == null)
                return null;

            if (EnableDebug) mUsedOptimizedKernels.Add(kernelName);

            return FindComputeShader(shaderName);
        }

        private ComputeShader FindComputeShader(string shaderName)
        {
            if (!mShaderNameToComputeShader.ContainsKey(shaderName))
            {
                Profiler.BeginSample(shaderName);
                mShaderNameToComputeShader[shaderName] = Resources.Load<ComputeShader>(shaderName);
                Profiler.EndSample();
            }

            return mShaderNameToComputeShader[shaderName];
        }

        /// <summary>
        /// Warmup reference kernels
        /// </summary>
        /// <param name="kernels">list of kernels to warm up</param>
        /// <returns>IEnumerator</returns>
        public IEnumerator WarmupReferenceKernels(List<string> kernels)
        {
            if (kernels?.Count > 0)
                FindComputeShader("Barracuda/BarracudaReferenceImpl");

            yield break;
        }

        /// <summary>
        /// Warmup optimized kernels
        /// </summary>
        /// <param name="kernels">list of kernels to warm up</param>
        /// <returns>IEnumerator</returns>
        public IEnumerator WarmupOptimizedKernels(List<string> kernels)
        {
            foreach (var kernel in kernels)
            {
                var shader = mKernelToShaderName[kernel];
                if (!mShaderNameToComputeShader.ContainsKey(shader))
                {
                    FindComputeShader(shader);
                    yield return null;
                }
            }
            yield break;
        }

        /// <summary>
        /// Get used reference kernels list
        /// </summary>
        /// <returns>list of kernels</returns>
        public List<string> GetUsedReferenceKernels()
        {
            if (!EnableDebug)
            {
                D.LogWarning("List of used kernels was requested while ComputeShaderSingleton.EnableDebug == false");
                return null;
            }

            return mUsedReferenceKernels.ToList();
        }

        /// <summary>
        /// Get used optimized kernels list
        /// </summary>
        /// <returns>list of kernels</returns>
        public List<string> GetUsedOptimizedKernels()
        {
            if (!EnableDebug)
            {
                D.LogWarning("List of used kernels was requested while ComputeShaderSingleton.EnableDebug == false");
                return null;
            }

            return mUsedOptimizedKernels.ToList();
        }

        /// <summary>
        /// Singleton
        /// </summary>
        public static ComputeShaderSingleton Instance {
            get { return instance; }
        }

        /// <summary>
        /// Check if GPU compute is supported
        /// </summary>
        public bool supported { get { return SystemInfo.supportsComputeShaders; } }
    }
}
