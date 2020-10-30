using System.Collections.Generic;
using UnityEngine;

namespace Unity.Barracuda
{
    /// <summary>
    /// Stores compute kernel cache for GPU compute backends
    /// </summary>
    public sealed class ComputeShaderSingleton
    {
        /// <summary>
        /// Reference compute kernels
        /// </summary>
        public readonly ComputeShader referenceKernels;

        /// <summary>
        /// Optimized kernels
        /// </summary>
        public readonly ComputeShader[] kernels;

        private static readonly ComputeShaderSingleton instance = new ComputeShaderSingleton ();

        private ComputeShaderSingleton ()
        {
            referenceKernels = LoadIf(ComputeInfo.supportsCompute, "BarracudaReferenceImpl");

            List<ComputeShader> kernelsList = new List<ComputeShader>();

            LoadIf(ComputeInfo.supportsCompute, "Generic", kernelsList);
            LoadIf(ComputeInfo.supportsCompute, "Activation", kernelsList);
            LoadIf(ComputeInfo.supportsCompute, "Broadcast", kernelsList);
            LoadIf(ComputeInfo.supportsCompute, "Pool", kernelsList);
            LoadIf(ComputeInfo.supportsCompute, "Pad", kernelsList);
            LoadIf(ComputeInfo.supportsCompute, "Dense", kernelsList);
            LoadIf(ComputeInfo.supportsCompute, "DenseFP16", kernelsList);
            LoadIf(ComputeInfo.supportsCompute, "Conv", kernelsList);

            kernels = kernelsList.ToArray();
        }

        /// <summary>
        /// Singleton
        /// </summary>
        public static ComputeShaderSingleton Instance {
            get { return instance; }
        }

        /// <summary>
        /// Load kernel if `condition` is met
        /// </summary>
        /// <param name="condition">condition to check</param>
        /// <param name="fileName">file name to load from</param>
        /// <returns>`ComputeShader`</returns>
        public static ComputeShader LoadIf(bool condition, string fileName)
        {
            if (condition)
                return (ComputeShader)Resources.Load(fileName);

            return null;
        }

        /// <summary>
        /// Load kernels if `condition` is met
        /// </summary>
        /// <param name="condition">condition to check</param>
        /// <param name="fileName">file name to load from</param>
        /// <param name="list">list to store loaded `ComputeShader` items</param>
        public static void LoadIf(bool condition, string fileName, List<ComputeShader> list)
        {
            ComputeShader shader = LoadIf(condition, fileName);

            if (shader)
                list.Add(shader);
        }

        /// <summary>
        /// Check if GPU compute is supported
        /// </summary>
        public bool supported { get { return SystemInfo.supportsComputeShaders; } }
    }
}
