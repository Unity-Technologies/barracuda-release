using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Profiling;

namespace Unity.Barracuda
{
    /// <summary>
    /// Stores compute kernel cache for GPU compute backends
    /// </summary>
    public sealed class ComputeShaderSingleton
    {


        private ComputeShader _referenceKernels;
        private ComputeShader _textureKernels;
        private ComputeShader[] _kernels;

        /// <summary>
        /// Reference compute kernels
        /// </summary>
        public ComputeShader referenceKernels
        {
            get
            {
                return  _referenceKernels ?? (_referenceKernels = LoadReferenceKernels());
            }
        }

        /// <summary>
        /// Optimized kernels
        /// </summary>
        public ComputeShader[] kernels
        {
            get
            {
                return  _kernels ?? (_kernels = LoadKernels());
            }
        }

        /// <summary>
        /// Texture kernels
        /// </summary>
        public ComputeShader texureKernels
        {
            get
            {
                return  _textureKernels ?? (_textureKernels = LoadTextureKernels());
            }
        }

        private static readonly ComputeShaderSingleton instance = new ComputeShaderSingleton ();

        private ComputeShaderSingleton ()
        {
        }

        private ComputeShader LoadReferenceKernels()
        {
            Profiler.BeginSample("Barracuda.LoadReferenceKernels");
            var res = LoadIf(ComputeInfo.supportsCompute, "Barracuda/BarracudaReferenceImpl");
            Profiler.EndSample();
            return res;
        }

        private ComputeShader LoadTextureKernels()
        {
            Profiler.BeginSample("Barracuda.LoadTextureKernels");
            var res = LoadIf(ComputeInfo.supportsCompute, "Barracuda/TextureUtils");
            Profiler.EndSample();
            return res;
        }
        private ComputeShader[] LoadKernels()
        {
            Profiler.BeginSample("Barracuda.LoadOptimizedKernels");
            List<ComputeShader> kernelsList = new List<ComputeShader>();

            LoadIf(ComputeInfo.supportsCompute, "Barracuda/Generic", kernelsList);
            LoadIf(ComputeInfo.supportsCompute, "Barracuda/Activation", kernelsList);
            LoadIf(ComputeInfo.supportsCompute, "Barracuda/Broadcast", kernelsList);
            LoadIf(ComputeInfo.supportsCompute, "Barracuda/Pool", kernelsList);
            LoadIf(ComputeInfo.supportsCompute, "Barracuda/Pad", kernelsList);
            LoadIf(ComputeInfo.supportsCompute, "Barracuda/Dense", kernelsList);
            LoadIf(ComputeInfo.supportsCompute, "Barracuda/DenseFP16", kernelsList);
            LoadIf(ComputeInfo.supportsCompute, "Barracuda/Conv", kernelsList);
            LoadIf(ComputeInfo.supportsCompute, "Barracuda/Conv3d", kernelsList);

            var res = kernelsList.ToArray();
            Profiler.EndSample();

            return res;
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
