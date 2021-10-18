using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Profiling;

namespace Unity.Barracuda
{
    /// <summary>
    /// Stores compute kernel cache for GPU pixel shader backends
    /// </summary>
    public sealed class PixelShaderSingleton
    {
        /// <summary>
        /// Enable kernel usage tracking
        /// </summary>
        public bool EnableDebug = false;

        private static readonly PixelShaderSingleton instance = new PixelShaderSingleton();

        // Maps shader name -> Shader
        private Dictionary<string, Shader> m_shaderNameToPixelShader = new Dictionary<string, Shader>();

        private HashSet<string> m_usedShaders = new HashSet<string>();

        internal Shader FindShader(string kernelName)
        {
            if (EnableDebug) m_usedShaders.Add(kernelName);

            if (!m_shaderNameToPixelShader.ContainsKey(kernelName))
            {
                Profiler.BeginSample(kernelName);
                m_shaderNameToPixelShader[kernelName] = Shader.Find(kernelName);
                Profiler.EndSample();
            }

            return m_shaderNameToPixelShader[kernelName];
        }

        /// <summary>
        /// Warmup pixel shaders
        /// </summary>
        /// <param name="shaders">list of shaders to warm up</param>
        /// <returns>IEnumerator</returns>
        public IEnumerator WarmupPixelShaderKernels(List<string> shaders)
        {
            foreach (var shader in shaders)
            {
                if (!m_shaderNameToPixelShader.ContainsKey(shader))
                {
                    FindShader(shader);
                    yield return null;
                }
            }
            yield break;
        }

        /// <summary>
        /// Get used pixel shader list
        /// </summary>
        /// <returns>list of kernels</returns>
        public List<string> GetUsedPixelShaders()
        {
            if (!EnableDebug)
            {
                D.LogWarning("List of used pixel shaders was requested while PixelShaderSingleton.EnableDebug == false");
                return null;
            }

            return m_usedShaders.ToList();
        }

        /// <summary>
        /// Singleton
        /// </summary>
        public static PixelShaderSingleton Instance {
            get { return instance; }
        }
    }
}
