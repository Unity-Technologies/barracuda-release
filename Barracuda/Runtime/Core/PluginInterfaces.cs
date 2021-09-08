using System;
using System.Collections.Generic;
using UnityEngine;
using Unity.Jobs;

namespace Unity.Barracuda
{
    /// <summary>
    /// BLAS plugin interface, allows to supply platform specific implementation of matrix multiplication
    /// </summary>
    public interface BLASPlugin
    {
        /// <summary>
        /// Query if BLAS implementation is coming from platform's native library
        /// </summary>
        /// <returns>`true` if BLAS implementation is coming from platform's native library</returns>
        bool IsNative();

        /// <summary>
        /// Query if current platform is supported by the BLAS plugin
        /// </summary>
        /// <returns>`true` if plugin supports current platform</returns>
        bool IsCurrentPlatformSupported();

        /// <summary>
        /// Perform matrix multiplication C = A x B + C
        /// </summary>
        /// <param name="Ap">pointer to the matrix A</param>
        /// <param name="AM">matrix A row count</param>
        /// <param name="AN">matrix A column count</param>
        /// <param name="Bp">pointer to the matrix B</param>
        /// <param name="BM">matrix B row count</param>
        /// <param name="BN">matrix B column count</param>
        /// <param name="Cp">pointer to the matrix C</param>
        /// <param name="CM">matrix C row count</param>
        /// <param name="CN">matrix C column count</param>
        /// <param name="bs">inner loop block size (if applicable) bs x bs</param>
        /// <param name="transposeA">matrix A data is in transposed layout</param>
        /// <param name="transposeB">matrix B data is in transposed layout</param>
        unsafe void SGEMM(float* Ap, int AM, int AN,
            float* Bp, int BM, int BN,
            float* Cp, int CM, int CN, int bs,
            bool transposeA = false, bool transposeB = false);

        /// <summary>
        /// Launches matrix multiplication C = A x B + C in async-manner
        /// </summary>
        /// <param name="dependsOn">input data dependency job handle</param>
        /// <param name="Ap">pointer to the matrix A</param>
        /// <param name="AM">matrix A row count</param>
        /// <param name="AN">matrix A column count</param>
        /// <param name="Bp">pointer to the matrix B</param>
        /// <param name="BM">matrix B row count</param>
        /// <param name="BN">matrix B column count</param>
        /// <param name="Cp">pointer to the matrix C</param>
        /// <param name="CM">matrix C row count</param>
        /// <param name="CN">matrix C column count</param>
        /// <param name="bs">inner loop block size (if applicable) bs x bs</param>
        /// <param name="transposeA">matrix A data is in transposed layout</param>
        /// <param name="transposeB">matrix B data is in transposed layout</param>
        /// <returns>job handle</returns>
        unsafe JobHandle ScheduleSGEMM(JobHandle dependsOn,
            float* Ap, int AM, int AN,
            float* Bp, int BM, int BN,
            float* Cp, int CM, int CN, int bs,
            bool transposeA = false, bool transposeB = false);
    }

    internal class BLASPluginFactory
    {
        public static BLASPlugin CreateBLASPlugin()
        {
            BLASPlugin blas = null;

            // TODO make plugins discoverable via custom attributes
            Stack<string> plugins = new Stack<string>();
            plugins.Push(typeof(CSharpBLAS).FullName);
            plugins.Push("Unity.Barracuda.BurstBLAS");

            if (Application.platform == RuntimePlatform.IPhonePlayer)
                plugins.Push("Unity.Barracuda.iOSBLAS");
            else if (Application.platform == RuntimePlatform.OSXPlayer || Application.platform == RuntimePlatform.OSXEditor)
                plugins.Push("Unity.Barracuda.MacBLAS");

            while (plugins.Count > 0)
            {
                var candidate = plugins.Pop();
                foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
                {
                    var t = assembly.GetType(candidate);
                    if (t != null)
                    {
                        try
                        {
                            var inst = Activator.CreateInstance(t) as BLASPlugin;

                            if (inst != null && inst.IsCurrentPlatformSupported())
                            {
                                blas = inst;
                            }
                        }
                        catch (Exception e)
                        {
                            D.LogWarning($"Failed to load {t} with exception {e}");
                            break;
                        }
                    }
                }

                // Found working candidate
                if (blas != null)
                    break;
            }

            return blas;
        }
    }
}
