using System;
using Unity.Burst;
using Unity.Jobs;
using UnityEngine.Scripting;

[assembly: AlwaysLinkAssembly]
[assembly: BurstCompile(OptimizeFor = OptimizeFor.FastCompilation)]

namespace Unity.Barracuda
{

    /// <summary>
    /// Burst specific BLAS implementation
    /// </summary>
    [Preserve]
    public class BurstBLAS : BLASPlugin
    {
        /// <inheritdoc/>
        public bool IsNative()
        {
            return false; // not a native fast BLAS implementation
        }

        /// <inheritdoc/>
        public bool IsCurrentPlatformSupported()
        {
            try
            {
                // Sanity test if all the dependencies of the job are met at runtime
                // Also prevent compiler from optimising this out
                new BurstCPUOps.MatrixMultiplyJob();
            }
            catch (Exception e)
            {
                D.Log($"C# Job system not found. Disabling {this.GetType()}. Error: {e}");
                return false;
            }

            return true;
        }

        /// <inheritdoc/>
        public unsafe void SGEMM(float* Ap, int AM, int AN, float* Bp, int BM, int BN, float* Cp, int CM, int CN,
            int bs,
            bool transposeA = false, bool transposeB = false)
        {
            var noDependencies = new JobHandle();
            var fence = ScheduleSGEMM(noDependencies, Ap, AM, AN, Bp, BM, BN, Cp, CM, CN, bs, transposeA, transposeB);
            fence.Complete();
        }

        /// <inheritdoc/>
        public unsafe JobHandle ScheduleSGEMM(JobHandle dependsOn,
            float* Ap, int AM, int AN, float* Bp, int BM, int BN, float* Cp, int CM, int CN,
            int bs, // NOTE: bs (block size) is ignored
            bool transposeA = false, bool transposeB = false)
        {
            var job = new BurstCPUOps.MatrixMultiplyJob();
            job.A = Ap; job.AM = AM; job.AN = AN;
            job.B = Bp; job.BM = BM; job.BN = BN;
            job.C = Cp; job.CM = CM; job.CN = CN;
            job.transposeA = transposeA;
            job.transposeB = transposeB;

            return job.Schedule(dependsOn);
        }
    }
}
