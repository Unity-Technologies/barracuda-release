#if UNITY_2018_1_OR_NEWER && (UNITY_STANDALONE_OSX || UNITY_EDITOR_OSX)
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Scripting;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;

[assembly: AlwaysLinkAssembly]

namespace Unity.Barracuda
{

    [Preserve]
    public class MacBLAS : BLASPlugin
    {
        [DllImport("macblas")]
        static extern unsafe void macsgemm(float* Ap, int AN, int AM,
            float* Bp, int BN, int BM,
            float* Cp, int CN, int CM,
            int bs, bool transposeA, bool transposeB);

        public bool IsNative()
        {
            return true;
        }

        public bool IsCurrentPlatformSupported()
        {
            return Application.platform == RuntimePlatform.OSXEditor ||
                   Application.platform == RuntimePlatform.OSXPlayer;
        }

        public unsafe void SGEMM(float* Ap, int AN, int AM, float* Bp, int BN, int BM, float* Cp, int CN, int CM,
            int bs,
            bool transposeA = false, bool transposeB = false)
        {
            macsgemm(Ap, AN, AM, Bp, BN, BM, Cp, CN, CM, bs, transposeA, transposeB);
        }

        public unsafe JobHandle ScheduleSGEMM(JobHandle dependsOn,
            float* Ap, int AN, int AM, float* Bp, int BN, int BM, float* Cp, int CN, int CM,
            int bs,
            bool transposeA = false, bool transposeB = false)
        {
            var job = new SGEMMJob();
            job.Ap = Ap; job.AN = AN; job.AM = AM;
            job.Bp = Bp; job.BN = BN; job.BM = BM;
            job.Cp = Cp; job.CN = CN; job.CM = CM;
            job.transposeA = transposeA;
            job.transposeB = transposeB;
            job.bs = bs;
            return job.Schedule(dependsOn);
        }

        unsafe struct SGEMMJob : IJob
        {
            [NativeDisableUnsafePtrRestriction][ReadOnly] public unsafe float* Ap;
            public int AN, AM;
            [NativeDisableUnsafePtrRestriction][ReadOnly] public unsafe float* Bp;
            public int BN, BM;
            [NativeDisableUnsafePtrRestriction]           public unsafe float* Cp;
            public int CN, CM;
            public int bs;
            public bool transposeA;
            public bool transposeB;

            public void Execute()
            {
                macsgemm(Ap, AN, AM, Bp, BN, BM, Cp, CN, CM, bs, transposeA, transposeB);    
            }
        }
    }
}
#endif // UNITY_OSX
