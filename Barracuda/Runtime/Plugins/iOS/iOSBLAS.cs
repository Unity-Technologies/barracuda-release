#if UNITY_IOS
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Scripting;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;

[assembly: AlwaysLinkAssembly]

namespace Unity.Barracuda {

    [Preserve]
    public class iOSBLAS : BLASPlugin
    {
        [DllImport("__Internal")]
        static extern unsafe void iossgemm(float* Ap, int AM, int AN,
                                            float* Bp, int BM, int BN,
                                            float* Cp, int CM, int CN,
                                            int bs, bool transposeA, bool transposeB);

        public bool IsNative()
        {
            return true;
        }

        public bool IsCurrentPlatformSupported()
        {
            return Application.platform == RuntimePlatform.IPhonePlayer;
        }

        public unsafe void SGEMM(float* Ap, int AM, int AN, float* Bp, int BM, int BN, float* Cp, int CM, int CN, int bs,
            bool transposeA = false, bool transposeB = false)
        {
            iossgemm(Ap, AM, AN, Bp, BM, BN, Cp, CM, CN, bs, transposeA, transposeB);
        }

        public unsafe JobHandle ScheduleSGEMM(JobHandle dependsOn,
            float* Ap, int AM, int AN, float* Bp, int BM, int BN, float* Cp, int CM, int CN,
            int bs,
            bool transposeA = false, bool transposeB = false)
        {
            var job = new SGEMMJob();
            job.Ap = Ap; job.AM = AM; job.AN = AN;
            job.Bp = Bp; job.BM = BM; job.BN = BN;
            job.Cp = Cp; job.CM = CM; job.CN = CN;
            job.transposeA = transposeA;
            job.transposeB = transposeB;
            job.bs = bs;
            return job.Schedule(dependsOn);
        }

        unsafe struct SGEMMJob : IJob
        {
            [NativeDisableUnsafePtrRestriction][ReadOnly] public unsafe float* Ap;
            public int AM, AN;
            [NativeDisableUnsafePtrRestriction][ReadOnly] public unsafe float* Bp;
            public int BM, BN;
            [NativeDisableUnsafePtrRestriction]           public unsafe float* Cp;
            public int CM, CN;
            public int bs;
            public bool transposeA;
            public bool transposeB;

            public void Execute()
            {
                iossgemm(Ap, AM, AN, Bp, BM, BN, Cp, CM, CN, bs, transposeA, transposeB);
            }
        }
    }
}
#endif // UNITY_IOS
