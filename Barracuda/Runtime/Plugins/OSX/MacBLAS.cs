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
        [DllImport("/System/Library/Frameworks/Accelerate.framework/Accelerate")]
        static extern unsafe void cblas_sgemm(CBLAS_ORDER __Order, CBLAS_TRANSPOSE __TransA, CBLAS_TRANSPOSE __TransB,
            int __M, int __N, int __K, float __alpha, float *__A, int __lda, float *__B, int __ldb,
            float __beta, float *__C, int __ldc);

        public bool IsNative()
        {
            return true;
        }

        public bool IsCurrentPlatformSupported()
        {
            return Application.platform == RuntimePlatform.OSXEditor ||
                   Application.platform == RuntimePlatform.OSXPlayer;
        }

        public unsafe void SGEMM(float* Ap, int AM, int AN, float* Bp, int BM, int BN, float* Cp, int CM, int CN,
            int bs,
            bool transposeA = false, bool transposeB = false)
        {
            cblas_sgemm(CBLAS_ORDER.CblasRowMajor, transposeA ? CBLAS_TRANSPOSE.CblasTrans : CBLAS_TRANSPOSE.CblasNoTrans,
                transposeB ? CBLAS_TRANSPOSE.CblasTrans : CBLAS_TRANSPOSE.CblasNoTrans,
                AM, BN, BM, 1.0f, Ap, AN, Bp, BN, 1.0f, Cp, CN);
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
                cblas_sgemm(CBLAS_ORDER.CblasRowMajor, transposeA ? CBLAS_TRANSPOSE.CblasTrans : CBLAS_TRANSPOSE.CblasNoTrans,
                    transposeB ? CBLAS_TRANSPOSE.CblasTrans : CBLAS_TRANSPOSE.CblasNoTrans,
                    AM, BN, BM, 1.0f, Ap, AN, Bp, BN, 1.0f, Cp, CN);
            }
        }

        internal enum CBLAS_ORDER
        {
            CblasRowMajor=101,
            CblasColMajor=102
        };

        internal enum CBLAS_TRANSPOSE
        {
            CblasNoTrans=111,
            CblasTrans=112,
            CblasConjTrans=113,
            AtlasConj=114
        };
    }
}
#endif // UNITY_OSX
