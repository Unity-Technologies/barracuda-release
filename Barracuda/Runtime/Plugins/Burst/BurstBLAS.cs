using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using UnityEngine.Scripting;

[assembly: AlwaysLinkAssembly]

namespace Unity.Barracuda
{

    [Preserve]
    public class BurstBLAS : BLASPlugin
    {
        public bool IsCurrentPlatformSupported()
        {
            try
            {
                // Sanity test if all the dependencies of the job are met at runtime
                // Also prevent compiler from optimising this out
                var test = new UnsafeMatrixBlockMultiplyUnrolled8xhJob();
                D.Log($"Loaded: {test}");
            }
            catch (Exception e)
            {
                D.Log($"C# Job system not found. Disabling {this.GetType()}. Error: {e}");
                return false;
            }

            return true;
        }

        public unsafe void SGEMM(float* Ap, int AN, int AM, float* Bp, int BN, int BM, float* Cp, int CN, int CM,
            int bs,
            bool transposeA = false, bool transposeB = false)
        {
            if (transposeA)
            {
                var tmp = AN;
                AN = AM;
                AM = tmp;
            }

            if (transposeB)
            {
                var tmp = BN;
                BN = BM;
                BM = tmp;
            }

            UnsafeMatrixBlockMultiplyUnrolled8xhJob job = new UnsafeMatrixBlockMultiplyUnrolled8xhJob();
            job.A = Ap;
            job.AN = AN;
            job.AM = AM;
            job.B = Bp;
            job.BN = BN;
            job.BM = BM;
            job.C = Cp;
            job.CN = CN;
            job.CM = CM;
            job.bs = bs;
            job.transposeA = transposeA;
            job.transposeB = transposeB;

            //D.Log($"Matrix mul {AN},{AM} x {BN},{BM}");

            JobHandle fence;

            if (AN < BM)
            {
                job.scheduleRowA = false;
                fence = job.Schedule((BM / bs) + (BM % bs > 0 ? 1 : 0), 1);
            }
            else
            {
                job.scheduleRowA = true;
                fence = job.Schedule((AN / bs) + (AN % bs > 0 ? 1 : 0), 1);
            }

            fence.Complete();
        }
    }

    [BurstCompile]
    struct UnsafeMatrixBlockMultiplyUnrolled8xhJob : IJobParallelFor
    {
        [NativeDisableParallelForRestriction] [NativeDisableUnsafePtrRestriction]
        public unsafe float* A;

        public int AN, AM;

        [NativeDisableParallelForRestriction] [NativeDisableUnsafePtrRestriction]
        public unsafe float* B;

        public int BN, BM;

        [NativeDisableParallelForRestriction] [NativeDisableUnsafePtrRestriction]
        public unsafe float* C;

        public int CN, CM;
        public int bs;
        public bool scheduleRowA;
        public bool transposeA;
        public bool transposeB;

        public void Execute(int n)
        {
            unsafe
            {
                int sz = bs * bs * 4;

                float* blockA = (float*) UnsafeUtility.Malloc(sz, 4, Allocator.TempJob);
                float* blockB = (float*) UnsafeUtility.Malloc(sz, 4, Allocator.TempJob);
                float* blockC = (float*) UnsafeUtility.Malloc(sz, 4, Allocator.TempJob);

                if (scheduleRowA)
                    ExecutOverRowA(blockA, blockB, blockC, n * bs);
                else
                    ExecutOverColB(blockA, blockB, blockC, n * bs);

                UnsafeUtility.Free(blockA, Allocator.TempJob);
                UnsafeUtility.Free(blockB, Allocator.TempJob);
                UnsafeUtility.Free(blockC, Allocator.TempJob);
            }
        }

        private unsafe void ExecutOverColB(float* blockA, float* blockB, float* blockC, int colB)
        {
            for (int rowA = 0; rowA < AN; rowA += bs)
            {
                //for (int colB = 0; colB < BM; colB += bs)
                {
                    for (int l = 0; l < AM; l += bs)
                    {

                        MatrixUtils.CopyBlockWithPadding(A, rowA, AN, l, AM, blockA, bs, transposeA);
                        MatrixUtils.CopyBlockWithPadding(B, l, BN, colB, BM, blockB, bs, transposeB);
                        MatrixUtils.CopyBlockWithPadding(C, rowA, CN, colB, CM, blockC, bs);

                        MatrixUtils.MultiplyBlockUnroll8xhPadded(blockA, blockB, blockC, bs);

                        MatrixUtils.CopyBlockWithPadding(blockC, C, rowA, CN, colB, CM, bs);
                    }
                }
            }
        }

        private unsafe void ExecutOverRowA(float* blockA, float* blockB, float* blockC, int rowA)
        {
            //for (int rowA = 0; rowA < AN; rowA += bs)
            {
                for (int colB = 0; colB < BM; colB += bs)
                {
                    for (int l = 0; l < AM; l += bs)
                    {

                        MatrixUtils.CopyBlockWithPadding(A, rowA, AN, l, AM, blockA, bs, transposeA);
                        MatrixUtils.CopyBlockWithPadding(B, l, BN, colB, BM, blockB, bs, transposeB);
                        MatrixUtils.CopyBlockWithPadding(C, rowA, CN, colB, CM, blockC, bs);

                        MatrixUtils.MultiplyBlockUnroll8xhPadded(blockA, blockB, blockC, bs);

                        MatrixUtils.CopyBlockWithPadding(blockC, C, rowA, CN, colB, CM, bs);
                    }
                }
            }
        }
    }
}
