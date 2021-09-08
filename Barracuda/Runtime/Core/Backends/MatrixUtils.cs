using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using UnityEngine.Assertions;
using UnityEngine.Scripting;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;

[assembly: InternalsVisibleTo("Unity.Barracuda.BurstBLAS")]

namespace Unity.Barracuda
{
    [Preserve]
    internal class CSharpBLAS : BLASPlugin
    {
        public bool IsNative()
        {
            return false; // reference implementation
        }

        public bool IsCurrentPlatformSupported()
        {
            return true;
        }

        public unsafe void SGEMM(float* Ap, int AM, int AN, float* Bp, int BM, int BN, float* Cp, int CM, int CN, int bs,
            bool transposeA = false, bool transposeB = false)
        {
            MatrixUtils.MultiplyBlockUnrollHx8ParallelWithPadding(Ap, AM, AN, Bp, BM, BN, Cp, CM, CN, bs,
                transposeA, transposeB);
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
                MatrixUtils.MultiplyBlockUnrollHx8ParallelWithPadding(
                    Ap, AM, AN,
                    Bp, BM, BN,
                    Cp, CM, CN, bs,
                    transposeA, transposeB);
            }
        }
    }

    internal class MatrixUtils
    {
        public static unsafe void CopyBlockWithPadding(float* matrixIn, int row, int M, int col, int N, float[] blockOut, int bs, bool transpose = false)
        {
            Array.Clear(blockOut, 0, bs * bs);

            var rowFinal = Math.Min(row + bs, M);
            var count = Math.Min(col + bs, N) - col;

            // @TODO: measure which one is better - sequential access over matrix memory or blockOut cache
            if (transpose)
            {
                // sequential access over blockOut, strided over matrixIn
                //for (var i = row; i < rowFinal; i++)
                //    for (var j = 0; j < count; ++j)
                //        blockOut[(i - row) * bs + j] = matrixIn[i + (col + j) * N];

                // sequential access over matrixIn, strided over blockOut
                for (var j = 0; j < count; ++j)
                for (var i = row; i < rowFinal; i++)
                    blockOut[(i - row) * bs + j] = matrixIn[i + (col + j) * M];
            }
            else
                for (var i = row; i < rowFinal; i++)
                {
                    //D.Log(string.Format("Copy[{3}] {0} -> {1} {2}", i * M + col, (i - row) * bs, count, i));
                    Marshal.Copy((IntPtr)(matrixIn + i * N + col), blockOut, (i - row) * bs, count);
                }

        }

        public static unsafe void ClearFloatArray(float* arr, float val, int count)
        {
            for (int i = 0; i < count; i++)
            {
                arr[i] = val;
            }
        }

        public static unsafe void CopyFloatArray(float* from, float* to, int count)
        {
            for (int i = 0; i < count; i++)
            {
                to[i] = from[i];
            }
        }

        public static unsafe void CopyBlockWithPadding(float* matrixIn, int row, int M, int col, int N, float* blockOut, int bs, bool transpose = false)
        {
            ClearFloatArray(blockOut, 0, bs * bs);

            var rowFinal = Math.Min(row + bs, M);
            var count = Math.Min(col + bs, N) - col;

            // @TODO: measure which one is better - sequential access over matrix memory or blockOut cache
            if (transpose)
            {
                // sequential access over blockOut, strided over matrixIn
                //for (var i = row; i < rowFinal; i++)
                //    for (var j = 0; j < count; ++j)
                //        blockOut[(i - row) * bs + j] = matrixIn[i + (col + j) * N];

                // sequential access over matrixIn, strided over blockOut
                for (var j = 0; j < count; ++j)
                for (var i = row; i < rowFinal; i++)
                    blockOut[(i - row) * bs + j] = matrixIn[i + (col + j) * M];
            }
            else
                for (var i = row; i < rowFinal; i++)
                {
                    //D.Log(string.Format("Copy[{3}] {0} -> {1} {2}", i * M + col, (i - row) * bs, count, i));
                    CopyFloatArray(matrixIn + i * N + col, blockOut + (i - row) * bs, count);
                }

        }

        public static unsafe void CopyBlockWithPadding(float[] blockOut, float* matrixIn, int row, int M, int col, int N, int bs)
        {
            var rowFinal = Math.Min(row + bs, M);
            var count = Math.Min(col + bs, N) - col;

            for (var i = row; i < rowFinal; i++)
                Marshal.Copy(blockOut, (i - row) * bs, (IntPtr)(matrixIn + i * N + col), count);
        }

        public static unsafe void CopyBlockWithPadding(float* blockOut, float* matrixIn, int row, int M, int col, int N, int bs)
        {
            var rowFinal = Math.Min(row + bs, M);
            var count = Math.Min(col + bs, N) - col;

            for (var i = row; i < rowFinal; i++)
                CopyFloatArray(blockOut + (i - row) * bs, matrixIn + i * N + col, count);
        }

        public static unsafe void MultiplyBlockUnrollHx8Padded(float* Ap,
            float* Bp,
            float* Cp, int bs)
        {
            for (int i = 0; i < bs; i++)
            {
                for (int j = 0; j < bs; j += 8)
                {
                    int baseC = i * bs + j;
                    float sum0 = *(Cp + baseC);
                    float sum1 = *(Cp + baseC + 1);
                    float sum2 = *(Cp + baseC + 2);
                    float sum3 = *(Cp + baseC + 3);
                    float sum4 = *(Cp + baseC + 4);
                    float sum5 = *(Cp + baseC + 5);
                    float sum6 = *(Cp + baseC + 6);
                    float sum7 = *(Cp + baseC + 7);

                    for (int l = 0; l < bs; l++)
                    {
                        float A = Ap[i * bs + l];
                        int baseB = l * bs + j;

                        sum0 += A * *(Bp + baseB);
                        sum1 += A * *(Bp + baseB + 1);
                        sum2 += A * *(Bp + baseB + 2);
                        sum3 += A * *(Bp + baseB + 3);
                        sum4 += A * *(Bp + baseB + 4);
                        sum5 += A * *(Bp + baseB + 5);
                        sum6 += A * *(Bp + baseB + 6);
                        sum7 += A * *(Bp + baseB + 7);
                    }

                    *(Cp + baseC) = sum0;
                    *(Cp + baseC + 1) = sum1;
                    *(Cp + baseC + 2) = sum2;
                    *(Cp + baseC + 3) = sum3;
                    *(Cp + baseC + 4) = sum4;
                    *(Cp + baseC + 5) = sum5;
                    *(Cp + baseC + 6) = sum6;
                    *(Cp + baseC + 7) = sum7;
                }
            }
        }

        public static unsafe void MultiplyBlockUnrollHx8ParallelWithPadding(float* Ap, int AM, int AN,
            float* Bp, int BM, int BN,
            float* Cp, int CM, int CN, int bs,
            bool transposeA = false, bool transposeB = false)
        {
            if (transposeA)
            {
                var tmp = AM; AM = AN; AN = tmp;
            }
            if (transposeB)
            {
                var tmp = BM; BM = BN; BN = tmp;
            }

            int N = AM;
            {
                Assert.IsTrue(bs >= 8, "Matrix Mul block size should be >= 8");

                Parallel.For(0, (BN / bs) + (BN % bs > 0 ? 1 : 0), colB =>
                {
                    float[] blockA = new float[bs * bs];
                    float[] blockB = new float[bs * bs];
                    float[] blockC = new float[bs * bs];

                    for (int rowA = 0; rowA < N; rowA += bs)
                    {
                        for (int l = 0; l < AN; l += bs)
                        {

                            CopyBlockWithPadding(Ap, rowA, AM, l, AN, blockA, bs, transposeA);
                            CopyBlockWithPadding(Bp, l, BM, colB * bs, BN, blockB, bs, transposeB);
                            CopyBlockWithPadding(Cp, rowA, CM, colB * bs, CN, blockC, bs);

                            fixed (float* blockAp = blockA, blockBp = blockB, blockCp = blockC)
                            {
                                MultiplyBlockUnrollHx8Padded(blockAp, blockBp, blockCp, bs);
                            }

                            CopyBlockWithPadding(blockC, Cp, rowA, CM, colB * bs, CN, bs);
                        }
                    }
                });
            }
        }
    }
}

