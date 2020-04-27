using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using UnityEngine.Assertions;
using UnityEngine.Scripting;

namespace Unity.Barracuda
{
    [Preserve]
    public class CSharpBLAS : BLASPlugin
    {
        public bool IsCurrentPlatformSupported()
        {
            return true;
        }

        public unsafe void SGEMM(float* Ap, int AN, int AM, float* Bp, int BN, int BM, float* Cp, int CN, int CM, int bs,
            bool transposeA = false, bool transposeB = false)
        {
            MatrixUtils.MultiplyBlockUnroll8xhParallelWithPadding(Ap, AN, AM, Bp, BN, BM, Cp, CN, CM, bs,
                transposeA, transposeB);
        }
    }

    public class MatrixUtils
    {
        public static unsafe void CopyBlockWithPadding(float* matrixIn, int row, int N, int col, int M, float[] blockOut, int bs, bool transpose = false)
        {
            Array.Clear(blockOut, 0, bs * bs);

            var rowFinal = Math.Min(row + bs, N);
            var count = Math.Min(col + bs, M) - col;

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
                    blockOut[(i - row) * bs + j] = matrixIn[i + (col + j) * N];
            }
            else
                for (var i = row; i < rowFinal; i++)
                {
                    //D.Log(string.Format("Copy[{3}] {0} -> {1} {2}", i * M + col, (i - row) * bs, count, i));
                    Marshal.Copy((IntPtr)(matrixIn + i * M + col), blockOut, (i - row) * bs, count);
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

        public static unsafe void CopyBlockWithPadding(float* matrixIn, int row, int N, int col, int M, float* blockOut, int bs, bool transpose = false)
        {
            ClearFloatArray(blockOut, 0, bs * bs);

            var rowFinal = Math.Min(row + bs, N);
            var count = Math.Min(col + bs, M) - col;

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
                    blockOut[(i - row) * bs + j] = matrixIn[i + (col + j) * N];
            }
            else
                for (var i = row; i < rowFinal; i++)
                {
                    //D.Log(string.Format("Copy[{3}] {0} -> {1} {2}", i * M + col, (i - row) * bs, count, i));
                    CopyFloatArray(matrixIn + i * M + col, blockOut + (i - row) * bs, count);
                }

        }

        public static unsafe void CopyBlockWithPadding(float[] blockOut, float* matrixIn, int row, int N, int col, int M, int bs)
        {
            var rowFinal = Math.Min(row + bs, N);
            var count = Math.Min(col + bs, M) - col;

            for (var i = row; i < rowFinal; i++)
                Marshal.Copy(blockOut, (i - row) * bs, (IntPtr)(matrixIn + i * M + col), count);
        }

        public static unsafe void CopyBlockWithPadding(float* blockOut, float* matrixIn, int row, int N, int col, int M, int bs)
        {
            var rowFinal = Math.Min(row + bs, N);
            var count = Math.Min(col + bs, M) - col;

            for (var i = row; i < rowFinal; i++)
                CopyFloatArray(blockOut + (i - row) * bs, matrixIn + i * M + col, count);
        }

        public static unsafe void MultiplyBlockUnroll8xhPadded(float* Ap,
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

        public static unsafe void MultiplyBlockUnroll8xhParallelWithPadding(float* Ap, int AN, int AM,
            float* Bp, int BN, int BM,
            float* Cp, int CN, int CM, int bs,
            bool transposeA = false, bool transposeB = false)
        {
            if (transposeA)
            {
                var tmp = AN; AN = AM; AM = tmp;
            }
            if (transposeB)
            {
                var tmp = BN; BN = BM; BM = tmp;
            }

            int N = AN;
            int M = AM;
            int K = BM;

            {
                Assert.IsTrue(bs >= 8, "Matrix Mul block size should be >= 8");

                Parallel.For(0, (BM / bs) + (BM % bs > 0 ? 1 : 0), colB =>
                {
                    float[] blockA = new float[bs * bs];
                    float[] blockB = new float[bs * bs];
                    float[] blockC = new float[bs * bs];

                    for (int rowA = 0; rowA < N; rowA += bs)
                    {
                        //for (int colB = 0; colB < BM; colB += bs)
                        {
                            for (int l = 0; l < AM; l += bs)
                            {

                                CopyBlockWithPadding(Ap, rowA, AN, l, AM, blockA, bs, transposeA);
                                CopyBlockWithPadding(Bp, l, BN, colB * bs, BM, blockB, bs, transposeB);
                                CopyBlockWithPadding(Cp, rowA, CN, colB * bs, CM, blockC, bs);

                                fixed (float* blockAp = blockA, blockBp = blockB, blockCp = blockC)
                                {
                                    MatrixUtils.MultiplyBlockUnroll8xhPadded(blockAp, blockBp, blockCp, bs);
                                }

                                CopyBlockWithPadding(blockC, Cp, rowA, CN, colB * bs, CM, bs);
                            }
                        }
                    }
                });
            }
        }
    }
}

