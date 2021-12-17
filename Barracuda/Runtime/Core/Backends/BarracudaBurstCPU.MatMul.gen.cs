// This is auto-generated -- do not modify directly
using UnityEngine;
using System;
using Unity.Burst;
using Unity.Burst.Intrinsics;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using static Unity.Burst.Intrinsics.X86.Avx;
using static Unity.Burst.Intrinsics.X86.Fma;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs.LowLevel.Unsafe;
using FencingHelperMode = Unity.Barracuda.BurstSchedulingHelper.FencingHelperMode;

namespace Unity.Barracuda {
public partial class BurstCPUOps
{
    static unsafe void MultiplyBlockUnroll1x8(
            [NoAlias] float* Ap, int Astride,
            [NoAlias] float* Bp, int Bstride,
            [NoAlias] float* Cp, int Cstride,
            int blockSizeM, int blockSizeK,
            int n)
    {
        n = Math.Max(8, n);
        int i = 0;
        for (; i < blockSizeM - 0; i += 1)
        {
            var i_0 = i + 0;

            for (int j = 0; j < n; j += 8)
            {
                int baseC_0 = i_0 * Cstride + j;
                // 0
                float sum0_0 = *(Cp + baseC_0 + 0);
                float sum1_0 = *(Cp + baseC_0 + 1);
                float sum2_0 = *(Cp + baseC_0 + 2);
                float sum3_0 = *(Cp + baseC_0 + 3);
                float sum4_0 = *(Cp + baseC_0 + 4);
                float sum5_0 = *(Cp + baseC_0 + 5);
                float sum6_0 = *(Cp + baseC_0 + 6);
                float sum7_0 = *(Cp + baseC_0 + 7);

                for (int l = 0; l < blockSizeK; l++)
                {
                    float A_0 = *(Ap + i_0 * Astride + l);
                    int baseB = l * Bstride + j;
                    float B_0 = (*(Bp + baseB + 0));
                    float B_1 = (*(Bp + baseB + 1));
                    float B_2 = (*(Bp + baseB + 2));
                    float B_3 = (*(Bp + baseB + 3));
                    float B_4 = (*(Bp + baseB + 4));
                    float B_5 = (*(Bp + baseB + 5));
                    float B_6 = (*(Bp + baseB + 6));
                    float B_7 = (*(Bp + baseB + 7));
                    sum0_0 += A_0 * B_0;
                    sum1_0 += A_0 * B_1;
                    sum2_0 += A_0 * B_2;
                    sum3_0 += A_0 * B_3;
                    sum4_0 += A_0 * B_4;
                    sum5_0 += A_0 * B_5;
                    sum6_0 += A_0 * B_6;
                    sum7_0 += A_0 * B_7;
                }
                // 0
                *(Cp + baseC_0 + 0) = sum0_0;
                *(Cp + baseC_0 + 1) = sum1_0;
                *(Cp + baseC_0 + 2) = sum2_0;
                *(Cp + baseC_0 + 3) = sum3_0;
                *(Cp + baseC_0 + 4) = sum4_0;
                *(Cp + baseC_0 + 5) = sum5_0;
                *(Cp + baseC_0 + 6) = sum6_0;
                *(Cp + baseC_0 + 7) = sum7_0;
            }
        }
    }

    static unsafe void MultiplyBlockUnroll1x8I(
        [NoAlias] float* Ap, int Astride,
        [NoAlias] float* Bp, int Bstride,
        [NoAlias] float* Cp, int Cstride,
        int blockSizeM, int blockSizeK,
        int n)
    {
        n = Math.Max(8, n);
        int i = 0;
        for (; i < blockSizeM - 0; i += 1)
        {
            var i_0 = i + 0;

            for (int j = 0; j < n; j += 8)
            {
                int baseC_0 = i_0 * Cstride + j;

                // row 0
                v256 gamma_0_0 = mm256_loadu_ps(Cp + baseC_0 + 0);

                for (int l = 0; l < blockSizeK; l++)
                {
                    v256 alpha_0_p = mm256_broadcast_ss(Ap + i_0 * Astride + l);

                    v256 beta_p_0 = mm256_loadu_ps(Bp + l * Bstride + j + 0);

                    gamma_0_0 = mm256_fmadd_ps(alpha_0_p, beta_p_0, gamma_0_0);
                }
                // row 0
                mm256_storeu_ps(Cp + baseC_0 + 0, gamma_0_0);
            }
        }
    }

    static unsafe void MultiplyBlockUnroll1x16(
            [NoAlias] float* Ap, int Astride,
            [NoAlias] float* Bp, int Bstride,
            [NoAlias] float* Cp, int Cstride,
            int blockSizeM, int blockSizeK,
            int n)
    {
        n = Math.Max(16, n);
        int i = 0;
        for (; i < blockSizeM - 0; i += 1)
        {
            var i_0 = i + 0;

            for (int j = 0; j < n; j += 16)
            {
                int baseC_0 = i_0 * Cstride + j;
                // 0
                float sum0_0 = *(Cp + baseC_0 + 0);
                float sum1_0 = *(Cp + baseC_0 + 1);
                float sum2_0 = *(Cp + baseC_0 + 2);
                float sum3_0 = *(Cp + baseC_0 + 3);
                float sum4_0 = *(Cp + baseC_0 + 4);
                float sum5_0 = *(Cp + baseC_0 + 5);
                float sum6_0 = *(Cp + baseC_0 + 6);
                float sum7_0 = *(Cp + baseC_0 + 7);
                float sum8_0 = *(Cp + baseC_0 + 8);
                float sum9_0 = *(Cp + baseC_0 + 9);
                float sum10_0 = *(Cp + baseC_0 + 10);
                float sum11_0 = *(Cp + baseC_0 + 11);
                float sum12_0 = *(Cp + baseC_0 + 12);
                float sum13_0 = *(Cp + baseC_0 + 13);
                float sum14_0 = *(Cp + baseC_0 + 14);
                float sum15_0 = *(Cp + baseC_0 + 15);

                for (int l = 0; l < blockSizeK; l++)
                {
                    float A_0 = *(Ap + i_0 * Astride + l);
                    int baseB = l * Bstride + j;
                    float B_0 = (*(Bp + baseB + 0));
                    float B_1 = (*(Bp + baseB + 1));
                    float B_2 = (*(Bp + baseB + 2));
                    float B_3 = (*(Bp + baseB + 3));
                    float B_4 = (*(Bp + baseB + 4));
                    float B_5 = (*(Bp + baseB + 5));
                    float B_6 = (*(Bp + baseB + 6));
                    float B_7 = (*(Bp + baseB + 7));
                    float B_8 = (*(Bp + baseB + 8));
                    float B_9 = (*(Bp + baseB + 9));
                    float B_10 = (*(Bp + baseB + 10));
                    float B_11 = (*(Bp + baseB + 11));
                    float B_12 = (*(Bp + baseB + 12));
                    float B_13 = (*(Bp + baseB + 13));
                    float B_14 = (*(Bp + baseB + 14));
                    float B_15 = (*(Bp + baseB + 15));
                    sum0_0 += A_0 * B_0;
                    sum1_0 += A_0 * B_1;
                    sum2_0 += A_0 * B_2;
                    sum3_0 += A_0 * B_3;
                    sum4_0 += A_0 * B_4;
                    sum5_0 += A_0 * B_5;
                    sum6_0 += A_0 * B_6;
                    sum7_0 += A_0 * B_7;
                    sum8_0 += A_0 * B_8;
                    sum9_0 += A_0 * B_9;
                    sum10_0 += A_0 * B_10;
                    sum11_0 += A_0 * B_11;
                    sum12_0 += A_0 * B_12;
                    sum13_0 += A_0 * B_13;
                    sum14_0 += A_0 * B_14;
                    sum15_0 += A_0 * B_15;
                }
                // 0
                *(Cp + baseC_0 + 0) = sum0_0;
                *(Cp + baseC_0 + 1) = sum1_0;
                *(Cp + baseC_0 + 2) = sum2_0;
                *(Cp + baseC_0 + 3) = sum3_0;
                *(Cp + baseC_0 + 4) = sum4_0;
                *(Cp + baseC_0 + 5) = sum5_0;
                *(Cp + baseC_0 + 6) = sum6_0;
                *(Cp + baseC_0 + 7) = sum7_0;
                *(Cp + baseC_0 + 8) = sum8_0;
                *(Cp + baseC_0 + 9) = sum9_0;
                *(Cp + baseC_0 + 10) = sum10_0;
                *(Cp + baseC_0 + 11) = sum11_0;
                *(Cp + baseC_0 + 12) = sum12_0;
                *(Cp + baseC_0 + 13) = sum13_0;
                *(Cp + baseC_0 + 14) = sum14_0;
                *(Cp + baseC_0 + 15) = sum15_0;
            }
        }
    }

    static unsafe void MultiplyBlockUnroll1x16I(
        [NoAlias] float* Ap, int Astride,
        [NoAlias] float* Bp, int Bstride,
        [NoAlias] float* Cp, int Cstride,
        int blockSizeM, int blockSizeK,
        int n)
    {
        n = Math.Max(16, n);
        int i = 0;
        for (; i < blockSizeM - 0; i += 1)
        {
            var i_0 = i + 0;

            for (int j = 0; j < n; j += 16)
            {
                int baseC_0 = i_0 * Cstride + j;

                // row 0
                v256 gamma_0_0 = mm256_loadu_ps(Cp + baseC_0 + 0);
                v256 gamma_0_8 = mm256_loadu_ps(Cp + baseC_0 + 8);

                for (int l = 0; l < blockSizeK; l++)
                {
                    v256 alpha_0_p = mm256_broadcast_ss(Ap + i_0 * Astride + l);

                    v256 beta_p_0 = mm256_loadu_ps(Bp + l * Bstride + j + 0);
                    v256 beta_p_8 = mm256_loadu_ps(Bp + l * Bstride + j + 8);

                    gamma_0_0 = mm256_fmadd_ps(alpha_0_p, beta_p_0, gamma_0_0);
                    gamma_0_8 = mm256_fmadd_ps(alpha_0_p, beta_p_8, gamma_0_8);
                }
                // row 0
                mm256_storeu_ps(Cp + baseC_0 + 0, gamma_0_0);
                mm256_storeu_ps(Cp + baseC_0 + 8, gamma_0_8);
            }
        }
    }

    static unsafe void MultiplyBlockUnroll2x24(
            [NoAlias] float* Ap, int Astride,
            [NoAlias] float* Bp, int Bstride,
            [NoAlias] float* Cp, int Cstride,
            int blockSizeM, int blockSizeK,
            int n)
    {
        n = Math.Max(24, n);
        int i = 0;
        for (; i < blockSizeM - 1; i += 2)
        {
            var i_0 = i + 0;
            var i_1 = i + 1;

            for (int j = 0; j < n; j += 24)
            {
                int baseC_0 = i_0 * Cstride + j;
                int baseC_1 = i_1 * Cstride + j;
                // 0
                float sum0_0 = *(Cp + baseC_0 + 0);
                float sum1_0 = *(Cp + baseC_0 + 1);
                float sum2_0 = *(Cp + baseC_0 + 2);
                float sum3_0 = *(Cp + baseC_0 + 3);
                float sum4_0 = *(Cp + baseC_0 + 4);
                float sum5_0 = *(Cp + baseC_0 + 5);
                float sum6_0 = *(Cp + baseC_0 + 6);
                float sum7_0 = *(Cp + baseC_0 + 7);
                float sum8_0 = *(Cp + baseC_0 + 8);
                float sum9_0 = *(Cp + baseC_0 + 9);
                float sum10_0 = *(Cp + baseC_0 + 10);
                float sum11_0 = *(Cp + baseC_0 + 11);
                float sum12_0 = *(Cp + baseC_0 + 12);
                float sum13_0 = *(Cp + baseC_0 + 13);
                float sum14_0 = *(Cp + baseC_0 + 14);
                float sum15_0 = *(Cp + baseC_0 + 15);
                float sum16_0 = *(Cp + baseC_0 + 16);
                float sum17_0 = *(Cp + baseC_0 + 17);
                float sum18_0 = *(Cp + baseC_0 + 18);
                float sum19_0 = *(Cp + baseC_0 + 19);
                float sum20_0 = *(Cp + baseC_0 + 20);
                float sum21_0 = *(Cp + baseC_0 + 21);
                float sum22_0 = *(Cp + baseC_0 + 22);
                float sum23_0 = *(Cp + baseC_0 + 23);
                // 1
                float sum0_1 = *(Cp + baseC_1 + 0);
                float sum1_1 = *(Cp + baseC_1 + 1);
                float sum2_1 = *(Cp + baseC_1 + 2);
                float sum3_1 = *(Cp + baseC_1 + 3);
                float sum4_1 = *(Cp + baseC_1 + 4);
                float sum5_1 = *(Cp + baseC_1 + 5);
                float sum6_1 = *(Cp + baseC_1 + 6);
                float sum7_1 = *(Cp + baseC_1 + 7);
                float sum8_1 = *(Cp + baseC_1 + 8);
                float sum9_1 = *(Cp + baseC_1 + 9);
                float sum10_1 = *(Cp + baseC_1 + 10);
                float sum11_1 = *(Cp + baseC_1 + 11);
                float sum12_1 = *(Cp + baseC_1 + 12);
                float sum13_1 = *(Cp + baseC_1 + 13);
                float sum14_1 = *(Cp + baseC_1 + 14);
                float sum15_1 = *(Cp + baseC_1 + 15);
                float sum16_1 = *(Cp + baseC_1 + 16);
                float sum17_1 = *(Cp + baseC_1 + 17);
                float sum18_1 = *(Cp + baseC_1 + 18);
                float sum19_1 = *(Cp + baseC_1 + 19);
                float sum20_1 = *(Cp + baseC_1 + 20);
                float sum21_1 = *(Cp + baseC_1 + 21);
                float sum22_1 = *(Cp + baseC_1 + 22);
                float sum23_1 = *(Cp + baseC_1 + 23);

                for (int l = 0; l < blockSizeK; l++)
                {
                    float A_0 = *(Ap + i_0 * Astride + l);
                    float A_1 = *(Ap + i_1 * Astride + l);
                    int baseB = l * Bstride + j;
                    float B_0 = (*(Bp + baseB + 0));
                    float B_1 = (*(Bp + baseB + 1));
                    float B_2 = (*(Bp + baseB + 2));
                    float B_3 = (*(Bp + baseB + 3));
                    float B_4 = (*(Bp + baseB + 4));
                    float B_5 = (*(Bp + baseB + 5));
                    float B_6 = (*(Bp + baseB + 6));
                    float B_7 = (*(Bp + baseB + 7));
                    float B_8 = (*(Bp + baseB + 8));
                    float B_9 = (*(Bp + baseB + 9));
                    float B_10 = (*(Bp + baseB + 10));
                    float B_11 = (*(Bp + baseB + 11));
                    float B_12 = (*(Bp + baseB + 12));
                    float B_13 = (*(Bp + baseB + 13));
                    float B_14 = (*(Bp + baseB + 14));
                    float B_15 = (*(Bp + baseB + 15));
                    float B_16 = (*(Bp + baseB + 16));
                    float B_17 = (*(Bp + baseB + 17));
                    float B_18 = (*(Bp + baseB + 18));
                    float B_19 = (*(Bp + baseB + 19));
                    float B_20 = (*(Bp + baseB + 20));
                    float B_21 = (*(Bp + baseB + 21));
                    float B_22 = (*(Bp + baseB + 22));
                    float B_23 = (*(Bp + baseB + 23));
                    sum0_0 += A_0 * B_0;	sum0_1 += A_1 * B_0;
                    sum1_0 += A_0 * B_1;	sum1_1 += A_1 * B_1;
                    sum2_0 += A_0 * B_2;	sum2_1 += A_1 * B_2;
                    sum3_0 += A_0 * B_3;	sum3_1 += A_1 * B_3;
                    sum4_0 += A_0 * B_4;	sum4_1 += A_1 * B_4;
                    sum5_0 += A_0 * B_5;	sum5_1 += A_1 * B_5;
                    sum6_0 += A_0 * B_6;	sum6_1 += A_1 * B_6;
                    sum7_0 += A_0 * B_7;	sum7_1 += A_1 * B_7;
                    sum8_0 += A_0 * B_8;	sum8_1 += A_1 * B_8;
                    sum9_0 += A_0 * B_9;	sum9_1 += A_1 * B_9;
                    sum10_0 += A_0 * B_10;	sum10_1 += A_1 * B_10;
                    sum11_0 += A_0 * B_11;	sum11_1 += A_1 * B_11;
                    sum12_0 += A_0 * B_12;	sum12_1 += A_1 * B_12;
                    sum13_0 += A_0 * B_13;	sum13_1 += A_1 * B_13;
                    sum14_0 += A_0 * B_14;	sum14_1 += A_1 * B_14;
                    sum15_0 += A_0 * B_15;	sum15_1 += A_1 * B_15;
                    sum16_0 += A_0 * B_16;	sum16_1 += A_1 * B_16;
                    sum17_0 += A_0 * B_17;	sum17_1 += A_1 * B_17;
                    sum18_0 += A_0 * B_18;	sum18_1 += A_1 * B_18;
                    sum19_0 += A_0 * B_19;	sum19_1 += A_1 * B_19;
                    sum20_0 += A_0 * B_20;	sum20_1 += A_1 * B_20;
                    sum21_0 += A_0 * B_21;	sum21_1 += A_1 * B_21;
                    sum22_0 += A_0 * B_22;	sum22_1 += A_1 * B_22;
                    sum23_0 += A_0 * B_23;	sum23_1 += A_1 * B_23;
                }
                // 0
                *(Cp + baseC_0 + 0) = sum0_0;
                *(Cp + baseC_0 + 1) = sum1_0;
                *(Cp + baseC_0 + 2) = sum2_0;
                *(Cp + baseC_0 + 3) = sum3_0;
                *(Cp + baseC_0 + 4) = sum4_0;
                *(Cp + baseC_0 + 5) = sum5_0;
                *(Cp + baseC_0 + 6) = sum6_0;
                *(Cp + baseC_0 + 7) = sum7_0;
                *(Cp + baseC_0 + 8) = sum8_0;
                *(Cp + baseC_0 + 9) = sum9_0;
                *(Cp + baseC_0 + 10) = sum10_0;
                *(Cp + baseC_0 + 11) = sum11_0;
                *(Cp + baseC_0 + 12) = sum12_0;
                *(Cp + baseC_0 + 13) = sum13_0;
                *(Cp + baseC_0 + 14) = sum14_0;
                *(Cp + baseC_0 + 15) = sum15_0;
                *(Cp + baseC_0 + 16) = sum16_0;
                *(Cp + baseC_0 + 17) = sum17_0;
                *(Cp + baseC_0 + 18) = sum18_0;
                *(Cp + baseC_0 + 19) = sum19_0;
                *(Cp + baseC_0 + 20) = sum20_0;
                *(Cp + baseC_0 + 21) = sum21_0;
                *(Cp + baseC_0 + 22) = sum22_0;
                *(Cp + baseC_0 + 23) = sum23_0;
                // 1
                *(Cp + baseC_1 + 0) = sum0_1;
                *(Cp + baseC_1 + 1) = sum1_1;
                *(Cp + baseC_1 + 2) = sum2_1;
                *(Cp + baseC_1 + 3) = sum3_1;
                *(Cp + baseC_1 + 4) = sum4_1;
                *(Cp + baseC_1 + 5) = sum5_1;
                *(Cp + baseC_1 + 6) = sum6_1;
                *(Cp + baseC_1 + 7) = sum7_1;
                *(Cp + baseC_1 + 8) = sum8_1;
                *(Cp + baseC_1 + 9) = sum9_1;
                *(Cp + baseC_1 + 10) = sum10_1;
                *(Cp + baseC_1 + 11) = sum11_1;
                *(Cp + baseC_1 + 12) = sum12_1;
                *(Cp + baseC_1 + 13) = sum13_1;
                *(Cp + baseC_1 + 14) = sum14_1;
                *(Cp + baseC_1 + 15) = sum15_1;
                *(Cp + baseC_1 + 16) = sum16_1;
                *(Cp + baseC_1 + 17) = sum17_1;
                *(Cp + baseC_1 + 18) = sum18_1;
                *(Cp + baseC_1 + 19) = sum19_1;
                *(Cp + baseC_1 + 20) = sum20_1;
                *(Cp + baseC_1 + 21) = sum21_1;
                *(Cp + baseC_1 + 22) = sum22_1;
                *(Cp + baseC_1 + 23) = sum23_1;
            }
        }
        for (; i < blockSizeM - 0; i += 1)
        {
            var i_0 = i + 0;

            for (int j = 0; j < n; j += 24)
            {
                int baseC_0 = i_0 * Cstride + j;
                // 0
                float sum0_0 = *(Cp + baseC_0 + 0);
                float sum1_0 = *(Cp + baseC_0 + 1);
                float sum2_0 = *(Cp + baseC_0 + 2);
                float sum3_0 = *(Cp + baseC_0 + 3);
                float sum4_0 = *(Cp + baseC_0 + 4);
                float sum5_0 = *(Cp + baseC_0 + 5);
                float sum6_0 = *(Cp + baseC_0 + 6);
                float sum7_0 = *(Cp + baseC_0 + 7);
                float sum8_0 = *(Cp + baseC_0 + 8);
                float sum9_0 = *(Cp + baseC_0 + 9);
                float sum10_0 = *(Cp + baseC_0 + 10);
                float sum11_0 = *(Cp + baseC_0 + 11);
                float sum12_0 = *(Cp + baseC_0 + 12);
                float sum13_0 = *(Cp + baseC_0 + 13);
                float sum14_0 = *(Cp + baseC_0 + 14);
                float sum15_0 = *(Cp + baseC_0 + 15);
                float sum16_0 = *(Cp + baseC_0 + 16);
                float sum17_0 = *(Cp + baseC_0 + 17);
                float sum18_0 = *(Cp + baseC_0 + 18);
                float sum19_0 = *(Cp + baseC_0 + 19);
                float sum20_0 = *(Cp + baseC_0 + 20);
                float sum21_0 = *(Cp + baseC_0 + 21);
                float sum22_0 = *(Cp + baseC_0 + 22);
                float sum23_0 = *(Cp + baseC_0 + 23);

                for (int l = 0; l < blockSizeK; l++)
                {
                    float A_0 = *(Ap + i_0 * Astride + l);
                    int baseB = l * Bstride + j;
                    float B_0 = (*(Bp + baseB + 0));
                    float B_1 = (*(Bp + baseB + 1));
                    float B_2 = (*(Bp + baseB + 2));
                    float B_3 = (*(Bp + baseB + 3));
                    float B_4 = (*(Bp + baseB + 4));
                    float B_5 = (*(Bp + baseB + 5));
                    float B_6 = (*(Bp + baseB + 6));
                    float B_7 = (*(Bp + baseB + 7));
                    float B_8 = (*(Bp + baseB + 8));
                    float B_9 = (*(Bp + baseB + 9));
                    float B_10 = (*(Bp + baseB + 10));
                    float B_11 = (*(Bp + baseB + 11));
                    float B_12 = (*(Bp + baseB + 12));
                    float B_13 = (*(Bp + baseB + 13));
                    float B_14 = (*(Bp + baseB + 14));
                    float B_15 = (*(Bp + baseB + 15));
                    float B_16 = (*(Bp + baseB + 16));
                    float B_17 = (*(Bp + baseB + 17));
                    float B_18 = (*(Bp + baseB + 18));
                    float B_19 = (*(Bp + baseB + 19));
                    float B_20 = (*(Bp + baseB + 20));
                    float B_21 = (*(Bp + baseB + 21));
                    float B_22 = (*(Bp + baseB + 22));
                    float B_23 = (*(Bp + baseB + 23));
                    sum0_0 += A_0 * B_0;
                    sum1_0 += A_0 * B_1;
                    sum2_0 += A_0 * B_2;
                    sum3_0 += A_0 * B_3;
                    sum4_0 += A_0 * B_4;
                    sum5_0 += A_0 * B_5;
                    sum6_0 += A_0 * B_6;
                    sum7_0 += A_0 * B_7;
                    sum8_0 += A_0 * B_8;
                    sum9_0 += A_0 * B_9;
                    sum10_0 += A_0 * B_10;
                    sum11_0 += A_0 * B_11;
                    sum12_0 += A_0 * B_12;
                    sum13_0 += A_0 * B_13;
                    sum14_0 += A_0 * B_14;
                    sum15_0 += A_0 * B_15;
                    sum16_0 += A_0 * B_16;
                    sum17_0 += A_0 * B_17;
                    sum18_0 += A_0 * B_18;
                    sum19_0 += A_0 * B_19;
                    sum20_0 += A_0 * B_20;
                    sum21_0 += A_0 * B_21;
                    sum22_0 += A_0 * B_22;
                    sum23_0 += A_0 * B_23;
                }
                // 0
                *(Cp + baseC_0 + 0) = sum0_0;
                *(Cp + baseC_0 + 1) = sum1_0;
                *(Cp + baseC_0 + 2) = sum2_0;
                *(Cp + baseC_0 + 3) = sum3_0;
                *(Cp + baseC_0 + 4) = sum4_0;
                *(Cp + baseC_0 + 5) = sum5_0;
                *(Cp + baseC_0 + 6) = sum6_0;
                *(Cp + baseC_0 + 7) = sum7_0;
                *(Cp + baseC_0 + 8) = sum8_0;
                *(Cp + baseC_0 + 9) = sum9_0;
                *(Cp + baseC_0 + 10) = sum10_0;
                *(Cp + baseC_0 + 11) = sum11_0;
                *(Cp + baseC_0 + 12) = sum12_0;
                *(Cp + baseC_0 + 13) = sum13_0;
                *(Cp + baseC_0 + 14) = sum14_0;
                *(Cp + baseC_0 + 15) = sum15_0;
                *(Cp + baseC_0 + 16) = sum16_0;
                *(Cp + baseC_0 + 17) = sum17_0;
                *(Cp + baseC_0 + 18) = sum18_0;
                *(Cp + baseC_0 + 19) = sum19_0;
                *(Cp + baseC_0 + 20) = sum20_0;
                *(Cp + baseC_0 + 21) = sum21_0;
                *(Cp + baseC_0 + 22) = sum22_0;
                *(Cp + baseC_0 + 23) = sum23_0;
            }
        }
    }

    static unsafe void MultiplyBlockUnroll2x24I(
        [NoAlias] float* Ap, int Astride,
        [NoAlias] float* Bp, int Bstride,
        [NoAlias] float* Cp, int Cstride,
        int blockSizeM, int blockSizeK,
        int n)
    {
        n = Math.Max(24, n);
        int i = 0;
        for (; i < blockSizeM - 1; i += 2)
        {
            var i_0 = i + 0;
            var i_1 = i + 1;

            for (int j = 0; j < n; j += 24)
            {
                int baseC_0 = i_0 * Cstride + j;
                int baseC_1 = i_1 * Cstride + j;

                // row 0
                v256 gamma_0_0 = mm256_loadu_ps(Cp + baseC_0 + 0);
                v256 gamma_0_8 = mm256_loadu_ps(Cp + baseC_0 + 8);
                v256 gamma_0_16 = mm256_loadu_ps(Cp + baseC_0 + 16);
                // row 1
                v256 gamma_1_0 = mm256_loadu_ps(Cp + baseC_1 + 0);
                v256 gamma_1_8 = mm256_loadu_ps(Cp + baseC_1 + 8);
                v256 gamma_1_16 = mm256_loadu_ps(Cp + baseC_1 + 16);

                for (int l = 0; l < blockSizeK; l++)
                {
                    v256 alpha_0_p = mm256_broadcast_ss(Ap + i_0 * Astride + l);
                    v256 alpha_1_p = mm256_broadcast_ss(Ap + i_1 * Astride + l);

                    v256 beta_p_0 = mm256_loadu_ps(Bp + l * Bstride + j + 0);
                    v256 beta_p_8 = mm256_loadu_ps(Bp + l * Bstride + j + 8);
                    v256 beta_p_16 = mm256_loadu_ps(Bp + l * Bstride + j + 16);

                    gamma_0_0 = mm256_fmadd_ps(alpha_0_p, beta_p_0, gamma_0_0);
                    gamma_1_0 = mm256_fmadd_ps(alpha_1_p, beta_p_0, gamma_1_0);
                    gamma_0_8 = mm256_fmadd_ps(alpha_0_p, beta_p_8, gamma_0_8);
                    gamma_1_8 = mm256_fmadd_ps(alpha_1_p, beta_p_8, gamma_1_8);
                    gamma_0_16 = mm256_fmadd_ps(alpha_0_p, beta_p_16, gamma_0_16);
                    gamma_1_16 = mm256_fmadd_ps(alpha_1_p, beta_p_16, gamma_1_16);
                }
                // row 0
                mm256_storeu_ps(Cp + baseC_0 + 0, gamma_0_0);
                mm256_storeu_ps(Cp + baseC_0 + 8, gamma_0_8);
                mm256_storeu_ps(Cp + baseC_0 + 16, gamma_0_16);
                // row 1
                mm256_storeu_ps(Cp + baseC_1 + 0, gamma_1_0);
                mm256_storeu_ps(Cp + baseC_1 + 8, gamma_1_8);
                mm256_storeu_ps(Cp + baseC_1 + 16, gamma_1_16);
            }
        }
        for (; i < blockSizeM - 0; i += 1)
        {
            var i_0 = i + 0;

            for (int j = 0; j < n; j += 24)
            {
                int baseC_0 = i_0 * Cstride + j;

                // row 0
                v256 gamma_0_0 = mm256_loadu_ps(Cp + baseC_0 + 0);
                v256 gamma_0_8 = mm256_loadu_ps(Cp + baseC_0 + 8);
                v256 gamma_0_16 = mm256_loadu_ps(Cp + baseC_0 + 16);

                for (int l = 0; l < blockSizeK; l++)
                {
                    v256 alpha_0_p = mm256_broadcast_ss(Ap + i_0 * Astride + l);

                    v256 beta_p_0 = mm256_loadu_ps(Bp + l * Bstride + j + 0);
                    v256 beta_p_8 = mm256_loadu_ps(Bp + l * Bstride + j + 8);
                    v256 beta_p_16 = mm256_loadu_ps(Bp + l * Bstride + j + 16);

                    gamma_0_0 = mm256_fmadd_ps(alpha_0_p, beta_p_0, gamma_0_0);
                    gamma_0_8 = mm256_fmadd_ps(alpha_0_p, beta_p_8, gamma_0_8);
                    gamma_0_16 = mm256_fmadd_ps(alpha_0_p, beta_p_16, gamma_0_16);
                }
                // row 0
                mm256_storeu_ps(Cp + baseC_0 + 0, gamma_0_0);
                mm256_storeu_ps(Cp + baseC_0 + 8, gamma_0_8);
                mm256_storeu_ps(Cp + baseC_0 + 16, gamma_0_16);
            }
        }
    }

    static unsafe void MultiplyBlockUnroll2x32(
            [NoAlias] float* Ap, int Astride,
            [NoAlias] float* Bp, int Bstride,
            [NoAlias] float* Cp, int Cstride,
            int blockSizeM, int blockSizeK,
            int n)
    {
        n = Math.Max(32, n);
        int i = 0;
        for (; i < blockSizeM - 1; i += 2)
        {
            var i_0 = i + 0;
            var i_1 = i + 1;

            for (int j = 0; j < n; j += 32)
            {
                int baseC_0 = i_0 * Cstride + j;
                int baseC_1 = i_1 * Cstride + j;
                // 0
                float sum0_0 = *(Cp + baseC_0 + 0);
                float sum1_0 = *(Cp + baseC_0 + 1);
                float sum2_0 = *(Cp + baseC_0 + 2);
                float sum3_0 = *(Cp + baseC_0 + 3);
                float sum4_0 = *(Cp + baseC_0 + 4);
                float sum5_0 = *(Cp + baseC_0 + 5);
                float sum6_0 = *(Cp + baseC_0 + 6);
                float sum7_0 = *(Cp + baseC_0 + 7);
                float sum8_0 = *(Cp + baseC_0 + 8);
                float sum9_0 = *(Cp + baseC_0 + 9);
                float sum10_0 = *(Cp + baseC_0 + 10);
                float sum11_0 = *(Cp + baseC_0 + 11);
                float sum12_0 = *(Cp + baseC_0 + 12);
                float sum13_0 = *(Cp + baseC_0 + 13);
                float sum14_0 = *(Cp + baseC_0 + 14);
                float sum15_0 = *(Cp + baseC_0 + 15);
                float sum16_0 = *(Cp + baseC_0 + 16);
                float sum17_0 = *(Cp + baseC_0 + 17);
                float sum18_0 = *(Cp + baseC_0 + 18);
                float sum19_0 = *(Cp + baseC_0 + 19);
                float sum20_0 = *(Cp + baseC_0 + 20);
                float sum21_0 = *(Cp + baseC_0 + 21);
                float sum22_0 = *(Cp + baseC_0 + 22);
                float sum23_0 = *(Cp + baseC_0 + 23);
                float sum24_0 = *(Cp + baseC_0 + 24);
                float sum25_0 = *(Cp + baseC_0 + 25);
                float sum26_0 = *(Cp + baseC_0 + 26);
                float sum27_0 = *(Cp + baseC_0 + 27);
                float sum28_0 = *(Cp + baseC_0 + 28);
                float sum29_0 = *(Cp + baseC_0 + 29);
                float sum30_0 = *(Cp + baseC_0 + 30);
                float sum31_0 = *(Cp + baseC_0 + 31);
                // 1
                float sum0_1 = *(Cp + baseC_1 + 0);
                float sum1_1 = *(Cp + baseC_1 + 1);
                float sum2_1 = *(Cp + baseC_1 + 2);
                float sum3_1 = *(Cp + baseC_1 + 3);
                float sum4_1 = *(Cp + baseC_1 + 4);
                float sum5_1 = *(Cp + baseC_1 + 5);
                float sum6_1 = *(Cp + baseC_1 + 6);
                float sum7_1 = *(Cp + baseC_1 + 7);
                float sum8_1 = *(Cp + baseC_1 + 8);
                float sum9_1 = *(Cp + baseC_1 + 9);
                float sum10_1 = *(Cp + baseC_1 + 10);
                float sum11_1 = *(Cp + baseC_1 + 11);
                float sum12_1 = *(Cp + baseC_1 + 12);
                float sum13_1 = *(Cp + baseC_1 + 13);
                float sum14_1 = *(Cp + baseC_1 + 14);
                float sum15_1 = *(Cp + baseC_1 + 15);
                float sum16_1 = *(Cp + baseC_1 + 16);
                float sum17_1 = *(Cp + baseC_1 + 17);
                float sum18_1 = *(Cp + baseC_1 + 18);
                float sum19_1 = *(Cp + baseC_1 + 19);
                float sum20_1 = *(Cp + baseC_1 + 20);
                float sum21_1 = *(Cp + baseC_1 + 21);
                float sum22_1 = *(Cp + baseC_1 + 22);
                float sum23_1 = *(Cp + baseC_1 + 23);
                float sum24_1 = *(Cp + baseC_1 + 24);
                float sum25_1 = *(Cp + baseC_1 + 25);
                float sum26_1 = *(Cp + baseC_1 + 26);
                float sum27_1 = *(Cp + baseC_1 + 27);
                float sum28_1 = *(Cp + baseC_1 + 28);
                float sum29_1 = *(Cp + baseC_1 + 29);
                float sum30_1 = *(Cp + baseC_1 + 30);
                float sum31_1 = *(Cp + baseC_1 + 31);

                for (int l = 0; l < blockSizeK; l++)
                {
                    float A_0 = *(Ap + i_0 * Astride + l);
                    float A_1 = *(Ap + i_1 * Astride + l);
                    int baseB = l * Bstride + j;
                    float B_0 = (*(Bp + baseB + 0));
                    float B_1 = (*(Bp + baseB + 1));
                    float B_2 = (*(Bp + baseB + 2));
                    float B_3 = (*(Bp + baseB + 3));
                    float B_4 = (*(Bp + baseB + 4));
                    float B_5 = (*(Bp + baseB + 5));
                    float B_6 = (*(Bp + baseB + 6));
                    float B_7 = (*(Bp + baseB + 7));
                    float B_8 = (*(Bp + baseB + 8));
                    float B_9 = (*(Bp + baseB + 9));
                    float B_10 = (*(Bp + baseB + 10));
                    float B_11 = (*(Bp + baseB + 11));
                    float B_12 = (*(Bp + baseB + 12));
                    float B_13 = (*(Bp + baseB + 13));
                    float B_14 = (*(Bp + baseB + 14));
                    float B_15 = (*(Bp + baseB + 15));
                    float B_16 = (*(Bp + baseB + 16));
                    float B_17 = (*(Bp + baseB + 17));
                    float B_18 = (*(Bp + baseB + 18));
                    float B_19 = (*(Bp + baseB + 19));
                    float B_20 = (*(Bp + baseB + 20));
                    float B_21 = (*(Bp + baseB + 21));
                    float B_22 = (*(Bp + baseB + 22));
                    float B_23 = (*(Bp + baseB + 23));
                    float B_24 = (*(Bp + baseB + 24));
                    float B_25 = (*(Bp + baseB + 25));
                    float B_26 = (*(Bp + baseB + 26));
                    float B_27 = (*(Bp + baseB + 27));
                    float B_28 = (*(Bp + baseB + 28));
                    float B_29 = (*(Bp + baseB + 29));
                    float B_30 = (*(Bp + baseB + 30));
                    float B_31 = (*(Bp + baseB + 31));
                    sum0_0 += A_0 * B_0;	sum0_1 += A_1 * B_0;
                    sum1_0 += A_0 * B_1;	sum1_1 += A_1 * B_1;
                    sum2_0 += A_0 * B_2;	sum2_1 += A_1 * B_2;
                    sum3_0 += A_0 * B_3;	sum3_1 += A_1 * B_3;
                    sum4_0 += A_0 * B_4;	sum4_1 += A_1 * B_4;
                    sum5_0 += A_0 * B_5;	sum5_1 += A_1 * B_5;
                    sum6_0 += A_0 * B_6;	sum6_1 += A_1 * B_6;
                    sum7_0 += A_0 * B_7;	sum7_1 += A_1 * B_7;
                    sum8_0 += A_0 * B_8;	sum8_1 += A_1 * B_8;
                    sum9_0 += A_0 * B_9;	sum9_1 += A_1 * B_9;
                    sum10_0 += A_0 * B_10;	sum10_1 += A_1 * B_10;
                    sum11_0 += A_0 * B_11;	sum11_1 += A_1 * B_11;
                    sum12_0 += A_0 * B_12;	sum12_1 += A_1 * B_12;
                    sum13_0 += A_0 * B_13;	sum13_1 += A_1 * B_13;
                    sum14_0 += A_0 * B_14;	sum14_1 += A_1 * B_14;
                    sum15_0 += A_0 * B_15;	sum15_1 += A_1 * B_15;
                    sum16_0 += A_0 * B_16;	sum16_1 += A_1 * B_16;
                    sum17_0 += A_0 * B_17;	sum17_1 += A_1 * B_17;
                    sum18_0 += A_0 * B_18;	sum18_1 += A_1 * B_18;
                    sum19_0 += A_0 * B_19;	sum19_1 += A_1 * B_19;
                    sum20_0 += A_0 * B_20;	sum20_1 += A_1 * B_20;
                    sum21_0 += A_0 * B_21;	sum21_1 += A_1 * B_21;
                    sum22_0 += A_0 * B_22;	sum22_1 += A_1 * B_22;
                    sum23_0 += A_0 * B_23;	sum23_1 += A_1 * B_23;
                    sum24_0 += A_0 * B_24;	sum24_1 += A_1 * B_24;
                    sum25_0 += A_0 * B_25;	sum25_1 += A_1 * B_25;
                    sum26_0 += A_0 * B_26;	sum26_1 += A_1 * B_26;
                    sum27_0 += A_0 * B_27;	sum27_1 += A_1 * B_27;
                    sum28_0 += A_0 * B_28;	sum28_1 += A_1 * B_28;
                    sum29_0 += A_0 * B_29;	sum29_1 += A_1 * B_29;
                    sum30_0 += A_0 * B_30;	sum30_1 += A_1 * B_30;
                    sum31_0 += A_0 * B_31;	sum31_1 += A_1 * B_31;
                }
                // 0
                *(Cp + baseC_0 + 0) = sum0_0;
                *(Cp + baseC_0 + 1) = sum1_0;
                *(Cp + baseC_0 + 2) = sum2_0;
                *(Cp + baseC_0 + 3) = sum3_0;
                *(Cp + baseC_0 + 4) = sum4_0;
                *(Cp + baseC_0 + 5) = sum5_0;
                *(Cp + baseC_0 + 6) = sum6_0;
                *(Cp + baseC_0 + 7) = sum7_0;
                *(Cp + baseC_0 + 8) = sum8_0;
                *(Cp + baseC_0 + 9) = sum9_0;
                *(Cp + baseC_0 + 10) = sum10_0;
                *(Cp + baseC_0 + 11) = sum11_0;
                *(Cp + baseC_0 + 12) = sum12_0;
                *(Cp + baseC_0 + 13) = sum13_0;
                *(Cp + baseC_0 + 14) = sum14_0;
                *(Cp + baseC_0 + 15) = sum15_0;
                *(Cp + baseC_0 + 16) = sum16_0;
                *(Cp + baseC_0 + 17) = sum17_0;
                *(Cp + baseC_0 + 18) = sum18_0;
                *(Cp + baseC_0 + 19) = sum19_0;
                *(Cp + baseC_0 + 20) = sum20_0;
                *(Cp + baseC_0 + 21) = sum21_0;
                *(Cp + baseC_0 + 22) = sum22_0;
                *(Cp + baseC_0 + 23) = sum23_0;
                *(Cp + baseC_0 + 24) = sum24_0;
                *(Cp + baseC_0 + 25) = sum25_0;
                *(Cp + baseC_0 + 26) = sum26_0;
                *(Cp + baseC_0 + 27) = sum27_0;
                *(Cp + baseC_0 + 28) = sum28_0;
                *(Cp + baseC_0 + 29) = sum29_0;
                *(Cp + baseC_0 + 30) = sum30_0;
                *(Cp + baseC_0 + 31) = sum31_0;
                // 1
                *(Cp + baseC_1 + 0) = sum0_1;
                *(Cp + baseC_1 + 1) = sum1_1;
                *(Cp + baseC_1 + 2) = sum2_1;
                *(Cp + baseC_1 + 3) = sum3_1;
                *(Cp + baseC_1 + 4) = sum4_1;
                *(Cp + baseC_1 + 5) = sum5_1;
                *(Cp + baseC_1 + 6) = sum6_1;
                *(Cp + baseC_1 + 7) = sum7_1;
                *(Cp + baseC_1 + 8) = sum8_1;
                *(Cp + baseC_1 + 9) = sum9_1;
                *(Cp + baseC_1 + 10) = sum10_1;
                *(Cp + baseC_1 + 11) = sum11_1;
                *(Cp + baseC_1 + 12) = sum12_1;
                *(Cp + baseC_1 + 13) = sum13_1;
                *(Cp + baseC_1 + 14) = sum14_1;
                *(Cp + baseC_1 + 15) = sum15_1;
                *(Cp + baseC_1 + 16) = sum16_1;
                *(Cp + baseC_1 + 17) = sum17_1;
                *(Cp + baseC_1 + 18) = sum18_1;
                *(Cp + baseC_1 + 19) = sum19_1;
                *(Cp + baseC_1 + 20) = sum20_1;
                *(Cp + baseC_1 + 21) = sum21_1;
                *(Cp + baseC_1 + 22) = sum22_1;
                *(Cp + baseC_1 + 23) = sum23_1;
                *(Cp + baseC_1 + 24) = sum24_1;
                *(Cp + baseC_1 + 25) = sum25_1;
                *(Cp + baseC_1 + 26) = sum26_1;
                *(Cp + baseC_1 + 27) = sum27_1;
                *(Cp + baseC_1 + 28) = sum28_1;
                *(Cp + baseC_1 + 29) = sum29_1;
                *(Cp + baseC_1 + 30) = sum30_1;
                *(Cp + baseC_1 + 31) = sum31_1;
            }
        }
        for (; i < blockSizeM - 0; i += 1)
        {
            var i_0 = i + 0;

            for (int j = 0; j < n; j += 32)
            {
                int baseC_0 = i_0 * Cstride + j;
                // 0
                float sum0_0 = *(Cp + baseC_0 + 0);
                float sum1_0 = *(Cp + baseC_0 + 1);
                float sum2_0 = *(Cp + baseC_0 + 2);
                float sum3_0 = *(Cp + baseC_0 + 3);
                float sum4_0 = *(Cp + baseC_0 + 4);
                float sum5_0 = *(Cp + baseC_0 + 5);
                float sum6_0 = *(Cp + baseC_0 + 6);
                float sum7_0 = *(Cp + baseC_0 + 7);
                float sum8_0 = *(Cp + baseC_0 + 8);
                float sum9_0 = *(Cp + baseC_0 + 9);
                float sum10_0 = *(Cp + baseC_0 + 10);
                float sum11_0 = *(Cp + baseC_0 + 11);
                float sum12_0 = *(Cp + baseC_0 + 12);
                float sum13_0 = *(Cp + baseC_0 + 13);
                float sum14_0 = *(Cp + baseC_0 + 14);
                float sum15_0 = *(Cp + baseC_0 + 15);
                float sum16_0 = *(Cp + baseC_0 + 16);
                float sum17_0 = *(Cp + baseC_0 + 17);
                float sum18_0 = *(Cp + baseC_0 + 18);
                float sum19_0 = *(Cp + baseC_0 + 19);
                float sum20_0 = *(Cp + baseC_0 + 20);
                float sum21_0 = *(Cp + baseC_0 + 21);
                float sum22_0 = *(Cp + baseC_0 + 22);
                float sum23_0 = *(Cp + baseC_0 + 23);
                float sum24_0 = *(Cp + baseC_0 + 24);
                float sum25_0 = *(Cp + baseC_0 + 25);
                float sum26_0 = *(Cp + baseC_0 + 26);
                float sum27_0 = *(Cp + baseC_0 + 27);
                float sum28_0 = *(Cp + baseC_0 + 28);
                float sum29_0 = *(Cp + baseC_0 + 29);
                float sum30_0 = *(Cp + baseC_0 + 30);
                float sum31_0 = *(Cp + baseC_0 + 31);

                for (int l = 0; l < blockSizeK; l++)
                {
                    float A_0 = *(Ap + i_0 * Astride + l);
                    int baseB = l * Bstride + j;
                    float B_0 = (*(Bp + baseB + 0));
                    float B_1 = (*(Bp + baseB + 1));
                    float B_2 = (*(Bp + baseB + 2));
                    float B_3 = (*(Bp + baseB + 3));
                    float B_4 = (*(Bp + baseB + 4));
                    float B_5 = (*(Bp + baseB + 5));
                    float B_6 = (*(Bp + baseB + 6));
                    float B_7 = (*(Bp + baseB + 7));
                    float B_8 = (*(Bp + baseB + 8));
                    float B_9 = (*(Bp + baseB + 9));
                    float B_10 = (*(Bp + baseB + 10));
                    float B_11 = (*(Bp + baseB + 11));
                    float B_12 = (*(Bp + baseB + 12));
                    float B_13 = (*(Bp + baseB + 13));
                    float B_14 = (*(Bp + baseB + 14));
                    float B_15 = (*(Bp + baseB + 15));
                    float B_16 = (*(Bp + baseB + 16));
                    float B_17 = (*(Bp + baseB + 17));
                    float B_18 = (*(Bp + baseB + 18));
                    float B_19 = (*(Bp + baseB + 19));
                    float B_20 = (*(Bp + baseB + 20));
                    float B_21 = (*(Bp + baseB + 21));
                    float B_22 = (*(Bp + baseB + 22));
                    float B_23 = (*(Bp + baseB + 23));
                    float B_24 = (*(Bp + baseB + 24));
                    float B_25 = (*(Bp + baseB + 25));
                    float B_26 = (*(Bp + baseB + 26));
                    float B_27 = (*(Bp + baseB + 27));
                    float B_28 = (*(Bp + baseB + 28));
                    float B_29 = (*(Bp + baseB + 29));
                    float B_30 = (*(Bp + baseB + 30));
                    float B_31 = (*(Bp + baseB + 31));
                    sum0_0 += A_0 * B_0;
                    sum1_0 += A_0 * B_1;
                    sum2_0 += A_0 * B_2;
                    sum3_0 += A_0 * B_3;
                    sum4_0 += A_0 * B_4;
                    sum5_0 += A_0 * B_5;
                    sum6_0 += A_0 * B_6;
                    sum7_0 += A_0 * B_7;
                    sum8_0 += A_0 * B_8;
                    sum9_0 += A_0 * B_9;
                    sum10_0 += A_0 * B_10;
                    sum11_0 += A_0 * B_11;
                    sum12_0 += A_0 * B_12;
                    sum13_0 += A_0 * B_13;
                    sum14_0 += A_0 * B_14;
                    sum15_0 += A_0 * B_15;
                    sum16_0 += A_0 * B_16;
                    sum17_0 += A_0 * B_17;
                    sum18_0 += A_0 * B_18;
                    sum19_0 += A_0 * B_19;
                    sum20_0 += A_0 * B_20;
                    sum21_0 += A_0 * B_21;
                    sum22_0 += A_0 * B_22;
                    sum23_0 += A_0 * B_23;
                    sum24_0 += A_0 * B_24;
                    sum25_0 += A_0 * B_25;
                    sum26_0 += A_0 * B_26;
                    sum27_0 += A_0 * B_27;
                    sum28_0 += A_0 * B_28;
                    sum29_0 += A_0 * B_29;
                    sum30_0 += A_0 * B_30;
                    sum31_0 += A_0 * B_31;
                }
                // 0
                *(Cp + baseC_0 + 0) = sum0_0;
                *(Cp + baseC_0 + 1) = sum1_0;
                *(Cp + baseC_0 + 2) = sum2_0;
                *(Cp + baseC_0 + 3) = sum3_0;
                *(Cp + baseC_0 + 4) = sum4_0;
                *(Cp + baseC_0 + 5) = sum5_0;
                *(Cp + baseC_0 + 6) = sum6_0;
                *(Cp + baseC_0 + 7) = sum7_0;
                *(Cp + baseC_0 + 8) = sum8_0;
                *(Cp + baseC_0 + 9) = sum9_0;
                *(Cp + baseC_0 + 10) = sum10_0;
                *(Cp + baseC_0 + 11) = sum11_0;
                *(Cp + baseC_0 + 12) = sum12_0;
                *(Cp + baseC_0 + 13) = sum13_0;
                *(Cp + baseC_0 + 14) = sum14_0;
                *(Cp + baseC_0 + 15) = sum15_0;
                *(Cp + baseC_0 + 16) = sum16_0;
                *(Cp + baseC_0 + 17) = sum17_0;
                *(Cp + baseC_0 + 18) = sum18_0;
                *(Cp + baseC_0 + 19) = sum19_0;
                *(Cp + baseC_0 + 20) = sum20_0;
                *(Cp + baseC_0 + 21) = sum21_0;
                *(Cp + baseC_0 + 22) = sum22_0;
                *(Cp + baseC_0 + 23) = sum23_0;
                *(Cp + baseC_0 + 24) = sum24_0;
                *(Cp + baseC_0 + 25) = sum25_0;
                *(Cp + baseC_0 + 26) = sum26_0;
                *(Cp + baseC_0 + 27) = sum27_0;
                *(Cp + baseC_0 + 28) = sum28_0;
                *(Cp + baseC_0 + 29) = sum29_0;
                *(Cp + baseC_0 + 30) = sum30_0;
                *(Cp + baseC_0 + 31) = sum31_0;
            }
        }
    }

    static unsafe void MultiplyBlockUnroll2x32I(
        [NoAlias] float* Ap, int Astride,
        [NoAlias] float* Bp, int Bstride,
        [NoAlias] float* Cp, int Cstride,
        int blockSizeM, int blockSizeK,
        int n)
    {
        n = Math.Max(32, n);
        int i = 0;
        for (; i < blockSizeM - 1; i += 2)
        {
            var i_0 = i + 0;
            var i_1 = i + 1;

            for (int j = 0; j < n; j += 32)
            {
                int baseC_0 = i_0 * Cstride + j;
                int baseC_1 = i_1 * Cstride + j;

                // row 0
                v256 gamma_0_0 = mm256_loadu_ps(Cp + baseC_0 + 0);
                v256 gamma_0_8 = mm256_loadu_ps(Cp + baseC_0 + 8);
                v256 gamma_0_16 = mm256_loadu_ps(Cp + baseC_0 + 16);
                v256 gamma_0_24 = mm256_loadu_ps(Cp + baseC_0 + 24);
                // row 1
                v256 gamma_1_0 = mm256_loadu_ps(Cp + baseC_1 + 0);
                v256 gamma_1_8 = mm256_loadu_ps(Cp + baseC_1 + 8);
                v256 gamma_1_16 = mm256_loadu_ps(Cp + baseC_1 + 16);
                v256 gamma_1_24 = mm256_loadu_ps(Cp + baseC_1 + 24);

                for (int l = 0; l < blockSizeK; l++)
                {
                    v256 alpha_0_p = mm256_broadcast_ss(Ap + i_0 * Astride + l);
                    v256 alpha_1_p = mm256_broadcast_ss(Ap + i_1 * Astride + l);

                    v256 beta_p_0 = mm256_loadu_ps(Bp + l * Bstride + j + 0);
                    v256 beta_p_8 = mm256_loadu_ps(Bp + l * Bstride + j + 8);
                    v256 beta_p_16 = mm256_loadu_ps(Bp + l * Bstride + j + 16);
                    v256 beta_p_24 = mm256_loadu_ps(Bp + l * Bstride + j + 24);

                    gamma_0_0 = mm256_fmadd_ps(alpha_0_p, beta_p_0, gamma_0_0);
                    gamma_1_0 = mm256_fmadd_ps(alpha_1_p, beta_p_0, gamma_1_0);
                    gamma_0_8 = mm256_fmadd_ps(alpha_0_p, beta_p_8, gamma_0_8);
                    gamma_1_8 = mm256_fmadd_ps(alpha_1_p, beta_p_8, gamma_1_8);
                    gamma_0_16 = mm256_fmadd_ps(alpha_0_p, beta_p_16, gamma_0_16);
                    gamma_1_16 = mm256_fmadd_ps(alpha_1_p, beta_p_16, gamma_1_16);
                    gamma_0_24 = mm256_fmadd_ps(alpha_0_p, beta_p_24, gamma_0_24);
                    gamma_1_24 = mm256_fmadd_ps(alpha_1_p, beta_p_24, gamma_1_24);
                }
                // row 0
                mm256_storeu_ps(Cp + baseC_0 + 0, gamma_0_0);
                mm256_storeu_ps(Cp + baseC_0 + 8, gamma_0_8);
                mm256_storeu_ps(Cp + baseC_0 + 16, gamma_0_16);
                mm256_storeu_ps(Cp + baseC_0 + 24, gamma_0_24);
                // row 1
                mm256_storeu_ps(Cp + baseC_1 + 0, gamma_1_0);
                mm256_storeu_ps(Cp + baseC_1 + 8, gamma_1_8);
                mm256_storeu_ps(Cp + baseC_1 + 16, gamma_1_16);
                mm256_storeu_ps(Cp + baseC_1 + 24, gamma_1_24);
            }
        }
        for (; i < blockSizeM - 0; i += 1)
        {
            var i_0 = i + 0;

            for (int j = 0; j < n; j += 32)
            {
                int baseC_0 = i_0 * Cstride + j;

                // row 0
                v256 gamma_0_0 = mm256_loadu_ps(Cp + baseC_0 + 0);
                v256 gamma_0_8 = mm256_loadu_ps(Cp + baseC_0 + 8);
                v256 gamma_0_16 = mm256_loadu_ps(Cp + baseC_0 + 16);
                v256 gamma_0_24 = mm256_loadu_ps(Cp + baseC_0 + 24);

                for (int l = 0; l < blockSizeK; l++)
                {
                    v256 alpha_0_p = mm256_broadcast_ss(Ap + i_0 * Astride + l);

                    v256 beta_p_0 = mm256_loadu_ps(Bp + l * Bstride + j + 0);
                    v256 beta_p_8 = mm256_loadu_ps(Bp + l * Bstride + j + 8);
                    v256 beta_p_16 = mm256_loadu_ps(Bp + l * Bstride + j + 16);
                    v256 beta_p_24 = mm256_loadu_ps(Bp + l * Bstride + j + 24);

                    gamma_0_0 = mm256_fmadd_ps(alpha_0_p, beta_p_0, gamma_0_0);
                    gamma_0_8 = mm256_fmadd_ps(alpha_0_p, beta_p_8, gamma_0_8);
                    gamma_0_16 = mm256_fmadd_ps(alpha_0_p, beta_p_16, gamma_0_16);
                    gamma_0_24 = mm256_fmadd_ps(alpha_0_p, beta_p_24, gamma_0_24);
                }
                // row 0
                mm256_storeu_ps(Cp + baseC_0 + 0, gamma_0_0);
                mm256_storeu_ps(Cp + baseC_0 + 8, gamma_0_8);
                mm256_storeu_ps(Cp + baseC_0 + 16, gamma_0_16);
                mm256_storeu_ps(Cp + baseC_0 + 24, gamma_0_24);
            }
        }
    }

    static unsafe void MultiplyBlockUnroll3x16(
            [NoAlias] float* Ap, int Astride,
            [NoAlias] float* Bp, int Bstride,
            [NoAlias] float* Cp, int Cstride,
            int blockSizeM, int blockSizeK,
            int n)
    {
        n = Math.Max(16, n);
        int i = 0;
        for (; i < blockSizeM - 2; i += 3)
        {
            var i_0 = i + 0;
            var i_1 = i + 1;
            var i_2 = i + 2;

            for (int j = 0; j < n; j += 16)
            {
                int baseC_0 = i_0 * Cstride + j;
                int baseC_1 = i_1 * Cstride + j;
                int baseC_2 = i_2 * Cstride + j;
                // 0
                float sum0_0 = *(Cp + baseC_0 + 0);
                float sum1_0 = *(Cp + baseC_0 + 1);
                float sum2_0 = *(Cp + baseC_0 + 2);
                float sum3_0 = *(Cp + baseC_0 + 3);
                float sum4_0 = *(Cp + baseC_0 + 4);
                float sum5_0 = *(Cp + baseC_0 + 5);
                float sum6_0 = *(Cp + baseC_0 + 6);
                float sum7_0 = *(Cp + baseC_0 + 7);
                float sum8_0 = *(Cp + baseC_0 + 8);
                float sum9_0 = *(Cp + baseC_0 + 9);
                float sum10_0 = *(Cp + baseC_0 + 10);
                float sum11_0 = *(Cp + baseC_0 + 11);
                float sum12_0 = *(Cp + baseC_0 + 12);
                float sum13_0 = *(Cp + baseC_0 + 13);
                float sum14_0 = *(Cp + baseC_0 + 14);
                float sum15_0 = *(Cp + baseC_0 + 15);
                // 1
                float sum0_1 = *(Cp + baseC_1 + 0);
                float sum1_1 = *(Cp + baseC_1 + 1);
                float sum2_1 = *(Cp + baseC_1 + 2);
                float sum3_1 = *(Cp + baseC_1 + 3);
                float sum4_1 = *(Cp + baseC_1 + 4);
                float sum5_1 = *(Cp + baseC_1 + 5);
                float sum6_1 = *(Cp + baseC_1 + 6);
                float sum7_1 = *(Cp + baseC_1 + 7);
                float sum8_1 = *(Cp + baseC_1 + 8);
                float sum9_1 = *(Cp + baseC_1 + 9);
                float sum10_1 = *(Cp + baseC_1 + 10);
                float sum11_1 = *(Cp + baseC_1 + 11);
                float sum12_1 = *(Cp + baseC_1 + 12);
                float sum13_1 = *(Cp + baseC_1 + 13);
                float sum14_1 = *(Cp + baseC_1 + 14);
                float sum15_1 = *(Cp + baseC_1 + 15);
                // 2
                float sum0_2 = *(Cp + baseC_2 + 0);
                float sum1_2 = *(Cp + baseC_2 + 1);
                float sum2_2 = *(Cp + baseC_2 + 2);
                float sum3_2 = *(Cp + baseC_2 + 3);
                float sum4_2 = *(Cp + baseC_2 + 4);
                float sum5_2 = *(Cp + baseC_2 + 5);
                float sum6_2 = *(Cp + baseC_2 + 6);
                float sum7_2 = *(Cp + baseC_2 + 7);
                float sum8_2 = *(Cp + baseC_2 + 8);
                float sum9_2 = *(Cp + baseC_2 + 9);
                float sum10_2 = *(Cp + baseC_2 + 10);
                float sum11_2 = *(Cp + baseC_2 + 11);
                float sum12_2 = *(Cp + baseC_2 + 12);
                float sum13_2 = *(Cp + baseC_2 + 13);
                float sum14_2 = *(Cp + baseC_2 + 14);
                float sum15_2 = *(Cp + baseC_2 + 15);

                for (int l = 0; l < blockSizeK; l++)
                {
                    float A_0 = *(Ap + i_0 * Astride + l);
                    float A_1 = *(Ap + i_1 * Astride + l);
                    float A_2 = *(Ap + i_2 * Astride + l);
                    int baseB = l * Bstride + j;
                    float B_0 = (*(Bp + baseB + 0));
                    float B_1 = (*(Bp + baseB + 1));
                    float B_2 = (*(Bp + baseB + 2));
                    float B_3 = (*(Bp + baseB + 3));
                    float B_4 = (*(Bp + baseB + 4));
                    float B_5 = (*(Bp + baseB + 5));
                    float B_6 = (*(Bp + baseB + 6));
                    float B_7 = (*(Bp + baseB + 7));
                    float B_8 = (*(Bp + baseB + 8));
                    float B_9 = (*(Bp + baseB + 9));
                    float B_10 = (*(Bp + baseB + 10));
                    float B_11 = (*(Bp + baseB + 11));
                    float B_12 = (*(Bp + baseB + 12));
                    float B_13 = (*(Bp + baseB + 13));
                    float B_14 = (*(Bp + baseB + 14));
                    float B_15 = (*(Bp + baseB + 15));
                    sum0_0 += A_0 * B_0;	sum0_1 += A_1 * B_0;	sum0_2 += A_2 * B_0;
                    sum1_0 += A_0 * B_1;	sum1_1 += A_1 * B_1;	sum1_2 += A_2 * B_1;
                    sum2_0 += A_0 * B_2;	sum2_1 += A_1 * B_2;	sum2_2 += A_2 * B_2;
                    sum3_0 += A_0 * B_3;	sum3_1 += A_1 * B_3;	sum3_2 += A_2 * B_3;
                    sum4_0 += A_0 * B_4;	sum4_1 += A_1 * B_4;	sum4_2 += A_2 * B_4;
                    sum5_0 += A_0 * B_5;	sum5_1 += A_1 * B_5;	sum5_2 += A_2 * B_5;
                    sum6_0 += A_0 * B_6;	sum6_1 += A_1 * B_6;	sum6_2 += A_2 * B_6;
                    sum7_0 += A_0 * B_7;	sum7_1 += A_1 * B_7;	sum7_2 += A_2 * B_7;
                    sum8_0 += A_0 * B_8;	sum8_1 += A_1 * B_8;	sum8_2 += A_2 * B_8;
                    sum9_0 += A_0 * B_9;	sum9_1 += A_1 * B_9;	sum9_2 += A_2 * B_9;
                    sum10_0 += A_0 * B_10;	sum10_1 += A_1 * B_10;	sum10_2 += A_2 * B_10;
                    sum11_0 += A_0 * B_11;	sum11_1 += A_1 * B_11;	sum11_2 += A_2 * B_11;
                    sum12_0 += A_0 * B_12;	sum12_1 += A_1 * B_12;	sum12_2 += A_2 * B_12;
                    sum13_0 += A_0 * B_13;	sum13_1 += A_1 * B_13;	sum13_2 += A_2 * B_13;
                    sum14_0 += A_0 * B_14;	sum14_1 += A_1 * B_14;	sum14_2 += A_2 * B_14;
                    sum15_0 += A_0 * B_15;	sum15_1 += A_1 * B_15;	sum15_2 += A_2 * B_15;
                }
                // 0
                *(Cp + baseC_0 + 0) = sum0_0;
                *(Cp + baseC_0 + 1) = sum1_0;
                *(Cp + baseC_0 + 2) = sum2_0;
                *(Cp + baseC_0 + 3) = sum3_0;
                *(Cp + baseC_0 + 4) = sum4_0;
                *(Cp + baseC_0 + 5) = sum5_0;
                *(Cp + baseC_0 + 6) = sum6_0;
                *(Cp + baseC_0 + 7) = sum7_0;
                *(Cp + baseC_0 + 8) = sum8_0;
                *(Cp + baseC_0 + 9) = sum9_0;
                *(Cp + baseC_0 + 10) = sum10_0;
                *(Cp + baseC_0 + 11) = sum11_0;
                *(Cp + baseC_0 + 12) = sum12_0;
                *(Cp + baseC_0 + 13) = sum13_0;
                *(Cp + baseC_0 + 14) = sum14_0;
                *(Cp + baseC_0 + 15) = sum15_0;
                // 1
                *(Cp + baseC_1 + 0) = sum0_1;
                *(Cp + baseC_1 + 1) = sum1_1;
                *(Cp + baseC_1 + 2) = sum2_1;
                *(Cp + baseC_1 + 3) = sum3_1;
                *(Cp + baseC_1 + 4) = sum4_1;
                *(Cp + baseC_1 + 5) = sum5_1;
                *(Cp + baseC_1 + 6) = sum6_1;
                *(Cp + baseC_1 + 7) = sum7_1;
                *(Cp + baseC_1 + 8) = sum8_1;
                *(Cp + baseC_1 + 9) = sum9_1;
                *(Cp + baseC_1 + 10) = sum10_1;
                *(Cp + baseC_1 + 11) = sum11_1;
                *(Cp + baseC_1 + 12) = sum12_1;
                *(Cp + baseC_1 + 13) = sum13_1;
                *(Cp + baseC_1 + 14) = sum14_1;
                *(Cp + baseC_1 + 15) = sum15_1;
                // 2
                *(Cp + baseC_2 + 0) = sum0_2;
                *(Cp + baseC_2 + 1) = sum1_2;
                *(Cp + baseC_2 + 2) = sum2_2;
                *(Cp + baseC_2 + 3) = sum3_2;
                *(Cp + baseC_2 + 4) = sum4_2;
                *(Cp + baseC_2 + 5) = sum5_2;
                *(Cp + baseC_2 + 6) = sum6_2;
                *(Cp + baseC_2 + 7) = sum7_2;
                *(Cp + baseC_2 + 8) = sum8_2;
                *(Cp + baseC_2 + 9) = sum9_2;
                *(Cp + baseC_2 + 10) = sum10_2;
                *(Cp + baseC_2 + 11) = sum11_2;
                *(Cp + baseC_2 + 12) = sum12_2;
                *(Cp + baseC_2 + 13) = sum13_2;
                *(Cp + baseC_2 + 14) = sum14_2;
                *(Cp + baseC_2 + 15) = sum15_2;
            }
        }
        for (; i < blockSizeM - 1; i += 2)
        {
            var i_0 = i + 0;
            var i_1 = i + 1;

            for (int j = 0; j < n; j += 16)
            {
                int baseC_0 = i_0 * Cstride + j;
                int baseC_1 = i_1 * Cstride + j;
                // 0
                float sum0_0 = *(Cp + baseC_0 + 0);
                float sum1_0 = *(Cp + baseC_0 + 1);
                float sum2_0 = *(Cp + baseC_0 + 2);
                float sum3_0 = *(Cp + baseC_0 + 3);
                float sum4_0 = *(Cp + baseC_0 + 4);
                float sum5_0 = *(Cp + baseC_0 + 5);
                float sum6_0 = *(Cp + baseC_0 + 6);
                float sum7_0 = *(Cp + baseC_0 + 7);
                float sum8_0 = *(Cp + baseC_0 + 8);
                float sum9_0 = *(Cp + baseC_0 + 9);
                float sum10_0 = *(Cp + baseC_0 + 10);
                float sum11_0 = *(Cp + baseC_0 + 11);
                float sum12_0 = *(Cp + baseC_0 + 12);
                float sum13_0 = *(Cp + baseC_0 + 13);
                float sum14_0 = *(Cp + baseC_0 + 14);
                float sum15_0 = *(Cp + baseC_0 + 15);
                // 1
                float sum0_1 = *(Cp + baseC_1 + 0);
                float sum1_1 = *(Cp + baseC_1 + 1);
                float sum2_1 = *(Cp + baseC_1 + 2);
                float sum3_1 = *(Cp + baseC_1 + 3);
                float sum4_1 = *(Cp + baseC_1 + 4);
                float sum5_1 = *(Cp + baseC_1 + 5);
                float sum6_1 = *(Cp + baseC_1 + 6);
                float sum7_1 = *(Cp + baseC_1 + 7);
                float sum8_1 = *(Cp + baseC_1 + 8);
                float sum9_1 = *(Cp + baseC_1 + 9);
                float sum10_1 = *(Cp + baseC_1 + 10);
                float sum11_1 = *(Cp + baseC_1 + 11);
                float sum12_1 = *(Cp + baseC_1 + 12);
                float sum13_1 = *(Cp + baseC_1 + 13);
                float sum14_1 = *(Cp + baseC_1 + 14);
                float sum15_1 = *(Cp + baseC_1 + 15);

                for (int l = 0; l < blockSizeK; l++)
                {
                    float A_0 = *(Ap + i_0 * Astride + l);
                    float A_1 = *(Ap + i_1 * Astride + l);
                    int baseB = l * Bstride + j;
                    float B_0 = (*(Bp + baseB + 0));
                    float B_1 = (*(Bp + baseB + 1));
                    float B_2 = (*(Bp + baseB + 2));
                    float B_3 = (*(Bp + baseB + 3));
                    float B_4 = (*(Bp + baseB + 4));
                    float B_5 = (*(Bp + baseB + 5));
                    float B_6 = (*(Bp + baseB + 6));
                    float B_7 = (*(Bp + baseB + 7));
                    float B_8 = (*(Bp + baseB + 8));
                    float B_9 = (*(Bp + baseB + 9));
                    float B_10 = (*(Bp + baseB + 10));
                    float B_11 = (*(Bp + baseB + 11));
                    float B_12 = (*(Bp + baseB + 12));
                    float B_13 = (*(Bp + baseB + 13));
                    float B_14 = (*(Bp + baseB + 14));
                    float B_15 = (*(Bp + baseB + 15));
                    sum0_0 += A_0 * B_0;	sum0_1 += A_1 * B_0;
                    sum1_0 += A_0 * B_1;	sum1_1 += A_1 * B_1;
                    sum2_0 += A_0 * B_2;	sum2_1 += A_1 * B_2;
                    sum3_0 += A_0 * B_3;	sum3_1 += A_1 * B_3;
                    sum4_0 += A_0 * B_4;	sum4_1 += A_1 * B_4;
                    sum5_0 += A_0 * B_5;	sum5_1 += A_1 * B_5;
                    sum6_0 += A_0 * B_6;	sum6_1 += A_1 * B_6;
                    sum7_0 += A_0 * B_7;	sum7_1 += A_1 * B_7;
                    sum8_0 += A_0 * B_8;	sum8_1 += A_1 * B_8;
                    sum9_0 += A_0 * B_9;	sum9_1 += A_1 * B_9;
                    sum10_0 += A_0 * B_10;	sum10_1 += A_1 * B_10;
                    sum11_0 += A_0 * B_11;	sum11_1 += A_1 * B_11;
                    sum12_0 += A_0 * B_12;	sum12_1 += A_1 * B_12;
                    sum13_0 += A_0 * B_13;	sum13_1 += A_1 * B_13;
                    sum14_0 += A_0 * B_14;	sum14_1 += A_1 * B_14;
                    sum15_0 += A_0 * B_15;	sum15_1 += A_1 * B_15;
                }
                // 0
                *(Cp + baseC_0 + 0) = sum0_0;
                *(Cp + baseC_0 + 1) = sum1_0;
                *(Cp + baseC_0 + 2) = sum2_0;
                *(Cp + baseC_0 + 3) = sum3_0;
                *(Cp + baseC_0 + 4) = sum4_0;
                *(Cp + baseC_0 + 5) = sum5_0;
                *(Cp + baseC_0 + 6) = sum6_0;
                *(Cp + baseC_0 + 7) = sum7_0;
                *(Cp + baseC_0 + 8) = sum8_0;
                *(Cp + baseC_0 + 9) = sum9_0;
                *(Cp + baseC_0 + 10) = sum10_0;
                *(Cp + baseC_0 + 11) = sum11_0;
                *(Cp + baseC_0 + 12) = sum12_0;
                *(Cp + baseC_0 + 13) = sum13_0;
                *(Cp + baseC_0 + 14) = sum14_0;
                *(Cp + baseC_0 + 15) = sum15_0;
                // 1
                *(Cp + baseC_1 + 0) = sum0_1;
                *(Cp + baseC_1 + 1) = sum1_1;
                *(Cp + baseC_1 + 2) = sum2_1;
                *(Cp + baseC_1 + 3) = sum3_1;
                *(Cp + baseC_1 + 4) = sum4_1;
                *(Cp + baseC_1 + 5) = sum5_1;
                *(Cp + baseC_1 + 6) = sum6_1;
                *(Cp + baseC_1 + 7) = sum7_1;
                *(Cp + baseC_1 + 8) = sum8_1;
                *(Cp + baseC_1 + 9) = sum9_1;
                *(Cp + baseC_1 + 10) = sum10_1;
                *(Cp + baseC_1 + 11) = sum11_1;
                *(Cp + baseC_1 + 12) = sum12_1;
                *(Cp + baseC_1 + 13) = sum13_1;
                *(Cp + baseC_1 + 14) = sum14_1;
                *(Cp + baseC_1 + 15) = sum15_1;
            }
        }
        for (; i < blockSizeM - 0; i += 1)
        {
            var i_0 = i + 0;

            for (int j = 0; j < n; j += 16)
            {
                int baseC_0 = i_0 * Cstride + j;
                // 0
                float sum0_0 = *(Cp + baseC_0 + 0);
                float sum1_0 = *(Cp + baseC_0 + 1);
                float sum2_0 = *(Cp + baseC_0 + 2);
                float sum3_0 = *(Cp + baseC_0 + 3);
                float sum4_0 = *(Cp + baseC_0 + 4);
                float sum5_0 = *(Cp + baseC_0 + 5);
                float sum6_0 = *(Cp + baseC_0 + 6);
                float sum7_0 = *(Cp + baseC_0 + 7);
                float sum8_0 = *(Cp + baseC_0 + 8);
                float sum9_0 = *(Cp + baseC_0 + 9);
                float sum10_0 = *(Cp + baseC_0 + 10);
                float sum11_0 = *(Cp + baseC_0 + 11);
                float sum12_0 = *(Cp + baseC_0 + 12);
                float sum13_0 = *(Cp + baseC_0 + 13);
                float sum14_0 = *(Cp + baseC_0 + 14);
                float sum15_0 = *(Cp + baseC_0 + 15);

                for (int l = 0; l < blockSizeK; l++)
                {
                    float A_0 = *(Ap + i_0 * Astride + l);
                    int baseB = l * Bstride + j;
                    float B_0 = (*(Bp + baseB + 0));
                    float B_1 = (*(Bp + baseB + 1));
                    float B_2 = (*(Bp + baseB + 2));
                    float B_3 = (*(Bp + baseB + 3));
                    float B_4 = (*(Bp + baseB + 4));
                    float B_5 = (*(Bp + baseB + 5));
                    float B_6 = (*(Bp + baseB + 6));
                    float B_7 = (*(Bp + baseB + 7));
                    float B_8 = (*(Bp + baseB + 8));
                    float B_9 = (*(Bp + baseB + 9));
                    float B_10 = (*(Bp + baseB + 10));
                    float B_11 = (*(Bp + baseB + 11));
                    float B_12 = (*(Bp + baseB + 12));
                    float B_13 = (*(Bp + baseB + 13));
                    float B_14 = (*(Bp + baseB + 14));
                    float B_15 = (*(Bp + baseB + 15));
                    sum0_0 += A_0 * B_0;
                    sum1_0 += A_0 * B_1;
                    sum2_0 += A_0 * B_2;
                    sum3_0 += A_0 * B_3;
                    sum4_0 += A_0 * B_4;
                    sum5_0 += A_0 * B_5;
                    sum6_0 += A_0 * B_6;
                    sum7_0 += A_0 * B_7;
                    sum8_0 += A_0 * B_8;
                    sum9_0 += A_0 * B_9;
                    sum10_0 += A_0 * B_10;
                    sum11_0 += A_0 * B_11;
                    sum12_0 += A_0 * B_12;
                    sum13_0 += A_0 * B_13;
                    sum14_0 += A_0 * B_14;
                    sum15_0 += A_0 * B_15;
                }
                // 0
                *(Cp + baseC_0 + 0) = sum0_0;
                *(Cp + baseC_0 + 1) = sum1_0;
                *(Cp + baseC_0 + 2) = sum2_0;
                *(Cp + baseC_0 + 3) = sum3_0;
                *(Cp + baseC_0 + 4) = sum4_0;
                *(Cp + baseC_0 + 5) = sum5_0;
                *(Cp + baseC_0 + 6) = sum6_0;
                *(Cp + baseC_0 + 7) = sum7_0;
                *(Cp + baseC_0 + 8) = sum8_0;
                *(Cp + baseC_0 + 9) = sum9_0;
                *(Cp + baseC_0 + 10) = sum10_0;
                *(Cp + baseC_0 + 11) = sum11_0;
                *(Cp + baseC_0 + 12) = sum12_0;
                *(Cp + baseC_0 + 13) = sum13_0;
                *(Cp + baseC_0 + 14) = sum14_0;
                *(Cp + baseC_0 + 15) = sum15_0;
            }
        }
    }

    static unsafe void MultiplyBlockUnroll3x16I(
        [NoAlias] float* Ap, int Astride,
        [NoAlias] float* Bp, int Bstride,
        [NoAlias] float* Cp, int Cstride,
        int blockSizeM, int blockSizeK,
        int n)
    {
        n = Math.Max(16, n);
        int i = 0;
        for (; i < blockSizeM - 2; i += 3)
        {
            var i_0 = i + 0;
            var i_1 = i + 1;
            var i_2 = i + 2;

            for (int j = 0; j < n; j += 16)
            {
                int baseC_0 = i_0 * Cstride + j;
                int baseC_1 = i_1 * Cstride + j;
                int baseC_2 = i_2 * Cstride + j;

                // row 0
                v256 gamma_0_0 = mm256_loadu_ps(Cp + baseC_0 + 0);
                v256 gamma_0_8 = mm256_loadu_ps(Cp + baseC_0 + 8);
                // row 1
                v256 gamma_1_0 = mm256_loadu_ps(Cp + baseC_1 + 0);
                v256 gamma_1_8 = mm256_loadu_ps(Cp + baseC_1 + 8);
                // row 2
                v256 gamma_2_0 = mm256_loadu_ps(Cp + baseC_2 + 0);
                v256 gamma_2_8 = mm256_loadu_ps(Cp + baseC_2 + 8);

                for (int l = 0; l < blockSizeK; l++)
                {
                    v256 alpha_0_p = mm256_broadcast_ss(Ap + i_0 * Astride + l);
                    v256 alpha_1_p = mm256_broadcast_ss(Ap + i_1 * Astride + l);
                    v256 alpha_2_p = mm256_broadcast_ss(Ap + i_2 * Astride + l);

                    v256 beta_p_0 = mm256_loadu_ps(Bp + l * Bstride + j + 0);
                    v256 beta_p_8 = mm256_loadu_ps(Bp + l * Bstride + j + 8);

                    gamma_0_0 = mm256_fmadd_ps(alpha_0_p, beta_p_0, gamma_0_0);
                    gamma_1_0 = mm256_fmadd_ps(alpha_1_p, beta_p_0, gamma_1_0);
                    gamma_2_0 = mm256_fmadd_ps(alpha_2_p, beta_p_0, gamma_2_0);
                    gamma_0_8 = mm256_fmadd_ps(alpha_0_p, beta_p_8, gamma_0_8);
                    gamma_1_8 = mm256_fmadd_ps(alpha_1_p, beta_p_8, gamma_1_8);
                    gamma_2_8 = mm256_fmadd_ps(alpha_2_p, beta_p_8, gamma_2_8);
                }
                // row 0
                mm256_storeu_ps(Cp + baseC_0 + 0, gamma_0_0);
                mm256_storeu_ps(Cp + baseC_0 + 8, gamma_0_8);
                // row 1
                mm256_storeu_ps(Cp + baseC_1 + 0, gamma_1_0);
                mm256_storeu_ps(Cp + baseC_1 + 8, gamma_1_8);
                // row 2
                mm256_storeu_ps(Cp + baseC_2 + 0, gamma_2_0);
                mm256_storeu_ps(Cp + baseC_2 + 8, gamma_2_8);
            }
        }
        for (; i < blockSizeM - 1; i += 2)
        {
            var i_0 = i + 0;
            var i_1 = i + 1;

            for (int j = 0; j < n; j += 16)
            {
                int baseC_0 = i_0 * Cstride + j;
                int baseC_1 = i_1 * Cstride + j;

                // row 0
                v256 gamma_0_0 = mm256_loadu_ps(Cp + baseC_0 + 0);
                v256 gamma_0_8 = mm256_loadu_ps(Cp + baseC_0 + 8);
                // row 1
                v256 gamma_1_0 = mm256_loadu_ps(Cp + baseC_1 + 0);
                v256 gamma_1_8 = mm256_loadu_ps(Cp + baseC_1 + 8);

                for (int l = 0; l < blockSizeK; l++)
                {
                    v256 alpha_0_p = mm256_broadcast_ss(Ap + i_0 * Astride + l);
                    v256 alpha_1_p = mm256_broadcast_ss(Ap + i_1 * Astride + l);

                    v256 beta_p_0 = mm256_loadu_ps(Bp + l * Bstride + j + 0);
                    v256 beta_p_8 = mm256_loadu_ps(Bp + l * Bstride + j + 8);

                    gamma_0_0 = mm256_fmadd_ps(alpha_0_p, beta_p_0, gamma_0_0);
                    gamma_1_0 = mm256_fmadd_ps(alpha_1_p, beta_p_0, gamma_1_0);
                    gamma_0_8 = mm256_fmadd_ps(alpha_0_p, beta_p_8, gamma_0_8);
                    gamma_1_8 = mm256_fmadd_ps(alpha_1_p, beta_p_8, gamma_1_8);
                }
                // row 0
                mm256_storeu_ps(Cp + baseC_0 + 0, gamma_0_0);
                mm256_storeu_ps(Cp + baseC_0 + 8, gamma_0_8);
                // row 1
                mm256_storeu_ps(Cp + baseC_1 + 0, gamma_1_0);
                mm256_storeu_ps(Cp + baseC_1 + 8, gamma_1_8);
            }
        }
        for (; i < blockSizeM - 0; i += 1)
        {
            var i_0 = i + 0;

            for (int j = 0; j < n; j += 16)
            {
                int baseC_0 = i_0 * Cstride + j;

                // row 0
                v256 gamma_0_0 = mm256_loadu_ps(Cp + baseC_0 + 0);
                v256 gamma_0_8 = mm256_loadu_ps(Cp + baseC_0 + 8);

                for (int l = 0; l < blockSizeK; l++)
                {
                    v256 alpha_0_p = mm256_broadcast_ss(Ap + i_0 * Astride + l);

                    v256 beta_p_0 = mm256_loadu_ps(Bp + l * Bstride + j + 0);
                    v256 beta_p_8 = mm256_loadu_ps(Bp + l * Bstride + j + 8);

                    gamma_0_0 = mm256_fmadd_ps(alpha_0_p, beta_p_0, gamma_0_0);
                    gamma_0_8 = mm256_fmadd_ps(alpha_0_p, beta_p_8, gamma_0_8);
                }
                // row 0
                mm256_storeu_ps(Cp + baseC_0 + 0, gamma_0_0);
                mm256_storeu_ps(Cp + baseC_0 + 8, gamma_0_8);
            }
        }
    }

    static unsafe void MultiplyBlockUnroll3x24(
            [NoAlias] float* Ap, int Astride,
            [NoAlias] float* Bp, int Bstride,
            [NoAlias] float* Cp, int Cstride,
            int blockSizeM, int blockSizeK,
            int n)
    {
        n = Math.Max(24, n);
        int i = 0;
        for (; i < blockSizeM - 2; i += 3)
        {
            var i_0 = i + 0;
            var i_1 = i + 1;
            var i_2 = i + 2;

            for (int j = 0; j < n; j += 24)
            {
                int baseC_0 = i_0 * Cstride + j;
                int baseC_1 = i_1 * Cstride + j;
                int baseC_2 = i_2 * Cstride + j;
                // 0
                float sum0_0 = *(Cp + baseC_0 + 0);
                float sum1_0 = *(Cp + baseC_0 + 1);
                float sum2_0 = *(Cp + baseC_0 + 2);
                float sum3_0 = *(Cp + baseC_0 + 3);
                float sum4_0 = *(Cp + baseC_0 + 4);
                float sum5_0 = *(Cp + baseC_0 + 5);
                float sum6_0 = *(Cp + baseC_0 + 6);
                float sum7_0 = *(Cp + baseC_0 + 7);
                float sum8_0 = *(Cp + baseC_0 + 8);
                float sum9_0 = *(Cp + baseC_0 + 9);
                float sum10_0 = *(Cp + baseC_0 + 10);
                float sum11_0 = *(Cp + baseC_0 + 11);
                float sum12_0 = *(Cp + baseC_0 + 12);
                float sum13_0 = *(Cp + baseC_0 + 13);
                float sum14_0 = *(Cp + baseC_0 + 14);
                float sum15_0 = *(Cp + baseC_0 + 15);
                float sum16_0 = *(Cp + baseC_0 + 16);
                float sum17_0 = *(Cp + baseC_0 + 17);
                float sum18_0 = *(Cp + baseC_0 + 18);
                float sum19_0 = *(Cp + baseC_0 + 19);
                float sum20_0 = *(Cp + baseC_0 + 20);
                float sum21_0 = *(Cp + baseC_0 + 21);
                float sum22_0 = *(Cp + baseC_0 + 22);
                float sum23_0 = *(Cp + baseC_0 + 23);
                // 1
                float sum0_1 = *(Cp + baseC_1 + 0);
                float sum1_1 = *(Cp + baseC_1 + 1);
                float sum2_1 = *(Cp + baseC_1 + 2);
                float sum3_1 = *(Cp + baseC_1 + 3);
                float sum4_1 = *(Cp + baseC_1 + 4);
                float sum5_1 = *(Cp + baseC_1 + 5);
                float sum6_1 = *(Cp + baseC_1 + 6);
                float sum7_1 = *(Cp + baseC_1 + 7);
                float sum8_1 = *(Cp + baseC_1 + 8);
                float sum9_1 = *(Cp + baseC_1 + 9);
                float sum10_1 = *(Cp + baseC_1 + 10);
                float sum11_1 = *(Cp + baseC_1 + 11);
                float sum12_1 = *(Cp + baseC_1 + 12);
                float sum13_1 = *(Cp + baseC_1 + 13);
                float sum14_1 = *(Cp + baseC_1 + 14);
                float sum15_1 = *(Cp + baseC_1 + 15);
                float sum16_1 = *(Cp + baseC_1 + 16);
                float sum17_1 = *(Cp + baseC_1 + 17);
                float sum18_1 = *(Cp + baseC_1 + 18);
                float sum19_1 = *(Cp + baseC_1 + 19);
                float sum20_1 = *(Cp + baseC_1 + 20);
                float sum21_1 = *(Cp + baseC_1 + 21);
                float sum22_1 = *(Cp + baseC_1 + 22);
                float sum23_1 = *(Cp + baseC_1 + 23);
                // 2
                float sum0_2 = *(Cp + baseC_2 + 0);
                float sum1_2 = *(Cp + baseC_2 + 1);
                float sum2_2 = *(Cp + baseC_2 + 2);
                float sum3_2 = *(Cp + baseC_2 + 3);
                float sum4_2 = *(Cp + baseC_2 + 4);
                float sum5_2 = *(Cp + baseC_2 + 5);
                float sum6_2 = *(Cp + baseC_2 + 6);
                float sum7_2 = *(Cp + baseC_2 + 7);
                float sum8_2 = *(Cp + baseC_2 + 8);
                float sum9_2 = *(Cp + baseC_2 + 9);
                float sum10_2 = *(Cp + baseC_2 + 10);
                float sum11_2 = *(Cp + baseC_2 + 11);
                float sum12_2 = *(Cp + baseC_2 + 12);
                float sum13_2 = *(Cp + baseC_2 + 13);
                float sum14_2 = *(Cp + baseC_2 + 14);
                float sum15_2 = *(Cp + baseC_2 + 15);
                float sum16_2 = *(Cp + baseC_2 + 16);
                float sum17_2 = *(Cp + baseC_2 + 17);
                float sum18_2 = *(Cp + baseC_2 + 18);
                float sum19_2 = *(Cp + baseC_2 + 19);
                float sum20_2 = *(Cp + baseC_2 + 20);
                float sum21_2 = *(Cp + baseC_2 + 21);
                float sum22_2 = *(Cp + baseC_2 + 22);
                float sum23_2 = *(Cp + baseC_2 + 23);

                for (int l = 0; l < blockSizeK; l++)
                {
                    float A_0 = *(Ap + i_0 * Astride + l);
                    float A_1 = *(Ap + i_1 * Astride + l);
                    float A_2 = *(Ap + i_2 * Astride + l);
                    int baseB = l * Bstride + j;
                    float B_0 = (*(Bp + baseB + 0));
                    float B_1 = (*(Bp + baseB + 1));
                    float B_2 = (*(Bp + baseB + 2));
                    float B_3 = (*(Bp + baseB + 3));
                    float B_4 = (*(Bp + baseB + 4));
                    float B_5 = (*(Bp + baseB + 5));
                    float B_6 = (*(Bp + baseB + 6));
                    float B_7 = (*(Bp + baseB + 7));
                    float B_8 = (*(Bp + baseB + 8));
                    float B_9 = (*(Bp + baseB + 9));
                    float B_10 = (*(Bp + baseB + 10));
                    float B_11 = (*(Bp + baseB + 11));
                    float B_12 = (*(Bp + baseB + 12));
                    float B_13 = (*(Bp + baseB + 13));
                    float B_14 = (*(Bp + baseB + 14));
                    float B_15 = (*(Bp + baseB + 15));
                    float B_16 = (*(Bp + baseB + 16));
                    float B_17 = (*(Bp + baseB + 17));
                    float B_18 = (*(Bp + baseB + 18));
                    float B_19 = (*(Bp + baseB + 19));
                    float B_20 = (*(Bp + baseB + 20));
                    float B_21 = (*(Bp + baseB + 21));
                    float B_22 = (*(Bp + baseB + 22));
                    float B_23 = (*(Bp + baseB + 23));
                    sum0_0 += A_0 * B_0;	sum0_1 += A_1 * B_0;	sum0_2 += A_2 * B_0;
                    sum1_0 += A_0 * B_1;	sum1_1 += A_1 * B_1;	sum1_2 += A_2 * B_1;
                    sum2_0 += A_0 * B_2;	sum2_1 += A_1 * B_2;	sum2_2 += A_2 * B_2;
                    sum3_0 += A_0 * B_3;	sum3_1 += A_1 * B_3;	sum3_2 += A_2 * B_3;
                    sum4_0 += A_0 * B_4;	sum4_1 += A_1 * B_4;	sum4_2 += A_2 * B_4;
                    sum5_0 += A_0 * B_5;	sum5_1 += A_1 * B_5;	sum5_2 += A_2 * B_5;
                    sum6_0 += A_0 * B_6;	sum6_1 += A_1 * B_6;	sum6_2 += A_2 * B_6;
                    sum7_0 += A_0 * B_7;	sum7_1 += A_1 * B_7;	sum7_2 += A_2 * B_7;
                    sum8_0 += A_0 * B_8;	sum8_1 += A_1 * B_8;	sum8_2 += A_2 * B_8;
                    sum9_0 += A_0 * B_9;	sum9_1 += A_1 * B_9;	sum9_2 += A_2 * B_9;
                    sum10_0 += A_0 * B_10;	sum10_1 += A_1 * B_10;	sum10_2 += A_2 * B_10;
                    sum11_0 += A_0 * B_11;	sum11_1 += A_1 * B_11;	sum11_2 += A_2 * B_11;
                    sum12_0 += A_0 * B_12;	sum12_1 += A_1 * B_12;	sum12_2 += A_2 * B_12;
                    sum13_0 += A_0 * B_13;	sum13_1 += A_1 * B_13;	sum13_2 += A_2 * B_13;
                    sum14_0 += A_0 * B_14;	sum14_1 += A_1 * B_14;	sum14_2 += A_2 * B_14;
                    sum15_0 += A_0 * B_15;	sum15_1 += A_1 * B_15;	sum15_2 += A_2 * B_15;
                    sum16_0 += A_0 * B_16;	sum16_1 += A_1 * B_16;	sum16_2 += A_2 * B_16;
                    sum17_0 += A_0 * B_17;	sum17_1 += A_1 * B_17;	sum17_2 += A_2 * B_17;
                    sum18_0 += A_0 * B_18;	sum18_1 += A_1 * B_18;	sum18_2 += A_2 * B_18;
                    sum19_0 += A_0 * B_19;	sum19_1 += A_1 * B_19;	sum19_2 += A_2 * B_19;
                    sum20_0 += A_0 * B_20;	sum20_1 += A_1 * B_20;	sum20_2 += A_2 * B_20;
                    sum21_0 += A_0 * B_21;	sum21_1 += A_1 * B_21;	sum21_2 += A_2 * B_21;
                    sum22_0 += A_0 * B_22;	sum22_1 += A_1 * B_22;	sum22_2 += A_2 * B_22;
                    sum23_0 += A_0 * B_23;	sum23_1 += A_1 * B_23;	sum23_2 += A_2 * B_23;
                }
                // 0
                *(Cp + baseC_0 + 0) = sum0_0;
                *(Cp + baseC_0 + 1) = sum1_0;
                *(Cp + baseC_0 + 2) = sum2_0;
                *(Cp + baseC_0 + 3) = sum3_0;
                *(Cp + baseC_0 + 4) = sum4_0;
                *(Cp + baseC_0 + 5) = sum5_0;
                *(Cp + baseC_0 + 6) = sum6_0;
                *(Cp + baseC_0 + 7) = sum7_0;
                *(Cp + baseC_0 + 8) = sum8_0;
                *(Cp + baseC_0 + 9) = sum9_0;
                *(Cp + baseC_0 + 10) = sum10_0;
                *(Cp + baseC_0 + 11) = sum11_0;
                *(Cp + baseC_0 + 12) = sum12_0;
                *(Cp + baseC_0 + 13) = sum13_0;
                *(Cp + baseC_0 + 14) = sum14_0;
                *(Cp + baseC_0 + 15) = sum15_0;
                *(Cp + baseC_0 + 16) = sum16_0;
                *(Cp + baseC_0 + 17) = sum17_0;
                *(Cp + baseC_0 + 18) = sum18_0;
                *(Cp + baseC_0 + 19) = sum19_0;
                *(Cp + baseC_0 + 20) = sum20_0;
                *(Cp + baseC_0 + 21) = sum21_0;
                *(Cp + baseC_0 + 22) = sum22_0;
                *(Cp + baseC_0 + 23) = sum23_0;
                // 1
                *(Cp + baseC_1 + 0) = sum0_1;
                *(Cp + baseC_1 + 1) = sum1_1;
                *(Cp + baseC_1 + 2) = sum2_1;
                *(Cp + baseC_1 + 3) = sum3_1;
                *(Cp + baseC_1 + 4) = sum4_1;
                *(Cp + baseC_1 + 5) = sum5_1;
                *(Cp + baseC_1 + 6) = sum6_1;
                *(Cp + baseC_1 + 7) = sum7_1;
                *(Cp + baseC_1 + 8) = sum8_1;
                *(Cp + baseC_1 + 9) = sum9_1;
                *(Cp + baseC_1 + 10) = sum10_1;
                *(Cp + baseC_1 + 11) = sum11_1;
                *(Cp + baseC_1 + 12) = sum12_1;
                *(Cp + baseC_1 + 13) = sum13_1;
                *(Cp + baseC_1 + 14) = sum14_1;
                *(Cp + baseC_1 + 15) = sum15_1;
                *(Cp + baseC_1 + 16) = sum16_1;
                *(Cp + baseC_1 + 17) = sum17_1;
                *(Cp + baseC_1 + 18) = sum18_1;
                *(Cp + baseC_1 + 19) = sum19_1;
                *(Cp + baseC_1 + 20) = sum20_1;
                *(Cp + baseC_1 + 21) = sum21_1;
                *(Cp + baseC_1 + 22) = sum22_1;
                *(Cp + baseC_1 + 23) = sum23_1;
                // 2
                *(Cp + baseC_2 + 0) = sum0_2;
                *(Cp + baseC_2 + 1) = sum1_2;
                *(Cp + baseC_2 + 2) = sum2_2;
                *(Cp + baseC_2 + 3) = sum3_2;
                *(Cp + baseC_2 + 4) = sum4_2;
                *(Cp + baseC_2 + 5) = sum5_2;
                *(Cp + baseC_2 + 6) = sum6_2;
                *(Cp + baseC_2 + 7) = sum7_2;
                *(Cp + baseC_2 + 8) = sum8_2;
                *(Cp + baseC_2 + 9) = sum9_2;
                *(Cp + baseC_2 + 10) = sum10_2;
                *(Cp + baseC_2 + 11) = sum11_2;
                *(Cp + baseC_2 + 12) = sum12_2;
                *(Cp + baseC_2 + 13) = sum13_2;
                *(Cp + baseC_2 + 14) = sum14_2;
                *(Cp + baseC_2 + 15) = sum15_2;
                *(Cp + baseC_2 + 16) = sum16_2;
                *(Cp + baseC_2 + 17) = sum17_2;
                *(Cp + baseC_2 + 18) = sum18_2;
                *(Cp + baseC_2 + 19) = sum19_2;
                *(Cp + baseC_2 + 20) = sum20_2;
                *(Cp + baseC_2 + 21) = sum21_2;
                *(Cp + baseC_2 + 22) = sum22_2;
                *(Cp + baseC_2 + 23) = sum23_2;
            }
        }
        for (; i < blockSizeM - 1; i += 2)
        {
            var i_0 = i + 0;
            var i_1 = i + 1;

            for (int j = 0; j < n; j += 24)
            {
                int baseC_0 = i_0 * Cstride + j;
                int baseC_1 = i_1 * Cstride + j;
                // 0
                float sum0_0 = *(Cp + baseC_0 + 0);
                float sum1_0 = *(Cp + baseC_0 + 1);
                float sum2_0 = *(Cp + baseC_0 + 2);
                float sum3_0 = *(Cp + baseC_0 + 3);
                float sum4_0 = *(Cp + baseC_0 + 4);
                float sum5_0 = *(Cp + baseC_0 + 5);
                float sum6_0 = *(Cp + baseC_0 + 6);
                float sum7_0 = *(Cp + baseC_0 + 7);
                float sum8_0 = *(Cp + baseC_0 + 8);
                float sum9_0 = *(Cp + baseC_0 + 9);
                float sum10_0 = *(Cp + baseC_0 + 10);
                float sum11_0 = *(Cp + baseC_0 + 11);
                float sum12_0 = *(Cp + baseC_0 + 12);
                float sum13_0 = *(Cp + baseC_0 + 13);
                float sum14_0 = *(Cp + baseC_0 + 14);
                float sum15_0 = *(Cp + baseC_0 + 15);
                float sum16_0 = *(Cp + baseC_0 + 16);
                float sum17_0 = *(Cp + baseC_0 + 17);
                float sum18_0 = *(Cp + baseC_0 + 18);
                float sum19_0 = *(Cp + baseC_0 + 19);
                float sum20_0 = *(Cp + baseC_0 + 20);
                float sum21_0 = *(Cp + baseC_0 + 21);
                float sum22_0 = *(Cp + baseC_0 + 22);
                float sum23_0 = *(Cp + baseC_0 + 23);
                // 1
                float sum0_1 = *(Cp + baseC_1 + 0);
                float sum1_1 = *(Cp + baseC_1 + 1);
                float sum2_1 = *(Cp + baseC_1 + 2);
                float sum3_1 = *(Cp + baseC_1 + 3);
                float sum4_1 = *(Cp + baseC_1 + 4);
                float sum5_1 = *(Cp + baseC_1 + 5);
                float sum6_1 = *(Cp + baseC_1 + 6);
                float sum7_1 = *(Cp + baseC_1 + 7);
                float sum8_1 = *(Cp + baseC_1 + 8);
                float sum9_1 = *(Cp + baseC_1 + 9);
                float sum10_1 = *(Cp + baseC_1 + 10);
                float sum11_1 = *(Cp + baseC_1 + 11);
                float sum12_1 = *(Cp + baseC_1 + 12);
                float sum13_1 = *(Cp + baseC_1 + 13);
                float sum14_1 = *(Cp + baseC_1 + 14);
                float sum15_1 = *(Cp + baseC_1 + 15);
                float sum16_1 = *(Cp + baseC_1 + 16);
                float sum17_1 = *(Cp + baseC_1 + 17);
                float sum18_1 = *(Cp + baseC_1 + 18);
                float sum19_1 = *(Cp + baseC_1 + 19);
                float sum20_1 = *(Cp + baseC_1 + 20);
                float sum21_1 = *(Cp + baseC_1 + 21);
                float sum22_1 = *(Cp + baseC_1 + 22);
                float sum23_1 = *(Cp + baseC_1 + 23);

                for (int l = 0; l < blockSizeK; l++)
                {
                    float A_0 = *(Ap + i_0 * Astride + l);
                    float A_1 = *(Ap + i_1 * Astride + l);
                    int baseB = l * Bstride + j;
                    float B_0 = (*(Bp + baseB + 0));
                    float B_1 = (*(Bp + baseB + 1));
                    float B_2 = (*(Bp + baseB + 2));
                    float B_3 = (*(Bp + baseB + 3));
                    float B_4 = (*(Bp + baseB + 4));
                    float B_5 = (*(Bp + baseB + 5));
                    float B_6 = (*(Bp + baseB + 6));
                    float B_7 = (*(Bp + baseB + 7));
                    float B_8 = (*(Bp + baseB + 8));
                    float B_9 = (*(Bp + baseB + 9));
                    float B_10 = (*(Bp + baseB + 10));
                    float B_11 = (*(Bp + baseB + 11));
                    float B_12 = (*(Bp + baseB + 12));
                    float B_13 = (*(Bp + baseB + 13));
                    float B_14 = (*(Bp + baseB + 14));
                    float B_15 = (*(Bp + baseB + 15));
                    float B_16 = (*(Bp + baseB + 16));
                    float B_17 = (*(Bp + baseB + 17));
                    float B_18 = (*(Bp + baseB + 18));
                    float B_19 = (*(Bp + baseB + 19));
                    float B_20 = (*(Bp + baseB + 20));
                    float B_21 = (*(Bp + baseB + 21));
                    float B_22 = (*(Bp + baseB + 22));
                    float B_23 = (*(Bp + baseB + 23));
                    sum0_0 += A_0 * B_0;	sum0_1 += A_1 * B_0;
                    sum1_0 += A_0 * B_1;	sum1_1 += A_1 * B_1;
                    sum2_0 += A_0 * B_2;	sum2_1 += A_1 * B_2;
                    sum3_0 += A_0 * B_3;	sum3_1 += A_1 * B_3;
                    sum4_0 += A_0 * B_4;	sum4_1 += A_1 * B_4;
                    sum5_0 += A_0 * B_5;	sum5_1 += A_1 * B_5;
                    sum6_0 += A_0 * B_6;	sum6_1 += A_1 * B_6;
                    sum7_0 += A_0 * B_7;	sum7_1 += A_1 * B_7;
                    sum8_0 += A_0 * B_8;	sum8_1 += A_1 * B_8;
                    sum9_0 += A_0 * B_9;	sum9_1 += A_1 * B_9;
                    sum10_0 += A_0 * B_10;	sum10_1 += A_1 * B_10;
                    sum11_0 += A_0 * B_11;	sum11_1 += A_1 * B_11;
                    sum12_0 += A_0 * B_12;	sum12_1 += A_1 * B_12;
                    sum13_0 += A_0 * B_13;	sum13_1 += A_1 * B_13;
                    sum14_0 += A_0 * B_14;	sum14_1 += A_1 * B_14;
                    sum15_0 += A_0 * B_15;	sum15_1 += A_1 * B_15;
                    sum16_0 += A_0 * B_16;	sum16_1 += A_1 * B_16;
                    sum17_0 += A_0 * B_17;	sum17_1 += A_1 * B_17;
                    sum18_0 += A_0 * B_18;	sum18_1 += A_1 * B_18;
                    sum19_0 += A_0 * B_19;	sum19_1 += A_1 * B_19;
                    sum20_0 += A_0 * B_20;	sum20_1 += A_1 * B_20;
                    sum21_0 += A_0 * B_21;	sum21_1 += A_1 * B_21;
                    sum22_0 += A_0 * B_22;	sum22_1 += A_1 * B_22;
                    sum23_0 += A_0 * B_23;	sum23_1 += A_1 * B_23;
                }
                // 0
                *(Cp + baseC_0 + 0) = sum0_0;
                *(Cp + baseC_0 + 1) = sum1_0;
                *(Cp + baseC_0 + 2) = sum2_0;
                *(Cp + baseC_0 + 3) = sum3_0;
                *(Cp + baseC_0 + 4) = sum4_0;
                *(Cp + baseC_0 + 5) = sum5_0;
                *(Cp + baseC_0 + 6) = sum6_0;
                *(Cp + baseC_0 + 7) = sum7_0;
                *(Cp + baseC_0 + 8) = sum8_0;
                *(Cp + baseC_0 + 9) = sum9_0;
                *(Cp + baseC_0 + 10) = sum10_0;
                *(Cp + baseC_0 + 11) = sum11_0;
                *(Cp + baseC_0 + 12) = sum12_0;
                *(Cp + baseC_0 + 13) = sum13_0;
                *(Cp + baseC_0 + 14) = sum14_0;
                *(Cp + baseC_0 + 15) = sum15_0;
                *(Cp + baseC_0 + 16) = sum16_0;
                *(Cp + baseC_0 + 17) = sum17_0;
                *(Cp + baseC_0 + 18) = sum18_0;
                *(Cp + baseC_0 + 19) = sum19_0;
                *(Cp + baseC_0 + 20) = sum20_0;
                *(Cp + baseC_0 + 21) = sum21_0;
                *(Cp + baseC_0 + 22) = sum22_0;
                *(Cp + baseC_0 + 23) = sum23_0;
                // 1
                *(Cp + baseC_1 + 0) = sum0_1;
                *(Cp + baseC_1 + 1) = sum1_1;
                *(Cp + baseC_1 + 2) = sum2_1;
                *(Cp + baseC_1 + 3) = sum3_1;
                *(Cp + baseC_1 + 4) = sum4_1;
                *(Cp + baseC_1 + 5) = sum5_1;
                *(Cp + baseC_1 + 6) = sum6_1;
                *(Cp + baseC_1 + 7) = sum7_1;
                *(Cp + baseC_1 + 8) = sum8_1;
                *(Cp + baseC_1 + 9) = sum9_1;
                *(Cp + baseC_1 + 10) = sum10_1;
                *(Cp + baseC_1 + 11) = sum11_1;
                *(Cp + baseC_1 + 12) = sum12_1;
                *(Cp + baseC_1 + 13) = sum13_1;
                *(Cp + baseC_1 + 14) = sum14_1;
                *(Cp + baseC_1 + 15) = sum15_1;
                *(Cp + baseC_1 + 16) = sum16_1;
                *(Cp + baseC_1 + 17) = sum17_1;
                *(Cp + baseC_1 + 18) = sum18_1;
                *(Cp + baseC_1 + 19) = sum19_1;
                *(Cp + baseC_1 + 20) = sum20_1;
                *(Cp + baseC_1 + 21) = sum21_1;
                *(Cp + baseC_1 + 22) = sum22_1;
                *(Cp + baseC_1 + 23) = sum23_1;
            }
        }
        for (; i < blockSizeM - 0; i += 1)
        {
            var i_0 = i + 0;

            for (int j = 0; j < n; j += 24)
            {
                int baseC_0 = i_0 * Cstride + j;
                // 0
                float sum0_0 = *(Cp + baseC_0 + 0);
                float sum1_0 = *(Cp + baseC_0 + 1);
                float sum2_0 = *(Cp + baseC_0 + 2);
                float sum3_0 = *(Cp + baseC_0 + 3);
                float sum4_0 = *(Cp + baseC_0 + 4);
                float sum5_0 = *(Cp + baseC_0 + 5);
                float sum6_0 = *(Cp + baseC_0 + 6);
                float sum7_0 = *(Cp + baseC_0 + 7);
                float sum8_0 = *(Cp + baseC_0 + 8);
                float sum9_0 = *(Cp + baseC_0 + 9);
                float sum10_0 = *(Cp + baseC_0 + 10);
                float sum11_0 = *(Cp + baseC_0 + 11);
                float sum12_0 = *(Cp + baseC_0 + 12);
                float sum13_0 = *(Cp + baseC_0 + 13);
                float sum14_0 = *(Cp + baseC_0 + 14);
                float sum15_0 = *(Cp + baseC_0 + 15);
                float sum16_0 = *(Cp + baseC_0 + 16);
                float sum17_0 = *(Cp + baseC_0 + 17);
                float sum18_0 = *(Cp + baseC_0 + 18);
                float sum19_0 = *(Cp + baseC_0 + 19);
                float sum20_0 = *(Cp + baseC_0 + 20);
                float sum21_0 = *(Cp + baseC_0 + 21);
                float sum22_0 = *(Cp + baseC_0 + 22);
                float sum23_0 = *(Cp + baseC_0 + 23);

                for (int l = 0; l < blockSizeK; l++)
                {
                    float A_0 = *(Ap + i_0 * Astride + l);
                    int baseB = l * Bstride + j;
                    float B_0 = (*(Bp + baseB + 0));
                    float B_1 = (*(Bp + baseB + 1));
                    float B_2 = (*(Bp + baseB + 2));
                    float B_3 = (*(Bp + baseB + 3));
                    float B_4 = (*(Bp + baseB + 4));
                    float B_5 = (*(Bp + baseB + 5));
                    float B_6 = (*(Bp + baseB + 6));
                    float B_7 = (*(Bp + baseB + 7));
                    float B_8 = (*(Bp + baseB + 8));
                    float B_9 = (*(Bp + baseB + 9));
                    float B_10 = (*(Bp + baseB + 10));
                    float B_11 = (*(Bp + baseB + 11));
                    float B_12 = (*(Bp + baseB + 12));
                    float B_13 = (*(Bp + baseB + 13));
                    float B_14 = (*(Bp + baseB + 14));
                    float B_15 = (*(Bp + baseB + 15));
                    float B_16 = (*(Bp + baseB + 16));
                    float B_17 = (*(Bp + baseB + 17));
                    float B_18 = (*(Bp + baseB + 18));
                    float B_19 = (*(Bp + baseB + 19));
                    float B_20 = (*(Bp + baseB + 20));
                    float B_21 = (*(Bp + baseB + 21));
                    float B_22 = (*(Bp + baseB + 22));
                    float B_23 = (*(Bp + baseB + 23));
                    sum0_0 += A_0 * B_0;
                    sum1_0 += A_0 * B_1;
                    sum2_0 += A_0 * B_2;
                    sum3_0 += A_0 * B_3;
                    sum4_0 += A_0 * B_4;
                    sum5_0 += A_0 * B_5;
                    sum6_0 += A_0 * B_6;
                    sum7_0 += A_0 * B_7;
                    sum8_0 += A_0 * B_8;
                    sum9_0 += A_0 * B_9;
                    sum10_0 += A_0 * B_10;
                    sum11_0 += A_0 * B_11;
                    sum12_0 += A_0 * B_12;
                    sum13_0 += A_0 * B_13;
                    sum14_0 += A_0 * B_14;
                    sum15_0 += A_0 * B_15;
                    sum16_0 += A_0 * B_16;
                    sum17_0 += A_0 * B_17;
                    sum18_0 += A_0 * B_18;
                    sum19_0 += A_0 * B_19;
                    sum20_0 += A_0 * B_20;
                    sum21_0 += A_0 * B_21;
                    sum22_0 += A_0 * B_22;
                    sum23_0 += A_0 * B_23;
                }
                // 0
                *(Cp + baseC_0 + 0) = sum0_0;
                *(Cp + baseC_0 + 1) = sum1_0;
                *(Cp + baseC_0 + 2) = sum2_0;
                *(Cp + baseC_0 + 3) = sum3_0;
                *(Cp + baseC_0 + 4) = sum4_0;
                *(Cp + baseC_0 + 5) = sum5_0;
                *(Cp + baseC_0 + 6) = sum6_0;
                *(Cp + baseC_0 + 7) = sum7_0;
                *(Cp + baseC_0 + 8) = sum8_0;
                *(Cp + baseC_0 + 9) = sum9_0;
                *(Cp + baseC_0 + 10) = sum10_0;
                *(Cp + baseC_0 + 11) = sum11_0;
                *(Cp + baseC_0 + 12) = sum12_0;
                *(Cp + baseC_0 + 13) = sum13_0;
                *(Cp + baseC_0 + 14) = sum14_0;
                *(Cp + baseC_0 + 15) = sum15_0;
                *(Cp + baseC_0 + 16) = sum16_0;
                *(Cp + baseC_0 + 17) = sum17_0;
                *(Cp + baseC_0 + 18) = sum18_0;
                *(Cp + baseC_0 + 19) = sum19_0;
                *(Cp + baseC_0 + 20) = sum20_0;
                *(Cp + baseC_0 + 21) = sum21_0;
                *(Cp + baseC_0 + 22) = sum22_0;
                *(Cp + baseC_0 + 23) = sum23_0;
            }
        }
    }

    static unsafe void MultiplyBlockUnroll3x24I(
        [NoAlias] float* Ap, int Astride,
        [NoAlias] float* Bp, int Bstride,
        [NoAlias] float* Cp, int Cstride,
        int blockSizeM, int blockSizeK,
        int n)
    {
        n = Math.Max(24, n);
        int i = 0;
        for (; i < blockSizeM - 2; i += 3)
        {
            var i_0 = i + 0;
            var i_1 = i + 1;
            var i_2 = i + 2;

            for (int j = 0; j < n; j += 24)
            {
                int baseC_0 = i_0 * Cstride + j;
                int baseC_1 = i_1 * Cstride + j;
                int baseC_2 = i_2 * Cstride + j;

                // row 0
                v256 gamma_0_0 = mm256_loadu_ps(Cp + baseC_0 + 0);
                v256 gamma_0_8 = mm256_loadu_ps(Cp + baseC_0 + 8);
                v256 gamma_0_16 = mm256_loadu_ps(Cp + baseC_0 + 16);
                // row 1
                v256 gamma_1_0 = mm256_loadu_ps(Cp + baseC_1 + 0);
                v256 gamma_1_8 = mm256_loadu_ps(Cp + baseC_1 + 8);
                v256 gamma_1_16 = mm256_loadu_ps(Cp + baseC_1 + 16);
                // row 2
                v256 gamma_2_0 = mm256_loadu_ps(Cp + baseC_2 + 0);
                v256 gamma_2_8 = mm256_loadu_ps(Cp + baseC_2 + 8);
                v256 gamma_2_16 = mm256_loadu_ps(Cp + baseC_2 + 16);

                for (int l = 0; l < blockSizeK; l++)
                {
                    v256 alpha_0_p = mm256_broadcast_ss(Ap + i_0 * Astride + l);
                    v256 alpha_1_p = mm256_broadcast_ss(Ap + i_1 * Astride + l);
                    v256 alpha_2_p = mm256_broadcast_ss(Ap + i_2 * Astride + l);

                    v256 beta_p_0 = mm256_loadu_ps(Bp + l * Bstride + j + 0);
                    v256 beta_p_8 = mm256_loadu_ps(Bp + l * Bstride + j + 8);
                    v256 beta_p_16 = mm256_loadu_ps(Bp + l * Bstride + j + 16);

                    gamma_0_0 = mm256_fmadd_ps(alpha_0_p, beta_p_0, gamma_0_0);
                    gamma_1_0 = mm256_fmadd_ps(alpha_1_p, beta_p_0, gamma_1_0);
                    gamma_2_0 = mm256_fmadd_ps(alpha_2_p, beta_p_0, gamma_2_0);
                    gamma_0_8 = mm256_fmadd_ps(alpha_0_p, beta_p_8, gamma_0_8);
                    gamma_1_8 = mm256_fmadd_ps(alpha_1_p, beta_p_8, gamma_1_8);
                    gamma_2_8 = mm256_fmadd_ps(alpha_2_p, beta_p_8, gamma_2_8);
                    gamma_0_16 = mm256_fmadd_ps(alpha_0_p, beta_p_16, gamma_0_16);
                    gamma_1_16 = mm256_fmadd_ps(alpha_1_p, beta_p_16, gamma_1_16);
                    gamma_2_16 = mm256_fmadd_ps(alpha_2_p, beta_p_16, gamma_2_16);
                }
                // row 0
                mm256_storeu_ps(Cp + baseC_0 + 0, gamma_0_0);
                mm256_storeu_ps(Cp + baseC_0 + 8, gamma_0_8);
                mm256_storeu_ps(Cp + baseC_0 + 16, gamma_0_16);
                // row 1
                mm256_storeu_ps(Cp + baseC_1 + 0, gamma_1_0);
                mm256_storeu_ps(Cp + baseC_1 + 8, gamma_1_8);
                mm256_storeu_ps(Cp + baseC_1 + 16, gamma_1_16);
                // row 2
                mm256_storeu_ps(Cp + baseC_2 + 0, gamma_2_0);
                mm256_storeu_ps(Cp + baseC_2 + 8, gamma_2_8);
                mm256_storeu_ps(Cp + baseC_2 + 16, gamma_2_16);
            }
        }
        for (; i < blockSizeM - 1; i += 2)
        {
            var i_0 = i + 0;
            var i_1 = i + 1;

            for (int j = 0; j < n; j += 24)
            {
                int baseC_0 = i_0 * Cstride + j;
                int baseC_1 = i_1 * Cstride + j;

                // row 0
                v256 gamma_0_0 = mm256_loadu_ps(Cp + baseC_0 + 0);
                v256 gamma_0_8 = mm256_loadu_ps(Cp + baseC_0 + 8);
                v256 gamma_0_16 = mm256_loadu_ps(Cp + baseC_0 + 16);
                // row 1
                v256 gamma_1_0 = mm256_loadu_ps(Cp + baseC_1 + 0);
                v256 gamma_1_8 = mm256_loadu_ps(Cp + baseC_1 + 8);
                v256 gamma_1_16 = mm256_loadu_ps(Cp + baseC_1 + 16);

                for (int l = 0; l < blockSizeK; l++)
                {
                    v256 alpha_0_p = mm256_broadcast_ss(Ap + i_0 * Astride + l);
                    v256 alpha_1_p = mm256_broadcast_ss(Ap + i_1 * Astride + l);

                    v256 beta_p_0 = mm256_loadu_ps(Bp + l * Bstride + j + 0);
                    v256 beta_p_8 = mm256_loadu_ps(Bp + l * Bstride + j + 8);
                    v256 beta_p_16 = mm256_loadu_ps(Bp + l * Bstride + j + 16);

                    gamma_0_0 = mm256_fmadd_ps(alpha_0_p, beta_p_0, gamma_0_0);
                    gamma_1_0 = mm256_fmadd_ps(alpha_1_p, beta_p_0, gamma_1_0);
                    gamma_0_8 = mm256_fmadd_ps(alpha_0_p, beta_p_8, gamma_0_8);
                    gamma_1_8 = mm256_fmadd_ps(alpha_1_p, beta_p_8, gamma_1_8);
                    gamma_0_16 = mm256_fmadd_ps(alpha_0_p, beta_p_16, gamma_0_16);
                    gamma_1_16 = mm256_fmadd_ps(alpha_1_p, beta_p_16, gamma_1_16);
                }
                // row 0
                mm256_storeu_ps(Cp + baseC_0 + 0, gamma_0_0);
                mm256_storeu_ps(Cp + baseC_0 + 8, gamma_0_8);
                mm256_storeu_ps(Cp + baseC_0 + 16, gamma_0_16);
                // row 1
                mm256_storeu_ps(Cp + baseC_1 + 0, gamma_1_0);
                mm256_storeu_ps(Cp + baseC_1 + 8, gamma_1_8);
                mm256_storeu_ps(Cp + baseC_1 + 16, gamma_1_16);
            }
        }
        for (; i < blockSizeM - 0; i += 1)
        {
            var i_0 = i + 0;

            for (int j = 0; j < n; j += 24)
            {
                int baseC_0 = i_0 * Cstride + j;

                // row 0
                v256 gamma_0_0 = mm256_loadu_ps(Cp + baseC_0 + 0);
                v256 gamma_0_8 = mm256_loadu_ps(Cp + baseC_0 + 8);
                v256 gamma_0_16 = mm256_loadu_ps(Cp + baseC_0 + 16);

                for (int l = 0; l < blockSizeK; l++)
                {
                    v256 alpha_0_p = mm256_broadcast_ss(Ap + i_0 * Astride + l);

                    v256 beta_p_0 = mm256_loadu_ps(Bp + l * Bstride + j + 0);
                    v256 beta_p_8 = mm256_loadu_ps(Bp + l * Bstride + j + 8);
                    v256 beta_p_16 = mm256_loadu_ps(Bp + l * Bstride + j + 16);

                    gamma_0_0 = mm256_fmadd_ps(alpha_0_p, beta_p_0, gamma_0_0);
                    gamma_0_8 = mm256_fmadd_ps(alpha_0_p, beta_p_8, gamma_0_8);
                    gamma_0_16 = mm256_fmadd_ps(alpha_0_p, beta_p_16, gamma_0_16);
                }
                // row 0
                mm256_storeu_ps(Cp + baseC_0 + 0, gamma_0_0);
                mm256_storeu_ps(Cp + baseC_0 + 8, gamma_0_8);
                mm256_storeu_ps(Cp + baseC_0 + 16, gamma_0_16);
            }
        }
    }

    static unsafe void MultiplyBlockUnroll3x32(
            [NoAlias] float* Ap, int Astride,
            [NoAlias] float* Bp, int Bstride,
            [NoAlias] float* Cp, int Cstride,
            int blockSizeM, int blockSizeK,
            int n)
    {
        n = Math.Max(32, n);
        int i = 0;
        for (; i < blockSizeM - 2; i += 3)
        {
            var i_0 = i + 0;
            var i_1 = i + 1;
            var i_2 = i + 2;

            for (int j = 0; j < n; j += 32)
            {
                int baseC_0 = i_0 * Cstride + j;
                int baseC_1 = i_1 * Cstride + j;
                int baseC_2 = i_2 * Cstride + j;
                // 0
                float sum0_0 = *(Cp + baseC_0 + 0);
                float sum1_0 = *(Cp + baseC_0 + 1);
                float sum2_0 = *(Cp + baseC_0 + 2);
                float sum3_0 = *(Cp + baseC_0 + 3);
                float sum4_0 = *(Cp + baseC_0 + 4);
                float sum5_0 = *(Cp + baseC_0 + 5);
                float sum6_0 = *(Cp + baseC_0 + 6);
                float sum7_0 = *(Cp + baseC_0 + 7);
                float sum8_0 = *(Cp + baseC_0 + 8);
                float sum9_0 = *(Cp + baseC_0 + 9);
                float sum10_0 = *(Cp + baseC_0 + 10);
                float sum11_0 = *(Cp + baseC_0 + 11);
                float sum12_0 = *(Cp + baseC_0 + 12);
                float sum13_0 = *(Cp + baseC_0 + 13);
                float sum14_0 = *(Cp + baseC_0 + 14);
                float sum15_0 = *(Cp + baseC_0 + 15);
                float sum16_0 = *(Cp + baseC_0 + 16);
                float sum17_0 = *(Cp + baseC_0 + 17);
                float sum18_0 = *(Cp + baseC_0 + 18);
                float sum19_0 = *(Cp + baseC_0 + 19);
                float sum20_0 = *(Cp + baseC_0 + 20);
                float sum21_0 = *(Cp + baseC_0 + 21);
                float sum22_0 = *(Cp + baseC_0 + 22);
                float sum23_0 = *(Cp + baseC_0 + 23);
                float sum24_0 = *(Cp + baseC_0 + 24);
                float sum25_0 = *(Cp + baseC_0 + 25);
                float sum26_0 = *(Cp + baseC_0 + 26);
                float sum27_0 = *(Cp + baseC_0 + 27);
                float sum28_0 = *(Cp + baseC_0 + 28);
                float sum29_0 = *(Cp + baseC_0 + 29);
                float sum30_0 = *(Cp + baseC_0 + 30);
                float sum31_0 = *(Cp + baseC_0 + 31);
                // 1
                float sum0_1 = *(Cp + baseC_1 + 0);
                float sum1_1 = *(Cp + baseC_1 + 1);
                float sum2_1 = *(Cp + baseC_1 + 2);
                float sum3_1 = *(Cp + baseC_1 + 3);
                float sum4_1 = *(Cp + baseC_1 + 4);
                float sum5_1 = *(Cp + baseC_1 + 5);
                float sum6_1 = *(Cp + baseC_1 + 6);
                float sum7_1 = *(Cp + baseC_1 + 7);
                float sum8_1 = *(Cp + baseC_1 + 8);
                float sum9_1 = *(Cp + baseC_1 + 9);
                float sum10_1 = *(Cp + baseC_1 + 10);
                float sum11_1 = *(Cp + baseC_1 + 11);
                float sum12_1 = *(Cp + baseC_1 + 12);
                float sum13_1 = *(Cp + baseC_1 + 13);
                float sum14_1 = *(Cp + baseC_1 + 14);
                float sum15_1 = *(Cp + baseC_1 + 15);
                float sum16_1 = *(Cp + baseC_1 + 16);
                float sum17_1 = *(Cp + baseC_1 + 17);
                float sum18_1 = *(Cp + baseC_1 + 18);
                float sum19_1 = *(Cp + baseC_1 + 19);
                float sum20_1 = *(Cp + baseC_1 + 20);
                float sum21_1 = *(Cp + baseC_1 + 21);
                float sum22_1 = *(Cp + baseC_1 + 22);
                float sum23_1 = *(Cp + baseC_1 + 23);
                float sum24_1 = *(Cp + baseC_1 + 24);
                float sum25_1 = *(Cp + baseC_1 + 25);
                float sum26_1 = *(Cp + baseC_1 + 26);
                float sum27_1 = *(Cp + baseC_1 + 27);
                float sum28_1 = *(Cp + baseC_1 + 28);
                float sum29_1 = *(Cp + baseC_1 + 29);
                float sum30_1 = *(Cp + baseC_1 + 30);
                float sum31_1 = *(Cp + baseC_1 + 31);
                // 2
                float sum0_2 = *(Cp + baseC_2 + 0);
                float sum1_2 = *(Cp + baseC_2 + 1);
                float sum2_2 = *(Cp + baseC_2 + 2);
                float sum3_2 = *(Cp + baseC_2 + 3);
                float sum4_2 = *(Cp + baseC_2 + 4);
                float sum5_2 = *(Cp + baseC_2 + 5);
                float sum6_2 = *(Cp + baseC_2 + 6);
                float sum7_2 = *(Cp + baseC_2 + 7);
                float sum8_2 = *(Cp + baseC_2 + 8);
                float sum9_2 = *(Cp + baseC_2 + 9);
                float sum10_2 = *(Cp + baseC_2 + 10);
                float sum11_2 = *(Cp + baseC_2 + 11);
                float sum12_2 = *(Cp + baseC_2 + 12);
                float sum13_2 = *(Cp + baseC_2 + 13);
                float sum14_2 = *(Cp + baseC_2 + 14);
                float sum15_2 = *(Cp + baseC_2 + 15);
                float sum16_2 = *(Cp + baseC_2 + 16);
                float sum17_2 = *(Cp + baseC_2 + 17);
                float sum18_2 = *(Cp + baseC_2 + 18);
                float sum19_2 = *(Cp + baseC_2 + 19);
                float sum20_2 = *(Cp + baseC_2 + 20);
                float sum21_2 = *(Cp + baseC_2 + 21);
                float sum22_2 = *(Cp + baseC_2 + 22);
                float sum23_2 = *(Cp + baseC_2 + 23);
                float sum24_2 = *(Cp + baseC_2 + 24);
                float sum25_2 = *(Cp + baseC_2 + 25);
                float sum26_2 = *(Cp + baseC_2 + 26);
                float sum27_2 = *(Cp + baseC_2 + 27);
                float sum28_2 = *(Cp + baseC_2 + 28);
                float sum29_2 = *(Cp + baseC_2 + 29);
                float sum30_2 = *(Cp + baseC_2 + 30);
                float sum31_2 = *(Cp + baseC_2 + 31);

                for (int l = 0; l < blockSizeK; l++)
                {
                    float A_0 = *(Ap + i_0 * Astride + l);
                    float A_1 = *(Ap + i_1 * Astride + l);
                    float A_2 = *(Ap + i_2 * Astride + l);
                    int baseB = l * Bstride + j;
                    float B_0 = (*(Bp + baseB + 0));
                    float B_1 = (*(Bp + baseB + 1));
                    float B_2 = (*(Bp + baseB + 2));
                    float B_3 = (*(Bp + baseB + 3));
                    float B_4 = (*(Bp + baseB + 4));
                    float B_5 = (*(Bp + baseB + 5));
                    float B_6 = (*(Bp + baseB + 6));
                    float B_7 = (*(Bp + baseB + 7));
                    float B_8 = (*(Bp + baseB + 8));
                    float B_9 = (*(Bp + baseB + 9));
                    float B_10 = (*(Bp + baseB + 10));
                    float B_11 = (*(Bp + baseB + 11));
                    float B_12 = (*(Bp + baseB + 12));
                    float B_13 = (*(Bp + baseB + 13));
                    float B_14 = (*(Bp + baseB + 14));
                    float B_15 = (*(Bp + baseB + 15));
                    float B_16 = (*(Bp + baseB + 16));
                    float B_17 = (*(Bp + baseB + 17));
                    float B_18 = (*(Bp + baseB + 18));
                    float B_19 = (*(Bp + baseB + 19));
                    float B_20 = (*(Bp + baseB + 20));
                    float B_21 = (*(Bp + baseB + 21));
                    float B_22 = (*(Bp + baseB + 22));
                    float B_23 = (*(Bp + baseB + 23));
                    float B_24 = (*(Bp + baseB + 24));
                    float B_25 = (*(Bp + baseB + 25));
                    float B_26 = (*(Bp + baseB + 26));
                    float B_27 = (*(Bp + baseB + 27));
                    float B_28 = (*(Bp + baseB + 28));
                    float B_29 = (*(Bp + baseB + 29));
                    float B_30 = (*(Bp + baseB + 30));
                    float B_31 = (*(Bp + baseB + 31));
                    sum0_0 += A_0 * B_0;	sum0_1 += A_1 * B_0;	sum0_2 += A_2 * B_0;
                    sum1_0 += A_0 * B_1;	sum1_1 += A_1 * B_1;	sum1_2 += A_2 * B_1;
                    sum2_0 += A_0 * B_2;	sum2_1 += A_1 * B_2;	sum2_2 += A_2 * B_2;
                    sum3_0 += A_0 * B_3;	sum3_1 += A_1 * B_3;	sum3_2 += A_2 * B_3;
                    sum4_0 += A_0 * B_4;	sum4_1 += A_1 * B_4;	sum4_2 += A_2 * B_4;
                    sum5_0 += A_0 * B_5;	sum5_1 += A_1 * B_5;	sum5_2 += A_2 * B_5;
                    sum6_0 += A_0 * B_6;	sum6_1 += A_1 * B_6;	sum6_2 += A_2 * B_6;
                    sum7_0 += A_0 * B_7;	sum7_1 += A_1 * B_7;	sum7_2 += A_2 * B_7;
                    sum8_0 += A_0 * B_8;	sum8_1 += A_1 * B_8;	sum8_2 += A_2 * B_8;
                    sum9_0 += A_0 * B_9;	sum9_1 += A_1 * B_9;	sum9_2 += A_2 * B_9;
                    sum10_0 += A_0 * B_10;	sum10_1 += A_1 * B_10;	sum10_2 += A_2 * B_10;
                    sum11_0 += A_0 * B_11;	sum11_1 += A_1 * B_11;	sum11_2 += A_2 * B_11;
                    sum12_0 += A_0 * B_12;	sum12_1 += A_1 * B_12;	sum12_2 += A_2 * B_12;
                    sum13_0 += A_0 * B_13;	sum13_1 += A_1 * B_13;	sum13_2 += A_2 * B_13;
                    sum14_0 += A_0 * B_14;	sum14_1 += A_1 * B_14;	sum14_2 += A_2 * B_14;
                    sum15_0 += A_0 * B_15;	sum15_1 += A_1 * B_15;	sum15_2 += A_2 * B_15;
                    sum16_0 += A_0 * B_16;	sum16_1 += A_1 * B_16;	sum16_2 += A_2 * B_16;
                    sum17_0 += A_0 * B_17;	sum17_1 += A_1 * B_17;	sum17_2 += A_2 * B_17;
                    sum18_0 += A_0 * B_18;	sum18_1 += A_1 * B_18;	sum18_2 += A_2 * B_18;
                    sum19_0 += A_0 * B_19;	sum19_1 += A_1 * B_19;	sum19_2 += A_2 * B_19;
                    sum20_0 += A_0 * B_20;	sum20_1 += A_1 * B_20;	sum20_2 += A_2 * B_20;
                    sum21_0 += A_0 * B_21;	sum21_1 += A_1 * B_21;	sum21_2 += A_2 * B_21;
                    sum22_0 += A_0 * B_22;	sum22_1 += A_1 * B_22;	sum22_2 += A_2 * B_22;
                    sum23_0 += A_0 * B_23;	sum23_1 += A_1 * B_23;	sum23_2 += A_2 * B_23;
                    sum24_0 += A_0 * B_24;	sum24_1 += A_1 * B_24;	sum24_2 += A_2 * B_24;
                    sum25_0 += A_0 * B_25;	sum25_1 += A_1 * B_25;	sum25_2 += A_2 * B_25;
                    sum26_0 += A_0 * B_26;	sum26_1 += A_1 * B_26;	sum26_2 += A_2 * B_26;
                    sum27_0 += A_0 * B_27;	sum27_1 += A_1 * B_27;	sum27_2 += A_2 * B_27;
                    sum28_0 += A_0 * B_28;	sum28_1 += A_1 * B_28;	sum28_2 += A_2 * B_28;
                    sum29_0 += A_0 * B_29;	sum29_1 += A_1 * B_29;	sum29_2 += A_2 * B_29;
                    sum30_0 += A_0 * B_30;	sum30_1 += A_1 * B_30;	sum30_2 += A_2 * B_30;
                    sum31_0 += A_0 * B_31;	sum31_1 += A_1 * B_31;	sum31_2 += A_2 * B_31;
                }
                // 0
                *(Cp + baseC_0 + 0) = sum0_0;
                *(Cp + baseC_0 + 1) = sum1_0;
                *(Cp + baseC_0 + 2) = sum2_0;
                *(Cp + baseC_0 + 3) = sum3_0;
                *(Cp + baseC_0 + 4) = sum4_0;
                *(Cp + baseC_0 + 5) = sum5_0;
                *(Cp + baseC_0 + 6) = sum6_0;
                *(Cp + baseC_0 + 7) = sum7_0;
                *(Cp + baseC_0 + 8) = sum8_0;
                *(Cp + baseC_0 + 9) = sum9_0;
                *(Cp + baseC_0 + 10) = sum10_0;
                *(Cp + baseC_0 + 11) = sum11_0;
                *(Cp + baseC_0 + 12) = sum12_0;
                *(Cp + baseC_0 + 13) = sum13_0;
                *(Cp + baseC_0 + 14) = sum14_0;
                *(Cp + baseC_0 + 15) = sum15_0;
                *(Cp + baseC_0 + 16) = sum16_0;
                *(Cp + baseC_0 + 17) = sum17_0;
                *(Cp + baseC_0 + 18) = sum18_0;
                *(Cp + baseC_0 + 19) = sum19_0;
                *(Cp + baseC_0 + 20) = sum20_0;
                *(Cp + baseC_0 + 21) = sum21_0;
                *(Cp + baseC_0 + 22) = sum22_0;
                *(Cp + baseC_0 + 23) = sum23_0;
                *(Cp + baseC_0 + 24) = sum24_0;
                *(Cp + baseC_0 + 25) = sum25_0;
                *(Cp + baseC_0 + 26) = sum26_0;
                *(Cp + baseC_0 + 27) = sum27_0;
                *(Cp + baseC_0 + 28) = sum28_0;
                *(Cp + baseC_0 + 29) = sum29_0;
                *(Cp + baseC_0 + 30) = sum30_0;
                *(Cp + baseC_0 + 31) = sum31_0;
                // 1
                *(Cp + baseC_1 + 0) = sum0_1;
                *(Cp + baseC_1 + 1) = sum1_1;
                *(Cp + baseC_1 + 2) = sum2_1;
                *(Cp + baseC_1 + 3) = sum3_1;
                *(Cp + baseC_1 + 4) = sum4_1;
                *(Cp + baseC_1 + 5) = sum5_1;
                *(Cp + baseC_1 + 6) = sum6_1;
                *(Cp + baseC_1 + 7) = sum7_1;
                *(Cp + baseC_1 + 8) = sum8_1;
                *(Cp + baseC_1 + 9) = sum9_1;
                *(Cp + baseC_1 + 10) = sum10_1;
                *(Cp + baseC_1 + 11) = sum11_1;
                *(Cp + baseC_1 + 12) = sum12_1;
                *(Cp + baseC_1 + 13) = sum13_1;
                *(Cp + baseC_1 + 14) = sum14_1;
                *(Cp + baseC_1 + 15) = sum15_1;
                *(Cp + baseC_1 + 16) = sum16_1;
                *(Cp + baseC_1 + 17) = sum17_1;
                *(Cp + baseC_1 + 18) = sum18_1;
                *(Cp + baseC_1 + 19) = sum19_1;
                *(Cp + baseC_1 + 20) = sum20_1;
                *(Cp + baseC_1 + 21) = sum21_1;
                *(Cp + baseC_1 + 22) = sum22_1;
                *(Cp + baseC_1 + 23) = sum23_1;
                *(Cp + baseC_1 + 24) = sum24_1;
                *(Cp + baseC_1 + 25) = sum25_1;
                *(Cp + baseC_1 + 26) = sum26_1;
                *(Cp + baseC_1 + 27) = sum27_1;
                *(Cp + baseC_1 + 28) = sum28_1;
                *(Cp + baseC_1 + 29) = sum29_1;
                *(Cp + baseC_1 + 30) = sum30_1;
                *(Cp + baseC_1 + 31) = sum31_1;
                // 2
                *(Cp + baseC_2 + 0) = sum0_2;
                *(Cp + baseC_2 + 1) = sum1_2;
                *(Cp + baseC_2 + 2) = sum2_2;
                *(Cp + baseC_2 + 3) = sum3_2;
                *(Cp + baseC_2 + 4) = sum4_2;
                *(Cp + baseC_2 + 5) = sum5_2;
                *(Cp + baseC_2 + 6) = sum6_2;
                *(Cp + baseC_2 + 7) = sum7_2;
                *(Cp + baseC_2 + 8) = sum8_2;
                *(Cp + baseC_2 + 9) = sum9_2;
                *(Cp + baseC_2 + 10) = sum10_2;
                *(Cp + baseC_2 + 11) = sum11_2;
                *(Cp + baseC_2 + 12) = sum12_2;
                *(Cp + baseC_2 + 13) = sum13_2;
                *(Cp + baseC_2 + 14) = sum14_2;
                *(Cp + baseC_2 + 15) = sum15_2;
                *(Cp + baseC_2 + 16) = sum16_2;
                *(Cp + baseC_2 + 17) = sum17_2;
                *(Cp + baseC_2 + 18) = sum18_2;
                *(Cp + baseC_2 + 19) = sum19_2;
                *(Cp + baseC_2 + 20) = sum20_2;
                *(Cp + baseC_2 + 21) = sum21_2;
                *(Cp + baseC_2 + 22) = sum22_2;
                *(Cp + baseC_2 + 23) = sum23_2;
                *(Cp + baseC_2 + 24) = sum24_2;
                *(Cp + baseC_2 + 25) = sum25_2;
                *(Cp + baseC_2 + 26) = sum26_2;
                *(Cp + baseC_2 + 27) = sum27_2;
                *(Cp + baseC_2 + 28) = sum28_2;
                *(Cp + baseC_2 + 29) = sum29_2;
                *(Cp + baseC_2 + 30) = sum30_2;
                *(Cp + baseC_2 + 31) = sum31_2;
            }
        }
        for (; i < blockSizeM - 1; i += 2)
        {
            var i_0 = i + 0;
            var i_1 = i + 1;

            for (int j = 0; j < n; j += 32)
            {
                int baseC_0 = i_0 * Cstride + j;
                int baseC_1 = i_1 * Cstride + j;
                // 0
                float sum0_0 = *(Cp + baseC_0 + 0);
                float sum1_0 = *(Cp + baseC_0 + 1);
                float sum2_0 = *(Cp + baseC_0 + 2);
                float sum3_0 = *(Cp + baseC_0 + 3);
                float sum4_0 = *(Cp + baseC_0 + 4);
                float sum5_0 = *(Cp + baseC_0 + 5);
                float sum6_0 = *(Cp + baseC_0 + 6);
                float sum7_0 = *(Cp + baseC_0 + 7);
                float sum8_0 = *(Cp + baseC_0 + 8);
                float sum9_0 = *(Cp + baseC_0 + 9);
                float sum10_0 = *(Cp + baseC_0 + 10);
                float sum11_0 = *(Cp + baseC_0 + 11);
                float sum12_0 = *(Cp + baseC_0 + 12);
                float sum13_0 = *(Cp + baseC_0 + 13);
                float sum14_0 = *(Cp + baseC_0 + 14);
                float sum15_0 = *(Cp + baseC_0 + 15);
                float sum16_0 = *(Cp + baseC_0 + 16);
                float sum17_0 = *(Cp + baseC_0 + 17);
                float sum18_0 = *(Cp + baseC_0 + 18);
                float sum19_0 = *(Cp + baseC_0 + 19);
                float sum20_0 = *(Cp + baseC_0 + 20);
                float sum21_0 = *(Cp + baseC_0 + 21);
                float sum22_0 = *(Cp + baseC_0 + 22);
                float sum23_0 = *(Cp + baseC_0 + 23);
                float sum24_0 = *(Cp + baseC_0 + 24);
                float sum25_0 = *(Cp + baseC_0 + 25);
                float sum26_0 = *(Cp + baseC_0 + 26);
                float sum27_0 = *(Cp + baseC_0 + 27);
                float sum28_0 = *(Cp + baseC_0 + 28);
                float sum29_0 = *(Cp + baseC_0 + 29);
                float sum30_0 = *(Cp + baseC_0 + 30);
                float sum31_0 = *(Cp + baseC_0 + 31);
                // 1
                float sum0_1 = *(Cp + baseC_1 + 0);
                float sum1_1 = *(Cp + baseC_1 + 1);
                float sum2_1 = *(Cp + baseC_1 + 2);
                float sum3_1 = *(Cp + baseC_1 + 3);
                float sum4_1 = *(Cp + baseC_1 + 4);
                float sum5_1 = *(Cp + baseC_1 + 5);
                float sum6_1 = *(Cp + baseC_1 + 6);
                float sum7_1 = *(Cp + baseC_1 + 7);
                float sum8_1 = *(Cp + baseC_1 + 8);
                float sum9_1 = *(Cp + baseC_1 + 9);
                float sum10_1 = *(Cp + baseC_1 + 10);
                float sum11_1 = *(Cp + baseC_1 + 11);
                float sum12_1 = *(Cp + baseC_1 + 12);
                float sum13_1 = *(Cp + baseC_1 + 13);
                float sum14_1 = *(Cp + baseC_1 + 14);
                float sum15_1 = *(Cp + baseC_1 + 15);
                float sum16_1 = *(Cp + baseC_1 + 16);
                float sum17_1 = *(Cp + baseC_1 + 17);
                float sum18_1 = *(Cp + baseC_1 + 18);
                float sum19_1 = *(Cp + baseC_1 + 19);
                float sum20_1 = *(Cp + baseC_1 + 20);
                float sum21_1 = *(Cp + baseC_1 + 21);
                float sum22_1 = *(Cp + baseC_1 + 22);
                float sum23_1 = *(Cp + baseC_1 + 23);
                float sum24_1 = *(Cp + baseC_1 + 24);
                float sum25_1 = *(Cp + baseC_1 + 25);
                float sum26_1 = *(Cp + baseC_1 + 26);
                float sum27_1 = *(Cp + baseC_1 + 27);
                float sum28_1 = *(Cp + baseC_1 + 28);
                float sum29_1 = *(Cp + baseC_1 + 29);
                float sum30_1 = *(Cp + baseC_1 + 30);
                float sum31_1 = *(Cp + baseC_1 + 31);

                for (int l = 0; l < blockSizeK; l++)
                {
                    float A_0 = *(Ap + i_0 * Astride + l);
                    float A_1 = *(Ap + i_1 * Astride + l);
                    int baseB = l * Bstride + j;
                    float B_0 = (*(Bp + baseB + 0));
                    float B_1 = (*(Bp + baseB + 1));
                    float B_2 = (*(Bp + baseB + 2));
                    float B_3 = (*(Bp + baseB + 3));
                    float B_4 = (*(Bp + baseB + 4));
                    float B_5 = (*(Bp + baseB + 5));
                    float B_6 = (*(Bp + baseB + 6));
                    float B_7 = (*(Bp + baseB + 7));
                    float B_8 = (*(Bp + baseB + 8));
                    float B_9 = (*(Bp + baseB + 9));
                    float B_10 = (*(Bp + baseB + 10));
                    float B_11 = (*(Bp + baseB + 11));
                    float B_12 = (*(Bp + baseB + 12));
                    float B_13 = (*(Bp + baseB + 13));
                    float B_14 = (*(Bp + baseB + 14));
                    float B_15 = (*(Bp + baseB + 15));
                    float B_16 = (*(Bp + baseB + 16));
                    float B_17 = (*(Bp + baseB + 17));
                    float B_18 = (*(Bp + baseB + 18));
                    float B_19 = (*(Bp + baseB + 19));
                    float B_20 = (*(Bp + baseB + 20));
                    float B_21 = (*(Bp + baseB + 21));
                    float B_22 = (*(Bp + baseB + 22));
                    float B_23 = (*(Bp + baseB + 23));
                    float B_24 = (*(Bp + baseB + 24));
                    float B_25 = (*(Bp + baseB + 25));
                    float B_26 = (*(Bp + baseB + 26));
                    float B_27 = (*(Bp + baseB + 27));
                    float B_28 = (*(Bp + baseB + 28));
                    float B_29 = (*(Bp + baseB + 29));
                    float B_30 = (*(Bp + baseB + 30));
                    float B_31 = (*(Bp + baseB + 31));
                    sum0_0 += A_0 * B_0;	sum0_1 += A_1 * B_0;
                    sum1_0 += A_0 * B_1;	sum1_1 += A_1 * B_1;
                    sum2_0 += A_0 * B_2;	sum2_1 += A_1 * B_2;
                    sum3_0 += A_0 * B_3;	sum3_1 += A_1 * B_3;
                    sum4_0 += A_0 * B_4;	sum4_1 += A_1 * B_4;
                    sum5_0 += A_0 * B_5;	sum5_1 += A_1 * B_5;
                    sum6_0 += A_0 * B_6;	sum6_1 += A_1 * B_6;
                    sum7_0 += A_0 * B_7;	sum7_1 += A_1 * B_7;
                    sum8_0 += A_0 * B_8;	sum8_1 += A_1 * B_8;
                    sum9_0 += A_0 * B_9;	sum9_1 += A_1 * B_9;
                    sum10_0 += A_0 * B_10;	sum10_1 += A_1 * B_10;
                    sum11_0 += A_0 * B_11;	sum11_1 += A_1 * B_11;
                    sum12_0 += A_0 * B_12;	sum12_1 += A_1 * B_12;
                    sum13_0 += A_0 * B_13;	sum13_1 += A_1 * B_13;
                    sum14_0 += A_0 * B_14;	sum14_1 += A_1 * B_14;
                    sum15_0 += A_0 * B_15;	sum15_1 += A_1 * B_15;
                    sum16_0 += A_0 * B_16;	sum16_1 += A_1 * B_16;
                    sum17_0 += A_0 * B_17;	sum17_1 += A_1 * B_17;
                    sum18_0 += A_0 * B_18;	sum18_1 += A_1 * B_18;
                    sum19_0 += A_0 * B_19;	sum19_1 += A_1 * B_19;
                    sum20_0 += A_0 * B_20;	sum20_1 += A_1 * B_20;
                    sum21_0 += A_0 * B_21;	sum21_1 += A_1 * B_21;
                    sum22_0 += A_0 * B_22;	sum22_1 += A_1 * B_22;
                    sum23_0 += A_0 * B_23;	sum23_1 += A_1 * B_23;
                    sum24_0 += A_0 * B_24;	sum24_1 += A_1 * B_24;
                    sum25_0 += A_0 * B_25;	sum25_1 += A_1 * B_25;
                    sum26_0 += A_0 * B_26;	sum26_1 += A_1 * B_26;
                    sum27_0 += A_0 * B_27;	sum27_1 += A_1 * B_27;
                    sum28_0 += A_0 * B_28;	sum28_1 += A_1 * B_28;
                    sum29_0 += A_0 * B_29;	sum29_1 += A_1 * B_29;
                    sum30_0 += A_0 * B_30;	sum30_1 += A_1 * B_30;
                    sum31_0 += A_0 * B_31;	sum31_1 += A_1 * B_31;
                }
                // 0
                *(Cp + baseC_0 + 0) = sum0_0;
                *(Cp + baseC_0 + 1) = sum1_0;
                *(Cp + baseC_0 + 2) = sum2_0;
                *(Cp + baseC_0 + 3) = sum3_0;
                *(Cp + baseC_0 + 4) = sum4_0;
                *(Cp + baseC_0 + 5) = sum5_0;
                *(Cp + baseC_0 + 6) = sum6_0;
                *(Cp + baseC_0 + 7) = sum7_0;
                *(Cp + baseC_0 + 8) = sum8_0;
                *(Cp + baseC_0 + 9) = sum9_0;
                *(Cp + baseC_0 + 10) = sum10_0;
                *(Cp + baseC_0 + 11) = sum11_0;
                *(Cp + baseC_0 + 12) = sum12_0;
                *(Cp + baseC_0 + 13) = sum13_0;
                *(Cp + baseC_0 + 14) = sum14_0;
                *(Cp + baseC_0 + 15) = sum15_0;
                *(Cp + baseC_0 + 16) = sum16_0;
                *(Cp + baseC_0 + 17) = sum17_0;
                *(Cp + baseC_0 + 18) = sum18_0;
                *(Cp + baseC_0 + 19) = sum19_0;
                *(Cp + baseC_0 + 20) = sum20_0;
                *(Cp + baseC_0 + 21) = sum21_0;
                *(Cp + baseC_0 + 22) = sum22_0;
                *(Cp + baseC_0 + 23) = sum23_0;
                *(Cp + baseC_0 + 24) = sum24_0;
                *(Cp + baseC_0 + 25) = sum25_0;
                *(Cp + baseC_0 + 26) = sum26_0;
                *(Cp + baseC_0 + 27) = sum27_0;
                *(Cp + baseC_0 + 28) = sum28_0;
                *(Cp + baseC_0 + 29) = sum29_0;
                *(Cp + baseC_0 + 30) = sum30_0;
                *(Cp + baseC_0 + 31) = sum31_0;
                // 1
                *(Cp + baseC_1 + 0) = sum0_1;
                *(Cp + baseC_1 + 1) = sum1_1;
                *(Cp + baseC_1 + 2) = sum2_1;
                *(Cp + baseC_1 + 3) = sum3_1;
                *(Cp + baseC_1 + 4) = sum4_1;
                *(Cp + baseC_1 + 5) = sum5_1;
                *(Cp + baseC_1 + 6) = sum6_1;
                *(Cp + baseC_1 + 7) = sum7_1;
                *(Cp + baseC_1 + 8) = sum8_1;
                *(Cp + baseC_1 + 9) = sum9_1;
                *(Cp + baseC_1 + 10) = sum10_1;
                *(Cp + baseC_1 + 11) = sum11_1;
                *(Cp + baseC_1 + 12) = sum12_1;
                *(Cp + baseC_1 + 13) = sum13_1;
                *(Cp + baseC_1 + 14) = sum14_1;
                *(Cp + baseC_1 + 15) = sum15_1;
                *(Cp + baseC_1 + 16) = sum16_1;
                *(Cp + baseC_1 + 17) = sum17_1;
                *(Cp + baseC_1 + 18) = sum18_1;
                *(Cp + baseC_1 + 19) = sum19_1;
                *(Cp + baseC_1 + 20) = sum20_1;
                *(Cp + baseC_1 + 21) = sum21_1;
                *(Cp + baseC_1 + 22) = sum22_1;
                *(Cp + baseC_1 + 23) = sum23_1;
                *(Cp + baseC_1 + 24) = sum24_1;
                *(Cp + baseC_1 + 25) = sum25_1;
                *(Cp + baseC_1 + 26) = sum26_1;
                *(Cp + baseC_1 + 27) = sum27_1;
                *(Cp + baseC_1 + 28) = sum28_1;
                *(Cp + baseC_1 + 29) = sum29_1;
                *(Cp + baseC_1 + 30) = sum30_1;
                *(Cp + baseC_1 + 31) = sum31_1;
            }
        }
        for (; i < blockSizeM - 0; i += 1)
        {
            var i_0 = i + 0;

            for (int j = 0; j < n; j += 32)
            {
                int baseC_0 = i_0 * Cstride + j;
                // 0
                float sum0_0 = *(Cp + baseC_0 + 0);
                float sum1_0 = *(Cp + baseC_0 + 1);
                float sum2_0 = *(Cp + baseC_0 + 2);
                float sum3_0 = *(Cp + baseC_0 + 3);
                float sum4_0 = *(Cp + baseC_0 + 4);
                float sum5_0 = *(Cp + baseC_0 + 5);
                float sum6_0 = *(Cp + baseC_0 + 6);
                float sum7_0 = *(Cp + baseC_0 + 7);
                float sum8_0 = *(Cp + baseC_0 + 8);
                float sum9_0 = *(Cp + baseC_0 + 9);
                float sum10_0 = *(Cp + baseC_0 + 10);
                float sum11_0 = *(Cp + baseC_0 + 11);
                float sum12_0 = *(Cp + baseC_0 + 12);
                float sum13_0 = *(Cp + baseC_0 + 13);
                float sum14_0 = *(Cp + baseC_0 + 14);
                float sum15_0 = *(Cp + baseC_0 + 15);
                float sum16_0 = *(Cp + baseC_0 + 16);
                float sum17_0 = *(Cp + baseC_0 + 17);
                float sum18_0 = *(Cp + baseC_0 + 18);
                float sum19_0 = *(Cp + baseC_0 + 19);
                float sum20_0 = *(Cp + baseC_0 + 20);
                float sum21_0 = *(Cp + baseC_0 + 21);
                float sum22_0 = *(Cp + baseC_0 + 22);
                float sum23_0 = *(Cp + baseC_0 + 23);
                float sum24_0 = *(Cp + baseC_0 + 24);
                float sum25_0 = *(Cp + baseC_0 + 25);
                float sum26_0 = *(Cp + baseC_0 + 26);
                float sum27_0 = *(Cp + baseC_0 + 27);
                float sum28_0 = *(Cp + baseC_0 + 28);
                float sum29_0 = *(Cp + baseC_0 + 29);
                float sum30_0 = *(Cp + baseC_0 + 30);
                float sum31_0 = *(Cp + baseC_0 + 31);

                for (int l = 0; l < blockSizeK; l++)
                {
                    float A_0 = *(Ap + i_0 * Astride + l);
                    int baseB = l * Bstride + j;
                    float B_0 = (*(Bp + baseB + 0));
                    float B_1 = (*(Bp + baseB + 1));
                    float B_2 = (*(Bp + baseB + 2));
                    float B_3 = (*(Bp + baseB + 3));
                    float B_4 = (*(Bp + baseB + 4));
                    float B_5 = (*(Bp + baseB + 5));
                    float B_6 = (*(Bp + baseB + 6));
                    float B_7 = (*(Bp + baseB + 7));
                    float B_8 = (*(Bp + baseB + 8));
                    float B_9 = (*(Bp + baseB + 9));
                    float B_10 = (*(Bp + baseB + 10));
                    float B_11 = (*(Bp + baseB + 11));
                    float B_12 = (*(Bp + baseB + 12));
                    float B_13 = (*(Bp + baseB + 13));
                    float B_14 = (*(Bp + baseB + 14));
                    float B_15 = (*(Bp + baseB + 15));
                    float B_16 = (*(Bp + baseB + 16));
                    float B_17 = (*(Bp + baseB + 17));
                    float B_18 = (*(Bp + baseB + 18));
                    float B_19 = (*(Bp + baseB + 19));
                    float B_20 = (*(Bp + baseB + 20));
                    float B_21 = (*(Bp + baseB + 21));
                    float B_22 = (*(Bp + baseB + 22));
                    float B_23 = (*(Bp + baseB + 23));
                    float B_24 = (*(Bp + baseB + 24));
                    float B_25 = (*(Bp + baseB + 25));
                    float B_26 = (*(Bp + baseB + 26));
                    float B_27 = (*(Bp + baseB + 27));
                    float B_28 = (*(Bp + baseB + 28));
                    float B_29 = (*(Bp + baseB + 29));
                    float B_30 = (*(Bp + baseB + 30));
                    float B_31 = (*(Bp + baseB + 31));
                    sum0_0 += A_0 * B_0;
                    sum1_0 += A_0 * B_1;
                    sum2_0 += A_0 * B_2;
                    sum3_0 += A_0 * B_3;
                    sum4_0 += A_0 * B_4;
                    sum5_0 += A_0 * B_5;
                    sum6_0 += A_0 * B_6;
                    sum7_0 += A_0 * B_7;
                    sum8_0 += A_0 * B_8;
                    sum9_0 += A_0 * B_9;
                    sum10_0 += A_0 * B_10;
                    sum11_0 += A_0 * B_11;
                    sum12_0 += A_0 * B_12;
                    sum13_0 += A_0 * B_13;
                    sum14_0 += A_0 * B_14;
                    sum15_0 += A_0 * B_15;
                    sum16_0 += A_0 * B_16;
                    sum17_0 += A_0 * B_17;
                    sum18_0 += A_0 * B_18;
                    sum19_0 += A_0 * B_19;
                    sum20_0 += A_0 * B_20;
                    sum21_0 += A_0 * B_21;
                    sum22_0 += A_0 * B_22;
                    sum23_0 += A_0 * B_23;
                    sum24_0 += A_0 * B_24;
                    sum25_0 += A_0 * B_25;
                    sum26_0 += A_0 * B_26;
                    sum27_0 += A_0 * B_27;
                    sum28_0 += A_0 * B_28;
                    sum29_0 += A_0 * B_29;
                    sum30_0 += A_0 * B_30;
                    sum31_0 += A_0 * B_31;
                }
                // 0
                *(Cp + baseC_0 + 0) = sum0_0;
                *(Cp + baseC_0 + 1) = sum1_0;
                *(Cp + baseC_0 + 2) = sum2_0;
                *(Cp + baseC_0 + 3) = sum3_0;
                *(Cp + baseC_0 + 4) = sum4_0;
                *(Cp + baseC_0 + 5) = sum5_0;
                *(Cp + baseC_0 + 6) = sum6_0;
                *(Cp + baseC_0 + 7) = sum7_0;
                *(Cp + baseC_0 + 8) = sum8_0;
                *(Cp + baseC_0 + 9) = sum9_0;
                *(Cp + baseC_0 + 10) = sum10_0;
                *(Cp + baseC_0 + 11) = sum11_0;
                *(Cp + baseC_0 + 12) = sum12_0;
                *(Cp + baseC_0 + 13) = sum13_0;
                *(Cp + baseC_0 + 14) = sum14_0;
                *(Cp + baseC_0 + 15) = sum15_0;
                *(Cp + baseC_0 + 16) = sum16_0;
                *(Cp + baseC_0 + 17) = sum17_0;
                *(Cp + baseC_0 + 18) = sum18_0;
                *(Cp + baseC_0 + 19) = sum19_0;
                *(Cp + baseC_0 + 20) = sum20_0;
                *(Cp + baseC_0 + 21) = sum21_0;
                *(Cp + baseC_0 + 22) = sum22_0;
                *(Cp + baseC_0 + 23) = sum23_0;
                *(Cp + baseC_0 + 24) = sum24_0;
                *(Cp + baseC_0 + 25) = sum25_0;
                *(Cp + baseC_0 + 26) = sum26_0;
                *(Cp + baseC_0 + 27) = sum27_0;
                *(Cp + baseC_0 + 28) = sum28_0;
                *(Cp + baseC_0 + 29) = sum29_0;
                *(Cp + baseC_0 + 30) = sum30_0;
                *(Cp + baseC_0 + 31) = sum31_0;
            }
        }
    }

    static unsafe void MultiplyBlockUnroll4x16(
            [NoAlias] float* Ap, int Astride,
            [NoAlias] float* Bp, int Bstride,
            [NoAlias] float* Cp, int Cstride,
            int blockSizeM, int blockSizeK,
            int n)
    {
        n = Math.Max(16, n);
        int i = 0;
        for (; i < blockSizeM - 3; i += 4)
        {
            var i_0 = i + 0;
            var i_1 = i + 1;
            var i_2 = i + 2;
            var i_3 = i + 3;

            for (int j = 0; j < n; j += 16)
            {
                int baseC_0 = i_0 * Cstride + j;
                int baseC_1 = i_1 * Cstride + j;
                int baseC_2 = i_2 * Cstride + j;
                int baseC_3 = i_3 * Cstride + j;
                // 0
                float sum0_0 = *(Cp + baseC_0 + 0);
                float sum1_0 = *(Cp + baseC_0 + 1);
                float sum2_0 = *(Cp + baseC_0 + 2);
                float sum3_0 = *(Cp + baseC_0 + 3);
                float sum4_0 = *(Cp + baseC_0 + 4);
                float sum5_0 = *(Cp + baseC_0 + 5);
                float sum6_0 = *(Cp + baseC_0 + 6);
                float sum7_0 = *(Cp + baseC_0 + 7);
                float sum8_0 = *(Cp + baseC_0 + 8);
                float sum9_0 = *(Cp + baseC_0 + 9);
                float sum10_0 = *(Cp + baseC_0 + 10);
                float sum11_0 = *(Cp + baseC_0 + 11);
                float sum12_0 = *(Cp + baseC_0 + 12);
                float sum13_0 = *(Cp + baseC_0 + 13);
                float sum14_0 = *(Cp + baseC_0 + 14);
                float sum15_0 = *(Cp + baseC_0 + 15);
                // 1
                float sum0_1 = *(Cp + baseC_1 + 0);
                float sum1_1 = *(Cp + baseC_1 + 1);
                float sum2_1 = *(Cp + baseC_1 + 2);
                float sum3_1 = *(Cp + baseC_1 + 3);
                float sum4_1 = *(Cp + baseC_1 + 4);
                float sum5_1 = *(Cp + baseC_1 + 5);
                float sum6_1 = *(Cp + baseC_1 + 6);
                float sum7_1 = *(Cp + baseC_1 + 7);
                float sum8_1 = *(Cp + baseC_1 + 8);
                float sum9_1 = *(Cp + baseC_1 + 9);
                float sum10_1 = *(Cp + baseC_1 + 10);
                float sum11_1 = *(Cp + baseC_1 + 11);
                float sum12_1 = *(Cp + baseC_1 + 12);
                float sum13_1 = *(Cp + baseC_1 + 13);
                float sum14_1 = *(Cp + baseC_1 + 14);
                float sum15_1 = *(Cp + baseC_1 + 15);
                // 2
                float sum0_2 = *(Cp + baseC_2 + 0);
                float sum1_2 = *(Cp + baseC_2 + 1);
                float sum2_2 = *(Cp + baseC_2 + 2);
                float sum3_2 = *(Cp + baseC_2 + 3);
                float sum4_2 = *(Cp + baseC_2 + 4);
                float sum5_2 = *(Cp + baseC_2 + 5);
                float sum6_2 = *(Cp + baseC_2 + 6);
                float sum7_2 = *(Cp + baseC_2 + 7);
                float sum8_2 = *(Cp + baseC_2 + 8);
                float sum9_2 = *(Cp + baseC_2 + 9);
                float sum10_2 = *(Cp + baseC_2 + 10);
                float sum11_2 = *(Cp + baseC_2 + 11);
                float sum12_2 = *(Cp + baseC_2 + 12);
                float sum13_2 = *(Cp + baseC_2 + 13);
                float sum14_2 = *(Cp + baseC_2 + 14);
                float sum15_2 = *(Cp + baseC_2 + 15);
                // 3
                float sum0_3 = *(Cp + baseC_3 + 0);
                float sum1_3 = *(Cp + baseC_3 + 1);
                float sum2_3 = *(Cp + baseC_3 + 2);
                float sum3_3 = *(Cp + baseC_3 + 3);
                float sum4_3 = *(Cp + baseC_3 + 4);
                float sum5_3 = *(Cp + baseC_3 + 5);
                float sum6_3 = *(Cp + baseC_3 + 6);
                float sum7_3 = *(Cp + baseC_3 + 7);
                float sum8_3 = *(Cp + baseC_3 + 8);
                float sum9_3 = *(Cp + baseC_3 + 9);
                float sum10_3 = *(Cp + baseC_3 + 10);
                float sum11_3 = *(Cp + baseC_3 + 11);
                float sum12_3 = *(Cp + baseC_3 + 12);
                float sum13_3 = *(Cp + baseC_3 + 13);
                float sum14_3 = *(Cp + baseC_3 + 14);
                float sum15_3 = *(Cp + baseC_3 + 15);

                for (int l = 0; l < blockSizeK; l++)
                {
                    float A_0 = *(Ap + i_0 * Astride + l);
                    float A_1 = *(Ap + i_1 * Astride + l);
                    float A_2 = *(Ap + i_2 * Astride + l);
                    float A_3 = *(Ap + i_3 * Astride + l);
                    int baseB = l * Bstride + j;
                    float B_0 = (*(Bp + baseB + 0));
                    float B_1 = (*(Bp + baseB + 1));
                    float B_2 = (*(Bp + baseB + 2));
                    float B_3 = (*(Bp + baseB + 3));
                    float B_4 = (*(Bp + baseB + 4));
                    float B_5 = (*(Bp + baseB + 5));
                    float B_6 = (*(Bp + baseB + 6));
                    float B_7 = (*(Bp + baseB + 7));
                    float B_8 = (*(Bp + baseB + 8));
                    float B_9 = (*(Bp + baseB + 9));
                    float B_10 = (*(Bp + baseB + 10));
                    float B_11 = (*(Bp + baseB + 11));
                    float B_12 = (*(Bp + baseB + 12));
                    float B_13 = (*(Bp + baseB + 13));
                    float B_14 = (*(Bp + baseB + 14));
                    float B_15 = (*(Bp + baseB + 15));
                    sum0_0 += A_0 * B_0;	sum0_1 += A_1 * B_0;	sum0_2 += A_2 * B_0;	sum0_3 += A_3 * B_0;
                    sum1_0 += A_0 * B_1;	sum1_1 += A_1 * B_1;	sum1_2 += A_2 * B_1;	sum1_3 += A_3 * B_1;
                    sum2_0 += A_0 * B_2;	sum2_1 += A_1 * B_2;	sum2_2 += A_2 * B_2;	sum2_3 += A_3 * B_2;
                    sum3_0 += A_0 * B_3;	sum3_1 += A_1 * B_3;	sum3_2 += A_2 * B_3;	sum3_3 += A_3 * B_3;
                    sum4_0 += A_0 * B_4;	sum4_1 += A_1 * B_4;	sum4_2 += A_2 * B_4;	sum4_3 += A_3 * B_4;
                    sum5_0 += A_0 * B_5;	sum5_1 += A_1 * B_5;	sum5_2 += A_2 * B_5;	sum5_3 += A_3 * B_5;
                    sum6_0 += A_0 * B_6;	sum6_1 += A_1 * B_6;	sum6_2 += A_2 * B_6;	sum6_3 += A_3 * B_6;
                    sum7_0 += A_0 * B_7;	sum7_1 += A_1 * B_7;	sum7_2 += A_2 * B_7;	sum7_3 += A_3 * B_7;
                    sum8_0 += A_0 * B_8;	sum8_1 += A_1 * B_8;	sum8_2 += A_2 * B_8;	sum8_3 += A_3 * B_8;
                    sum9_0 += A_0 * B_9;	sum9_1 += A_1 * B_9;	sum9_2 += A_2 * B_9;	sum9_3 += A_3 * B_9;
                    sum10_0 += A_0 * B_10;	sum10_1 += A_1 * B_10;	sum10_2 += A_2 * B_10;	sum10_3 += A_3 * B_10;
                    sum11_0 += A_0 * B_11;	sum11_1 += A_1 * B_11;	sum11_2 += A_2 * B_11;	sum11_3 += A_3 * B_11;
                    sum12_0 += A_0 * B_12;	sum12_1 += A_1 * B_12;	sum12_2 += A_2 * B_12;	sum12_3 += A_3 * B_12;
                    sum13_0 += A_0 * B_13;	sum13_1 += A_1 * B_13;	sum13_2 += A_2 * B_13;	sum13_3 += A_3 * B_13;
                    sum14_0 += A_0 * B_14;	sum14_1 += A_1 * B_14;	sum14_2 += A_2 * B_14;	sum14_3 += A_3 * B_14;
                    sum15_0 += A_0 * B_15;	sum15_1 += A_1 * B_15;	sum15_2 += A_2 * B_15;	sum15_3 += A_3 * B_15;
                }
                // 0
                *(Cp + baseC_0 + 0) = sum0_0;
                *(Cp + baseC_0 + 1) = sum1_0;
                *(Cp + baseC_0 + 2) = sum2_0;
                *(Cp + baseC_0 + 3) = sum3_0;
                *(Cp + baseC_0 + 4) = sum4_0;
                *(Cp + baseC_0 + 5) = sum5_0;
                *(Cp + baseC_0 + 6) = sum6_0;
                *(Cp + baseC_0 + 7) = sum7_0;
                *(Cp + baseC_0 + 8) = sum8_0;
                *(Cp + baseC_0 + 9) = sum9_0;
                *(Cp + baseC_0 + 10) = sum10_0;
                *(Cp + baseC_0 + 11) = sum11_0;
                *(Cp + baseC_0 + 12) = sum12_0;
                *(Cp + baseC_0 + 13) = sum13_0;
                *(Cp + baseC_0 + 14) = sum14_0;
                *(Cp + baseC_0 + 15) = sum15_0;
                // 1
                *(Cp + baseC_1 + 0) = sum0_1;
                *(Cp + baseC_1 + 1) = sum1_1;
                *(Cp + baseC_1 + 2) = sum2_1;
                *(Cp + baseC_1 + 3) = sum3_1;
                *(Cp + baseC_1 + 4) = sum4_1;
                *(Cp + baseC_1 + 5) = sum5_1;
                *(Cp + baseC_1 + 6) = sum6_1;
                *(Cp + baseC_1 + 7) = sum7_1;
                *(Cp + baseC_1 + 8) = sum8_1;
                *(Cp + baseC_1 + 9) = sum9_1;
                *(Cp + baseC_1 + 10) = sum10_1;
                *(Cp + baseC_1 + 11) = sum11_1;
                *(Cp + baseC_1 + 12) = sum12_1;
                *(Cp + baseC_1 + 13) = sum13_1;
                *(Cp + baseC_1 + 14) = sum14_1;
                *(Cp + baseC_1 + 15) = sum15_1;
                // 2
                *(Cp + baseC_2 + 0) = sum0_2;
                *(Cp + baseC_2 + 1) = sum1_2;
                *(Cp + baseC_2 + 2) = sum2_2;
                *(Cp + baseC_2 + 3) = sum3_2;
                *(Cp + baseC_2 + 4) = sum4_2;
                *(Cp + baseC_2 + 5) = sum5_2;
                *(Cp + baseC_2 + 6) = sum6_2;
                *(Cp + baseC_2 + 7) = sum7_2;
                *(Cp + baseC_2 + 8) = sum8_2;
                *(Cp + baseC_2 + 9) = sum9_2;
                *(Cp + baseC_2 + 10) = sum10_2;
                *(Cp + baseC_2 + 11) = sum11_2;
                *(Cp + baseC_2 + 12) = sum12_2;
                *(Cp + baseC_2 + 13) = sum13_2;
                *(Cp + baseC_2 + 14) = sum14_2;
                *(Cp + baseC_2 + 15) = sum15_2;
                // 3
                *(Cp + baseC_3 + 0) = sum0_3;
                *(Cp + baseC_3 + 1) = sum1_3;
                *(Cp + baseC_3 + 2) = sum2_3;
                *(Cp + baseC_3 + 3) = sum3_3;
                *(Cp + baseC_3 + 4) = sum4_3;
                *(Cp + baseC_3 + 5) = sum5_3;
                *(Cp + baseC_3 + 6) = sum6_3;
                *(Cp + baseC_3 + 7) = sum7_3;
                *(Cp + baseC_3 + 8) = sum8_3;
                *(Cp + baseC_3 + 9) = sum9_3;
                *(Cp + baseC_3 + 10) = sum10_3;
                *(Cp + baseC_3 + 11) = sum11_3;
                *(Cp + baseC_3 + 12) = sum12_3;
                *(Cp + baseC_3 + 13) = sum13_3;
                *(Cp + baseC_3 + 14) = sum14_3;
                *(Cp + baseC_3 + 15) = sum15_3;
            }
        }
        for (; i < blockSizeM - 2; i += 3)
        {
            var i_0 = i + 0;
            var i_1 = i + 1;
            var i_2 = i + 2;

            for (int j = 0; j < n; j += 16)
            {
                int baseC_0 = i_0 * Cstride + j;
                int baseC_1 = i_1 * Cstride + j;
                int baseC_2 = i_2 * Cstride + j;
                // 0
                float sum0_0 = *(Cp + baseC_0 + 0);
                float sum1_0 = *(Cp + baseC_0 + 1);
                float sum2_0 = *(Cp + baseC_0 + 2);
                float sum3_0 = *(Cp + baseC_0 + 3);
                float sum4_0 = *(Cp + baseC_0 + 4);
                float sum5_0 = *(Cp + baseC_0 + 5);
                float sum6_0 = *(Cp + baseC_0 + 6);
                float sum7_0 = *(Cp + baseC_0 + 7);
                float sum8_0 = *(Cp + baseC_0 + 8);
                float sum9_0 = *(Cp + baseC_0 + 9);
                float sum10_0 = *(Cp + baseC_0 + 10);
                float sum11_0 = *(Cp + baseC_0 + 11);
                float sum12_0 = *(Cp + baseC_0 + 12);
                float sum13_0 = *(Cp + baseC_0 + 13);
                float sum14_0 = *(Cp + baseC_0 + 14);
                float sum15_0 = *(Cp + baseC_0 + 15);
                // 1
                float sum0_1 = *(Cp + baseC_1 + 0);
                float sum1_1 = *(Cp + baseC_1 + 1);
                float sum2_1 = *(Cp + baseC_1 + 2);
                float sum3_1 = *(Cp + baseC_1 + 3);
                float sum4_1 = *(Cp + baseC_1 + 4);
                float sum5_1 = *(Cp + baseC_1 + 5);
                float sum6_1 = *(Cp + baseC_1 + 6);
                float sum7_1 = *(Cp + baseC_1 + 7);
                float sum8_1 = *(Cp + baseC_1 + 8);
                float sum9_1 = *(Cp + baseC_1 + 9);
                float sum10_1 = *(Cp + baseC_1 + 10);
                float sum11_1 = *(Cp + baseC_1 + 11);
                float sum12_1 = *(Cp + baseC_1 + 12);
                float sum13_1 = *(Cp + baseC_1 + 13);
                float sum14_1 = *(Cp + baseC_1 + 14);
                float sum15_1 = *(Cp + baseC_1 + 15);
                // 2
                float sum0_2 = *(Cp + baseC_2 + 0);
                float sum1_2 = *(Cp + baseC_2 + 1);
                float sum2_2 = *(Cp + baseC_2 + 2);
                float sum3_2 = *(Cp + baseC_2 + 3);
                float sum4_2 = *(Cp + baseC_2 + 4);
                float sum5_2 = *(Cp + baseC_2 + 5);
                float sum6_2 = *(Cp + baseC_2 + 6);
                float sum7_2 = *(Cp + baseC_2 + 7);
                float sum8_2 = *(Cp + baseC_2 + 8);
                float sum9_2 = *(Cp + baseC_2 + 9);
                float sum10_2 = *(Cp + baseC_2 + 10);
                float sum11_2 = *(Cp + baseC_2 + 11);
                float sum12_2 = *(Cp + baseC_2 + 12);
                float sum13_2 = *(Cp + baseC_2 + 13);
                float sum14_2 = *(Cp + baseC_2 + 14);
                float sum15_2 = *(Cp + baseC_2 + 15);

                for (int l = 0; l < blockSizeK; l++)
                {
                    float A_0 = *(Ap + i_0 * Astride + l);
                    float A_1 = *(Ap + i_1 * Astride + l);
                    float A_2 = *(Ap + i_2 * Astride + l);
                    int baseB = l * Bstride + j;
                    float B_0 = (*(Bp + baseB + 0));
                    float B_1 = (*(Bp + baseB + 1));
                    float B_2 = (*(Bp + baseB + 2));
                    float B_3 = (*(Bp + baseB + 3));
                    float B_4 = (*(Bp + baseB + 4));
                    float B_5 = (*(Bp + baseB + 5));
                    float B_6 = (*(Bp + baseB + 6));
                    float B_7 = (*(Bp + baseB + 7));
                    float B_8 = (*(Bp + baseB + 8));
                    float B_9 = (*(Bp + baseB + 9));
                    float B_10 = (*(Bp + baseB + 10));
                    float B_11 = (*(Bp + baseB + 11));
                    float B_12 = (*(Bp + baseB + 12));
                    float B_13 = (*(Bp + baseB + 13));
                    float B_14 = (*(Bp + baseB + 14));
                    float B_15 = (*(Bp + baseB + 15));
                    sum0_0 += A_0 * B_0;	sum0_1 += A_1 * B_0;	sum0_2 += A_2 * B_0;
                    sum1_0 += A_0 * B_1;	sum1_1 += A_1 * B_1;	sum1_2 += A_2 * B_1;
                    sum2_0 += A_0 * B_2;	sum2_1 += A_1 * B_2;	sum2_2 += A_2 * B_2;
                    sum3_0 += A_0 * B_3;	sum3_1 += A_1 * B_3;	sum3_2 += A_2 * B_3;
                    sum4_0 += A_0 * B_4;	sum4_1 += A_1 * B_4;	sum4_2 += A_2 * B_4;
                    sum5_0 += A_0 * B_5;	sum5_1 += A_1 * B_5;	sum5_2 += A_2 * B_5;
                    sum6_0 += A_0 * B_6;	sum6_1 += A_1 * B_6;	sum6_2 += A_2 * B_6;
                    sum7_0 += A_0 * B_7;	sum7_1 += A_1 * B_7;	sum7_2 += A_2 * B_7;
                    sum8_0 += A_0 * B_8;	sum8_1 += A_1 * B_8;	sum8_2 += A_2 * B_8;
                    sum9_0 += A_0 * B_9;	sum9_1 += A_1 * B_9;	sum9_2 += A_2 * B_9;
                    sum10_0 += A_0 * B_10;	sum10_1 += A_1 * B_10;	sum10_2 += A_2 * B_10;
                    sum11_0 += A_0 * B_11;	sum11_1 += A_1 * B_11;	sum11_2 += A_2 * B_11;
                    sum12_0 += A_0 * B_12;	sum12_1 += A_1 * B_12;	sum12_2 += A_2 * B_12;
                    sum13_0 += A_0 * B_13;	sum13_1 += A_1 * B_13;	sum13_2 += A_2 * B_13;
                    sum14_0 += A_0 * B_14;	sum14_1 += A_1 * B_14;	sum14_2 += A_2 * B_14;
                    sum15_0 += A_0 * B_15;	sum15_1 += A_1 * B_15;	sum15_2 += A_2 * B_15;
                }
                // 0
                *(Cp + baseC_0 + 0) = sum0_0;
                *(Cp + baseC_0 + 1) = sum1_0;
                *(Cp + baseC_0 + 2) = sum2_0;
                *(Cp + baseC_0 + 3) = sum3_0;
                *(Cp + baseC_0 + 4) = sum4_0;
                *(Cp + baseC_0 + 5) = sum5_0;
                *(Cp + baseC_0 + 6) = sum6_0;
                *(Cp + baseC_0 + 7) = sum7_0;
                *(Cp + baseC_0 + 8) = sum8_0;
                *(Cp + baseC_0 + 9) = sum9_0;
                *(Cp + baseC_0 + 10) = sum10_0;
                *(Cp + baseC_0 + 11) = sum11_0;
                *(Cp + baseC_0 + 12) = sum12_0;
                *(Cp + baseC_0 + 13) = sum13_0;
                *(Cp + baseC_0 + 14) = sum14_0;
                *(Cp + baseC_0 + 15) = sum15_0;
                // 1
                *(Cp + baseC_1 + 0) = sum0_1;
                *(Cp + baseC_1 + 1) = sum1_1;
                *(Cp + baseC_1 + 2) = sum2_1;
                *(Cp + baseC_1 + 3) = sum3_1;
                *(Cp + baseC_1 + 4) = sum4_1;
                *(Cp + baseC_1 + 5) = sum5_1;
                *(Cp + baseC_1 + 6) = sum6_1;
                *(Cp + baseC_1 + 7) = sum7_1;
                *(Cp + baseC_1 + 8) = sum8_1;
                *(Cp + baseC_1 + 9) = sum9_1;
                *(Cp + baseC_1 + 10) = sum10_1;
                *(Cp + baseC_1 + 11) = sum11_1;
                *(Cp + baseC_1 + 12) = sum12_1;
                *(Cp + baseC_1 + 13) = sum13_1;
                *(Cp + baseC_1 + 14) = sum14_1;
                *(Cp + baseC_1 + 15) = sum15_1;
                // 2
                *(Cp + baseC_2 + 0) = sum0_2;
                *(Cp + baseC_2 + 1) = sum1_2;
                *(Cp + baseC_2 + 2) = sum2_2;
                *(Cp + baseC_2 + 3) = sum3_2;
                *(Cp + baseC_2 + 4) = sum4_2;
                *(Cp + baseC_2 + 5) = sum5_2;
                *(Cp + baseC_2 + 6) = sum6_2;
                *(Cp + baseC_2 + 7) = sum7_2;
                *(Cp + baseC_2 + 8) = sum8_2;
                *(Cp + baseC_2 + 9) = sum9_2;
                *(Cp + baseC_2 + 10) = sum10_2;
                *(Cp + baseC_2 + 11) = sum11_2;
                *(Cp + baseC_2 + 12) = sum12_2;
                *(Cp + baseC_2 + 13) = sum13_2;
                *(Cp + baseC_2 + 14) = sum14_2;
                *(Cp + baseC_2 + 15) = sum15_2;
            }
        }
        for (; i < blockSizeM - 1; i += 2)
        {
            var i_0 = i + 0;
            var i_1 = i + 1;

            for (int j = 0; j < n; j += 16)
            {
                int baseC_0 = i_0 * Cstride + j;
                int baseC_1 = i_1 * Cstride + j;
                // 0
                float sum0_0 = *(Cp + baseC_0 + 0);
                float sum1_0 = *(Cp + baseC_0 + 1);
                float sum2_0 = *(Cp + baseC_0 + 2);
                float sum3_0 = *(Cp + baseC_0 + 3);
                float sum4_0 = *(Cp + baseC_0 + 4);
                float sum5_0 = *(Cp + baseC_0 + 5);
                float sum6_0 = *(Cp + baseC_0 + 6);
                float sum7_0 = *(Cp + baseC_0 + 7);
                float sum8_0 = *(Cp + baseC_0 + 8);
                float sum9_0 = *(Cp + baseC_0 + 9);
                float sum10_0 = *(Cp + baseC_0 + 10);
                float sum11_0 = *(Cp + baseC_0 + 11);
                float sum12_0 = *(Cp + baseC_0 + 12);
                float sum13_0 = *(Cp + baseC_0 + 13);
                float sum14_0 = *(Cp + baseC_0 + 14);
                float sum15_0 = *(Cp + baseC_0 + 15);
                // 1
                float sum0_1 = *(Cp + baseC_1 + 0);
                float sum1_1 = *(Cp + baseC_1 + 1);
                float sum2_1 = *(Cp + baseC_1 + 2);
                float sum3_1 = *(Cp + baseC_1 + 3);
                float sum4_1 = *(Cp + baseC_1 + 4);
                float sum5_1 = *(Cp + baseC_1 + 5);
                float sum6_1 = *(Cp + baseC_1 + 6);
                float sum7_1 = *(Cp + baseC_1 + 7);
                float sum8_1 = *(Cp + baseC_1 + 8);
                float sum9_1 = *(Cp + baseC_1 + 9);
                float sum10_1 = *(Cp + baseC_1 + 10);
                float sum11_1 = *(Cp + baseC_1 + 11);
                float sum12_1 = *(Cp + baseC_1 + 12);
                float sum13_1 = *(Cp + baseC_1 + 13);
                float sum14_1 = *(Cp + baseC_1 + 14);
                float sum15_1 = *(Cp + baseC_1 + 15);

                for (int l = 0; l < blockSizeK; l++)
                {
                    float A_0 = *(Ap + i_0 * Astride + l);
                    float A_1 = *(Ap + i_1 * Astride + l);
                    int baseB = l * Bstride + j;
                    float B_0 = (*(Bp + baseB + 0));
                    float B_1 = (*(Bp + baseB + 1));
                    float B_2 = (*(Bp + baseB + 2));
                    float B_3 = (*(Bp + baseB + 3));
                    float B_4 = (*(Bp + baseB + 4));
                    float B_5 = (*(Bp + baseB + 5));
                    float B_6 = (*(Bp + baseB + 6));
                    float B_7 = (*(Bp + baseB + 7));
                    float B_8 = (*(Bp + baseB + 8));
                    float B_9 = (*(Bp + baseB + 9));
                    float B_10 = (*(Bp + baseB + 10));
                    float B_11 = (*(Bp + baseB + 11));
                    float B_12 = (*(Bp + baseB + 12));
                    float B_13 = (*(Bp + baseB + 13));
                    float B_14 = (*(Bp + baseB + 14));
                    float B_15 = (*(Bp + baseB + 15));
                    sum0_0 += A_0 * B_0;	sum0_1 += A_1 * B_0;
                    sum1_0 += A_0 * B_1;	sum1_1 += A_1 * B_1;
                    sum2_0 += A_0 * B_2;	sum2_1 += A_1 * B_2;
                    sum3_0 += A_0 * B_3;	sum3_1 += A_1 * B_3;
                    sum4_0 += A_0 * B_4;	sum4_1 += A_1 * B_4;
                    sum5_0 += A_0 * B_5;	sum5_1 += A_1 * B_5;
                    sum6_0 += A_0 * B_6;	sum6_1 += A_1 * B_6;
                    sum7_0 += A_0 * B_7;	sum7_1 += A_1 * B_7;
                    sum8_0 += A_0 * B_8;	sum8_1 += A_1 * B_8;
                    sum9_0 += A_0 * B_9;	sum9_1 += A_1 * B_9;
                    sum10_0 += A_0 * B_10;	sum10_1 += A_1 * B_10;
                    sum11_0 += A_0 * B_11;	sum11_1 += A_1 * B_11;
                    sum12_0 += A_0 * B_12;	sum12_1 += A_1 * B_12;
                    sum13_0 += A_0 * B_13;	sum13_1 += A_1 * B_13;
                    sum14_0 += A_0 * B_14;	sum14_1 += A_1 * B_14;
                    sum15_0 += A_0 * B_15;	sum15_1 += A_1 * B_15;
                }
                // 0
                *(Cp + baseC_0 + 0) = sum0_0;
                *(Cp + baseC_0 + 1) = sum1_0;
                *(Cp + baseC_0 + 2) = sum2_0;
                *(Cp + baseC_0 + 3) = sum3_0;
                *(Cp + baseC_0 + 4) = sum4_0;
                *(Cp + baseC_0 + 5) = sum5_0;
                *(Cp + baseC_0 + 6) = sum6_0;
                *(Cp + baseC_0 + 7) = sum7_0;
                *(Cp + baseC_0 + 8) = sum8_0;
                *(Cp + baseC_0 + 9) = sum9_0;
                *(Cp + baseC_0 + 10) = sum10_0;
                *(Cp + baseC_0 + 11) = sum11_0;
                *(Cp + baseC_0 + 12) = sum12_0;
                *(Cp + baseC_0 + 13) = sum13_0;
                *(Cp + baseC_0 + 14) = sum14_0;
                *(Cp + baseC_0 + 15) = sum15_0;
                // 1
                *(Cp + baseC_1 + 0) = sum0_1;
                *(Cp + baseC_1 + 1) = sum1_1;
                *(Cp + baseC_1 + 2) = sum2_1;
                *(Cp + baseC_1 + 3) = sum3_1;
                *(Cp + baseC_1 + 4) = sum4_1;
                *(Cp + baseC_1 + 5) = sum5_1;
                *(Cp + baseC_1 + 6) = sum6_1;
                *(Cp + baseC_1 + 7) = sum7_1;
                *(Cp + baseC_1 + 8) = sum8_1;
                *(Cp + baseC_1 + 9) = sum9_1;
                *(Cp + baseC_1 + 10) = sum10_1;
                *(Cp + baseC_1 + 11) = sum11_1;
                *(Cp + baseC_1 + 12) = sum12_1;
                *(Cp + baseC_1 + 13) = sum13_1;
                *(Cp + baseC_1 + 14) = sum14_1;
                *(Cp + baseC_1 + 15) = sum15_1;
            }
        }
        for (; i < blockSizeM - 0; i += 1)
        {
            var i_0 = i + 0;

            for (int j = 0; j < n; j += 16)
            {
                int baseC_0 = i_0 * Cstride + j;
                // 0
                float sum0_0 = *(Cp + baseC_0 + 0);
                float sum1_0 = *(Cp + baseC_0 + 1);
                float sum2_0 = *(Cp + baseC_0 + 2);
                float sum3_0 = *(Cp + baseC_0 + 3);
                float sum4_0 = *(Cp + baseC_0 + 4);
                float sum5_0 = *(Cp + baseC_0 + 5);
                float sum6_0 = *(Cp + baseC_0 + 6);
                float sum7_0 = *(Cp + baseC_0 + 7);
                float sum8_0 = *(Cp + baseC_0 + 8);
                float sum9_0 = *(Cp + baseC_0 + 9);
                float sum10_0 = *(Cp + baseC_0 + 10);
                float sum11_0 = *(Cp + baseC_0 + 11);
                float sum12_0 = *(Cp + baseC_0 + 12);
                float sum13_0 = *(Cp + baseC_0 + 13);
                float sum14_0 = *(Cp + baseC_0 + 14);
                float sum15_0 = *(Cp + baseC_0 + 15);

                for (int l = 0; l < blockSizeK; l++)
                {
                    float A_0 = *(Ap + i_0 * Astride + l);
                    int baseB = l * Bstride + j;
                    float B_0 = (*(Bp + baseB + 0));
                    float B_1 = (*(Bp + baseB + 1));
                    float B_2 = (*(Bp + baseB + 2));
                    float B_3 = (*(Bp + baseB + 3));
                    float B_4 = (*(Bp + baseB + 4));
                    float B_5 = (*(Bp + baseB + 5));
                    float B_6 = (*(Bp + baseB + 6));
                    float B_7 = (*(Bp + baseB + 7));
                    float B_8 = (*(Bp + baseB + 8));
                    float B_9 = (*(Bp + baseB + 9));
                    float B_10 = (*(Bp + baseB + 10));
                    float B_11 = (*(Bp + baseB + 11));
                    float B_12 = (*(Bp + baseB + 12));
                    float B_13 = (*(Bp + baseB + 13));
                    float B_14 = (*(Bp + baseB + 14));
                    float B_15 = (*(Bp + baseB + 15));
                    sum0_0 += A_0 * B_0;
                    sum1_0 += A_0 * B_1;
                    sum2_0 += A_0 * B_2;
                    sum3_0 += A_0 * B_3;
                    sum4_0 += A_0 * B_4;
                    sum5_0 += A_0 * B_5;
                    sum6_0 += A_0 * B_6;
                    sum7_0 += A_0 * B_7;
                    sum8_0 += A_0 * B_8;
                    sum9_0 += A_0 * B_9;
                    sum10_0 += A_0 * B_10;
                    sum11_0 += A_0 * B_11;
                    sum12_0 += A_0 * B_12;
                    sum13_0 += A_0 * B_13;
                    sum14_0 += A_0 * B_14;
                    sum15_0 += A_0 * B_15;
                }
                // 0
                *(Cp + baseC_0 + 0) = sum0_0;
                *(Cp + baseC_0 + 1) = sum1_0;
                *(Cp + baseC_0 + 2) = sum2_0;
                *(Cp + baseC_0 + 3) = sum3_0;
                *(Cp + baseC_0 + 4) = sum4_0;
                *(Cp + baseC_0 + 5) = sum5_0;
                *(Cp + baseC_0 + 6) = sum6_0;
                *(Cp + baseC_0 + 7) = sum7_0;
                *(Cp + baseC_0 + 8) = sum8_0;
                *(Cp + baseC_0 + 9) = sum9_0;
                *(Cp + baseC_0 + 10) = sum10_0;
                *(Cp + baseC_0 + 11) = sum11_0;
                *(Cp + baseC_0 + 12) = sum12_0;
                *(Cp + baseC_0 + 13) = sum13_0;
                *(Cp + baseC_0 + 14) = sum14_0;
                *(Cp + baseC_0 + 15) = sum15_0;
            }
        }
    }

    static unsafe void MultiplyBlockUnroll4x16I(
        [NoAlias] float* Ap, int Astride,
        [NoAlias] float* Bp, int Bstride,
        [NoAlias] float* Cp, int Cstride,
        int blockSizeM, int blockSizeK,
        int n)
    {
        n = Math.Max(16, n);
        int i = 0;
        for (; i < blockSizeM - 3; i += 4)
        {
            var i_0 = i + 0;
            var i_1 = i + 1;
            var i_2 = i + 2;
            var i_3 = i + 3;

            for (int j = 0; j < n; j += 16)
            {
                int baseC_0 = i_0 * Cstride + j;
                int baseC_1 = i_1 * Cstride + j;
                int baseC_2 = i_2 * Cstride + j;
                int baseC_3 = i_3 * Cstride + j;

                // row 0
                v256 gamma_0_0 = mm256_loadu_ps(Cp + baseC_0 + 0);
                v256 gamma_0_8 = mm256_loadu_ps(Cp + baseC_0 + 8);
                // row 1
                v256 gamma_1_0 = mm256_loadu_ps(Cp + baseC_1 + 0);
                v256 gamma_1_8 = mm256_loadu_ps(Cp + baseC_1 + 8);
                // row 2
                v256 gamma_2_0 = mm256_loadu_ps(Cp + baseC_2 + 0);
                v256 gamma_2_8 = mm256_loadu_ps(Cp + baseC_2 + 8);
                // row 3
                v256 gamma_3_0 = mm256_loadu_ps(Cp + baseC_3 + 0);
                v256 gamma_3_8 = mm256_loadu_ps(Cp + baseC_3 + 8);

                for (int l = 0; l < blockSizeK; l++)
                {
                    v256 alpha_0_p = mm256_broadcast_ss(Ap + i_0 * Astride + l);
                    v256 alpha_1_p = mm256_broadcast_ss(Ap + i_1 * Astride + l);
                    v256 alpha_2_p = mm256_broadcast_ss(Ap + i_2 * Astride + l);
                    v256 alpha_3_p = mm256_broadcast_ss(Ap + i_3 * Astride + l);

                    v256 beta_p_0 = mm256_loadu_ps(Bp + l * Bstride + j + 0);
                    v256 beta_p_8 = mm256_loadu_ps(Bp + l * Bstride + j + 8);

                    gamma_0_0 = mm256_fmadd_ps(alpha_0_p, beta_p_0, gamma_0_0);
                    gamma_1_0 = mm256_fmadd_ps(alpha_1_p, beta_p_0, gamma_1_0);
                    gamma_2_0 = mm256_fmadd_ps(alpha_2_p, beta_p_0, gamma_2_0);
                    gamma_3_0 = mm256_fmadd_ps(alpha_3_p, beta_p_0, gamma_3_0);
                    gamma_0_8 = mm256_fmadd_ps(alpha_0_p, beta_p_8, gamma_0_8);
                    gamma_1_8 = mm256_fmadd_ps(alpha_1_p, beta_p_8, gamma_1_8);
                    gamma_2_8 = mm256_fmadd_ps(alpha_2_p, beta_p_8, gamma_2_8);
                    gamma_3_8 = mm256_fmadd_ps(alpha_3_p, beta_p_8, gamma_3_8);
                }
                // row 0
                mm256_storeu_ps(Cp + baseC_0 + 0, gamma_0_0);
                mm256_storeu_ps(Cp + baseC_0 + 8, gamma_0_8);
                // row 1
                mm256_storeu_ps(Cp + baseC_1 + 0, gamma_1_0);
                mm256_storeu_ps(Cp + baseC_1 + 8, gamma_1_8);
                // row 2
                mm256_storeu_ps(Cp + baseC_2 + 0, gamma_2_0);
                mm256_storeu_ps(Cp + baseC_2 + 8, gamma_2_8);
                // row 3
                mm256_storeu_ps(Cp + baseC_3 + 0, gamma_3_0);
                mm256_storeu_ps(Cp + baseC_3 + 8, gamma_3_8);
            }
        }
        for (; i < blockSizeM - 2; i += 3)
        {
            var i_0 = i + 0;
            var i_1 = i + 1;
            var i_2 = i + 2;

            for (int j = 0; j < n; j += 16)
            {
                int baseC_0 = i_0 * Cstride + j;
                int baseC_1 = i_1 * Cstride + j;
                int baseC_2 = i_2 * Cstride + j;

                // row 0
                v256 gamma_0_0 = mm256_loadu_ps(Cp + baseC_0 + 0);
                v256 gamma_0_8 = mm256_loadu_ps(Cp + baseC_0 + 8);
                // row 1
                v256 gamma_1_0 = mm256_loadu_ps(Cp + baseC_1 + 0);
                v256 gamma_1_8 = mm256_loadu_ps(Cp + baseC_1 + 8);
                // row 2
                v256 gamma_2_0 = mm256_loadu_ps(Cp + baseC_2 + 0);
                v256 gamma_2_8 = mm256_loadu_ps(Cp + baseC_2 + 8);

                for (int l = 0; l < blockSizeK; l++)
                {
                    v256 alpha_0_p = mm256_broadcast_ss(Ap + i_0 * Astride + l);
                    v256 alpha_1_p = mm256_broadcast_ss(Ap + i_1 * Astride + l);
                    v256 alpha_2_p = mm256_broadcast_ss(Ap + i_2 * Astride + l);

                    v256 beta_p_0 = mm256_loadu_ps(Bp + l * Bstride + j + 0);
                    v256 beta_p_8 = mm256_loadu_ps(Bp + l * Bstride + j + 8);

                    gamma_0_0 = mm256_fmadd_ps(alpha_0_p, beta_p_0, gamma_0_0);
                    gamma_1_0 = mm256_fmadd_ps(alpha_1_p, beta_p_0, gamma_1_0);
                    gamma_2_0 = mm256_fmadd_ps(alpha_2_p, beta_p_0, gamma_2_0);
                    gamma_0_8 = mm256_fmadd_ps(alpha_0_p, beta_p_8, gamma_0_8);
                    gamma_1_8 = mm256_fmadd_ps(alpha_1_p, beta_p_8, gamma_1_8);
                    gamma_2_8 = mm256_fmadd_ps(alpha_2_p, beta_p_8, gamma_2_8);
                }
                // row 0
                mm256_storeu_ps(Cp + baseC_0 + 0, gamma_0_0);
                mm256_storeu_ps(Cp + baseC_0 + 8, gamma_0_8);
                // row 1
                mm256_storeu_ps(Cp + baseC_1 + 0, gamma_1_0);
                mm256_storeu_ps(Cp + baseC_1 + 8, gamma_1_8);
                // row 2
                mm256_storeu_ps(Cp + baseC_2 + 0, gamma_2_0);
                mm256_storeu_ps(Cp + baseC_2 + 8, gamma_2_8);
            }
        }
        for (; i < blockSizeM - 1; i += 2)
        {
            var i_0 = i + 0;
            var i_1 = i + 1;

            for (int j = 0; j < n; j += 16)
            {
                int baseC_0 = i_0 * Cstride + j;
                int baseC_1 = i_1 * Cstride + j;

                // row 0
                v256 gamma_0_0 = mm256_loadu_ps(Cp + baseC_0 + 0);
                v256 gamma_0_8 = mm256_loadu_ps(Cp + baseC_0 + 8);
                // row 1
                v256 gamma_1_0 = mm256_loadu_ps(Cp + baseC_1 + 0);
                v256 gamma_1_8 = mm256_loadu_ps(Cp + baseC_1 + 8);

                for (int l = 0; l < blockSizeK; l++)
                {
                    v256 alpha_0_p = mm256_broadcast_ss(Ap + i_0 * Astride + l);
                    v256 alpha_1_p = mm256_broadcast_ss(Ap + i_1 * Astride + l);

                    v256 beta_p_0 = mm256_loadu_ps(Bp + l * Bstride + j + 0);
                    v256 beta_p_8 = mm256_loadu_ps(Bp + l * Bstride + j + 8);

                    gamma_0_0 = mm256_fmadd_ps(alpha_0_p, beta_p_0, gamma_0_0);
                    gamma_1_0 = mm256_fmadd_ps(alpha_1_p, beta_p_0, gamma_1_0);
                    gamma_0_8 = mm256_fmadd_ps(alpha_0_p, beta_p_8, gamma_0_8);
                    gamma_1_8 = mm256_fmadd_ps(alpha_1_p, beta_p_8, gamma_1_8);
                }
                // row 0
                mm256_storeu_ps(Cp + baseC_0 + 0, gamma_0_0);
                mm256_storeu_ps(Cp + baseC_0 + 8, gamma_0_8);
                // row 1
                mm256_storeu_ps(Cp + baseC_1 + 0, gamma_1_0);
                mm256_storeu_ps(Cp + baseC_1 + 8, gamma_1_8);
            }
        }
        for (; i < blockSizeM - 0; i += 1)
        {
            var i_0 = i + 0;

            for (int j = 0; j < n; j += 16)
            {
                int baseC_0 = i_0 * Cstride + j;

                // row 0
                v256 gamma_0_0 = mm256_loadu_ps(Cp + baseC_0 + 0);
                v256 gamma_0_8 = mm256_loadu_ps(Cp + baseC_0 + 8);

                for (int l = 0; l < blockSizeK; l++)
                {
                    v256 alpha_0_p = mm256_broadcast_ss(Ap + i_0 * Astride + l);

                    v256 beta_p_0 = mm256_loadu_ps(Bp + l * Bstride + j + 0);
                    v256 beta_p_8 = mm256_loadu_ps(Bp + l * Bstride + j + 8);

                    gamma_0_0 = mm256_fmadd_ps(alpha_0_p, beta_p_0, gamma_0_0);
                    gamma_0_8 = mm256_fmadd_ps(alpha_0_p, beta_p_8, gamma_0_8);
                }
                // row 0
                mm256_storeu_ps(Cp + baseC_0 + 0, gamma_0_0);
                mm256_storeu_ps(Cp + baseC_0 + 8, gamma_0_8);
            }
        }
    }

    static unsafe void MultiplyBlockUnroll4x24(
            [NoAlias] float* Ap, int Astride,
            [NoAlias] float* Bp, int Bstride,
            [NoAlias] float* Cp, int Cstride,
            int blockSizeM, int blockSizeK,
            int n)
    {
        n = Math.Max(24, n);
        int i = 0;
        for (; i < blockSizeM - 3; i += 4)
        {
            var i_0 = i + 0;
            var i_1 = i + 1;
            var i_2 = i + 2;
            var i_3 = i + 3;

            for (int j = 0; j < n; j += 24)
            {
                int baseC_0 = i_0 * Cstride + j;
                int baseC_1 = i_1 * Cstride + j;
                int baseC_2 = i_2 * Cstride + j;
                int baseC_3 = i_3 * Cstride + j;
                // 0
                float sum0_0 = *(Cp + baseC_0 + 0);
                float sum1_0 = *(Cp + baseC_0 + 1);
                float sum2_0 = *(Cp + baseC_0 + 2);
                float sum3_0 = *(Cp + baseC_0 + 3);
                float sum4_0 = *(Cp + baseC_0 + 4);
                float sum5_0 = *(Cp + baseC_0 + 5);
                float sum6_0 = *(Cp + baseC_0 + 6);
                float sum7_0 = *(Cp + baseC_0 + 7);
                float sum8_0 = *(Cp + baseC_0 + 8);
                float sum9_0 = *(Cp + baseC_0 + 9);
                float sum10_0 = *(Cp + baseC_0 + 10);
                float sum11_0 = *(Cp + baseC_0 + 11);
                float sum12_0 = *(Cp + baseC_0 + 12);
                float sum13_0 = *(Cp + baseC_0 + 13);
                float sum14_0 = *(Cp + baseC_0 + 14);
                float sum15_0 = *(Cp + baseC_0 + 15);
                float sum16_0 = *(Cp + baseC_0 + 16);
                float sum17_0 = *(Cp + baseC_0 + 17);
                float sum18_0 = *(Cp + baseC_0 + 18);
                float sum19_0 = *(Cp + baseC_0 + 19);
                float sum20_0 = *(Cp + baseC_0 + 20);
                float sum21_0 = *(Cp + baseC_0 + 21);
                float sum22_0 = *(Cp + baseC_0 + 22);
                float sum23_0 = *(Cp + baseC_0 + 23);
                // 1
                float sum0_1 = *(Cp + baseC_1 + 0);
                float sum1_1 = *(Cp + baseC_1 + 1);
                float sum2_1 = *(Cp + baseC_1 + 2);
                float sum3_1 = *(Cp + baseC_1 + 3);
                float sum4_1 = *(Cp + baseC_1 + 4);
                float sum5_1 = *(Cp + baseC_1 + 5);
                float sum6_1 = *(Cp + baseC_1 + 6);
                float sum7_1 = *(Cp + baseC_1 + 7);
                float sum8_1 = *(Cp + baseC_1 + 8);
                float sum9_1 = *(Cp + baseC_1 + 9);
                float sum10_1 = *(Cp + baseC_1 + 10);
                float sum11_1 = *(Cp + baseC_1 + 11);
                float sum12_1 = *(Cp + baseC_1 + 12);
                float sum13_1 = *(Cp + baseC_1 + 13);
                float sum14_1 = *(Cp + baseC_1 + 14);
                float sum15_1 = *(Cp + baseC_1 + 15);
                float sum16_1 = *(Cp + baseC_1 + 16);
                float sum17_1 = *(Cp + baseC_1 + 17);
                float sum18_1 = *(Cp + baseC_1 + 18);
                float sum19_1 = *(Cp + baseC_1 + 19);
                float sum20_1 = *(Cp + baseC_1 + 20);
                float sum21_1 = *(Cp + baseC_1 + 21);
                float sum22_1 = *(Cp + baseC_1 + 22);
                float sum23_1 = *(Cp + baseC_1 + 23);
                // 2
                float sum0_2 = *(Cp + baseC_2 + 0);
                float sum1_2 = *(Cp + baseC_2 + 1);
                float sum2_2 = *(Cp + baseC_2 + 2);
                float sum3_2 = *(Cp + baseC_2 + 3);
                float sum4_2 = *(Cp + baseC_2 + 4);
                float sum5_2 = *(Cp + baseC_2 + 5);
                float sum6_2 = *(Cp + baseC_2 + 6);
                float sum7_2 = *(Cp + baseC_2 + 7);
                float sum8_2 = *(Cp + baseC_2 + 8);
                float sum9_2 = *(Cp + baseC_2 + 9);
                float sum10_2 = *(Cp + baseC_2 + 10);
                float sum11_2 = *(Cp + baseC_2 + 11);
                float sum12_2 = *(Cp + baseC_2 + 12);
                float sum13_2 = *(Cp + baseC_2 + 13);
                float sum14_2 = *(Cp + baseC_2 + 14);
                float sum15_2 = *(Cp + baseC_2 + 15);
                float sum16_2 = *(Cp + baseC_2 + 16);
                float sum17_2 = *(Cp + baseC_2 + 17);
                float sum18_2 = *(Cp + baseC_2 + 18);
                float sum19_2 = *(Cp + baseC_2 + 19);
                float sum20_2 = *(Cp + baseC_2 + 20);
                float sum21_2 = *(Cp + baseC_2 + 21);
                float sum22_2 = *(Cp + baseC_2 + 22);
                float sum23_2 = *(Cp + baseC_2 + 23);
                // 3
                float sum0_3 = *(Cp + baseC_3 + 0);
                float sum1_3 = *(Cp + baseC_3 + 1);
                float sum2_3 = *(Cp + baseC_3 + 2);
                float sum3_3 = *(Cp + baseC_3 + 3);
                float sum4_3 = *(Cp + baseC_3 + 4);
                float sum5_3 = *(Cp + baseC_3 + 5);
                float sum6_3 = *(Cp + baseC_3 + 6);
                float sum7_3 = *(Cp + baseC_3 + 7);
                float sum8_3 = *(Cp + baseC_3 + 8);
                float sum9_3 = *(Cp + baseC_3 + 9);
                float sum10_3 = *(Cp + baseC_3 + 10);
                float sum11_3 = *(Cp + baseC_3 + 11);
                float sum12_3 = *(Cp + baseC_3 + 12);
                float sum13_3 = *(Cp + baseC_3 + 13);
                float sum14_3 = *(Cp + baseC_3 + 14);
                float sum15_3 = *(Cp + baseC_3 + 15);
                float sum16_3 = *(Cp + baseC_3 + 16);
                float sum17_3 = *(Cp + baseC_3 + 17);
                float sum18_3 = *(Cp + baseC_3 + 18);
                float sum19_3 = *(Cp + baseC_3 + 19);
                float sum20_3 = *(Cp + baseC_3 + 20);
                float sum21_3 = *(Cp + baseC_3 + 21);
                float sum22_3 = *(Cp + baseC_3 + 22);
                float sum23_3 = *(Cp + baseC_3 + 23);

                for (int l = 0; l < blockSizeK; l++)
                {
                    float A_0 = *(Ap + i_0 * Astride + l);
                    float A_1 = *(Ap + i_1 * Astride + l);
                    float A_2 = *(Ap + i_2 * Astride + l);
                    float A_3 = *(Ap + i_3 * Astride + l);
                    int baseB = l * Bstride + j;
                    float B_0 = (*(Bp + baseB + 0));
                    float B_1 = (*(Bp + baseB + 1));
                    float B_2 = (*(Bp + baseB + 2));
                    float B_3 = (*(Bp + baseB + 3));
                    float B_4 = (*(Bp + baseB + 4));
                    float B_5 = (*(Bp + baseB + 5));
                    float B_6 = (*(Bp + baseB + 6));
                    float B_7 = (*(Bp + baseB + 7));
                    float B_8 = (*(Bp + baseB + 8));
                    float B_9 = (*(Bp + baseB + 9));
                    float B_10 = (*(Bp + baseB + 10));
                    float B_11 = (*(Bp + baseB + 11));
                    float B_12 = (*(Bp + baseB + 12));
                    float B_13 = (*(Bp + baseB + 13));
                    float B_14 = (*(Bp + baseB + 14));
                    float B_15 = (*(Bp + baseB + 15));
                    float B_16 = (*(Bp + baseB + 16));
                    float B_17 = (*(Bp + baseB + 17));
                    float B_18 = (*(Bp + baseB + 18));
                    float B_19 = (*(Bp + baseB + 19));
                    float B_20 = (*(Bp + baseB + 20));
                    float B_21 = (*(Bp + baseB + 21));
                    float B_22 = (*(Bp + baseB + 22));
                    float B_23 = (*(Bp + baseB + 23));
                    sum0_0 += A_0 * B_0;	sum0_1 += A_1 * B_0;	sum0_2 += A_2 * B_0;	sum0_3 += A_3 * B_0;
                    sum1_0 += A_0 * B_1;	sum1_1 += A_1 * B_1;	sum1_2 += A_2 * B_1;	sum1_3 += A_3 * B_1;
                    sum2_0 += A_0 * B_2;	sum2_1 += A_1 * B_2;	sum2_2 += A_2 * B_2;	sum2_3 += A_3 * B_2;
                    sum3_0 += A_0 * B_3;	sum3_1 += A_1 * B_3;	sum3_2 += A_2 * B_3;	sum3_3 += A_3 * B_3;
                    sum4_0 += A_0 * B_4;	sum4_1 += A_1 * B_4;	sum4_2 += A_2 * B_4;	sum4_3 += A_3 * B_4;
                    sum5_0 += A_0 * B_5;	sum5_1 += A_1 * B_5;	sum5_2 += A_2 * B_5;	sum5_3 += A_3 * B_5;
                    sum6_0 += A_0 * B_6;	sum6_1 += A_1 * B_6;	sum6_2 += A_2 * B_6;	sum6_3 += A_3 * B_6;
                    sum7_0 += A_0 * B_7;	sum7_1 += A_1 * B_7;	sum7_2 += A_2 * B_7;	sum7_3 += A_3 * B_7;
                    sum8_0 += A_0 * B_8;	sum8_1 += A_1 * B_8;	sum8_2 += A_2 * B_8;	sum8_3 += A_3 * B_8;
                    sum9_0 += A_0 * B_9;	sum9_1 += A_1 * B_9;	sum9_2 += A_2 * B_9;	sum9_3 += A_3 * B_9;
                    sum10_0 += A_0 * B_10;	sum10_1 += A_1 * B_10;	sum10_2 += A_2 * B_10;	sum10_3 += A_3 * B_10;
                    sum11_0 += A_0 * B_11;	sum11_1 += A_1 * B_11;	sum11_2 += A_2 * B_11;	sum11_3 += A_3 * B_11;
                    sum12_0 += A_0 * B_12;	sum12_1 += A_1 * B_12;	sum12_2 += A_2 * B_12;	sum12_3 += A_3 * B_12;
                    sum13_0 += A_0 * B_13;	sum13_1 += A_1 * B_13;	sum13_2 += A_2 * B_13;	sum13_3 += A_3 * B_13;
                    sum14_0 += A_0 * B_14;	sum14_1 += A_1 * B_14;	sum14_2 += A_2 * B_14;	sum14_3 += A_3 * B_14;
                    sum15_0 += A_0 * B_15;	sum15_1 += A_1 * B_15;	sum15_2 += A_2 * B_15;	sum15_3 += A_3 * B_15;
                    sum16_0 += A_0 * B_16;	sum16_1 += A_1 * B_16;	sum16_2 += A_2 * B_16;	sum16_3 += A_3 * B_16;
                    sum17_0 += A_0 * B_17;	sum17_1 += A_1 * B_17;	sum17_2 += A_2 * B_17;	sum17_3 += A_3 * B_17;
                    sum18_0 += A_0 * B_18;	sum18_1 += A_1 * B_18;	sum18_2 += A_2 * B_18;	sum18_3 += A_3 * B_18;
                    sum19_0 += A_0 * B_19;	sum19_1 += A_1 * B_19;	sum19_2 += A_2 * B_19;	sum19_3 += A_3 * B_19;
                    sum20_0 += A_0 * B_20;	sum20_1 += A_1 * B_20;	sum20_2 += A_2 * B_20;	sum20_3 += A_3 * B_20;
                    sum21_0 += A_0 * B_21;	sum21_1 += A_1 * B_21;	sum21_2 += A_2 * B_21;	sum21_3 += A_3 * B_21;
                    sum22_0 += A_0 * B_22;	sum22_1 += A_1 * B_22;	sum22_2 += A_2 * B_22;	sum22_3 += A_3 * B_22;
                    sum23_0 += A_0 * B_23;	sum23_1 += A_1 * B_23;	sum23_2 += A_2 * B_23;	sum23_3 += A_3 * B_23;
                }
                // 0
                *(Cp + baseC_0 + 0) = sum0_0;
                *(Cp + baseC_0 + 1) = sum1_0;
                *(Cp + baseC_0 + 2) = sum2_0;
                *(Cp + baseC_0 + 3) = sum3_0;
                *(Cp + baseC_0 + 4) = sum4_0;
                *(Cp + baseC_0 + 5) = sum5_0;
                *(Cp + baseC_0 + 6) = sum6_0;
                *(Cp + baseC_0 + 7) = sum7_0;
                *(Cp + baseC_0 + 8) = sum8_0;
                *(Cp + baseC_0 + 9) = sum9_0;
                *(Cp + baseC_0 + 10) = sum10_0;
                *(Cp + baseC_0 + 11) = sum11_0;
                *(Cp + baseC_0 + 12) = sum12_0;
                *(Cp + baseC_0 + 13) = sum13_0;
                *(Cp + baseC_0 + 14) = sum14_0;
                *(Cp + baseC_0 + 15) = sum15_0;
                *(Cp + baseC_0 + 16) = sum16_0;
                *(Cp + baseC_0 + 17) = sum17_0;
                *(Cp + baseC_0 + 18) = sum18_0;
                *(Cp + baseC_0 + 19) = sum19_0;
                *(Cp + baseC_0 + 20) = sum20_0;
                *(Cp + baseC_0 + 21) = sum21_0;
                *(Cp + baseC_0 + 22) = sum22_0;
                *(Cp + baseC_0 + 23) = sum23_0;
                // 1
                *(Cp + baseC_1 + 0) = sum0_1;
                *(Cp + baseC_1 + 1) = sum1_1;
                *(Cp + baseC_1 + 2) = sum2_1;
                *(Cp + baseC_1 + 3) = sum3_1;
                *(Cp + baseC_1 + 4) = sum4_1;
                *(Cp + baseC_1 + 5) = sum5_1;
                *(Cp + baseC_1 + 6) = sum6_1;
                *(Cp + baseC_1 + 7) = sum7_1;
                *(Cp + baseC_1 + 8) = sum8_1;
                *(Cp + baseC_1 + 9) = sum9_1;
                *(Cp + baseC_1 + 10) = sum10_1;
                *(Cp + baseC_1 + 11) = sum11_1;
                *(Cp + baseC_1 + 12) = sum12_1;
                *(Cp + baseC_1 + 13) = sum13_1;
                *(Cp + baseC_1 + 14) = sum14_1;
                *(Cp + baseC_1 + 15) = sum15_1;
                *(Cp + baseC_1 + 16) = sum16_1;
                *(Cp + baseC_1 + 17) = sum17_1;
                *(Cp + baseC_1 + 18) = sum18_1;
                *(Cp + baseC_1 + 19) = sum19_1;
                *(Cp + baseC_1 + 20) = sum20_1;
                *(Cp + baseC_1 + 21) = sum21_1;
                *(Cp + baseC_1 + 22) = sum22_1;
                *(Cp + baseC_1 + 23) = sum23_1;
                // 2
                *(Cp + baseC_2 + 0) = sum0_2;
                *(Cp + baseC_2 + 1) = sum1_2;
                *(Cp + baseC_2 + 2) = sum2_2;
                *(Cp + baseC_2 + 3) = sum3_2;
                *(Cp + baseC_2 + 4) = sum4_2;
                *(Cp + baseC_2 + 5) = sum5_2;
                *(Cp + baseC_2 + 6) = sum6_2;
                *(Cp + baseC_2 + 7) = sum7_2;
                *(Cp + baseC_2 + 8) = sum8_2;
                *(Cp + baseC_2 + 9) = sum9_2;
                *(Cp + baseC_2 + 10) = sum10_2;
                *(Cp + baseC_2 + 11) = sum11_2;
                *(Cp + baseC_2 + 12) = sum12_2;
                *(Cp + baseC_2 + 13) = sum13_2;
                *(Cp + baseC_2 + 14) = sum14_2;
                *(Cp + baseC_2 + 15) = sum15_2;
                *(Cp + baseC_2 + 16) = sum16_2;
                *(Cp + baseC_2 + 17) = sum17_2;
                *(Cp + baseC_2 + 18) = sum18_2;
                *(Cp + baseC_2 + 19) = sum19_2;
                *(Cp + baseC_2 + 20) = sum20_2;
                *(Cp + baseC_2 + 21) = sum21_2;
                *(Cp + baseC_2 + 22) = sum22_2;
                *(Cp + baseC_2 + 23) = sum23_2;
                // 3
                *(Cp + baseC_3 + 0) = sum0_3;
                *(Cp + baseC_3 + 1) = sum1_3;
                *(Cp + baseC_3 + 2) = sum2_3;
                *(Cp + baseC_3 + 3) = sum3_3;
                *(Cp + baseC_3 + 4) = sum4_3;
                *(Cp + baseC_3 + 5) = sum5_3;
                *(Cp + baseC_3 + 6) = sum6_3;
                *(Cp + baseC_3 + 7) = sum7_3;
                *(Cp + baseC_3 + 8) = sum8_3;
                *(Cp + baseC_3 + 9) = sum9_3;
                *(Cp + baseC_3 + 10) = sum10_3;
                *(Cp + baseC_3 + 11) = sum11_3;
                *(Cp + baseC_3 + 12) = sum12_3;
                *(Cp + baseC_3 + 13) = sum13_3;
                *(Cp + baseC_3 + 14) = sum14_3;
                *(Cp + baseC_3 + 15) = sum15_3;
                *(Cp + baseC_3 + 16) = sum16_3;
                *(Cp + baseC_3 + 17) = sum17_3;
                *(Cp + baseC_3 + 18) = sum18_3;
                *(Cp + baseC_3 + 19) = sum19_3;
                *(Cp + baseC_3 + 20) = sum20_3;
                *(Cp + baseC_3 + 21) = sum21_3;
                *(Cp + baseC_3 + 22) = sum22_3;
                *(Cp + baseC_3 + 23) = sum23_3;
            }
        }
        for (; i < blockSizeM - 2; i += 3)
        {
            var i_0 = i + 0;
            var i_1 = i + 1;
            var i_2 = i + 2;

            for (int j = 0; j < n; j += 24)
            {
                int baseC_0 = i_0 * Cstride + j;
                int baseC_1 = i_1 * Cstride + j;
                int baseC_2 = i_2 * Cstride + j;
                // 0
                float sum0_0 = *(Cp + baseC_0 + 0);
                float sum1_0 = *(Cp + baseC_0 + 1);
                float sum2_0 = *(Cp + baseC_0 + 2);
                float sum3_0 = *(Cp + baseC_0 + 3);
                float sum4_0 = *(Cp + baseC_0 + 4);
                float sum5_0 = *(Cp + baseC_0 + 5);
                float sum6_0 = *(Cp + baseC_0 + 6);
                float sum7_0 = *(Cp + baseC_0 + 7);
                float sum8_0 = *(Cp + baseC_0 + 8);
                float sum9_0 = *(Cp + baseC_0 + 9);
                float sum10_0 = *(Cp + baseC_0 + 10);
                float sum11_0 = *(Cp + baseC_0 + 11);
                float sum12_0 = *(Cp + baseC_0 + 12);
                float sum13_0 = *(Cp + baseC_0 + 13);
                float sum14_0 = *(Cp + baseC_0 + 14);
                float sum15_0 = *(Cp + baseC_0 + 15);
                float sum16_0 = *(Cp + baseC_0 + 16);
                float sum17_0 = *(Cp + baseC_0 + 17);
                float sum18_0 = *(Cp + baseC_0 + 18);
                float sum19_0 = *(Cp + baseC_0 + 19);
                float sum20_0 = *(Cp + baseC_0 + 20);
                float sum21_0 = *(Cp + baseC_0 + 21);
                float sum22_0 = *(Cp + baseC_0 + 22);
                float sum23_0 = *(Cp + baseC_0 + 23);
                // 1
                float sum0_1 = *(Cp + baseC_1 + 0);
                float sum1_1 = *(Cp + baseC_1 + 1);
                float sum2_1 = *(Cp + baseC_1 + 2);
                float sum3_1 = *(Cp + baseC_1 + 3);
                float sum4_1 = *(Cp + baseC_1 + 4);
                float sum5_1 = *(Cp + baseC_1 + 5);
                float sum6_1 = *(Cp + baseC_1 + 6);
                float sum7_1 = *(Cp + baseC_1 + 7);
                float sum8_1 = *(Cp + baseC_1 + 8);
                float sum9_1 = *(Cp + baseC_1 + 9);
                float sum10_1 = *(Cp + baseC_1 + 10);
                float sum11_1 = *(Cp + baseC_1 + 11);
                float sum12_1 = *(Cp + baseC_1 + 12);
                float sum13_1 = *(Cp + baseC_1 + 13);
                float sum14_1 = *(Cp + baseC_1 + 14);
                float sum15_1 = *(Cp + baseC_1 + 15);
                float sum16_1 = *(Cp + baseC_1 + 16);
                float sum17_1 = *(Cp + baseC_1 + 17);
                float sum18_1 = *(Cp + baseC_1 + 18);
                float sum19_1 = *(Cp + baseC_1 + 19);
                float sum20_1 = *(Cp + baseC_1 + 20);
                float sum21_1 = *(Cp + baseC_1 + 21);
                float sum22_1 = *(Cp + baseC_1 + 22);
                float sum23_1 = *(Cp + baseC_1 + 23);
                // 2
                float sum0_2 = *(Cp + baseC_2 + 0);
                float sum1_2 = *(Cp + baseC_2 + 1);
                float sum2_2 = *(Cp + baseC_2 + 2);
                float sum3_2 = *(Cp + baseC_2 + 3);
                float sum4_2 = *(Cp + baseC_2 + 4);
                float sum5_2 = *(Cp + baseC_2 + 5);
                float sum6_2 = *(Cp + baseC_2 + 6);
                float sum7_2 = *(Cp + baseC_2 + 7);
                float sum8_2 = *(Cp + baseC_2 + 8);
                float sum9_2 = *(Cp + baseC_2 + 9);
                float sum10_2 = *(Cp + baseC_2 + 10);
                float sum11_2 = *(Cp + baseC_2 + 11);
                float sum12_2 = *(Cp + baseC_2 + 12);
                float sum13_2 = *(Cp + baseC_2 + 13);
                float sum14_2 = *(Cp + baseC_2 + 14);
                float sum15_2 = *(Cp + baseC_2 + 15);
                float sum16_2 = *(Cp + baseC_2 + 16);
                float sum17_2 = *(Cp + baseC_2 + 17);
                float sum18_2 = *(Cp + baseC_2 + 18);
                float sum19_2 = *(Cp + baseC_2 + 19);
                float sum20_2 = *(Cp + baseC_2 + 20);
                float sum21_2 = *(Cp + baseC_2 + 21);
                float sum22_2 = *(Cp + baseC_2 + 22);
                float sum23_2 = *(Cp + baseC_2 + 23);

                for (int l = 0; l < blockSizeK; l++)
                {
                    float A_0 = *(Ap + i_0 * Astride + l);
                    float A_1 = *(Ap + i_1 * Astride + l);
                    float A_2 = *(Ap + i_2 * Astride + l);
                    int baseB = l * Bstride + j;
                    float B_0 = (*(Bp + baseB + 0));
                    float B_1 = (*(Bp + baseB + 1));
                    float B_2 = (*(Bp + baseB + 2));
                    float B_3 = (*(Bp + baseB + 3));
                    float B_4 = (*(Bp + baseB + 4));
                    float B_5 = (*(Bp + baseB + 5));
                    float B_6 = (*(Bp + baseB + 6));
                    float B_7 = (*(Bp + baseB + 7));
                    float B_8 = (*(Bp + baseB + 8));
                    float B_9 = (*(Bp + baseB + 9));
                    float B_10 = (*(Bp + baseB + 10));
                    float B_11 = (*(Bp + baseB + 11));
                    float B_12 = (*(Bp + baseB + 12));
                    float B_13 = (*(Bp + baseB + 13));
                    float B_14 = (*(Bp + baseB + 14));
                    float B_15 = (*(Bp + baseB + 15));
                    float B_16 = (*(Bp + baseB + 16));
                    float B_17 = (*(Bp + baseB + 17));
                    float B_18 = (*(Bp + baseB + 18));
                    float B_19 = (*(Bp + baseB + 19));
                    float B_20 = (*(Bp + baseB + 20));
                    float B_21 = (*(Bp + baseB + 21));
                    float B_22 = (*(Bp + baseB + 22));
                    float B_23 = (*(Bp + baseB + 23));
                    sum0_0 += A_0 * B_0;	sum0_1 += A_1 * B_0;	sum0_2 += A_2 * B_0;
                    sum1_0 += A_0 * B_1;	sum1_1 += A_1 * B_1;	sum1_2 += A_2 * B_1;
                    sum2_0 += A_0 * B_2;	sum2_1 += A_1 * B_2;	sum2_2 += A_2 * B_2;
                    sum3_0 += A_0 * B_3;	sum3_1 += A_1 * B_3;	sum3_2 += A_2 * B_3;
                    sum4_0 += A_0 * B_4;	sum4_1 += A_1 * B_4;	sum4_2 += A_2 * B_4;
                    sum5_0 += A_0 * B_5;	sum5_1 += A_1 * B_5;	sum5_2 += A_2 * B_5;
                    sum6_0 += A_0 * B_6;	sum6_1 += A_1 * B_6;	sum6_2 += A_2 * B_6;
                    sum7_0 += A_0 * B_7;	sum7_1 += A_1 * B_7;	sum7_2 += A_2 * B_7;
                    sum8_0 += A_0 * B_8;	sum8_1 += A_1 * B_8;	sum8_2 += A_2 * B_8;
                    sum9_0 += A_0 * B_9;	sum9_1 += A_1 * B_9;	sum9_2 += A_2 * B_9;
                    sum10_0 += A_0 * B_10;	sum10_1 += A_1 * B_10;	sum10_2 += A_2 * B_10;
                    sum11_0 += A_0 * B_11;	sum11_1 += A_1 * B_11;	sum11_2 += A_2 * B_11;
                    sum12_0 += A_0 * B_12;	sum12_1 += A_1 * B_12;	sum12_2 += A_2 * B_12;
                    sum13_0 += A_0 * B_13;	sum13_1 += A_1 * B_13;	sum13_2 += A_2 * B_13;
                    sum14_0 += A_0 * B_14;	sum14_1 += A_1 * B_14;	sum14_2 += A_2 * B_14;
                    sum15_0 += A_0 * B_15;	sum15_1 += A_1 * B_15;	sum15_2 += A_2 * B_15;
                    sum16_0 += A_0 * B_16;	sum16_1 += A_1 * B_16;	sum16_2 += A_2 * B_16;
                    sum17_0 += A_0 * B_17;	sum17_1 += A_1 * B_17;	sum17_2 += A_2 * B_17;
                    sum18_0 += A_0 * B_18;	sum18_1 += A_1 * B_18;	sum18_2 += A_2 * B_18;
                    sum19_0 += A_0 * B_19;	sum19_1 += A_1 * B_19;	sum19_2 += A_2 * B_19;
                    sum20_0 += A_0 * B_20;	sum20_1 += A_1 * B_20;	sum20_2 += A_2 * B_20;
                    sum21_0 += A_0 * B_21;	sum21_1 += A_1 * B_21;	sum21_2 += A_2 * B_21;
                    sum22_0 += A_0 * B_22;	sum22_1 += A_1 * B_22;	sum22_2 += A_2 * B_22;
                    sum23_0 += A_0 * B_23;	sum23_1 += A_1 * B_23;	sum23_2 += A_2 * B_23;
                }
                // 0
                *(Cp + baseC_0 + 0) = sum0_0;
                *(Cp + baseC_0 + 1) = sum1_0;
                *(Cp + baseC_0 + 2) = sum2_0;
                *(Cp + baseC_0 + 3) = sum3_0;
                *(Cp + baseC_0 + 4) = sum4_0;
                *(Cp + baseC_0 + 5) = sum5_0;
                *(Cp + baseC_0 + 6) = sum6_0;
                *(Cp + baseC_0 + 7) = sum7_0;
                *(Cp + baseC_0 + 8) = sum8_0;
                *(Cp + baseC_0 + 9) = sum9_0;
                *(Cp + baseC_0 + 10) = sum10_0;
                *(Cp + baseC_0 + 11) = sum11_0;
                *(Cp + baseC_0 + 12) = sum12_0;
                *(Cp + baseC_0 + 13) = sum13_0;
                *(Cp + baseC_0 + 14) = sum14_0;
                *(Cp + baseC_0 + 15) = sum15_0;
                *(Cp + baseC_0 + 16) = sum16_0;
                *(Cp + baseC_0 + 17) = sum17_0;
                *(Cp + baseC_0 + 18) = sum18_0;
                *(Cp + baseC_0 + 19) = sum19_0;
                *(Cp + baseC_0 + 20) = sum20_0;
                *(Cp + baseC_0 + 21) = sum21_0;
                *(Cp + baseC_0 + 22) = sum22_0;
                *(Cp + baseC_0 + 23) = sum23_0;
                // 1
                *(Cp + baseC_1 + 0) = sum0_1;
                *(Cp + baseC_1 + 1) = sum1_1;
                *(Cp + baseC_1 + 2) = sum2_1;
                *(Cp + baseC_1 + 3) = sum3_1;
                *(Cp + baseC_1 + 4) = sum4_1;
                *(Cp + baseC_1 + 5) = sum5_1;
                *(Cp + baseC_1 + 6) = sum6_1;
                *(Cp + baseC_1 + 7) = sum7_1;
                *(Cp + baseC_1 + 8) = sum8_1;
                *(Cp + baseC_1 + 9) = sum9_1;
                *(Cp + baseC_1 + 10) = sum10_1;
                *(Cp + baseC_1 + 11) = sum11_1;
                *(Cp + baseC_1 + 12) = sum12_1;
                *(Cp + baseC_1 + 13) = sum13_1;
                *(Cp + baseC_1 + 14) = sum14_1;
                *(Cp + baseC_1 + 15) = sum15_1;
                *(Cp + baseC_1 + 16) = sum16_1;
                *(Cp + baseC_1 + 17) = sum17_1;
                *(Cp + baseC_1 + 18) = sum18_1;
                *(Cp + baseC_1 + 19) = sum19_1;
                *(Cp + baseC_1 + 20) = sum20_1;
                *(Cp + baseC_1 + 21) = sum21_1;
                *(Cp + baseC_1 + 22) = sum22_1;
                *(Cp + baseC_1 + 23) = sum23_1;
                // 2
                *(Cp + baseC_2 + 0) = sum0_2;
                *(Cp + baseC_2 + 1) = sum1_2;
                *(Cp + baseC_2 + 2) = sum2_2;
                *(Cp + baseC_2 + 3) = sum3_2;
                *(Cp + baseC_2 + 4) = sum4_2;
                *(Cp + baseC_2 + 5) = sum5_2;
                *(Cp + baseC_2 + 6) = sum6_2;
                *(Cp + baseC_2 + 7) = sum7_2;
                *(Cp + baseC_2 + 8) = sum8_2;
                *(Cp + baseC_2 + 9) = sum9_2;
                *(Cp + baseC_2 + 10) = sum10_2;
                *(Cp + baseC_2 + 11) = sum11_2;
                *(Cp + baseC_2 + 12) = sum12_2;
                *(Cp + baseC_2 + 13) = sum13_2;
                *(Cp + baseC_2 + 14) = sum14_2;
                *(Cp + baseC_2 + 15) = sum15_2;
                *(Cp + baseC_2 + 16) = sum16_2;
                *(Cp + baseC_2 + 17) = sum17_2;
                *(Cp + baseC_2 + 18) = sum18_2;
                *(Cp + baseC_2 + 19) = sum19_2;
                *(Cp + baseC_2 + 20) = sum20_2;
                *(Cp + baseC_2 + 21) = sum21_2;
                *(Cp + baseC_2 + 22) = sum22_2;
                *(Cp + baseC_2 + 23) = sum23_2;
            }
        }
        for (; i < blockSizeM - 1; i += 2)
        {
            var i_0 = i + 0;
            var i_1 = i + 1;

            for (int j = 0; j < n; j += 24)
            {
                int baseC_0 = i_0 * Cstride + j;
                int baseC_1 = i_1 * Cstride + j;
                // 0
                float sum0_0 = *(Cp + baseC_0 + 0);
                float sum1_0 = *(Cp + baseC_0 + 1);
                float sum2_0 = *(Cp + baseC_0 + 2);
                float sum3_0 = *(Cp + baseC_0 + 3);
                float sum4_0 = *(Cp + baseC_0 + 4);
                float sum5_0 = *(Cp + baseC_0 + 5);
                float sum6_0 = *(Cp + baseC_0 + 6);
                float sum7_0 = *(Cp + baseC_0 + 7);
                float sum8_0 = *(Cp + baseC_0 + 8);
                float sum9_0 = *(Cp + baseC_0 + 9);
                float sum10_0 = *(Cp + baseC_0 + 10);
                float sum11_0 = *(Cp + baseC_0 + 11);
                float sum12_0 = *(Cp + baseC_0 + 12);
                float sum13_0 = *(Cp + baseC_0 + 13);
                float sum14_0 = *(Cp + baseC_0 + 14);
                float sum15_0 = *(Cp + baseC_0 + 15);
                float sum16_0 = *(Cp + baseC_0 + 16);
                float sum17_0 = *(Cp + baseC_0 + 17);
                float sum18_0 = *(Cp + baseC_0 + 18);
                float sum19_0 = *(Cp + baseC_0 + 19);
                float sum20_0 = *(Cp + baseC_0 + 20);
                float sum21_0 = *(Cp + baseC_0 + 21);
                float sum22_0 = *(Cp + baseC_0 + 22);
                float sum23_0 = *(Cp + baseC_0 + 23);
                // 1
                float sum0_1 = *(Cp + baseC_1 + 0);
                float sum1_1 = *(Cp + baseC_1 + 1);
                float sum2_1 = *(Cp + baseC_1 + 2);
                float sum3_1 = *(Cp + baseC_1 + 3);
                float sum4_1 = *(Cp + baseC_1 + 4);
                float sum5_1 = *(Cp + baseC_1 + 5);
                float sum6_1 = *(Cp + baseC_1 + 6);
                float sum7_1 = *(Cp + baseC_1 + 7);
                float sum8_1 = *(Cp + baseC_1 + 8);
                float sum9_1 = *(Cp + baseC_1 + 9);
                float sum10_1 = *(Cp + baseC_1 + 10);
                float sum11_1 = *(Cp + baseC_1 + 11);
                float sum12_1 = *(Cp + baseC_1 + 12);
                float sum13_1 = *(Cp + baseC_1 + 13);
                float sum14_1 = *(Cp + baseC_1 + 14);
                float sum15_1 = *(Cp + baseC_1 + 15);
                float sum16_1 = *(Cp + baseC_1 + 16);
                float sum17_1 = *(Cp + baseC_1 + 17);
                float sum18_1 = *(Cp + baseC_1 + 18);
                float sum19_1 = *(Cp + baseC_1 + 19);
                float sum20_1 = *(Cp + baseC_1 + 20);
                float sum21_1 = *(Cp + baseC_1 + 21);
                float sum22_1 = *(Cp + baseC_1 + 22);
                float sum23_1 = *(Cp + baseC_1 + 23);

                for (int l = 0; l < blockSizeK; l++)
                {
                    float A_0 = *(Ap + i_0 * Astride + l);
                    float A_1 = *(Ap + i_1 * Astride + l);
                    int baseB = l * Bstride + j;
                    float B_0 = (*(Bp + baseB + 0));
                    float B_1 = (*(Bp + baseB + 1));
                    float B_2 = (*(Bp + baseB + 2));
                    float B_3 = (*(Bp + baseB + 3));
                    float B_4 = (*(Bp + baseB + 4));
                    float B_5 = (*(Bp + baseB + 5));
                    float B_6 = (*(Bp + baseB + 6));
                    float B_7 = (*(Bp + baseB + 7));
                    float B_8 = (*(Bp + baseB + 8));
                    float B_9 = (*(Bp + baseB + 9));
                    float B_10 = (*(Bp + baseB + 10));
                    float B_11 = (*(Bp + baseB + 11));
                    float B_12 = (*(Bp + baseB + 12));
                    float B_13 = (*(Bp + baseB + 13));
                    float B_14 = (*(Bp + baseB + 14));
                    float B_15 = (*(Bp + baseB + 15));
                    float B_16 = (*(Bp + baseB + 16));
                    float B_17 = (*(Bp + baseB + 17));
                    float B_18 = (*(Bp + baseB + 18));
                    float B_19 = (*(Bp + baseB + 19));
                    float B_20 = (*(Bp + baseB + 20));
                    float B_21 = (*(Bp + baseB + 21));
                    float B_22 = (*(Bp + baseB + 22));
                    float B_23 = (*(Bp + baseB + 23));
                    sum0_0 += A_0 * B_0;	sum0_1 += A_1 * B_0;
                    sum1_0 += A_0 * B_1;	sum1_1 += A_1 * B_1;
                    sum2_0 += A_0 * B_2;	sum2_1 += A_1 * B_2;
                    sum3_0 += A_0 * B_3;	sum3_1 += A_1 * B_3;
                    sum4_0 += A_0 * B_4;	sum4_1 += A_1 * B_4;
                    sum5_0 += A_0 * B_5;	sum5_1 += A_1 * B_5;
                    sum6_0 += A_0 * B_6;	sum6_1 += A_1 * B_6;
                    sum7_0 += A_0 * B_7;	sum7_1 += A_1 * B_7;
                    sum8_0 += A_0 * B_8;	sum8_1 += A_1 * B_8;
                    sum9_0 += A_0 * B_9;	sum9_1 += A_1 * B_9;
                    sum10_0 += A_0 * B_10;	sum10_1 += A_1 * B_10;
                    sum11_0 += A_0 * B_11;	sum11_1 += A_1 * B_11;
                    sum12_0 += A_0 * B_12;	sum12_1 += A_1 * B_12;
                    sum13_0 += A_0 * B_13;	sum13_1 += A_1 * B_13;
                    sum14_0 += A_0 * B_14;	sum14_1 += A_1 * B_14;
                    sum15_0 += A_0 * B_15;	sum15_1 += A_1 * B_15;
                    sum16_0 += A_0 * B_16;	sum16_1 += A_1 * B_16;
                    sum17_0 += A_0 * B_17;	sum17_1 += A_1 * B_17;
                    sum18_0 += A_0 * B_18;	sum18_1 += A_1 * B_18;
                    sum19_0 += A_0 * B_19;	sum19_1 += A_1 * B_19;
                    sum20_0 += A_0 * B_20;	sum20_1 += A_1 * B_20;
                    sum21_0 += A_0 * B_21;	sum21_1 += A_1 * B_21;
                    sum22_0 += A_0 * B_22;	sum22_1 += A_1 * B_22;
                    sum23_0 += A_0 * B_23;	sum23_1 += A_1 * B_23;
                }
                // 0
                *(Cp + baseC_0 + 0) = sum0_0;
                *(Cp + baseC_0 + 1) = sum1_0;
                *(Cp + baseC_0 + 2) = sum2_0;
                *(Cp + baseC_0 + 3) = sum3_0;
                *(Cp + baseC_0 + 4) = sum4_0;
                *(Cp + baseC_0 + 5) = sum5_0;
                *(Cp + baseC_0 + 6) = sum6_0;
                *(Cp + baseC_0 + 7) = sum7_0;
                *(Cp + baseC_0 + 8) = sum8_0;
                *(Cp + baseC_0 + 9) = sum9_0;
                *(Cp + baseC_0 + 10) = sum10_0;
                *(Cp + baseC_0 + 11) = sum11_0;
                *(Cp + baseC_0 + 12) = sum12_0;
                *(Cp + baseC_0 + 13) = sum13_0;
                *(Cp + baseC_0 + 14) = sum14_0;
                *(Cp + baseC_0 + 15) = sum15_0;
                *(Cp + baseC_0 + 16) = sum16_0;
                *(Cp + baseC_0 + 17) = sum17_0;
                *(Cp + baseC_0 + 18) = sum18_0;
                *(Cp + baseC_0 + 19) = sum19_0;
                *(Cp + baseC_0 + 20) = sum20_0;
                *(Cp + baseC_0 + 21) = sum21_0;
                *(Cp + baseC_0 + 22) = sum22_0;
                *(Cp + baseC_0 + 23) = sum23_0;
                // 1
                *(Cp + baseC_1 + 0) = sum0_1;
                *(Cp + baseC_1 + 1) = sum1_1;
                *(Cp + baseC_1 + 2) = sum2_1;
                *(Cp + baseC_1 + 3) = sum3_1;
                *(Cp + baseC_1 + 4) = sum4_1;
                *(Cp + baseC_1 + 5) = sum5_1;
                *(Cp + baseC_1 + 6) = sum6_1;
                *(Cp + baseC_1 + 7) = sum7_1;
                *(Cp + baseC_1 + 8) = sum8_1;
                *(Cp + baseC_1 + 9) = sum9_1;
                *(Cp + baseC_1 + 10) = sum10_1;
                *(Cp + baseC_1 + 11) = sum11_1;
                *(Cp + baseC_1 + 12) = sum12_1;
                *(Cp + baseC_1 + 13) = sum13_1;
                *(Cp + baseC_1 + 14) = sum14_1;
                *(Cp + baseC_1 + 15) = sum15_1;
                *(Cp + baseC_1 + 16) = sum16_1;
                *(Cp + baseC_1 + 17) = sum17_1;
                *(Cp + baseC_1 + 18) = sum18_1;
                *(Cp + baseC_1 + 19) = sum19_1;
                *(Cp + baseC_1 + 20) = sum20_1;
                *(Cp + baseC_1 + 21) = sum21_1;
                *(Cp + baseC_1 + 22) = sum22_1;
                *(Cp + baseC_1 + 23) = sum23_1;
            }
        }
        for (; i < blockSizeM - 0; i += 1)
        {
            var i_0 = i + 0;

            for (int j = 0; j < n; j += 24)
            {
                int baseC_0 = i_0 * Cstride + j;
                // 0
                float sum0_0 = *(Cp + baseC_0 + 0);
                float sum1_0 = *(Cp + baseC_0 + 1);
                float sum2_0 = *(Cp + baseC_0 + 2);
                float sum3_0 = *(Cp + baseC_0 + 3);
                float sum4_0 = *(Cp + baseC_0 + 4);
                float sum5_0 = *(Cp + baseC_0 + 5);
                float sum6_0 = *(Cp + baseC_0 + 6);
                float sum7_0 = *(Cp + baseC_0 + 7);
                float sum8_0 = *(Cp + baseC_0 + 8);
                float sum9_0 = *(Cp + baseC_0 + 9);
                float sum10_0 = *(Cp + baseC_0 + 10);
                float sum11_0 = *(Cp + baseC_0 + 11);
                float sum12_0 = *(Cp + baseC_0 + 12);
                float sum13_0 = *(Cp + baseC_0 + 13);
                float sum14_0 = *(Cp + baseC_0 + 14);
                float sum15_0 = *(Cp + baseC_0 + 15);
                float sum16_0 = *(Cp + baseC_0 + 16);
                float sum17_0 = *(Cp + baseC_0 + 17);
                float sum18_0 = *(Cp + baseC_0 + 18);
                float sum19_0 = *(Cp + baseC_0 + 19);
                float sum20_0 = *(Cp + baseC_0 + 20);
                float sum21_0 = *(Cp + baseC_0 + 21);
                float sum22_0 = *(Cp + baseC_0 + 22);
                float sum23_0 = *(Cp + baseC_0 + 23);

                for (int l = 0; l < blockSizeK; l++)
                {
                    float A_0 = *(Ap + i_0 * Astride + l);
                    int baseB = l * Bstride + j;
                    float B_0 = (*(Bp + baseB + 0));
                    float B_1 = (*(Bp + baseB + 1));
                    float B_2 = (*(Bp + baseB + 2));
                    float B_3 = (*(Bp + baseB + 3));
                    float B_4 = (*(Bp + baseB + 4));
                    float B_5 = (*(Bp + baseB + 5));
                    float B_6 = (*(Bp + baseB + 6));
                    float B_7 = (*(Bp + baseB + 7));
                    float B_8 = (*(Bp + baseB + 8));
                    float B_9 = (*(Bp + baseB + 9));
                    float B_10 = (*(Bp + baseB + 10));
                    float B_11 = (*(Bp + baseB + 11));
                    float B_12 = (*(Bp + baseB + 12));
                    float B_13 = (*(Bp + baseB + 13));
                    float B_14 = (*(Bp + baseB + 14));
                    float B_15 = (*(Bp + baseB + 15));
                    float B_16 = (*(Bp + baseB + 16));
                    float B_17 = (*(Bp + baseB + 17));
                    float B_18 = (*(Bp + baseB + 18));
                    float B_19 = (*(Bp + baseB + 19));
                    float B_20 = (*(Bp + baseB + 20));
                    float B_21 = (*(Bp + baseB + 21));
                    float B_22 = (*(Bp + baseB + 22));
                    float B_23 = (*(Bp + baseB + 23));
                    sum0_0 += A_0 * B_0;
                    sum1_0 += A_0 * B_1;
                    sum2_0 += A_0 * B_2;
                    sum3_0 += A_0 * B_3;
                    sum4_0 += A_0 * B_4;
                    sum5_0 += A_0 * B_5;
                    sum6_0 += A_0 * B_6;
                    sum7_0 += A_0 * B_7;
                    sum8_0 += A_0 * B_8;
                    sum9_0 += A_0 * B_9;
                    sum10_0 += A_0 * B_10;
                    sum11_0 += A_0 * B_11;
                    sum12_0 += A_0 * B_12;
                    sum13_0 += A_0 * B_13;
                    sum14_0 += A_0 * B_14;
                    sum15_0 += A_0 * B_15;
                    sum16_0 += A_0 * B_16;
                    sum17_0 += A_0 * B_17;
                    sum18_0 += A_0 * B_18;
                    sum19_0 += A_0 * B_19;
                    sum20_0 += A_0 * B_20;
                    sum21_0 += A_0 * B_21;
                    sum22_0 += A_0 * B_22;
                    sum23_0 += A_0 * B_23;
                }
                // 0
                *(Cp + baseC_0 + 0) = sum0_0;
                *(Cp + baseC_0 + 1) = sum1_0;
                *(Cp + baseC_0 + 2) = sum2_0;
                *(Cp + baseC_0 + 3) = sum3_0;
                *(Cp + baseC_0 + 4) = sum4_0;
                *(Cp + baseC_0 + 5) = sum5_0;
                *(Cp + baseC_0 + 6) = sum6_0;
                *(Cp + baseC_0 + 7) = sum7_0;
                *(Cp + baseC_0 + 8) = sum8_0;
                *(Cp + baseC_0 + 9) = sum9_0;
                *(Cp + baseC_0 + 10) = sum10_0;
                *(Cp + baseC_0 + 11) = sum11_0;
                *(Cp + baseC_0 + 12) = sum12_0;
                *(Cp + baseC_0 + 13) = sum13_0;
                *(Cp + baseC_0 + 14) = sum14_0;
                *(Cp + baseC_0 + 15) = sum15_0;
                *(Cp + baseC_0 + 16) = sum16_0;
                *(Cp + baseC_0 + 17) = sum17_0;
                *(Cp + baseC_0 + 18) = sum18_0;
                *(Cp + baseC_0 + 19) = sum19_0;
                *(Cp + baseC_0 + 20) = sum20_0;
                *(Cp + baseC_0 + 21) = sum21_0;
                *(Cp + baseC_0 + 22) = sum22_0;
                *(Cp + baseC_0 + 23) = sum23_0;
            }
        }
    }

}
}
