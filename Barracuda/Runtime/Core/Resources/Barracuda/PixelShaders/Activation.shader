Shader "Barracuda/Activation"
{
    Properties
    {
    }
    SubShader
    {
        // No culling or depth
        Cull Off ZWrite Off ZTest Always

        Pass
        {
            CGPROGRAM
            #pragma multi_compile None Relu Selu Abs Neg Ceil Floor Round Reciprocal Swish Tanh Softplus Sigmoid HardSigmoid Relu6 Elu LeakyRelu Exp Log Sqrt Acos Acosh Asin Asinh Atan Atanh Cos Cosh Sin Sinh Tan Pow Clip Erf Sign LogicalNot

            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"

            #include "TensorTexture.cginc"

            TENSOR_DECL_O(O)
            TENSOR_DECL(X)

            float signed_pow(float f, float e)
            {
                // handle negative f
                float v = pow(abs(f), e);
                float s = (e % 2 == 1) ?
                    sign(f) :    // exponent is odd  => sign(f) * pow(abs(f), e)
                    1;            // exponent is even => pow(abs(f), e)
                return v * s;
            }

            float erf(float v)
            {
                // Abramowitz/Stegun approximations
                // erf(x) = -erf(-x)
                float x = abs(v);

                float p = 0.3275911f;
                float a1 = 0.254829592f; float a2 = -0.284496736f; float a3 = 1.421413741f;
                float a4 = -1.453152027f; float a5 = 1.061405429f;

                float t = 1.0f / (1.0f + p * x);
                float t2 = t * t;
                float t3 = t2 * t;
                float t4 = t3 * t;
                float t5 = t4 * t;

                return sign(v)*(1 - (a1*t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5)*exp(-x * x));
            }

            float _Alpha;
            float _Beta;

            fixed4 frag (v2f i) : SV_Target
            {
                TENSOR_ARGS2(X, O);
				
                uint n, h, w, c4;
                O.GetPositionFromUV(i.uv, n, h, w, c4);
                float4 v = X.Get4(n, h, w, c4);

                #ifdef Relu
                    v = 0.5f * (v + abs(v));
                #endif
                #ifdef Selu
                    v = _Beta * (max(v, 0.0f) + min(_Alpha * (exp(v) - 1.0f), 0.0f));
                #endif
                #ifdef Abs
                    v = abs(v);
                #endif
                #ifdef Neg
                    v = -v;
                #endif
                #ifdef Ceil
                    v = ceil(v);
                #endif
                #ifdef Floor
                    v = floor(v);
                #endif
                #ifdef Round
                    v = round(v);
                #endif
                #ifdef Reciprocal
                    v = 1.0f / v;
                #endif
                #ifdef Swish
                    v = v / (1 + exp(-v));
                #endif
                #ifdef Tanh
                    v = tanh(clamp(v,-16.0f,16.0f));//clamp to avoid NaNs for large values.
                #endif
                #ifdef Softplus
                    v = log(exp(v) + 1);
                #endif
                #ifdef Sigmoid
                    v = 1 / (1 + exp(-v));
                #endif
                #ifdef HardSigmoid
                    v = max(0.0f, min(1.0f, _Alpha * v + _Beta));
                #endif
                #ifdef Relu6
                    v = min(max(0, v), 6);
                #endif
                #ifdef Elu
                    if (v.x <= 0)
                        v.x = _Alpha * (exp(v.x) - 1);
                    if (v.y <= 0)
                        v.y = _Alpha * (exp(v.y) - 1);
                    if (v.z <= 0)
                        v.z = _Alpha * (exp(v.z) - 1);
                    if (v.w <= 0)
                        v.w = _Alpha * (exp(v.w) - 1);
                #endif
                #ifdef LeakyRelu
                    v = max(v, _Alpha * v);
                #endif
                #ifdef Exp
                    v = exp(v);
                #endif
                #ifdef Log
                    v = log(v);
                #endif
                #ifdef Sqrt
                    v = sqrt(v);
                #endif
                #ifdef Acos
                    v = acos(v);
                #endif
                #ifdef Acosh
                    v = log(v + sqrt(v * v - 1.0f));
                #endif
                #ifdef Asin
                    v = asin(v);
                #endif
                #ifdef Asinh
                    v = log(v + sqrt(v*v + 1.0f));
                #endif
                #ifdef Atan
                    v = atan(v);
                #endif
                #ifdef Atanh
                    v = 0.5f * log((1.0f + v) / (1.0f - v));
                #endif
                #ifdef Cos
                    v = cos(v);
                #endif
                #ifdef Cosh
                    v = 0.5f * (exp(v) + exp(-v));
                #endif
                #ifdef Sin
                    v = sin(v);
                #endif
                #ifdef Sinh
                    v = 0.5f * (exp(v) - exp(-v));
                #endif
                #ifdef Tan
                    v = tan(v);
                #endif
                #ifdef Pow
                    v.x = signed_pow(v.x, _Alpha);
                    v.y = signed_pow(v.y, _Alpha);
                    v.z = signed_pow(v.z, _Alpha);
                    v.w = signed_pow(v.w, _Alpha);
                #endif
                #ifdef Clip
                    v = clamp(v, _Alpha, _Beta);
                #endif
                #ifdef Erf
                    v.x = erf(v.x);
                    v.y = erf(v.y);
                    v.z = erf(v.z);
                    v.w = erf(v.w);
                #endif
                #ifdef Sign
                    v = sign(v);
                #endif
                #ifdef LogicalNot
                    v = (v == 0.0f) ? 1.0f : 0.0f;
                #endif

                if (4 * c4 >= X.channels)
                    v.x = 0.0f;
                if (4 * c4 + 1 >= X.channels)
                    v.y = 0.0f;
                if (4 * c4 + 2 >= X.channels)
                    v.z = 0.0f;
                if (4 * c4 + 3 >= X.channels)
                    v.w = 0.0f;

                return v;
            }
            ENDCG
        }
    }
}
