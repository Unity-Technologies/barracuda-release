Shader "Barracuda/LRN"
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
            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"

            #include "TensorTexture.cginc"

            float _Alpha;
            float _Beta;
            float _Epsilon;
            uint _Axis;

            TENSOR_DECL_O(O)
            TENSOR_DECL(X)

            float signed_pow(float f, float e)
            {
                // handle negative f
                float v = pow(abs(f), e);
                float s = (e % 2 == 1) ?
                    sign(f):    // exponent is odd  => sign(f) * pow(abs(f), e)
                    1;            // exponent is even => pow(abs(f), e)
                return v * s;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                TENSOR_ARGS2(X, O);
				
                uint n, h, w, c4;
                O.GetPositionFromUV(i.uv, n, h, w, c4);

                float bias = _Epsilon;
                float sizef = (float)_Axis;

                float regionCenter = (sizef - 1.0f) / 2.0f;

                float4 v = X.Get4(n, h, w, c4);
                [unroll]
                for (uint cc = 0; cc < 4; cc++)
                {
                    uint c = 4 * c4 + cc;
                    uint regionStart = max(0, c - (uint)floor(regionCenter));
                    uint regionEnd = min(X.channels, c + (uint)ceil(regionCenter) + 1);
                    float sumOfSquared = 0.0f;

                    for (uint ci = regionStart; ci < regionEnd; ++ci)
                    {
                        float regionValue = X.Get(n, h, w, ci);
                        sumOfSquared += regionValue * regionValue;
                    }

                    v[cc] /= signed_pow(bias + _Alpha * sumOfSquared / sizef, _Beta);
                }

                return v;
            }
            ENDCG
        }
    }
}
