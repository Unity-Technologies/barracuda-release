Shader "Barracuda/DepthwiseConv2D"
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

            TENSOR_DECL_O(O)
            TENSOR_DECL(X)
            TENSOR_DECL(K)
            TENSOR_DECL(B)

            uint4 _Pad;
            uint4 _Stride;

            fixed4 frag (v2f i) : SV_Target
            {
                TENSOR_ARGS4(X, K, B, O);
				
                uint n, h, w, k4;
                O.GetPositionFromUV(i.uv, n, h, w, k4);

                float4 acc4 = B.Get4(0, 0, 0, k4);

                for (uint dy = 0; dy < K.GetKernelHeight(); ++dy)
                {
                    for (uint dx = 0; dx < K.GetKernelWidth(); ++dx)
                    {
                        uint2 pos = uint2(w, h) * _Stride.xy + uint2(dx, dy);
                        float4 v = X.SafeGet4(n, pos, k4, _Pad.xy);
				
                        float4 w0 = K.Get4(dy, dx, 0, k4);

                        acc4 += v * w0;
                    }
                }

                return ApplyFusedActivation(acc4);
            }
            ENDCG
        }
    }
}
