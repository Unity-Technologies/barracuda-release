Shader "Barracuda/ScaleBias"
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
            TENSOR_DECL(W)
            TENSOR_DECL(B)

            fixed4 frag (v2f i) : SV_Target
            {
                TENSOR_ARGS4(X, W, B, O);
				
                uint n, h, w, c4;
                O.GetPositionFromUV(i.uv, n, h, w, c4);

                float4 scale = W.Get4(0,0,0,c4);
                float4 bias = B.Get4(0,0,0,c4);

                float4 v = X.Get4(n, h, w, c4);

                return scale * v + bias;
            }
            ENDCG
        }
    }
}
