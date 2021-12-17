Shader "Barracuda/BroadcastWhere"
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
            TENSOR_DECL(K)

            fixed4 frag (v2f i) : SV_Target
            {
                TENSOR_ARGS4(X, W, K, O);

                uint n, h, w, c4;
                O.GetPositionFromUV(i.uv, n, h, w, c4);

                float4 cond = (X.BroadcastGet4(n, h, w, c4) != 0.0f);
                float4 a = W.BroadcastGet4(n, h, w, c4);
                float4 b = K.BroadcastGet4(n, h, w, c4);

                float4 v = 0.0f;
                v.x = cond.x ? a.x : b.x;
                v.y = cond.y ? a.y : b.y;
                v.z = cond.z ? a.z : b.z;
                v.w = cond.w ? a.w : b.w;

                if (4 * c4 >= O.channels)
                    v.x = 0.0f;
                if (4 * c4 + 1 >= O.channels)
                    v.y = 0.0f;
                if (4 * c4 + 2 >= O.channels)
                    v.z = 0.0f;
                if (4 * c4 + 3 >= O.channels)
                    v.w = 0.0f;

                return v;
            }
            ENDCG
        }
    }
}
