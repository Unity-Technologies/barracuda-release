Shader "Barracuda/PRelu"
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

            fixed4 frag (v2f i) : SV_Target
            {
                TENSOR_ARGS3(X, W, O);
				
                uint n, h, w, c4;
                O.GetPositionFromUV(i.uv, n, h, w, c4);
                float4 v = X.Get4(n, h, w, c4);
                float4 slope = W.BroadcastGet4(n, h, w, c4);

                v.x = max(0.0f, v.x) + slope.x * min(0.0f, v.x);
                v.y = max(0.0f, v.y) + slope.y * min(0.0f, v.y);
                v.z = max(0.0f, v.z) + slope.z * min(0.0f, v.z);
                v.w = max(0.0f, v.w) + slope.w * min(0.0f, v.w);

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
