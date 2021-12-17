Shader "Barracuda/Sigmoid"
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

            TENSOR_DECL_O(O)
            TENSOR_DECL(X)

            fixed4 frag (v2f i) : SV_Target
            {
                TENSOR_ARGS2(X, O);
				
                uint n, h, w, c4;
                O.GetPositionFromUV(i.uv, n, h, w, c4);
                float4 v = X.Get4(n, h, w, c4);

                v.x = max(0.0f, min(1.0f, _Alpha * v.x + _Beta));
                v.y = max(0.0f, min(1.0f, _Alpha * v.y + _Beta));
                v.z = max(0.0f, min(1.0f, _Alpha * v.z + _Beta));
                v.w = max(0.0f, min(1.0f, _Alpha * v.w + _Beta));

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
