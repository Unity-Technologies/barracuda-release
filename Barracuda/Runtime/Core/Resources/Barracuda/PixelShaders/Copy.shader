Shader "Barracuda/Copy"
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

            fixed4 frag (v2f i) : SV_Target
            {
                TENSOR_ARGS2(X, O);
				
                uint n, h, w, c4;
                O.GetPositionFromUV(i.uv, n, h, w, c4);

                float4 v = 0.0f;
                [unroll]
                for (uint cc = 0; cc < 4; cc++)
                {
                    if (c4 * 4 + cc >= O.channels)
                        break;

                    uint index = n * O.height * O.width * O.channels + h * O.width * O.channels + w * O.channels + (4 * c4 + cc);

                    uint cX = index % X.channels;
                    uint wX = (index / X.channels) % X.width;
                    uint hX = (index / X.channels / X.width) % X.height;
                    uint nX = (index / X.channels / X.width / X.height);

                    v[cc] = X.Get(nX, hX, wX, cX);
                }

                return v;
            }
            ENDCG
        }
    }
}
