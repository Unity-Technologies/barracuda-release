Shader "Barracuda/BufferToTensor"
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

            uint _InputHeight;
            uint _InputWidth;

            Texture2D<float> Xtex2D;

            fixed4 frag (v2f i) : SV_Target
            {
                TENSOR_O(O);

                uint n, h, w, c4;
                O.GetPositionFromUV(i.uv, n, h, w, c4);
				
                float4 v = 0.0f;
	
                [unroll]
                for (uint cc = 0; cc < 4; cc++)
                {
                    if (c4*4+cc >= O.channels)
                        break;

                    uint index = n * O.height * O.width * O.channels + h * O.width * O.channels + w * O.channels + 4 * c4 + cc;

                    uint x = (index) % _InputWidth;
                    uint y = (index) / _InputWidth;

                    v[cc] = Xtex2D.Load(uint3(x, y, 0)).r;
                }

                return v;
            }
            ENDCG
        }
    }
}
