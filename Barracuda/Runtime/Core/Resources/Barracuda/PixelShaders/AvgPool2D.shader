Shader "Barracuda/AvgPool2D"
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

            uint4 _Pool;
            uint4 _Pad;
            uint4 _Stride;

            fixed4 frag (v2f i) : SV_Target
            {
                TENSOR_ARGS2(X, O);
				
                uint n, h, w, c4;
                O.GetPositionFromUV(i.uv, n, h, w, c4);


                uint2 leftCorner = _Pad.xy;
                uint2 rightCorner = uint2(X.width, X.height) + _Pad.xy;

                float4 acc4 = 0;
                float counter = 0;
                for (uint dy = 0; dy < _Pool.y; ++dy)
                    for (uint dx = 0; dx < _Pool.x; ++dx)
                    {
                        uint oy = h * _Stride.y + dy;
                        uint ox = w * _Stride.x + dx;

                        bool mask = (oy >= leftCorner.y) && (ox >= leftCorner.x) && (oy < rightCorner.y) && (ox < rightCorner.x);
                        acc4 += (mask) ? X.Get4(n, min(oy - leftCorner.y, X.height - 1), min(ox - leftCorner.x, X.width - 1), c4) : 0;
                        counter += (mask) ? 1 : 0;
                    }

                acc4 /= counter;
				
                return acc4;
            }
            ENDCG
        }
    }
}
