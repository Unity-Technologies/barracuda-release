Shader "Barracuda/SpaceToDepth"
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
			
            fixed4 frag(v2f i) : SV_Target
            {
                TENSOR_ARGS2(X, O);

                uint n, y, x, c4;
                O.GetPositionFromUV(i.uv, n, y, x, c4);

                uint bsX = _Pool.x;
                uint bsY = _Pool.y;

                float4 v = 0;
                [unroll]
                for (uint cc = 0; cc < 4; cc++)
                {
                    uint c = 4 * c4 + cc;
                    int ic = c % X.channels;
                    int bx = c / X.channels % bsX;
                    int by = c / X.channels / bsX;
                    int ix = x * bsX + bx;
                    int iy = y * bsY + by;

                    if (c < O.channels)
                        v[cc] = X.Get(n, iy, ix, ic);
                }

                return v;
            }
            ENDCG
        }
    }
}
