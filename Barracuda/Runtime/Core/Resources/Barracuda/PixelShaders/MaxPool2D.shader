Shader "Barracuda/MaxPool2D"
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

                float4 maxV = -FLT_MAX;
                for (uint dy = 0; dy < _Pool.y; ++dy)
                    for (uint dx = 0; dx < _Pool.x; ++dx)
                    {
                        uint2 pos = uint2(w, h) * _Stride.xy + uint2(dx, dy);
                        float4 v = X.SafeGet4(n, pos, c4, _Pad.xy, -FLT_MAX);
                        maxV = max(v, maxV);
                    }
				
                return maxV;
            }
            ENDCG
        }
    }
}
