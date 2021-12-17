Shader "Barracuda/GlobalMaxPool2D"
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

                float4 maxV4 = -FLT_MAX;
                for (uint y = 0; y < X.height; ++y)
                    for (uint x = 0; x < X.width; ++x)
                    {
                        float4 v = X.Get4(n, y, x, c4);
                        maxV4.x = max(v.x, maxV4.x);
                        maxV4.y = max(v.y, maxV4.y);
                        maxV4.z = max(v.z, maxV4.z);
                        maxV4.w = max(v.w, maxV4.w);
                    }
						
                return maxV4;
            }
            ENDCG
        }
    }
}
