Shader "Barracuda/Dense3"
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
				
            fixed4 frag(v2f i) : SV_Target
            {
                TENSOR_O(O);
                TENSOR_ARG(X);
                TENSOR_ARG(W);
                TENSOR_ARG(B);

                uint n, h, w, k4;
                O.GetPositionFromUV(i.uv, n, h, w, k4);

                float4 acc4 = B.Get(0, 0, 0, w);
                for (uint j = 0; j < X.width; ++j)
                {
                    acc4 += X.Get4(n, 0, j, k4) * W.Get(j, 0, 0, w);
                }

                return ApplyFusedActivation(acc4);
            }
            ENDCG
        }
    }
}
