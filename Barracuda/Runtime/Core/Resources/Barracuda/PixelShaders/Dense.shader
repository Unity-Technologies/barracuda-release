Shader "Barracuda/Dense"
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

                float4 acc4 = B.Get4(0, 0, 0, k4);
                for (uint c4 = 0; c4 < X.channels4; c4++)
                {
                    float4 v = X.Get4(n, 0, 0, c4);
                    float4 w0 = W.Get4(4 * c4 + 0, 0, 0, k4);
                    float4 w1 = W.Get4(4 * c4 + 1, 0, 0, k4);
                    float4 w2 = W.Get4(4 * c4 + 2, 0, 0, k4);
                    float4 w3 = W.Get4(4 * c4 + 3, 0, 0, k4);

                    acc4.x += dot(v, float4(w0.x, w1.x, w2.x, w3.x));
                    acc4.y += dot(v, float4(w0.y, w1.y, w2.y, w3.y));
                    acc4.z += dot(v, float4(w0.z, w1.z, w2.z, w3.z));
                    acc4.w += dot(v, float4(w0.w, w1.w, w2.w, w3.w));
                }

                return ApplyFusedActivation(acc4);
            }
            ENDCG
        }
    }
}
