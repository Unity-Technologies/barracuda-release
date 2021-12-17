Shader "Barracuda/MatMul"
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
            #pragma multi_compile xTranspose_OFF xTranspose_ON
            #pragma multi_compile yTranspose_OFF yTranspose_ON

            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
			
            #include "TensorTexture.cginc"
			
            TENSOR_DECL_O(O)
            TENSOR_DECL(X)
            TENSOR_DECL(Y)
				
            fixed4 frag(v2f i) : SV_Target
            {
                TENSOR_O(O);
                TENSOR_ARG(X);
                TENSOR_ARG(Y);

                uint n, h, w, k4;
                O.GetPositionFromUV(i.uv, n, h, w, k4);

                float4 acc4 = 0.0f;
                for (uint c4 = 0; c4 < X.channels4; c4++)
                {
                    float4 a = X.Get4(n, 0, 0, c4);
                    #ifdef xTranspose_ON
                    a.x = X.Get(4 * c4 + 0, 0, 0, n);
                    a.y = X.Get(4 * c4 + 1, 0, 0, n);
                    a.z = X.Get(4 * c4 + 2, 0, 0, n);
                    a.w = X.Get(4 * c4 + 3, 0, 0, n);
                    #endif

                    float4 b0 = Y.Get4(4 * c4 + 0, 0, 0, k4);
                    float4 b1 = Y.Get4(4 * c4 + 1, 0, 0, k4);
                    float4 b2 = Y.Get4(4 * c4 + 2, 0, 0, k4);
                    float4 b3 = Y.Get4(4 * c4 + 3, 0, 0, k4);
                    #ifdef yTranspose_ON
                    b0.x = Y.Get(4 * k4 + 0, 0, 0, 4 * c4 + 0);
                    b0.y = Y.Get(4 * k4 + 1, 0, 0, 4 * c4 + 0);
                    b0.z = Y.Get(4 * k4 + 2, 0, 0, 4 * c4 + 0);
                    b0.w = Y.Get(4 * k4 + 3, 0, 0, 4 * c4 + 0);
                  
                    b1.x = Y.Get(4 * k4 + 0, 0, 0, 4 * c4 + 1);
                    b1.y = Y.Get(4 * k4 + 1, 0, 0, 4 * c4 + 1);
                    b1.z = Y.Get(4 * k4 + 2, 0, 0, 4 * c4 + 1);
                    b1.w = Y.Get(4 * k4 + 3, 0, 0, 4 * c4 + 1);
                  
                    b2.x = Y.Get(4 * k4 + 0, 0, 0, 4 * c4 + 2);
                    b2.y = Y.Get(4 * k4 + 1, 0, 0, 4 * c4 + 2);
                    b2.z = Y.Get(4 * k4 + 2, 0, 0, 4 * c4 + 2);
                    b2.w = Y.Get(4 * k4 + 3, 0, 0, 4 * c4 + 2);
                  
                    b3.x = Y.Get(4 * k4 + 0, 0, 0, 4 * c4 + 3);
                    b3.y = Y.Get(4 * k4 + 1, 0, 0, 4 * c4 + 3);
                    b3.z = Y.Get(4 * k4 + 2, 0, 0, 4 * c4 + 3);
                    b3.w = Y.Get(4 * k4 + 3, 0, 0, 4 * c4 + 3);
                    #endif


                    acc4.x += dot(a, float4(b0.x, b1.x, b2.x, b3.x));
                    acc4.y += dot(a, float4(b0.y, b1.y, b2.y, b3.y));
                    acc4.z += dot(a, float4(b0.z, b1.z, b2.z, b3.z));
                    acc4.w += dot(a, float4(b0.w, b1.w, b2.w, b3.w));
                }

                return acc4;
            }
            ENDCG
        }
    }
}
