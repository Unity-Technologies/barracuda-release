Shader "Barracuda/Conv2D"
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
            TENSOR_DECL(K)
            TENSOR_DECL(B)
	
            uint4 _Pad;
            uint4 _Stride;
			
            fixed4 frag(v2f i) : SV_Target
            {
                TENSOR_O(O);
                TENSOR_ARG(X);
                TENSOR_ARG(K);
                TENSOR_ARG(B);

                uint n, h, w, k4;
                O.GetPositionFromUV(i.uv, n, h, w, k4);

                float4 acc4 =  B.Get4(0, 0, 0, k4);

                for (uint c4 = 0; c4 < X.channels4; c4++)
                {
                    for (uint dy = 0; dy < K.GetKernelHeight(); ++dy)
                    {
                        for (uint dx = 0; dx < K.GetKernelWidth(); ++dx)
                        {
                            uint2 pos = uint2(w, h) * _Stride.xy + uint2(dx, dy);
                            float4 v = X.SafeGet4(n, pos, c4, _Pad.xy);
				
                            float4 w0 = K.Get4(dy, dx, 4 * c4 + 0, k4);
                            float4 w1 = K.Get4(dy, dx, 4 * c4 + 1, k4);
                            float4 w2 = K.Get4(dy, dx, 4 * c4 + 2, k4);
                            float4 w3 = K.Get4(dy, dx, 4 * c4 + 3, k4);

                            acc4.x += dot(v, float4(w0.x, w1.x, w2.x, w3.x));
                            acc4.y += dot(v, float4(w0.y, w1.y, w2.y, w3.y));
                            acc4.z += dot(v, float4(w0.z, w1.z, w2.z, w3.z));
                            acc4.w += dot(v, float4(w0.w, w1.w, w2.w, w3.w));
                        }
                    }
                }

                return ApplyFusedActivation(acc4);
            }
            ENDCG
        }
    }
}
