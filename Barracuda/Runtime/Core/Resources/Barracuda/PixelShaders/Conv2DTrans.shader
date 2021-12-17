Shader "Barracuda/Conv2DTrans"
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

                uint2 strideMask = _Stride.xy - 1;

                float4 acc4 =  B.Get4(0, 0, 0, k4);

                uint strideH = 1;
                uint strideW = 1;

                for (uint c4 = 0; c4 < X.channels4; c4++)
                {
                    for (uint dy = 0; dy < K.GetKernelHeight(); dy += strideH)
                    {
                        for (uint dx = 0; dx < K.GetKernelWidth(); dx += strideW)
                        {
                            uint readX = (w + dx - _Pad.x) / _Stride.x;
                            uint readY = (h + dy - _Pad.y) / _Stride.y;

                            // early out if read input index fall upon leftmost outer zero padding
                            if ((w + dx) < _Pad.x) continue;
                            if ((h + dy) < _Pad.y) continue;

                            // early out if read input index fall upon rightmost outer zero padding
                            if (readX >= X.width) continue;
                            if (readY >= X.height) continue;

                            if ((w + dx - _Pad.x) % _Stride.x != 0) continue;
                            if ((h + dy - _Pad.y) % _Stride.y != 0) continue;

                            float4 v = X.Get4(n, readY, readX, c4);
				
                            float4 w0 = K.Get4(K.GetKernelHeight() - 1 - dy, K.GetKernelWidth() - 1 - dx, 4 * c4 + 0, k4);
                            float4 w1 = K.Get4(K.GetKernelHeight() - 1 - dy, K.GetKernelWidth() - 1 - dx, 4 * c4 + 1, k4);
                            float4 w2 = K.Get4(K.GetKernelHeight() - 1 - dy, K.GetKernelWidth() - 1 - dx, 4 * c4 + 2, k4);
                            float4 w3 = K.Get4(K.GetKernelHeight() - 1 - dy, K.GetKernelWidth() - 1 - dx, 4 * c4 + 3, k4);

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
