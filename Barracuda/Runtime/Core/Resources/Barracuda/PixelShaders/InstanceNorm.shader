Shader "Barracuda/InstanceNorm"
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

            float _Epsilon;

            fixed4 frag (v2f i) : SV_Target
            {
                TENSOR_ARGS4(X, W, B, O);

                uint n, h, w, c4;
                O.GetPositionFromUV(i.uv, n, h, w, c4);

                float4 gamma = W.Get4(0, 0, 0, c4);
                float4 beta = B.Get4(0, 0, 0, c4);

                float4 alpha = X.Get4(n, 0, 0, c4);

                uint y, x;

                float4 sum = 0, sumSq = 0;
                for (y = 0; y < X.height; ++y)
                    for (x = 0; x < X.width; ++x)
                    {
                        float4 delta = X.Get4(n, y, x, c4) - alpha;
                        sum += delta;
                        sumSq += delta * delta;
                    }

                float4 mean = alpha + sum / (X.width * X.height);
                float4 var = (sumSq - (sum * sum) / (X.width * X.height)) / (X.width * X.height);

                float4 v = X.Get4(n, h, w, c4);
                v = gamma * (v - mean) / sqrt(var + _Epsilon) + beta;
                return ApplyFusedActivation(v);
            }
            ENDCG
        }
    }
}
