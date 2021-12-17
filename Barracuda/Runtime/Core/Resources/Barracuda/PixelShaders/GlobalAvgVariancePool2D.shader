Shader "Barracuda/GlobalAvgVariancePool2D"
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

                float mean = 0;
                float mean2 = 0;
                for (uint y = 0; y < X.height; ++y)
                {
                    for (uint x = 0; x < X.width; ++x)
                    {
                        float4 v = X.Get4(n, y, x, c4);
                        mean += v;
                        mean2 += v * v;
                    }
                }

                mean /= (X.height * X.width);
                mean2 /= (X.height * X.width);

                if (h == 0)
                    return mean;
                else if (h == 1)
                    return mean2;
                else
                    return 0;
            }
            ENDCG
        }
    }
}
