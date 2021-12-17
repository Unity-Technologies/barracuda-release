Shader "Barracuda/Softmax"
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
            #pragma multi_compile ReduceN ReduceH ReduceW ReduceC

            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"

            #include "TensorTexture.cginc"

            TENSOR_DECL_O(O)
            TENSOR_DECL(X)

            uint _Axis;
            

            fixed4 frag (v2f i) : SV_Target
            {
                TENSOR_ARGS2(X, O);

                uint n, h, w, c4;
                O.GetPositionFromUV(i.uv, n, h, w, c4);

                float4 maxV = -FLT_MAX;

                uint j = 0;
                #ifdef ReduceN
                for (j = 0; j < X.batch; j++)
                #endif
                #ifdef ReduceH
                for (j = 0; j < X.height; j++)
                #endif
                #ifdef ReduceW
                for (j = 0; j < X.width; j++)
                #endif
                #ifdef ReduceC
                for (j = 0; j < X.channels4; j++)
                #endif
                {
                    float4 v = 0.0f;
                    #ifdef ReduceN
                    v = X.SafeGet4(j, uint2(w, h), c4, uint2(0, 0), -FLT_MAX);
                    #endif
                    #ifdef ReduceH
                    v = X.SafeGet4(n, uint2(w, j), c4, uint2(0, 0), -FLT_MAX);
                    #endif
                    #ifdef ReduceW
                    v = X.SafeGet4(n, uint2(j, h), c4, uint2(0, 0), -FLT_MAX);
                    #endif
                    #ifdef ReduceC
                    v = X.SafeGet4(n, uint2(w, h), j, uint2(0, 0), -FLT_MAX);
                    #endif

                    maxV = max(maxV, v);
                }
                #ifdef ReduceC
                maxV = max(maxV.x, max(maxV.y, max(maxV.z, maxV.w)));
                #endif

                float4 acc = 0.0f;
                #ifdef ReduceN
                for (j = 0; j < X.batch; j++)
                #endif
                #ifdef ReduceH
                for (j = 0; j < X.height; j++)
                #endif
                #ifdef ReduceW
                for (j = 0; j < X.width; j++)
                #endif
                #ifdef ReduceC
                for (j = 0; j < X.channels4; j++)
                #endif
                {
                    float4 v = 0.0f;
                    #ifdef ReduceN
                    v = X.Get4(j, h, w, c4);
                    #endif
                    #ifdef ReduceH
                    v = X.Get4(n, j, w, c4);
                    #endif
                    #ifdef ReduceW
                    v = X.Get4(n, h, j, c4);
                    #endif
                    #ifdef ReduceC
                    v = X.Get4(n, h, w, j);
                    #endif

                    #ifdef ReduceC
                    if (4 * j + 0 < X.channels)
                        acc.x += exp(v.x - maxV.x);
                    if (4 * j + 1 < X.channels)
                        acc.y += exp(v.y - maxV.y);
                    if (4 * j + 2 < X.channels)
                        acc.z += exp(v.z - maxV.z);
                    if (4 * j + 3 < X.channels)
                        acc.w += exp(v.w - maxV.w);
                    #else
                    acc += exp(v - maxV);
                    #endif
                }
                #ifdef ReduceC
                acc = acc.x + acc.y + acc.z + acc.w;
                #endif

                float4  v = X.Get4(n, h, w, c4);
                v = exp(v - maxV) / acc;

				return v;
            }
            ENDCG
        }
    }
}
