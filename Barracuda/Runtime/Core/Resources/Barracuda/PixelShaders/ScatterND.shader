Shader "Barracuda/ScatterND"
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
            #pragma multi_compile ReduceNone ReduceAdd ReduceMul

            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"

            #include "TensorTexture.cginc"

			
            TENSOR_DECL_O(O)
            TENSOR_DECL(X)
            TENSOR_DECL(K)
            TENSOR_DECL(W)

            fixed4 frag (v2f i) : SV_Target
            {
                TENSOR_ARGS4(X, K, W, O);

                uint n, h, w, c4;
                O.GetPositionFromUV(i.uv, n, h, w, c4);

                float4 v = X.Get4(n, h, w, c4);

                for (uint idx = 0; idx < K.GetFlatWidth(); idx++)
                {
                    uint cK = idx % K.channels;
                    uint wK = (idx / K.channels) % K.width;
                    uint hK = (idx / K.channels / K.width) % K.height;

                    uint indexRemap = (uint)(K.Get(0, hK, wK, cK));

                    if (4 * c4 + 0 == indexRemap)
                    {
                        float vw = W.Get(n, h, w, idx);
                        
                        #ifdef ReduceNone
                        v[0] = vw;
                        #endif
                        #ifdef ReduceAdd
                        v[0] += vw;
                        #endif
                        #ifdef ReduceMul
                        v[0] += vw;
                        #endif
                    }

                    if (4 * c4 + 1 == indexRemap)
                    {
                        float vw = W.Get(n, h, w, idx);
                        
                        #ifdef ReduceNone
                        v[1] = vw;
                        #endif
                        #ifdef ReduceAdd
                        v[1] += vw;
                        #endif
                        #ifdef ReduceMul
                        v[1] += vw;
                        #endif
                    }

                    if (4 * c4 + 2 == indexRemap)
                    {
                        float vw = W.Get(n, h, w, idx);
                        
                        #ifdef ReduceNone
                        v[2] = vw;
                        #endif
                        #ifdef ReduceAdd
                        v[2] += vw;
                        #endif
                        #ifdef ReduceMul
                        v[2] += vw;
                        #endif
                    }

                    if (4 * c4 + 3 == indexRemap)
                    {
                        float vw = W.Get(n, h, w, idx);
                        
                        #ifdef ReduceNone
                        v[3] = vw;
                        #endif
                        #ifdef ReduceAdd
                        v[3] += vw;
                        #endif
                        #ifdef ReduceMul
                        v[3] += vw;
                        #endif
                    }
                }
                
                if (4 * c4 >= O.channels)
                    v.x = 0.0f;
                if (4 * c4 + 1 >= O.channels)
                    v.y = 0.0f;
                if (4 * c4 + 2 >= O.channels)
                    v.z = 0.0f;
                if (4 * c4 + 3 >= O.channels)
                    v.w = 0.0f;

                return v;
            }
            ENDCG
        }
    }
}
