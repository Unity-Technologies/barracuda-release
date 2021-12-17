Shader "Barracuda/Reduce"
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
            #pragma multi_compile ArgMax ArgMin ReduceMin ReduceMax ReduceSum ReduceMean ReduceProd
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

                #ifdef ArgMin
                uint4 minIdx = uint4(0, 1, 2, 3);
                #endif
                #ifdef ArgMax
                uint4 maxIdx = uint4(0, 1, 2, 3);
                #endif

                float defaultValue = 0.0f;
                #ifdef ArgMin
                defaultValue = FLT_MAX;
                #endif
                #ifdef ArgMax
                defaultValue = -FLT_MAX;
                #endif
                #ifdef ReduceMin
                defaultValue = FLT_MAX;
                #endif
                #ifdef ReduceMax
                defaultValue = -FLT_MAX;
                #endif
                #ifdef ReduceProd
                defaultValue = 1.0f;
                #endif

                float4 acc4 = defaultValue;

                #ifdef ReduceN
                for (uint j = 0; j < X.batch; j++)
                #endif
                #ifdef ReduceH
                for (uint j = 0; j < X.height; j++)
                #endif
                #ifdef ReduceW
                for (uint j = 0; j < X.width; j++)
                #endif
                #ifdef ReduceC
                for (uint j = 0; j < X.channels4; j++)
                #endif
                {
                    float4 v = 0.0f;
                    #ifdef ReduceN
                    v = X.SafeGet4(j, uint2(w, h), c4, uint2(0, 0), defaultValue);
                    #endif
                    #ifdef ReduceH
                    v = X.SafeGet4(n, uint2(w, j), c4, uint2(0, 0), defaultValue);
                    #endif
                    #ifdef ReduceW
                    v = X.SafeGet4(n, uint2(j, h), c4, uint2(0, 0), defaultValue);
                    #endif
                    #ifdef ReduceC
                    v = X.SafeGet4(n, uint2(w, h), j, uint2(0, 0), defaultValue);
                    #endif

                    #ifdef ArgMin
                    uint4 index = j;
                    #ifdef ReduceC
                    index = uint4(0, 1, 2, 3) + 4 * j;
                    #endif
                    if (v.x < acc4.x)
                    {
                        acc4.x = v.x;
                        minIdx.x = index.x;
                    }
                    if (v.y < acc4.y)
                    {
                        acc4.y = v.y;
                        minIdx.y = index.y;
                    }
                    if (v.z < acc4.z)
                    {
                        acc4.z = v.z;
                        minIdx.z = index.z;
                    }
                    if (v.w < acc4.w)
                    {
                        acc4.w = v.w;
                        minIdx.w = index.w;
                    }
                    #endif
                    #ifdef ArgMax
                    uint4 index = j;
                    #ifdef ReduceC
                    index = uint4(0, 1, 2, 3) + 4 * j;
                    #endif
                    if (v.x > acc4.x)
                    {
                        acc4.x = v.x;
                        maxIdx.x = index.x;
                    }
                    if (v.y > acc4.y)
                    {
                        acc4.y = v.y;
                        maxIdx.y = index.y;
                    }
                    if (v.z > acc4.z)
                    {
                        acc4.z = v.z;
                        maxIdx.z = index.z;
                    }
                    if (v.w > acc4.w)
                    {
                        acc4.w = v.w;
                        maxIdx.w = index.w;
                    }
                    #endif
                    #ifdef ReduceMin
                    acc4 = min(acc4, v);
                    #endif
                    #ifdef ReduceMax
                    acc4 = max(acc4, v);
                    #endif
                    #ifdef ReduceSum
                    acc4 = acc4 + v;
                    #endif
                    #ifdef ReduceMean
                    acc4 = acc4 + v;
                    #endif
                    #ifdef ReduceProd
                    acc4 = acc4 * v;
                    #endif
                }

                #ifdef ReduceC
                #ifdef ArgMin
                    if (acc4[1] < acc4[0])
                    {
                        acc4[0] = acc4[1];
                        minIdx[0] = minIdx[1];
                    }
                    if (acc4[2] < acc4[0])
                    {
                        acc4[0] = acc4[2];
                        minIdx[0] = minIdx[2];
                    }
                    if (acc4[3] < acc4[0])
                    {
                        acc4[0] = acc4[3];
                        minIdx[0] = minIdx[3];
                    }
                    acc4.x = minIdx.x;
                    acc4.yzw = 0;
                #endif
                #ifdef ArgMax
                    if (acc4[1] > acc4[0])
                    {
                        acc4[0] = acc4[1];
                        maxIdx[0] = maxIdx[1];
                    }
                    if (acc4[2] > acc4[0])
                    {
                        acc4[0] = acc4[2];
                        maxIdx[0] = maxIdx[2];
                    }
                    if (acc4[3] > acc4[0])
                    {
                        acc4[0] = acc4[3];
                        maxIdx[0] = maxIdx[3];
                    }
                    acc4.x = maxIdx.x;
                    acc4.yzw = 0;
                #endif
                #ifdef ReduceMin
                    acc4.x = min(acc4.x, min(acc4.y, min(acc4.z, acc4.w)));
                    acc4.yzw = 0;
                #endif
                #ifdef ReduceMax
                    acc4.x = max(acc4.x, max(acc4.y, max(acc4.z, acc4.w)));
                    acc4.yzw = 0;
                #endif
                #ifdef ReduceSum
                    acc4.x = acc4.x + acc4.y + acc4.z + acc4.w;
                    acc4.yzw = 0;
                #endif
                #ifdef ReduceMean
                    acc4.x = acc4.x + acc4.y + acc4.z + acc4.w;
                    acc4.yzw = 0;
                #endif
                #ifdef ReduceProd
                    acc4.x = acc4.x * acc4.y * acc4.z * acc4.w;
                    acc4.yzw = 0;
                #endif
                #endif

                #ifdef ReduceMean
                #ifdef ReduceN
                    acc4 /= X.batch;
                #endif
                #ifdef ReduceH
                    acc4 /= X.height;
                #endif
                #ifdef ReduceW
                    acc4 /= X.width;
                #endif
                #ifdef ReduceC
                    acc4 /= X.channels;
                #endif
                #endif

				return acc4;
            }
            ENDCG
        }
    }
}
