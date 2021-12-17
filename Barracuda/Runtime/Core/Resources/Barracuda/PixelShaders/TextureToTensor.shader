Shader "Barracuda/TextureToTensor"
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

            float4 _Scale;
            float4 _Bias;
            bool _FlipY;
            int4 _ChannelReadMap;
            int4 _ChannelWriteMask;
            int4 _ChannelWriteMap;
            uint2 _Pool;

            Texture2D<float4> Xtex2D;
            SamplerState samplerXtex2D { Filter = MIN_MAG_LINEAR_MIP_POINT; AddressU = Clamp; AddressV = Clamp; };

            fixed4 frag (v2f i) : SV_Target
            {
                TENSOR_O(O);

                uint n, h, w, c4;
                O.GetPositionFromUV(i.uv, n, h, w, c4);
				
                float2 uv = float2(w, h) + float2(0.5f, 0.5f);
                uv.xy /= _Pool.xy;

                if (_FlipY)
                    uv.y = 1 - uv.y;

                float4 v = Xtex2D.SampleLevel(samplerXtex2D, uv.xy, 0);

                bool specialCaseWhenChannelMaskIsEmptyStoresAverage = true;

                float4 value = 0;
                if (_ChannelWriteMask.x == 1)
                {
                    float v0 = 0.0f;
                    if (_ChannelReadMap.x >= 0)
                        v0 = _Scale[_ChannelReadMap.x] * v[_ChannelReadMap.x] + _Bias[_ChannelReadMap.x];

                    if (_ChannelWriteMap.x == 0)
                        value[0] = v0;
                    else if (_ChannelWriteMap.x == 1)
                        value[1] = v0;
                    else if (_ChannelWriteMap.x == 2)
                        value[2] = v0;
                    else if (_ChannelWriteMap.x == 3)
                        value[3] = v0;

                    specialCaseWhenChannelMaskIsEmptyStoresAverage = false;
                }
                if (_ChannelWriteMask.y == 1)
                {
                    float v1 = 0.0f;
                    if (_ChannelReadMap.y >= 0)
                        v1 = _Scale[_ChannelReadMap.y] * v[_ChannelReadMap.y] + _Bias[_ChannelReadMap.y];

                    if (_ChannelWriteMap.y == 0)
                        value[0] = v1;
                    else if (_ChannelWriteMap.y == 1)
                        value[1] = v1;
                    else if (_ChannelWriteMap.y == 2)
                        value[2] = v1;
                    else if (_ChannelWriteMap.y == 3)
                        value[3] = v1;

                    specialCaseWhenChannelMaskIsEmptyStoresAverage = false;
                }
                if (_ChannelWriteMask.z == 1)
                {
                    float v2 = 0.0f;
                    if (_ChannelReadMap.z >= 0)
                        v2 = _Scale[_ChannelReadMap.z] * v[_ChannelReadMap.z] + _Bias[_ChannelReadMap.z];

                    if (_ChannelWriteMap.z == 0)
                        value[0] = v2;
                    else if (_ChannelWriteMap.z == 1)
                        value[1] = v2;
                    else if (_ChannelWriteMap.z == 2)
                        value[2] = v2;
                    else if (_ChannelWriteMap.z == 3)
                        value[3] = v2;

                    specialCaseWhenChannelMaskIsEmptyStoresAverage = false;
                }
                if (_ChannelWriteMask.w == 1)
                {
                    float v3 = 1.0f;
                    if (_ChannelReadMap.w >= 0)
                        v3 = _Scale[_ChannelReadMap.w] * v[_ChannelReadMap.w] + _Bias[_ChannelReadMap.w];

                    if (_ChannelWriteMap.w == 0)
                        value[0] = v3;
                    else if (_ChannelWriteMap.w == 1)
                        value[1] = v3;
                    else if (_ChannelWriteMap.w == 2)
                        value[2] = v3;
                    else if (_ChannelWriteMap.w == 3)
                        value[3] = v3;

                    specialCaseWhenChannelMaskIsEmptyStoresAverage = false;
                }

                if (specialCaseWhenChannelMaskIsEmptyStoresAverage)
                {
                    v = _Scale * v + _Bias;
                    float avg = (v.r + v.g + v.b) / 3.0f;
                    value = avg;
                }

                return value;
            }
            ENDCG
        }
    }
}
