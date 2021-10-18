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
            uint4 _Pool;
            uint4 _Pad;

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

                float4 value = 0;
				
                value.x = _Scale[_ChannelReadMap.x] * v[_ChannelReadMap.x] + _Bias[_ChannelReadMap.x];
                value.x *= (_ChannelWriteMask.x == 1 ? 1.0f : 0.0f);

                value.y = _Scale[_ChannelReadMap.y] * v[_ChannelReadMap.y] + _Bias[_ChannelReadMap.y];
                value.y *= (_ChannelWriteMask.y == 1 ? 1.0f : 0.0f);

                value.z = _Scale[_ChannelReadMap.z] * v[_ChannelReadMap.z] + _Bias[_ChannelReadMap.z];
                value.z *= (_ChannelWriteMask.z == 1 ? 1.0f : 0.0f);

                value.w = _Scale[_ChannelReadMap.w] * v[_ChannelReadMap.w] + _Bias[_ChannelReadMap.w];
                value.w *= (_ChannelWriteMask.w == 1 ? 1.0f : 0.0f);


                if (all(_ChannelReadMap != 1))
                {
                    v = _Scale * v + _Bias;
                    float avg = (v.r + v.g + v.b) / 3.0f;
                    value.r = avg;
                    value.gba = 0;
                }


                if (4 * c4 + 0 < _Pad.w)
                value.x = 0;
                if (4 * c4 + 1 < _Pad.w)
                value.y = 0;
                if (4 * c4 + 2 < _Pad.w)
                value.z = 0;
                if (4 * c4 + 3 < _Pad.w)
                value.w = 0;

                return value;
            }
            ENDCG
        }
    }
}
