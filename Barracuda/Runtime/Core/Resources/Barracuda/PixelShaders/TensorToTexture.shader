Shader "Barracuda/TensorToTexture"
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

            TENSOR_DECL(X)

            uint _OutputHeight;
            uint _OutputWidth;

            float4 _Scale;
            float4 _Bias;
            uint4 _Pad;
            int _FlipY;

            fixed4 frag (v2f i) : SV_Target
            {
                TENSOR_ARG(X);
				
                uint b = _Pad.x;
                uint x = floor(i.uv.x * _OutputWidth);
                uint y = floor(i.uv.y * _OutputHeight);
				
                if (_FlipY == 1)
                    y = floor((1 - i.uv.y) * _OutputHeight);
				
                float4 v = 0;
				
                uint c = _Pad.y;
                uint c4 = c / 4;
				
                int channelRemainder = X.channels - c;
                if (channelRemainder == 1)
                {
                    // broadcast to all channels
                    v = _Scale.x * X.Get4(b, y, x, c4).x + _Bias.x;
                }
                else if (channelRemainder == 2)
                {
                    v = _Scale * X.Get4(b, y, x, c4) + _Bias;
                    v.b = 0;
                    v.a = 1;
                }
                else if (channelRemainder == 3)
                {
                    v = _Scale * X.Get4(b, y, x, c4) + _Bias;
                    v.a = 1;
                }
                else if (channelRemainder >= 4)
                {
                    v = _Scale * X.Get4(b, y, x, c4) + _Bias;
                }

                return v;
            }
            ENDCG
        }
    }
}
