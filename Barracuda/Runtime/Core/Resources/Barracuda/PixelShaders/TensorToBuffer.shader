Shader "Barracuda/TensorToBuffer"
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

            fixed frag (v2f i) : SV_Target
            {
                TENSOR_ARG(X);

                uint x = floor(i.uv.x * _OutputWidth);
                uint y = floor(i.uv.y * _OutputHeight);

                uint index = x + y * _OutputWidth;

                uint c = index % X.channels;
                uint w = (index / X.channels) % X.width;
                uint h = (index / (X.channels * X.width)) % X.height;
                uint n = (index / (X.channels * X.width * X.height)) % X.batch;

                uint c4 = c / 4;
                uint c0 = c % 4;

                float4 v = X.Get4(n, h, w, c4);

                return v[c0];
            }
            ENDCG
        }
    }
}
