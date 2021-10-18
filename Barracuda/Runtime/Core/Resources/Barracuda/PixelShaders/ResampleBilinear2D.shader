Shader "Barracuda/ResampleBilinear2D"
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

            int4 _Pool;

            fixed4 frag(v2f i) : SV_Target
            {
                TENSOR_ARGS2(X, O);

                uint n, h, w, c4;
                O.GetPositionFromUV(i.uv, n, h, w, c4);


                float2 dstSize = float2(O.width, O.height);
                float2 srcSize = float2(X.width, X.height);
                float2 dstPos = float2(w, h);
                float2 srcPos = (dstPos + 0.5) * (srcSize / dstSize) - 0.5;

                float4 p00 = X.ClampGet4(n, floor(srcPos) + float2(0, 0), c4);
                float4 p01 = X.ClampGet4(n, floor(srcPos) + float2(0, 1), c4);
                float4 p10 = X.ClampGet4(n, floor(srcPos) + float2(1, 0), c4);
                float4 p11 = X.ClampGet4(n, floor(srcPos) + float2(1, 1), c4);

                float v = p00 * (1 - frac(srcPos.x)) * (1 - frac(srcPos.y)) +
                          p01 * (1 - frac(srcPos.x)) * frac(srcPos.y) +
                          p10 * frac(srcPos.x) * (1 - frac(srcPos.y)) +
                          p11 * frac(srcPos.x) * frac(srcPos.y);

                return v;
            }
            ENDCG
        }
    }
}
