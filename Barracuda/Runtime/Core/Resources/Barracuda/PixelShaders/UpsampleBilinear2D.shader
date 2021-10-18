Shader "Barracuda/UpsampleBilinear2D"
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

            float4 BilinearInterpolation(float fracSrcPosX, float fracSrcPosY, float4 p00, float4 p01, float4 p10, float4 p11)
            {
                float4 v = p00 * (1.0f - fracSrcPosX) * (1.0f - fracSrcPosY) +
                           p01 * (1.0f - fracSrcPosX) * fracSrcPosY +
                           p10 * fracSrcPosX * (1.0f - fracSrcPosY) +
                           p11 * fracSrcPosX * fracSrcPosY;
                return v;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                TENSOR_ARGS2(X, O);

                uint n, h, w, c4;
                O.GetPositionFromUV(i.uv, n, h, w, c4);

                float2 dstPos = float2(w, h);
                float2 srcPos = (dstPos + 0.5) / _Pool.xy - 0.5;

                float4 p00 = X.ClampGet4(n, floor(srcPos) + float2(0, 0), c4);
                float4 p01 = X.ClampGet4(n, floor(srcPos) + float2(0, 1), c4);
                float4 p10 = X.ClampGet4(n, floor(srcPos) + float2(1, 0), c4);
                float4 p11 = X.ClampGet4(n, floor(srcPos) + float2(1, 1), c4);

                float4 v = BilinearInterpolation(frac(srcPos.x), frac(srcPos.y), p00, p01, p10, p11);

                return v;
            }
            ENDCG
        }
    }
}
