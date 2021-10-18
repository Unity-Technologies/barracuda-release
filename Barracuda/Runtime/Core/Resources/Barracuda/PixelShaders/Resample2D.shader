Shader "Barracuda/Resample2D"
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
                float2 srcPos = floor(dstPos / (dstSize / srcSize));

                float4 v = X.ClampGet4(n, srcPos, c4);

                return v;
            }
            ENDCG
        }
    }
}
