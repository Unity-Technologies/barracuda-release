Shader "Barracuda/Transpose"
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

            fixed4 frag (v2f i) : SV_Target
            {
                TENSOR_ARGS2(X, O);
				
                uint n, h, w, c4;
                O.GetPositionFromUV(i.uv, n, h, w, c4);

                uint4 index = uint4(n, h, w, c4);

                float4 v = 0;
                v.x = X.Get(index[_Pool.x], index[_Pool.y], index[_Pool.z], index[4 * _Pool.w + 0]);
                v.y = X.Get(index[_Pool.x], index[_Pool.y], index[_Pool.z], index[4 * _Pool.w + 1]);
                v.z = X.Get(index[_Pool.x], index[_Pool.y], index[_Pool.z], index[4 * _Pool.w + 2]);
                v.w = X.Get(index[_Pool.x], index[_Pool.y], index[_Pool.z], index[4 * _Pool.w + 3]);
				
                return v;
            }
            ENDCG
        }
    }
}
