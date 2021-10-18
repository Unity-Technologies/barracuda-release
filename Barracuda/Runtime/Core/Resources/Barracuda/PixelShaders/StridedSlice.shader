Shader "Barracuda/StridedSlice"
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

            int4 _Starts;
            int4 _Stride;


            fixed4 frag (v2f i) : SV_Target
            {
                TENSOR_ARGS2(X, O);
				
                uint n, h, w, c4;
                O.GetPositionFromUV(i.uv, n, h, w, c4);

                float4 v = 0;

                v.x = X.Get(_Starts.x + n * _Stride.x, _Starts.y + h * _Stride.y, _Starts.z + w * _Stride.z, _Starts.w + (4 * c4 + 0) * _Stride.w);
                v.y = X.Get(_Starts.x + n * _Stride.x, _Starts.y + h * _Stride.y, _Starts.z + w * _Stride.z, _Starts.w + (4 * c4 + 1) * _Stride.w);
                v.z = X.Get(_Starts.x + n * _Stride.x, _Starts.y + h * _Stride.y, _Starts.z + w * _Stride.z, _Starts.w + (4 * c4 + 2) * _Stride.w);
                v.w = X.Get(_Starts.x + n * _Stride.x, _Starts.y + h * _Stride.y, _Starts.z + w * _Stride.z, _Starts.w + (4 * c4 + 3) * _Stride.w);
                
                v.x = 4 * c4 + 0 < X.channels ? v.x : 0.0f;
                v.y = 4 * c4 + 1 < X.channels ? v.y : 0.0f;
                v.z = 4 * c4 + 2 < X.channels ? v.z : 0.0f;
                v.w = 4 * c4 + 3 < X.channels ? v.w : 0.0f;

                return v;
            }
            ENDCG
        }
    }
}
