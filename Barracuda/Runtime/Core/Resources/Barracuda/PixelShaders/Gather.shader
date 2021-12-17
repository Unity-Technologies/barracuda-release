Shader "Barracuda/Gather"
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
            #pragma multi_compile Input1D Input2D

            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"

            #include "TensorTexture.cginc"

			
            TENSOR_DECL_O(O)
            TENSOR_DECL(X)
            TENSOR_DECL(K)

            uint _Axis;

            fixed4 frag (v2f i) : SV_Target
            {
                TENSOR_ARGS3(X, K, O);

                uint n, h, w, c4;
                O.GetPositionFromUV(i.uv, n, h, w, c4);

                float4 v = 0.0f;
                if (_Axis == 0)
                    v = X.Get4((uint)K.Get(n,0,0,0), h, w, c4);
                else if (_Axis == 1)
                    v = X.Get4(n, (uint)K.Get(h,0,0,0), w, c4);
                else if (_Axis == 2)
                    v = X.Get4(n, h, (uint)K.Get(w,0,0,0), c4);
                else if (_Axis == 3)
                {
                    v.x = X.Get(n, h, w, (uint)K.Get(4 * c4 + 0, 0, 0, 0));
                    v.y = X.Get(n, h, w, (uint)K.Get(4 * c4 + 1, 0, 0, 0));
                    v.z = X.Get(n, h, w, (uint)K.Get(4 * c4 + 2, 0, 0, 0));
                    v.w = X.Get(n, h, w, (uint)K.Get(4 * c4 + 3, 0, 0, 0));
                }

                if (4 * c4 >= O.channels)
                    v.x = 0.0f;
                if (4 * c4 + 1 >= O.channels)
                    v.y = 0.0f;
                if (4 * c4 + 2 >= O.channels)
                    v.z = 0.0f;
                if (4 * c4 + 3 >= O.channels)
                    v.w = 0.0f;

                return v;
            }
            ENDCG
        }
    }
}
