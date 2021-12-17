Shader "Barracuda/OneHot"
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
            #pragma multi_compile Input1D Input2D Input3D

            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"

            #include "TensorTexture.cginc"

			
            TENSOR_DECL_O(O)
            TENSOR_DECL(X)

            float _Alpha;
            float _Beta;

            fixed4 frag (v2f i) : SV_Target
            {
                TENSOR_ARGS2(X, O);

                uint n, h, w, c4;
                O.GetPositionFromUV(i.uv, n, h, w, c4);

                float4 v = 0.0f;
                #ifdef Input1D
                    // O = (X.flatHeight, 1, 1, depth)
                    uint index = (uint)(X.Get(n, 0, 0, 0));
                    v.x = ((4 * c4 + 0) == index) ? _Alpha : _Beta;
                    v.y = ((4 * c4 + 1) == index) ? _Alpha : _Beta;
                    v.z = ((4 * c4 + 2) == index) ? _Alpha : _Beta;
                    v.w = ((4 * c4 + 3) == index) ? _Alpha : _Beta;
                #endif
                #ifdef Input2D
                    // O = (X.flatHeight, 1, depth, X.flatWidth)
                    uint4 index = (uint4)(X.Get4(n, 0, 0, c4));
                    v.x = (w == index.x) ? _Alpha : _Beta;
                    v.y = (w == index.y) ? _Alpha : _Beta;
                    v.z = (w == index.z) ? _Alpha : _Beta;
                    v.w = (w == index.w) ? _Alpha : _Beta;
                #endif
                #ifdef Input3D
                    // O = (X.batch, X.height, depth, X.channels
                    uint4 index = (uint4)(X.Get4(n, 0, w, c4));
                    v.x = (w == index.x) ? _Alpha : _Beta;
                    v.y = (w == index.y) ? _Alpha : _Beta;
                    v.z = (w == index.z) ? _Alpha : _Beta;
                    v.w = (w == index.w) ? _Alpha : _Beta;
                #endif

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
