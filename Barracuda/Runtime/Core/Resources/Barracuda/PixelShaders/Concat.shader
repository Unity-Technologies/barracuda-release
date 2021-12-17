Shader "Barracuda/Concat"
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
            TENSOR_DECL(OPred)

            uint4 _Pad;
            uint _IsFirstPass;

            fixed4 frag (v2f i) : SV_Target
            {
                TENSOR_ARGS3(X, OPred, O);
				
                uint n, h, w, c4;
                O.GetPositionFromUV(i.uv, n, h, w, c4);

                uint c;

                float4 v = 0;

                if (_IsFirstPass == 1)
                    v = 0;
                else
                    v = OPred.Get4(n, h, w, c4);

                if ((n >= _Pad.x && n - _Pad.x < X.batch) &&
                    (h >= _Pad.y && h - _Pad.y < X.height) &&
                    (w >= _Pad.z && w - _Pad.z < X.width))
                {
                    c = 4 * c4 + 0;
                    if (c >= _Pad.w && c - _Pad.w < X.channels)
                        v.x = X.Get(n - _Pad.x, h - _Pad.y, w - _Pad.z, c - _Pad.w);

                    c = 4 * c4 + 1;
                    if (c >= _Pad.w && c - _Pad.w < X.channels)
                        v.y = X.Get(n - _Pad.x, h - _Pad.y, w - _Pad.z, c - _Pad.w);

                    c = 4 * c4 + 2;
                    if (c >= _Pad.w && c - _Pad.w < X.channels)
                        v.z = X.Get(n - _Pad.x, h - _Pad.y, w - _Pad.z, c - _Pad.w);

                    c = 4 * c4 + 3;
                    if (c >= _Pad.w && c - _Pad.w < X.channels)
                        v.w = X.Get(n - _Pad.x, h - _Pad.y, w - _Pad.z, c - _Pad.w);
                }
               
                return v;
            }
            ENDCG
        }
    }
}
