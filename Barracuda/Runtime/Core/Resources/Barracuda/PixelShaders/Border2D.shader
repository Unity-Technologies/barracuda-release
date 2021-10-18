Shader "Barracuda/Border2D"
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

            int4 _Pad;
            int4 _Pool;
            float _Beta;

            fixed4 frag (v2f i) : SV_Target
            {
                TENSOR_ARGS2(X, O);
				
                uint n, h, w, c4;
                O.GetPositionFromUV(i.uv, n, h, w, c4);

                int croppedWidth = _Pool.x;
                int croppedHeight = _Pool.y;

                int readX = (int)(w) - _Pad.x;
                int readY = (int)(h) - _Pad.y;

                float4 v = 0.0f;
                if (readX < 0 || readX >= croppedWidth ||
                    readY < 0 || readY >= croppedHeight)
                {
                    v = _Beta;
                }
                else
                {
                    v = X.Get4(n, readY, readX, c4);
                }
				
                return v;
            }
            ENDCG
        }
    }
}
