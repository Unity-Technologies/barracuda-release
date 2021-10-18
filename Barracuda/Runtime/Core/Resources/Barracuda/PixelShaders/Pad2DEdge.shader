Shader "Barracuda/Pad2DEdge"
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

            void ClampHWToTensorShape(uint2 shape, inout int height, inout int width)
            {
                width = clamp(width, 0, (int)shape.x - 1);
                height = clamp(height, 0, (int)shape.y - 1);
            }

            fixed4 frag (v2f i) : SV_Target
            {
                TENSOR_ARGS2(X, O);
				
                uint n, h, w, c4;
                O.GetPositionFromUV(i.uv, n, h, w, c4);

                int readX = w - _Pad.x;
                int readY = h - _Pad.y;
                uint2 Xshape = uint2(X.width, X.height);

                //clamp read indices to source
                ClampHWToTensorShape(Xshape, readY, readX);

                float4 v = X.Get4(n, readY, readX, c4);
				
                return v;
            }
            ENDCG
        }
    }
}
