Shader "Barracuda/Broadcast"
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
            #pragma multi_compile Sub Pow Mul Min Mean Max LogicalXor LogicalOr LogicalAnd LessEqual Less GreaterEqual Greater Equal Div Add

            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            
            #include "TensorTexture.cginc"

            TENSOR_DECL_O(O)
            TENSOR_DECL(X)
            TENSOR_DECL(B)

            int _IsFirstDispatch;
            float _Alpha;

            float signed_pow(float f, float e)
            {
                // handle negative f
                float v = pow(abs(f), e);
                float s = (e % 2 == 1) ?
                    sign(f) :    // exponent is odd  => sign(f) * pow(abs(f), e)
                    1;            // exponent is even => pow(abs(f), e)
                return v * s;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                TENSOR_ARGS3(X, B, O);

                uint n, h, w, c4;
                O.GetPositionFromUV(i.uv, n, h, w, c4);

                float4 v = 0.0;
                #ifdef Sub
                    v = X.BroadcastGet4(n, h, w, c4) - B.BroadcastGet4(n, h, w, c4);
                #endif
                #ifdef Pow
                    float4 a = X.BroadcastGet4(n, h, w, c4);
                    float4 b = B.BroadcastGet4(n, h, w, c4);
                    v.x = signed_pow(a.x, b.x);
                    v.y = signed_pow(a.y, b.y);
                    v.z = signed_pow(a.z, b.z);
                    v.w = signed_pow(a.w, b.w);
                #endif
                #ifdef Mul
                    v = X.BroadcastGet4(n, h, w, c4) * B.BroadcastGet4(n, h, w, c4);
                #endif
                #ifdef Min
                    v = min(X.BroadcastGet4(n, h, w, c4), B.BroadcastGet4(n, h, w, c4));
                #endif
                #ifdef Mean
                    float4 a = X.BroadcastGet4(n, h, w, c4);
                    a *= _IsFirstDispatch ? _Alpha : 1.0f;
                    float4 b = B.BroadcastGet4(n, h, w, c4) * _Alpha;
                    v = a + b;
                #endif
                #ifdef Max
                    v = max(X.BroadcastGet4(n, h, w, c4), B.BroadcastGet4(n, h, w, c4));
                #endif
                #ifdef LogicalXor
                    float4 a = X.BroadcastGet4(n, h, w, c4);
                    float4 b = B.BroadcastGet4(n, h, w, c4);

                    a.x = (a.x == 0.0f) ? 0.0f : 1.0f;
                    a.y = (a.y == 0.0f) ? 0.0f : 1.0f;
                    a.z = (a.z == 0.0f) ? 0.0f : 1.0f;
                    a.w = (a.w == 0.0f) ? 0.0f : 1.0f;

                    b.x = (b.x == 0.0f) ? 0.0f : 1.0f;
                    b.y = (b.y == 0.0f) ? 0.0f : 1.0f;
                    b.z = (b.z == 0.0f) ? 0.0f : 1.0f;
                    b.w = (b.w == 0.0f) ? 0.0f : 1.0f;

                    v = a * (1 - 2 * b) + b;
                #endif
                #ifdef LogicalOr
                    float4 a = X.BroadcastGet4(n, h, w, c4);
                    float4 b = B.BroadcastGet4(n, h, w, c4);

                    a.x = (a.x == 0.0f) ? 0.0f : 1.0f;
                    a.y = (a.y == 0.0f) ? 0.0f : 1.0f;
                    a.z = (a.z == 0.0f) ? 0.0f : 1.0f;
                    a.w = (a.w == 0.0f) ? 0.0f : 1.0f;

                    b.x = (b.x == 0.0f) ? 0.0f : 1.0f;
                    b.y = (b.y == 0.0f) ? 0.0f : 1.0f;
                    b.z = (b.z == 0.0f) ? 0.0f : 1.0f;
                    b.w = (b.w == 0.0f) ? 0.0f : 1.0f;

                    v = a * (1 - b) + b;
                #endif
                #ifdef LogicalAnd
                    float4 a = X.BroadcastGet4(n, h, w, c4);
                    float4 b = B.BroadcastGet4(n, h, w, c4);

                    a.x = (a.x == 0.0f) ? 0.0f : 1.0f;
                    a.y = (a.y == 0.0f) ? 0.0f : 1.0f;
                    a.z = (a.z == 0.0f) ? 0.0f : 1.0f;
                    a.w = (a.w == 0.0f) ? 0.0f : 1.0f;

                    b.x = (b.x == 0.0f) ? 0.0f : 1.0f;
                    b.y = (b.y == 0.0f) ? 0.0f : 1.0f;
                    b.z = (b.z == 0.0f) ? 0.0f : 1.0f;
                    b.w = (b.w == 0.0f) ? 0.0f : 1.0f;

                    v.x = a.x * b.x != 0.0 ? 1.0f : 0.0f;
                    v.y = a.y * b.y != 0.0 ? 1.0f : 0.0f;
                    v.z = a.z * b.z != 0.0 ? 1.0f : 0.0f;
                    v.w = a.w * b.w != 0.0 ? 1.0f : 0.0f;
                #endif
                #ifdef LessEqual
                    float4 a = X.BroadcastGet4(n, h, w, c4);
                    float4 b = B.BroadcastGet4(n, h, w, c4);

                    v.x = (a.x <= b.x) ? 1.0f : 0.0f;
                    v.y = (a.y <= b.y) ? 1.0f : 0.0f;
                    v.z = (a.z <= b.z) ? 1.0f : 0.0f;
                    v.w = (a.w <= b.w) ? 1.0f : 0.0f;
                #endif
                #ifdef Less
                    float4 a = X.BroadcastGet4(n, h, w, c4);
                    float4 b = B.BroadcastGet4(n, h, w, c4);

                    v.x = (a.x < b.x) ? 1.0f : 0.0f;
                    v.y = (a.y < b.y) ? 1.0f : 0.0f;
                    v.z = (a.z < b.z) ? 1.0f : 0.0f;
                    v.w = (a.w < b.w) ? 1.0f : 0.0f;
                #endif
                #ifdef GreaterEqual
                    float4 a = X.BroadcastGet4(n, h, w, c4);
                    float4 b = B.BroadcastGet4(n, h, w, c4);

                    v.x = (a.x >= b.x) ? 1.0f : 0.0f;
                    v.y = (a.y >= b.y) ? 1.0f : 0.0f;
                    v.z = (a.z >= b.z) ? 1.0f : 0.0f;
                    v.w = (a.w >= b.w) ? 1.0f : 0.0f;
                #endif
                #ifdef Greater
                    float4 a = X.BroadcastGet4(n, h, w, c4);
                    float4 b = B.BroadcastGet4(n, h, w, c4);

                    v.x = (a.x > b.x) ? 1.0f : 0.0f;
                    v.y = (a.y > b.y) ? 1.0f : 0.0f;
                    v.z = (a.z > b.z) ? 1.0f : 0.0f;
                    v.w = (a.w > b.w) ? 1.0f : 0.0f;
                #endif
                #ifdef Equal
                    float4 a = X.BroadcastGet4(n, h, w, c4);
                    float4 b = B.BroadcastGet4(n, h, w, c4);

                    v.x = (a.x == b.x) ? 1.0f : 0.0f;
                    v.y = (a.y == b.y) ? 1.0f : 0.0f;
                    v.z = (a.z == b.z) ? 1.0f : 0.0f;
                    v.w = (a.w == b.w) ? 1.0f : 0.0f;
                #endif
                #ifdef Div
                    v = X.BroadcastGet4(n, h, w, c4) / B.BroadcastGet4(n, h, w, c4);
                #endif
                #ifdef Add
                    v = X.BroadcastGet4(n, h, w, c4) + B.BroadcastGet4(n, h, w, c4);
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
