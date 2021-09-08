#include "Tensor.cginc"

float _Alpha;
int _IsFirstDispatch;
uint4 _XStrides;
uint4 _SStrides;
uint4 _BStrides;

TENSOR_DECL(X)
TENSOR_DECL(S)
TENSOR_DECL(B)
TENSOR_DECL_RW(O)

void DispatchThreadIdToTensorIndices(uint3 dispatchThreadID, out uint c, out uint x, out uint y)
{
#if CHANNELS_FIRST
    //DISPATCH ARGS(O.width, O.height, O.channels);
    x = dispatchThreadID.x;
    y = dispatchThreadID.y;
    c = dispatchThreadID.z;
#else
    //DISPATCH ARGS(O.channels, O.width, O.height);
    c = dispatchThreadID.x;
    x = dispatchThreadID.y;
    y = dispatchThreadID.z;
#endif
}

NUMTHREADS((4,8,8), (4,8,4), (4,4,4))
void KERNEL_FUNC(BroadcastAdd)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    TENSOR_TWOINPUTS(X, B, O);
    uint c, x, y;
    DispatchThreadIdToTensorIndices(dispatchThreadID, c, x, y);

    if (c >= O.channels) return;    if (x >= O.width) return;       if (y >= O.height) return;

    for (uint n = 0; n < O.batch; ++n)
    {
        float v =
            X.FastGet(dot(uint4(n, y, x, c), _XStrides)) +
            B.FastGet(dot(uint4(n, y, x, c), _BStrides));
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4,8,8), (4,8,4), (4,4,4))
void KERNEL_FUNC(BroadcastSub)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    TENSOR_TWOINPUTS(X, B, O);
    uint c, x, y;
    DispatchThreadIdToTensorIndices(dispatchThreadID, c, x, y);
    if (c >= O.channels) return;    if (x >= O.width) return;       if (y >= O.height) return;

    for (uint n = 0; n < O.batch; ++n)
    {
        float v =
            X.FastGet(dot(uint4(n, y, x, c), _XStrides)) -
            B.FastGet(dot(uint4(n, y, x, c), _BStrides));
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4,8,8), (4,8,4), (4,4,4))
void KERNEL_FUNC(BroadcastMul)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    TENSOR_TWOINPUTS(X, B, O);
    uint c, x, y;
    DispatchThreadIdToTensorIndices(dispatchThreadID, c, x, y);
    if (c >= O.channels) return;    if (x >= O.width) return;       if (y >= O.height) return;

    for (uint n = 0; n < O.batch; ++n)
    {
        float v =
            X.FastGet(dot(uint4(n, y, x, c), _XStrides)) *
            B.FastGet(dot(uint4(n, y, x, c), _BStrides));
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4,8,8), (4,8,4), (4,4,4))
void KERNEL_FUNC(BroadcastDiv)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    TENSOR_TWOINPUTS(X, B, O);
    uint c, x, y;
    DispatchThreadIdToTensorIndices(dispatchThreadID, c, x, y);
    if (c >= O.channels) return;    if (x >= O.width) return;       if (y >= O.height) return;

    for (uint n = 0; n < O.batch; ++n)
    {
        float v =
            X.FastGet(dot(uint4(n, y, x, c), _XStrides)) /
            B.FastGet(dot(uint4(n, y, x, c), _BStrides));
        O.Set(n, y, x, c, v);
    }
}

float signed_pow(float f, float e)
{
    // handle negative f
    float v = pow(abs(f), e);
    float s = (e % 2 == 1) ?
        sign(f):    // exponent is odd  => sign(f) * pow(abs(f), e)
        1;            // exponent is even => pow(abs(f), e)
    return v * s;
}

NUMTHREADS((4,8,8), (4,8,4), (4,4,4))
void KERNEL_FUNC(BroadcastPow)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    TENSOR_TWOINPUTS(X, B, O);
    uint c, x, y;
    DispatchThreadIdToTensorIndices(dispatchThreadID, c, x, y);
    if (c >= O.channels) return;    if (x >= O.width) return;       if (y >= O.height) return;

    for (uint n = 0; n < O.batch; ++n)
    {
        float v = signed_pow(
            X.FastGet(dot(uint4(n, y, x, c), _XStrides)),
            B.FastGet(dot(uint4(n, y, x, c), _BStrides)));
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4,8,8), (4,8,4), (4,4,4))
void KERNEL_FUNC(BroadcastMin)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    TENSOR_TWOINPUTS(X, B, O);
    uint c, x, y;
    DispatchThreadIdToTensorIndices(dispatchThreadID, c, x, y);
    if (c >= O.channels) return;    if (x >= O.width) return;       if (y >= O.height) return;

    for (uint n = 0; n < O.batch; ++n)
    {
        float v = min(
            X.FastGet(dot(uint4(n, y, x, c), _XStrides)),
            B.FastGet(dot(uint4(n, y, x, c), _BStrides)));
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4,8,8), (4,8,4), (4,4,4))
void KERNEL_FUNC(BroadcastMax)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    TENSOR_TWOINPUTS(X, B, O);
    uint c, x, y;
    DispatchThreadIdToTensorIndices(dispatchThreadID, c, x, y);
    if (c >= O.channels) return;    if (x >= O.width) return;       if (y >= O.height) return;

    for (uint n = 0; n < O.batch; ++n)
    {
        float v = max(
            X.FastGet(dot(uint4(n, y, x, c), _XStrides)),
            B.FastGet(dot(uint4(n, y, x, c), _BStrides)));
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4, 8, 8), (4, 8, 4), (4, 4, 4))
void KERNEL_FUNC(BroadcastMean)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    TENSOR_TWOINPUTS(X, B, O);
    uint c, x, y;
    DispatchThreadIdToTensorIndices(dispatchThreadID, c, x, y);
    if (c >= O.channels) return;    if (x >= O.width) return;       if (y >= O.height) return;

    for (uint n = 0; n < O.batch; ++n)
    {
        float a = X.FastGet(dot(uint4(n, y, x, c), _XStrides));
        a *= _IsFirstDispatch ? _Alpha : 1.0f;
        float b = B.FastGet(dot(uint4(n, y, x, c), _BStrides)) * _Alpha;
        float v = a + b;
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4,8,8), (4,8,4), (4,4,4))
void KERNEL_FUNC(BroadcastGreater)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    TENSOR_TWOINPUTS(X, B, O);
    uint c, x, y;
    DispatchThreadIdToTensorIndices(dispatchThreadID, c, x, y);
    if (c >= O.channels)
        return;
    if (x >= O.width)
        return;
    if (y >= O.height)
        return;

    for (uint n = 0; n < O.batch; ++n)
    {
        float a = X.FastGet(dot(uint4(n, y, x, c), _XStrides));
        float b = B.FastGet(dot(uint4(n, y, x, c), _BStrides));
        float v = (a > b) ? 1.0f : 0.0f;
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4,8,8), (4,8,4), (4,4,4))
void KERNEL_FUNC(BroadcastGreaterEqual)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    TENSOR_TWOINPUTS(X, B, O);
    uint c, x, y;
    DispatchThreadIdToTensorIndices(dispatchThreadID, c, x, y);
    if (c >= O.channels)
        return;
    if (x >= O.width)
        return;
    if (y >= O.height)
        return;

    for (uint n = 0; n < O.batch; ++n)
    {
        float a = X.FastGet(dot(uint4(n, y, x, c), _XStrides));
        float b = B.FastGet(dot(uint4(n, y, x, c), _BStrides));
        float v = (a >= b) ? 1.0f : 0.0f;
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4,8,8), (4,8,4), (4,4,4))
void KERNEL_FUNC(BroadcastLess)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    TENSOR_TWOINPUTS(X, B, O);
    uint c, x, y;
    DispatchThreadIdToTensorIndices(dispatchThreadID, c, x, y);
    if (c >= O.channels)
        return;
    if (x >= O.width)
        return;
    if (y >= O.height)
        return;

    for (uint n = 0; n < O.batch; ++n)
    {
        float a = X.FastGet(dot(uint4(n, y, x, c), _XStrides));
        float b = B.FastGet(dot(uint4(n, y, x, c), _BStrides));
        float v = (a < b) ? 1.0f : 0.0f;
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4,8,8), (4,8,4), (4,4,4))
void KERNEL_FUNC(BroadcastLessEqual)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    TENSOR_TWOINPUTS(X, B, O);
    uint c, x, y;
    DispatchThreadIdToTensorIndices(dispatchThreadID, c, x, y);
    if (c >= O.channels)
        return;
    if (x >= O.width)
        return;
    if (y >= O.height)
        return;

    for (uint n = 0; n < O.batch; ++n)
    {
        float a = X.FastGet(dot(uint4(n, y, x, c), _XStrides));
        float b = B.FastGet(dot(uint4(n, y, x, c), _BStrides));
        float v = (a <= b) ? 1.0f : 0.0f;
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4,8,8), (4,8,4), (4,4,4))
void KERNEL_FUNC(BroadcastEqual)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    TENSOR_TWOINPUTS(X, B, O);
    uint c, x, y;
    DispatchThreadIdToTensorIndices(dispatchThreadID, c, x, y);
    if (c >= O.channels)
        return;
    if (x >= O.width)
        return;
    if (y >= O.height)
        return;

    for (uint n = 0; n < O.batch; ++n)
    {
        float a = X.FastGet(dot(uint4(n, y, x, c), _XStrides));
        float b = B.FastGet(dot(uint4(n, y, x, c), _BStrides));
        float v = (a == b) ? 1.0f : 0.0f;
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4,8,8), (4,8,4), (4,4,4))
void KERNEL_FUNC(BroadcastLogicalOr)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    TENSOR_TWOINPUTS(X, B, O);
    uint c, x, y;
    DispatchThreadIdToTensorIndices(dispatchThreadID, c, x, y);
    if (c >= O.channels)
        return;
    if (x >= O.width)
        return;
    if (y >= O.height)
        return;

    for (uint n = 0; n < O.batch; ++n)
    {
        float a = (X.FastGet(dot(uint4(n, y, x, c), _XStrides)) == 0.0f) ? 0.0f : 1.0f;
        float b = (B.FastGet(dot(uint4(n, y, x, c), _BStrides)) == 0.0f) ? 0.0f : 1.0f;
        float v = a * (1 - b) + b;
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4,8,8), (4,8,4), (4,4,4))
void KERNEL_FUNC(BroadcastLogicalAnd)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    TENSOR_TWOINPUTS(X, B, O);
    uint c, x, y;
    DispatchThreadIdToTensorIndices(dispatchThreadID, c, x, y);
    if (c >= O.channels)
        return;
    if (x >= O.width)
        return;
    if (y >= O.height)
        return;

    for (uint n = 0; n < O.batch; ++n)
    {
        float a = X.FastGet(dot(uint4(n, y, x, c), _XStrides));
        float b = B.FastGet(dot(uint4(n, y, x, c), _BStrides));
        float v = a * b != 0.0 ? 1.0f : 0.0f;
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4,8,8), (4,8,4), (4,4,4))
void KERNEL_FUNC(BroadcastLogicalXor)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    TENSOR_TWOINPUTS(X, B, O);
    uint c, x, y;
    DispatchThreadIdToTensorIndices(dispatchThreadID, c, x, y);
    if (c >= O.channels)
        return;
    if (x >= O.width)
        return;
    if (y >= O.height)
        return;

    for (uint n = 0; n < O.batch; ++n)
    {
        float a = X.FastGet(dot(uint4(n, y, x, c), _XStrides)) != 0.0f ? 1.0f : 0.0f;
        float b = B.FastGet(dot(uint4(n, y, x, c), _BStrides)) != 0.0f ? 1.0f : 0.0f;
        float v = a * (1 - 2 * b) + b;
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4, 8, 8), (4, 8, 4), (4, 4, 4))
void KERNEL_FUNC(BroadcastWhere)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    TENSOR_THREEINPUTS(X, S, B, O);
    uint c, x, y;
    DispatchThreadIdToTensorIndices(dispatchThreadID, c, x, y);
    if (c >= O.channels)
        return;
    if (x >= O.width)
        return;
    if (y >= O.height)
        return;

    for (uint n = 0; n < O.batch; ++n)
    {
        bool cond = (X.FastGet(dot(uint4(n, y, x, c), _XStrides)) != 0.0f);
        float a = S.FastGet(dot(uint4(n, y, x, c), _SStrides));
        float b = B.FastGet(dot(uint4(n, y, x, c), _BStrides));
        float v = cond ? a : b;
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4, 8, 8), (4, 8, 4), (4, 4, 4))
void KERNEL_FUNC(BroadcastDivExpSub)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    TENSOR_THREEINPUTS(X, B, S, O);
    uint c, x, y;
    DispatchThreadIdToTensorIndices(dispatchThreadID, c, x, y);
    if (c >= O.channels) return;    if (x >= O.width) return;       if (y >= O.height) return;

    for (uint n = 0; n < O.batch; ++n)
    {
        float v =
            X.FastGet(dot(uint4(n, y, x, c), _XStrides)) -
            B.FastGet(dot(uint4(n, y, x, c), _BStrides));
        v = exp(v) / S.FastGet(dot(uint4(n, y, x, c), _SStrides));
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4, 8, 8), (4, 8, 4), (4, 4, 4))
void KERNEL_FUNC(LogSoftmaxEnd)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    TENSOR_THREEINPUTS(X, B, S, O);
    uint c, x, y;
    DispatchThreadIdToTensorIndices(dispatchThreadID, c, x, y);
    if (c >= O.channels) return;    if (x >= O.width) return;       if (y >= O.height) return;

    for (uint n = 0; n < O.batch; ++n)
    {
        float v =
            X.FastGet(dot(uint4(n, y, x, c), _XStrides)) -
            B.FastGet(dot(uint4(n, y, x, c), _BStrides));
        v = v - log(S.FastGet(dot(uint4(n, y, x, c), _SStrides)));
        O.Set(n, y, x, c, v);
    }
}
