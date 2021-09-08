#include "Tensor.cginc"

TENSOR_DECL(X)
TENSOR_DECL(W)
TENSOR_DECL(B)
TENSOR_DECL(WBK)
TENSOR_DECL_RW(O)

uint4 _Pool;
uint4 _Stride;
uint4 _Pad;

NUMTHREADS((4,8,8), (4,8,4), (4,4,4))
void KERNEL_FUNC(MaxPool2D)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.channels, O.width, O.height);
    TENSOR_ARGS2(X, O);

    uint c = dispatchThreadID.x;
    uint x = dispatchThreadID.y;
    uint y = dispatchThreadID.z;

    if (c >= O.channels) return;
    if (x >= O.width) return;
    if (y >= O.height) return;

    for (uint n = 0; n < X.batch; ++n)
    {
        float maxV = -FLT_MAX;
        for (uint dy = 0; dy < _Pool.y; ++dy)
            for (uint dx = 0; dx < _Pool.x; ++dx)
            {
                uint oy = y * _Stride.y + dy;
                uint ox = x * _Stride.x + dx;

                bool mask = (oy >= _Pad.y) && (ox >= _Pad.x) && (oy - _Pad.y < X.height) && (ox - _Pad.x < X.width);
                float v = (mask)? X.Get(n, min(oy - _Pad.y, X.height-1), min(ox - _Pad.x, X.width-1), c): -FLT_MAX;

                maxV = max(v, maxV);
            }

        O.Set(n, y, x, c, maxV);
    }
}

NUMTHREADS((4,8,8), (4,8,4), (4,4,4))
void KERNEL_FUNC(AvgPool2D)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.channels, O.width, O.height);
    TENSOR_ARGS2(X, O);

    uint c = dispatchThreadID.x;
    uint x = dispatchThreadID.y;
    uint y = dispatchThreadID.z;

    if (c >= O.channels) return;
    if (x >= O.width) return;
    if (y >= O.height) return;

    for (uint n = 0; n < X.batch; ++n)
    {
        float acc = 0;
        float counter = 0;
        for (uint dy = 0; dy < _Pool.y; ++dy)
            for (uint dx = 0; dx < _Pool.x; ++dx)
            {
                uint oy = y * _Stride.y + dy;
                uint ox = x * _Stride.x + dx;

                bool mask = (oy >= _Pad.y) && (ox >= _Pad.x) && (oy - _Pad.y < X.height) && (ox - _Pad.x < X.width);
                acc += (mask)? X.Get(n, min(oy - _Pad.y, X.height-1), min(ox - _Pad.x, X.width-1), c): 0;
                counter += (mask)? 1: 0;
            }

        acc /= counter;
        O.Set(n, y, x, c, acc);
    }
}

#undef POOL_SIZE
#define POOL_SIZE 8

groupshared float AvPool2D_PartialSum[POOL_SIZE*POOL_SIZE];

inline void AvgPoolInternalReduce(uint gtz, uint s)
{
    if (gtz < s)
    {
        AvPool2D_PartialSum[gtz] += AvPool2D_PartialSum[gtz + s];
    }
    GroupMemoryBarrierWithGroupSync();
}

[numthreads(1, POOL_SIZE, POOL_SIZE)]
void KERNEL_FUNC(AvgPool2DReduce)(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    //DISPATCH ARGS(O.channels, O.width, O.height);
    TENSOR_ARGS2(X, O);

    uint c = dispatchThreadID.x;

    uint gtx = groupThreadID.y;
    uint gty = groupThreadID.z;

    uint gx = groupId.y;
    uint gy = groupId.z;

    // https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    // half the number of blocks (x and y) replaced with 4 loads
    uint x = gx * POOL_SIZE * 2 + gtx;
    uint y = gy * POOL_SIZE * 2 + gty;

    // 2D -> 1D indexing
    uint gtz = POOL_SIZE * gty + gtx;

    for (uint n = 0; n < X.batch; ++n)
    {
        float v0 = X.SafeGetHW(n, y, x, c);
        float v1 = X.SafeGetHW(n, y + POOL_SIZE, x, c);
        float v2 = X.SafeGetHW(n, y, x + POOL_SIZE, c);
        float v3 = X.SafeGetHW(n, y + POOL_SIZE, x + POOL_SIZE, c);
        AvPool2D_PartialSum[gtz] = v0 + v1 + v2 + v3;

        GroupMemoryBarrierWithGroupSync();

        // sequential addressing
        // mem = [x0...xn y0..yn]
        //     = [x0+y0...xn+yn ...]
        // last sum saved for last
        // following code is unrolled:
        // for s = (POOL_SIZE*POOL_SIZE) / 2; s > 1; s >>= 1
        AvgPoolInternalReduce(gtz, 32);
        AvgPoolInternalReduce(gtz, 16);
        AvgPoolInternalReduce(gtz, 8);
        AvgPoolInternalReduce(gtz, 4);
        AvgPoolInternalReduce(gtz, 2);


        if (gtz == 0)
        {
            float v = AvPool2D_PartialSum[0] + AvPool2D_PartialSum[1];
            O.Set(n, gy, gx, c, v);
        }
    }
}

#undef POOL_SIZE
#define POOL_SIZE 8

groupshared float GlobalAvgPool2D_PartialSum[POOL_SIZE*POOL_SIZE];

inline void GlobalAvgPoolInternalReduce(uint gtz, uint s)
{
    if (gtz < s)
    {
        GlobalAvgPool2D_PartialSum[gtz] += GlobalAvgPool2D_PartialSum[gtz + s];
    }
    GroupMemoryBarrierWithGroupSync();
}

[numthreads(1, POOL_SIZE, POOL_SIZE)]
void KERNEL_FUNC(GlobalAvgPool2D)(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    //DISPATCH ARGS(O.channels, 1, 1);
    TENSOR_ARGS2(X, O);

    uint c = dispatchThreadID.x;

    uint gtx = groupThreadID.y;
    uint gty = groupThreadID.z;

    uint gx = groupId.y;
    uint gy = groupId.z;

    uint x = gx * POOL_SIZE * 2 + gtx;
    uint y = gy * POOL_SIZE * 2 + gty;

    // 2D -> 1D indexing
    uint gtz = POOL_SIZE * gty + gtx;

    for (uint n = 0; n < X.batch; ++n)
    {
        float v0 = X.SafeGetHW(n, y, x, c);
        float v1 = X.SafeGetHW(n, y + POOL_SIZE, x, c);
        float v2 = X.SafeGetHW(n, y, x + POOL_SIZE, c);
        float v3 = X.SafeGetHW(n, y + POOL_SIZE, x + POOL_SIZE, c);
        GlobalAvgPool2D_PartialSum[gtz] = v0 + v1 + v2 + v3;
        GroupMemoryBarrierWithGroupSync();

        // sequential addressing
        // mem = [x0...xn y0..yn]
        //     = [x0+y0...xn+yn ...]
        // last sum saved for last
        // following code is unrolled:
        // for s = (POOL_SIZE*POOL_SIZE) / 2; s > 1; s >>= 1
        GlobalAvgPoolInternalReduce(gtz, 32);
        GlobalAvgPoolInternalReduce(gtz, 16);
        GlobalAvgPoolInternalReduce(gtz, 8);
        GlobalAvgPoolInternalReduce(gtz, 4);
        GlobalAvgPoolInternalReduce(gtz, 2);


        if (gtz == 0)
        {
            float v = GlobalAvgPool2D_PartialSum[0] + GlobalAvgPool2D_PartialSum[1];
            float poolSize = (_Pool[0] * _Pool[1]);
            v /= poolSize;
            O.Set(n, 0, 0, c, v);
        }
    }
}


#undef POOL_SIZE
#define POOL_SIZE 8

groupshared float MaxPool2D_PartialSum[POOL_SIZE*POOL_SIZE];

inline void MaxPoolInternalReduce(uint gtz, uint s)
{
    if (gtz < s)
    {
        MaxPool2D_PartialSum[gtz] = max(MaxPool2D_PartialSum[gtz], MaxPool2D_PartialSum[gtz + s]);
    }
    GroupMemoryBarrierWithGroupSync();
}

[numthreads(1, POOL_SIZE, POOL_SIZE)]
void KERNEL_FUNC(MaxPool2DReduce)(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    //DISPATCH ARGS(O.channels, O.width, O.height);
    TENSOR_ARGS2(X, O);

    uint c = dispatchThreadID.x;

    uint gtx = groupThreadID.y;
    uint gty = groupThreadID.z;

    uint gx = groupId.y;
    uint gy = groupId.z;

    // https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    // half the number of blocks (x and y) replaced with 4 loads
    uint x = gx * POOL_SIZE * 2 + gtx;
    uint y = gy * POOL_SIZE * 2 + gty;

    // 2D -> 1D indexing
    uint gtz = POOL_SIZE * gty + gtx;

    for (uint n = 0; n < X.batch; ++n)
    {
        float v0 = X.SafeGetHW(n, y, x, c, -FLT_MAX);
        float v1 = X.SafeGetHW(n, y + POOL_SIZE, x, c, -FLT_MAX);
        float v2 = X.SafeGetHW(n, y, x + POOL_SIZE, c, -FLT_MAX);
        float v3 = X.SafeGetHW(n, y + POOL_SIZE, x + POOL_SIZE, c, -FLT_MAX);
        MaxPool2D_PartialSum[gtz] = max(max(max(v0, v1), v2), v3);

        GroupMemoryBarrierWithGroupSync();

        // sequential addressing
        // mem = [x0...xn y0..yn]
        //     = [x0+y0...xn+yn ...]
        // last sum saved for last
        // following code is unrolled:
        // for s = (POOL_SIZE*POOL_SIZE) / 2; s > 1; s >>= 1
        MaxPoolInternalReduce(gtz, 32);
        MaxPoolInternalReduce(gtz, 16);
        MaxPoolInternalReduce(gtz, 8);
        MaxPoolInternalReduce(gtz, 4);
        MaxPoolInternalReduce(gtz, 2);

        if (gtz == 0)
        {
            float v = max(MaxPool2D_PartialSum[0], MaxPool2D_PartialSum[1]);
            O.Set(n, gy, gx, c, v);
        }
    }
}

#undef POOL_SIZE
#define POOL_SIZE 8

groupshared float GlobalMaxPool2D_PartialMax[POOL_SIZE*POOL_SIZE];

inline void GlobalMaxPoolInternalReduce(uint gtz, uint s)
{
    if (gtz < s)
    {
        GlobalMaxPool2D_PartialMax[gtz] = max(GlobalMaxPool2D_PartialMax[gtz], GlobalMaxPool2D_PartialMax[gtz + s]);
    }
    GroupMemoryBarrierWithGroupSync();
}

[numthreads(1, POOL_SIZE, POOL_SIZE)]
void KERNEL_FUNC(GlobalMaxPool2D)(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    //DISPATCH ARGS(O.channels, 1, 1);
    TENSOR_ARGS2(X, O);

    uint c = dispatchThreadID.x;

    uint gtx = groupThreadID.y;
    uint gty = groupThreadID.z;

    uint gx = groupId.y;
    uint gy = groupId.z;

    uint x = gx * POOL_SIZE * 2 + gtx;
    uint y = gy * POOL_SIZE * 2 + gty;

    // 2D -> 1D indexing
    uint gtz = POOL_SIZE * gty + gtx;

    for (uint n = 0; n < X.batch; ++n)
    {
        float v0 = X.SafeGetHW(n, y, x, c, -FLT_MAX);
        float v1 = X.SafeGetHW(n, y + POOL_SIZE, x, c, -FLT_MAX);
        float v2 = X.SafeGetHW(n, y, x + POOL_SIZE, c, -FLT_MAX);
        float v3 = X.SafeGetHW(n, y + POOL_SIZE, x + POOL_SIZE, c, -FLT_MAX);
        GlobalMaxPool2D_PartialMax[gtz] = max(max(max(v0, v1), v2), v3);
        GroupMemoryBarrierWithGroupSync();

        // sequential addressing
        // mem = [x0...xn y0..yn]
        //     = [x0+y0...xn+yn ...]
        // last sum saved for last
        // following code is unrolled:
        // for s = (POOL_SIZE*POOL_SIZE) / 2; s > 1; s >>= 1
        GlobalMaxPoolInternalReduce(gtz, 32);
        GlobalMaxPoolInternalReduce(gtz, 16);
        GlobalMaxPoolInternalReduce(gtz, 8);
        GlobalMaxPoolInternalReduce(gtz, 4);
        GlobalMaxPoolInternalReduce(gtz, 2);

        if (gtz == 0)
        {
            float maxV = max(GlobalMaxPool2D_PartialMax[0], GlobalMaxPool2D_PartialMax[1]);
            O.Set(n, 0, 0, c, maxV);
        }
    }
}

int _IsFirstDispatch;

#undef POOL_SIZE
#define POOL_SIZE 8

TENSOR_DECL(X2)
TENSOR_DECL_RW(O2)

groupshared float AvgVariancePool2D_PartialSum[POOL_SIZE*POOL_SIZE];
groupshared float AvgVariancePool2D_PartialSumSq[POOL_SIZE*POOL_SIZE];

inline void AvgVarianceInternalReduce(uint gtz, uint s)
{
    if (gtz < s)
    {
        AvgVariancePool2D_PartialSum[gtz]  += AvgVariancePool2D_PartialSum[gtz + s];
        AvgVariancePool2D_PartialSumSq[gtz] += AvgVariancePool2D_PartialSumSq[gtz + s];
    }
    GroupMemoryBarrierWithGroupSync();
}


[numthreads(1, POOL_SIZE, POOL_SIZE)]
void KERNEL_FUNC(AvgVariancePool2DReduce)(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    //DISPATCH ARGS(O.channels, O.width, O.height);
    TENSOR_ARG(X); TENSOR_ARG(X2); TENSOR_ARG_RW(O); TENSOR_ARG_RW(O2);

    uint c = dispatchThreadID.x;

    uint gtx = groupThreadID.y;
    uint gty = groupThreadID.z;

    uint gx = groupId.y;
    uint gy = groupId.z;

    // https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    // half the number of blocks (x and y) replaced with 4 loads
    uint x = gx * POOL_SIZE * 2 + gtx;
    uint y = gy * POOL_SIZE * 2 + gty;

    // 2D -> 1D indexing
    uint gtz = POOL_SIZE * gty + gtx;

    for (uint n = 0; n < X.batch; ++n)
    {
        float v0 = X.SafeGetHW(n, y, x, c);
        float v1 = X.SafeGetHW(n, y + POOL_SIZE, x, c);
        float v2 = X.SafeGetHW(n, y, x + POOL_SIZE, c);
        float v3 = X.SafeGetHW(n, y + POOL_SIZE, x + POOL_SIZE, c);
        AvgVariancePool2D_PartialSum[gtz] = v0 + v1 + v2 + v3;

        float w0 = X2.SafeGetHW(n, y, x, c);
        float w1 = X2.SafeGetHW(n, y + POOL_SIZE, x, c);
        float w2 = X2.SafeGetHW(n, y, x + POOL_SIZE, c);
        float w3 = X2.SafeGetHW(n, y + POOL_SIZE, x + POOL_SIZE, c);
        if (_IsFirstDispatch)
        {
            // to avoid X^2 dispatch, first call squares X inplace
            AvgVariancePool2D_PartialSumSq[gtz]  = w0 * w0 + w1 * w1 + w2 * w2 + w3 * w3;
        }
        else
        {
            AvgVariancePool2D_PartialSumSq[gtz] = w0 + w1 + w2 + w3;
        }
        GroupMemoryBarrierWithGroupSync();

        // sequential addressing
        // mem = [x0...xn y0..yn]
        //     = [x0+y0...xn+yn ...]
        // last sum saved for last
        // following code is unrolled:
        // for s = (POOL_SIZE*POOL_SIZE) / 2; s > 1; s >>= 1
        AvgVarianceInternalReduce(gtz, 32);
        AvgVarianceInternalReduce(gtz, 16);
        AvgVarianceInternalReduce(gtz, 8);
        AvgVarianceInternalReduce(gtz, 4);
        AvgVarianceInternalReduce(gtz, 2);


        if (gtz == 0)
        {
            float v  = AvgVariancePool2D_PartialSum[0]  + AvgVariancePool2D_PartialSum[1];
            float v2 = AvgVariancePool2D_PartialSumSq[0] + AvgVariancePool2D_PartialSumSq[1];

            O.Set(n, gy, gx, c, v);
            O2.Set(n, gy, gx, c, v2);
        }
    }
}

#undef POOL_SIZE
#define POOL_SIZE 8

groupshared float GlobalAvgVariancePool2D_PartialSum[POOL_SIZE*POOL_SIZE];
groupshared float GlobalAvgVariancePool2D_PartialSumSq[POOL_SIZE*POOL_SIZE];

inline void GlobalAvgVarianceInternalReduce(uint gtz, uint s)
{
    if (gtz < s)
    {
        GlobalAvgVariancePool2D_PartialSum[gtz] += GlobalAvgVariancePool2D_PartialSum[gtz + s];
        GlobalAvgVariancePool2D_PartialSumSq[gtz] += GlobalAvgVariancePool2D_PartialSumSq[gtz + s];
    }
    GroupMemoryBarrierWithGroupSync();
}

[numthreads(1, POOL_SIZE, POOL_SIZE)]
void KERNEL_FUNC(GlobalAvgVariancePool2D)(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    //DISPATCH ARGS(O.channels, 1, 1);
    TENSOR_TWOINPUTS(X, X2, O);

    uint c = dispatchThreadID.x;

    uint gtx = groupThreadID.y;
    uint gty = groupThreadID.z;

    uint gx = groupId.y;
    uint gy = groupId.z;

    uint x = gx * POOL_SIZE * 2 + gtx;
    uint y = gy * POOL_SIZE * 2 + gty;

    // 2D -> 1D indexing
    uint gtz = POOL_SIZE * gty + gtx;

    for (uint n = 0; n < X.batch; ++n)
    {
        float v0 = X.SafeGetHW(n, y, x, c);
        float v1 = X.SafeGetHW(n, y + POOL_SIZE, x, c);
        float v2 = X.SafeGetHW(n, y, x + POOL_SIZE, c);
        float v3 = X.SafeGetHW(n, y + POOL_SIZE, x + POOL_SIZE, c);
        GlobalAvgVariancePool2D_PartialSum[gtz] = v0 + v1 + v2 + v3;

        float w0 = X2.SafeGetHW(n, y, x, c);
        float w1 = X2.SafeGetHW(n, y + POOL_SIZE, x, c);
        float w2 = X2.SafeGetHW(n, y, x + POOL_SIZE, c);
        float w3 = X2.SafeGetHW(n, y + POOL_SIZE, x + POOL_SIZE, c);
        if (_IsFirstDispatch)
        {
            // to avoid X^2 dispatch, first call squares X inplace
            GlobalAvgVariancePool2D_PartialSumSq[gtz] = w0 * w0 + w1 * w1 + w2 * w2 + w3 * w3;
        }
        else
        {
            GlobalAvgVariancePool2D_PartialSumSq[gtz]  = w0 + w1 + w2 + w3;
        }
        GroupMemoryBarrierWithGroupSync();

        // sequential addressing
        // mem = [x0...xn y0..yn]
        //     = [x0+y0...xn+yn ...]
        // last sum saved for last
        // following code is unrolled:
        // for s = (POOL_SIZE*POOL_SIZE) / 2; s > 1; s >>= 1
        GlobalAvgVarianceInternalReduce(gtz, 32);
        GlobalAvgVarianceInternalReduce(gtz, 16);
        GlobalAvgVarianceInternalReduce(gtz, 8);
        GlobalAvgVarianceInternalReduce(gtz, 4);
        GlobalAvgVarianceInternalReduce(gtz, 2);


        if (gtz == 0)
        {
            float v = GlobalAvgVariancePool2D_PartialSum[0] + GlobalAvgVariancePool2D_PartialSum[1];
            float v2 = GlobalAvgVariancePool2D_PartialSumSq[0] + GlobalAvgVariancePool2D_PartialSumSq[1];

            float poolSize = (_Pool[0] * _Pool[1]);
            v /= poolSize;
            v2 /= poolSize;

            float mean = v;
            float variance = v2 - mean * mean;
            O.Set(n, 0, 0, c, mean);
            O.Set(n, 1, 0, c, variance);
        }
    }
}
