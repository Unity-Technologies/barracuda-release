#include "DebugUtils.cginc"

#define BARRACUDA_MAX_THREAD_COUNT 64
#if (BARRACUDA_MAX_THREAD_COUNT>=256)
#define NUMTHREADS(t256,t128,t64) [numthreads t256]
#define NUMTHREAD(t256, t128, t64) t256
#elif (BARRACUDA_MAX_THREAD_COUNT>=128)
#define NUMTHREADS(t256,t128,t64) [numthreads t128]
#define NUMTHREAD(t256,t128,t64) t128
#elif (BARRACUDA_MAX_THREAD_COUNT>=64)
#define NUMTHREADS(t256,t128,t64) [numthreads t64]
#define NUMTHREAD(t256,t128,t64) t64
#endif


//Keep in sync with Model.cs enum Layer.FusedActivation
#define ACTIVATION_NONE 0
#define ACTIVATION_RELU 1

int _ActivationMode;
float ApplyFusedActivation(float v)
{
    if (_ActivationMode == ACTIVATION_RELU)
        v = max(v, 0.0f);
    return v;
}

struct Tensor
{
    // @TODO: actually uint seems not like a good idea anymore, consider going to int
    uint batch, height, width, channels;

    void Init(uint4 nhwc)
    {
        batch = nhwc.x;
        height = nhwc.y;
        width = nhwc.z;
        channels = nhwc.w;
    }

    uint4 Dims()
    {
        return uint4(batch, height, width, channels);
    }
    uint GetFlatHeight()
    {
        return batch;
    }
    uint GetFlatWidth()
    {
        return height * width * channels;
    }
    uint GetKernelHeight()
    {
        // kernels storage: {kernel_width * kernel_height * kernel_channels * kernel_count}
        uint kernelHeight = batch;
        return kernelHeight;
    }
    uint GetKernelWidth()
    {
        // kernels storage: {kernel_width * kernel_height * kernel_channels * kernel_count}
        uint kernelWidth = height;
        return kernelWidth;
    }
    uint GetKernelDepth()
    {
        // kernels storage: {kernel_width * kernel_height * kernel_channels * kernel_count}
        uint kernelDepth = width;
        return kernelDepth;
    }
    uint GetKernelCount()
    {
        // kernels storage: {kernel_width * kernel_height * kernel_channels * kernel_count}
        uint kernelCount = channels;
        return kernelCount;
    }
    uint GetLength()
    {
        return batch * height * width * channels;
    }

    uint IndexCHW(uint b, uint h, uint w, uint c)
    {
        uint index =
            b * channels * height * width +
            c * height * width +
            h * width +
            w;
        return index;
    }

    uint IndexCHW(uint b, uint i, uint c)
    {
        uint index =
            b * channels * height * width +
            c * height * width +
            i;
        return index;
    }

    uint IndexHWC(uint b, uint h, uint w, uint c)
    {
        uint index =
            b * height * width * channels +
            h * width * channels +
            w * channels +
            c;
        return index;
    }

    uint IndexHWC(uint b, uint i, uint c)
    {
        uint index =
            b * height * width * channels +
            i * channels +
            c;
        return index;
    }

    uint Index(uint b, uint i)
    {
        uint index =
            b * height * width * channels +
            i;
        return index;
    }

    void GetPositionFromIndexNCHW(uint index, out uint n, out uint h, out uint w, out uint c)
    {
        w = index % width;
        h = (index / width) % height;
        c = (index / (width * height)) % channels;
        n = (index / (width * height * channels)) % batch;
    }

    void GetPositionFromIndexNHWC(uint index, out uint n, out uint h, out uint w, out uint c)
    {
        c = index % channels;
        w = (index / channels) % width;
        h = (index / (channels * width)) % height;
        n = (index / (channels * width * height)) % batch;
    }
};

struct ReadonlyTensor : Tensor
{
    StructuredBuffer<float> data;

    void Init(uint4 nhwc, StructuredBuffer<float> data_)
    {
        Tensor::Init(nhwc);
        data = data_;
    }

    float Get(uint b, uint h, uint w, uint ch)
    {
        #if CHANNELS_FIRST
            uint index = IndexCHW(b,h,w,ch);
        #else
            uint index = IndexHWC(b,h,w,ch);
        #endif
        float value;
        TENSOR_READ(value, index, KERNEL_ASSERT_CONTEXT_READONLY_READ);
        return value;
    }
    float Get(uint b, uint2 pos, uint ch)
    {
        return Get(b, pos.y, pos.x, ch);
    }
    float Get(uint b, uint i, uint ch)
    {
        #if CHANNELS_FIRST
            uint index = IndexCHW(b, i, ch);
        #else
            uint index = IndexHWC(b, i, ch);
        #endif
        float value;
        TENSOR_READ(value, index, KERNEL_ASSERT_CONTEXT_READONLY_READ);
        return value;
    }
    float Get(uint b, uint i)
    {
        uint index = Index(b,i);
        float value;
        TENSOR_READ(value, index, KERNEL_ASSERT_CONTEXT_READONLY_READ);
        return value;
    }
    float FastGet(uint i)
    {
        float value;
        TENSOR_READ(value, i, KERNEL_ASSERT_CONTEXT_READONLY_READ);
        return value;
    }

    float BroadcastGet(uint b, uint h, uint w, uint ch)
    {
        return Get(b % batch, h % height, w % width, ch % channels);
    }
    float BroadcastGet(uint b, uint2 pos, uint ch)
    {
        return BroadcastGet(b, pos.y, pos.x, ch);
    }
    float BroadcastGet(uint b, uint i)
    {
        return Get(b % GetFlatHeight(), i % GetFlatWidth());
    }

    float ClampGet(int b, int2 pos, int ch, int2 pad = int2(0,0))
    {
        b = clamp(b, 0, (int)batch - 1);
        pos = clamp(pos, pad, int2(width, height) + pad - 1);
        ch = clamp(ch, 0, (int)channels - 1);

        pos -= pad;
        return Get(b, pos.y, pos.x, ch);
    }
    float ClampGet(int b, int h, int w, int ch, int2 pad = int2(0,0))
    {
        return ClampGet(b, int2(w, h), ch, pad);
    }
    float ClampGet(int b, int i)
    {
        b = clamp(b, 0, (int)batch - 1);
        i = clamp(i, 0, (int)(height * width * channels) - 1);
        return Get(b,i);
    }
    float ClampGet(int i)
    {
        i = clamp(i, 0, (int)(batch * height * width * channels) - 1);
        return FastGet(i);
    }

    float SafeGetHW(uint b, uint h, uint w, uint c, float def = 0.0f)
    {
        return (h >= height || w >= width) ? def : Get(b, min(h, height-1), min(w, width-1), c);
    }
    float SafeGet(uint b, uint2 pos, uint ch, uint2 pad, float def = 0)
    {
        bool cond =
            (b >= batch || ch >= channels ||
            any(pos < pad) ||
            any(pos >= uint2(width, height) + pad));

        if (cond)
            return def;
        else
            return Get(b, pos - pad, ch);
    }
    float SafeGet(uint b, uint2 pos, uint ch, float def = 0)
    {
        bool cond =
            (b >= batch || ch >= channels ||
            any(pos >= uint2(width, height)));

        if (cond)
            return def;
        else
            return Get(b, pos, ch);
    }
    float SafeGet(uint b, uint h, uint w, uint ch, uint2 pad, float def = 0)
    {
        return SafeGet(b, uint2(w, h), ch, pad, def);
    }
    float SafeGet(uint b, uint h, uint w, uint ch, float def = 0)
    {
        return SafeGet(b, uint2(w, h), ch, def);
    }
    float SafeGet(uint b, uint i, float def = 0)
    {
        if (b >= batch || i >= height * width * channels)
            return def;
        else
            return Get(b,i);
    }
    float SafeGet(uint i, float def = 0)
    {
        if (i >= batch * height * width * channels)
            return def;
        else
            return FastGet(i);
    }

    float MaskedGet(bool cond, uint i, float def = 0)
    {
        if (cond)
            return FastGet(i);
        else
            return def;
    }

    uint GetChannelFromIndex(uint index)
    {
        #if CHANNELS_FIRST
            index /= height*width;
        #endif
        return index % channels;
    }
};

struct ReadWriteTensor : Tensor
{
    RWStructuredBuffer<float> data;

    void Init(int4 nhwc, RWStructuredBuffer<float> data_)
    {
        Tensor::Init(nhwc);
        data = data_;
    }

    float Get(uint b, uint h, uint w, uint ch)
    {
        #if CHANNELS_FIRST
            uint index = IndexCHW(b,h,w,ch);
        #else
            uint index = IndexHWC(b,h,w,ch);
        #endif
        float value;
        TENSOR_READ(value, index, KERNEL_ASSERT_CONTEXT_READWRITE_READ);
        return value;
    }
    float Get(uint b, uint2 pos, uint ch)
    {
        return Get(b, pos.y, pos.x, ch);
    }
    float Get(uint b, uint i)
    {
        uint index = Index(b,i);
        float value;
        TENSOR_READ(value, index, KERNEL_ASSERT_CONTEXT_READWRITE_READ);
        return value;
    }
    float FastGet(uint i)
    {
        float value;
        TENSOR_READ(value, i, KERNEL_ASSERT_CONTEXT_READWRITE_READ);
        return value;
    }

    float BroadcastGet(uint b, uint h, uint w, uint ch)
    {
        return Get(b % batch, h % height, w % width, ch % channels);
    }
    float BroadcastGet(uint b, uint2 pos, uint ch)
    {
        return BroadcastGet(b, pos.y, pos.x, ch);
    }
    float BroadcastGet(uint b, uint i)
    {
        return Get(b % GetFlatHeight(), i % GetFlatWidth());
    }

    float SafeGet(uint b, uint2 pos, uint ch, uint2 pad, float def = 0)
    {
        bool cond =
            (b >= batch || ch >= channels ||
            any(pos < pad) ||
            any(pos >= uint2(width, height) + pad));

        if (cond)
            return def;
        else
            return Get(b, pos - pad, ch);
    }
    float SafeGet(uint b, uint h, uint w, uint ch, uint2 pad, float def = 0)
    {
        return SafeGet(b, uint2(w, h), ch, pad, def);
    }
    float SafeGet(uint b, uint i, float def = 0)
    {
        if (b >= batch || i >= height * width * channels)
            return def;
        else
            return Get(b,i);
    }
    float SafeGet(uint i, float def = 0)
    {
        if (i >= batch * height * width * channels)
            return def;
        else
            return FastGet(i);
    }

    float MaskedGet(bool cond, uint i, float def=0)
    {
        if (cond)
            return FastGet(i);
        else
            return def;
    }

    void Set(uint b, uint h, uint w, uint ch, float v)
    {
        #if CHANNELS_FIRST
            uint index = IndexCHW(b,h,w,ch);
        #else
            uint index = IndexHWC(b,h,w,ch);
        #endif
        TENSOR_WRITE(v, index, KERNEL_ASSERT_CONTEXT_READWRITE_WRITE);
    }
    void Set(uint b, uint2 pos, uint ch, float v)
    {
        Set(b, pos.y, pos.x, ch, v);
    }
    void Set(uint b, uint i, uint ch, float v)
    {
        #if CHANNELS_FIRST
            uint index = IndexCHW(b, i, ch);
        #else
            uint index = IndexHWC(b, i, ch);
        #endif
        TENSOR_WRITE(v, index, KERNEL_ASSERT_CONTEXT_READWRITE_WRITE);
    }
    void Set(uint y, uint x, float v)
    {
        data[Index(y,x)] = v;
    }
    void FastSet(uint i, float v)
    {
        TENSOR_WRITE(v, i, KERNEL_ASSERT_CONTEXT_READWRITE_WRITE);
    }

    void SetWithActivation(uint b, uint h, uint w, uint ch, float v)
    {
        v = ApplyFusedActivation(v);
        Set(b,h,w,ch,v);
    }
    void SetWithActivation(uint b, uint2 pos, uint ch, float v)
    {
        v = ApplyFusedActivation(v);
        Set(b,pos,ch,v);
    }
    void SetWithActivation(uint b, uint i, uint ch, float v)
    {
        v = ApplyFusedActivation(v);
        Set(b,i,ch,v);
    }
    void SetWithActivation(uint y, uint x, float v)
    {
        v = ApplyFusedActivation(v);
        Set(y,x,v);
    }
    void FastSetWithActivation(uint i, float v)
    {
        v = ApplyFusedActivation(v);
        FastSet(i,v);
    }
};

struct SharedTensor : Tensor
{
    StructuredBuffer<float> data;
    uint offset;

    void Init(uint4 nhwc, uint4 info, StructuredBuffer<float> data_)
    {
        Tensor::Init(nhwc);
        data = data_;
        offset = info.x;
    }

    float Get(uint b, uint h, uint w, uint ch)
    {
        uint index = IndexHWC(b,h,w,ch) + offset;
        float value;
        TENSOR_READ(value, index, KERNEL_ASSERT_CONTEXT_SHARED_READ);
        return value;
    }
    float Get(uint b, uint2 pos, uint ch)
    {
        return Get(b, pos.y, pos.x, ch);
    }
    float Get(uint b, uint i)
    {
        uint index = Index(b,i) + offset;
        float value;
        TENSOR_READ(value, index, KERNEL_ASSERT_CONTEXT_SHARED_READ);
        return value;
    }
    float FastGet(uint i)
    {
        float value;
        TENSOR_READ(value, i + offset, KERNEL_ASSERT_CONTEXT_SHARED_READ);
        return value;
    }

    float BroadcastGet(uint b, uint h, uint w, uint ch)
    {
        return Get(b % batch, h % height, w % width, ch % channels);
    }
    float BroadcastGet(uint b, uint2 pos, uint ch)
    {
        return BroadcastGet(b, pos.y, pos.x, ch);
    }
    float BroadcastGet(uint b, uint i)
    {
        return Get(b % GetFlatHeight(), i % GetFlatWidth());
    }
    float FastBroadcastGet(uint i)
    {
        uint index = i % GetFlatWidth() + offset;
        float value;
        TENSOR_READ(value, index, KERNEL_ASSERT_CONTEXT_SHARED_READ);
        return value;
    }

    float SafeGet(uint b, uint2 pos, uint ch, uint2 pad, float def = 0)
    {
        if (b >= batch || ch >= channels ||
            any(pos < pad) ||
            any(pos >= uint2(width, height) + pad))
        {
            return def;
        }
        else
            return Get(b, pos - pad, ch);
    }
    float SafeGet(uint b, uint h, uint w, uint ch, uint2 pad, float def = 0)
    {
        return SafeGet(b, uint2(w, h), ch, pad, def);
    }
    float SafeGet(uint b, uint i, float def = 0)
    {
        if (b >= batch || i >= height * width * channels)
            return def;
        else
            return Get(b,i);
    }
    float SafeGet(uint i, float def = 0)
    {
        if (i >= batch * height * width * channels)
            return def;
        else
            return FastGet(i);
    }

    float MaskedGet(bool cond, uint i, float def=0)
    {
        if (cond)
            return FastGet(i);
        else
            return def;
    }
};

#if CHANNELS_FIRST
    #define KERNEL_FUNC(name) name##_NCHW
#else
    #define KERNEL_FUNC(name) name##_NHWC
#endif

#define TENSOR_DECL(X) uint4 X##declShape; uint4 X##declInfo; StructuredBuffer<float> X##data;
#define TENSOR_DECL_RW(X) uint4 X##declShape; uint4 X##declInfo; RWStructuredBuffer<float> X##data;

// readonly with channel order support (for inputs).
#define TENSOR_ARG(X) ReadonlyTensor X; X.Init(X##declShape, X##data);
// readonly with offset, no channel order support (for weights and biases).
#define TENSOR_MODEL(X) SharedTensor X; X.Init(X##declShape, X##declInfo, X##data);
// read/write with channel order support (for outputs).
#define TENSOR_ARG_RW(X) ReadWriteTensor X; X.Init(X##declShape, X##data);

#define TENSOR_ARGS2(X, O) TENSOR_ARG(X); TENSOR_ARG_RW(O);
#define TENSOR_ARGS3(X, A, O) TENSOR_ARG(X); TENSOR_MODEL(A); TENSOR_ARG_RW(O);
#define TENSOR_TWOINPUTS(X, X1, O) TENSOR_ARG(X); TENSOR_ARG(X1); TENSOR_ARG_RW(O);
#define TENSOR_ARGS4(X, A, B, O) TENSOR_ARG(X); TENSOR_MODEL(A); TENSOR_MODEL(B); TENSOR_ARG_RW(O);

// shared model tensors
#define TENSOR_SHARED_MODEL(X, S) SharedTensor X; X.Init(X##declShape, X##declInfo, S##data);
#define TENSOR_SHARED2_ARGS4(X, A, B, S, O) TENSOR_ARG(X); TENSOR_SHARED_MODEL(A, S); TENSOR_SHARED_MODEL(B, S); TENSOR_ARG_RW(O);


// Purely informational - declares contract between caller of Dispatch() and kernel
// Temporarily disabled due to failure in shader preprocessor in 2020.2
// @TODO: reenable
//#define DISPATCH_ARGS(threadGroupsX, threadGroupsY, threadGroupsZ)


// @TODO: move all code below into a separate and appropriately named file(s)
//
#define FLT_MAX 3.402823466e+38F
#define FLT_EPSILON 1e-6

float fastfma(float a, float b, float c)
{
    return dot(float2(a,c), float2(b, 1));
}

// Neumaier's improved Kahan–Babuška algorithm for compensated summation
// see: https://en.wikipedia.org/wiki/Kahan_summation_algorithm
float neumaierAdd(float sum, float value, inout float floatingPointAccuracyCompensation)
{
    float newSum = sum + value;
    if (abs(sum) >= abs(value))
        floatingPointAccuracyCompensation += (sum - newSum) + value;
    else
        floatingPointAccuracyCompensation += (value - newSum) + sum;
    return newSum;
}
