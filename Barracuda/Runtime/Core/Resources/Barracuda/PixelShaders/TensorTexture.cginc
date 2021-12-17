#pragma multi_compile BATCHTILLING_OFF BATCHTILLING_ON
#define ACTIVATION_NONE 0
#define ACTIVATION_RELU 1

int _ActivationMode;
float4 ApplyFusedActivation(float4 v)
{
	if (_ActivationMode == ACTIVATION_RELU)
	{
		v.x = max(v.x, 0.0f);
		v.y = max(v.y, 0.0f);
		v.z = max(v.z, 0.0f);
		v.w = max(v.w, 0.0f);
	}
	return v;
}

struct Tensor
{
    uint batch;
	uint height;
	uint width;
	uint channels;

#if  BATCHTILLING_ON
    uint batchw;
    uint batchh;
#endif

	uint channels4;
	uint channels4w;
	uint channels4h;

	void Init(uint4 nhwc)
	{
		batch = nhwc.x;
		height = nhwc.y;
		width = nhwc.z;
		channels = nhwc.w;


		channels4 = (channels + 4 - 1) / 4;
		channels4w = channels4;
		channels4h = 1;
		
		if (channels4w * width > 16384)
		{
			channels4w = floor(16384 / ((float)width));
			channels4h = (channels4 + channels4w - 1) / channels4w;
		}

#if  BATCHTILLING_ON
        batchh = batch;
        batchw = 1;

        if (batchh * channels4h * height > 16384)
        {
            batchh = floor(16384 / ((float)(channels4h * height)));
            batchw = (batch + batchh - 1) / batchh;
        }
#endif
	}

	void GetPositionFromUV(float2 uv, out int n, out uint h, out uint w, out uint c4)
	{
#if  BATCHTILLING_ON
		uint2 tid2 = (uint2)(floor(uv * float2(width * channels4w * batchw, channels4h * height * batchh)));
#else
        uint2 tid2 = (uint2)(floor(uv * float2(width * channels4w, channels4h * height * batch)));
#endif
		w = tid2.x % width;
		uint c4w = tid2.x / width;


		h = tid2.y % height;
		uint c4h = (tid2.y / height) % channels4h;

#if  BATCHTILLING_ON
        uint bw = (tid2.x / width) / channels4w;
        uint bh = (tid2.y / height) / channels4h;
		n = bw + batchw * bh;
#else
        n = (tid2.y / height) / channels4h;
#endif

		c4 = c4w + channels4w * c4h;
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
};

struct ReadonlyTensor : Tensor
{
	Texture2D<float4> data;

	void Init(uint4 nhwc, Texture2D<float4> data_)
	{
		Tensor::Init(nhwc);
		data = data_;
	}

	float4 FastGet4(float2 uv)
	{
		return data.Load(uint3(uv.x * channels4w * width, uv.y * channels4h * height * batch, 0));
	}

	float4 Get4(uint n, uint h, uint w, uint c4)
	{
		int c4w = clamp(0, c4 % channels4w, channels4w);
		int c4h = clamp(0, c4 / channels4w, channels4h);

#if  BATCHTILLING_ON
        int bw = clamp(0, n % batchw, batchw);
        int bh = clamp(0, n / batchw, batchh);

		uint2 tid = uint2(bw * width * channels4w + c4w * width + w, bh * channels4h * height + c4h * height + h);
#else
        uint2 tid = uint2(c4w * width + w, n * channels4h * height + c4h * height + h);
#endif

		return data.Load(uint3(tid.x, tid.y, 0));
	}

	float Get(uint n, uint h, uint w, uint c)
	{
        uint c4 = c / 4;
		uint cr4 = c % 4;
		return Get4(n, h, w, c4)[cr4];
	}

	float Get(uint b, uint2 pos, uint c)
	{
		return Get(b, pos.y, pos.x, c);
	}

	float4 Get4(uint b, uint2 pos, uint c)
	{
		return Get4(b, pos.y, pos.x, c);
	}

	float BroadcastGet(uint b, uint h, uint w, uint c)
	{
		return Get(b % batch, h % height, w % width, c % channels);
	}

	float4 BroadcastGet4(uint b, uint h, uint w, uint c4)
	{
        float4 v = Get4(b % batch, h % height, w % width, c4 % channels4);
       // v.x = Get(b % batch, h % height, w % width, (4 * c4 + 0) % channels);
       // v.y = Get(b % batch, h % height, w % width, (4 * c4 + 1) % channels);
       // v.z = Get(b % batch, h % height, w % width, (4 * c4 + 2) % channels);
       // v.w = Get(b % batch, h % height, w % width, (4 * c4 + 3) % channels);
        v[1] = v[((4 * c4 + 1) % channels) % 4];
        v[2] = v[((4 * c4 + 2) % channels) % 4];
        v[3] = v[((4 * c4 + 3) % channels) % 4];

        return v;
	}

	float4 ClampGet4(int b, int2 pos, int ch, int2 pad = int2(0, 0))
	{
		b = clamp(b, 0, (int)batch - 1);
		pos = clamp(pos, pad, int2(width, height) + pad - 1);
		ch = clamp(ch, 0, (int)channels - 1);

		pos -= pad;
		return Get4(b, pos.y, pos.x, ch);
	}

	float4 ClampGet4(int b, int h, int w, int ch, int2 pad = int2(0, 0))
	{
		return ClampGet4(b, int2(w, h), ch, pad);
	}

	float SafeGetHW(uint b, uint h, uint w, uint c, float def = 0.0f)
	{
		return (h >= height || w >= width) ? def : Get(b, min(h, height - 1), min(w, width - 1), c);
	}

	float4 SafeGet4(uint b, uint2 pos, uint c4, uint2 pad, float def = 0)
	{
		if (b >= batch || 
			any(pos < pad) || 
			any(pos >= uint2(width, height) + pad))
			return def;

        float4 v = Get4(b, pos - pad, c4);
        v.x = 4 * c4 + 0 >= channels ? def : v.x;
        v.y = 4 * c4 + 1 >= channels ? def : v.y;
        v.z = 4 * c4 + 2 >= channels ? def : v.z;
        v.w = 4 * c4 + 3 >= channels ? def : v.w;

        return v;
	}

	float SafeGet(uint b, uint2 pos, uint c, uint2 pad, float def = 0)
	{
		uint cr4 = (int)c % 4;
		return SafeGet4(b, pos, c, pad, def = 0)[cr4];
	}
};

#define TENSOR_DECL(X) uint4 X##declShape; Texture2D<float4> X##data;
#define TENSOR_ARG(X) ReadonlyTensor X; X.Init(X##declShape, X##data);

#define TENSOR_DECL_O(X) uint4 X##declShape;
#define TENSOR_O(X) Tensor X; X.Init(X##declShape);

#define TENSOR_ARGS2(X, O) TENSOR_ARG(X); TENSOR_O(O);
#define TENSOR_ARGS3(X, A, O) TENSOR_ARG(X); TENSOR_ARG(A); TENSOR_O(O);
#define TENSOR_ARGS4(X, A, B, O) TENSOR_ARG(X); TENSOR_ARG(A); TENSOR_ARG(B); TENSOR_O(O);

#define FLT_MAX 3.402823466e+38F
#define FLT_EPSILON 1e-6
