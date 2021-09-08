#include "Tensor.cginc"

TENSOR_DECL(X)
TENSOR_DECL_RW(O)

float _Alpha;
float _Beta;
uint _LoopStride;

//DISPATCH ARGS(O.length, 1, 1);
#define FLAT_ACTIVATION(name, op_name) \
void name##_Flat(uint3 dispatchThreadID : SV_DispatchThreadID)\
{\
    TENSOR_ARGS2(X, O);\
\
    uint i = dispatchThreadID.x;\
    if (i >= O.GetLength()) return;\
\
    float v = X.FastGet(i);\
    v = op_name (v);\
    O.FastSet(i, v);\
}

//DISPATCH ARGS(O.length/2, 1, 1)
#define FLAT_ACTIVATION_STRICT(name, op_name) \
void name##_FlatStrict(uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadID)\
{\
    TENSOR_ARGS2(X, O);\
\
    uint numThreadsPerTG = NUMTHREAD(512, 128, 64);\
    uint i1 = (groupId.x * 2 + 0) * numThreadsPerTG + groupThreadId.x;\
	uint i2 = (groupId.x * 2 + 1) * numThreadsPerTG + groupThreadId.x;\
    float v1 = X.FastGet(i1);\
	float v2 = X.FastGet(i2);\
    v1 = op_name (v1);\
	v2 = op_name (v2);\
    O.FastSet(i1, v1);\
	O.FastSet(i2, v2);\
}

//DISPATCH ARGS(O.length, 1, 1);
#define LOOP_ACTIVATION(name, op_name) \
void name##_Loop(uint3 dispatchThreadID : SV_DispatchThreadID)\
{\
    TENSOR_ARGS2(X, O);\
\
    uint i = dispatchThreadID.x;\
    uint len = O.GetLength();\
\
    while (i < len) {\
        float v = X.FastGet(i); \
        v = op_name (v); \
        O.FastSet(i, v); \
        i += _LoopStride; \
    }\
}

#define ACTIVATION(name, op_name) \
NUMTHREADS((512,1,1), (128,1,1), (64,1,1))\
FLAT_ACTIVATION(name, op_name)\
NUMTHREADS((512,1,1), (128,1,1), (64,1,1))\
FLAT_ACTIVATION_STRICT(name, op_name)\
NUMTHREADS((512,1,1), (128,1,1), (64,1,1))\
LOOP_ACTIVATION(name, op_name)

float relu(float v)
{
    return max(v, 0.0f);
}

float relu6(float v)
{
    return min(max(v, 0.0f), 6.0f);
}

float swish(float v)
{
    return v / (1.f + exp(-v));
}

float prelu(float v, float alpha)
{
	return max(v, 0.0f) + alpha * min(v, 0.0f);
}

float selu(float v)
{
	return _Beta * (max(v, 0.0f) + min(_Alpha * (exp(v) - 1.0f), 0.0f));
}

float softplus(float v)
{
    return log(exp(v) + 1.f);
}

float sigmoid(float v)
{
    return rcp(1.f + exp(-v));
}

float hardsigmoid(float v)
{
    return max(0.0f, min(1.0f, _Alpha * v + _Beta));
}

float elu(float v)
{
    return (v <= 0.f) ? _Alpha * (exp(v) - 1.f) : v;
}

float lrelu(float v)
{
    return max(v, _Alpha * v);
}

float signed_pow(float f)
{
	return pow(abs(f), _Alpha);
}

float logical_not(float v)
{
    return (v == 0.0f) ? 1.0f : 0.0f;
}

float neg(float v)
{
    return -v;
}

float tanh_safe(float x)
{
    return tanh(clamp(x,-16.0f,16.0f));//clamp to avoid NaNs for large values.
}

float activation_clip(float v)
{
	return clamp(v, _Alpha, _Beta);
}

float acosh(float v)
{
    return log(v + sqrt(v*v - 1.0f));
}

float asinh(float v)
{
    return log(v + sqrt(v*v + 1.0f));
}

float atanh(float v)
{
    return 0.5f * log((1.0f + v) / (1.0f - v));
}

float erf(float v)
{
    // Abramowitz/Stegun approximations
    // erf(x) = -erf(-x)
    float x = abs(v);

    float p = 0.3275911f;
    float a1 = 0.254829592f; float a2 = -0.284496736f; float a3 = 1.421413741f;
    float a4 = -1.453152027f; float a5 = 1.061405429f;

    float t = 1.0f / (1.0f + p * x);
    float t2 = t * t;
    float t3 = t2 * t;
    float t4 = t3 * t;
    float t5 = t4 * t;

    return sign(v)*(1 - (a1*t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5)*exp(-x * x));
}


ACTIVATION(Abs, abs)
ACTIVATION(Neg, neg)
ACTIVATION(Ceil, ceil)
ACTIVATION(Floor, floor)
ACTIVATION(Round, round)
ACTIVATION(Reciprocal, rcp)
ACTIVATION(Relu, relu)
ACTIVATION(Relu6, relu6)
ACTIVATION(Tanh, tanh_safe)
ACTIVATION(Softplus, softplus)
ACTIVATION(Sigmoid, sigmoid)
ACTIVATION(HardSigmoid, hardsigmoid)
ACTIVATION(Swish, swish)
ACTIVATION(Elu, elu)
ACTIVATION(Selu, selu)
ACTIVATION(LeakyRelu, lrelu)
ACTIVATION(Exp, exp)
ACTIVATION(Log, log)
ACTIVATION(Sqrt, sqrt)
ACTIVATION(Pow, signed_pow)
ACTIVATION(LogicalNot, logical_not)
ACTIVATION(Sign, sign)
ACTIVATION(Clip, activation_clip)
ACTIVATION(Acos, acos)
ACTIVATION(Acosh, acosh)
ACTIVATION(Asin, asin)
ACTIVATION(Asinh, asinh)
ACTIVATION(Atan, atan)
ACTIVATION(Atanh, atanh)
ACTIVATION(Cos, cos)
ACTIVATION(Cosh, cosh)
ACTIVATION(Sin, sin)
ACTIVATION(Sinh, sinh)
ACTIVATION(Tan, tan)
ACTIVATION(Erf, erf)

// -------------------

NUMTHREADS((4,8,8), (4,8,4), (4,4,4))
void KERNEL_FUNC(Relu)(uint3 dispatchThreadID : SV_DispatchThreadID)
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
        float v = X.Get(n, y, x, c);
        v = relu(v);
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4,8,8), (4,8,4), (4,4,4))
void KERNEL_FUNC(Relu6)(uint3 dispatchThreadID : SV_DispatchThreadID)
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
        float v = X.Get(n, y, x, c);
        v = relu6(v);
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4, 8, 8), (4, 8, 4), (4, 4, 4))
void KERNEL_FUNC(Selu)(uint3 dispatchThreadID : SV_DispatchThreadID)
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
		float v = X.Get(n, y, x, c);
		v = selu(v);
		O.Set(n, y, x, c, v);
	}
}

NUMTHREADS((4,8,8), (4,8,4), (4,4,4))
void KERNEL_FUNC(Tanh)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.channels, O.width, O.height);
    TENSOR_ARGS2(X, O);

    uint c = dispatchThreadID.x;    uint x = dispatchThreadID.y;    uint y = dispatchThreadID.z;
    if (c >= O.channels) return;    if (x >= O.width) return;        if (y >= O.height) return;

    for (uint n = 0; n < X.batch; ++n)
    {
        float v = X.Get(n, y, x, c);
        v = tanh_safe(v);
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4, 8, 8), (4, 8, 4), (4, 4, 4))
void KERNEL_FUNC(Softplus)(uint3 dispatchThreadID : SV_DispatchThreadID)
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
        float v = X.Get(n, y, x, c);
        v = softplus(v);
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4,8,8), (4,8,4), (4,4,4))
 void KERNEL_FUNC(Sigmoid)(uint3 dispatchThreadID : SV_DispatchThreadID)
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
         float v = X.Get(n, y, x, c);
         v = sigmoid(v);
         O.Set(n, y, x, c, v);
     }
 }

NUMTHREADS((4, 8, 8), (4, 8, 4), (4, 4, 4))
void KERNEL_FUNC(HardSigmoid)(uint3 dispatchThreadID : SV_DispatchThreadID)
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
        float v = X.Get(n, y, x, c);
        v = hardsigmoid(v);
        O.Set(n, y, x, c, v);
    }
}

 NUMTHREADS((4,8,8), (4,8,4), (4,4,4))
void KERNEL_FUNC(Swish)(uint3 dispatchThreadID : SV_DispatchThreadID)
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
        float v = X.Get(n, y, x, c);
        v = swish(v);
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4,8,8), (4,8,4), (4,4,4))
void KERNEL_FUNC(Elu)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.channels, O.width, O.height);
    TENSOR_ARGS2(X, O);

    uint c = dispatchThreadID.x;    uint x = dispatchThreadID.y;    uint y = dispatchThreadID.z;
    if (c >= O.channels) return;    if (x >= O.width) return;        if (y >= O.height) return;

    for (uint n = 0; n < X.batch; ++n)
    {
        float v = X.Get(n, y, x, c);
        v = elu(v);
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4,8,8), (4,8,4), (4,4,4))
void KERNEL_FUNC(LeakyRelu)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.channels, O.width, O.height);
    TENSOR_ARGS2(X, O);

    uint c = dispatchThreadID.x;    uint x = dispatchThreadID.y;    uint y = dispatchThreadID.z;
    if (c >= O.channels) return;    if (x >= O.width) return;        if (y >= O.height) return;

    for (uint n = 0; n < X.batch; ++n)
    {
        float v = X.Get(n, y, x, c);
        v = lrelu(v);
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4,8,8), (4,8,4), (4,4,4))
void KERNEL_FUNC(Exp)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.channels, O.width, O.height);
    TENSOR_ARGS2(X, O);

    uint c = dispatchThreadID.x;    uint x = dispatchThreadID.y;    uint y = dispatchThreadID.z;
    if (c >= O.channels) return;    if (x >= O.width) return;        if (y >= O.height) return;

    for (uint n = 0; n < X.batch; ++n)
    {
        float v = X.Get(n, y, x, c);
        v = exp(v);
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4,8,8), (4,8,4), (4,4,4))
void KERNEL_FUNC(Log)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.channels, O.width, O.height);
    TENSOR_ARGS2(X, O);

    uint c = dispatchThreadID.x;    uint x = dispatchThreadID.y;    uint y = dispatchThreadID.z;
    if (c >= O.channels) return;    if (x >= O.width) return;        if (y >= O.height) return;

    for (uint n = 0; n < X.batch; ++n)
    {
        float v = X.Get(n, y, x, c);
        v = log(v);
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4, 8, 8), (4, 8, 4), (4, 4, 4))
void KERNEL_FUNC(Sqrt)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	//DISPATCH ARGS(O.channels, O.width, O.height);
	TENSOR_ARGS2(X, O);

	uint c = dispatchThreadID.x;    uint x = dispatchThreadID.y;    uint y = dispatchThreadID.z;
	if (c >= O.channels) return;    if (x >= O.width) return;        if (y >= O.height) return;

	for (uint n = 0; n < X.batch; ++n)
	{
		float v = X.Get(n, y, x, c);
		v = sqrt(v);
		O.Set(n, y, x, c, v);
	}
}

NUMTHREADS((4,8,8), (4,8,4), (4,4,4))
void KERNEL_FUNC(Pow)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.channels, O.width, O.height);
    TENSOR_ARGS2(X, O);

    uint c = dispatchThreadID.x;    uint x = dispatchThreadID.y;    uint y = dispatchThreadID.z;
    if (c >= O.channels) return;    if (x >= O.width) return;        if (y >= O.height) return;

    for (uint n = 0; n < X.batch; ++n)
    {
        float v = X.Get(n, y, x, c);
        v = signed_pow(v);
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4, 8, 8), (4, 8, 4), (4, 4, 4))
void KERNEL_FUNC(Clip)(uint3 dispatchThreadID : SV_DispatchThreadID)
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
		float v = X.Get(n, y, x, c);
		v = activation_clip(v);
		O.Set(n, y, x, c, v);
	}
}

NUMTHREADS((4, 8, 8), (4, 8, 4), (4, 4, 4))
void KERNEL_FUNC(Acos)(uint3 dispatchThreadID : SV_DispatchThreadID)
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
        float v = X.Get(n, y, x, c);
        v = acos(v);
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4, 8, 8), (4, 8, 4), (4, 4, 4))
void KERNEL_FUNC(Acosh)(uint3 dispatchThreadID : SV_DispatchThreadID)
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
        float v = X.Get(n, y, x, c);
        v = log(v + sqrt(v * v - 1.0f));
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4, 8, 8), (4, 8, 4), (4, 4, 4))
void KERNEL_FUNC(Asin)(uint3 dispatchThreadID : SV_DispatchThreadID)
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
        float v = X.Get(n, y, x, c);
        v = asin(v);
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4, 8, 8), (4, 8, 4), (4, 4, 4))
void KERNEL_FUNC(Asinh)(uint3 dispatchThreadID : SV_DispatchThreadID)
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
        float v = X.Get(n, y, x, c);
        v = log(v + sqrt(v*v + 1.0f));
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4, 8, 8), (4, 8, 4), (4, 4, 4))
void KERNEL_FUNC(Atan)(uint3 dispatchThreadID : SV_DispatchThreadID)
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
        float v = X.Get(n, y, x, c);
        v = atan(v);
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4, 8, 8), (4, 8, 4), (4, 4, 4))
void KERNEL_FUNC(Atanh)(uint3 dispatchThreadID : SV_DispatchThreadID)
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
        float v = X.Get(n, y, x, c);
        v = 0.5f * log((1.0f + v) / (1.0f - v));
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4, 8, 8), (4, 8, 4), (4, 4, 4))
void KERNEL_FUNC(Cos)(uint3 dispatchThreadID : SV_DispatchThreadID)
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
        float v = X.Get(n, y, x, c);
        v = cos(v);
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4, 8, 8), (4, 8, 4), (4, 4, 4))
void KERNEL_FUNC(Cosh)(uint3 dispatchThreadID : SV_DispatchThreadID)
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
        float v = X.Get(n, y, x, c);
        v = 0.5f * (exp(v) + exp(-v));
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4, 8, 8), (4, 8, 4), (4, 4, 4))
void KERNEL_FUNC(Sin)(uint3 dispatchThreadID : SV_DispatchThreadID)
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
        float v = X.Get(n, y, x, c);
        v = sin(v);
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4, 8, 8), (4, 8, 4), (4, 4, 4))
void KERNEL_FUNC(Sinh)(uint3 dispatchThreadID : SV_DispatchThreadID)
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
        float v = X.Get(n, y, x, c);
        v = 0.5f * (exp(v) - exp(-v));
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4, 8, 8), (4, 8, 4), (4, 4, 4))
void KERNEL_FUNC(Tan)(uint3 dispatchThreadID : SV_DispatchThreadID)
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
        float v = X.Get(n, y, x, c);
        v = tan(v);
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((4, 8, 8), (4, 8, 4), (4, 4, 4))
void KERNEL_FUNC(Erf)(uint3 dispatchThreadID : SV_DispatchThreadID)
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
        float v = X.Get(n, y, x, c);
        v = erf(x);
        O.Set(n, y, x, c, v);
    }
}

NUMTHREADS((16,16,1), (16,8,1), (16,4,1))
void KERNEL_FUNC(Relu_CNyx)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.channels, O.batch * O.height * O.width, 1);
    TENSOR_ARGS2(X, O);

    uint c = dispatchThreadID.x;
    uint nyx = dispatchThreadID.y;

    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (c >= X.channels) return;
    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = relu(v);
    O.Set(n, y, x, c, v);
}

NUMTHREADS((512,1,1), (128,1,1), (64,1,1))
void KERNEL_FUNC(Relu_Nyxc)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.batch * O.height * O.width * O.channels, 1, 1);
    TENSOR_ARGS2(X, O);

    uint nyxc = dispatchThreadID.x;

    uint c = nyxc % X.channels;
    uint nyx = nyxc / X.channels;
    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = relu(v);
    O.Set(n, y, x, c, v);
}


NUMTHREADS((16,16,1), (16,8,1), (16,4,1))
void KERNEL_FUNC(Relu6_CNyx)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.channels, O.batch * O.height * O.width, 1);
    TENSOR_ARGS2(X, O);

    uint c = dispatchThreadID.x;
    uint nyx = dispatchThreadID.y;

    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (c >= X.channels) return;
    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = relu6(v);
    O.Set(n, y, x, c, v);
}

NUMTHREADS((512,1,1), (128,1,1), (64,1,1))
void KERNEL_FUNC(Relu6_Nyxc)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.batch * O.height * O.width * O.channels, 1, 1);
    TENSOR_ARGS2(X, O);

    uint nyxc = dispatchThreadID.x;

    uint c = nyxc % X.channels;
    uint nyx = nyxc / X.channels;
    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = relu6(v);
    O.Set(n, y, x, c, v);
}

NUMTHREADS((16, 16, 1), (16, 8, 1), (16, 4, 1))
void KERNEL_FUNC(Selu_CNyx)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	//DISPATCH ARGS(O.channels, O.batch * O.height * O.width, 1);
	TENSOR_ARGS2(X, O);

	uint c = dispatchThreadID.x;
	uint nyx = dispatchThreadID.y;

	uint x = nyx % X.width;
	uint ny = nyx / X.width;
	uint y = ny % X.height;
	uint n = ny / X.height;

	if (c >= X.channels) return;
	if (n >= X.batch) return;

	float v = X.Get(n, y, x, c);
	v = selu(v);
	O.Set(n, y, x, c, v);
}

NUMTHREADS((512, 1, 1), (128, 1, 1), (64, 1, 1))
void KERNEL_FUNC(Selu_Nyxc)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	//DISPATCH ARGS(O.batch * O.height * O.width * O.channels, 1, 1);
	TENSOR_ARGS2(X, O);

	uint nyxc = dispatchThreadID.x;

	uint c = nyxc % X.channels;
	uint nyx = nyxc / X.channels;
	uint x = nyx % X.width;
	uint ny = nyx / X.width;
	uint y = ny % X.height;
	uint n = ny / X.height;

	if (n >= X.batch) return;

	float v = X.Get(n, y, x, c);
	v = selu(v);
	O.Set(n, y, x, c, v);
}

NUMTHREADS((16,16,1), (16,8,1), (16,4,1))
void KERNEL_FUNC(Tanh_CNyx)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.channels, O.batch * O.height * O.width, 1);
    TENSOR_ARGS2(X, O);

    uint c = dispatchThreadID.x;
    uint nyx = dispatchThreadID.y;

    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (c >= X.channels) return;
    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = tanh_safe(v);
    O.Set(n, y, x, c, v);
}

NUMTHREADS((512,1,1), (128,1,1), (64,1,1))
void KERNEL_FUNC(Tanh_Nyxc)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.batch * O.height * O.width * O.channels, 1, 1);
    TENSOR_ARGS2(X, O);

    uint nyxc = dispatchThreadID.x;

    uint c = nyxc % X.channels;
    uint nyx = nyxc / X.channels;
    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = tanh_safe(v);
    O.Set(n, y, x, c, v);
}

NUMTHREADS((16, 16, 1), (16, 8, 1), (16, 4, 1))
void KERNEL_FUNC(Erf_CNyx)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.channels, O.batch * O.height * O.width, 1);
    TENSOR_ARGS2(X, O);

    uint c = dispatchThreadID.x;
    uint nyx = dispatchThreadID.y;

    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (c >= X.channels) return;
    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = erf(v);
    O.Set(n, y, x, c, v);
}

NUMTHREADS((512, 1, 1), (128, 1, 1), (64, 1, 1))
void KERNEL_FUNC(Erf_Nyxc)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.batch * O.height * O.width * O.channels, 1, 1);
    TENSOR_ARGS2(X, O);

    uint nyxc = dispatchThreadID.x;

    uint c = nyxc % X.channels;
    uint nyx = nyxc / X.channels;
    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = erf(v);
    O.Set(n, y, x, c, v);
}

NUMTHREADS((16, 16, 1), (16, 8, 1), (16, 4, 1))
void KERNEL_FUNC(Softplus_CNyx)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.channels, O.batch * O.height * O.width, 1);
    TENSOR_ARGS2(X, O);

    uint c = dispatchThreadID.x;
    uint nyx = dispatchThreadID.y;

    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (c >= X.channels) return;
    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = softplus(v);
    O.Set(n, y, x, c, v);
}

NUMTHREADS((512, 1, 1), (128, 1, 1), (64, 1, 1))
void KERNEL_FUNC(Softplus_Nyxc)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.batch * O.height * O.width * O.channels, 1, 1);
    TENSOR_ARGS2(X, O);

    uint nyxc = dispatchThreadID.x;

    uint c = nyxc % X.channels;
    uint nyx = nyxc / X.channels;
    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = softplus(v);
    O.Set(n, y, x, c, v);
}

NUMTHREADS((16,16,1), (16,8,1), (16,4,1))
void KERNEL_FUNC(Sigmoid_CNyx)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.channels, O.batch * O.height * O.width, 1);
    TENSOR_ARGS2(X, O);

    uint c = dispatchThreadID.x;
    uint nyx = dispatchThreadID.y;

    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (c >= X.channels) return;
    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = sigmoid(v);
    O.Set(n, y, x, c, v);
}

NUMTHREADS((512,1,1), (128,1,1), (64,1,1))
void KERNEL_FUNC(Sigmoid_Nyxc)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.batch * O.height * O.width * O.channels, 1, 1);
    TENSOR_ARGS2(X, O);

    uint nyxc = dispatchThreadID.x;

    uint c = nyxc % X.channels;
    uint nyx = nyxc / X.channels;
    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = sigmoid(v);
    O.Set(n, y, x, c, v);
}

NUMTHREADS((16, 16, 1), (16, 8, 1), (16, 4, 1))
void KERNEL_FUNC(HardSigmoid_CNyx)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.channels, O.batch * O.height * O.width, 1);
    TENSOR_ARGS2(X, O);

    uint c = dispatchThreadID.x;
    uint nyx = dispatchThreadID.y;

    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (c >= X.channels) return;
    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = hardsigmoid(v);
    O.Set(n, y, x, c, v);
}

NUMTHREADS((512, 1, 1), (128, 1, 1), (64, 1, 1))
void KERNEL_FUNC(HardSigmoid_Nyxc)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.batch * O.height * O.width * O.channels, 1, 1);
    TENSOR_ARGS2(X, O);

    uint nyxc = dispatchThreadID.x;

    uint c = nyxc % X.channels;
    uint nyx = nyxc / X.channels;
    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = hardsigmoid(v);
    O.Set(n, y, x, c, v);
}

NUMTHREADS((16,16,1), (16,8,1), (16,4,1))
void KERNEL_FUNC(Swish_CNyx)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.channels, O.batch * O.height * O.width, 1);
    TENSOR_ARGS2(X, O);

    uint c = dispatchThreadID.x;
    uint nyx = dispatchThreadID.y;

    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (c >= X.channels) return;
    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = swish(v);
    O.Set(n, y, x, c, v);
}

NUMTHREADS((512,1,1), (128,1,1), (64,1,1))
void KERNEL_FUNC(Swish_Nyxc)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.batch * O.height * O.width * O.channels, 1, 1);
    TENSOR_ARGS2(X, O);

    uint nyxc = dispatchThreadID.x;

    uint c = nyxc % X.channels;
    uint nyx = nyxc / X.channels;
    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = swish(v);
    O.Set(n, y, x, c, v);
}

NUMTHREADS((16,16,1), (16,8,1), (16,4,1))
void KERNEL_FUNC(Elu_CNyx)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.channels, O.batch * O.height * O.width, 1);
    TENSOR_ARGS2(X, O);

    uint c = dispatchThreadID.x;
    uint nyx = dispatchThreadID.y;

    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (c >= X.channels) return;
    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = elu(v);
    O.Set(n, y, x, c, v);
}

NUMTHREADS((512,1,1), (128,1,1), (64,1,1))
void KERNEL_FUNC(Elu_Nyxc)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.batch * O.height * O.width * O.channels, 1, 1);
    TENSOR_ARGS2(X, O);

    uint nyxc = dispatchThreadID.x;

    uint c = nyxc % X.channels;
    uint nyx = nyxc / X.channels;
    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = elu(v);
    O.Set(n, y, x, c, v);
}

NUMTHREADS((16,16,1), (16,8,1), (16,4,1))
void KERNEL_FUNC(LeakyRelu_CNyx)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.channels, O.batch * O.height * O.width, 1);
    TENSOR_ARGS2(X, O);

    uint c = dispatchThreadID.x;
    uint nyx = dispatchThreadID.y;

    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (c >= X.channels) return;
    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = lrelu(v);
    O.Set(n, y, x, c, v);
}

NUMTHREADS((512,1,1), (128,1,1), (64,1,1))
void KERNEL_FUNC(LeakyRelu_Nyxc)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.batch * O.height * O.width * O.channels, 1, 1);
    TENSOR_ARGS2(X, O);

    uint nyxc = dispatchThreadID.x;

    uint c = nyxc % X.channels;
    uint nyx = nyxc / X.channels;
    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = lrelu(v);
    O.Set(n, y, x, c, v);
}

NUMTHREADS((16,16,1), (16,8,1), (16,4,1))
void KERNEL_FUNC(Exp_CNyx)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.channels, O.batch * O.height * O.width, 1);
    TENSOR_ARGS2(X, O);

    uint c = dispatchThreadID.x;
    uint nyx = dispatchThreadID.y;

    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (c >= X.channels) return;
    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = exp(v);
    O.Set(n, y, x, c, v);
}

NUMTHREADS((512,1,1), (128,1,1), (64,1,1))
void KERNEL_FUNC(Exp_Nyxc)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.batch * O.height * O.width * O.channels, 1, 1);
    TENSOR_ARGS2(X, O);

    uint nyxc = dispatchThreadID.x;

    uint c = nyxc % X.channels;
    uint nyx = nyxc / X.channels;
    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = exp(v);
    O.Set(n, y, x, c, v);
}

NUMTHREADS((16,16,1), (16,8,1), (16,4,1))
void KERNEL_FUNC(Log_CNyx)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.channels, O.batch * O.height * O.width, 1);
    TENSOR_ARGS2(X, O);

    uint c = dispatchThreadID.x;
    uint nyx = dispatchThreadID.y;

    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (c >= X.channels) return;
    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = log(v);
    O.Set(n, y, x, c, v);
}

NUMTHREADS((512,1,1), (128,1,1), (64,1,1))
void KERNEL_FUNC(Log_Nyxc)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.batch * O.height * O.width * O.channels, 1, 1);
    TENSOR_ARGS2(X, O);

    uint nyxc = dispatchThreadID.x;

    uint c = nyxc % X.channels;
    uint nyx = nyxc / X.channels;
    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = log(v);
    O.Set(n, y, x, c, v);
}

NUMTHREADS((16, 16, 1), (16, 8, 1), (16, 4, 1))
void KERNEL_FUNC(Sqrt_CNyx)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	//DISPATCH ARGS(O.channels, O.batch * O.height * O.width, 1);
	TENSOR_ARGS2(X, O);

	uint c = dispatchThreadID.x;
	uint nyx = dispatchThreadID.y;

	uint x = nyx % X.width;
	uint ny = nyx / X.width;
	uint y = ny % X.height;
	uint n = ny / X.height;

	if (c >= X.channels) return;
	if (n >= X.batch) return;

	float v = X.Get(n, y, x, c);
	v = sqrt(v);
	O.Set(n, y, x, c, v);
}

NUMTHREADS((512, 1, 1), (128, 1, 1), (64, 1, 1))
void KERNEL_FUNC(Sqrt_Nyxc)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	//DISPATCH ARGS(O.batch * O.height * O.width * O.channels, 1, 1);
	TENSOR_ARGS2(X, O);

	uint nyxc = dispatchThreadID.x;

	uint c = nyxc % X.channels;
	uint nyx = nyxc / X.channels;
	uint x = nyx % X.width;
	uint ny = nyx / X.width;
	uint y = ny % X.height;
	uint n = ny / X.height;

	if (n >= X.batch) return;

	float v = X.Get(n, y, x, c);
	v = sqrt(v);
	O.Set(n, y, x, c, v);
}

NUMTHREADS((16, 16, 1), (16, 8, 1), (16, 4, 1))
void KERNEL_FUNC(Acos_CNyx)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.channels, O.batch * O.height * O.width, 1);
    TENSOR_ARGS2(X, O);

    uint c = dispatchThreadID.x;
    uint nyx = dispatchThreadID.y;

    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (c >= X.channels) return;
    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = acos(v);
    O.Set(n, y, x, c, v);
}

NUMTHREADS((512, 1, 1), (128, 1, 1), (64, 1, 1))
void KERNEL_FUNC(Acos_Nyxc)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.batch * O.height * O.width * O.channels, 1, 1);
    TENSOR_ARGS2(X, O);

    uint nyxc = dispatchThreadID.x;

    uint c = nyxc % X.channels;
    uint nyx = nyxc / X.channels;
    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = acos(v);
    O.Set(n, y, x, c, v);
}

NUMTHREADS((16, 16, 1), (16, 8, 1), (16, 4, 1))
void KERNEL_FUNC(Acosh_CNyx)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.channels, O.batch * O.height * O.width, 1);
    TENSOR_ARGS2(X, O);

    uint c = dispatchThreadID.x;
    uint nyx = dispatchThreadID.y;

    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (c >= X.channels) return;
    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = log(v + sqrt(v * v - 1.0f));
    O.Set(n, y, x, c, v);
}

NUMTHREADS((512, 1, 1), (128, 1, 1), (64, 1, 1))
void KERNEL_FUNC(Acosh_Nyxc)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.batch * O.height * O.width * O.channels, 1, 1);
    TENSOR_ARGS2(X, O);

    uint nyxc = dispatchThreadID.x;

    uint c = nyxc % X.channels;
    uint nyx = nyxc / X.channels;
    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = log(v + sqrt(v * v - 1.0f));
    O.Set(n, y, x, c, v);
}

NUMTHREADS((16, 16, 1), (16, 8, 1), (16, 4, 1))
void KERNEL_FUNC(Asin_CNyx)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.channels, O.batch * O.height * O.width, 1);
    TENSOR_ARGS2(X, O);

    uint c = dispatchThreadID.x;
    uint nyx = dispatchThreadID.y;

    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (c >= X.channels) return;
    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = asin(v);
    O.Set(n, y, x, c, v);
}

NUMTHREADS((512, 1, 1), (128, 1, 1), (64, 1, 1))
void KERNEL_FUNC(Asin_Nyxc)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.batch * O.height * O.width * O.channels, 1, 1);
    TENSOR_ARGS2(X, O);

    uint nyxc = dispatchThreadID.x;

    uint c = nyxc % X.channels;
    uint nyx = nyxc / X.channels;
    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = asin(v);
    O.Set(n, y, x, c, v);
}

NUMTHREADS((16, 16, 1), (16, 8, 1), (16, 4, 1))
void KERNEL_FUNC(Asinh_CNyx)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.channels, O.batch * O.height * O.width, 1);
    TENSOR_ARGS2(X, O);

    uint c = dispatchThreadID.x;
    uint nyx = dispatchThreadID.y;

    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (c >= X.channels) return;
    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = log(v + sqrt(v*v + 1.0f));
    O.Set(n, y, x, c, v);
}

NUMTHREADS((512, 1, 1), (128, 1, 1), (64, 1, 1))
void KERNEL_FUNC(Asinh_Nyxc)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.batch * O.height * O.width * O.channels, 1, 1);
    TENSOR_ARGS2(X, O);

    uint nyxc = dispatchThreadID.x;

    uint c = nyxc % X.channels;
    uint nyx = nyxc / X.channels;
    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = log(v + sqrt(v*v + 1.0f));
    O.Set(n, y, x, c, v);
}

NUMTHREADS((16, 16, 1), (16, 8, 1), (16, 4, 1))
void KERNEL_FUNC(Atan_CNyx)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.channels, O.batch * O.height * O.width, 1);
    TENSOR_ARGS2(X, O);

    uint c = dispatchThreadID.x;
    uint nyx = dispatchThreadID.y;

    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (c >= X.channels) return;
    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = atan(v);
    O.Set(n, y, x, c, v);
}

NUMTHREADS((512, 1, 1), (128, 1, 1), (64, 1, 1))
void KERNEL_FUNC(Atan_Nyxc)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.batch * O.height * O.width * O.channels, 1, 1);
    TENSOR_ARGS2(X, O);

    uint nyxc = dispatchThreadID.x;

    uint c = nyxc % X.channels;
    uint nyx = nyxc / X.channels;
    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = atan(v);
    O.Set(n, y, x, c, v);
}

NUMTHREADS((16, 16, 1), (16, 8, 1), (16, 4, 1))
void KERNEL_FUNC(Atanh_CNyx)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.channels, O.batch * O.height * O.width, 1);
    TENSOR_ARGS2(X, O);

    uint c = dispatchThreadID.x;
    uint nyx = dispatchThreadID.y;

    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (c >= X.channels) return;
    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = 0.5f * log((1.0f + v) / (1.0f - v));
    O.Set(n, y, x, c, v);
}

NUMTHREADS((512, 1, 1), (128, 1, 1), (64, 1, 1))
void KERNEL_FUNC(Atanh_Nyxc)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.batch * O.height * O.width * O.channels, 1, 1);
    TENSOR_ARGS2(X, O);

    uint nyxc = dispatchThreadID.x;

    uint c = nyxc % X.channels;
    uint nyx = nyxc / X.channels;
    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = 0.5f * log((1.0f + v) / (1.0f - v));
    O.Set(n, y, x, c, v);
}

NUMTHREADS((16, 16, 1), (16, 8, 1), (16, 4, 1))
void KERNEL_FUNC(Cos_CNyx)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.channels, O.batch * O.height * O.width, 1);
    TENSOR_ARGS2(X, O);

    uint c = dispatchThreadID.x;
    uint nyx = dispatchThreadID.y;

    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (c >= X.channels) return;
    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = cos(v);
    O.Set(n, y, x, c, v);
}

NUMTHREADS((512, 1, 1), (128, 1, 1), (64, 1, 1))
void KERNEL_FUNC(Cos_Nyxc)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.batch * O.height * O.width * O.channels, 1, 1);
    TENSOR_ARGS2(X, O);

    uint nyxc = dispatchThreadID.x;

    uint c = nyxc % X.channels;
    uint nyx = nyxc / X.channels;
    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = cos(v);
    O.Set(n, y, x, c, v);
}

NUMTHREADS((16, 16, 1), (16, 8, 1), (16, 4, 1))
void KERNEL_FUNC(Cosh_CNyx)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.channels, O.batch * O.height * O.width, 1);
    TENSOR_ARGS2(X, O);

    uint c = dispatchThreadID.x;
    uint nyx = dispatchThreadID.y;

    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (c >= X.channels) return;
    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = 0.5f * (exp(v) + exp(-v));
    O.Set(n, y, x, c, v);
}

NUMTHREADS((512, 1, 1), (128, 1, 1), (64, 1, 1))
void KERNEL_FUNC(Cosh_Nyxc)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.batch * O.height * O.width * O.channels, 1, 1);
    TENSOR_ARGS2(X, O);

    uint nyxc = dispatchThreadID.x;

    uint c = nyxc % X.channels;
    uint nyx = nyxc / X.channels;
    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = 0.5f * (exp(v) + exp(-v));
    O.Set(n, y, x, c, v);
}

NUMTHREADS((16, 16, 1), (16, 8, 1), (16, 4, 1))
void KERNEL_FUNC(Sin_CNyx)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.channels, O.batch * O.height * O.width, 1);
    TENSOR_ARGS2(X, O);

    uint c = dispatchThreadID.x;
    uint nyx = dispatchThreadID.y;

    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (c >= X.channels) return;
    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = sin(v);
    O.Set(n, y, x, c, v);
}

NUMTHREADS((512, 1, 1), (128, 1, 1), (64, 1, 1))
void KERNEL_FUNC(Sin_Nyxc)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.batch * O.height * O.width * O.channels, 1, 1);
    TENSOR_ARGS2(X, O);

    uint nyxc = dispatchThreadID.x;

    uint c = nyxc % X.channels;
    uint nyx = nyxc / X.channels;
    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = sin(v);
    O.Set(n, y, x, c, v);
}

NUMTHREADS((16, 16, 1), (16, 8, 1), (16, 4, 1))
void KERNEL_FUNC(Sinh_CNyx)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.channels, O.batch * O.height * O.width, 1);
    TENSOR_ARGS2(X, O);

    uint c = dispatchThreadID.x;
    uint nyx = dispatchThreadID.y;

    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (c >= X.channels) return;
    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = 0.5f * (exp(v) - exp(-v));
    O.Set(n, y, x, c, v);
}

NUMTHREADS((512, 1, 1), (128, 1, 1), (64, 1, 1))
void KERNEL_FUNC(Sinh_Nyxc)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.batch * O.height * O.width * O.channels, 1, 1);
    TENSOR_ARGS2(X, O);

    uint nyxc = dispatchThreadID.x;

    uint c = nyxc % X.channels;
    uint nyx = nyxc / X.channels;
    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = 0.5f * (exp(v) - exp(-v));
    O.Set(n, y, x, c, v);
}

NUMTHREADS((16, 16, 1), (16, 8, 1), (16, 4, 1))
void KERNEL_FUNC(Tan_CNyx)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.channels, O.batch * O.height * O.width, 1);
    TENSOR_ARGS2(X, O);

    uint c = dispatchThreadID.x;
    uint nyx = dispatchThreadID.y;

    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (c >= X.channels) return;
    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = tan(v);
    O.Set(n, y, x, c, v);
}

NUMTHREADS((512, 1, 1), (128, 1, 1), (64, 1, 1))
void KERNEL_FUNC(Tan_Nyxc)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.batch * O.height * O.width * O.channels, 1, 1);
    TENSOR_ARGS2(X, O);

    uint nyxc = dispatchThreadID.x;

    uint c = nyxc % X.channels;
    uint nyx = nyxc / X.channels;
    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = tan(v);
    O.Set(n, y, x, c, v);
}

NUMTHREADS((16,16,1), (16,8,1), (16,4,1))
void KERNEL_FUNC(Pow_CNyx)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.channels, O.batch * O.height * O.width, 1);
    TENSOR_ARGS2(X, O);

    uint c = dispatchThreadID.x;
    uint nyx = dispatchThreadID.y;

    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (c >= X.channels) return;
    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = signed_pow(v);
    O.Set(n, y, x, c, v);
}

NUMTHREADS((512,1,1), (128,1,1), (64,1,1))
void KERNEL_FUNC(Pow_Nyxc)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //DISPATCH ARGS(O.batch * O.height * O.width * O.channels, 1, 1);
    TENSOR_ARGS2(X, O);

    uint nyxc = dispatchThreadID.x;

    uint c = nyxc % X.channels;
    uint nyx = nyxc / X.channels;
    uint x = nyx % X.width;
    uint ny = nyx / X.width;
    uint y = ny % X.height;
    uint n = ny / X.height;

    if (n >= X.batch) return;

    float v = X.Get(n, y, x, c);
    v = signed_pow(v);
    O.Set(n, y, x, c, v);
}

NUMTHREADS((16, 16, 1), (16, 8, 1), (16, 4, 1))
void KERNEL_FUNC(Clip_CNyx)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	//DISPATCH ARGS(O.channels, O.batch * O.height * O.width, 1);
	TENSOR_ARGS2(X, O);

	uint c = dispatchThreadID.x;
	uint nyx = dispatchThreadID.y;

	uint x = nyx % X.width;
	uint ny = nyx / X.width;
	uint y = ny % X.height;
	uint n = ny / X.height;

	if (c >= X.channels) return;
	if (n >= X.batch) return;

	float v = X.Get(n, y, x, c);
	v = activation_clip(v);
	O.Set(n, y, x, c, v);
}

NUMTHREADS((512, 1, 1), (128, 1, 1), (64, 1, 1))
void KERNEL_FUNC(Clip_Nyxc)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	//DISPATCH ARGS(O.batch * O.height * O.width * O.channels, 1, 1);
	TENSOR_ARGS2(X, O);

	uint nyxc = dispatchThreadID.x;

	uint c = nyxc % X.channels;
	uint nyx = nyxc / X.channels;
	uint x = nyx % X.width;
	uint ny = nyx / X.width;
	uint y = ny % X.height;
	uint n = ny / X.height;

	if (n >= X.batch) return;

	float v = X.Get(n, y, x, c);
	v = activation_clip(v);
	O.Set(n, y, x, c, v);
}

TENSOR_DECL(W)
TENSOR_DECL(B)
TENSOR_DECL(WBK)

NUMTHREADS((4, 8, 8), (4, 8, 4), (4, 4, 4))
void KERNEL_FUNC(PRelu)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	//DISPATCH ARGS(O.channels, O.width, O.height);
	TENSOR_TWOINPUTS(X, W, O);

	uint c = dispatchThreadID.x;
	uint x = dispatchThreadID.y;
	uint y = dispatchThreadID.z;

	if (c >= O.channels) return;
	if (x >= O.width) return;
	if (y >= O.height) return;

	float slope = W.Get(0, 0, 0, c);

	for (uint n = 0; n < X.batch; ++n)
	{
		float slope = W.BroadcastGet(n, y, x, c);
		float v = X.Get(n, y, x, c);
		v = prelu(v,slope);
		O.Set(n, y, x, c, v);
	}

}


NUMTHREADS((256, 1, 1), (128, 1, 1), (64, 1, 1))
void PRelu_Flat(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	//DISPATCH ARGS(O.length, 1, 1);
	TENSOR_ARGS3(X, W, O);

	uint i = dispatchThreadID.x;
	if (i >= O.GetLength()) return;

	float slope = W.FastBroadcastGet(i);
	float v = X.FastGet(i);
	v = prelu(v, slope);
	O.FastSet(i, v);

}

NUMTHREADS((256, 1, 1), (128, 1, 1), (64, 1, 1))
void PRelu_Loop(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	//DISPATCH ARGS(O.length, 1, 1);
	TENSOR_ARGS3(X, W, O);

	uint i = dispatchThreadID.x;
	uint len = O.GetLength();

	while (i < len)
	{
		float slope = W.FastBroadcastGet(i);
		float v = X.FastGet(i);
		v = prelu(v, slope);
		O.FastSet(i, v);

		i += _LoopStride;
	}

}

NUMTHREADS((32, 4, 1), (32, 2, 1), (16, 2, 1))
void KERNEL_FUNC(PRelu_CNyx2)(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	//DISPATCH ARGS(O.channels, O.batch * O.height * O.width, 1);
	TENSOR_ARGS3(X, W, O);

	uint c = dispatchThreadID.x;
	uint i = dispatchThreadID.y * X.channels + c;

	if (c >= X.channels) return;
	if (i >= X.GetLength()) return;

	float slope = W.FastBroadcastGet(i);
	float v = X.FastGet(i);
	v = prelu(v, slope);
	O.FastSet(i, v);

}
