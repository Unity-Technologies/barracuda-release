/// DEBUG ONLY:
/// `KERNEL_ASSERTS` allow to track out of bound read/write and assertion
/// in all kernels with the exception of those where FORCE_NO_DEBUG is defined.
/// to debug only a few kernel one can also define FORCE_NO_DEBUG per kernel rather.
/// To debug kernel be sure to set ComputeDebugUtils.debugKernels to true BarracudaComputeDebugUtils.cs also.
/// Production code should not define this as this will significantly degrade performances.
/// Defining those require Shader model 5.0 and not Metal (Metal does not support GetDimensions on buffer)
/// aka `#pragma target 5.0` see https://docs.unity3d.com/Manual/SL-ShaderCompileTargets.html.
//#define KERNEL_ASSERTS

//Keep in sync with BarracudaComputeDebugUtils.cs enum ComputeDebugUtils.KernelAssertContext
#define KERNEL_ASSERT_CONTEXT_READONLY_READ 0
#define KERNEL_ASSERT_CONTEXT_READWRITE_READ 1
#define KERNEL_ASSERT_CONTEXT_READWRITE_WRITE 2
#define KERNEL_ASSERT_CONTEXT_SHARED_READ 3
#define KERNEL_ASSERT_CONTEXT_ASSERTION 4

//Keep in sync with BarracudaComputeDebugUtils.cs enum ComputeDebugUtils.KernelAssertInfo
struct KernelAssertInfo
{
    uint lockValue;
    //context
    uint lineNumber;
    uint context;
    //specific to read/write OOB detection
    uint index;
    uint bufferSize;
    //padding
    uint padding0;
    uint padding1;
    uint padding2;
};

#if (defined(KERNEL_ASSERTS) && !defined(FORCE_NO_DEBUG)) || defined(FORCE_DEBUG)

    RWStructuredBuffer<KernelAssertInfo> KernelAssertInfoBuffer;
    void LogAssertion(uint index, uint bufferSize, uint lineNumber, uint context)
    {
        uint anAssertionIsAlreadyLogged;
        InterlockedAdd(KernelAssertInfoBuffer[0].lockValue, 1, anAssertionIsAlreadyLogged);
        if (!anAssertionIsAlreadyLogged)
        {
            KernelAssertInfoBuffer[0].lineNumber = lineNumber;
            KernelAssertInfoBuffer[0].context = context;
            KernelAssertInfoBuffer[0].index = index;
            KernelAssertInfoBuffer[0].bufferSize = bufferSize;
        }
    }

    uint GetSafeTensorIndex(uint index, uint bufferSize, uint lineNumber, uint context)
    {
        bool isIndexValid = (index >= 0 && index < bufferSize);
        if (isIndexValid)
            return index;

        LogAssertion(index, bufferSize, lineNumber, context);

        //Always return a valid index to avoid GPU crashs so CPU get a chance to catch the error.
        return 0;
    }

    void KernelAssert(bool isOk, int lineNumber)
    {
        if (isOk)
            return;

        LogAssertion(0, 0, lineNumber, KERNEL_ASSERT_CONTEXT_ASSERTION);
    }

    #define ASSERT_TENSOR_INDEX(index, context) \
            uint dataNumStructs, dataStride; \
            data.GetDimensions(dataNumStructs, dataStride); \
            uint safeIndex = GetSafeTensorIndex(index, dataNumStructs, __LINE__, context);
    #define TENSOR_READ(varName, index, context) ASSERT_TENSOR_INDEX(index, context); varName = data[safeIndex]
    #define TENSOR_WRITE(varName, index, context) ASSERT_TENSOR_INDEX(index, context); data[safeIndex] = varName

    #define KERNEL_ASSERT(condition) KernelAssert(condition, __LINE__)
#else
    #define TENSOR_READ(varName, index, context) varName = data[index]
    #define TENSOR_WRITE(varName, index, context) data[index] = varName
    #define KERNEL_ASSERT(condition)
#endif
