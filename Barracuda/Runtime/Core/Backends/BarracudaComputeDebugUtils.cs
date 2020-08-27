using System.Diagnostics;
using UnityEngine;
using System.Runtime.InteropServices;

namespace Unity.Barracuda {

public class ComputeDebugUtils
{
    /// <summary>
    /// DEBUG ONLY: `debugKernels` allow to track out of bound read/write and assertion in kernels.
    /// When set to true be sure to define KERNEL_ASSERTS or FORCE_DEBUG in the particular kernel(s)
    /// you want to debug (see in DebugUtils.cginc).
    /// Production code should not set this to 'true' as this will significantly degrade performances.
    /// </summary>
    public static bool debugKernels = false;

    /// <summary>
    /// DEBUG ONLY: if ComputeDebugUtils.debugKernels is true and debugger is attached, debugger will break when a kernel assertion is catch.
    /// </summary>
    public static bool breakOnAssertion = false;

    //Keep in sync with DebugUtils.cginc KERNEL_ASSERT_CONTEXT defines
    private enum KernelAssertContext
    {
        ReadOnlyTensor_Read = 0,
        ReadWriteTensor_Read = 1,
        ReadWriteTensor_Write = 2,
        SharedTensor_Read = 3,
        Assertion = 4,
        AssertionWithValue = 5
    }

    static ComputeDebugUtils()
    {
        string[] args = System.Environment.GetCommandLineArgs ();
        for (int i = 0; i < args.Length; i++) {
            if (args [i] == "-barracuda-debug-gpu-kernels")
            {
                debugKernels = true;
            }
        }
    }

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct KernelAssertInfo
    {
        public KernelAssertInfo(uint[] data)
        {
            UnityEngine.Debug.Assert(numUintInKernelAssertInfo == data.Length);
            UnityEngine.Debug.Assert(numUintInKernelAssertInfo == 8,
                "Please change KernelAssertInfo constructor if altering the struct.");
            lockValue = data[0];
            lineNumber = data[1];
            context = data[2];
            index = data[3];
            bufferSize = data[4];
            debugValue = data[5];
            padding1 = data[6];
            padding2 = data[7];
        }

        public readonly uint lockValue;
        public readonly uint lineNumber;
        public readonly uint context;
        public readonly uint index;
        public readonly uint bufferSize;
        public readonly uint debugValue;
        public readonly uint padding1;
        public readonly uint padding2;
    }
    private static readonly int numUintInKernelAssertInfo = Marshal.SizeOf(typeof(KernelAssertInfo))/sizeof(uint);

    private static ComputeBuffer kernelDebugInfo = null;

    private static void LogAssertion(KernelAssertInfo info, string kernelName)
    {
        if (info.lockValue != 0)
        {
            string source;
            switch (info.context)
            {
                case (int) KernelAssertContext.ReadOnlyTensor_Read:
                    source = $"Out of bound while Reading a ReadonlyTensor of length {info.bufferSize} at index {info.index} (at Tensor.cginc line {info.lineNumber})";
                    break;
                case (int) KernelAssertContext.ReadWriteTensor_Read:
                    source = $"Out of bound while Reading a ReadWriteTensor of length {info.bufferSize} at index {info.index} (at Tensor.cginc line {info.lineNumber})";
                    break;
                case (int) KernelAssertContext.ReadWriteTensor_Write:
                    source = $"Out of bound while Writing to a ReadWriteTensor of length {info.bufferSize} at index {info.index} (at Tensor.cginc line {info.lineNumber})";
                    break;
                case (int) KernelAssertContext.SharedTensor_Read:
                    source = $"Out of bound while Reading a SharedTensor of length {info.bufferSize} at index {info.index} (at Tensor.cginc line {info.lineNumber})";
                    break;
                case (int) KernelAssertContext.Assertion:
                    source = $"Assertion at line {info.lineNumber}";
                    break;
                case (int) KernelAssertContext.AssertionWithValue:
                    source = $"Assertion at line {info.lineNumber}, debug value is {info.debugValue}";
                    break;
                default:
                    source = "Unknown error";
                    break;
            }

            string message = $"{source} in kernel {kernelName}.";
            D.LogError(message);

            if (breakOnAssertion)
            {
                Debugger.Break();
            }
        }
    }


    public static void PrepareDispatch()
    {
        //Lazy alloc, will be released by GC.
        if (debugKernels && kernelDebugInfo == null)
        {
            kernelDebugInfo = new ComputeBuffer(1, numUintInKernelAssertInfo*sizeof(uint));
        }

        if (debugKernels)
        {
            Shader.SetGlobalBuffer("KernelAssertInfoBuffer", kernelDebugInfo);
            kernelDebugInfo.SetData(new uint[numUintInKernelAssertInfo]); //TODO use a kernel to zero out the buffer to avoid a extra sync.
        }
    }

    public static void VerifyDispatch(string kernelName)
    {
        if (debugKernels)
        {
            UnityEngine.Debug.Assert(kernelDebugInfo != null);
            var data = new uint[numUintInKernelAssertInfo];
            kernelDebugInfo.GetData(data, 0, 0, numUintInKernelAssertInfo);
            LogAssertion(new KernelAssertInfo(data), kernelName);
        }
    }
}

} // namespace Unity.Barracuda
