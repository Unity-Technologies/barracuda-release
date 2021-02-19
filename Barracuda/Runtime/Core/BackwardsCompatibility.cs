using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Assertions;

namespace Unity.Barracuda {

/// <summary>
/// Deprecated APIs, left here only for backwards compatibility
/// </summary>
public static class DeprecatedTensorExtensions
{
    /// <summary>
    /// Deprecated, use `AdjustPadToPool` version with pool as an array instead
    /// </summary>
    /// <param name="tensor">`Tensor`</param>
    /// <param name="pool">pool tuple</param>
    /// <param name="stride">stride</param>
    /// <param name="pad">padding</param>
    /// <returns>shape as int array</returns>
    [ObsoleteAttribute("Use AdjustPadToPool version with pool as an array instead.", false)]
    public static int[] AdjustPadToPool(this Tensor tensor, ValueTuple<int,int> pool, int[] stride, int[] pad)
    {
        unsafe
        {
            int* pPool = stackalloc int[2];
            pPool[0] = pool.Item1;
            pPool[1] = pool.Item2;
            return tensor.shape.AdjustPadToPool(pPool, stride, pad);
        }
    }

    /// <summary>
    /// Deprecated, use `AdjustPadToPool` version with pool as an array instead
    /// </summary>
    /// <param name="shape">`TensorShape`</param>
    /// <param name="pool">pool tuple</param>
    /// <param name="stride">stride</param>
    /// <param name="pad">padding</param>
    /// <returns>shape as int array</returns>
    [ObsoleteAttribute("Use AdjustPadToPool version with pool as an array instead.", false)]
    public static int[] AdjustPadToPool(this TensorShape shape, ValueTuple<int,int> pool, int[] stride, int[] pad)
    {
        unsafe
        {
            int* pPool = stackalloc int[2];
            pPool[0] = pool.Item1;
            pPool[1] = pool.Item2;

            return shape.AdjustPadToPool(pPool, stride, pad);
        }
    }

    /// <summary>
    /// Deprecated. Use <c>UploadToDevice</c> instead
    /// </summary>
    /// <param name="self">Tensor</param>
    /// <param name="onDevice">ITensorData</param>
    /// <param name="forceInvalidateCache">Force cache invalidation</param>
    [ObsoleteAttribute("Use UploadToDevice instead.", false)]
    public static void PinToDeviceAndUploadToIt(this Tensor self, ITensorData onDevice, bool forceInvalidateCache = true)
    {
        self.UploadToDevice(onDevice, forceInvalidateCache);
    }

    /// <summary>
    /// Deprecated. Use <c>AttachToDevice</c> instead
    /// </summary>
    /// <param name="self">Tensor</param>
    /// <param name="onDevice">ITensorData</param>
    [ObsoleteAttribute("Use AttachToDevice instead.", false)]
    public static void PinToDeviceAndDownloadFromIt(this Tensor self, ITensorData onDevice)
    {
        self.AttachToDevice(onDevice);
    }

    /// <summary>
    /// Deprecated. Use <c>DetachFromDevice</c> instead
    /// </summary>
    /// <param name="self">Tensor</param>
    /// <param name="disposeUnpinned">Call dispose when unpinned</param>
    /// <returns></returns>
    [ObsoleteAttribute("Use DetachFromDevice instead.", false)]
    public static ITensorData Unpin(this Tensor self, bool disposeUnpinned = true)
    {
        return self.DetachFromDevice(disposeUnpinned);
    }

    /// <summary>
    /// Deprecated. Use <c>AttachToDevice</c> instead
    /// </summary>
    /// <param name="self">Tensor</param>
    /// <param name="onDevice">ITensorData</param>
    [ObsoleteAttribute("Use AttachToDevice instead.", false)]
    public static void CastOnDevice(this Tensor self, ITensorData onDevice)
    {
        self.AttachToDevice(onDevice);
    }

    #region Tensor
    // @SEE: Tensor.cs
    // public ITensorData UnpinAndDisposeTensor()
    // public float[] readonlyArray { get { PrepareCacheForAccess(); return m_Cache; } }
    // public int readonlyArrayOffset { get { return 0; } }
    #endregion
}

/// <summary>
/// Deprecated `TestSet` extensions
/// </summary>
public static class DeprecatedTestSetExtensions
{
    /// <summary>
    /// Deprecated. Use `GetInputShape` version returning a TensorShape instead
    /// </summary>
    /// <param name="self">`TestSet`</param>
    /// <param name="idx">input index</param>
    /// <returns>input shape as array</returns>
    [ObsoleteAttribute("Use GetInputShape version returning a TensorShape instead.", false)]
    public static int[] GetInputShape(this TestSet self, int idx = 0)
    {
        var shape = self.GetInputShape(idx);
        Assert.IsTrue(shape.Is4D());
        return shape.ToArray();
    }

    /// <summary>
    /// Deprecated. Use `GetOutputShape` version returning a TensorShape instead
    /// </summary>
    /// <param name="self">`TestSet`</param>
    /// <param name="idx">output index</param>
    /// <returns>shape as int array</returns>
    [ObsoleteAttribute("Use GetOutputShape version returning a TensorShape instead.", false)]
    public static int[] GetOutputShape(this TestSet self, int idx = 0)
    {
        var shape = self.GetOutputShape(idx);
        Assert.IsTrue(shape.Is4D());
        return shape.ToArray();
    }
}

/// <summary>
/// Deprecated <c>ITensorData</c> extensions
/// </summary>
public static class DeprecatedTensorDataExtensions
{
    /// <summary>
    /// Deprecated. Use <c>maxCapacity</c> extensions
    /// </summary>
    /// <param name="self">Tensor</param>
    /// <returns>max Tensor capacity</returns>
    [ObsoleteAttribute("Use maxCapacity instead.", false)]
    public static int GetMaxCount(this ITensorData self)
    {
        return self.maxCapacity;
    }
}

/// <summary>
/// Deprecated <c>IWorker</c> extensions
/// </summary>
public static class DeprecatedWorkerExtensions
{
    #region Inputs
    /// <summary>
    /// Deprecated. Use <c>SetInput</c> instead
    /// </summary>
    /// <param name="worker">IWorker</param>
    /// <param name="x">input Tensor</param>
    [ObsoleteAttribute("Use SetInput instead.", false)]
    public static void AddInput(this IWorker worker, Tensor x)
    {
        worker.SetInput(x);
    }

    /// <summary>
    /// Deprecated. Use <c>SetInput</c> instead
    /// </summary>
    /// <param name="worker">IWorker</param>
    /// <param name="name">input Tensor name</param>
    /// <param name="x">input Tensor</param>
    [ObsoleteAttribute("Use SetInput instead.", false)]
    public static void AddInput(this IWorker worker, string name, Tensor x)
    {
        worker.SetInput(name, x);
    }
    #endregion

    #region Outputs
    /// <summary>
    /// Deprecated. Use <c>PeekOutput</c> instead
    /// </summary>
    /// <param name="worker">IWorker</param>
    /// <returns>output Tensor</returns>
    [ObsoleteAttribute("Use PeekOutput instead.", false)]
    public static Tensor Peek(this IWorker worker)
    {
        return worker.PeekOutput();
    }

    /// <summary>
    /// Deprecated. Use <c>PeekOutput</c> instead
    /// </summary>
    /// <param name="worker">IWorker</param>
    /// <param name="name">output Tensor name</param>
    /// <returns>output Tensor</returns>
    [ObsoleteAttribute("Use PeekOutput instead.", false)]
    public static Tensor Peek(this IWorker worker, string name)
    {
        return worker.PeekOutput(name);
    }
    #endregion

    #region Schedule one layer at a time
    /// <summary>
    /// Deprecated. Use <c>StartManualSchedule</c> instead
    /// </summary>
    /// <param name="worker">IWorker</param>
    /// <returns>Manual schedule iterator</returns>
    [ObsoleteAttribute("Use StartManualSchedule instead.", false)]
    public static IEnumerator ExecuteAsync(this IWorker worker)
    {
        return worker.StartManualSchedule();
    }

    /// <summary>
    /// Deprecated. Use <c>StartManualSchedule</c> instead
    /// </summary>
    /// <param name="worker">IWorker</param>
    /// <param name="input">input Tensor</param>
    /// <returns>Manual schedule iterator</returns>
    [ObsoleteAttribute("Use StartManualSchedule instead.", false)]
    public static IEnumerator ExecuteAsync(this IWorker worker, Tensor input)
    {
        return worker.StartManualSchedule(input);
    }

    /// <summary>
    /// Deprecated. Use <c>StartManualSchedule</c> instead
    /// </summary>
    /// <param name="worker">IWorker</param>
    /// <param name="inputs">input Tensor Dictionary</param>
    /// <returns>Manual schedule iterator</returns>
    [ObsoleteAttribute("Use StartManualSchedule instead.", false)]
    public static IEnumerator ExecuteAsync(this IWorker worker, IDictionary<string, Tensor> inputs)
    {
        return worker.StartManualSchedule(inputs);
    }

    /// <summary>
    /// Deprecated. Use <c>FlushSchedule</c> instead
    /// </summary>
    /// <param name="worker">IWorker</param>
    [ObsoleteAttribute("Use FlushSchedule instead.", false)]
    public static void WaitForCompletion(this IWorker worker)
    {
        worker.FlushSchedule(blocking:true);
    }

    /// <summary>
    /// Deprecated. Use <c>scheduleProgress</c> instead
    /// </summary>
    /// <param name="worker">IWorker</param>
    /// <returns>Manual schedule progress (0 = 0%, 1 = 100% complete)</returns>
    [ObsoleteAttribute("Use scheduleProgress instead.", false)]
    public static float GetAsyncProgress(this IWorker worker)
    {
        return worker.scheduleProgress;
    }
    #endregion

    #region Outputs

    /// <summary>
    /// Deprecated. Use <c>Execute</c> followed by <c>CopyOutput</c> and <c>PrepareCacheForAccess</c> instead
    /// </summary>
    /// <param name="worker">IWorker</param>
    /// <param name="input">input Tensor</param>
    /// <returns>output Tensor</returns>
    [ObsoleteAttribute("Use Execute followed by CopyOutput and PrepareCacheForAccess instead.", false)]
    public static Tensor ExecuteAndWaitForCompletion(this IWorker worker, Tensor input)
    {
        worker.Execute(input);
        return worker.CopyOutput();
    }

    /// <summary>
    /// Deprecated. Use <c>Execute</c> followed by <c>CopyOutput</c> and <c>PrepareCacheForAccess</c> instead
    /// </summary>
    /// <param name="worker">IWorker</param>
    /// <param name="inputs">input Tensor Dictionary</param>
    /// <returns>output Tensor</returns>
    [ObsoleteAttribute("Use Execute followed by CopyOutput and PrepareCacheForAccess instead.", false)]
    public static Tensor ExecuteAndWaitForCompletion(this IWorker worker, IDictionary<string, Tensor> inputs)
    {
        worker.Execute(inputs);
        return worker.CopyOutput();
    }

    /// <summary>
    /// Deprecated. Use <c>PeekOutput</c> followed by <c>TakeOwnership</c> or <c>DeepCopy</c> instead
    /// </summary>
    /// <param name="worker">IWorker</param>
    /// <returns>output Tensor</returns>
    [ObsoleteAttribute("Use PeekOutput followed by TakeOwnership or DeepCopy instead.", false)]
    public static Tensor FetchAndTakeOwnership(this IWorker worker)
    {
        var output = worker.PeekOutput();
        output.TakeOwnership();
        return output;

    }

    /// <summary>
    /// Deprecated. Use <c>PeekOutput</c> followed by <c>TakeOwnership</c> or <c>DeepCopy</c> instead
    /// </summary>
    /// <param name="worker">IWorker</param>
    /// <param name="name">output Tensor name</param>
    /// <returns>output Tensor</returns>
    [ObsoleteAttribute("Use PeekOutput followed by TakeOwnership or DeepCopy instead.", false)]
    public static Tensor FetchAndTakeOwnership(this IWorker worker, string name)
    {
        var output = worker.PeekOutput(name);
        output.TakeOwnership();
        return output;
    }

    /// <summary>
    /// Deprecated. Use <c>CopyOutput</c> instead
    /// </summary>
    /// <param name="worker">IWorker</param>
    /// <returns>copy of the output Tensor</returns>
    [ObsoleteAttribute("Use CopyOutput instead.", false)]
    public static Tensor Fetch(this IWorker worker)
    {
        return worker.CopyOutput();
    }

    /// <summary>
    /// Deprecated. Use <c>CopyOutput</c> instead
    /// </summary>
    /// <param name="worker">IWorker</param>
    /// <param name="name">output Tensor name</param>
    /// <returns>copy of the output Tensor</returns>
    [ObsoleteAttribute("Use CopyOutput instead.", false)]
    public static Tensor Fetch(this IWorker worker, string name)
    {
        return worker.CopyOutput(name);
    }
    #endregion
}

/// <summary>
/// Deprecated. Use <c>WorkerFactory</c> class instead
/// </summary>
[ObsoleteAttribute("Use WorkerFactory class instead.", false)]
public class BarracudaWorkerFactory : WorkerFactory
{
    /// <summary>
    /// Device type enum
    /// </summary>
    public enum Flags
    {
        /// <summary>
        /// GPU
        /// </summary>
        Compute = Device.GPU,

        /// <summary>
        /// CPU
        /// </summary>
        CSharp  = Device.CPU
    }

    /// <summary>
    /// Compare against <c>Flags</c> enum
    /// </summary>
    /// <param name="type">type</param>
    /// <param name="flags">flags</param>
    /// <returns>True if matches</returns>
    public static bool IsType(Type type, Flags flags)
    {
        return IsType(type, (Device)flags);
    }
}

/// <summary>
/// Deprecated. Use <c>Tensor.ToRenderTexture</c> method instead
/// </summary>
[ObsoleteAttribute("Use Tensor.ToRenderTexture method instead.", false)]
public class BarracudaTextureUtils
{
    /// <summary>
    /// Copy Tensor data to RenderTexture
    /// </summary>
    /// <param name="x">Tensor</param>
    /// <param name="target">target RenderTexture</param>
    /// <param name="batch">batch</param>
    /// <param name="fromChannel">from channel</param>
    /// <param name="scale">scale</param>
    /// <param name="bias">bias</param>
    public static void TensorToRenderTexture(Tensor x, RenderTexture target,
                                            int batch = 0, int fromChannel = 0, float scale = 1.0f, float bias = 0f)
    {
        x.ToRenderTexture(target, batch, fromChannel, scale, bias);
    }

    /// <summary>
    /// Copy Tensor data to RenderTexture
    /// </summary>
    /// <param name="x">Tensor</param>
    /// <param name="batch">batch</param>
    /// <param name="fromChannel">from channel</param>
    /// <param name="scale">scale</param>
    /// <param name="bias">bias</param>
    /// <returns>RenderTexture created from Tensor data</returns>
    public static RenderTexture TensorToRenderTexture(Tensor x,
                                                int batch = 0, int fromChannel = 0, float scale = 1.0f, float bias = 0f)
    {
        return x.ToRenderTexture(batch, fromChannel, scale, bias);
    }
}


} // namespace Unity.Barracuda
