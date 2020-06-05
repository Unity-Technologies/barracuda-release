using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

namespace Unity.Barracuda {

// Deprecated APIs, left here only for backwards compatibility
public static class DeprecatedTensorExtensions
{
    [ObsoleteAttribute("Use UploadToDevice instead.", false)]
    public static void PinToDeviceAndUploadToIt(this Tensor self, ITensorData onDevice, bool forceInvalidateCache = true)
    {
        self.UploadToDevice(onDevice, forceInvalidateCache);
    }

    [ObsoleteAttribute("Use AttachToDevice instead.", false)]
    public static void PinToDeviceAndDownloadFromIt(this Tensor self, ITensorData onDevice)
    {
        self.AttachToDevice(onDevice);
    }

    [ObsoleteAttribute("Use DetachFromDevice instead.", false)]
    public static ITensorData Unpin(this Tensor self, bool disposeUnpinned = true)
    {
        return self.DetachFromDevice(disposeUnpinned);
    }

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

public static class DeprecatedTensorDataExtensions
{
    [ObsoleteAttribute("Use maxCapacity instead.", false)]
    public static int GetMaxCount(this ITensorData self)
    {
        return self.maxCapacity;
    }
}

public static class DeprecatedWorkerExtensions
{
    #region Inputs
    [ObsoleteAttribute("Use SetInput instead.", false)]
    public static void AddInput(this IWorker worker, Tensor x)
    {
        worker.SetInput(x);
    }
    [ObsoleteAttribute("Use SetInput instead.", false)]
    public static void AddInput(this IWorker worker, string name, Tensor x)
    {
        worker.SetInput(name, x);
    }
    #endregion

    #region Outputs
    [ObsoleteAttribute("Use PeekOutput instead.", false)]
    public static Tensor Peek(this IWorker worker)
    {
        return worker.PeekOutput();
    }
    [ObsoleteAttribute("Use PeekOutput instead.", false)]
    public static Tensor Peek(this IWorker worker, string name)
    {
        return worker.PeekOutput(name);
    }
    #endregion

    #region Schedule one layer at a time
    [ObsoleteAttribute("Use StartManualSchedule instead.", false)]
    public static IEnumerator ExecuteAsync(this IWorker worker)
    {
        return worker.StartManualSchedule();
    }
    [ObsoleteAttribute("Use StartManualSchedule instead.", false)]
    public static IEnumerator ExecuteAsync(this IWorker worker, Tensor input)
    {
        return worker.StartManualSchedule(input);
    }
    [ObsoleteAttribute("Use StartManualSchedule instead.", false)]
    public static IEnumerator ExecuteAsync(this IWorker worker, IDictionary<string, Tensor> inputs)
    {
        return worker.StartManualSchedule(inputs);
    }
    [ObsoleteAttribute("Use FlushSchedule instead.", false)]
    public static void WaitForCompletion(this IWorker worker)
    {
        worker.FlushSchedule(blocking:true);
    }
    [ObsoleteAttribute("Use scheduleProgress instead.", false)]
    public static float GetAsyncProgress(this IWorker worker)
    {
        return worker.scheduleProgress;
    }
    #endregion

    #region Outputs

    [ObsoleteAttribute("Use Execute followed by CopyOutput and PrepareCacheForAccess instead.", false)]
    public static Tensor ExecuteAndWaitForCompletion(this IWorker worker, Tensor input)
    {
        worker.Execute(input);
        return worker.CopyOutput();
    }
    [ObsoleteAttribute("Use Execute followed by CopyOutput and PrepareCacheForAccess instead.", false)]
    public static Tensor ExecuteAndWaitForCompletion(this IWorker worker, IDictionary<string, Tensor> inputs)
    {
        worker.Execute(inputs);
        return worker.CopyOutput();
    }

    [ObsoleteAttribute("Use PeekOutput followed by TakeOwnership or DeepCopy instead.", false)]
    public static Tensor FetchAndTakeOwnership(this IWorker worker)
    {
        var output = worker.PeekOutput();
        output.TakeOwnership();
        return output;

    }
    [ObsoleteAttribute("Use PeekOutput followed by TakeOwnership or DeepCopy instead.", false)]
    public static Tensor FetchAndTakeOwnership(this IWorker worker, string name)
    {
        var output = worker.PeekOutput(name);
        output.TakeOwnership();
        return output;
    }

    [ObsoleteAttribute("Use CopyOutput instead.", false)]
    public static Tensor Fetch(this IWorker worker)
    {
        return worker.CopyOutput();
    }
    [ObsoleteAttribute("Use CopyOutput instead.", false)]
    public static Tensor Fetch(this IWorker worker, string name)
    {
        return worker.CopyOutput(name);
    }
    #endregion
}

[ObsoleteAttribute("Use WorkerFactory class instead.", false)]
public class BarracudaWorkerFactory : WorkerFactory
{
    public enum Flags
    {
        Compute = Device.GPU,
        CSharp  = Device.CPU
    }

    public static bool IsType(Type type, Flags flags)
    {
        return IsType(type, (Device)flags);
    }
}

[ObsoleteAttribute("Use Tensor.ToRenderTexture method instead.", false)]
public class BarracudaTextureUtils
{
    public static void TensorToRenderTexture(Tensor x, RenderTexture target,
                                            int batch = 0, int fromChannel = 0, float scale = 1.0f, float bias = 0f)
    {
        x.ToRenderTexture(target, batch, fromChannel, scale, bias);
    }

    public static RenderTexture TensorToRenderTexture(Tensor x,
                                                int batch = 0, int fromChannel = 0, float scale = 1.0f, float bias = 0f)
    {
        return x.ToRenderTexture(batch, fromChannel, scale, bias);
    }
}


} // namespace Unity.Barracuda
