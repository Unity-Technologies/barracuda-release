using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

namespace Barracuda {

// Deprecated APIs, left here only for backwards compatibility
public static class DeprecatedWorkerExtensions
{
    #region Inputs
    /// <summary>
    /// Specify single tensor value as the input for the network.
    /// Useful when network has only one input and caller does not need to know input's name.
    /// </summary>
    [ObsoleteAttribute("Use SetInput instead.", false)]
    public static void AddInput(this IWorker worker, Tensor x)
    {
        worker.SetInput(x);
    }
    /// <summary>
    /// Specify tensor value for the named input of the network.
    /// </summary>
    [ObsoleteAttribute("Use SetInput instead.", false)]
    public static void AddInput(this IWorker worker, string name, Tensor x)
    {
        worker.SetInput(name, x);
    }
    #endregion

    #region Outputs
    /// <summary>
    /// Returns a reference to the first output tensor.
    /// Useful when network has only one output.
    /// IMPORTANT: follow with TakeOwnership() call, if you want tensor to outlive worker or make tensor copy with DeepCopy()
    /// </summary>
    [ObsoleteAttribute("Use PeekOutput instead.", false)]
    public static Tensor Peek(this IWorker worker)
    {
        return worker.PeekOutput();
    }
    /// <summary>
    /// Returns a reference to output tensor by name.
    /// IMPORTANT: follow with TakeOwnership() call, if you want tensor to outlive worker or make tensor copy with DeepCopy()
    /// </summary>
    [ObsoleteAttribute("Use PeekOutput instead.", false)]
    public static Tensor Peek(this IWorker worker, string name)
    {
        return worker.PeekOutput(name);
    }
    #endregion

    #region Non-blocking APIs
    /// <summary>
    /// Returns the output tensor and takes ownership of memory to outlive the worker.
    /// Useful when network has only one output.
    /// </summary>
    [ObsoleteAttribute("Use PeekOutput followed by TakeOwnership or DeepCopy instead.", false)]
    public static Tensor FetchAndTakeOwnership(this IWorker worker)
    {
        var output = worker.PeekOutput();
        output.TakeOwnership();
        return output;

    }
    /// <summary>
    /// Returns output tensor by name and takes ownership of memory to outlive worker.
    /// </summary>
    [ObsoleteAttribute("Use PeekOutput followed by TakeOwnership or DeepCopy instead.", false)]
    public static Tensor FetchAndTakeOwnership(this IWorker worker, string name)
    {
        var output = worker.PeekOutput(name);
        output.TakeOwnership();
        return output;
    }
    #endregion

    #region Blocking APIs
    /// <summary>
    /// This method is a blocking call while FetchAndTakeOwnership() is not.
    /// </summary>
    [ObsoleteAttribute("Use CopyOutput instead.", false)]
    public static Tensor Fetch(this IWorker worker)
    {
        return worker.CopyOutput();
    }
    /// <summary>
    /// This method is a blocking call while FetchAndTakeOwnership() is not.
    /// </summary>
    [ObsoleteAttribute("Use CopyOutput instead.", false)]
    public static Tensor Fetch(this IWorker worker, string name)
    {
        return worker.CopyOutput(name);
    }
    #endregion
}

// Deprecated, left here only for backwards compatibility
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

// Deprecated, left here only for backwards compatibility
[ObsoleteAttribute("Use Tensor.ToRenderTexture method instead.", false)]
public class BarracudaTextureUtils
{
    public static void TensorToRenderTexture(Tensor x, RenderTexture target,
                                            int batch = 0, int fromChannel = 0, float scale = 1.0f, float bias = 0f)
    {
        x.ToRenderTexture(target, batch, fromChannel, scale, bias);
    }

    /// <summary>
    /// Create a RenderTexture from a slice/batch of a tensor.
    /// </summary>
    public static RenderTexture TensorToRenderTexture(Tensor x,
                                                int batch = 0, int fromChannel = 0, float scale = 1.0f, float bias = 0f)
    {
        return x.ToRenderTexture(batch, fromChannel, scale, bias);
    }
}


} // namespace Barracuda
