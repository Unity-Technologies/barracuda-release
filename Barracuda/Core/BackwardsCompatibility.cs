using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

namespace Barracuda {

// @TODO: deprecate, left here only for backwards compatibility
public static class WorkerExtensions
{
    #region Inputs
    /// <summary>
    /// Specify single tensor value as the input for the network.
    /// Useful when network has only one input and caller does not need to know input's name.
    /// </summary>
    public static void AddInput(this IWorker worker, Tensor x)
    {
        worker.SetInput(x);
    }
    /// <summary>
    /// Specify tensor value for the named input of the network.
    /// </summary>
    public static void AddInput(this IWorker worker, string name, Tensor x)
    {
        worker.SetInput(name, x);
    }
    #endregion

    #region Outputs
    /// <summary>
    /// Returns a reference to tensor from the last layer of the network
    /// Useful when network has only one output.
    /// IMPORTANT: follow with TakeOwnership() call, if you want tensor to outlive worker or make tensor copy with DeepCopy()
    /// see also WorkerExtensions.FetchAndTakeOwnership()
    /// </summary>
    public static Tensor Peek(this IWorker worker)
    {
        return worker.PeekOutput();
    }
    /// <summary>
    /// Returns a reference to tensor by name.
    /// IMPORTANT: follow with TakeOwnership() call, if you want tensor to outlive worker or make tensor copy with DeepCopy()
    /// see also WorkerExtensions.FetchAndTakeOwnership()
    /// </summary>
    public static Tensor Peek(this IWorker worker, string name)
    {
        return worker.PeekOutput(name);
    }
    #endregion


    #region Blocking APIs
    /// <summary>
    /// Schedules network execution in one go and waits for result to be available.
    /// Useful when network has only one input and caller does not need to know input's name.
    /// </summary>
    public static Tensor ExecuteAndWaitForCompletion(this IWorker worker, Tensor input)
    {
        worker.Execute(input);
        return worker.Fetch();
    }
    /// <summary>
    /// Schedules network execution in one go and waits for result to be available.
    /// </summary>
    public static Tensor ExecuteAndWaitForCompletion(this IWorker worker, IDictionary<string, Tensor> inputs)
    {
        worker.Execute(inputs);
        return worker.Fetch();
    }
    #endregion

    #region Non-blocking APIs
    /// <summary>
    /// Returns first output tensor and takes ownership of memory to outlive worker.
    /// Useful when network has only one output.
    /// </summary>
    public static Tensor FetchAndTakeOwnership(this IWorker worker)
    {
        var output = worker.Peek();
        output.TakeOwnership();
        return output;

    }
    /// <summary>
    /// Returns output tensor by name and takes ownership of memory to outlive worker.
    /// </summary>
    public static Tensor FetchAndTakeOwnership(this IWorker worker, string name)
    {
        var output = worker.Peek(name);
        output.TakeOwnership();
        return output;
    }
    #endregion

    // @TODO: rename these APIs, Fetch() name kept for backwards compatibility
    #region Backward compatiblity
    /// <summary>
    /// DEPRECATED: Use FetchAndTakeOwnership() instead.
    /// This method is a blocking call while FetchAndTakeOwnership() is not.
    /// </summary>
    public static Tensor Fetch(this IWorker worker)
    {
        var output = worker.Peek();
        output.Unpin(); // unpin will readback to CPU and
                        // give allocator a chance to reuse allocated buffer
        output.TakeOwnership();
        return output;
    }
    /// <summary>
    /// DEPRECATED: Use FetchAndTakeOwnership() instead.
    /// This method is a blocking call while FetchAndTakeOwnership() is not.
    /// </summary>
    public static Tensor Fetch(this IWorker worker, string name)
    {
        var output = worker.Peek(name);
        output.Unpin(); // unpin will readback to CPU and
                        // give allocator a chance to reuse allocated buffer
        output.TakeOwnership();
        return output;
    }
    #endregion
}

// @TODO: deprecate, left here only for backwards compatibility
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

// @TODO: make internal or remove completely. Left here for backwards compatibility.
public class BarracudaTextureUtils
{
    public static void TensorToRenderTexture(Tensor x, RenderTexture target,
                                            int batch = 0, int fromChannel = 0, float scale = 1.0f, float bias = 0f)
    {
        new ReferenceComputeOps(ComputeShaderSingleton.Instance.referenceKernels).TensorToRenderTexture(x, target, batch, fromChannel, scale, bias);
    }

    /// <summary>
    /// Create a RenderTexture from a slice/batch of a tensor.
    /// </summary>
    public static RenderTexture TensorToRenderTexture(Tensor x,
                                                int batch = 0, int fromChannel = 0, float scale = 1.0f, float bias = 0f)
    {
        var target = new RenderTexture(x.width, x.height, 0);
        TensorToRenderTexture(x, target, batch, fromChannel, scale, bias);
        return target;
    }
}


} // namespace Barracuda
