using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine.Assertions;

namespace Barracuda {

/// <summary>
/// Public interface for Workers. A worker is able to schedule models execution for a given backend.
/// Use `WorkerFactory` to instantiate a worker.
/// </summary>
public interface IWorker : IDisposable
{
    #region Inputs
    /// <summary>
    /// Optional method to prepare network for particular input dimensions
    /// </summary>
    void PrepareForInput(IDictionary<string, TensorShape> inputShapes);
    /// <summary>
    /// Specify single tensor value as the input for the network.
    /// Useful when network has only one input and caller does not need to know input's name.
    /// </summary>
    void SetInput(Tensor x);
    /// <summary>
    /// Specify tensor value for the named input of the network.
    /// </summary>
    void SetInput(string name, Tensor x);
    #endregion

    #region Schedule whole network
    /// <summary>
    /// Non-blocking API that schedules network execution in one go
    /// Remark: This API will only be non-blocking for GPU inference.
    /// </summary>
    void Execute();
    /// <summary>
    /// Non-blocking API that schedules network execution in one go, using the provider tensor as input.
    /// Remark: This API will only be non-blocking for GPU inference.
    /// Useful when network have only one input as input name is not needed.
    /// </summary>
    void Execute(Tensor input);
    /// <summary>
    /// Non-blocking API that schedules network execution in one go, using the provider tensor dictionary for inputs.
    /// Remark: This API will only be non-blocking for GPU inference.
    /// </summary>
    void Execute(IDictionary<string, Tensor> inputs);
    #endregion

    #region Schedule one layer at a time
    /// <summary>
    /// Non-blocking API that schedules network execution one layer at the time.
    /// Remark: This API will only be non-blocking for GPU inference.
    /// Check GetAsyncProgress() for progress.
    /// </summary>
    IEnumerator ExecuteAsync();
    /// <summary>
    /// Non-blocking API that schedules network execution one layer at the time, using the provider tensor as input.
    /// Remark: This API will only be non-blocking for GPU inference.
    /// Useful when network have only one input as input name is not needed.
    /// Check GetAsyncProgress() for progress.
    /// </summary>
    IEnumerator ExecuteAsync(Tensor input);
    /// <summary>
    /// Non-blocking API that schedules network execution one layer at the time, using the provider tensor dictionary for inputs.
    /// Remark: This API will only be non-blocking for GPU inference.
    /// Check GetAsyncProgress() for progress.
    /// </summary>
    IEnumerator ExecuteAsync(IDictionary<string, Tensor> inputs);
    /// <summary>
    /// Wait for completion of part of the network that was scheduled via `ExecuteAsync()`
    /// </summary>
    void WaitForCompletion();
    /// <summary>
    /// Progress of the scheduling, 0.0 = 0%, 1.0 = 100%
    /// </summary>
    float GetAsyncProgress();
    #endregion

    #region Outputs
    /// <summary>
    /// Returns a reference to the first output tensor. This reference will be valid only until the next `Execute()` or `Dispose()` method is called on the worker.
    /// Useful when network has only one output.
    /// IMPORTANT: if you want tensor to outlive the worker, use `CopyOutput()` method or follow with `TakeOwnership()` call on the tensor.
    /// </summary>
    Tensor PeekOutput();
    /// <summary>
    /// Returns a reference to output tensor by name. This reference will be valid only until the next `Execute()` or `Dispose()` method is called on the worker.
    /// IMPORTANT: if you want tensor to outlive the worker, use `CopyOutput()` method or follow with `TakeOwnership()` call on the tensor.
    /// </summary>
    Tensor PeekOutput(string name);
    #endregion

    /// <summary>
    /// Returns a string summary after execution.
    /// </summary>
    string Summary();
}

public static class WorkerExtensions
{
    #region Blocking APIs
    /// <summary>
    /// Returns CPU copy of the first output tensor.
    /// This method is a blocking call and will wait until network execution is completed.
    /// Useful when network has only one output.
    /// </summary>
    public static Tensor CopyOutput(this IWorker worker)
    {
        // @TODO: consider using PeekOutput()+DeepCopy() instead of Unpin()+TakeOwnership()
        var output = worker.PeekOutput();
        output.Unpin(); // unpin will readback to CPU and
                        // give allocator a chance to reuse allocated buffer
        output.TakeOwnership();
        return output;
    }
    /// <summary>
    /// Returns CPU copy of output tensor by name.
    /// This method is a blocking call and will wait until network execution is completed.
    /// </summary>
    public static Tensor CopyOutput(this IWorker worker, string name)
    {
        // @TODO: consider using PeekOutput()+DeepCopy() instead of Unpin()+TakeOwnership()
        var output = worker.PeekOutput(name);
        output.Unpin(); // unpin will readback to CPU and
                        // give allocator a chance to reuse allocated buffer
        output.TakeOwnership();
        return output;
    }

    /// <summary>
    /// Schedules network execution in one go and waits for result to be available.
    /// Useful when network has only one input and caller does not need to know input's name.
    /// </summary>
    public static Tensor ExecuteAndWaitForCompletion(this IWorker worker, Tensor input)
    {
        worker.Execute(input);
        return worker.CopyOutput();
    }
    /// <summary>
    /// Schedules network execution in one go and waits for result to be available.
    /// This method supports multiple inputs.
    /// </summary>
    public static Tensor ExecuteAndWaitForCompletion(this IWorker worker, IDictionary<string, Tensor> inputs)
    {
        worker.Execute(inputs);
        return worker.CopyOutput();
    }
    #endregion
}

/// <summary>
/// Interface for device dependent representation of Tensor data.
/// </summary>
public interface ITensorData : IDisposable
{
    /// <summary>
    /// Reserve uninitialized memory.
    /// </summary>
    void Reserve(int count);
    /// <summary>
    /// Initialize with `data`.
    /// `offset` is the offset where to start the copy in the `data`
    /// `count` is the number of element to copy. If count is -1 (default) number of element will be (data.length - offset).
    /// </summary>
    void Upload(float[] data, int offset = 0, int count = -1);
    /// <summary>
    /// Schedule an asynchronous download from device memory.
    /// `count` is the number of element to readback.
    /// return `true` if the request was successfully scheduled.
    /// </summary>
    bool ScheduleAsyncDownload(int count);
    /// <summary>
    /// Return a copy of the data. This is a blocking call.
    /// `count` is the number of element to readback.
    /// Prefer a call to ScheduleAsyncDownload() before.
    /// </summary>
    float[] Download(int count);
    /// <summary>
    /// Return a copy of the full shared tensorData,
    /// and an offset where this tensorData data is starting.
    /// Prefer a call to ScheduleAsyncDownload() before.
    /// </summary>
    float[] SharedAccess(out int offset);
    /// <summary>
    /// Return the maximum number of element this tensorData can contain.
    /// </summary>
    int GetMaxCount();
}

/// <summary>
/// Object that represent memory (recurrent state) between the executions of a given model.
/// </summary>
public class RecurrentState : IDisposable
{
    private int m_BatchSize = 1;
    private Model m_Model;
    private Tensor[] m_Memories;

    /// <summary>
    /// Constructs recurrent state for a specific model
    /// `model` is the associated model.
    /// `batchSize` has to match the batch dimension of the input tensor(s).
    /// `grabFromInputs` optional dictionary of named tensors that can be used as a memory. If name of the tensor matches the memory, tensor will be removed from the dictionary and used as memory.
    /// </summary>
    public RecurrentState(Model model, int batchSize = 1, Dictionary<string, Tensor> grabFromInputs = null)
    {
        m_BatchSize = batchSize;
        m_Model = model;
        m_Memories = new Tensor[m_Model.memories.Count];

        var index = 0;
        foreach (var memory in m_Model.memories)
        {
            if (grabFromInputs != null && grabFromInputs.ContainsKey(memory.input))
            {
                m_Memories[index++] = grabFromInputs[memory.input];
                grabFromInputs.Remove(memory.input);
            }
            else
            {
                Assert.AreEqual(memory.shape.batch, 1);
                var shape = new TensorShape(memory.shape.batch * batchSize, memory.shape.height, memory.shape.width, memory.shape.channels);
                m_Memories[index++] = new Tensor(shape);
            }
        }
    }

    ~RecurrentState()
    {
        Dispose();
    }

    public virtual void Dispose()
    {
        if (m_Memories == null)
            return;

        foreach (var x in m_Memories)
            x.Dispose();

        m_Memories = null;
    }

    /// <summary>
    /// Returns batch dimension used for the memories.
    /// </summary>
    public int GetBatchSize()
    {
        return m_BatchSize;
    }

    /// <summary>
    /// Internal callback called before the execution of the model.
    /// This callback prepares model for the next iteration according to the memory.
    /// </summary>
    public void BeforeExecution(IWorker worker)
    {
        Assert.AreEqual(m_Model.memories.Count, m_Memories.Length);

        var index = 0;
        foreach (var memory in m_Model.memories)
            worker.SetInput(memory.input, m_Memories[index++]);
    }

    /// <summary>
    /// Internal callback called after execution of the model finished.
    /// This callback stores results of the current iteration in the memory.
    /// </summary>
    public void AfterExecution(IWorker worker)
    {
        Assert.AreEqual(m_Model.memories.Count, m_Memories.Length);

        var index = 0;
        foreach (var memory in m_Model.memories)
        {
            var newTensor = worker.CopyOutput(memory.output);
            Assert.IsTrue(newTensor.tensorOnDevice != m_Memories[index]);
            m_Memories[index].Dispose();
            m_Memories[index] = newTensor;
            index++;
        }
    }
}

/// <summary>
/// Factory to create worker that executes specified model on a particular device  (GPU, CPU, etc) using particular backend.
/// See `IWorker` for usage of the worker itself.
/// </summary>
public class WorkerFactory
{
    /// <summary>
    /// Supported device type
    /// </summary>
    public enum Device
    {
        GPU                 = 1 << 8,
        CPU                 = 1 << 9,
        Auto                = 1 << 15,

        // aliases
        Compute             = GPU,
        CSharp              = CPU,
    }

    /// <summary>
    /// Backend type
    /// </summary>
    public enum Type
    {
        Auto                = 0 | Device.Auto,

        ComputePrecompiled  = 0 | Device.GPU,
        Compute             = 1 | Device.GPU,
        ComputeRef          = 2 | Device.GPU,

        CSharp              = 0 | Device.CPU,
        CSharpRef           = 1 | Device.CPU
    }

    /// <summary>
    /// Create a worker with explicitly specified backend `type` to execute the given `model`.
    /// `type` is backend type to use. For example `WorkerFactory.Type.Compute` specifies the fast GPU path.
    /// `model` is the associated model. See ModelLoader.cs.
    /// `additionalOutputs` are the additional outputs to track but not directly specified by the model.
    /// `trimOutputs` are the outputs not discard even if they are specified by the model.
    /// `verbose` will log scheduling of layers execution to the console.
    /// `compareAgainstType` if different than `type` model will be run on those two backend and the result of every layer will be compared, checking for divergence. Great for debugging, but very slow because of the sync needed.
    /// </summary>
    public static IWorker CreateWorker(Type type, Model model, string[] additionalOutputs, string[] trimOutputs, bool verbose, Type compareAgainstType)
    {
        return BarracudaBackendsFactory.CreateWorker(type, model, additionalOutputs, trimOutputs, verbose, compareAgainstType);
    }

    /// <summary>
    /// Create a worker that will execute `model` using the best backend that is available for a given `device` type.
    /// `model` is the associated model. See ModelLoader.cs.
    /// `additionalOutputs` are the additional outputs to track but not directly specified by the model.
    /// `trimOutputs` are the outputs not discard even if they are specified by the model.
    /// `device` is the device type to run worker on. For example `WorkerFactory.Device.GPU` specifies the fast GPU path.
    /// `verbose` will log scheduling of layers execution to the console (default == false)
    /// </summary>
    public static IWorker CreateWorker(Model model, string[] additionalOutputs, string[] trimOutputs, Device device = Device.Auto, bool verbose = false)
    {
        var type = GetBestTypeForDevice(device);
        return CreateWorker(type, model, additionalOutputs, trimOutputs, verbose, type);
    }

    /// <summary>
    /// Create a worker with explicitly specified backend `type` to execute the given `model`.
    /// `type` is backend type to use. For example `WorkerFactory.Type.Compute` specifies the fast GPU path.
    /// `model` is the associated model. See ModelLoader.cs.
    /// `verbose` will log scheduling of layers execution to the console.
    /// </summary>
    public static IWorker CreateWorker(Type type, Model model, bool verbose)
    {
        return CreateWorker(type, model, null, null, verbose, type);
    }

    /// <summary>
    /// Create a worker with explicitly specified backend `type` to execute the given `model`.
    /// `type` is backend type to use. For example `WorkerFactory.Type.Compute` specifies the fast GPU path.
    /// `model` is the associated model. See ModelLoader.cs.
    /// `additionalOutputs` are the additional outputs to track but not directly specified by the model.
    /// `verbose` will log scheduling of layers execution to the console (default == false)
    /// </summary>
    public static IWorker CreateWorker(Type type, Model model, string[] additionalOutputs, bool verbose = false)
    {
        return CreateWorker(type, model, additionalOutputs, null, verbose, type);
    }

    /// <summary>
    /// Create a worker with explicitly specified backend `type` to execute the given `model`.
    /// `type` is backend type to use. For example `WorkerFactory.Type.Compute` specifies the fast GPU path.
    /// `model` is the associated model. See ModelLoader.cs.
    /// `additionalOutputs` are the additional outputs to track but not directly specified by the model.
    /// `trimOutputs` are the outputs not discard even if they are specified by the model.
    /// `verbose` will log scheduling of layers execution to the console (default == false)
    /// </summary>
    public static IWorker CreateWorker(Type type, Model model, string[] additionalOutputs = null, string[] trimOutputs = null, bool verbose = false)
    {
        return CreateWorker(type, model, additionalOutputs, trimOutputs, verbose, type);
    }

    /// <summary>
    /// Create a worker with explicitly specified backend `type` to execute the given `model`.
    /// `type` is backend type to use. For example `WorkerFactory.Type.Compute` specifies the fast GPU path.
    /// `model` is the associated model. See ModelLoader.cs.
    /// `verbose` will log scheduling of layers execution to the console.
    /// `compareAgainstType` if different than `type` model will be run on those two backend and the result of every layer will be compared, checking for divergence. Great for debugging, but very slow because of the sync needed.
    /// </summary>
    public static IWorker CreateWorker(Type type, Model model, bool verbose, Type compareAgainstType)
    {
        return CreateWorker(type, model, additionalOutputs:null, trimOutputs:null, verbose, compareAgainstType);
    }

    /// <summary>
    /// Create a worker that will execute `model` using the best backend that is available for a given `device` type.
    /// `model` is the associated model. See ModelLoader.cs.
    /// `verbose` will log scheduling of layers execution to the console.
    /// </summary>
    public static IWorker CreateWorker(Model model, bool verbose = false)
    {;
        return CreateWorker(model, Device.Auto, verbose);
    }

    /// <summary>
    /// Create a worker that will execute `model` using the best backend that is available for a given `device` type.
    /// `model` is the associated model. See ModelLoader.cs.
    /// `device` is the preferred device for execution. For example `WorkerFactory.Device.GPU` specifies the fast GPU path.
    /// `verbose` will log scheduling of layers execution to the console.
    /// </summary>
    public static IWorker CreateWorker(Model model, Device device, bool verbose = false)
    {
        return CreateWorker(model, additionalOutputs:null, device, verbose);
    }

    /// <summary>
    /// Create a worker that will execute `model` using the best backend that is available for a given `device` type.
    /// `model` is the associated model. See ModelLoader.cs.
    /// `additionalOutputs` are the additional outputs to track but not directly specified by the model.
    /// `device` is the device type to run worker on. For example `WorkerFactory.Device.GPU` specifies the fast GPU path.
    /// `verbose` will log scheduling of layers execution to the console (default == false)
    /// </summary>
    public static IWorker CreateWorker(Model model, string[] additionalOutputs, Device device = Device.Auto, bool verbose = false)
    {
        return CreateWorker(model, additionalOutputs, trimOutputs:null, device, verbose);
    }

    /// <summary>
    /// Create a worker using the reference CPU backend for the given `model`.
    /// `model` is the associated model. See ModelLoader.cs.
    /// `verbose` will log scheduling of layers execution to the console (default == false)
    /// </summary>
    public static IWorker CreateReferenceCPUWorker(Model model, bool verbose = false)
    {
        return CreateWorker(Type.CSharpRef, model, verbose);
    }

    /// <summary>
    /// Create a worker using the reference GPU backend for the given `model`.
    /// `model` is the associated model. See ModelLoader.cs.
    /// `verbose` will log scheduling of layers execution to the console (default == false)
    /// </summary>
    public static IWorker CreateReferenceComputeWorker(Model model, bool verbose = false)
    {
        return CreateWorker(Type.ComputeRef, model, verbose);
    }

    /// <summary>
    /// Create a worker using the precompiled GPU backend for the given `model`.
    /// `model` is the associated model. See ModelLoader.cs.
    /// `verbose` will log scheduling of layers execution to the console (default == false)
    /// </summary>
    public static IWorker CreateComputeWorker(Model model, bool verbose = false)
    {
        return CreateWorker(Type.ComputePrecompiled, model, verbose);
    }

    /// <summary>
    /// Check if a backend is of a given type.
    /// For example: IsType(Type.CSharpRef, Device.GPU) == true
    /// </summary>
    public static bool IsType(Type type, Device device)
    {
        type = BarracudaBackendsFactory.ResolveAutoType(type);
        if (type == Type.Auto)
            throw new ArgumentException($"Auto type is ambiguous in this context and not supported");
        return ((int)type & (int)device) == (int)device;
    }

    /// <summary>
    /// Returns the best backend type that can run on a `device` given the `model`.
    /// </summary>
    public static Type GetBestTypeForDevice(Device device)
    {
        return BarracudaBackendsFactory.GetBestTypeForDevice(device);
    }

    /// <summary>
    /// Validate if a backend of `type` is supported, otherwise return a fallback type.
    /// </summary>
    public static Type ValidateType(Type type)
    {
        return BarracudaBackendsFactory.ValidateType(type);
    }
}

public static class NNModelExtensions
{
    /// <summary>
    /// Create a worker that will execute `asset` using the best backend that is available for a given `device` type.
    /// This is just a convenience function that internally calls `ModelLoader.Load` followed by ``WorkerFactory.CreateWorker`.
    /// `asset` is the associated NNModel asset.
    /// `device` is the preferred device for execution. For example `WorkerFactory.Device.GPU` specifies the fast GPU path.
    /// `verbose` will log scheduling of layers execution to the console.
    /// </summary>
    public static IWorker CreateWorker(this NNModel asset,
        WorkerFactory.Device device = WorkerFactory.Device.Auto, bool verbose = false)
    {
        var model = ModelLoader.Load(asset);
        return WorkerFactory.CreateWorker(model, device, verbose);
    }

    /// <summary>
    /// Create a worker that will execute `asset` using the best backend that is available for a given `device` type.
    /// This is just a convenience function that internally calls `ModelLoader.Load` followed by ``WorkerFactory.CreateWorker`.
    /// `asset` is the associated NNModel asset.
    /// `additionalOutputs` are the additional outputs to track but not directly specified by the model.
    /// `trimOutputs` are the outputs not discard even if they are specified by the model.
    /// `device` is the device type to run worker on. For example `WorkerFactory.Device.GPU` specifies the fast GPU path.
    /// `verbose` will log scheduling of layers execution to the console (default == false)
    /// </summary>
    public static IWorker CreateWorker(this NNModel asset,
        string[] additionalOutputs, string[] trimOutputs, WorkerFactory.Device device = WorkerFactory.Device.Auto, bool verbose = false)
    {
        var model = ModelLoader.Load(asset);
        return WorkerFactory.CreateWorker(model, additionalOutputs, trimOutputs, device, verbose);
    }
}

} // namespace Barracuda
