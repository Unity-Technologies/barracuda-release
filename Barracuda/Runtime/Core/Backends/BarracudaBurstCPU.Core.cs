using UnityEngine;
using Unity.Jobs;

namespace Unity.Barracuda {

// BarracudaBurstCPU.Core.cs -- definition of class BurstCPUOps, Pin(), BurstTensorData
// BarracudaBurstCPU.Ops.cs  -- impl. IOps, job schedulers
// BarracudaBurstCPU.Jobs.cs -- impl. jobs

public class BurstTensorData : UnsafeArrayTensorData, IDependableTensorData
{
    private JobHandle m_ReadFence;
    private JobHandle m_WriteFence;
    private bool m_SafeToDispose = true;
    public JobHandle fence { get { return m_ReadFence; }  set { m_ReadFence = value; m_WriteFence = value; m_SafeToDispose = false; } }
    public JobHandle reuse { get { return m_WriteFence; } set { m_WriteFence = value;                      m_SafeToDispose = false; } }

    // creates new array
    public BurstTensorData(int count) : base(count)
    {
    }

    // creates new array
    public BurstTensorData(TensorShape shape) : base(shape)
    {
    }

    // uses shared array
    public BurstTensorData(ArrayTensorData sharedArray) : base(sharedArray)
    {
    }

    // uses shared array
    public BurstTensorData(SharedArrayTensorData sharedArray) : base(sharedArray)
    {
    }

    // uses unsafe array
    public BurstTensorData(UnsafeArrayTensorData unsafeArray) : base(unsafeArray.array, unsafeArray.offset, unsafeArray.count, unsafeArray.m_Readonly)
    {
    }

    ~BurstTensorData()
    {
        if (!m_SafeToDispose)
            D.LogWarning($"Found unreferenced, but undisposed Tensor data that potentially participates in an unfinished job and might lead to hazardous memory overwrites: {ToString()}");
    }

    public override void Dispose()
    {
        try
        {
            CompleteAllPendingOperations();
        }
        catch (UnityException)
        {
            // if Dispose() is called from the finalizer thread, exception will be thrown from Complete()
            // @TODO: rethrow exception, if Dispose was called from the user code
        }

        base.Dispose();
    }

    internal void CompleteAllPendingOperations()
    {
        fence.Complete();
        reuse.Complete();
        m_SafeToDispose = true;
    }

    public override void Reserve(int count)
    {
        if (count > m_Array.Length)
        {
            // going to reallocate memory in base.Reserve()
            // thus need to finish current work
            CompleteAllPendingOperations();
        }

        base.Reserve(count);
    }

    public override void Upload(float[] data, TensorShape shape, int managedBufferStartIndex = 0)
    {
        CompleteAllPendingOperations();
        base.Upload(data, shape, managedBufferStartIndex);
    }

    public override float[] Download(TensorShape shape)
    {
        // Download() as optimization gives direct access to the internal buffer
        // thus need to prepare internal buffer for potential writes
        CompleteAllPendingOperations();
        return base.Download(shape);
    }

    public override float[] SharedAccess(out int offset)
    {
        // SharedAccess() by design gives direct access to the interna
        // thus need to prepare internal buffer for potential writes
        CompleteAllPendingOperations();
        return base.SharedAccess(out offset);
    }

    public override bool ScheduleAsyncDownload(int count)
    {
        return fence.IsCompleted;
    }

    public override string ToString()
    {
        string readyToRead = m_SafeToDispose ? "true": "unknown";
        string readyForReuse = m_SafeToDispose ? "true": "unknown";
        try
        {
            readyToRead = fence.IsCompleted.ToString();
            readyForReuse = reuse.IsCompleted.ToString();
        }
        catch (UnityException) {}
        return string.Format("(CPU burst: {0} length: {1} offset: {2} uploaded: {3} ready-to-read: {4} ready-for-reuse: {5})",
            GetHashCode(), m_Array.Length, m_Offset, m_Count, readyToRead, readyForReuse);
    }
}

public partial class BurstCPUOps : UnsafeArrayCPUOps
{
    private bool m_UseBlas;

    public BurstCPUOps(ITensorAllocator allocator = null)
    : base(allocator)
    {
        m_UseBlas = blas.IsNative();
    }

    new public static BurstTensorData Pin(Tensor X)
    {
        X.FlushCache();

        var onDevice = X.tensorOnDevice as BurstTensorData;
        if (onDevice == null)
        {
            // try to adopt CPU arrays
            var asUnsafeArray = X.tensorOnDevice as UnsafeArrayTensorData;
            var asSharedArray = X.tensorOnDevice as SharedArrayTensorData;
            var asArray = X.tensorOnDevice as ArrayTensorData;
            if (asUnsafeArray != null) X.AttachToDevice(new BurstTensorData(asUnsafeArray));
            else if (asSharedArray != null) X.AttachToDevice(new BurstTensorData(asSharedArray));
            else if (asArray != null) X.AttachToDevice(new BurstTensorData(asArray));
            else
                X.UploadToDevice(new BurstTensorData(X.shape)); // device is not compatible, create new array and upload
        }

        return X.tensorOnDevice as BurstTensorData;
    }

    public override Tensor Prepare(Tensor X)
    {
        Pin(X);
        return X;
    }
}

} // namespace Barracuda
