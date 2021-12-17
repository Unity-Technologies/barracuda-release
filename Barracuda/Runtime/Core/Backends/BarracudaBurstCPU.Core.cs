using System.Threading;
using UnityEngine;
using Unity.Jobs;

namespace Unity.Barracuda {

// BarracudaBurstCPU.Core.cs -- definition of class BurstCPUOps, Pin(), BurstTensorData
// BarracudaBurstCPU.Ops.cs  -- impl. IOps, job schedulers
// BarracudaBurstCPU.Jobs.cs -- impl. jobs

/// <summary>
/// Burst specific internal `Tensor` data storage
/// </summary>
public class BurstTensorData : UnsafeArrayTensorData, IDependableTensorData
{
    private JobHandle m_ReadFence;
    private JobHandle m_WriteFence;
    private bool m_SafeToDispose = true;

    /// <inheritdoc/>
    public JobHandle fence { get { return m_ReadFence; }  set { m_ReadFence = value; m_WriteFence = value; m_SafeToDispose = false; } }

    /// <inheritdoc/>
    public JobHandle reuse { get { return m_WriteFence; } set { m_WriteFence = BurstCPUOps.Dependencies(value, m_WriteFence); m_SafeToDispose = false; } }

    /// <inheritdoc/>
    public unsafe void* rawPtr => array.RawAddressAt(offset);

    /// <summary>
    /// Creates new array
    /// </summary>
    /// <param name="count">count</param>
    public BurstTensorData(int count, DataType dataType) : base(count, dataType)
    {
    }

    /// <summary>
    /// Creates new array
    /// </summary>
    /// <param name="shape">shape</param>
    public BurstTensorData(TensorShape shape, DataType dataType) : base(shape, dataType)
    {
    }

    /// <summary>
    /// Uses shared array
    /// </summary>
    /// <param name="sharedArray">shared array</param>
    public BurstTensorData(ArrayTensorData sharedArray) : base(sharedArray)
    {
    }

    /// <summary>
    /// Uses shared array
    /// </summary>
    /// <param name="sharedArray">shared array</param>
    public BurstTensorData(SharedArrayTensorData sharedArray) : base(sharedArray)
    {
    }

    /// <summary>
    /// Uses unsafe array
    /// </summary>
    /// <param name="unsafeArray">unsafe array</param>
    public BurstTensorData(UnsafeArrayTensorData unsafeArray) : base(unsafeArray.array, unsafeArray.offset, unsafeArray.count, unsafeArray.m_Readonly)
    {
    }

    /// <summary>
    /// Finalizer
    /// </summary>
    ~BurstTensorData()
    {
        if (!m_SafeToDispose)
            D.LogWarning($"Found unreferenced, but undisposed Tensor data that potentially participates in an unfinished job and might lead to hazardous memory overwrites: {ToString()}");
    }

    /// <summary>
    /// Dispose contents
    /// </summary>
    public override void Dispose()
    {
        // It isn't safe to Complete jobs from a finalizer thread, so
        if (Thread.CurrentThread == BurstCPUOps.MainThread)
            CompleteAllPendingOperations();

        base.Dispose();
    }

    internal void CompleteAllPendingOperations()
    {
        fence.Complete();
        reuse.Complete();
        m_SafeToDispose = true;
    }

    /// <summary>
    /// Reserve (allocate) storage for `count` elements
    /// </summary>
    /// <param name="count">count</param>
    public override void Reserve(int count)
    {
        if (count > maxCapacity)
        {
            // going to reallocate memory in base.Reserve()
            // thus need to finish current work
            CompleteAllPendingOperations();
        }

        base.Reserve(count);
    }

    /// <summary>
    /// Upload data to internal storage
    /// </summary>
    /// <param name="data">data</param>
    /// <param name="shape">shape</param>
    /// <param name="managedBufferStartIndex">`data` start index</param>
    public override void Upload(float[] data, TensorShape shape, int managedBufferStartIndex = 0)
    {
        CompleteAllPendingOperations();
        base.Upload(data, shape, managedBufferStartIndex);
    }

    /// <summary>
    /// Return data from internal storage
    /// </summary>
    /// <param name="shape">shape</param>
    /// <returns>managed array</returns>
    public override float[] Download(TensorShape shape)
    {
        // Download() as optimization gives direct access to the internal buffer
        // thus need to prepare internal buffer for potential writes
        CompleteAllPendingOperations();
        return base.Download(shape);
    }

    /// <summary>
    /// Return shared array from internal storage
    /// </summary>
    /// <returns>shared array from internal storage</returns>
    public override BarracudaArray SharedAccess(out int offset)
    {
        // SharedAccess() by design gives direct access to the interna
        // thus need to prepare internal buffer for potential writes
        CompleteAllPendingOperations();
        return base.SharedAccess(out offset);
    }

    /// <summary>
    /// Schedule async internal data download
    /// </summary>
    /// <param name="count">count to download</param>
    /// <returns>`true` if download is completed</returns>
    public override bool ScheduleAsyncDownload(int count)
    {
        return fence.IsCompleted;
    }

    /// <summary>
    /// Object summary as string
    /// </summary>
    /// <returns>object summary</returns>
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
            GetHashCode(), m_Array?.Length, m_Offset, m_Count, readyToRead, readyForReuse);
    }
}

/// <summary>
/// Burst specific implementation of `IOps`
/// </summary>
public partial class BurstCPUOps : UnsafeArrayCPUOps
{
    /// <summary>
    /// Create `BurstCPUOps`
    /// </summary>
    /// <param name="allocator">allocator</param>
    public BurstCPUOps(ITensorAllocator allocator = null)
    : base(allocator)
    {
        if (PreferBLAS == BLAS.Native && !blas.IsNative())
            PreferBLAS = BLAS.Disabled;
    }

    /// <summary>
    /// Pin `Tensor` to Burst backend device, if `uploadCache` is false, data is not uploaded to device
    /// </summary>
    /// <param name="X">`Tensor`</param>
    /// <param name="uploadCache">`bool`</param>
    /// <returns>`BurstTensorData`</returns>
    new public static BurstTensorData Pin(Tensor X, bool uploadCache = true)
    {
        X.FlushCache(uploadCache);

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
            {
                if (uploadCache)
                    X.UploadToDevice(new BurstTensorData(X.shape, X.dataType)); // device is not compatible, create new array and upload
                else
                    X.AllocateOnDevice(new BurstTensorData(X.shape, X.dataType)); // device is not compatible, create new array but do not upload
            }
        }

        return X.tensorOnDevice as BurstTensorData;
    }

    /// <summary>
    /// Prepare `Tensor` for use with Burst backend
    /// </summary>
    /// <param name="X">`Tensor`</param>
    /// <returns>`Tensor`</returns>
    public override Tensor Prepare(Tensor X)
    {
        Pin(X);
        return X;
    }

    public override Tensor PrepareNoAlloc(Tensor X)
    {
        Pin(X, uploadCache: false);
        return X;
    }
}

} // namespace Barracuda
