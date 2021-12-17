using UnityEngine;
using UnityEngine.Assertions;
using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;

namespace Unity.Barracuda {

//#region Job output context helper

internal static class BurstSchedulingHelper
{
    #region Private scheduling helpers with pointer aliasing verification

    private static unsafe JobHandle ScheduleXSBOInternal<T>(T jobData,
        JobHandle fenceBeforeJobStart,
        void* ptrX,
        void* ptrS,
        void* ptrB,
        void* ptrO,
        int arrayLength, int innerloopBatchCount)
        where T : struct, IJobParallelFor, BurstCPUOps.IJobResourceDeclarationXSBO
    {
        T jobDataInternalCopy = jobData;
        jobDataInternalCopy.X = new BurstCPUOps.ReadOnlyMemResource() {ptr = ptrX};
        jobDataInternalCopy.S = new BurstCPUOps.ReadOnlyMemResource() {ptr = ptrS};
        jobDataInternalCopy.B = new BurstCPUOps.ReadOnlyMemResource() {ptr = ptrB};
        jobDataInternalCopy.O = new BurstCPUOps.ReadWriteMemResource() {ptr = ptrO};
        return jobDataInternalCopy.Schedule(arrayLength, innerloopBatchCount, fenceBeforeJobStart);
    }

    private static unsafe JobHandle ScheduleXBOInternal<T>(T jobData,
        JobHandle fenceBeforeJobStart,
        void* ptrX,
        void* ptrB,
        void* ptrO,
        int arrayLength, int innerloopBatchCount)
        where T : struct, IJobParallelFor, BurstCPUOps.IJobResourceDeclarationXBO
    {
        T jobDataInternalCopy = jobData;
        jobDataInternalCopy.X = new BurstCPUOps.ReadOnlyMemResource() {ptr = ptrX};
        jobDataInternalCopy.B = new BurstCPUOps.ReadOnlyMemResource() {ptr = ptrB};
        jobDataInternalCopy.O = new BurstCPUOps.ReadWriteMemResource() {ptr = ptrO};
        return jobDataInternalCopy.Schedule(arrayLength, innerloopBatchCount, fenceBeforeJobStart);
    }

    private static unsafe JobHandle ScheduleXOInternal<T>(T jobData,
        JobHandle fenceBeforeJobStart,
        void* ptrX,
        void* ptrO,
        int arrayLength, int innerloopBatchCount)
        where T : struct, IJobParallelFor, BurstCPUOps.IJobResourceDeclarationXO
    {
        T jobDataInternalCopy = jobData;
        jobDataInternalCopy.X = new BurstCPUOps.ReadOnlyMemResource() {ptr = ptrX};
        jobDataInternalCopy.O = new BurstCPUOps.ReadWriteMemResource() {ptr = ptrO};
        return jobDataInternalCopy.Schedule(arrayLength, innerloopBatchCount, fenceBeforeJobStart);
    }

    private static unsafe JobHandle ScheduleXOInternal<T>(T jobData,
        JobHandle fenceBeforeJobStart,
        void* ptrX,
        void* ptrO)
        where T : struct, IJob, BurstCPUOps.IJobResourceDeclarationXO
    {
        Assert.IsTrue(ptrO != ptrX);
        T jobDataInternalCopy = jobData;
        jobDataInternalCopy.X = new BurstCPUOps.ReadOnlyMemResource() {ptr = ptrX};
        jobDataInternalCopy.O = new BurstCPUOps.ReadWriteMemResource() {ptr = ptrO};
        return jobDataInternalCopy.Schedule(fenceBeforeJobStart);
    }

    private static unsafe JobHandle ScheduleOInternal<T>(T jobData,
        JobHandle fenceBeforeJobStart,
        void* ptrO)
        where T : struct, IJob, BurstCPUOps.IJobResourceDeclarationO
    {
        T jobDataInternalCopy = jobData;
        jobDataInternalCopy.O = new BurstCPUOps.ReadWriteMemResource() {ptr = ptrO};
        return jobDataInternalCopy.Schedule(fenceBeforeJobStart);
    }

    private static unsafe JobHandle ScheduleOInternal<T>(T jobData,
        JobHandle fenceBeforeJobStart,
        void* ptrO,
        int arrayLength, int innerloopBatchCount)
        where T : struct, IJobParallelFor, BurstCPUOps.IJobResourceDeclarationO
    {
        T jobDataInternalCopy = jobData;
        jobDataInternalCopy.O = new BurstCPUOps.ReadWriteMemResource() {ptr = ptrO};
        return jobDataInternalCopy.Schedule(arrayLength, innerloopBatchCount, fenceBeforeJobStart);
    }

    #endregion

    #region Private fencing helper for readability
    private static JobHandle GetFenceBeforeJobStartXSBO(
        IDependableMemoryResource pinX,
        IDependableMemoryResource pinS,
        IDependableMemoryResource pinB,
        IDependableMemoryResource pinO)
    {
        return BurstCPUOps.Dependencies(pinX.fence, pinS.fence, pinB.fence, pinO.reuse);
    }

    private static JobHandle GetFenceBeforeJobStartXBO(
        IDependableMemoryResource pinX,
        IDependableMemoryResource pinB,
        IDependableMemoryResource pinO)
    {
        return BurstCPUOps.Dependencies(pinX.fence, pinB.fence, pinO.reuse);
    }

    private static JobHandle GetFenceBeforeJobStartXO(
        IDependableMemoryResource pinX,
        IDependableMemoryResource pinO)
    {
        return BurstCPUOps.Dependencies(pinX.fence, pinO.reuse);
    }

    private static void SetXSBOFences(this JobHandle jobFence,
        IDependableMemoryResource pinX,
        IDependableMemoryResource pinS,
        IDependableMemoryResource pinB,
        IDependableMemoryResource pinO)
    {
        pinX.reuse = jobFence;
        pinS.reuse = jobFence;
        pinB.reuse = jobFence;
        pinO.fence = jobFence;
    }

    private static void SetXBOFences(this JobHandle jobFence,
        IDependableMemoryResource pinX,
        IDependableMemoryResource pinB,
        IDependableMemoryResource pinO)
    {
        pinX.reuse = jobFence;
        pinB.reuse = jobFence;
        pinO.fence = jobFence;
    }

    private static void SetXOFences(this JobHandle jobFence,
        IDependableMemoryResource pinX,
        IDependableMemoryResource pinO)
    {
        pinX.reuse = jobFence;
        pinO.fence = jobFence;
    }
    #endregion

    #region Immediate scheduling helper
    internal enum FencingHelperMode
    {
        UpdateResourcesFencesOnScheduling,
        CustomResourcesFencesHandling,
    }

    internal static unsafe JobHandle ScheduleXSBO<T>(this T jobData,
        IDependableMemoryResource rX,
        IDependableMemoryResource rS,
        IDependableMemoryResource rB,
        IDependableMemoryResource rO,
        int arrayLength, int innerloopBatchCount,
        FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        where T : struct, IJobParallelFor, BurstCPUOps.IJobResourceDeclarationXSBO
    {
        var fenceBeforeJobStart = GetFenceBeforeJobStartXSBO(rX, rS, rB, rO);

        JobHandle jobFence;
        {
            jobFence = ScheduleXSBOInternal(jobData, fenceBeforeJobStart, rX.rawPtr, rS.rawPtr, rB.rawPtr, rO.rawPtr, arrayLength, innerloopBatchCount);
        }

        if (fencingMode==FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            jobFence.SetXSBOFences(rX, rS, rB, rO);
        }

        return jobFence;
    }

    internal static unsafe JobHandle ScheduleXBO<T>(this T jobData,
        IDependableMemoryResource X,
        IDependableMemoryResource B,
        IDependableMemoryResource O,
        int arrayLength, int innerloopBatchCount,
        FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        where T : struct, IJobParallelFor, BurstCPUOps.IJobResourceDeclarationXBO
    {
        var fenceBeforeJobStart = GetFenceBeforeJobStartXBO(X, B, O);

        JobHandle jobFence;
        {
            jobFence = ScheduleXBOInternal(jobData, fenceBeforeJobStart, X.rawPtr, B.rawPtr, O.rawPtr, arrayLength, innerloopBatchCount);
        }

        if (fencingMode==FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            jobFence.SetXBOFences(X, B, O);
        }

        return jobFence;
    }

    internal static unsafe JobHandle ScheduleO<T>(this T jobData,
        IDependableMemoryResource O,
        FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        where T : struct, IJob, BurstCPUOps.IJobResourceDeclarationO
    {
        var fenceBeforeJobStart = O.reuse;

        JobHandle jobFence;
        {
            jobFence = ScheduleOInternal(jobData, fenceBeforeJobStart, O.rawPtr);
        }

        if (fencingMode==FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            O.fence = jobFence;
        }

        return jobFence;
    }

    internal static unsafe JobHandle ScheduleXO<T>(this T jobData,
        IDependableMemoryResource X,
        IDependableMemoryResource O,
        int arrayLength, int innerloopBatchCount,
        FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        where T : struct, IJobParallelFor, BurstCPUOps.IJobResourceDeclarationXO
    {
        var fenceBeforeJobStart = GetFenceBeforeJobStartXO(X, O);

        JobHandle jobFence;
        {
            jobFence = ScheduleXOInternal(jobData, fenceBeforeJobStart, X.rawPtr, O.rawPtr, arrayLength, innerloopBatchCount);
        }

        if (fencingMode==FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            jobFence.SetXOFences(X, O);
        }

        return jobFence;
    }

    internal static unsafe JobHandle ScheduleO<T>(this T jobData,
        BurstTensorData pinO,
        int offsetO,
        int arrayLength, int innerloopBatchCount,
        FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        where T : struct, IJobParallelFor, BurstCPUOps.IJobResourceDeclarationO
    {
        var fenceBeforeJobStart = pinO.reuse;

        JobHandle jobFence;
        {
            void* ptrO = pinO.array.RawAddressAt(pinO.offset+offsetO);
            jobFence = ScheduleOInternal(jobData, fenceBeforeJobStart, ptrO, arrayLength, innerloopBatchCount);
        }

        if (fencingMode==FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            pinO.fence = jobFence;
        }

        return jobFence;
    }

    internal static unsafe JobHandle ScheduleXO<T>(this T jobData,
        BurstTensorData pinX,
        int offsetX,
        BurstTensorData pinO,
        int offsetO,
        FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        where T : struct, IJob, BurstCPUOps.IJobResourceDeclarationXO
    {
        var fenceBeforeJobStart = GetFenceBeforeJobStartXO(pinX, pinO);

        JobHandle jobFence;
        {
            void* ptrX = pinX.array.RawAddressAt(pinX.offset+offsetX);
            void* ptrO = pinO.array.RawAddressAt(pinO.offset+offsetO);
            jobFence = ScheduleXOInternal(jobData, fenceBeforeJobStart, ptrX, ptrO);
        }

        if (fencingMode==FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            jobFence.SetXOFences(pinX, pinO);
        }

        return jobFence;
    }

    internal static unsafe JobHandle ScheduleXO<T>(this T jobData,
        IDependableMemoryResource X,
        IDependableMemoryResource O,
        FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        where T : struct, IJob, BurstCPUOps.IJobResourceDeclarationXO
    {
        var fenceBeforeJobStart = GetFenceBeforeJobStartXO(X, O);

        JobHandle jobFence;
        {
            jobFence = ScheduleXOInternal(jobData, fenceBeforeJobStart, X.rawPtr, O.rawPtr);
        }

        if (fencingMode==FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            jobFence.SetXOFences(X, O);
        }

        return jobFence;
    }

    #endregion
}

#region Schedulling helper for parrallel jobs

internal struct ParallelJobsContext : IDisposable
{
    internal static Dictionary<IDependableMemoryResource, JobHandle> s_ReadDependencyTracker =
        new Dictionary<IDependableMemoryResource, JobHandle>(100);

    private readonly IDependableMemoryResource outputResource;
    private JobHandle combinedJobFence;

    public ParallelJobsContext(IDependableMemoryResource output)
    {
        outputResource = output;
        combinedJobFence = new JobHandle();
        Assert.AreEqual(0, s_ReadDependencyTracker.Count,
            "s_ReadDependencyTracker should be empty meaning ParrallelJobs was not disposed properly.");
    }

    //For now only CopyStrideJobHelper and tests need ParallelJobsContext. If this code need to be duplicated for more case in the future:
    //- Maybe add generic version by having CopyStrideJobHelper and other helper struct implement an interface (but beware of GC).
    //- Or make ParallelJobsContext partial and code generated by jobs template.
    public JobHandle ScheduleXO(
        BurstCPUOps.CopyStrideJobHelper jobData,//See comment above.
        BurstTensorData pinX, int offsetX,
        BurstTensorData pinO, int offsetO)
    {
        Assert.IsTrue(pinO == outputResource);
        var jobFence = jobData.ScheduleXO(pinX, offsetX, pinO, offsetO, BurstSchedulingHelper.FencingHelperMode.CustomResourcesFencesHandling);
        TrackJobReadDependencies(pinX, jobFence);
        AddJobDependencyToOutputFence(jobFence);
        return jobFence;
    }

    public JobHandle ScheduleXO<T>(
        T jobData,
        BurstTensorData pinX,
        BurstTensorData pinO,
        int arrayLength, int innerloopBatchCount)
        where T : struct, IJobParallelFor, BurstCPUOps.IJobResourceDeclarationXO
    {
        Assert.IsTrue(pinO == outputResource);
        var jobFence = jobData.ScheduleXO(pinX, pinO, arrayLength, innerloopBatchCount, BurstSchedulingHelper.FencingHelperMode.CustomResourcesFencesHandling);
        TrackJobReadDependencies(pinX, jobFence);
        AddJobDependencyToOutputFence(jobFence);
        return jobFence;
    }


    public JobHandle ScheduleXBO<T>(
        T jobData,
        BurstTensorData pinX,
        BurstTensorData pinB,
        BurstTensorData pinO,
        int arrayLength, int innerloopBatchCount)
        where T : struct, IJobParallelFor, BurstCPUOps.IJobResourceDeclarationXBO
    {
        Assert.IsTrue(pinO == outputResource);
        var jobFence = jobData.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerloopBatchCount, BurstSchedulingHelper.FencingHelperMode.CustomResourcesFencesHandling);
        TrackJobReadDependencies(pinX, jobFence);
        TrackJobReadDependencies(pinB, jobFence);
        AddJobDependencyToOutputFence(jobFence);
        return jobFence;
    }

    internal void AddJobDependencyToOutputFence(JobHandle jobFence)
    {
        //Once all jobs writing to O will be done, further jobs will be able to read from O.
        //We combine job fences from all job writing to O here and assign to O.fence in Dispose().
        combinedJobFence = JobHandle.CombineDependencies(combinedJobFence, jobFence);
    }

    internal void TrackJobReadDependencies(IDependableMemoryResource T, JobHandle jobFence)
    {
        //Once all jobs reading from T will be done, further jobs will be able to write to T.
        //We combine job fences from all jobs reading from T here and assign to T.reuse in Dispose().
        if (T != null)
        {
            if (s_ReadDependencyTracker.ContainsKey(T))
                s_ReadDependencyTracker[T] = JobHandle.CombineDependencies(s_ReadDependencyTracker[T], jobFence);
            else
                s_ReadDependencyTracker[T] = jobFence;
        }
    }

    public void Dispose()
    {
        foreach (var key in s_ReadDependencyTracker.Keys)
        {
            key.reuse = s_ReadDependencyTracker[key];
        }
        outputResource.fence = combinedJobFence;
        s_ReadDependencyTracker.Clear();
    }
}

#endregion

#region Memory allocation wrapper usable by job fencing helpers

internal unsafe class FencedMemoryAlloc : IDependableMemoryResource
{
    private JobHandle m_ReadFence;
    private JobHandle m_WriteFence;
    private void* data;
    public void* rawPtr => data;
    public half* halfdata { get { Assert.AreEqual(DataType.Half, type); return (half*) data; } }
    public float* floatdata { get { Assert.AreEqual(DataType.Float, type);return (float*) data; } }
    public DataType type;
    public int elementCount;
    public int elementSize;

    /// <inheritdoc/>
    public JobHandle fence { get { return m_ReadFence; }  set { m_ReadFence = value; m_WriteFence = value; } }

    /// <inheritdoc/>
    public JobHandle reuse { get { return m_WriteFence; } set { m_WriteFence = value; } }

    public void Allocate(int numElement, DataType dataType, int alignment, Allocator allocator)
    {
        m_ReadFence = new JobHandle();
        m_WriteFence = new JobHandle();
        elementCount = numElement;
        elementSize = BarracudaArray.DataItemSize(dataType);
        type = dataType;
        Assert.IsTrue(data == null, "Please call ClearState() when freeing underlying memory.");
        Assert.IsTrue(alignment % elementSize == 0);
        data = UnsafeUtility.Malloc(elementCount * elementSize, alignment, allocator);
        Assert.IsTrue(data != null);
    }

    public void ClearState()
    {
        m_ReadFence = new JobHandle();
        m_WriteFence = new JobHandle();
        elementCount = 0;
        elementSize = 0;
        type = DataType.Float;
        data = null;
    }

    public FencedMemoryAlloc()
    {
        ClearState();
    }
}

#endregion

} // namespace Barracuda
