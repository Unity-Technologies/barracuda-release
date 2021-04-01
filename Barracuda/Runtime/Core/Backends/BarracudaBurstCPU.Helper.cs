using UnityEngine;
using UnityEngine.Assertions;
using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;

namespace Unity.Barracuda {

//#region Job output context helper

internal static class BurstSchedulingHelper
{
    #region Private scheduling helpers with pointer aliasing verification
    private static unsafe JobHandle ScheduleXSBOInternal<T>(T jobData,
        JobHandle fenceBeforeJobStart,
        float* ptrX,
        float* ptrS,
        float* ptrB,
        float* ptrO,
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
        float* ptrX,
        float* ptrB,
        float* ptrO,
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
        float* ptrX,
        float* ptrO,
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
        float* ptrX,
        float* ptrO)
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
        float* ptrO)
        where T : struct, IJob, BurstCPUOps.IJobResourceDeclarationO
    {
        T jobDataInternalCopy = jobData;
        jobDataInternalCopy.O = new BurstCPUOps.ReadWriteMemResource() {ptr = ptrO};
        return jobDataInternalCopy.Schedule(fenceBeforeJobStart);
    }

    private static unsafe JobHandle ScheduleOInternal<T>(T jobData,
        JobHandle fenceBeforeJobStart,
        float* ptrO,
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
        BurstTensorData pinX,
        BurstTensorData pinS,
        BurstTensorData pinB,
        BurstTensorData pinO,
        int arrayLength, int innerloopBatchCount,
        FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        where T : struct, IJobParallelFor, BurstCPUOps.IJobResourceDeclarationXSBO
    {
        var fenceBeforeJobStart = GetFenceBeforeJobStartXSBO(pinX, pinS, pinB, pinO);

        JobHandle jobFence;
        fixed (float*
            ptrX = &pinX.array[pinX.offset],
            ptrS = &pinS.array[pinS.offset],
            ptrB = &pinB.array[pinB.offset],
            ptrO = &pinO.array[pinO.offset])
        {
            jobFence = ScheduleXSBOInternal(jobData, fenceBeforeJobStart, ptrX, ptrS, ptrB, ptrO, arrayLength, innerloopBatchCount);
        }

        if (fencingMode==FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            jobFence.SetXSBOFences(pinX, pinS, pinB, pinO);
        }

        return jobFence;
    }

    internal static unsafe JobHandle ScheduleXSBO<T>(this T jobData,
        BurstTensorData pinX,
        FencedMemoryAlloc pinS,
        FencedMemoryAlloc pinB,
        BurstTensorData pinO,
        int arrayLength, int innerloopBatchCount,
        FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        where T : struct, IJobParallelFor, BurstCPUOps.IJobResourceDeclarationXSBO
    {
        var fenceBeforeJobStart = GetFenceBeforeJobStartXSBO(pinX, pinS, pinB, pinO);

        JobHandle jobFence;
        fixed (float*
            ptrX = &pinX.array[pinX.offset],
            ptrO = &pinO.array[pinO.offset])
        {
            var ptrS = pinS.data;
            var ptrB = pinB.data;
            jobFence = ScheduleXSBOInternal(jobData, fenceBeforeJobStart, ptrX, ptrS, ptrB, ptrO, arrayLength, innerloopBatchCount);
        }

        if (fencingMode==FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            jobFence.SetXSBOFences(pinX, pinS, pinB, pinO);
        }

        return jobFence;
    }

    internal static unsafe JobHandle ScheduleXBO<T>(this T jobData,
        BurstTensorData pinX,
        BurstTensorData pinB,
        BurstTensorData pinO,
        int arrayLength, int innerloopBatchCount,
        FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        where T : struct, IJobParallelFor, BurstCPUOps.IJobResourceDeclarationXBO
    {
        var fenceBeforeJobStart = GetFenceBeforeJobStartXBO(pinX, pinB, pinO);

        JobHandle jobFence;
        fixed (float*
            ptrX = &pinX.array[pinX.offset],
            ptrB = &pinB.array[pinB.offset],
            ptrO = &pinO.array[pinO.offset])
        {
            jobFence = ScheduleXBOInternal(jobData, fenceBeforeJobStart, ptrX, ptrB, ptrO, arrayLength, innerloopBatchCount);
        }

        if (fencingMode==FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            jobFence.SetXBOFences(pinX, pinB, pinO);
        }

        return jobFence;
    }

    internal static unsafe JobHandle ScheduleXBO<T>(this T jobData,
        BurstTensorData pinX,
        FencedMemoryAlloc pinB,
        FencedMemoryAlloc pinO,
        int arrayLength, int innerloopBatchCount,
        FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        where T : struct, IJobParallelFor, BurstCPUOps.IJobResourceDeclarationXBO
    {
        var fenceBeforeJobStart = GetFenceBeforeJobStartXBO(pinX, pinB, pinO);

        JobHandle jobFence;
        fixed (float*
            ptrX = &pinX.array[pinX.offset])
        {
            var ptrB = pinB.data;
            var ptrO = pinO.data;
            jobFence = ScheduleXBOInternal(jobData, fenceBeforeJobStart, ptrX, ptrB, ptrO, arrayLength, innerloopBatchCount);
        }

        if (fencingMode==FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            jobFence.SetXBOFences(pinX, pinB, pinO);
        }

        return jobFence;
    }

    internal static unsafe JobHandle ScheduleXBO<T>(this T jobData,
        FencedMemoryAlloc pinX,
        FencedMemoryAlloc pinB,
        FencedMemoryAlloc pinO,
        int arrayLength, int innerloopBatchCount,
        FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        where T : struct, IJobParallelFor, BurstCPUOps.IJobResourceDeclarationXBO
    {
        var fenceBeforeJobStart = GetFenceBeforeJobStartXBO(pinX, pinB, pinO);

        var ptrX = pinX.data;
        var ptrB = pinB.data;
        var ptrO = pinO.data;
        JobHandle jobFence = ScheduleXBOInternal(jobData, fenceBeforeJobStart, ptrX, ptrB, ptrO, arrayLength, innerloopBatchCount);

        if (fencingMode==FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            jobFence.SetXBOFences(pinX, pinB, pinO);
        }

        return jobFence;
    }

    internal static unsafe JobHandle ScheduleO<T>(this T jobData,
        BurstTensorData pinO,
        FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        where T : struct, IJob, BurstCPUOps.IJobResourceDeclarationO
    {
        var fenceBeforeJobStart = pinO.reuse;

        JobHandle jobFence;
        fixed (float* ptrO = &pinO.array[pinO.offset])
        {
            jobFence = ScheduleOInternal(jobData, fenceBeforeJobStart, ptrO);
        }

        if (fencingMode==FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            pinO.fence = jobFence;
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
        fixed (float* ptrO = &pinO.array[pinO.offset])
        {
            jobFence = ScheduleOInternal(jobData, fenceBeforeJobStart, ptrO+offsetO, arrayLength, innerloopBatchCount);
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
        fixed (float*
            ptrX = &pinX.array[pinX.offset],
            ptrO = &pinO.array[pinO.offset])
        {
            jobFence = ScheduleXOInternal(jobData, fenceBeforeJobStart, ptrX+offsetX, ptrO+offsetO);
        }

        if (fencingMode==FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            jobFence.SetXOFences(pinX, pinO);
        }

        return jobFence;
    }

    internal static unsafe JobHandle ScheduleXO<T>(this T jobData,
        BurstTensorData pinX,
        BurstTensorData pinO,
        int arrayLength, int innerloopBatchCount,
        FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        where T : struct, IJobParallelFor, BurstCPUOps.IJobResourceDeclarationXO
    {
        var fenceBeforeJobStart = GetFenceBeforeJobStartXO(pinX, pinO);

        JobHandle jobFence;
        fixed (float*
            ptrX = &pinX.array[pinX.offset],
            ptrO = &pinO.array[pinO.offset])
        {
            jobFence = ScheduleXOInternal(jobData, fenceBeforeJobStart, ptrX, ptrO, arrayLength, innerloopBatchCount);
        }

        if (fencingMode==FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            jobFence.SetXOFences(pinX, pinO);
        }

        return jobFence;
    }

    internal static unsafe JobHandle ScheduleXO<T>(this T jobData,
        FencedMemoryAlloc pinX,
        FencedMemoryAlloc pinO,
        int arrayLength, int innerloopBatchCount,
        FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        where T : struct, IJobParallelFor, BurstCPUOps.IJobResourceDeclarationXO
    {
        var fenceBeforeJobStart = GetFenceBeforeJobStartXO(pinX, pinO);

        var ptrX = pinX.data;
        var ptrO = pinO.data;
        JobHandle jobFence = ScheduleXOInternal(jobData, fenceBeforeJobStart, ptrX, ptrO, arrayLength, innerloopBatchCount);

        if (fencingMode==FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            jobFence.SetXOFences(pinX, pinO);
        }

        return jobFence;
    }

    internal static unsafe JobHandle ScheduleXO<T>(this T jobData,
        BurstTensorData pinX,
        BurstTensorData pinO,
        FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        where T : struct, IJob, BurstCPUOps.IJobResourceDeclarationXO
    {
        var fenceBeforeJobStart = GetFenceBeforeJobStartXO(pinX, pinO);

        JobHandle jobFence;
        fixed (float*
            ptrX = &pinX.array[pinX.offset],
            ptrO = &pinO.array[pinO.offset])
        {
            jobFence = ScheduleXOInternal(jobData, fenceBeforeJobStart, ptrX, ptrO);
        }

        if (fencingMode==FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            jobFence.SetXOFences(pinX, pinO);
        }

        return jobFence;
    }

    internal static unsafe JobHandle ScheduleXO<T>(this T jobData,
        BurstTensorData pinX,
        FencedMemoryAlloc pinO,
        int arrayLength, int innerloopBatchCount,
        FencingHelperMode fencingMode=FencingHelperMode.UpdateResourcesFencesOnScheduling)
        where T : struct, IJobParallelFor, BurstCPUOps.IJobResourceDeclarationXO
    {
        var fenceBeforeJobStart = GetFenceBeforeJobStartXO(pinX, pinO);

        JobHandle jobFence;
        fixed (float*
            ptrX = &pinX.array[pinX.offset])
        {
            var ptrO = pinO.data;
            jobFence = ScheduleXOInternal(jobData, fenceBeforeJobStart, ptrX, ptrO, arrayLength, innerloopBatchCount);
        }

        if (fencingMode==FencingHelperMode.UpdateResourcesFencesOnScheduling)
        {
            jobFence.SetXOFences(pinX, pinO);
        }

        return jobFence;
    }

    internal static void ScheduleXO<T>(this T jobData, Tensor X, Tensor O)
        where T : struct, IJob, BurstCPUOps.IJobResourceDeclarationXO
    {
        var pinX = BurstCPUOps.Pin(X);
        var pinO = BurstCPUOps.Pin(O);
        jobData.ScheduleXO(pinX, pinO);
    }

    internal static void ScheduleXO<T>(this T jobData, Tensor X, Tensor O,
        int arrayLength, int innerloopBatchCount)
        where T : struct, IJobParallelFor, BurstCPUOps.IJobResourceDeclarationXO
    {
        var pinX = BurstCPUOps.Pin(X);
        var pinO = BurstCPUOps.Pin(O);
        jobData.ScheduleXO(pinX, pinO, arrayLength, innerloopBatchCount);
    }

    internal static void ScheduleXBO<T>(this T jobData, Tensor X, Tensor B, Tensor O,
        int arrayLength, int innerloopBatchCount)
        where T : struct, IJobParallelFor, BurstCPUOps.IJobResourceDeclarationXBO
    {
        var pinX = BurstCPUOps.Pin(X);
        var pinB = BurstCPUOps.Pin(B);
        var pinO = BurstCPUOps.Pin(O);
        jobData.ScheduleXBO(pinX, pinB, pinO, arrayLength, innerloopBatchCount);
    }

    internal static void ScheduleXSBO<T>(this T jobData, Tensor X, Tensor S, Tensor B, Tensor O,
        int arrayLength, int innerloopBatchCount)
        where T : struct, IJobParallelFor, BurstCPUOps.IJobResourceDeclarationXSBO
    {
        var pinX = BurstCPUOps.Pin(X);
        var pinS = BurstCPUOps.Pin(S);
        var pinB = BurstCPUOps.Pin(B);
        var pinO = BurstCPUOps.Pin(O);
        jobData.ScheduleXSBO(pinX, pinS, pinB, pinO, arrayLength, innerloopBatchCount);
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

    public void ScheduleO<T>(
        T jobData,
        BurstTensorData pinO, int offsetO,
        int arrayLength, int innerloopBatchCount)
        where T : struct, IJobParallelFor, BurstCPUOps.IJobResourceDeclarationO
    {
        Assert.IsTrue(pinO == outputResource);
        var jobFence = jobData.ScheduleO(pinO, offsetO, arrayLength, innerloopBatchCount, BurstSchedulingHelper.FencingHelperMode.CustomResourcesFencesHandling);
        AddJobDependencyToOutputFence(jobFence);
    }

    public void ScheduleXO<T>(
        T jobData,
        BurstTensorData pinX, int offsetX,
        BurstTensorData pinO, int offsetO)
        where T : struct, IJob, BurstCPUOps.IJobResourceDeclarationXO
    {
        Assert.IsTrue(pinO == outputResource);
        var jobFence = jobData.ScheduleXO(pinX, offsetX, pinO, offsetO, BurstSchedulingHelper.FencingHelperMode.CustomResourcesFencesHandling);
        TrackJobReadDependencies(pinX, jobFence);
        AddJobDependencyToOutputFence(jobFence);
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

    public JobHandle ScheduleXSBO<T>(
        T jobData,
        BurstTensorData pinX,
        BurstTensorData pinS,
        BurstTensorData pinB,
        BurstTensorData pinO,
        int arrayLength, int innerloopBatchCount)
        where T : struct, IJobParallelFor, BurstCPUOps.IJobResourceDeclarationXSBO
    {
        Assert.IsTrue(pinO == outputResource);
        var jobFence = jobData.ScheduleXSBO(pinX, pinS, pinB, pinO, arrayLength, innerloopBatchCount, BurstSchedulingHelper.FencingHelperMode.CustomResourcesFencesHandling);
        TrackJobReadDependencies(pinX, jobFence);
        TrackJobReadDependencies(pinS, jobFence);
        TrackJobReadDependencies(pinB, jobFence);
        AddJobDependencyToOutputFence(jobFence);
        return jobFence;
    }

    private void AddJobDependencyToOutputFence(JobHandle jobFence)
    {
        //Once all jobs writing to O will be done, further jobs will be able to read from O.
        //We combine job fences from all job writing to O here and assign to O.fence in Dispose().
        combinedJobFence = JobHandle.CombineDependencies(combinedJobFence, jobFence);
    }

    private void TrackJobReadDependencies(IDependableMemoryResource T, JobHandle jobFence)
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
    public float* data;

    /// <inheritdoc/>
    public JobHandle fence { get { return m_ReadFence; }  set { m_ReadFence = value; m_WriteFence = value; } }

    /// <inheritdoc/>
    public JobHandle reuse { get { return m_WriteFence; } set { m_WriteFence = value; } }

    public void Malloc(long size, int alignment, Allocator allocator)
    {
        Assert.IsTrue(data == null, "Please call ClearState() when freeing underlying memory.");
        m_ReadFence = new JobHandle();
        m_WriteFence = new JobHandle();
        data = (float*)UnsafeUtility.Malloc(size, alignment, allocator);
    }

    public void ClearState()
    {
        m_ReadFence = new JobHandle();
        m_WriteFence = new JobHandle();
        data = null;
    }

    public FencedMemoryAlloc()
    {
        ClearState();
    }

    public FencedMemoryAlloc(long size, int alignment, Allocator allocator)
    {
        Malloc(size, alignment, allocator);
    }
}

#endregion

} // namespace Barracuda
