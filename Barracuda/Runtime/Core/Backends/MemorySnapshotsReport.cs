#if ENABLE_BARRACUDA_STATS

using System.Collections.Generic;
using System.Text;

namespace Unity.Barracuda {

public class TensorDataMemoryInfo
{
    public int UniqueId { get; }
    public int MaxBytes  { get; }
    public bool InUse  { get; }
    public bool IsGPUMem  { get; }

    internal TensorDataMemoryInfo(ITensorDataStatistics tensorDataStatistics)
    {
        UniqueId = tensorDataStatistics.uniqueId;
        MaxBytes = tensorDataStatistics.maxCapacity * sizeof(float);
        InUse = tensorDataStatistics.inUse;
        IsGPUMem = tensorDataStatistics.isGPUMem;
    }

    public override string ToString()
    {
        return $"TensorData of maxBytes {MaxBytes}, inUse:{InUse}, onGPU:{IsGPUMem}, uniqueId:{UniqueId}";
    }
}

public class TempMemoryInfo
{
    public int UniqueId { get; }
    public string Name { get; }
    public long TotalBytes { get; }
    public bool IsGPUMem  { get; }

    internal TempMemoryInfo(TempMemoryStatistics tempMemoryStatistics)
    {
        UniqueId = tempMemoryStatistics.uniqueId;
        Name = tempMemoryStatistics.name;
        TotalBytes = tempMemoryStatistics.size;
        IsGPUMem = tempMemoryStatistics.isGPUMem;
    }

    public override string ToString()
    {
        return $"Temp memory '{Name}' of totalBytes {TotalBytes}";
    }
}

public class AllocatorMemoryInfo
{
    public int UniqueId { get; }
    public string Name { get; }
    public long UsedBytes { get; }
    public long BusyBytes { get; }
    public long FreeBytes { get; }
    public long TotalBytes { get; }
    public List<TensorDataMemoryInfo> TensorDatasMemoryInfo { get; }
    public List<TensorMemoryInfo> TensorsMemoryInfo { get; }
    public long BytesLostToFragmentation => BusyBytes - UsedBytes;

    internal AllocatorMemoryInfo(IAllocatorStatistics allocatorStatistics)
    {
        UniqueId = allocatorStatistics.uniqueId;
        Name = allocatorStatistics.name;
        UsedBytes = allocatorStatistics.usedBytes;
        BusyBytes = allocatorStatistics.busyBytes;
        FreeBytes = allocatorStatistics.freeBytes;
        TotalBytes = allocatorStatistics.totalBytes;
        TensorDatasMemoryInfo = new List<TensorDataMemoryInfo>();
        foreach (var tensorDataStatistics in allocatorStatistics.GetTensorDatasStatistics())
        {
            TensorDatasMemoryInfo.Add(new TensorDataMemoryInfo(tensorDataStatistics));
        }
        TensorsMemoryInfo = new List<TensorMemoryInfo>();
        foreach (var tensorStatistics in allocatorStatistics.GetTensorsStatistics())
        {
            TensorsMemoryInfo.Add(new TensorMemoryInfo(tensorStatistics));
        }
    }

    public override string ToString()
    {
        return $"Allocator '{Name}' of totalBytes {TotalBytes}, usedBytes:{UsedBytes}, lostToFragmentation:{BytesLostToFragmentation}, free:{FreeBytes}";
    }
}

public class TensorMemoryInfo
{
    public int UniqueId { get; }
    public string Name { get; }
    public TensorShape Shape { get; }
    public int CacheBytes { get; }
    public TensorDataMemoryInfo tensorDataMemoryInfo { get; }

    internal TensorMemoryInfo(ITensorStatistics tensorStatistics)
    {
        UniqueId = tensorStatistics.uniqueId;
        Name = tensorStatistics.name;
        Shape = tensorStatistics.shape;
        CacheBytes = tensorStatistics.cacheBytes;
        var tensorDataStats = tensorStatistics.GetTensorDataStatistics();
        if (tensorDataStats != null)
            tensorDataMemoryInfo = new TensorDataMemoryInfo(tensorDataStats);
    }

    public override string ToString()
    {
        var tensorDataStr = (tensorDataMemoryInfo != null) ? tensorDataMemoryInfo.ToString() : "";
        return $"Tensor: {Name} of shape {Shape.ToString()}, cacheBytes: {CacheBytes} (data: {tensorDataStr})";
    }
}

public class MemorySnapshotReport
{
    public string ContextType { get; }
    public string ContextName  { get; }
    public List<TensorMemoryInfo> TensorsMemoryInfo  { get; }
    public List<AllocatorMemoryInfo> AllocatorsMemoryInfo  { get; }
    public List<TempMemoryInfo> TempMemoriesInfo  { get; }

    internal MemorySnapshotReport(IOps ops, IVarsStatistics vars, string context, Layer layer)
    {
        ContextType = context;
        ContextName = "";
        if (layer != null)
        {
            ContextType += ": " + layer.type + ((layer.type == Layer.Type.Activation) ? ("." + layer.activation) : "");
            ContextName += layer.name;
        }

        TensorsMemoryInfo = new List<TensorMemoryInfo>();
        AllocatorsMemoryInfo = new List<AllocatorMemoryInfo>();
        TempMemoriesInfo = new List<TempMemoryInfo>();

        foreach (var allocatorsStatistic in vars.GetAllocatorsStatistics())
        {
            AllocatorsMemoryInfo.Add(new AllocatorMemoryInfo(allocatorsStatistic));
        }

        foreach (var tensorStatistic in vars.GetTensorsStatistics())
        {
            TensorsMemoryInfo.Add(new TensorMemoryInfo(tensorStatistic));
        }

        foreach (var tempMemoryStatistic in ops.GetTempMemoryStatistics())
        {
            TempMemoriesInfo.Add(new TempMemoryInfo(tempMemoryStatistic));
        }
    }
}

public class MemorySnapshotsReport
{
    public List<MemorySnapshotReport> MemorySnapshotsReports { get; private set; }

    public MemorySnapshotsReport()
    {
        Reset();
    }

    public void Reset()
    {
        MemorySnapshotsReports = new List<MemorySnapshotReport>();
    }

    public void TakeMemorySnapshot(IOps ops, IVars vars, string context, Layer layer)
    {
        var varsWithStatistics = vars as IVarsStatistics;
        if (varsWithStatistics == null)
            return;

        MemorySnapshotsReports.Add(new MemorySnapshotReport(ops, varsWithStatistics, context, layer));
    }

    public MemoryPeakSummary GenerateStringReport(StringBuilder stringBuilder, bool spreadSheetFormat)
    {
        stringBuilder.Append("**************** MEMORY SNAPSHOTS REPORTS - START ****************\n");
        stringBuilder.Append($"Number of snapshots : {MemorySnapshotsReports.Count}\n\n");

        var memoryPeakSummary = MemoryAndExecutionReportHelper.GenerateStringReport(stringBuilder, MemorySnapshotsReports, spreadSheetFormat);
        stringBuilder.Append("**************** MEMORY SNAPSHOTS REPORTS - STOP ****************\n");
        return memoryPeakSummary;
    }

    public override string ToString()
    {
        var stringBuilder = new StringBuilder(10000);
        GenerateStringReport(stringBuilder, spreadSheetFormat:false);
        return stringBuilder.ToString();
    }
}

} // namespace Unity.Barracuda

#endif //ENABLE_BARRACUDA_STATS
