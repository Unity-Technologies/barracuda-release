#if ENABLE_BARRACUDA_STATS

using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Assertions;

namespace Unity.Barracuda {

public readonly struct DispatchInfo
{
    public readonly string backend;
    public readonly string kernel;
    public readonly int workItemsX;
    public readonly int workItemsY;
    public readonly int workItemsZ;

    public DispatchInfo(string backend, string kernel, int workItemsX, int workItemsY, int workItemsZ)
    {
        this.backend = backend;
        this.kernel = kernel;
        this.workItemsX = workItemsX;
        this.workItemsY = workItemsY;
        this.workItemsZ = workItemsZ;
    }

    public override string ToString()
    {
        return $"{backend}:{kernel}({workItemsX},{workItemsY},{workItemsZ})";
    }

    internal static DispatchInfo CreateFromComputeFunc(ComputeFunc computeFunc, int x, int y, int z)
    {
        var backend = computeFunc.computeShaderContext==ComputeShaderContext.Reference?"REF":"OPT";
        return new DispatchInfo(backend, computeFunc.kernelName, x, y, z);
    }
}

public class LayerExecutionReport
{
    public string LayerType { get; }
    public string LayerName { get; }
    public string DispatchInfos { get; private set; }
    public string Summary { get; private set; }
    public long NumAlu { get; private set; }
    public long NumBytes { get; private set; }

    internal LayerExecutionReport(Layer l)
    {
        LayerType = l.type + ((l.type == Layer.Type.Activation) ? ("." + l.activation) : "");
        LayerName = l.name;
        Summary = "";
        DispatchInfos = "";
        NumAlu = 0;
        NumBytes = 0;
    }

    internal void SetSummary(string message)
    {
        Summary = message;
    }

    internal void SetALUAndMemStats(long alu, long bytes)
    {
        NumAlu = alu;
        NumBytes = bytes;
    }

    internal void AddDispatch(DispatchInfo dispatchInfo)
    {
        if (DispatchInfos.Length != 0)
            DispatchInfos = DispatchInfos + " / ";
        DispatchInfos = DispatchInfos + dispatchInfo;
    }
}

public class ModelExecutionReport
{
    public List<LayerExecutionReport> CompletedLayerExecutionReports { get; }
    public LayerExecutionReport CurrentLayerExecutionReport { get; private set; }

    internal ModelExecutionReport()
    {
        CompletedLayerExecutionReports = new List<LayerExecutionReport>();
        CurrentLayerExecutionReport = null;
    }

    internal void LayerExecutionStarted(Layer layer)
    {
        Assert.IsNull(CurrentLayerExecutionReport);
        CurrentLayerExecutionReport = new LayerExecutionReport(layer);
    }

    internal void LayerExecutionCompleted()
    {
        CompletedLayerExecutionReports.Add(CurrentLayerExecutionReport);
        CurrentLayerExecutionReport = null;
    }

    internal void SetLayerSummary(string message)
    {
        Assert.IsNotNull(CurrentLayerExecutionReport);
        CurrentLayerExecutionReport.SetSummary(message);
    }

    internal void SetLayerALUAndMemStats(long alu, long bytes)
    {
        Assert.IsNotNull(CurrentLayerExecutionReport);
        CurrentLayerExecutionReport.SetALUAndMemStats(alu, bytes);
    }

    internal void AddLayerDispatch(DispatchInfo dispatchInfo)
    {
        Assert.IsNotNull(CurrentLayerExecutionReport);
        CurrentLayerExecutionReport.AddDispatch(dispatchInfo);
    }
}

public class ModelExecutionsReporter : IModelExecutionsReporter
{
    //Tabs separator make importing into spreadsheet software easy.
    public static readonly string SpreadSheetFieldSeparator = "\t";
    public static readonly string TextFormatFieldSeparator = " / ";
    public static readonly string TextIndentation = "    ";

    public List<ModelExecutionReport> CompletedModelExecutionReports { get; private set; }
    public ModelExecutionReport CurrentModelExecutionReport { get; private set; }
    public MemorySnapshotsReport MemorySnapshotsReport { get; private set; }

    public ModelExecutionsReporter()
    {
        Reset();
    }

    public void Reset()
    {
        CompletedModelExecutionReports = new List<ModelExecutionReport>();
        CurrentModelExecutionReport = null;
        MemorySnapshotsReport = new MemorySnapshotsReport();
    }

    public void TakeMemorySnapshot(IOps ops, IVars vars, string context, Layer layer)
    {
        MemorySnapshotsReport.TakeMemorySnapshot(ops, vars, context, layer);
    }

    public void ModelExecutionStarted()
    {
        Assert.IsNull(CurrentModelExecutionReport);
        CurrentModelExecutionReport = new ModelExecutionReport();
    }

    public void ModelExecutionCompleted()
    {
        CompletedModelExecutionReports.Add(CurrentModelExecutionReport);
        CurrentModelExecutionReport = null;
    }

    public void LayerExecutionStarted(Layer layer)
    {
        Assert.IsNotNull(CurrentModelExecutionReport);
        CurrentModelExecutionReport.LayerExecutionStarted(layer);
    }

    public void LayerExecutionCompleted()
    {
        Assert.IsNotNull(CurrentModelExecutionReport);
        CurrentModelExecutionReport.LayerExecutionCompleted();
    }

    public void SetLayerSummary(string message)
    {
        Assert.IsNotNull(CurrentModelExecutionReport);
        CurrentModelExecutionReport.SetLayerSummary(message);
    }

    public void SetLayerALUAndMemStats(long alu, long bytes)
    {
        Assert.IsNotNull(CurrentModelExecutionReport);
        CurrentModelExecutionReport.SetLayerALUAndMemStats(alu, bytes);
    }

    public void AddLayerDispatch(DispatchInfo dispatchInfo)
    {
        Assert.IsNotNull(CurrentModelExecutionReport);
        CurrentModelExecutionReport.AddLayerDispatch(dispatchInfo);
    }

    public override string ToString()
    {
        return GenerateStringReport(out var memoryPeakSummary, false);
    }

    public string GenerateStringReport(out MemoryPeakSummary memoryPeakSummary, bool spreadsheetFormat)
    {
        var stringBuilder = new StringBuilder(1000);

        //**************** MODEL EXECUTIONS REPORT - START ****************
        stringBuilder.Append($"**************** MODEL EXECUTIONS REPORT - START ****************\n");
        stringBuilder.Append($"Number of completed executions : {CompletedModelExecutionReports.Count}\n");
        if (CurrentModelExecutionReport != null)
            stringBuilder.Append("Warning: last model execution was not completed. It will be logged, but information might be incomplete.\n");
        stringBuilder.Append("\n");
        int i = 0;
        for (; i < CompletedModelExecutionReports.Count; ++i)
        {
            stringBuilder.Append($"--------- Execution index : {i} - START ---------\n");
            MemoryAndExecutionReportHelper.GenerateStringReport(stringBuilder, CompletedModelExecutionReports[i], spreadsheetFormat);
            stringBuilder.Append($"--------- Execution index : {i} - STOP ---------\n");
            stringBuilder.Append("\n");
        }
        if (CurrentModelExecutionReport != null)
        {
            stringBuilder.Append($"--------- Uncompleted execution - START ---------\n");
            MemoryAndExecutionReportHelper.GenerateStringReport(stringBuilder, CurrentModelExecutionReport, spreadsheetFormat);
            stringBuilder.Append($"--------- Uncompleted execution - STOP ---------\n");
            stringBuilder.Append("\n");
        }
        stringBuilder.Append($"**************** MODEL EXECUTION REPORT - STOP ****************\n");
        stringBuilder.Append("\n");
        //**************** MODEL EXECUTIONS REPORT - STOP ****************

        //**************** MEMORY SNAPSHOTS REPORTS - START ****************
        memoryPeakSummary = MemorySnapshotsReport.GenerateStringReport(stringBuilder, spreadsheetFormat);
        //**************** MEMORY SNAPSHOTS REPORTS - STOP ****************

        return stringBuilder.ToString();
    }

    #if UNITY_EDITOR
    public static string ToTextFile(IModelExecutionsReporter report, bool spreadsheetFormat, out MemoryPeakSummary memoryPeakSummary, string filename = null)
    {
        string stringToSave = report.GenerateStringReport(out memoryPeakSummary, spreadsheetFormat);
        string fullPath = Application.temporaryCachePath;
        if (filename == null)
        {
            fullPath = Path.Combine(fullPath, "ModelExecutionReport");
            fullPath = Path.ChangeExtension(fullPath, "txt");
        }
        else
        {
            fullPath = Path.Combine(fullPath, filename);
        }
        File.WriteAllText(fullPath, stringToSave);
        return fullPath;
    }
    #endif
}

} // namespace Unity.Barracuda

#endif //ENABLE_BARRACUDA_STATS
