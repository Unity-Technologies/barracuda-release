using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEngine.Assertions;

namespace Unity.Barracuda {

internal static class MemoryAndExecutionReportHelper
{
    public static void GenerateStringReport(StringBuilder stringBuilder, ModelExecutionReport modelExecutionReport,
        bool spreadSheetFormat)
    {
        stringBuilder.Append($"Number of completed layers : {modelExecutionReport.CompletedLayerExecutionReports.Count}\n");
        if (modelExecutionReport.CurrentLayerExecutionReport != null)
            stringBuilder.Append("Warning: last layer was not completed. It will be logged, but it's information might be incomplete or erroneous.\n");
        stringBuilder.Append("\n");

        List<LayerExecutionReport> allLayerReports = new List<LayerExecutionReport>();
        allLayerReports.AddRange(modelExecutionReport.CompletedLayerExecutionReports);
        if (modelExecutionReport.CurrentLayerExecutionReport != null)
            allLayerReports.Add(modelExecutionReport.CurrentLayerExecutionReport);

        var layerExecutionViews = GenerateExecutionViews(allLayerReports, modelExecutionReport.CompletedLayerExecutionReports.Count);
        GenerateReportForViews(stringBuilder, layerExecutionViews, spreadSheetFormat, "", false);
    }

    public static void GenerateStringReport(StringBuilder stringBuilder, List<MemorySnapshotReport> memorySnapshots,
        bool spreadSheetFormat)
    {
        CollectAllAsFirstSeen(in memorySnapshots,
            out var allTensorAsFirstSeen,
            out var allAllocatorAsFirstSeen,
            out var allTensorDataAsFirstSeen);

        var summaryViews = GenerateSummaryViews(memorySnapshots, allTensorAsFirstSeen, allTensorDataAsFirstSeen);
        GenerateHeaderForSummaryViews(stringBuilder, summaryViews, spreadSheetFormat);
        GenerateReportForViews(stringBuilder, summaryViews, spreadSheetFormat, "Tensors allocation and deallocation (diff from previous snapshot):", isSummaryView:true);
        stringBuilder.Append("\n");
        stringBuilder.Append("\n");

        var tensorViews = GenerateTensorsViews(memorySnapshots, allTensorAsFirstSeen);
        GenerateHeaderForTensorViews(stringBuilder, tensorViews, spreadSheetFormat);
        GenerateReportForViews(stringBuilder, tensorViews, spreadSheetFormat, "All Tensors:", isSummaryView:false);
        stringBuilder.Append("\n");
        stringBuilder.Append("\n");

        var allocatorViews = GenerateAllocatorViews(memorySnapshots, allAllocatorAsFirstSeen);
        GenerateHeaderForAllocatorsViews(stringBuilder, allocatorViews, spreadSheetFormat);
        GenerateReportForViews(stringBuilder, allocatorViews, spreadSheetFormat, "All Allocators:", isSummaryView:false);
        stringBuilder.Append("\n");
        stringBuilder.Append("\n");

        var tensorDatasViews = GenerateTensorDatasViews(memorySnapshots, allTensorDataAsFirstSeen);
        GenerateHeaderForTensorDatasViews(stringBuilder, tensorDatasViews, spreadSheetFormat);
        GenerateReportForViews(stringBuilder, tensorDatasViews, spreadSheetFormat, "All TensorDatas:", isSummaryView:false);
        stringBuilder.Append("\n");
        stringBuilder.Append("\n");
    }

    #region `Internal data format` declaration
    private class SnapshotFields
    {
        public readonly string[] Titles;
        public readonly Dictionary<string, string> Items;

        public SnapshotFields(string[] titles)
        {
            Titles = titles;
            Items = new Dictionary<string, string>();
            foreach (var title in titles)
            {
                Items[title] = "";
            }
        }

        public string this[string title]
        {
            set {
                Assert.IsTrue(Items.ContainsKey(title));
                Assert.IsTrue(Items[title] == "");
                Items[title] = value;
            }
            get => Items[title];
        }

        public void AddTitlesToReport(StringBuilder stringBuilder, string separator)
        {
            foreach (var title in Titles)
            {
                stringBuilder.Append(title);
                stringBuilder.Append(separator);
            }
        }

        public void AddValuesToReport(StringBuilder stringBuilder, string separator)
        {
            foreach (var title in Titles)
            {
                stringBuilder.Append(Items[title]);
                stringBuilder.Append(separator);
            }
        }

        public void AddAllToReport(StringBuilder stringBuilder, string suffix, string prefix="")
        {
            bool first = true;
            foreach (var title in Titles)
            {
                if (!first)
                    stringBuilder.Append(suffix);

                stringBuilder.Append(prefix);
                stringBuilder.Append(title);
                stringBuilder.Append(": ");
                stringBuilder.Append(Items[title]);
                first = false;
            }
        }
    }

    private class SnapshotFieldsWithContexts
    {
        public readonly string[] FieldTitles;
        public readonly string[] ContextTitles;
        public SortedDictionary<int, SnapshotFields> Fields { get; }
        public SortedDictionary<int, SnapshotFields> Contexts { get; }

        public SnapshotFieldsWithContexts(string[] fieldsTitles, string[] contextTitles)
        {
            FieldTitles = fieldsTitles;
            ContextTitles = contextTitles;
            Contexts = new SortedDictionary<int, SnapshotFields>();
            Fields = new SortedDictionary<int, SnapshotFields>();
        }

        public void AddContext(int uniqueId)
        {
            Assert.IsFalse(Contexts.ContainsKey(uniqueId));
            Contexts[uniqueId] = new SnapshotFields(ContextTitles);
            Fields[uniqueId] = new SnapshotFields(FieldTitles);
        }

        public void SetContext(int uniqueId, string title, string value)
        {
            Assert.IsTrue(Contexts.ContainsKey(uniqueId));
            Contexts[uniqueId][title] = value;
        }

        public string this[int uniqueId, string title]
        {
            set
            {
                Assert.IsTrue(Fields.ContainsKey(uniqueId));
                Fields[uniqueId][title] = value;
            }
        }
    }

    private class SnapshotView
    {
        public SnapshotFields context;
        public SnapshotFields summary;
        public SnapshotFieldsWithContexts sections;

        public SnapshotView(int snapShotIndex, MemorySnapshotReport report)
        {
            context = new SnapshotFields( new [] {"Snapshot index", "Type", "Name"} );
            context["Snapshot index"] = snapShotIndex.ToString();
            context["Type"] = report.ContextType;
            context["Name"] = report.ContextName;
        }

        public SnapshotView(int snapShotIndex, LayerExecutionReport report)
        {
            context = new SnapshotFields( new [] {"Layer index", "Type", "Name"} );
            context["Layer index"] = snapShotIndex.ToString();
            context["Type"] = report.LayerType;
            context["Name"] = report.LayerName;
        }
    }
    #endregion

    #region Helpers to find information in Reports

    private static AllocatorMemoryInfo FindAllocatorInSnapshot(MemorySnapshotReport memorySnapshot, int allocatorId)
    {
        return memorySnapshot.AllocatorMemoryInfo.Find(memoryInfo => memoryInfo.UniqueId == allocatorId);
    }


    private static string FindTensorDataAllocatorInSnapshot(MemorySnapshotReport memorySnapshot, int tensorDataId)
    {
        foreach (var allocatorMemoryInfo in memorySnapshot.AllocatorMemoryInfo)
        {
            var foundTensorData = allocatorMemoryInfo.TensorDatasMemoryInfo.Find(memoryInfo => memoryInfo.UniqueId == tensorDataId);
            if (foundTensorData != null)
                return $"{allocatorMemoryInfo.Name} + / Id: {allocatorMemoryInfo.UniqueId}";
        }
        return "";
    }

    private static TensorDataMemoryInfo FindTensorDataInSnapshot(MemorySnapshotReport memorySnapshot, int tensorDataId)
    {
        bool MatchTensorDataGuidForTensor(TensorMemoryInfo memoryInfo) =>
            memoryInfo.tensorDataMemoryInfo != null && memoryInfo.tensorDataMemoryInfo.UniqueId == tensorDataId;

        var foundTensor = memorySnapshot.TensorsMemoryInfo.Find(MatchTensorDataGuidForTensor);
        if (foundTensor != null)
            return foundTensor.tensorDataMemoryInfo;

        foreach (var allocatorMemoryInfo in memorySnapshot.AllocatorMemoryInfo)
        {
            var foundTensorData = allocatorMemoryInfo.TensorDatasMemoryInfo.Find(memoryInfo => memoryInfo.UniqueId == tensorDataId);
            if (foundTensorData != null)
                return foundTensorData;
        }

        return null;
    }

    private static TensorMemoryInfo FindTensorInSnapshot(MemorySnapshotReport memorySnapshot, int tensorId)
    {
        var foundTensor = memorySnapshot.TensorsMemoryInfo.Find(memoryInfo => memoryInfo.UniqueId == tensorId);
        if (foundTensor != null)
            return foundTensor;

        foreach (var allocatorMemoryInfo in memorySnapshot.AllocatorMemoryInfo)
        {
            foundTensor = allocatorMemoryInfo.TensorsMemoryInfo.Find(memoryInfo => memoryInfo.UniqueId == tensorId);
            if (foundTensor != null)
                return foundTensor;
        }

        return null;
    }

    private static void CollectAllAsFirstSeen(in List<MemorySnapshotReport> memorySnapshots,
        out SortedDictionary<int,TensorMemoryInfo> tensors,
        out SortedDictionary<int,AllocatorMemoryInfo> allocators,
        out SortedDictionary<int,TensorDataMemoryInfo> tensorDatas)
    {
        tensors = new SortedDictionary<int, TensorMemoryInfo>();
        allocators = new SortedDictionary<int, AllocatorMemoryInfo>();
        tensorDatas = new SortedDictionary<int, TensorDataMemoryInfo>();

        //Collect all unique tensors, tensors and allocator
        foreach (var snapshot in memorySnapshots)
        {
            //From Vars
            foreach (var tensor in snapshot.TensorsMemoryInfo)
            {
                tensors[tensor.UniqueId] = tensor;
                if (tensor.tensorDataMemoryInfo != null)
                    tensorDatas[tensor.tensorDataMemoryInfo.UniqueId] = tensor.tensorDataMemoryInfo;
            }

            //From allocators
            foreach (var allocator in snapshot.AllocatorMemoryInfo)
            {
                allocators[allocator.UniqueId] = allocator;
                foreach (var tensor in allocator.TensorsMemoryInfo)
                {
                    tensors[tensor.UniqueId] = tensor;
                    if (tensor.tensorDataMemoryInfo != null)
                        tensorDatas[tensor.tensorDataMemoryInfo.UniqueId] = tensor.tensorDataMemoryInfo;
                }

                foreach (var tensorData in allocator.TensorDatasMemoryInfo)
                {
                    tensorDatas[tensorData.UniqueId] = tensorData;
                }
            }
        }
    }
    #endregion

    #region Reports -> internal data format

    private static List<SnapshotView> GenerateAllocatorViews(List<MemorySnapshotReport> memorySnapshots,
        SortedDictionary<int, AllocatorMemoryInfo> allAllocatorAsFirstSeen)
    {
        List<SnapshotView> views = new List<SnapshotView>();
        for (var memorySnapshotIndex = 0; memorySnapshotIndex < memorySnapshots.Count; memorySnapshotIndex++)
        {
            long allTotal = 0L;
            long allBusy = 0L;
            long allUsed = 0L;
            long allFragmented = 0L;
            long allFree = 0L;
            var snapshot = memorySnapshots[memorySnapshotIndex];

            //Titles and contexts
            SnapshotView view = new SnapshotView(memorySnapshotIndex, snapshot);
            view.sections = new SnapshotFieldsWithContexts(
                fieldsTitles: new[]
                {
                    "Memory pressure in bytes (sum of allocated tensorDatas capacities)",
                    "Busy bytes, for all allocators (sum of 'in use' tensorDatas capacities)",
                    "Needed bytes, for all allocators (sum of sizes of the part of the tensorDatas used by Tensors)",
                    "Unusable bytes, for all allocators (sum of the part of tensorData lost because of allocator fragmentation)",
                    "Ready bytes, for all allocators (sum of capacities of tensorData not used but allocated)"
                },
                contextTitles: new[] {"Name", "Id"});
            foreach (var allocatorMemoryInfo in allAllocatorAsFirstSeen)
            {
                var id = allocatorMemoryInfo.Key;
                view.sections.AddContext(id);
                view.sections.SetContext(id, "Name", allocatorMemoryInfo.Value.Name);
                view.sections.SetContext(id, "Id", id.ToString());
            }
            view.summary = new SnapshotFields(new[]
            {
                "Memory pressure in bytes, for all allocators (sum of allocated tensorDatas capacities)",
                "Busy bytes, for all allocators (sum of 'in use' tensorDatas capacities)",
                "Needed bytes, for all allocators (sum of sizes of the part of the tensorDatas used by Tensors)",
                "Unusable bytes, for all allocators (sum of the part of tensorData lost because of allocator fragmentation)",
                "Ready bytes, for all allocators (sum of capacities of tensorData not used but allocated)"
            });

            //Details
            foreach (var alloc in allAllocatorAsFirstSeen)
            {
                var allocator = FindAllocatorInSnapshot(snapshot, alloc.Key);
                if (allocator != null)
                {
                    allTotal += allocator.TotalBytes;
                    allBusy += allocator.BusyBytes;
                    allUsed += allocator.UsedBytes;
                    allFragmented += allocator.BytesLostToFragmentation;
                    allFree += allocator.FreeBytes;
                    view.sections[allocator.UniqueId, "Memory pressure in bytes (sum of allocated tensorDatas capacities)"] = allocator.TotalBytes.ToString();
                    view.sections[allocator.UniqueId, "Busy bytes, for all allocators (sum of 'in use' tensorDatas capacities)"] = allocator.BusyBytes.ToString();
                    view.sections[allocator.UniqueId, "Needed bytes, for all allocators (sum of sizes of the part of the tensorDatas used by Tensors)"] = allocator.UsedBytes.ToString();
                    view.sections[allocator.UniqueId, "Unusable bytes, for all allocators (sum of the part of tensorData lost because of allocator fragmentation)"] = allocator.BytesLostToFragmentation.ToString();
                    view.sections[allocator.UniqueId, "Ready bytes, for all allocators (sum of capacities of tensorData not used but allocated)"] = allocator.FreeBytes.ToString();
                }
            }

            //Summary
            view.summary["Memory pressure in bytes, for all allocators (sum of allocated tensorDatas capacities)"] = allTotal.ToString();
            view.summary["Busy bytes, for all allocators (sum of 'in use' tensorDatas capacities)"] = allBusy.ToString();
            view.summary["Needed bytes, for all allocators (sum of sizes of the part of the tensorDatas used by Tensors)"] = allUsed.ToString();
            view.summary["Unusable bytes, for all allocators (sum of the part of tensorData lost because of allocator fragmentation)"] = allFragmented.ToString();
            view.summary["Ready bytes, for all allocators (sum of capacities of tensorData not used but allocated)"] = allFree.ToString();
            views.Add(view);
        }

        return views;
    }

    private static List<SnapshotView> GenerateTensorDatasViews(List<MemorySnapshotReport> memorySnapshots,
        SortedDictionary<int,TensorDataMemoryInfo> allTensorDataAsFirstSeen)
    {
        List<SnapshotView> views = new List<SnapshotView>();
        for (var memorySnapshotIndex = 0; memorySnapshotIndex < memorySnapshots.Count; memorySnapshotIndex++)
        {
            long allGPUInBytes = 0L;
            long allCPUInBytes = 0L;
            long allUsedGPUInBytes = 0L;
            long allUsedCPUInBytes = 0L;

            var snapshot = memorySnapshots[memorySnapshotIndex];

            //Titles and contexts
            SnapshotView view = new SnapshotView(memorySnapshotIndex, snapshot);
            view.sections = new SnapshotFieldsWithContexts(
                fieldsTitles: new[]
                {
                    "In use",
                    "Capacity (bytes)",
                    "On GPU",
                    "Allocator"
                },
                contextTitles: new[] {"Id"});
            foreach (var tensorData in allTensorDataAsFirstSeen)
            {
                var id = tensorData.Key;
                view.sections.AddContext(id);
                view.sections.SetContext(id, "Id", id.ToString());
            }
            view.summary = new SnapshotFields(new[]
            {
                "GPU sum of all allocated tensorData capacities (bytes)",
                "CPU sum of all allocated tensorData capacities (bytes)",
                "GPU sum of all 'in use' tensorData (bytes)",
                "CPU sum of all 'in use' tensorData (bytes)"
            });

            foreach (var tData in allTensorDataAsFirstSeen)
            {
                TensorDataMemoryInfo tensorData = FindTensorDataInSnapshot(snapshot, tData.Key);
                if (tensorData != null)
                {
                    if (tensorData.IsGPUMem)
                    {
                        allGPUInBytes += tensorData.MaxBytes;
                        if (tensorData.InUse)
                            allUsedGPUInBytes += tensorData.MaxBytes;
                    }
                    else
                    {
                        allCPUInBytes += tensorData.MaxBytes;
                        if (tensorData.InUse)
                            allUsedCPUInBytes += tensorData.MaxBytes;
                    }

                    view.sections[tensorData.UniqueId, "In use"] = tensorData.InUse ? "Yes" : "";
                    view.sections[tensorData.UniqueId, "Capacity (bytes)"] = tensorData.MaxBytes.ToString();
                    view.sections[tensorData.UniqueId, "On GPU"] = tensorData.IsGPUMem ? "GPU" : "CPU";
                    view.sections[tensorData.UniqueId, "Allocator"] = FindTensorDataAllocatorInSnapshot(snapshot, tensorData.UniqueId);
                }
            }

            //Summary
            view.summary["GPU sum of all allocated tensorData capacities (bytes)"] = allGPUInBytes.ToString();
            view.summary["CPU sum of all allocated tensorData capacities (bytes)"] = allCPUInBytes.ToString();
            view.summary["GPU sum of all 'in use' tensorData (bytes)"] = allUsedGPUInBytes.ToString();
            view.summary["CPU sum of all 'in use' tensorData (bytes)"] = allUsedCPUInBytes.ToString();
            views.Add(view);
        }

        return views;
    }

    private static List<SnapshotView> GenerateTensorsViews(List<MemorySnapshotReport> memorySnapshots,
        SortedDictionary<int, TensorMemoryInfo> allTensorAsFirstSeen)
    {
        List<SnapshotView> views = new List<SnapshotView>();
        for (var memorySnapshotIndex = 0; memorySnapshotIndex < memorySnapshots.Count; memorySnapshotIndex++)
        {
            var snapshot = memorySnapshots[memorySnapshotIndex];

            //Titles and contexts
            SnapshotView view = new SnapshotView(memorySnapshotIndex, snapshot);
            view.sections = new SnapshotFieldsWithContexts(
                fieldsTitles: new[] {"Allocated", "Cache size (bytes)", "TensorData Id"},
                contextTitles: new[] {"Name", "Shape", "Id"});
            foreach (var tensorMemoryInfo in allTensorAsFirstSeen)
            {
                var id = tensorMemoryInfo.Key;
                view.sections.AddContext(id);
                view.sections.SetContext(id, "Name", tensorMemoryInfo.Value.Name);
                view.sections.SetContext(id, "Shape", tensorMemoryInfo.Value.Shape.ToString());
                view.sections.SetContext(id, "Id", id.ToString());
            }
            view.summary = new SnapshotFields(new[]
            {
                "Tensor memory on GPU (in bytes)",
                "Tensor memory on CPU (in bytes)",
                "On CPU tensor cache (in bytes)"
            });

            //Details
            long cacheMemInBytes = 0L;
            long gpuMem = 0L;
            long cpuMem = 0L;
            foreach (var tensorFromDict in allTensorAsFirstSeen)
            {
                var tensor = FindTensorInSnapshot(snapshot, tensorFromDict.Key);
                if (tensor != null)
                {
                    cacheMemInBytes += tensor.CacheBytes;

                    view.sections[tensor.UniqueId, "Allocated"] = "Yes";
                    view.sections[tensor.UniqueId, "Cache size (bytes)"] = tensor.CacheBytes.ToString();
                    if (tensor.tensorDataMemoryInfo != null)
                    {
                        view.sections[tensor.UniqueId, "TensorData Id"] =
                            tensor.tensorDataMemoryInfo.UniqueId.ToString();
                        if (tensor.tensorDataMemoryInfo.IsGPUMem)
                            gpuMem += tensor.Shape.length;
                        else
                            cpuMem += tensor.Shape.length;
                    }
                }
            }

            //Summary
            view.summary["Tensor memory on GPU (in bytes)"] = gpuMem.ToString();
            view.summary["Tensor memory on CPU (in bytes)"] = cpuMem.ToString();
            view.summary["On CPU tensor cache (in bytes)"] = cacheMemInBytes.ToString();
            views.Add(view);
        }

        return views;
    }

    private static List<SnapshotView> GenerateExecutionViews(List<LayerExecutionReport> layerReports, int numCompletedLayer)
    {
        List<SnapshotView> views = new List<SnapshotView>();
        for (var layerIndex = 0; layerIndex < layerReports.Count; layerIndex++)
        {
            var report = layerReports[layerIndex];

            //Titles
            SnapshotView view = new SnapshotView(layerIndex, report);
            view.sections = new SnapshotFieldsWithContexts(null, null);
            view.summary = new SnapshotFields(new[]
            {
                "Summary",
                "Compute Kernels(workItems:X,Y,Z)",
                "Theoretical ALU count",
                "Theoretical Bandwidth (bytes)",
                "Note"
            });

            //Summary
            view.summary["Summary"] = report.Summary==""?"NA":report.Summary;
            view.summary["Compute Kernels(workItems:X,Y,Z)"] = report.DispatchInfos;
            view.summary["Theoretical ALU count"] = report.NumAlu.ToString();
            view.summary["Theoretical Bandwidth (bytes)"] = report.NumBytes.ToString();
            if (layerIndex >= numCompletedLayer)
                view.summary["Note"] = "UNCOMPLETED LAYER";
            views.Add(view);
        }

        return views;
    }

    private static List<SnapshotView> GenerateSummaryViews(List<MemorySnapshotReport> memorySnapshots,
        SortedDictionary<int, TensorMemoryInfo> allTensorsAsFirstSeen,
        SortedDictionary<int, TensorDataMemoryInfo> allTensorDatasAsFirstSeen)
    {
        HashSet<int> previousSnapshotTensorIds = new HashSet<int>();
        List<SnapshotView> views = new List<SnapshotView>();

        for (var memorySnapshotIndex = 0; memorySnapshotIndex < memorySnapshots.Count; memorySnapshotIndex++)
        {
            var snapshot = memorySnapshots[memorySnapshotIndex];

            //Titles and contexts
            SnapshotView view = new SnapshotView(memorySnapshotIndex, snapshot);
            view.sections = new SnapshotFieldsWithContexts(
                fieldsTitles: new[] {"Allocated", "Released"},
                contextTitles: new[] {"Type" });
            view.sections.AddContext(0);
            view.sections.SetContext(0, "Type", "Tensor");
            view.summary = new SnapshotFields(new[]
            {
                "Total memory pressure on GPU (in bytes)",
                "Total memory pressure on CPU (in bytes)",
                "On CPU tensor cache (in bytes)"
            });

            //Summary
            HashSet<int> currentSnapshotTensorIds = new HashSet<int>();
            long cacheMemInBytes = 0L;
            foreach (var tensor in snapshot.TensorsMemoryInfo)
            {
                cacheMemInBytes += tensor.CacheBytes;
                currentSnapshotTensorIds.Add(tensor.UniqueId);
            }
            long gpuMem = 0L;
            long cpuMem = 0L;
            foreach (var tData in allTensorDatasAsFirstSeen)
            {
                TensorDataMemoryInfo tensorData = FindTensorDataInSnapshot(snapshot, tData.Key);
                if (tensorData != null)
                {
                    if (tensorData.IsGPUMem)
                        gpuMem += tensorData.MaxBytes;
                    else
                        cpuMem += tensorData.MaxBytes;
                }
            }
            view.summary["Total memory pressure on GPU (in bytes)"] = gpuMem.ToString();
            view.summary["Total memory pressure on CPU (in bytes)"] = cpuMem.ToString();
            view.summary["On CPU tensor cache (in bytes)"] = cacheMemInBytes.ToString();

            if (memorySnapshotIndex != 0)
            {
                //Tensor allocated and freed (diff from snapshot to snapshot)
                var allocatedTensorsId = currentSnapshotTensorIds.Except(previousSnapshotTensorIds);
                var releasedTensorsId = previousSnapshotTensorIds.Except(currentSnapshotTensorIds);
                StringBuilder tensorDiff = new StringBuilder();
                bool first = true;
                foreach (var tensorId in allocatedTensorsId)
                {
                    var tensor = FindTensorInSnapshot(snapshot, tensorId);
                    string tensorDataInfo = "none";
                    if (tensor.tensorDataMemoryInfo != null)
                    {
                        var data = tensor.tensorDataMemoryInfo;
                        var memType = data.IsGPUMem ? "GPU" : "CPU";
                        tensorDataInfo = $"id:{data.UniqueId} bytes:{data.MaxBytes} on:{memType}";
                    }
                    if (!first) tensorDiff.Append(" / ");
                    first = false;
                    tensorDiff.Append($"{tensor.Name} {tensor.Shape} id:{tensor.UniqueId} tensorData:[{tensorDataInfo}]");

                }
                view.sections[0, "Allocated"] = tensorDiff.ToString();
                tensorDiff.Clear();

                first = true;
                foreach (var tensorId in releasedTensorsId)
                {
                    var tensor = allTensorsAsFirstSeen[tensorId];
                    if (!first) tensorDiff.Append(" / ");
                    first = false;
                    tensorDiff.Append($"{tensor.Name} {tensor.Shape} id:{tensor.UniqueId}");
                }
                view.sections[0, "Released"] = tensorDiff.ToString();
            }

            views.Add(view);
            previousSnapshotTensorIds = currentSnapshotTensorIds;
        }

        return views;
    }

    #endregion

    #region Internal data format -> text

    private static void Append(this StringBuilder sb, string str, int repeatCount)
    {
        for (int i = 0; i < repeatCount; ++i)
            sb.Append(str);
    }

    private static void Append(this StringBuilder sb, string str, string separator)
    {
        sb.Append(str);
        sb.Append(separator);
    }

    private static void GenerateReportForViews(StringBuilder stringBuilder, List<SnapshotView> views, bool spreadSheetFormat, string sectionTitle, bool isSummaryView)
    {
        if (spreadSheetFormat)
        {
            //Columns Titles
            views[0].context.AddTitlesToReport(stringBuilder, ModelExecutionsReporter.SpreadSheetFieldSeparator);
            views[0].summary.AddTitlesToReport(stringBuilder, ModelExecutionsReporter.SpreadSheetFieldSeparator);
            stringBuilder.Append("|", ModelExecutionsReporter.SpreadSheetFieldSeparator);
            foreach (var tensorFields in views[0].sections.Fields)
            {
                tensorFields.Value.AddTitlesToReport(stringBuilder, ModelExecutionsReporter.SpreadSheetFieldSeparator);
                stringBuilder.Append("|", ModelExecutionsReporter.SpreadSheetFieldSeparator);
            }
            stringBuilder.Append("\n");

            //All snapshots
            foreach (var view in views)
            {
                view.context.AddValuesToReport(stringBuilder, ModelExecutionsReporter.SpreadSheetFieldSeparator);
                view.summary.AddValuesToReport(stringBuilder, ModelExecutionsReporter.SpreadSheetFieldSeparator);
                stringBuilder.Append("|", ModelExecutionsReporter.SpreadSheetFieldSeparator);
                foreach (var tensorFields in view.sections.Fields)
                {
                    tensorFields.Value.AddValuesToReport(stringBuilder, ModelExecutionsReporter.SpreadSheetFieldSeparator);
                    stringBuilder.Append("|", ModelExecutionsReporter.SpreadSheetFieldSeparator);
                }
                stringBuilder.Append("\n");
            }

        }
        else
        {
            string doubleIndentation = ModelExecutionsReporter.TextIndentation + ModelExecutionsReporter.TextIndentation;

            foreach (var view in views)
            {
                view.context.AddAllToReport(stringBuilder, ModelExecutionsReporter.TextFormatFieldSeparator);
                stringBuilder.Append("\n");
                view.summary.AddAllToReport(stringBuilder, suffix:"\n", prefix: ModelExecutionsReporter.TextIndentation);
                stringBuilder.Append("\n"+ModelExecutionsReporter.TextIndentation + sectionTitle +"\n");

                foreach (var context in view.sections.Contexts)
                {
                    stringBuilder.Append(doubleIndentation);
                    if (isSummaryView)
                    {
                        view.sections.Fields[context.Key].AddAllToReport(stringBuilder, "\n"+doubleIndentation);
                    }
                    else
                    {
                        context.Value.AddAllToReport(stringBuilder, ModelExecutionsReporter.TextFormatFieldSeparator);
                        stringBuilder.Append("\n"+doubleIndentation +"=> ");
                        view.sections.Fields[context.Key].AddAllToReport(stringBuilder, ModelExecutionsReporter.TextFormatFieldSeparator);
                        stringBuilder.Append("\n");
                    }
                }
                stringBuilder.Append("\n");
            }
        }
    }

    private static void GenerateHeaderForSummaryViews(StringBuilder stringBuilder, List<SnapshotView> views, bool spreadSheetFormat)
    {
        if (views.Count == 0)
        {
            stringBuilder.Append("<******** Summary info ********> NONE!\n");
            return;
        }

        if (!spreadSheetFormat)
        {
            stringBuilder.Append("<******** Summary info ********>\n");
            return;
        }

        //Tensor contexts
        int ctxFieldCount = views[0].context.Titles.Length + views[0].summary.Titles.Length;
        int sectionFieldCount = views[0].sections.FieldTitles.Length;

        stringBuilder.Append("<******** Summary info ********>");
        stringBuilder.Append(ModelExecutionsReporter.SpreadSheetFieldSeparator, ctxFieldCount);
        stringBuilder.Append("|", ModelExecutionsReporter.SpreadSheetFieldSeparator);
        foreach (var context in views[0].sections.Contexts)
        {
            stringBuilder.Append(context.Value["Type"], ModelExecutionsReporter.SpreadSheetFieldSeparator);
            stringBuilder.Append(ModelExecutionsReporter.SpreadSheetFieldSeparator, sectionFieldCount-1);
            stringBuilder.Append("|", ModelExecutionsReporter.SpreadSheetFieldSeparator);
        }
        stringBuilder.Append("\n");
    }

    private static void GenerateHeaderForTensorViews(StringBuilder stringBuilder, List<SnapshotView> views, bool spreadSheetFormat)
    {
        if (views.Count == 0)
        {
            stringBuilder.Append("<******** Tensors info ********> NONE!\n");
            return;
        }

        if (!spreadSheetFormat)
        {
            stringBuilder.Append("<******** Tensors info ********>\n");
            return;
        }

        //Tensor contexts
        int ctxFieldCount = views[0].context.Titles.Length + views[0].summary.Titles.Length;
        int sectionFieldCount = views[0].sections.FieldTitles.Length;

        stringBuilder.Append(ModelExecutionsReporter.SpreadSheetFieldSeparator, ctxFieldCount);
        stringBuilder.Append("|", ModelExecutionsReporter.SpreadSheetFieldSeparator);
        stringBuilder.Append("Tensors names and shapes:");
        stringBuilder.Append("\n");

        stringBuilder.Append(ModelExecutionsReporter.SpreadSheetFieldSeparator, ctxFieldCount);
        stringBuilder.Append("|", ModelExecutionsReporter.SpreadSheetFieldSeparator);
        foreach (var context in views[0].sections.Contexts)
        {
            stringBuilder.Append(context.Value["Name"], ModelExecutionsReporter.SpreadSheetFieldSeparator);
            stringBuilder.Append(ModelExecutionsReporter.SpreadSheetFieldSeparator, sectionFieldCount-1);
            stringBuilder.Append("|", ModelExecutionsReporter.SpreadSheetFieldSeparator);
        }
        stringBuilder.Append("\n");

        stringBuilder.Append("<******** Tensors info ********>");
        stringBuilder.Append(ModelExecutionsReporter.SpreadSheetFieldSeparator, ctxFieldCount);
        stringBuilder.Append("|", ModelExecutionsReporter.SpreadSheetFieldSeparator);
        foreach (var context in views[0].sections.Contexts)
        {
            stringBuilder.Append(context.Value["Shape"], " / Id: ");
            stringBuilder.Append(context.Value["Id"], ModelExecutionsReporter.SpreadSheetFieldSeparator);
            stringBuilder.Append(ModelExecutionsReporter.SpreadSheetFieldSeparator, sectionFieldCount-1);
            stringBuilder.Append("|", ModelExecutionsReporter.SpreadSheetFieldSeparator);
        }
        stringBuilder.Append("\n");
    }

    private static void GenerateHeaderForTensorDatasViews(StringBuilder stringBuilder, List<SnapshotView> views, bool spreadSheetFormat)
    {
        if (views.Count == 0)
        {
            stringBuilder.Append("<******** TensorDatas info ********> NONE!\n");
            return;
        }

        if (!spreadSheetFormat)
        {
            stringBuilder.Append("<******** TensorDatas info ********>\n");
            return;
        }

        //Tensor contexts
        int ctxFieldCount = views[0].context.Titles.Length + views[0].summary.Titles.Length;
        int sectionFieldCount = views[0].sections.FieldTitles.Length;

        stringBuilder.Append("<******** TensorDatas info ********>");
        stringBuilder.Append(ModelExecutionsReporter.SpreadSheetFieldSeparator, ctxFieldCount);
        stringBuilder.Append("|", ModelExecutionsReporter.SpreadSheetFieldSeparator);
        foreach (var context in views[0].sections.Contexts)
        {
            stringBuilder.Append("Id: ");
            stringBuilder.Append(context.Value["Id"], ModelExecutionsReporter.SpreadSheetFieldSeparator);
            stringBuilder.Append(ModelExecutionsReporter.SpreadSheetFieldSeparator, sectionFieldCount-1);
            stringBuilder.Append("|", ModelExecutionsReporter.SpreadSheetFieldSeparator);
        }
        stringBuilder.Append("\n");
    }

    private static void GenerateHeaderForAllocatorsViews(StringBuilder stringBuilder, List<SnapshotView> views, bool spreadSheetFormat)
    {
        if (views.Count == 0)
        {
            stringBuilder.Append("<******** Allocators info ********> NONE!\n");
            return;
        }

        if (!spreadSheetFormat)
        {
            stringBuilder.Append("<******** Allocators info ********>\n");
            return;
        }

        //Tensor contexts
        int ctxFieldCount = views[0].context.Titles.Length + views[0].summary.Titles.Length;
        int sectionFieldCount = views[0].sections.FieldTitles.Length;

        stringBuilder.Append(ModelExecutionsReporter.SpreadSheetFieldSeparator, ctxFieldCount);
        stringBuilder.Append("|", ModelExecutionsReporter.SpreadSheetFieldSeparator);
        stringBuilder.Append("Allocators names and shapes:");
        stringBuilder.Append("\n");

        stringBuilder.Append("<******** Allocators info ********>");
        stringBuilder.Append(ModelExecutionsReporter.SpreadSheetFieldSeparator, ctxFieldCount);
        stringBuilder.Append("|", ModelExecutionsReporter.SpreadSheetFieldSeparator);
        foreach (var context in views[0].sections.Contexts)
        {
            stringBuilder.Append(context.Value["Name"], " / Id: ");
            stringBuilder.Append(context.Value["Id"], ModelExecutionsReporter.SpreadSheetFieldSeparator);
            stringBuilder.Append(ModelExecutionsReporter.SpreadSheetFieldSeparator, sectionFieldCount-1);
            stringBuilder.Append("|", ModelExecutionsReporter.SpreadSheetFieldSeparator);
        }
        stringBuilder.Append("\n");
    }

    #endregion
}

} // namespace Unity.Barracuda
