using System;
using System.Collections.Generic;
using System.Linq; // ToList()
using UnityEngine;
using UnityEngine.Assertions;

namespace Unity.Barracuda {


internal class BarracudaBackendsFactory
{
    public static WorkerFactory.Type ResolveAutoType(WorkerFactory.Type type)
    {
        if (type != WorkerFactory.Type.Auto)
            return type;
        return GetBestTypeForDevice(WorkerFactory.Device.Auto);
    }

    internal static WorkerFactory.Type GetBestTypeForDevice(WorkerFactory.Device device)
    {
        switch (device)
        {
            case WorkerFactory.Device.Auto:
            case WorkerFactory.Device.GPU:
                return WorkerFactory.Type.ComputePrecompiled;
            default:
                return WorkerFactory.Type.CSharpBurst;
        }
    }

    internal static WorkerFactory.Type ValidateType(WorkerFactory.Type type)
    {
        type = ResolveAutoType(type);
        Assert.AreNotEqual(type, WorkerFactory.Type.Auto);

        if (WorkerFactory.IsType(type, WorkerFactory.Device.GPU) && !ComputeShaderSingleton.Instance.supported)
        {
            D.LogWarning(
                $"SystemInfo.supportsComputeShaders: {SystemInfo.supportsComputeShaders}. Falling back to {WorkerFactory.Type.CSharp}");
            type = WorkerFactory.Type.CSharp;
        }

        return type;
    }

    private static IOps CreateOps(WorkerFactory.Type type, ITensorAllocator allocator, bool verbose)
    {
        switch(type)
        {
        case WorkerFactory.Type.ComputePrecompiled:
            return new PrecompiledComputeOps(ComputeShaderSingleton.Instance.kernels,
                                            ComputeShaderSingleton.Instance.referenceKernels, allocator, verbose);

        case WorkerFactory.Type.Compute:
            return new ComputeOps(ComputeShaderSingleton.Instance.kernels,
                                 ComputeShaderSingleton.Instance.referenceKernels, allocator, verbose);

        case WorkerFactory.Type.ComputeRef:
            return new ReferenceComputeOps(ComputeShaderSingleton.Instance.referenceKernels, allocator);

        case WorkerFactory.Type.CSharpBurst:
            return new BurstCPUOps(allocator);

        case WorkerFactory.Type.CSharp:
            return new UnsafeArrayCPUOps(allocator);

        default:
            return new ReferenceCPUOps(allocator);
        }
    }

    internal static IWorker CreateWorker(WorkerFactory.Type type, Model model, string[] additionalOutputs, string[] trimOutputs, WorkerFactory.WorkerConfiguration workerConfiguration)
    {
        type = ResolveAutoType(type);
        var compareAgainstType = ResolveAutoType(workerConfiguration.compareAgainstType);
        Assert.AreNotEqual(type, WorkerFactory.Type.Auto);
        Assert.AreNotEqual(compareAgainstType, WorkerFactory.Type.Auto);

        bool compare = type != compareAgainstType;

        if (WorkerFactory.IsType(type, WorkerFactory.Device.GPU) && !SystemInfo.supportsComputeShaders && !Application.isEditor)
        {
            D.LogWarning("Compute shaders are not supported on current platform. Falling back to CSharpFast.");
            type = WorkerFactory.Type.CSharpBurst;
        }

        IVars vars;
        if (WorkerFactory.IsType(type, WorkerFactory.Device.GPU) || WorkerFactory.IsType(compareAgainstType, WorkerFactory.Device.GPU))
            vars = new ComputeVarsWithSharedModel();
        else
            vars = new DefaultVars();

        ITensorAllocator allocator = vars.GetAllocator();

        if (workerConfiguration.verbose)
            D.Log($"Storage type: {vars.GetType()}. Allocator type: {allocator.GetType()}.");

        IOps ops = CreateOps(type, allocator, workerConfiguration.verbose);

        if (compare)
            ops = new CompareOps(ops,
                CreateOps(compareAgainstType, allocator, workerConfiguration.verbose), workerConfiguration.compareLogLevel, workerConfiguration.compareEpsilon);

        if (workerConfiguration.verbose)
            ops = new VerboseOps(ops);

        if (Application.isEditor)
            ops = new StatsOps(ops);

        model = ValidateModel(
            PatchModel(model, additionalOutputs, trimOutputs));

        return new GenericWorker(model, ops, vars, workerConfiguration.verbose);
    }

    internal static Model PatchModel(Model model, string[] additionalOutputs, string[] trimOutputs = null)
    {
        bool trimModel = trimOutputs != null;

        if (trimOutputs != null)
        {
            foreach (var o in trimOutputs.Except(model.outputs))
                if (additionalOutputs == null || !additionalOutputs.Contains(o))
                    D.LogWarning($"Output specified in trimOutputs was not found in the model: {o}");

            var newModel = model.ShallowCopy();
            newModel.outputs = trimOutputs.Intersect(model.outputs).ToList();
            model = newModel;
        }

        if (additionalOutputs != null)
        {
            foreach (var o in additionalOutputs.Except(model.layers.Select(l => l.name)))
                D.LogWarning($"Layer specified in additionalOutputs was not found in the model: {o}");

            // 'new' means that output name does not yet exist in model.outputs
            // 'valid' means that output name matches one of the existing model.layer names
             var newAndValidAdditionalOutputs =
                additionalOutputs.Except(model.outputs).Intersect(model.layers.Select(l => l.name));

            var newModel = model.ShallowCopy();
            newModel.outputs.AddRange(newAndValidAdditionalOutputs);
            model = newModel;
        }

        if (trimModel)
        {
            var newModel = model.ShallowCopy();
            var upstream = ModelAnalyzer.FindUpstreamLayers(model, newModel.outputs.ToArray());
            foreach (var l in model.layers)
                if (!upstream.Contains(l))
                    newModel.layers.Remove(l);

            model = newModel;
        }

        model = ModelOptimizer.RemoveNoop(model);

        return model;
    }

    internal static Model ValidateModel(Model model)
    {
        // validate, model contains no broken links
        var brokenLinks = ModelAnalyzer.FindBrokenLinks(model);
        if (brokenLinks.Length > 0)
            D.LogWarning($"Model contains {brokenLinks.Length} broken links: {string.Join(",", brokenLinks)}");

        // validate, all model outputs are unique
        // https://stackoverflow.com/questions/18547354/c-sharp-linq-find-duplicates-in-list
        var duplicateOutputs = model.outputs.GroupBy(x => x)
            .Where(g => g.Count() > 1)
            .Select(y => y.Key);
        foreach (var o in duplicateOutputs)
            D.LogWarning($"Output is specified more than once in the model: {o}");

        // validate, model contains no unconnected layers
        var unconnectedOutputs = ModelAnalyzer.FindUnconnectedOutputs(model);
        foreach (var o in unconnectedOutputs)
            D.LogWarning($"Layer is specified as output, but is missing in the model: {o}");

        return model;
    }
}


} // namespace Unity.Barracuda
