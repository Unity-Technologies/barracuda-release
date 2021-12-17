using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Profiling;

using System.Runtime.CompilerServices;

[assembly: InternalsVisibleTo("Unity.Barracuda.PerformanceTests")]
[assembly: InternalsVisibleTo("Unity.Barracuda.Tests")]

namespace Unity.Barracuda
{

/// <summary>
/// Generic `IWorker` implementation
/// </summary>
public class GenericWorker : IWorker
{
    private Model m_Model;
    private string m_DefaultInputName;
    private string m_DefaultOutputName;
    private Dictionary<string, TensorShape> m_InputShapes = new Dictionary<string, TensorShape>();
    private DataType m_ActivationsDataType = DataType.Float;
    private IOps m_Ops;
    private IVars m_Vars;
    private IModelCompiler m_ModelCompiler;
    private Tensor m_DummyInput;

    private bool m_AllocatorIsStale = false;
    private bool m_AllocatorIsOccupied = false;
    private bool m_Verbose;
    private bool m_TakeoverWeights;
    private float m_Progress = 0f;

    private Tensor m_SyncTensor;

    // Heuristic size for a small tensor. Small tensors are more likely to be accessed on CPU,
    // thus PeekOutput() for such small tensor will auto schedule non-blocking download from GPU/NPU to CPU
    const int m_MaxBatchThatAutoTriggersAsyncDownload = 64;
    const int m_MaxFlatWidthThatAutoTriggersAsyncDownload = 1000;

    /// <summary>
    /// Create `GenericWorker` for specified `model` and `ops`
    /// </summary>
    /// <param name="model">`Model`</param>
    /// <param name="ops">`IOps`</param>
    /// <param name="vars">`IVars`</param>
    /// <param name="verbose">verbose execution flag</param>
    /// <param name="takeoverWeights">takeover weights execution flag</param>
    public GenericWorker(Model model, IOps ops, IVars vars, bool verbose = false, bool takeoverWeights = false)
    {
        m_Model = model;
        m_DefaultInputName = ModelAnalyzer.GetDefaultInputName(model);
        m_DefaultOutputName = ModelAnalyzer.GetDefaultOutputName(model);
        m_Ops = ops;
        m_Vars = vars;
        m_ModelCompiler = ops as IModelCompiler;
        m_DummyInput = new Tensor();
        m_Verbose = verbose;
        m_TakeoverWeights = takeoverWeights;

        m_AllocatorIsStale = true;

        SetupTensorLeaksTracking();
    }

    private void SetupTensorLeaksTracking()
    {
        //Reference backends are not targeting optimal memory usage
        //and should not be tracked for tensor leaks

        //Note: duplicate test (considering inheritance) for clarity
        bool isProductionBackend =
            m_Ops is UnsafeArrayCPUOps || m_Ops is BurstCPUOps ||
            m_Ops is ComputeOps || m_Ops is PrecompiledComputeOps;

        var genericVarsWithPreallocation = m_Vars as GenericVarsWithPreallocation;
        if (genericVarsWithPreallocation != null)
        {
            genericVarsWithPreallocation.ShouldTrackTensorLeaks = isProductionBackend;
        }
    }

    /// <summary>
    /// Finalizer
    /// </summary>
    ~GenericWorker()
    {
        Dispose();
    }

    internal void OccupyAllocator()
    {
        m_AllocatorIsOccupied = true;
    }

    internal void ResetAllocatorIfStale()
    {
        if (m_AllocatorIsStale)
        {
            m_Ops.ResetAllocator();
            m_AllocatorIsStale = false;
            m_AllocatorIsOccupied = false;
        }
    }

    internal void ResetAllocatorIfStaleAndNotOccupied()
    {
        if (!m_AllocatorIsOccupied)
            ResetAllocatorIfStale();
    }

    /// <summary>
    /// Dispose all internal storage structures
    /// </summary>
    public virtual void Dispose()
    {
        m_Vars?.Dispose();
        m_Ops?.ResetAllocator(false); // clear allocator's memory
        m_InputShapes?.Clear();
        m_DummyInput?.Dispose();

        m_Vars = null;
        m_Ops = null;
        m_InputShapes = null;
    }

    /// <inheritdoc/>
    public virtual void PrepareForInput(IDictionary<string, TensorShape> inputShapes, DataType dataType)
    {
        m_InputShapes.Clear();
        foreach (var input in inputShapes)
            m_InputShapes.Add(input.Key, input.Value);
        m_ActivationsDataType = dataType;//TODO fp16. for now all activations are expected to share the same data type
        m_Vars.PrepareStorage(m_Model, m_Ops, m_InputShapes, m_TakeoverWeights, m_ActivationsDataType);
    }

    /// <inheritdoc/>
    public virtual void SetInput(string name, Tensor x)
    {
        ResetAllocatorIfStale();
        OccupyAllocator();

        m_Ops.Prepare(x);
        m_Vars.SetInput(name, x);

        // if single input network, then we have enough information to prepare network for execution
        if (m_Model.inputs.Count <= 1 && name == m_DefaultInputName)
        {
            m_ActivationsDataType = x.dataType;
            PrepareForInput(new Dictionary<string, TensorShape> { { name, x.shape } }, m_ActivationsDataType); // @TODO: get rid of allocation
        }

        m_InputShapes[name] = x.shape;
    }

    /// <inheritdoc/>
    public virtual void SetInput(Tensor x)
    {
        SetInput(m_DefaultInputName, x);
    }

    /// <inheritdoc/>
    public virtual IWorker Execute(IDictionary<string, Tensor> inputs)
    {
        foreach (var entry in inputs)
            SetInput(entry.Key, entry.Value);
        return Execute();
    }

    /// <inheritdoc/>
    public virtual IWorker Execute(Tensor input)
    {
        SetInput(input);
        return Execute();
    }

    /// <inheritdoc/>
    public virtual IWorker Execute()
    {
        Profiler.BeginSample ("Barracuda.Execute");
        var enumerator = StartManualSchedule();
        while (enumerator.MoveNext()) {};
        Profiler.EndSample ();
        return this;
    }

    /// <inheritdoc/>
    public virtual IEnumerator StartManualSchedule(IDictionary<string, Tensor> inputs)
    {
        foreach (var entry in inputs)
            SetInput(entry.Key, entry.Value);
        return StartManualSchedule();
    }

    /// <inheritdoc/>
    public virtual void FlushSchedule(bool blocking)
    {
        // force execution of scheduled ops by requesting results of the intermediate tensor from the device
        m_SyncTensor.PrepareCacheForAccess(blocking);
    }

    /// <inheritdoc/>
    public virtual IEnumerator StartManualSchedule(Tensor input)
    {
        SetInput(input);
        return StartManualSchedule();
    }

    /// <inheritdoc/>
    public virtual float scheduleProgress
    {
        get
        {
            return m_Progress;
        }
    }

    private static Layer.FusedActivation GetAndVerifyFusedActivation(Layer l)
    {
        Assert.IsTrue(ModelOptimizer.IsLayerSupportingActivationFusing(l.type));
        if (!ModelOptimizer.IsActivationFusable(l.activation))
            throw new NotImplementedException("This activation function is not implemented as a fusable one yet! Check Layer.FusedActivation for supported ones.");

        return (Layer.FusedActivation) l.activation;
    }

    /// <inheritdoc/>
    public virtual IEnumerator StartManualSchedule()
    {
        ResetAllocatorIfStaleAndNotOccupied();
        m_AllocatorIsStale = true;

#if ENABLE_BARRACUDA_STATS
        m_Ops.GetModelExecutionsReporter()?.ModelExecutionStarted();
        m_Ops.GetModelExecutionsReporter()?.TakeMemorySnapshot(m_Ops, m_Vars, "Before model execution, step1: After Allocator reset");
#endif //ENABLE_BARRACUDA_STATS

        m_Vars.PrepareStorage(m_Model, m_Ops, m_InputShapes, m_TakeoverWeights, m_ActivationsDataType);

        if (m_ModelCompiler != null)
            m_ModelCompiler.PrepareModel(m_Model, m_InputShapes, m_Vars);

#if ENABLE_BARRACUDA_STATS
        m_Ops.GetModelExecutionsReporter()?.TakeMemorySnapshot(m_Ops, m_Vars, "Before model execution, step2: After Model preparation");
#endif //ENABLE_BARRACUDA_STATS

        int idx = 0;
        foreach (var l in m_Model.layers)
        {
            idx++;

            m_Progress = idx / (float)m_Model.layers.Count;

#if ENABLE_BARRACUDA_STATS
            m_Ops.GetModelExecutionsReporter()?.LayerExecutionStarted(l);
#endif //ENABLE_BARRACUDA_STATS

            Profiler.BeginSample(l.name);

            var inputs = m_Vars.GatherInputs(l);

            Tensor X = inputs.Length > 0 ? inputs[0] : m_DummyInput;

            if (m_Verbose)
                D.Log("Layer: " + l.type + ((l.type == Layer.Type.Activation) ? ("." + l.activation) : "") + " " + l.name );

            m_Vars.PrepareStorage(l);
            if (m_ModelCompiler != null)
                m_ModelCompiler.PreExecuteLayer(l, inputs);

            // No operation, identity
            if (l.type == Layer.Type.Nop)
            {
                Profiler.BeginSample ("Barracuda.Nop");
                X = m_Ops.Copy(X);
            }
            // Load const
            else if (l.type == Layer.Type.Load)
            {
                Profiler.BeginSample ("Barracuda.Load");
            }
            // GEMM
            else if (l.type == Layer.Type.Dense)
            {
                Assert.AreEqual(inputs.Length, 3);
                Profiler.BeginSample ("Barracuda.Dense");
                X = m_Ops.Dense(X, inputs[1], inputs[2], GetAndVerifyFusedActivation(l));
            }
            // GEMM - optimized rank3 path
            else if (l.type == Layer.Type.Dense3)
            {
                Assert.AreEqual(inputs.Length, 3);
                Profiler.BeginSample ("Barracuda.Dense3");
                X = m_Ops.Dense3(X, inputs[1], inputs[2]);
            }
            // MatMul
            else if (l.type == Layer.Type.MatMul)
            {
                Assert.AreEqual(inputs.Length, 2);
                Profiler.BeginSample ("Barracuda.MatMul");

                if (l.pool == null || l.pool.Length == 0)
                    X = m_Ops.MatMul(X, -1, inputs[1], -1);
                else
                    X = m_Ops.MatMul(X, l.pool[0], inputs[1], l.pool[1]);
            }
            // 2D
            else if (l.type == Layer.Type.Conv2D)
            {
                Assert.AreEqual(inputs.Length, 3);
                Profiler.BeginSample ("Barracuda.Conv2D");
                var pad = X.AdjustPadToKernel(inputs[1], l.stride, l.pad);
                X = m_Ops.Conv2D(X, inputs[1], inputs[2], l.stride, pad, GetAndVerifyFusedActivation(l));
            }
            else if (l.type == Layer.Type.DepthwiseConv2D)
            {
                Assert.AreEqual(inputs.Length, 3);
                Profiler.BeginSample ("Barracuda.DepthwiseConv2D");
                var pad = X.AdjustPadToKernel(inputs[1], l.stride, l.pad);
                X = m_Ops.DepthwiseConv2D(X, inputs[1], inputs[2], l.stride, pad, GetAndVerifyFusedActivation(l));
            }
            else if (l.type == Layer.Type.Conv2DTrans)
            {
                Assert.AreEqual(inputs.Length, 3);
                Profiler.BeginSample ("Barracuda.Conv2DTrans");
                // pool size is treated as output_adjustment aka output_padding here
                var outputAdjustment = l.pool;
                var pad = X.AdjustPadToKernel(inputs[1], l.stride, l.pad);
                X = m_Ops.Conv2DTrans(X, inputs[1], inputs[2], l.stride, pad, outputAdjustment, GetAndVerifyFusedActivation(l));
            }
            else if (l.type == Layer.Type.Upsample2D)
            {
                Profiler.BeginSample ("Barracuda.Upsample2D");
                // pool size is treated as upsample scale coefficient here
                var scale = l.pool;
                // axis is treated as upsample point/bilinear flag
                var bilinear = l.axis > 0;
                if (scale.Length == 0 && inputs.Length > 1)
                {
                    var scaleTensor = inputs[1];
                    Assert.AreEqual(scaleTensor.length, 4);
                    scale = new int[] {(int)scaleTensor[2], (int)scaleTensor[1]};
                }
                X = m_Ops.Upsample2D(X, scale, bilinear);
            }
            else if (l.type == Layer.Type.Resample2D)
            {
                Profiler.BeginSample("Barracuda.Resample2D");
                // pool size is treated as resample size here
                var size = l.pool;
                // axis is treated as upsample point/bilinear flag
                var bilinear = l.axis > 0;
                if (inputs.Length > 1)
                {
                    var sizeTensor = inputs[1];
                    Assert.IsTrue(sizeTensor.length == 4 || sizeTensor.length == 8);
                    if (sizeTensor.length == 4)
                        size = new int[] {(int)sizeTensor[2], (int)sizeTensor[1]};
                    else
                        size = new int[] {(int)sizeTensor[6], (int)sizeTensor[5]};
                }
                X = m_Ops.Resample2D(X, size, bilinear);
            }
            else if (l.type == Layer.Type.DepthToSpace)
            {
                Profiler.BeginSample("Barracuda.DepthToSpace");
                // pool size is treated as blocksize
                var blocksize = l.pool;
                // axis is treated as mode enum
                var mode = (Layer.DepthToSpaceMode) l.axis;
                X = m_Ops.DepthToSpace(X, blocksize, mode);
            }
            else if (l.type == Layer.Type.SpaceToDepth)
            {
                Profiler.BeginSample("Barracuda.SpaceToDepth");
                // pool size is treated as blocksize
                var blocksize = l.pool;
                X = m_Ops.SpaceToDepth(X, blocksize);
            }
            else if (l.type == Layer.Type.MaxPool2D)
            {
                Profiler.BeginSample ("Barracuda.MaxPool2D");
                var pad = X.AdjustPadToPool(l.pool, l.stride, l.pad);
                X = m_Ops.MaxPool2D(X, l.pool, l.stride, pad);
            }
            else if (l.type == Layer.Type.AvgPool2D)
            {
                Profiler.BeginSample ("Barracuda.AvgPool2D");
                var pad = X.AdjustPadToPool(l.pool, l.stride, l.pad);
                X = m_Ops.AvgPool2D(X, l.pool, l.stride, pad);
            }
            else if (l.type == Layer.Type.GlobalMaxPool2D)
            {
                Profiler.BeginSample ("Barracuda.GlobalMaxPool2D");
                X = m_Ops.GlobalMaxPool2D(X);
            }
            else if (l.type == Layer.Type.GlobalAvgPool2D)
            {
                Profiler.BeginSample ("Barracuda.GlobalAvgPool2D");
                X = m_Ops.GlobalAvgPool2D(X);
            }
            else if (l.type == Layer.Type.Border3D)
            {
                Profiler.BeginSample ("Barracuda.Border3D");

                Assert.IsNotNull(l.pad);
                // NOTE: beta is used to retrieve fillin value
                // because beta is 0 by default (while alpha is 1 by default)
                // 0 value is more inline with zero padding
                float fillValue = l.beta;
                // legacy support
                if (l.pad.Length == 6)
                    X = m_Ops.Border3D(X, new[] { l.pad[0], l.pad[1], l.pad[2], 0, l.pad[3], l.pad[4], l.pad[5], 0 }, fillValue);
                else
                    X = m_Ops.Border3D(X, l.pad, fillValue);
            }
            else if (l.type == Layer.Type.Border2D)
            {
                Profiler.BeginSample ("Barracuda.Border2D");

                Assert.IsNotNull(l.pad);
                // NOTE: beta is used to retrieve filling value
                // because beta is 0 by default (while alpha is 1 by default)
                // 0 value is more inline with zero padding
                float fillValue = l.beta;

                // legacy support
                if(l.pad.Length == 4)
                    X = m_Ops.Border2D(X, new[] { l.pad[0], l.pad[1], 0, l.pad[2], l.pad[3], 0 }, fillValue);
                else
                    X = m_Ops.Border2D(X, l.pad, fillValue);
            }
            else if (l.type == Layer.Type.Pad2DReflect)
            {
                Profiler.BeginSample ("Barracuda.Pad2DReflect");

                Assert.IsNotNull(l.pad);

                // legacy support
                if(l.pad.Length == 4)
                    X = m_Ops.Pad2DReflect(X, new[] { l.pad[0], l.pad[1], 0, l.pad[2], l.pad[3], 0 });
                else
                    X = m_Ops.Pad2DReflect(X, l.pad);
            }
            else if (l.type == Layer.Type.Pad2DSymmetric)
            {
                Profiler.BeginSample ("Barracuda.Pad2DSymmetric");

                Assert.IsNotNull(l.pad);

                // legacy support
                if(l.pad.Length == 4)
                    X = m_Ops.Pad2DSymmetric(X, new[] { l.pad[0], l.pad[1], 0, l.pad[2], l.pad[3], 0 });
                else
                    X = m_Ops.Pad2DSymmetric(X, l.pad);
            }
            else if (l.type == Layer.Type.Pad2DEdge)
            {
                Profiler.BeginSample ("Barracuda.Pad2DEdge");

                Assert.IsNotNull(l.pad);

                // legacy support
                if(l.pad.Length == 4)
                    X = m_Ops.Pad2DEdge(X, new[] { l.pad[0], l.pad[1], 0, l.pad[2], l.pad[3], 0 });
                else
                    X = m_Ops.Pad2DEdge(X, l.pad);
            }
            // 3D
            else if (l.type == Layer.Type.Upsample3D)
            {
                Profiler.BeginSample ("Barracuda.Upsample3D");
                // pool size is treated as upsample scale coefficient here
                var scale = l.pool;
                // axis is treated as upsample point/bilinear flag
                var trilinear = l.axis > 0;
                if (scale.Length == 0 && inputs.Length > 1)
                {
                    var scaleTensor = inputs[1];
                    Assert.AreEqual(scaleTensor.length, 5);
                    scale = new int[] {(int)scaleTensor[3], (int)scaleTensor[2], (int)scaleTensor[1]};
                }
                X = m_Ops.Upsample3D(X, scale, trilinear);
            }
            else if (l.type == Layer.Type.Conv3D)
            {
                Assert.AreEqual(inputs.Length, 3);
                Profiler.BeginSample ("Barracuda.Conv3D");
                var pad = X.AdjustPadToKernel(inputs[1], l.stride, l.pad);
                X = m_Ops.Conv3D(X, inputs[1], inputs[2], l.stride, pad, GetAndVerifyFusedActivation(l));
            }
            else if (l.type == Layer.Type.Conv3DTrans ||
                l.type == Layer.Type.MaxPool3D ||
                l.type == Layer.Type.AvgPool3D ||
                l.type == Layer.Type.GlobalMaxPool3D ||
                l.type == Layer.Type.GlobalAvgPool3D ||
                l.type == Layer.Type.Border3D)
            {
                throw new NotImplementedException($"{l.type} operations are not implemented yet!");
            }
            else if (l.type == Layer.Type.ScaleBias)
            {
                Assert.AreEqual(inputs.Length, 3);
                Profiler.BeginSample ("Barracuda.ScaleBias");
                X = m_Ops.ScaleBias(X, inputs[1], inputs[2]);
            }
            else if (l.type == Layer.Type.Normalization)
            {
                Assert.AreEqual(inputs.Length, 3);
                Profiler.BeginSample ("Barracuda.Normalization");
                // @TODO: support other types of Normalization at test time.
                // Currently supported only pool=1 (InstanceNormalization)

                // NOTE: beta is used to retrieve epsilon value
                // because beta is 0 by default (while alpha is 1 by default)
                // 0 value is more inline with very small epsilon
                var epsilon = l.beta;
                if (epsilon == 0)
                    epsilon = Mathf.Epsilon; // safety check to prevent division by zero

                X = m_Ops.Normalization(X, inputs[1], inputs[2], 1, l.axis, epsilon, GetAndVerifyFusedActivation(l));
            }
            else if (l.type == Layer.Type.LRN)
            {
                Profiler.BeginSample ("Barracuda.LRN");

                Assert.IsNotNull(l.pool);
                Assert.AreEqual(l.pool.Length, 1);
                int count = l.pool[0];
                float bias = (l.weights.Length > 0) ? l.weights[l.datasets[0].offset + 0] : 1.0f;
                X = m_Ops.LRN(X, l.alpha, l.beta, bias, count);
            }
            // Stochastic layers
            else if (l.type == Layer.Type.Dropout)
            {
                Profiler.BeginSample ("Barracuda.Dropout");

                X = m_Ops.Dropout(X, l.alpha);
            }
            else if (l.type == Layer.Type.RandomNormal)
            {
                Profiler.BeginSample ("Barracuda.RandomNormal");

                Assert.IsNotNull(l.pool);
                // pool size is treated as shape constant, if not empty
                // otherwise shape of the previous tensor is used
                var shape = X.shape;
                if (l.pool.Length > 0)
                    shape = new TensorShape(l.pool);

                int seed = (l.pad.Length > 0) ? l.pad[0] : 1337;
                    seed = seed == 0 ? 1337 : seed;
                float scale = l.alpha, mean = l.beta;
                X = m_Ops.RandomNormal(shape, mean, scale, seed);
            }
            else if (l.type == Layer.Type.RandomUniform)
            {
                Profiler.BeginSample ("Barracuda.RandomUniform");

                Assert.IsNotNull(l.pool);
                // pool size is treated as shape constant, if not empty
                // otherwise shape of the previous tensor is used
                var shape = X.shape;
                if (l.pool.Length > 0)
                    shape = new TensorShape(l.pool);

                int seed = (l.pad.Length > 0) ? l.pad[0] : 1337;
                    seed = seed == 0 ? 1337 : seed;
                float scale = l.alpha, mean = l.beta;
                X = m_Ops.RandomUniform(shape, mean, scale, seed);
            }
            else if (l.type == Layer.Type.Multinomial)
            {
                Profiler.BeginSample ("Barracuda.Multinomial");

                Assert.IsNotNull(l.pool);
                Assert.AreEqual(l.pool.Length, 1);

                int count = l.pool[0];
                int seed = (l.pad.Length > 0) ? l.pad[0] : 1337;
                    seed = seed == 0 ? 1337 : seed;
                X = m_Ops.Multinomial(X, count, seed);
            }
            else if (l.type == Layer.Type.OneHot)
            {
                Profiler.BeginSample ("Barracuda.OneHot");

                Assert.IsNotNull(l.pool);
                Assert.AreEqual(l.pool.Length, 1);
                int depth = l.pool[0];
                float on = l.alpha, off = l.beta;
                int inputRank = l.axis;
                inputRank = inputRank < 0 ? X.dimensions : inputRank;
                X = m_Ops.OneHot(X, depth, on, off, inputRank);
            }
            else if (l.type == Layer.Type.RoiAlign)
            {
                Profiler.BeginSample ("Barracuda.RoiAlign");

                X = m_Ops.RoiAlign(X, inputs[1], inputs[2], l.pool[0], l.pool[1], l.axis, l.alpha);
            }
            else if (l.type == Layer.Type.TopKIndices)
            {
                Profiler.BeginSample ("Barracuda.TopKIndices");

                bool largest = (l.pad[0] == 1);
                bool sorted = (l.pad[1] == 1);

                X = m_Ops.TopKIndices(X, (int)inputs[1][0], l.axis, largest, sorted);
            }
            else if (l.type == Layer.Type.TopKValues)
            {
                Profiler.BeginSample ("Barracuda.TopKValues");

                X = m_Ops.TopKValues(X, inputs[1], l.axis);
            }
            else if (l.type == Layer.Type.NonZero)
            {
                Profiler.BeginSample ("Barracuda.NonZero");

                X = m_Ops.NonZero(X);
            }
            // Broadcast layers
            else if (l.type == Layer.Type.Add)
            {
                Profiler.BeginSample ("Barracuda.Add");

                X = m_Ops.Add(inputs);
            }
            else if (l.type == Layer.Type.Sub)
            {
                Profiler.BeginSample ("Barracuda.Sub");

                X = m_Ops.Sub(inputs);
            }
            else if (l.type == Layer.Type.Mul)
            {
                Profiler.BeginSample ("Barracuda.Mul");

                X = m_Ops.Mul(inputs);
            }
            else if (l.type == Layer.Type.Div)
            {
                Profiler.BeginSample ("Barracuda.Div");

                X = m_Ops.Div(inputs);
            }
            else if (l.type == Layer.Type.Pow)
            {
                Profiler.BeginSample ("Barracuda.Pow");

                X = m_Ops.Pow(inputs);
            }
            else if (l.type == Layer.Type.Min)
            {
                Profiler.BeginSample ("Barracuda.Min");

                X = m_Ops.Min(inputs);
            }
            else if (l.type == Layer.Type.Max)
            {
                Profiler.BeginSample ("Barracuda.Max");

                X = m_Ops.Max(inputs);
            }
            else if (l.type == Layer.Type.Mean)
            {
                Profiler.BeginSample ("Barracuda.Mean");

                X = m_Ops.Mean(inputs);
            }
            // Reduction layers
            else if (l.type == Layer.Type.ReduceMax  ||
                     l.type == Layer.Type.ReduceMean ||
                     l.type == Layer.Type.ReduceMin  ||
                     l.type == Layer.Type.ReduceProd ||
                     l.type == Layer.Type.ReduceSum ||
                     l.type == Layer.Type.ArgMax ||
                     l.type == Layer.Type.ArgMin)
            {
                Profiler.BeginSample ("Barracuda.Reduce");
                switch (l.type)
                {
                    case Layer.Type.ReduceMax:
                        X = m_Ops.ReduceMax(X, l.axis);
                        break;
                    case Layer.Type.ReduceMean:
                        X = m_Ops.ReduceMean(X, l.axis);
                        break;
                    case Layer.Type.ReduceMin:
                        X = m_Ops.ReduceMin(X, l.axis);
                        break;
                    case Layer.Type.ReduceProd:
                        X = m_Ops.ReduceProd(X, l.axis);
                        break;
                    case Layer.Type.ReduceSum:
                        X = m_Ops.ReduceSum(X, l.axis);
                        break;
                    case Layer.Type.ArgMax:
                        X = m_Ops.ArgMax(X, l.axis);
                        break;
                    case Layer.Type.ArgMin:
                        X = m_Ops.ArgMin(X, l.axis);
                        break;
                }
            }
            else if (
                l.type == Layer.Type.ReduceL1 ||
                l.type == Layer.Type.ReduceL2 ||
                l.type == Layer.Type.ReduceLogSum ||
                l.type == Layer.Type.ReduceLogSumExp ||
                l.type == Layer.Type.ReduceSumSquare)
            {
                throw new NotImplementedException("This reduction operation is not implemented yet!");
            }
            // Logical operators with broadcast
            else if (l.type == Layer.Type.Greater)
            {
                Assert.AreEqual(inputs.Length, 2);
                Profiler.BeginSample ("Barracuda.Greater");
                X = m_Ops.Greater(X, inputs[1]);
            }
            else if (l.type == Layer.Type.GreaterEqual)
            {
                Assert.AreEqual(inputs.Length, 2);
                Profiler.BeginSample("Barracuda.GreaterEqual");
                X = m_Ops.GreaterEqual(X, inputs[1]);
            }
            else if (l.type == Layer.Type.Less)
            {
                Assert.AreEqual(inputs.Length, 2);
                Profiler.BeginSample("Barracuda.Less");
                X = m_Ops.Less(X, inputs[1]);
            }
            else if (l.type == Layer.Type.LessEqual)
            {
                Assert.AreEqual(inputs.Length, 2);
                Profiler.BeginSample("Barracuda.LessEqual");
                X = m_Ops.LessEqual(X, inputs[1]);
            }
            else if (l.type == Layer.Type.Equal)
            {
                Assert.AreEqual(inputs.Length, 2);
                Profiler.BeginSample("Barracuda.Equal");
                X = m_Ops.Equal(X, inputs[1]);
            }
            else if (l.type == Layer.Type.LogicalOr)
            {
                Assert.AreEqual(inputs.Length, 2);
                Profiler.BeginSample("Barracuda.LogicalOr");
                X = m_Ops.LogicalOr(X, inputs[1]);
            }
            else if (l.type == Layer.Type.LogicalAnd)
            {
                Assert.AreEqual(inputs.Length, 2);
                Profiler.BeginSample("Barracuda.LogicalAnd");
                X = m_Ops.LogicalAnd(X, inputs[1]);
            }
            else if (l.type == Layer.Type.LogicalXor)
            {
                Assert.AreEqual(inputs.Length, 2);
                Profiler.BeginSample("Barracuda.LogicalXor");
                X = m_Ops.LogicalXor(X, inputs[1]);
            }
            else if (l.type == Layer.Type.LogicalNot)
            {
                Profiler.BeginSample("Barracuda.LogicalNot");
                X = m_Ops.LogicalNot(X);
            }
            else if (l.type == Layer.Type.Sign)
            {
                Profiler.BeginSample("Barracuda.Sign");
                X = m_Ops.Sign(X);
            }
            else if (l.type == Layer.Type.Where)
            {
                Assert.AreEqual(inputs.Length, 3);
                Profiler.BeginSample("Barracuda.Where");
                X = m_Ops.Where(X, inputs[1], inputs[2]);
            }
            // Shape affecting layers
            else if (l.type == Layer.Type.Flatten)
            {
                Profiler.BeginSample ("Barracuda.Flatten");
                X = m_Ops.Flatten(X);
            }
            else if (l.type == Layer.Type.Reshape)
            {
                Profiler.BeginSample ("Barracuda.Reshape");

                // pool is treated as the shape, if not empty
                var size = l.pool;

                Assert.IsNotNull(size);
                if (size.Length == 0 && inputs.Length > 1)
                {
                    switch (l.axis)
                    {
                        // Legacy - use the shape of the input tensor as the shape
                        case -1:
                            size = inputs[1].shape.ToArray();
                            break;

                        // Use the tensor values as the shape
                        case 1:
                            Tensor shapeTensor = inputs[1];
                            size = new [] { 1, 1, 1, 1 };
                            for (var i = 0; i < shapeTensor.length; i++)
                            {
                                size[i] = (int)shapeTensor[i];
                            }
                            break;
                    }
                }

                var newShape = X.shape.Reshape(size);
                X = m_Ops.Reshape(X, newShape);
            }
            else if (l.type == Layer.Type.Expand)
            {
                Profiler.BeginSample("Barracuda.Expand");

                var shape = l.pool;
                if (inputs.Length == 1)
                {
                    // pool size is treated as new shape
                    Assert.IsNotNull(shape);
                    Assert.IsTrue(shape.Length == 8 || shape.Length == 4);

                    if (shape.Length == 4)
                        shape = new[] { 1, 1, l.pool[0], 1, 1, l.pool[1], l.pool[2], l.pool[3] };
                }
                else
                {
                    // dynamic shape support: shape operations cannot be performed on padded shapes, need to expand it here
                    var refShape = new float[inputs[1].length];
                    Array.Copy(inputs[1].ToReadOnlyArray(), refShape, inputs[1].length);
                    shape = Compiler.IRShapeInferenceHelper.ShapeInference.OnnxLayoutToBarracudaTensorShape(Array.ConvertAll(refShape, x => (int)x)).ToArray();
                }

                var inputShape = new[] { X.shape.sequenceLength, X.shape.numberOfDirections, X.shape.batch, X.shape.extraDimension, X.shape.depth, X.shape.height, X.shape.width, X.shape.channels };
                var tiledShape = new int[8];

                for (int i = 0; i < 8; i++)
                    tiledShape[i] = Mathf.Max(shape[i], inputShape[i]);

                if (Enumerable.SequenceEqual(tiledShape, X.shape.ToArray()))
                    X = m_Ops.Copy(X);
                else
                    X = m_Ops.Expand(X, new TensorShape(tiledShape));
            }
            else if (l.type == Layer.Type.Shape)
            {
                Profiler.BeginSample("Barracuda.Shape");

                X = m_Ops.Shape(X, l.axis);
            }
            else if (l.type == Layer.Type.Transpose)
            {
                Profiler.BeginSample ("Barracuda.Transpose");

                var permutations = l.pool;
                if (permutations == null)
                    X = m_Ops.Transpose(X);
                else
                {
                    // if transpose does not change internal memory layout, skip
                    if(ModelAnalyzer.DoesTransposeChangeTensorLayout(X.shape, permutations))
                        X = m_Ops.Reshape(X, X.shape.Permute(permutations));
                    else
                        X = m_Ops.Transpose(X, permutations);
                }
            }
            else if (l.type == Layer.Type.Gather)
            {
                Profiler.BeginSample ("Barracuda.Gather");
                X = m_Ops.Gather(inputs, l.axis);

                // Gather assume flat indices, if indices has a rank > 1, we need to expand the generated tensor
                if (l.pool != null && l.pool.Length == 2 && l.pool[1] > 1)
                {
                    int xRank = l.pool[0];
                    int indicesRank = l.pool[1];
                    var xShape = Compiler.IRShapeInferenceHelper.ShapeInference.BarracudaShapeToList(X.shape, xRank);
                    var indicesShape = Compiler.IRShapeInferenceHelper.ShapeInference.BarracudaShapeToList(inputs[1].shape, indicesRank);

                    int axis = Compiler.IRShapeInferenceHelper.ShapeInference.BarracudaAxisToTensor(l.axis, xRank);
                    xShape.InsertRange(axis, indicesShape);
                    xShape.RemoveAt(axis + indicesShape.Count);

                    X = m_Ops.Reshape(X, new TensorShape(Compiler.IRShapeInferenceHelper.ShapeInference.BarracudaLayoutToTensorShapeLayout(xShape.ToArray())));

                    // rank 2 -> 3
                    if (xRank == 2 && xShape.Count == 3)
                        X = m_Ops.Transpose(X, new int[] {0,1,3,2});
                }
            }
            else if (l.type == Layer.Type.ScatterND)
            {
                Profiler.BeginSample ("Barracuda.ScatterND");

                X = m_Ops.ScatterND(X, inputs[1], inputs[2], (Layer.ScatterNDReductionMode)l.axis);
            }
            else if (l.type == Layer.Type.NonMaxSuppression)
            {
                Profiler.BeginSample("Barracuda.NonMaxSuppression");

                int maxOutputBoxesPerClass = 0;
                float iouThreshold = 0f;
                float scoreThreshold = 0f;

                if (l.pool.Length > 0)
                {
                    maxOutputBoxesPerClass = l.pool[0];
                    iouThreshold = l.alpha;
                    scoreThreshold = l.beta;
                }
                else
                {
                    if (inputs.Length > 2)
                        maxOutputBoxesPerClass = (int)inputs[2][0];

                    if (inputs.Length > 3)
                        iouThreshold = inputs[3][0];

                    if (inputs.Length > 4)
                        scoreThreshold = inputs[4][0];
                }

                X = m_Ops.NonMaxSuppression(inputs, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, l.axis);
            }
            else if (l.type == Layer.Type.LSTM)
            {
                Profiler.BeginSample("Barracuda.LSTM");

                bool constantWRB = l.datasets.Length > 0;

                int hidden_index;
                int cell_index;

                Tensor[] w, r, wb, rb;

                using (var td = new TensorScope())
                {
                    TensorScope.F _ = td._; // Shorthand

                    if (constantWRB)
                    {
                        w = new[]
                        {
                            l.DataSetToTensor(0),
                            l.DataSetToTensor(1),
                            l.DataSetToTensor(2),
                            l.DataSetToTensor(3)
                        };

                        r = new[]
                        {
                            l.DataSetToTensor(4),
                            l.DataSetToTensor(5),
                            l.DataSetToTensor(6),
                            l.DataSetToTensor(7)
                        };

                        wb = new[]
                        {
                            l.DataSetToTensor(8),
                            l.DataSetToTensor(9),
                            l.DataSetToTensor(10),
                            l.DataSetToTensor(11)
                        };

                        rb = new[]
                        {
                            l.DataSetToTensor(12),
                            l.DataSetToTensor(13),
                            l.DataSetToTensor(14),
                            l.DataSetToTensor(15)
                        };

                        hidden_index = 1;
                        cell_index = 2;
                    }
                    else
                    {
                        // Barracuda N1WC [num_directions, 4*hidden_size, input_size] -> Barracuda NC [4*hidden_size, input_size]
                        // (i.e. drop directions since they are unsupported)
                        Tensor W = _(m_Ops.Transpose(inputs[1], new[] { 2, 0, 1, 3 }));

                        // Barracuda N1WC [num_directions, 4*hidden_size, hidden_size] -> Barracuda NC [4*hidden_size, input_size]
                        // (i.e. drop directions since they are unsupported)
                        Tensor R = _(m_Ops.Transpose(inputs[2], new[] { 2, 0, 1, 3 }));
                        Tensor B = inputs[3];

                        OpsUtils.SplitWRBForLSTM(m_Ops, W, R, B, out w, out r, out wb, out rb);

                        hidden_index = 4;
                        cell_index = 5;
                    }

                    // Tag for auto-disposal
                    for (int i = 0; i < w.Length; i++)
                    {
                        _(w[i]);
                        _(r[i]);
                        _(wb[i]);
                        _(rb[i]);
                    }

                    Tensor originalHidden = inputs[hidden_index];
                    Tensor originalCell = inputs[cell_index];

                    Tensor[] Y = m_Ops.LSTM(X, w, r, wb, rb, originalHidden, originalCell);

                    X = Y[0];
                    Tensor hiddenFinal = Y[1];
                    Tensor cellFinal = Y[2];

                    // We don't support multiple outputs from layers, so set memories directly, which gets picked
                    // up by subsequent output layers that load memories
                    var memories = m_Model.memories;
                    for (int m = 0; m < memories.Count; m++)
                    {
                        Model.Memory memory = memories[m];
                        if (l.inputs[hidden_index].Contains(memory.input))
                        {
                            _(originalHidden);
                            m_Vars.SetInput(memory.input, hiddenFinal);
                        }
                        else if (l.inputs[cell_index].Contains(memory.input))
                        {
                            _(originalCell);
                            m_Vars.SetInput(memory.input, cellFinal);
                        }
                    }
                }
            }
            else if (l.type == Layer.Type.Concat)
            {
                Profiler.BeginSample ("Barracuda.Concat");
                X = m_Ops.Concat(inputs, l.axis);
            }
            else if (l.type == Layer.Type.StridedSlice)
            {
                Profiler.BeginSample ("Barracuda.StridedSlice");

                Assert.IsNotNull(l.pad);
                Assert.IsNotNull(l.pool);
                Assert.IsNotNull(l.stride);
                X = m_Ops.StridedSlice(X, l.pad, l.pool, l.stride);
            }
            else if (l.type == Layer.Type.Tile)
            {
                Profiler.BeginSample ("Barracuda.Tile");

                var size = l.pool;
                if (size.Length == 0 && inputs.Length > 1)
                {
                    // dynamic shape support: shape operations cannot be performed on padded shapes, need to expand it here
                    var inputShape = new float[inputs[1].length];
                    Array.Copy(inputs[1].ToReadOnlyArray(), inputShape, inputs[1].length);
                    size = Compiler.IRShapeInferenceHelper.ShapeInference.OnnxLayoutToBarracudaTensorShape(Array.ConvertAll(inputShape, x => (int)x)).ToArray();
                }

                X = m_Ops.Tile(X, size);
            }
            else if(l.type == Layer.Type.ConstantOfShape)
            {
                Profiler.BeginSample ("Barracuda.ConstantOfShape");

                var size = inputs[0].shape;
                if (l.axis != 1)
                {
                    // dynamic shape support: shape operations cannot be performed on padded shapes, need to expand it here
                    var inputShape = new float[inputs[0].length];
                    Array.Copy(inputs[0].ToReadOnlyArray(), inputShape, inputs[0].length);
                    size = Compiler.IRShapeInferenceHelper.ShapeInference.OnnxLayoutToBarracudaTensorShape(Array.ConvertAll(inputShape, x => (int)x));
                }

                X = m_Ops.ConstantOfShape(size, X.dataType, l.alpha);
            }
            // Activations
            else if (l.type == Layer.Type.Activation)
            {
                Profiler.BeginSample ("Barracuda.Activation");

                if (l.activation == Layer.Activation.Relu)
                {
                    X = m_Ops.Relu(X);
                }
                else if (l.activation == Layer.Activation.Softmax)
                {
                    X = m_Ops.Softmax(X, l.axis);
                }
                else if (l.activation == Layer.Activation.LogSoftmax)
                {
                    X = m_Ops.LogSoftmax(X, l.axis);
                }
                else if (l.activation == Layer.Activation.Tanh)
                {
                    X = m_Ops.Tanh(X);
                }
                else if (l.activation == Layer.Activation.Softplus)
                {
                    X = m_Ops.Softplus(X);
                }
                else if (l.activation == Layer.Activation.Sigmoid)
                {
                    X = m_Ops.Sigmoid(X);
                }
                else if (l.activation == Layer.Activation.HardSigmoid)
                {
                    X = m_Ops.HardSigmoid(X, l.alpha, l.beta);
                }
                else if (l.activation == Layer.Activation.Relu6)
                {
                    X = m_Ops.Relu6(X);
                }
                else if (l.activation == Layer.Activation.Elu)
                {
                    X = m_Ops.Elu(X, l.alpha);
                }
                else if (l.activation == Layer.Activation.LeakyRelu)
                {
                    X = m_Ops.LeakyRelu(X, l.alpha);
                }
                else if (l.activation == Layer.Activation.Selu)
                {
                    X = m_Ops.Selu(X, l.alpha, l.beta);
                }
                else if (l.activation == Layer.Activation.Swish)
                {
                    X = m_Ops.Swish(X);
                }
                else if (l.activation == Layer.Activation.PRelu)
                {
                    Assert.AreEqual(inputs.Length, 2);
                    X = m_Ops.PRelu(X, inputs[1]);
                }
                else if (
                    l.activation == Layer.Activation.Softsign ||
                    l.activation == Layer.Activation.Hardmax)
                {
                    throw new NotImplementedException("This activation function is not implemented yet!");
                }
                else if (l.activation == Layer.Activation.Abs)
                {
                    X = m_Ops.Abs(X);
                }
                else if (l.activation == Layer.Activation.Neg)
                {
                    X = m_Ops.Neg(X);
                }
                else if (l.activation == Layer.Activation.Ceil)
                {
                    X = m_Ops.Ceil(X);
                }
                else if (l.activation == Layer.Activation.Clip)
                {
                    X = m_Ops.Clip(X, l.alpha, l.beta);
                }
                else if (l.activation == Layer.Activation.Floor)
                {
                    X = m_Ops.Floor(X);
                }
                else if (l.activation == Layer.Activation.Round)
                {
                    X = m_Ops.Round(X);
                }
                else if (l.activation == Layer.Activation.Reciprocal)
                {
                    X = m_Ops.Reciprocal(X);
                }
                else if (l.activation == Layer.Activation.Pow)
                {
                    X = m_Ops.Pow(X, l.alpha);
                }
                else if (l.activation == Layer.Activation.Exp)
                {
                    X = m_Ops.Exp(X);
                }
                else if (l.activation == Layer.Activation.Log)
                {
                    X = m_Ops.Log(X);
                }
                else if (l.activation == Layer.Activation.Sqrt)
                {
                    X = m_Ops.Sqrt(X);
                }
                else if (l.activation == Layer.Activation.Acos)
                {
                    X = m_Ops.Acos(X);
                }
                else if (l.activation == Layer.Activation.Acosh)
                {
                    X = m_Ops.Acosh(X);
                }
                else if (l.activation == Layer.Activation.Asin)
                {
                    X = m_Ops.Asin(X);
                }
                else if (l.activation == Layer.Activation.Asinh)
                {
                    X = m_Ops.Asinh(X);
                }
                else if (l.activation == Layer.Activation.Atan)
                {
                    X = m_Ops.Atan(X);
                }
                else if (l.activation == Layer.Activation.Atanh)
                {
                    X = m_Ops.Atanh(X);
                }
                else if (l.activation == Layer.Activation.Cos)
                {
                    X = m_Ops.Cos(X);
                }
                else if (l.activation == Layer.Activation.Cosh)
                {
                    X = m_Ops.Cosh(X);
                }
                else if (l.activation == Layer.Activation.Sin)
                {
                    X = m_Ops.Sin(X);
                }
                else if (l.activation == Layer.Activation.Sinh)
                {
                    X = m_Ops.Sinh(X);
                }
                else if (l.activation == Layer.Activation.Tan)
                {
                    X = m_Ops.Tan(X);
                }
                else if (l.activation == Layer.Activation.Erf)
                {
                    X = m_Ops.Erf(X);
                }
                else
                {
                    X = m_Ops.Copy(X);
                }
            }
            else
            {
                Profiler.BeginSample ("Barracuda.NotImplemented");
                Assert.IsTrue(l.type == Layer.Type.Nop, $"Layer type {l.type} not explicitly handled");
            }

#if ENABLE_BARRACUDA_STATS
            m_Ops.GetModelExecutionsReporter()?.TakeMemorySnapshot(m_Ops, m_Vars, "After layer",l);
#endif //ENABLE_BARRACUDA_STATS
            m_Vars.DisposeAfterLayer(l);
            m_Vars.Store(l, X);
            m_SyncTensor = X;

            // optype
            Profiler.EndSample();

            // layer.name
            Profiler.EndSample();
#if ENABLE_BARRACUDA_STATS
            m_Ops.GetModelExecutionsReporter()?.LayerExecutionCompleted();
#endif //ENABLE_BARRACUDA_STATS

            yield return null;
        }

        // request ResetAllocator before next Execute() starts
        m_AllocatorIsOccupied = false;

        if (m_Verbose)
            D.Log(m_Vars.GetAllocator());
#if ENABLE_BARRACUDA_STATS
        m_Ops.GetModelExecutionsReporter()?.ModelExecutionCompleted();
        m_Ops.GetModelExecutionsReporter()?.TakeMemorySnapshot(m_Ops, m_Vars, "After model execution");
#endif //ENABLE_BARRACUDA_STATS
    }

    /// <inheritdoc/>
    public virtual Tensor PeekOutput()
    {
        Profiler.BeginSample("Barracuda.PeekOutput");
        var X = m_Vars.PeekOutput(m_DefaultOutputName);

        if (X.batch <= m_MaxBatchThatAutoTriggersAsyncDownload &&
            X.flatWidth <= m_MaxFlatWidthThatAutoTriggersAsyncDownload) // tensor is small and most likely will be accessed on CPU,
            X.PrepareCacheForAccess(blocking:false);                    // thus schedule non-blocking download from GPU/NPU to CPU
        Profiler.EndSample();

        return X;
    }

    /// <inheritdoc/>
    public virtual Tensor PeekOutput(string name)
    {
        Profiler.BeginSample("Barracuda.PeekOutput");
        var X = m_Vars.PeekOutput(name);

        if (X.batch <= m_MaxBatchThatAutoTriggersAsyncDownload &&
            X.flatWidth <= m_MaxFlatWidthThatAutoTriggersAsyncDownload) // tensor is small and most likely will be accessed on CPU,
            X.PrepareCacheForAccess(blocking:false);                    // thus schedule non-blocking download from GPU/NPU to CPU
        Profiler.EndSample();

        return X;
    }

    /// <inheritdoc/>
    public virtual Tensor[] PeekConstants(string layerName)
    {
        Profiler.BeginSample("Barracuda.PeekConstants");
        return m_Vars.PeekConstants(layerName);
    }

    /// <summary>
    /// Execution summary
    /// </summary>
    /// <returns>execution summary</returns>
    public virtual string Summary()
    {
        return m_Vars.GetAllocator().ToString() + "\n" + m_Ops.ToString();
    }
}


internal class GenericVars : IVars, IVarsStatistics
{
    private Dictionary<string, Tensor> m_TensorsByName = new Dictionary<string, Tensor>();
    protected HashSet<Tensor> m_ModelTensors = new HashSet<Tensor>();
    protected Dictionary<Layer, Tensor[]> m_InputTensorsByLayer = new Dictionary<Layer, Tensor[]>();
    private Dictionary<string, int> m_LayerNameToId = new Dictionary<string, int>();
    private Dictionary<string, List<int>> m_LayerNameToDisposeWhenDone = new Dictionary<string, List<int>>();
    private Dictionary<int, Layer> m_LayerIdToLayer = new Dictionary<int, Layer>();
    protected StringCache m_StringCache = new StringCache();

    public GenericVars()
    {
    }

    ~GenericVars()
    {
        Dispose();
    }

    public virtual void Dispose()
    {
        foreach (var t in m_ModelTensors)
            t.Dispose();
        m_ModelTensors.Clear();

        // don't dispose input/user-owned tensors
        foreach (var ts in m_InputTensorsByLayer.Values)
            foreach (var t in ts)
            {
                if (IsTensorOwnedByInternalAllocator(t))
                    t.Dispose();
            }
        m_InputTensorsByLayer.Clear();

        m_LayerNameToId.Clear();
        m_LayerNameToDisposeWhenDone.Clear();
        m_LayerIdToLayer.Clear();
        m_StringCache.Clear();

        m_Allocator.Dispose();
    }

    private TensorCachingAllocator m_Allocator = new DefaultTensorAllocator();
    public virtual ITensorAllocator GetAllocator()
    {
        return m_Allocator;
    }

    public IEnumerable<IAllocatorStatistics> GetAllocatorsStatistics()
    {
        yield return m_Allocator;
    }

    public IEnumerable<ITensorStatistics> GetTensorsStatistics()
    {
        var tensors = new SortedDictionary<int, Tensor>();
        foreach (var modelTensor in m_ModelTensors)
        {
            tensors[modelTensor.uniqueId] = modelTensor;
        }
        foreach (var inputTensors in m_InputTensorsByLayer)
        {
            foreach (var inputTensor in inputTensors.Value)
            {
                tensors[inputTensor.uniqueId] = inputTensor;
            }
        }
        foreach (var tensorByName in m_TensorsByName)
        {
            tensors[tensorByName.Value.uniqueId] = tensorByName.Value;
        }

        foreach (var tensor in tensors)
        {
            yield return tensor.Value;
        }
    }

    protected virtual bool IsTensorOwnedByInternalAllocator(Tensor tensor)
    {
        return tensor.allocator == GetAllocator();
    }

    protected bool ValidateGlobalInputs(Model model, IDictionary<string, TensorShape> inputShapes)
    {
        bool valid = true;
        foreach (var i in model.inputs)
        {
            if (m_TensorsByName.ContainsKey(i.name) ||
                (inputShapes != null && inputShapes.ContainsKey(i.name)))
                continue;

            D.LogWarning("Global input is missing: " + i.name);
            valid = false;
        }
        return valid;
    }

    protected virtual Tensor[] PrepareLayerInputTensors(Model model, Layer layer, IOps ops)
    {
        int tensorIndex = 0;
        var tensors = new Tensor[layer.inputs.Length + layer.datasets.Length];

        foreach (var name in layer.inputs)
        {
            tensors[tensorIndex++] = new Tensor(1, 1, 1, 1, m_StringCache.Lookup(layer.name, "_dummy_in", tensorIndex));
        }
        foreach (var arg in layer.datasets)
        {
            var tensor = new Tensor(arg.shape, new SharedArrayTensorData(layer.weights, arg.shape, (int)arg.offset), arg.name);
            if (ops != null)
                tensor = ops.Prepare(tensor);
            m_ModelTensors.Add(tensor);
            tensors[tensorIndex++] = tensor;
        }
        return tensors;
    }

    public virtual void SetInput(string name, Tensor x)
    {
        m_TensorsByName[name] = x;
    }

    public virtual void PrepareStorage(Model model, IOps ops, IDictionary<string, TensorShape> inputShapes, bool takeoverWeights, DataType dataType)
    {
        ValidateGlobalInputs(model, inputShapes);

        m_LayerNameToId.Clear();
        m_LayerNameToDisposeWhenDone.Clear();
        m_LayerIdToLayer.Clear();

        for (var i = 0; i < model.layers.Count; i++)
        {
            var layer = model.layers[i];

            // prepare input placeholders and argument tensors only once per layer
            if (m_InputTensorsByLayer.ContainsKey(layer))
                continue;

            var tensors = PrepareLayerInputTensors(model, layer, ops);
            m_InputTensorsByLayer.Add(layer, tensors);
            if (takeoverWeights)
                layer.weights = null;
        }

        foreach (var mem in model.memories)
        {
            if (!m_TensorsByName.ContainsKey(mem.input))
            {
                // initialize memories that haven't been explicitly set
                var tensor = m_Allocator.Alloc(mem.shape, AllocScope.LayerOutput, dataType);
                SetInput(mem.input, tensor);
                m_ModelTensors.Add(tensor);
            }
        }

        // For each layer we find the latest downstream layer that has said layer as input
        // ex:
        // 0 -> 1 -> 4 -> 5 -> 8
        //   -> 2 -> 3  /     |
        //   -> 7 ------------/
        // latestDownstreamLayer:
        //  0 -> 7, 1 -> 4, 2 -> 3, 4 -> 5, 5 -> 8, 7 -> 8
        Dictionary<string, int> latestDownstreamLayer = new Dictionary<string, int>();
        for (var i = 0; i < model.layers.Count; i++)
        {
            var forLayer = model.layers[i];
            m_LayerNameToId[forLayer.name] = i;
            m_LayerIdToLayer[i] = forLayer;

            for (int j = 0; j < forLayer.inputs.Length; j++)
            {
                string input = forLayer.inputs[j];
                if (latestDownstreamLayer.ContainsKey(input))
                    latestDownstreamLayer[input] = Math.Max(latestDownstreamLayer[input], i);
                else
                    latestDownstreamLayer[input] = i;
            }
        }

        // now that we have the latestDownstreamLayer, we inverse the map
        // and compute when we reach a layer, what layers can I delete
        // in this case
        // 3 -> [2], 4 -> [1], 5 -> [4,3] , 7 -> [0], 8 -> [5,7]

        // keep layer if output or memories
        var preserve = new HashSet<string>(
            model.memories.Select(mem => mem.input).Concat(
            model.memories.Select(mem => mem.output)).Concat(
            model.inputs.Select(i => i.name)).Concat(
            model.outputs));

        foreach (var entry in latestDownstreamLayer)
        {
            if(preserve.Contains(entry.Key))
                continue;
            // input can be not specificed
            if(!m_LayerNameToId.ContainsKey(entry.Key))
                continue;

            var forLayer = m_LayerIdToLayer[entry.Value];
            if (m_LayerNameToDisposeWhenDone.ContainsKey(forLayer.name))
                m_LayerNameToDisposeWhenDone[forLayer.name].Add(m_LayerNameToId[entry.Key]);
            else
                m_LayerNameToDisposeWhenDone[forLayer.name] = new List<int>() { m_LayerNameToId[entry.Key] };
        }
    }

    public virtual Tensor[] GatherInputs(Layer forLayer)
    {
        var tensors = m_InputTensorsByLayer[forLayer];

        // fill in input variables
        int index = 0;
        foreach (var name in forLayer.inputs)
            tensors[index++] = PeekOutput(name);

        return tensors;
    }

    public virtual void PrepareStorage(Layer forLayer) {}

    public virtual void DisposeAfterLayer(Layer forLayer)
    {
        if(!m_LayerNameToDisposeWhenDone.ContainsKey(forLayer.name))
            return;

        foreach (var layerIdxToDispose in m_LayerNameToDisposeWhenDone[forLayer.name])
        {
            var l = m_LayerIdToLayer[layerIdxToDispose];
            var key = l.name;

            if (!(m_TensorsByName.ContainsKey(key) && !m_ModelTensors.Contains(m_TensorsByName[key])))
                continue;

            if (IsTensorOwnedByInternalAllocator(m_TensorsByName[key]))
                m_TensorsByName[key].Dispose();
            m_TensorsByName.Remove(key);
        }
    }

    public virtual void Store(Layer fromLayer, Tensor result)
    {
        // assign debug name
        result.name = fromLayer.name;

        // @TODO: implement Disposal of the old tensor that is going to be overwritten with new one
        // NOTE: need to make IWorker.CopyOutput to do real copy before enabling code below
        // otherwise there is a risk of Disposing tensor that is already owned by the user, if one calls CopyOutput on m_TensorsByName
        // if (m_TensorsByName.ContainsKey(fromLayer.name))
        // {
        //     var oldTensor = m_TensorsByName[fromLayer.name];
        //     if (oldTensor != result && IsTensorOwnedByInternalAllocator(oldTensor))
        //         oldTensor.Dispose();
        // }

        m_TensorsByName[fromLayer.name] = result;
    }

    public virtual Tensor PeekOutput(string name)
    {
        if (!m_TensorsByName.ContainsKey(name))
            D.LogWarning("GenericVars missing variable: " + name);

        return m_TensorsByName[name];
    }

    public virtual Tensor[] PeekConstants(string layerName)
    {
        if (!m_LayerNameToId.ContainsKey(layerName))
            D.LogWarning("GenericVars missing layer: " + layerName);

        var layerId = m_LayerNameToId[layerName];
        var l = m_LayerIdToLayer[layerId];
        var layerTensors = m_InputTensorsByLayer[l];
        var constantsTensors = new List<Tensor>();
        for (int i = 0; i < layerTensors.Length; ++i)
        {
            if (i < l.inputs.Length)
            {
                string inputLayerName = l.inputs[i];
                var inputLayerId = m_LayerNameToId[inputLayerName];
                var inputLayer = m_LayerIdToLayer[inputLayerId];
                if (inputLayer.type != Layer.Type.Load)
                    continue;
            }

            constantsTensors.Add(layerTensors[i]);
        }
        return constantsTensors.ToArray();
    }
}

internal class GenericVarsWithReuse : GenericVars
{
    private Model m_CachedModel;
    private bool m_LayerRequiresStorage = false;
    private HashSet<Layer> m_LayersWithStorage;
    private Tensor m_Temporary;
    private string m_TemporaryName = null;
    protected IDictionary<string, TensorShape> m_CachedInputShapes;

    internal bool layerRequiresStorage { get { return m_LayerRequiresStorage; } }
    protected Tensor temporary { get { return m_Temporary; } }

    protected void ReleaseTemporary()
    {
        m_TemporaryName = null;
        if (m_Temporary == null)
            return;

        if (IsTensorOwnedByInternalAllocator(m_Temporary))
            m_Temporary.Dispose();
        m_Temporary = null;
    }

    public override void PrepareStorage(Model model, IOps ops, IDictionary<string, TensorShape> inputShapes, bool takeoverWeights, DataType dataType)
    {
        if(m_CachedInputShapes != inputShapes)
        {
            m_CachedInputShapes = inputShapes;
            base.PrepareStorage(model, ops, inputShapes, takeoverWeights, dataType);
        }

        ReleaseTemporary();

        if (m_CachedModel != model)
            m_LayersWithStorage = ModelAnalyzer.FindLayersThatRequireStorage(model);
        m_CachedModel = model;

        Assert.AreEqual(m_Temporary, null);
    }

    public override void PrepareStorage(Layer forLayer)
    {
        base.PrepareStorage(forLayer);
        m_LayerRequiresStorage = m_LayersWithStorage.Contains(forLayer);
    }

    public override void Store(Layer fromLayer, Tensor result)
    {
        if (result != m_Temporary)
            ReleaseTemporary();

        // assign debug name
        result.name = fromLayer.name;

        if (layerRequiresStorage)
        {
            Assert.IsNotNull(result);
            base.Store(fromLayer, result);

            m_Temporary = null;
            m_TemporaryName = null;
        }
        else
        {
            Assert.IsTrue(m_Temporary == null || m_Temporary.tensorOnDevice == result.tensorOnDevice);

            m_Temporary = result;
            m_TemporaryName = fromLayer.name;
        }
    }

    public override Tensor PeekOutput(string name)
    {
        if (m_TemporaryName == name)
        {
            Assert.IsNotNull(m_Temporary);
            return m_Temporary;
        }
        return base.PeekOutput(name);
    }
}

internal class GenericVarsWithPreallocation : GenericVarsWithReuse, ITensorAllocator, IVarsStatistics
{
    public bool ShouldTrackTensorLeaks;
    private Model m_CachedModel;

    private DefaultTensorAllocator m_InferenceScopedPingPongAllocator = new DefaultTensorAllocator();
    private DefaultTensorAllocator m_InferenceScopedStorageAllocator = new DefaultTensorAllocator();
    private DefaultTensorAllocator m_LayerScopedAllocator = new DefaultTensorAllocator();

    public GenericVarsWithPreallocation()
    {
        m_InferenceScopedPingPongAllocator.name = "Inference ping pong Allocator";
        m_InferenceScopedStorageAllocator.name = "Inference storage Allocator";
        m_LayerScopedAllocator.name = "Layer scoped Allocator";
        ShouldTrackTensorLeaks = false;
    }

    public new IEnumerable<IAllocatorStatistics> GetAllocatorsStatistics()
    {
        yield return m_InferenceScopedPingPongAllocator;
        yield return m_InferenceScopedStorageAllocator;
        yield return m_LayerScopedAllocator;
    }

    /// <inheritdoc/>
    public virtual void PostLayerCleanup()
    {
        m_LayerScopedAllocator.Reset(keepCachedMemory:true);

        m_InferenceScopedPingPongAllocator.PostLayerCleanup();
        m_InferenceScopedStorageAllocator.PostLayerCleanup();
        m_LayerScopedAllocator.PostLayerCleanup();
    }

    public override void PrepareStorage(Model model, IOps ops, IDictionary<string, TensorShape> inputShapes, bool takeoverWeights, DataType dataType)
    {
        base.PrepareStorage(model, ops, inputShapes, takeoverWeights, dataType);

        if (m_CachedModel != model)
        {
            // pre-allocate 2 buffers that can be cycled for temporaries
            var allocator = m_InferenceScopedPingPongAllocator;

            var maxShape = ModelAnalyzer.FindLargestNecessaryTensorShape(model, inputShapes);
            var alloc1 = allocator.Alloc(maxShape, AllocScope.LayerOutput, dataType);
            var alloc2 = allocator.Alloc(maxShape, AllocScope.LayerOutput, dataType);
            alloc1 = ops.PrepareNoAlloc(alloc1);
            alloc2 = ops.PrepareNoAlloc(alloc2);
            allocator.Release(alloc1, false);
            allocator.Release(alloc2, false);
        }
        m_CachedModel = model;

        m_InferenceScopedPingPongAllocator.PostLayerCleanup();//reset allocation count
    }

    public override void DisposeAfterLayer(Layer forLayer)
    {
#if ENABLE_BARRACUDA_ERROR_ON_LEAKS
        if (ShouldTrackTensorLeaks && m_InferenceScopedPingPongAllocator.NumAllocatedBufferSinceCleanup != 0)
        {
            D.LogError($"TensorData leak detected: {m_InferenceScopedPingPongAllocator.NumAllocatedBufferSinceCleanup} tensorData(s)" +
                         $" was/were allocated in the ping pong allocator during execution of layer {forLayer} of type {forLayer.type}.");
        }
#endif

        PostLayerCleanup();

        base.DisposeAfterLayer(forLayer);
    }

    public override void Store(Layer fromLayer, Tensor result)
    {
        base.Store(fromLayer, result);

#if ENABLE_BARRACUDA_ERROR_ON_LEAKS
        if (ShouldTrackTensorLeaks && !m_InferenceScopedPingPongAllocator.IsPingPongReady)
        {
            D.LogError($"TensorData leak detected, one of the ping pong buffer was not released in layer {fromLayer} of type {fromLayer.type}.");
        }
#endif
    }

    public override ITensorAllocator GetAllocator()
    {
        return this;
    }
    protected override bool IsTensorOwnedByInternalAllocator(Tensor tensor)
    {
        var allocator = tensor.allocator;
        return allocator == m_InferenceScopedPingPongAllocator ||
               allocator == m_InferenceScopedStorageAllocator ||
               allocator == m_LayerScopedAllocator;
    }

    public virtual Tensor Alloc(TensorShape shape, AllocScope scope, DataType dataType)
    {
        if (scope == AllocScope.InternalToLayer)
            return m_LayerScopedAllocator.Alloc(shape, scope, dataType);

        if (layerRequiresStorage)
            return m_InferenceScopedStorageAllocator.Alloc(shape, scope, dataType);
        else
            return m_InferenceScopedPingPongAllocator.Alloc(shape, scope, dataType);
    }
    public virtual Tensor Alloc(TensorShape shape, ITensorData buffer, AllocScope scope, DataType dataType)
    {
        if (scope == AllocScope.InternalToLayer)
            return m_LayerScopedAllocator.Alloc(shape, buffer, scope, dataType);

        if (layerRequiresStorage)
            return m_InferenceScopedStorageAllocator.Alloc(shape, buffer, scope, dataType);
        else
            return m_InferenceScopedPingPongAllocator.Alloc(shape, buffer, scope, dataType);
    }
    public virtual void MoveToDevice(Tensor x, ITensorData newBuffer, ITensorData oldBuffer, bool disposeDetachedBufferHint)
    {
        x.allocator.MoveToDevice(x, newBuffer, oldBuffer, disposeDetachedBufferHint);
    }
    public virtual void Release(Tensor x, bool calledFromTensorDispose)
    {
        x.allocator.Release(x, calledFromTensorDispose);
    }
    public virtual void WaiveOwnership(Tensor x)
    {
        x.allocator.WaiveOwnership(x);
    }
    public virtual void Reset(bool keepCachedMemory)
    {
        m_InferenceScopedPingPongAllocator.Reset(keepCachedMemory);
        m_InferenceScopedStorageAllocator.Reset(keepCachedMemory);
        m_LayerScopedAllocator.Reset(keepCachedMemory);
    }

    public override void Dispose()
    {
        base.Dispose();

        m_InferenceScopedPingPongAllocator.Dispose();
        m_InferenceScopedStorageAllocator.Dispose();
        m_LayerScopedAllocator.Dispose();
    }

#if ENABLE_BARRACUDA_STATS
    public long usedBytes
    { get {
        return m_InferenceScopedPingPongAllocator.usedBytes + m_InferenceScopedStorageAllocator.usedBytes + m_LayerScopedAllocator.usedBytes;
    } }
    public long busyBytes
    { get {
        return m_InferenceScopedPingPongAllocator.busyBytes + m_InferenceScopedStorageAllocator.busyBytes + m_LayerScopedAllocator.busyBytes;
    } }
    public long freeBytes
    { get {
        return m_InferenceScopedPingPongAllocator.freeBytes + m_InferenceScopedStorageAllocator.freeBytes + m_LayerScopedAllocator.freeBytes;
    } }
    public long totalBytes
    { get {
        return m_InferenceScopedPingPongAllocator.totalBytes + m_InferenceScopedStorageAllocator.totalBytes + m_LayerScopedAllocator.totalBytes;
    } }
    public override string ToString()
    {
        return $"Total allocated: {totalBytes} busy: {busyBytes}";
    }
#endif //ENABLE_BARRACUDA_STATS
}

//public class DefaultTensorAllocator : TensorOperatorNewAllocator {}
//public class DefaultTensorAllocator : TensorCachingByShapeAllocator {}
internal class DefaultTensorAllocator : TensorCachingAllocator {}

//public class DefaultVars : GenericVars {}
//public class DefaultVars : GenericVarsWithReuse {}
internal class DefaultVars : GenericVarsWithPreallocation {}


} // namespace Unity.Barracuda
