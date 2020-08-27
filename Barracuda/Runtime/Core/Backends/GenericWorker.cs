using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq; // ToArray()

using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Profiling;

using System.Runtime.CompilerServices;


[assembly: InternalsVisibleTo("Unity.Barracuda.PerformanceTests")]
[assembly: InternalsVisibleTo("Unity.Barracuda.Tests")]

namespace Unity.Barracuda
{


public class GenericWorker : IWorker
{
    private Model m_Model;
    private string m_DefaultInputName;
    private string m_DefaultOutputName;
    private Dictionary<string, TensorShape> m_InputShapes = new Dictionary<string, TensorShape>();
    private IOps m_Ops;
    private IVars m_Vars;
    private IModelCompiler m_ModelCompiler;
    private bool m_RequestResetAllocator;
    private bool m_Verbose;
    private float m_Progress = 0f;

    private Tensor m_SyncTensor;

    // Heuristic size for a small tensor. Small tensors are more likely to be accessed on CPU,
    // thus PeekOutput() for such small tensor will auto schedule non-blocking download from GPU/NPU to CPU
    const int m_MaxBatchThatAutoTriggersAsyncDownload = 64;
    const int m_MaxFlatWidthThatAutoTriggersAsyncDownload = 1000;

    public GenericWorker(Model model, IOps ops, IVars vars, bool verbose = false)
    {
        m_Model = model;
        m_DefaultInputName = ModelAnalyzer.GetDefaultInputName(model);
        m_DefaultOutputName = ModelAnalyzer.GetDefaultOutputName(model);
        m_Ops = ops;
        m_Vars = vars;
        m_ModelCompiler = ops as IModelCompiler;
        m_Verbose = verbose;

        m_RequestResetAllocator = true;
    }

    ~GenericWorker()
    {
        Dispose();
    }

    protected void ResetAllocatorIfRequested()
    {
        if (m_RequestResetAllocator)
            m_Ops.ResetAllocator();
        m_RequestResetAllocator = false;
    }

    public virtual void Dispose()
    {
        m_Vars?.Dispose();
        m_Ops?.ResetAllocator(false); // clear allocator's memory
        m_InputShapes?.Clear();

        m_Vars = null;
        m_Ops = null;
        m_InputShapes = null;
    }

    public virtual void PrepareForInput(IDictionary<string, TensorShape> inputShapes)
    {
        m_InputShapes.Clear();
        foreach (var input in inputShapes)
            m_InputShapes.Add(input.Key, input.Value);
        m_Vars.PrepareStorage(m_Model, m_Ops, m_InputShapes);
    }

    public virtual void SetInput(string name, Tensor x)
    {
        ResetAllocatorIfRequested();
        m_Ops.Prepare(x);
        m_Vars.SetInput(name, x);

        // if single input network, then we have enough information to prepare network for execution
        if (m_Model.inputs.Count <= 1 && name == m_DefaultInputName)
            PrepareForInput(new Dictionary<string, TensorShape> { { name, x.shape } }); // @TODO: get rid of allocation

        m_InputShapes[name] = x.shape;
    }

    public virtual void SetInput(Tensor x)
    {
        SetInput(m_DefaultInputName, x);
    }

    public virtual IWorker Execute(IDictionary<string, Tensor> inputs)
    {
        foreach (var entry in inputs)
            SetInput(entry.Key, entry.Value);
        return Execute();
    }

    public virtual IWorker Execute(Tensor input)
    {
        SetInput(input);
        return Execute();
    }

    public virtual IWorker Execute()
    {
        var enumerator = StartManualSchedule();
        while (enumerator.MoveNext()) {};
        return this;
    }

    public virtual IEnumerator StartManualSchedule(IDictionary<string, Tensor> inputs)
    {
        foreach (var entry in inputs)
            SetInput(entry.Key, entry.Value);
        return StartManualSchedule();
    }

    public virtual void FlushSchedule(bool blocking)
    {
        // force execution of scheduled ops by requesting results of the intermedite tensor from the device
        m_SyncTensor.PrepareCacheForAccess(blocking);
    }

    public virtual IEnumerator StartManualSchedule(Tensor input)
    {
        SetInput(input);
        return StartManualSchedule();
    }

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

    public virtual IEnumerator StartManualSchedule()
    {
        Profiler.BeginSample ("Barracuda.Execute");

        ResetAllocatorIfRequested();
        m_Vars.PrepareStorage(m_Model, m_Ops, m_InputShapes);

        if (m_ModelCompiler != null)
            m_ModelCompiler.PrepareModel(m_Model, m_InputShapes);

        int idx = 0;
        foreach (var l in m_Model.layers)
        {
            idx++;

            m_Progress = idx / (float)m_Model.layers.Count;

            Profiler.BeginSample(l.name);
            var inputs = m_Vars.GatherInputs(l);

            Tensor X = inputs.Length > 0 ? inputs[0] : new Tensor();

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
            // MatMul
            else if (l.type == Layer.Type.MatMul)
            {
                Assert.AreEqual(inputs.Length, 2);
                Profiler.BeginSample ("Barracuda.MatMul");
                X = m_Ops.MatMul(X, false, inputs[1], false);
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
                    Assert.AreEqual(sizeTensor.length, 4);
                    size = new int[] {(int)sizeTensor[2], (int)sizeTensor[1]};
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
            else if (l.type == Layer.Type.Border2D)
            {
                Profiler.BeginSample ("Barracuda.Border2D");

                Assert.IsNotNull(l.pad);
                // NOTE: beta is used to retrieve fillin value
                // because beta is 0 by default (while alpha is 1 by default)
                // 0 value is more inline with zero padding
                float fillValue = l.beta;
                X = m_Ops.Border2D(X, l.pad, fillValue);
            }
            else if (l.type == Layer.Type.Pad2DReflect)
            {
                Profiler.BeginSample ("Barracuda.Pad2DReflect");

                Assert.IsNotNull(l.pad);
                X = m_Ops.Pad2DReflect(X, l.pad);
            }
            else if (l.type == Layer.Type.Pad2DSymmetric)
            {
                Profiler.BeginSample ("Barracuda.Pad2DSymmetric");

                Assert.IsNotNull(l.pad);
                X = m_Ops.Pad2DSymmetric(X, l.pad);
            }
            else if (l.type == Layer.Type.Pad2DEdge)
            {
                Profiler.BeginSample ("Barracuda.Pad2DEdge");

                Assert.IsNotNull(l.pad);
                X = m_Ops.Pad2DEdge(X, l.pad);
            }
            // 3D
            else if (l.type == Layer.Type.Conv3D ||
                l.type == Layer.Type.Conv3DTrans ||
                l.type == Layer.Type.Upsample3D ||
                l.type == Layer.Type.MaxPool3D ||
                l.type == Layer.Type.AvgPool3D ||
                l.type == Layer.Type.GlobalMaxPool3D ||
                l.type == Layer.Type.GlobalAvgPool3D ||
                l.type == Layer.Type.Border3D)
            {
                throw new NotImplementedException("3D operations are not implemented yet!");
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
                float bias = (l.weights.Length > 0) ? l.weights[0] : 1.0f;
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
                X = m_Ops.Multinomial(X, count, seed);
            }
            else if (l.type == Layer.Type.OneHot)
            {
                Profiler.BeginSample ("Barracuda.OneHot");

                Assert.IsNotNull(l.pool);
                Assert.AreEqual(l.pool.Length, 1);
                int depth = l.pool[0];
                float on = l.alpha, off = l.beta;
                X = m_Ops.OneHot(X, depth, on, off);
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
                     l.type == Layer.Type.ReduceSum)
            {
                Profiler.BeginSample ("Barracuda.Reduce");

                if(X.shape[l.axis] == 1)
                    break;

                var Xshape = X.shape;
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
            // Shape affecting layers
            else if (l.type == Layer.Type.Flatten)
            {
                Profiler.BeginSample ("Barracuda.Flatten");
                X = m_Ops.Flatten(X);
            }
            else if (l.type == Layer.Type.Reshape)
            {
                Profiler.BeginSample ("Barracuda.Reshape");

                // pool size is treated as reshape coefficient, if not empty
                // otherwise shape of the 2nd input tensor is used
                var size = l.pool;

                Assert.IsNotNull(size);
                if (size.Length == 0 && inputs.Length > 1)
                    size = inputs[1].shape.ToArray();

                var newShape = X.shape.Reshape(size);
                X = m_Ops.Reshape(X, newShape);
            }
            else if (l.type == Layer.Type.Expand)
            {
                Profiler.BeginSample("Barracuda.Expand");

                // pool size is treated as new shape
                var newShape = l.pool;

                Assert.IsNotNull(newShape);
                Assert.IsTrue(newShape.Length == 8 || newShape.Length == 4);

                X = m_Ops.Expand(X, new TensorShape(newShape));
            }
            else if (l.type == Layer.Type.Transpose)
            {
                Profiler.BeginSample ("Barracuda.Transpose");

                var permutations = l.pool;
                if (permutations == null)
                    X = m_Ops.Transpose(X);
                else
                    X = m_Ops.Transpose(X, permutations);
            }
            else if (l.type == Layer.Type.Gather)
            {
                Profiler.BeginSample ("Barracuda.Gather");
                X = m_Ops.Gather(inputs, l.axis);
            }
            else if (l.type == Layer.Type.Squeeze ||
                l.type == Layer.Type.Unsqueeze)
            {
                throw new NotImplementedException();
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
                throw new NotImplementedException();
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
                    X = m_Ops.Softmax(X);
                }
                else if (l.activation == Layer.Activation.LogSoftmax)
                {
                    X = m_Ops.LogSoftmax(X);
                }
                else if (l.activation == Layer.Activation.Tanh)
                {
                    X = m_Ops.Tanh(X);
                }
                else if (l.activation == Layer.Activation.Sigmoid)
                {
                    X = m_Ops.Sigmoid(X);
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
                    l.activation == Layer.Activation.Softplus ||
                    l.activation == Layer.Activation.Softsign ||
                    l.activation == Layer.Activation.Hardmax ||
                    l.activation == Layer.Activation.HardSigmoid)
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
                else
                {
                    X = m_Ops.Copy(X);
                }
            }
            else
            {
                Profiler.BeginSample ("Barracuda.Dummy");
                Assert.AreEqual(l.activation, Layer.Activation.None);
            }

            m_Vars.Store(l, X);
            m_SyncTensor = X;

            // optype
            Profiler.EndSample();

            // layer.name
            Profiler.EndSample();

            yield return null;
        }

        // request ResetAllocator before next Execute() starts
        m_RequestResetAllocator = true;
        Profiler.EndSample ();

        if (m_Verbose)
            D.Log(m_Vars.GetAllocator());
    }

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

    public virtual Tensor[] PeekConstants(string layerName)
    {
        Profiler.BeginSample("Barracuda.PeekConstants");
        return m_Vars.PeekConstants(layerName);
    }

    public virtual string Summary()
    {
        return m_Vars.GetAllocator().ToString() + "\n" + m_Ops.ToString();
    }
}


internal class GenericVars : IVars
{
    private Dictionary<string, Tensor> m_TensorsByName = new Dictionary<string, Tensor>();
    protected HashSet<Tensor> m_ModelTensors = new HashSet<Tensor>();
    protected Dictionary<Layer, Tensor[]> m_InputTensorsByLayer = new Dictionary<Layer, Tensor[]>();
    private Dictionary<string, int> m_LayerNameToId = new Dictionary<string, int>();
    private Dictionary<string, int> m_LayerNameToKeepUntilId = new Dictionary<string, int>();
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
    }

    private ITensorAllocator m_Allocator = new DefaultTensorAllocator();
    public virtual ITensorAllocator GetAllocator()
    {
        return m_Allocator;
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
            var tensor = new Tensor(arg.shape, new SharedArrayTensorData(layer.weights, (int)arg.offset,
                                                                        (int)arg.shape.length),
                                                                        arg.name);
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

    public virtual void PrepareStorage(Model model, IOps ops, IDictionary<string, TensorShape> inputShapes)
    {
        ValidateGlobalInputs(model, inputShapes);

        m_LayerNameToId.Clear();
        m_LayerNameToKeepUntilId.Clear();
        m_LayerIdToLayer.Clear();

        for (var idx = 0; idx < model.layers.Count; idx++)
        {
            var forLayer = model.layers[idx];
            m_LayerIdToLayer[idx] = forLayer;

            // prepare input placeholders and argument tensors only once per layer
            if (m_InputTensorsByLayer.ContainsKey(forLayer))
                continue;

            var tensors = PrepareLayerInputTensors(model, forLayer, ops);
            m_InputTensorsByLayer.Add(forLayer, tensors);
        }

        for (var i = 0; i < model.layers.Count; i++)
        {
            var layer = model.layers[i];
            m_LayerNameToId[layer.name] = i;

            for (var j = 0; j < layer.inputs.Length; j++)
            {
                m_LayerNameToKeepUntilId[layer.inputs[j]] = i;
            }
        }

        // inputs should always be preserved
        foreach (var input in model.inputs)
        {
            m_LayerNameToKeepUntilId[input.name] = model.layers.Count;
        }

        // outputs should always be preserved
        foreach (var outname in model.outputs)
        {
            m_LayerNameToKeepUntilId[outname] = model.layers.Count;
        }

        // memories should always be preserved
        foreach (var mem in model.memories)
        {
            m_LayerNameToKeepUntilId[mem.input] = model.layers.Count;
            m_LayerNameToKeepUntilId[mem.output] = model.layers.Count;
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

    public virtual void PrepareStorage(Layer forLayer)
    {
        // Current layer Id
        var layerId = m_LayerNameToId[forLayer.name];

        for (var idx = 0; idx < layerId; idx++)
        {
            var l = m_LayerIdToLayer[idx];
            var key = l.name;

            // Remove all allocated tensors for layer storage, but
            // global constants might not exist in this dictionary,
            // so lets just ignore them
            if (m_TensorsByName.ContainsKey(key) &&
                m_LayerNameToKeepUntilId.ContainsKey(key) &&
                m_LayerNameToKeepUntilId[key] < layerId &&
                !m_ModelTensors.Contains(m_TensorsByName[key]))
            {
                if (IsTensorOwnedByInternalAllocator(m_TensorsByName[key]))
                    m_TensorsByName[key].Dispose();
                m_TensorsByName.Remove(key);
            }
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

    protected bool layerRequiresStorage { get { return m_LayerRequiresStorage; } }
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

    public override void PrepareStorage(Model model, IOps ops, IDictionary<string, TensorShape> inputShapes)
    {
        base.PrepareStorage(model, ops, inputShapes);

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
        if (result.tensorOnDevice != m_Temporary?.tensorOnDevice)
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

internal class GenericVarsWithPreallocation : GenericVarsWithReuse, ITensorAllocator
{
    private Model m_CachedModel;

    private DefaultTensorAllocator m_TemporaryAllocator = new DefaultTensorAllocator();
    private DefaultTensorAllocator m_StorageAllocator = new DefaultTensorAllocator();

    public override void PrepareStorage(Model model, IOps ops, IDictionary<string, TensorShape> inputShapes)
    {
        base.PrepareStorage(model, ops, inputShapes);
        if (m_CachedModel != model)
        {
            // pre-allocate 2 buffers that can be cycled for temporaries
            var allocator = m_TemporaryAllocator;

            var maxShape = ModelAnalyzer.FindLargestNecessaryTensorShape(model, inputShapes);
            var alloc1 = allocator.Alloc(maxShape);
            var alloc2 = allocator.Alloc(maxShape);
            alloc1 = ops.Prepare(alloc1);
            alloc2 = ops.Prepare(alloc2);
            allocator.Release(alloc1, false);
            allocator.Release(alloc2, false);
        }
        m_CachedModel = model;
    }

    public override ITensorAllocator GetAllocator()
    {
        return this;
    }
    protected override bool IsTensorOwnedByInternalAllocator(Tensor tensor)
    {
        var allocator = tensor.allocator;
        return allocator == m_TemporaryAllocator ||
               allocator == m_StorageAllocator;
    }

    public virtual Tensor Alloc(TensorShape shape)
    {
        if (layerRequiresStorage)
            return m_StorageAllocator.Alloc(shape);
        else
            return m_TemporaryAllocator.Alloc(shape);
    }
    public virtual Tensor Alloc(TensorShape shape, ITensorData buffer)
    {
        if (layerRequiresStorage)
            return m_StorageAllocator.Alloc(shape, buffer);
        else
            return m_TemporaryAllocator.Alloc(shape, buffer);
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
        m_TemporaryAllocator.Reset(keepCachedMemory);
        m_StorageAllocator.Reset(keepCachedMemory);
    }

    public long busyBytes
    { get {
        return m_TemporaryAllocator.busyBytes + m_StorageAllocator.busyBytes;
    } }
    public long freeBytes
    { get {
        return m_TemporaryAllocator.freeBytes + m_StorageAllocator.freeBytes;
    } }
    public long totalBytes
    { get {
        return m_TemporaryAllocator.totalBytes + m_StorageAllocator.totalBytes;
    } }
    public override string ToString()
    {
        return $"Total allocated: {totalBytes} busy: {busyBytes}";
    }
}

//public class DefaultTensorAllocator : TensorOperatorNewAllocator {}
//public class DefaultTensorAllocator : TensorCachingByShapeAllocator {}
internal class DefaultTensorAllocator : TensorCachingAllocator {}

//public class DefaultVars : GenericVars {}
//public class DefaultVars : GenericVarsWithReuse {}
internal class DefaultVars : GenericVarsWithPreallocation {}


} // namespace Unity.Barracuda
