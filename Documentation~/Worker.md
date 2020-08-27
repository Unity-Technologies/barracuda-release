# IWorker interface: core of the engine

The core engine interface in Barracuda is called `IWorker`. `IWorker` breaks down the model into executable tasks and schedules them on GPU or CPU.

**Warning:**  Some platforms might not support some backends. See [Supported platforms](SupportedPlatforms.md) for more info.

## Create the inference engine (Worker)
You can create a `Worker` from the `WorkerFactory`. You must specify a backend and a [loaded model](Loading.md).
```Csharp
Model model = ...

// GPU
var worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model)
var worker = WorkerFactory.CreateWorker(WorkerFactory.Type.Compute, model)
// slow - GPU path
var worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputeRef, model)

// CPU
var worker = WorkerFactory.CreateWorker(WorkerFactory.Type.CSharpBurst, model)
var worker = WorkerFactory.CreateWorker(WorkerFactory.Type.CSharp, model)
// very slow - CPU path
var worker = WorkerFactory.CreateWorker(WorkerFactory.Type.CSharpRef, model)
```


## Worker type

There a number of different backends you can choose to run your network:
* CPU
    * `CSharpBurst` : highly efficient, jobified and parallelized CPU code compiled via [Burst](https://docs.unity3d.com/Packages/com.unity.burst@0.2/manual/index.html).
    * `CSharp` : slightly less efficient CPU code.
    * `CSharpRef`: a less efficient but more stable reference implementation.
* GPU
  * `ComputePrecompiled` : highly efficient GPU code with all overhead code stripped away and precompiled into the worker.
  * `Compute` : highly efficient GPU but with some logic overhead.
  * `ComputeRef` : a less efficient but more stable reference implementation.
* Auto 
  * At the moment defaults to GPU

**Note**:  You can use reference implementations as a stable baseline for comparison with other implementations. 

If you notice a bug or incorrect inference, see if choosing a simpler worker solves the issue. Please report any bugs in the [Barracuda GitHub repository](https://github.com/Unity-Technologies/barracuda-release/issues).
