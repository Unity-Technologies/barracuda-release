# Frequently Asked Questions (FAQ)

**Q.** Does Barracuda work on iPhone / Android / Magic Leap / Switch / PS4 / Xbox?

* **A.** Yes. Barracuda supports all platforms that Unity supports. Some restrictions apply however: for more information see [Supported platforms](SupportedPlatforms.md).

**Q.** Can I import my model from TensorFlow / Pytorch / Keras?

* **A.** Yes. You must first convert it to ONNX and then load it into Unity.
For more information see [Model exporting](Exporting.md).

**Q.** What operators does Barracuda support?

* **A.** See the [list of supported operators](SupportedOperators.md).

**Q.** Does Barracuda support training?

* **A.** No. Barracuda currently only supports inference.

**Q.** Can Barracuda run on a GPU?

* **A.** Yes. You must specify the GPU backend when [creating a `Worker`](Worker.md).

**Q.** Can Barracuda run on a CPU?

* **A.** Yes. You must specify the CPU backend when [creating a `Worker`](Worker.md).

**Q.** What is the difference between `ComputeRef`,  `Compute` and `ComputePrecompiled`?

* **A.** A `Compute<***>` `WorkerType` is a worker that operates on the GPU.  
  * `ComputeRef` is a slow reference implementation. It is useful for debugging but uses more memory.  
  * `Compute` is a fast GPU implementation. It has some CPU overhead but is flexible to a change of input dimensions.  
  * `ComputePrecompiled` strips out all CPU overhead, but optimizes the execution for a given input dimension. If the input dimensions change, there is a compile overhead. 

**Q.** What is the difference between `CSharpRef` , `CSharp` and`CSharpBurst`?

* **A.** A `CSharp<***>` `WorkerType` is a worker that operates on the CPU. 
  *  `CSharpRef` is a very slow reference implementation. It is useful for debugging.
  * `CSharp` is a fast C# synchronous implementation.
  * `CSharpBurst` is a very fast asynchronous CPU implementation.

**Q.** My operator is not supported, what do I do?

* **A.** Please post the issue on the [Barracuda GitHub repository](https://github.com/Unity-Technologies/barracuda-release/issues) and the Unity Barracuda team will do their best to help.

**Q.** I found a bug. What do I do?

* **A.** Please post the issue on the [Barracuda GitHub repository](https://github.com/Unity-Technologies/barracuda-release/issues) and the Unity Barracuda team will do their best to help.

**Q.** Execution is slow.

* **A.** 
  * GPU execution can be slower than CPU inference on mobile.
  * When reading back data from the GPU, there needs to be synchronization, which can induce lag. 
  * Keep all data on the same pipe as much as possible.
  * Please post the issue on the [Barracuda GitHub repository](https://github.com/Unity-Technologies/barracuda-release/issues) and the Unity Barracuda team will do their best to help.

**Q.** I get a lot of garbage collection warnings when using Barracuda.

* **A.** Ensure that you `Dispose` all allocated `Tensor` and `Worker` objects. For more information see [Memory management](MemoryManagement.md).
