# Working with outputs

After [executing a model](ModelExecution.md) you can then retrieve outputs or intermediate information.

## Fetching outputs
If the model has a single output, you can use  `worker.PeekOutput()`. Otherwise, provide output names:

```Csharp
var output = worker.PeekOutput(outputName);
// or
var output = worker.PeekOutput(); // warning: returns outputs[0]
```
**Note:** `worker.PeekOutput()` does not transfer ownership of the tensor to you: it is still owned by the `worker`. Calling `worker.PeekOutput()` is preferable and reduces memory allocations. However, if you expect to use the tensor for a long period, call `worker.Fetch()` . Otherwise, tensor values are lost after the next call to `worker.Execute()` or after the call to `worker.Dispose()`.


## Introspecting intermediate nodes

You can also analyse intermediate node values by passing a list of node names that you want to query when creating the `worker`.
```Csharp
// list of nodes to querry
var additionalOutputs = new string[] {"layer0", "layer1"}
m_Worker = WorkerFactory.CreateWorker(<WorkerFactory.Type>, m_RuntimeModel, additionalOutputs);
...
var outputLayer0 = worker.PeekOutput("layer0");
var outputLayer1 = worker.PeekOutput("layer1");
// you can also querry the original output of the network
var output = worker.PeekOutput(outputName);
```

## Introspecting the Barracuda model
The Barracuda model has simple memory representation. When the model loads you can query for inputs and outputs:
```Csharp
string[] inputNames = model.inputs;   // query model inputs
string[] outputNames = model.outputs; // query model outputs
```
Alternatively, you can directly iterate through the layers and investigate what the model is going to do:
```Csharp
foreach (var layer in model.layers)
	Debug.Log(layer.name + " does " + layer.type);
```

You can also query constants:
```Csharp
Tensor[] constants = worker.PeekConstants(layerName);
```


## Verbose mode
You can enable verbose mode for different parts of Barracuda:
```Csharp
bool verbose = true;
var model = ModelLoader.LoadModel(onnxAsset, verbose); // verbose loader
var worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model, verbose); // verbose execution
```
