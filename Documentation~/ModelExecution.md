# Model execution

In order to execute a model in Barracuda you must first [load the model](Loading.md) and [create a `Worker`](Worker.md).

## Basic execution
You can provide inputs either as a sole `Tensor` object (if the model has a single input) or as a dictionary of name and `Tensor` pairs.

```Csharp	
    Tensor input = new Tensor(batch, height, width, channels); 
    worker.Execute(input);
```
```Csharp
    var inputs = new Dictionary<string, Tensor>();
    inputs[name1] = new Tensor(batch, height1, width1, channels1);
    inputs[name2] = new Tensor(batch, height2, width2, channels2);
    worker.Execute(inputs);
```
Execution is asynchronous for GPU backends. Execution is asynchronous for CPU burst backends and synchronous for the rest of the CPU backends.


## Scheduled execution

Execution can be scheduled over a few frames. You can schedule your worker with the following code:
```Csharp
Tensor ExecuteInParts(IWorker worker, Tensor I, int syncEveryNthLayer = 5)
{
    var executor = worker.ExecuteAsync(I);
    var it = 0;
    bool hasMoreWork;

    do
    {
        hasMoreWork = executor.MoveNext();
        if (++it % syncEveryNthLayer == 0)
            worker.WaitForCompletion();

    } while (hasMoreWork);

    return worker.CopyOutput();
}
```

### Burst backend (scheduled execution)

For the `CSharpBurst` backend you may sometimes see warnings of the form (or some variant thereof):
`JobTempAlloc has allocations that are more than 4 frames old`

These warnings from the job system are benign and are meant mainly to help developers catch when memory is leaking. In the case of Barracuda this might occur when running a model via [`StartManualExecution`](https://docs.unity3d.com/Packages/com.unity.barracuda@1.0/api/Unity.Barracuda.GenericWorker.html#Unity_Barracuda_GenericWorker_StartManualSchedule), which only executes a single layer at a time in any given frame. We use a memory allocator that keeps previously allocated buffers around for re-use, so when that memory finally gets released from a previous job it may warn about it. _No memory is actually leaking in this case, though_.

If you are wanting to clear up these warnings, then you could either execute your model synchronously via (as mentioned in _Basic execution_):

```csharp
worker.Execute(inputs);
```

or choose to complete jobs and flush memory every 4 frames (similar to the general example above):

```csharp
var workerExecution = worker.StartManualSchedule(inputs);
int frameCount = 0;
while (workerExecution.MoveNext())
{
    frameCount++;
    if (frameCount == 4)
    {
        frameCount = 0;
        worker.FlushSchedule(true);
        yield return null;
    }
}
worker.FlushSchedule(true);
```

However, be aware that both approaches may affect the performance of your application.

## LSTM execution

Models with LSTM nodes can use either explicit cellular and hidden memory inputs or they can be implicit. In the former case you have no choice but to specify these inputs. In the latter case you can either rely on implicit initialization or override initialization. You can obtain the implicit names of the LSTM memories by iterating over `model.memories`. Please note that in the case of implicit memories, the memories will maintain state across execution of the worker. An example of overriding implicit initialization is below:

```Csharp
    var inputs = new Dictionary<string, Tensor>();
    
    // Standard inputs
    inputs[name1] = new Tensor(batch, height1, width1, channels1);
    inputs[name2] = new Tensor(batch, height2, width2, channels2);
    
    // Override implicit memory initialization
    foreach (var m in model.memories)
    {
        Tensor memory = new Tensor(m.shape);
        // ...
        inputs.Add(m.input, memory);
    }

    worker.Execute(inputs);
```