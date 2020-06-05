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
