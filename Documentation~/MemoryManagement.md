# Memory management
As a Barracuda user you are responsible for calling `Dispose()` on any worker, inputs and sometimes outputs. You must call `Dispose()` on outputs if you obtain them via `worker.CopyOutput()` or if you take ownership of them by calling `tensor.TakeOwnership()`.  

**Note:** Calling `Dispose()` is necessary to properly free up GPU resources.

Here is an example:
```Csharp
public void OnDestroy()
{
    worker?.Dispose();

    // Assuming model with multiple inputs that were passed as a Dictionary
    foreach (var key in inputs.Keys)
    {
        inputs[key].Dispose();
    }
    
    inputs.Clear();
}
```
**Note:** You don't need to call `Dispose()` for tensors that you received via the `worker.PeekOutput()` call.