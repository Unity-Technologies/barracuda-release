# Working with data

## Tensor
In Barracuda, you can access Tensor values via the  `batch`, `height`, `width`, `channels` NHWC layout also known as channels-last: 

![Channels last](images/ChannelsLast.png)

**Note:**  The native ONNX data layout is NCHW, or channels-first. Barracuda automatically converts ONNX models to NHWC layout.

### Data access

You can interact with `Tensor` data via multi-dimensional array operators:
```Csharp
var tensor = new Tensor(batchCount, height, width, channelCount);

// as N batches of 3 dimensional data: N x {X, Y, C}
tensor[n, y, x, c] = 1.0f;
// as N batches of 1 dimensional data: N x {C}
tensor[n,       c] = 2.0f; 
// as flat array
tensor[         i] = 3.0f;
```

### Constructor
Multiple `Tensor` constructors cover a variety of scenarios. By default tensors initialize with `0` upon construction, unless you provide an initialization `Array`:
```Csharp
// batch of 3 dimensional data, 0 initialized: batchCount x {height, width, channelCount}
tensor = new Tensor(batchCount, height, width, channelCount);    
// batch of 1 dimensional data, 0 initialized: batchCount x {elementCount}
tensor = new Tensor(batchCount, elementCount);                   
```
```Csharp
var stridedArray = new float[batchCount * elementCount] { ... };
// batch of 1 dimensional data, initialized from strided array
tensor = new Tensor(batchCount, elementCount, stridedArray);     
```
```Csharp
var jaggedArray = new float[batchCount][elementCount] { ... };
// batch of 1 dimensional data, initialized from jagged array
tensor = new Tensor(batchCount, elementCount, jaggedArray);      
```
```Csharp
Texture2D texture = ...;
// tensor initialized with texture data: 1 x { texture.width, texture.height, 3}
tensor = new Tensor(texture);                                    
```

You can query the shape of the `Tensor` object, but you cannot change the shape of the `Tensor`. If you need a different shape of `Tensor`, you must construct a new instance of the `Tensor` object:

```C#
var shape = tensor.shape;
Debug.Log(shape + " or " + shape.batch + shape.height + shape.width + shape.channels);
```

## Texture constructor
You can create a `Tensor` from an input `Texture` or save a `Tensor` to a `Texture` directly. 

**Note:** When you do this, pixel values are in the range [0,1]. Convert data accordingly if your network is expecting values between [0,255], for example.

### Texture as input
You can directly pass `Texture2D`, `Texture2DArray`, `Texture3D` or `RenderTexture` to Barracuda without accessing individual pixels on the CPU:
```Csharp
// you can treat input pixels as 1 (grayscale), 3 (color) or 4 (color with alpha) channels
var channelCount = 3; 
var tensor = new Tensor(texture, channelCount);
```
You can batch multiple textures into the single `Tensor` object:
```Csharp
// these textures form a batch
var textures = new [] { texture0, texture1, texture2, texture3 }; 
var tensor = new Tensor(textures, channelCount);
```
**Note:** All textures in a batch must have the same width and height dimensions.

### Texture as output
If you want to use Barracuda execution results further in the graphics pipeline, you can copy data from `Tensor` into a `RenderTexture` without stalling the CPU or GPU:
```Csharp
var tensor = worker.PeekOutput();
var texture = BarracudaTextureUtils.TensorToRenderTexture(tensor);
```
You can reuse the same `RenderTexture` multiple times:
```Csharp
var texture = new RenderTexture(width, height, 0);
// ...
tensor = worker.PeekOutput();
BarracudaTextureUtils.TensorToRenderTexture(tensor, texture);
```


## Cleanup
As a Barracuda user you are responsible for calling `Dispose()` on inputs, outputs and any data that you created, received via `worker.Fetch()` or have taken ownership of by calling `tensor.TakeOwnership()`.  

**Note:** This is necessary to free up GPU resources properly.

```C#
tensor.Dispose();
```
**Note:** You do not need to call `Dispose()` on tensors that you received via the ``worker.PeekOutput()`` call.