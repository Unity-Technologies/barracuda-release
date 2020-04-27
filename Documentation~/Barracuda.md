<!---TODO:
	Advanced topics
	* how to trim networks at runtime (multi brain models)
	* recurrent state
--->

# Unity Barracuda

**Unity Barracuda** is a lightweight and **cross-platform** Neural Net **inference library for Unity**. Barracuda can run Neural Nets both on GPU and CPU. Currently Barracuda is in the preview development stage, so adventures are expected.

## Getting Unity Barracuda
- via `Unity Package Manager`: open Package Manager window in Unity Editor, enable `preview` packages, select Barracuda and install it.
- from `GitHub`: edit your Unity Project's `Packages/manifest.json` and add dependency to the Barracuda GitHub repo:
`"com.unity.barracuda" : "https://github.com/Unity-Technologies/barracuda-release.git"`

## Using Barracuda
Typically the following steps are needed to use Barracuda in application:
1. add .onnx file to your project - it will behave like any other regular asset,
2. load model from the asset,
3. create inference engine (the worker),
4. execute model and
5. fetch results.

You can import ONNX models simply by adding .onnx file directly to your project, however Tensorflow models require additional attention by running python script for now. See _Converting Tensorflow models to Barracuda format_ paragraph below for more information.


### Load Model into Barracuda
To load ONNX model, first add its .onnx file to your project. It will be imported and show up as an asset of type `NNModel`.

Next, add public `NNModel` field to your C# script and assign reference to your asset via editor UI. Load model with `ModelLoader`:
```C#
public NNModel modelSource;
<..>
var model = ModelLoader.Load(modelSource);
```

To load Tensorflow models that were manually converted with `tensorflow_to_barracuda.py` python script, use the following code:
```C#
var model = ModelLoader.LoadFromStreamingAssets(modelName + ".nn");
```

### Create inference engine (Worker)
Inference engine in Barracuda is called Worker. Worker is responsible for breaking down model into executable tasks and scheduling them on GPU or CPU.
```C#
var worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model)
```

### Execute the model
Inputs can be provided both as sole `Tensor` object (assuming Model has only one input) or as a dictionary of name and `Tensor` pairs.

```C#
var inputs = new Dictionary<string, Tensor>();
inputs[name1] = new Tensor(...);
inputs[name2] = new Tensor(...);
worker.Execute(inputs);
```
Execution is asynchronous for GPU backends. Currently implementation is synchronous for CPU backends, however it is good to assume that execution will be async for all backends in the future.

### Fetch outputs
If model has only single output, then simple `worker.PeekOutput()` can be used, otherwise output names should be provided.
```C#
var O = worker.PeekOutput(outputName);
```
_Note:_ `worker.PeekOutput()` does not transfer ownership of the tensor to you and tensor will still be owned by the `worker`. Calling `worker.PeekOutput()` is preferable way and allows to reduce memory allocations. However, if you expect to use tensor for longer time, call `worker.Fetch()` - otherwise tensor values will be lost after the next call to `worker.Execute()` or after call to `worker.Dispose()`

### Cleanup
As a Barracuda client you are responsible to `Dispose` _worker_, _inputs_ and _outputs_ you created, received via `worker.Fetch()` or taken ownership by calling `tensor.TakeOwnership()`. This is necessary to properly free GPU resources.
```C#
O.Dispose();
worker.Dispose();
```
_Note:_ It is not necessary to `Dispose` tensor that you received via ``worker.PeekOutput()`` call.

## Working with data

### Tensor
Tensor values in Barracuda are accessed via  `batch`,`height`,`width`,`channels` layout also known as _NHWC_ or _channels-last_. You can interact with `Tensor` data via multi-dimensional array operators:
```C#
var tensor = new Tensor(batchCount, height, width, channelCount);
tensor[n, y, x, c] = 1.0f; // as N batches of 3 dimensional data: N x {X, Y, C}
tensor[n,       c] = 2.0f; // as N batches of 1 dimensional data: N x {C}
tensor[         i] = 3.0f; // as flat array
```

There are number of `Tensor` constructors that cover a variety of scenarios. By default tensors are initialized with `0` upon construction, unless initialization `Array` is provided.
```C#
tensor = new Tensor(batchCount, height, width, channelCount);    // batch of 3 dimensional data, 0 initialized: batchCount x {height, width, channelCount}
tensor = new Tensor(batchCount, elementCount);                   // batch of 1 dimensional data, 0 initialized: batchCount x {elementCount}

var stridedArray = new float[batchCount * elementCount] { ... };
tensor = new Tensor(batchCount, elementCount, stridedArray);     // batch of 1 dimensional data, initialized from strided array

var jaggedArray = new float[batchCount][elementCount] { ... };
tensor = new Tensor(batchCount, elementCount, jaggedArray);      // batch of 1 dimensional data, initialized from jagged array

Texture2D texture = ...;
tensor = new Tensor(texture);                                    // tensor initialized with texture data: 1 x { texture.width, texture.height, 3}
```

You can query shape of the `Tensor` object, but you can not change it. Shape of the `Tensor` is immutable. If you want to have different shape of `Tensor`, you have to construct a new instance of `Tensor` object.
```C#
var shape = tensor.shape;
Debug.Log(shape + " or " + shape.batch + shape.height + shape.width + shape.channels);
```

### Texture as input
You can directly pass `Texture2D`, `Texture2DArray`, `Texture3D` or `RenderTexture` to Barracuda without accessing individual pixels on CPU:
```C#
var channelCount = 3; // you can treat input pixels as 1 (grayscale), 3 (color) or 4 (color with alpha) channels
var tensor = new Tensor(texture, channelCount);
```
You can batch multiple textures into the single `Tensor` object:
```C#
var textures = new [] { texture0, texture1, texture2, texture3 }; // these textures will form a batch
var tensor = new Tensor(textures, channelCount);
```
Note that to form a batch all textures must have the same width and height dimensions.

### Texture as output
If you want to use Barracuda execution results further in the graphics pipeline, you can copy data from `Tensor` into `RenderTexture` without stalling CPU or GPU:
```C#
	var tensor = worker.PeekOutput();
	var texture = BarracudaTextureUtils.TensorToRenderTexture(tensor);
```
If you wish, you can reuse the same `RenderTexture` multiple times:
```C#
	var texture = new RenderTexture(width, height, 0);
	// ...
	var tensor = worker.PeekOutput();
	BarracudaTextureUtils.TensorToRenderTexture(tensor, texture);
```

## Introspecting Barracuda models
Barracuda model has very simple memory representation. Once model is loaded you can query for inputs and outputs:
```C#
string[] inputNames = model.inputs;   // query model inputs
string[] outputNames = model.outputs; // query model outputs
```
Or you can directly iterate through the layers and investigate what model is going to do:
```C#
foreach (var layer in model.layers)
	Debug.Log(layer.name + " does " + layer.type);
```

## Verbose mode
You can turn on verbose mode for different parts of Barracuda:
```C#
bool verbose = true;
var model = ModelLoader.LoadModel(onnxAsset, verbose); // verbose loader
var worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model, verbose); // verbose execution
```

## Converting TensorFlow models to Barracuda format
Barracuda comes with dedicated python scripts to convert pre-trained constant `.pb` TensorFlow graph.

To produce a constant graph from a trained model:

- In TF1.x use [freeze_graph](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py). Example from a `SavedModel` (more [examples](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph_test.py)).

```python
from tensorflow.python.tools import freeze_graph
freeze_graph.freeze_graph(None, None, <input_is_binary>, None, None
                          <output_name>, None, None, <output_path>, False,
                          clear_devices, None, None, None, False, False, 
                          <saved_model_dir>, tag_constants.SERVING)
```

- In TF2.x use [convert_to_constants](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/convert_to_constants.py). Example from a Keras `Model` (more [examples](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/convert_to_constants_test.py)):

```python
from tensorflow.python.framework import convert_to_constants
@tf.function(input_signature=[tf.TensorSpec(shape=[<input_shape>], dtype=tf.float32)])
def to_save(x):
    return model(x)
f = to_save.get_concrete_function()
constantGraph = convert_to_constants.convert_variables_to_constants_v2(f)
tf.io.write_graph(constantGraph.graph.as_graph_def(), <output_dir>, <output_file>) 
```

Convert constant graph to Barracuda:

```bash
python tensorflow_to_barracuda.py Models/3DBall-tf-model.pb Destination/3DBall-bc.nn
```

If network has multiple outputs, but you need only particular ones during the inference, there is an optional `-trim` flag to remove unused outputs and calculations.
For example:
```bash
python tensorflow_to_barracuda.py Models/3DBall-tf-model.pb Destination/3DBall-bc.bytes -trim action$
```
First, trim will remove outputs from the graph that do not match regexp pattern. Second trim will strip all nodes that do not participate in the evaluation of the output.
In the example above only outputs that end with `action` will be left.

You could pass `--print-supported-ops` to get approximate list of supported operations/activations for specific converter.

P.S. Python 3.5 or 3.6 is recommended
P.P.S. We plan to migrate Tensorflow converter from Python to C# in the future.

## Supported platforms

*CPU inference*: *all Unity platforms are supported*.

*GPU inference*: all Unity platforms are supported except OpenGL ES on Android/iOS (use Vulkan/Metal), OpenGL Core on Mac (use Metal), WebGL (use CPU inference).

## Supported Neural architectures and Models

- All ML-Agents models.
- MobileNet v1/v2 image classifiers.
- Tiny YOLO v2 object detector.
- UNet type of models.
- Fully convolutional models.
- Fully dense models.

## Approximate list of ONNX operations supported by Barracuda

### Operations
```
Add
Sum
Sub
Mul
Div
Pow
Min
Max
Mean
AveragePool
MaxPool
GlobalAveragePool
GlobalMaxPool
Upsample
Gemm
MatMul
Conv
ConvTranspose
BatchNormalization
InstanceNormalization
Greater
Less
Equal
Or
And 
Not 
Xor
Pad
Constant
Identity
Cast
Dropout
Reshape
Unsqueeze
Squeeze
Flatten
Concat
Slice
```
P.S. some of these operations are under limited support

### Activations
```Relu
Softmax
Tanh
Sigmoid
Elu
LeakyRelu
Selu
```

## Approximate list of TensorFlow nodes supported by Barracuda script converter
### Operations
```
Add
AvgPool
BatchNormalization
BatchNormalizationRuntime
BiasAdd
Ceil
Concat
Conv2D
Conv2DBackpropInput
Dense
DepthwiseConv2dNative
Exp
Flatten
Floor
FusedBatchNorm
GlobalAveragePool
GlobalAvgPool
InstanceNormalization
MatMul
Max
MaxPool
Maximum
Mean
Min
Minimum
Mul
Multinomial
Nop
Neg
OneHot
Pad
Pow
Prod
RandomStandardNormal
RandomUniform
RealDiv
Reshape
ResizeBicubic
ResizeBilinear
ResizeNearestNeighbor
StridedSlice
Sqrt
Sub
Sum
```

### Activations
```
Elu
LeakyRelu
Linear
Log
LogSoftmax
Relu
Relu6
Selu
Sigmoid
Softmax
Softplus
Softsign
Swish
```
P.S. some of these nodes are under limited support 

## Reporting issues

If you encounter issues running Barracuda in your Unity project, please report them on our [GitHub repo](https://github.com/Unity-Technologies/barracuda-release/issues).


