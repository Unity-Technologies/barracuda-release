# Release notes

## [1.0.3] - 2020-09-15
- Fix: Fixed matrix multiplication issue in Universal Mac Standalone builds. 

## [1.0.2] - 2020-07-30
- Upgraded Burst dependency to 1.3.4.
- Docs: Minor fixes. 

## [1.0.1] - 2020-07-02
- Fix in regard to package dependencies for Unity 2020.2 and newer.

## [1.0.0] - 2020-06-01
- First verified release

## [0.8.0] - 2020-05-28
- Breaking change: API cleanup, number of internal implementation classes marked as `internal`.
- API change: `IWorker.ExecuteAsync` renamed to `IWorker.StartManualSchedule`. `IWorker.ExecuteAsync` is marked as obsolete and will issue warning if used.
- API change: `IWorker.WaitForCompletion` renamed to `IWorker.FlushSchedule`. `IWorker.WaitForCompletion` is marked as obsolete and will issue warning if used.
- API change: `IWorker.GetAsyncProgress` renamed to `IWorker.scheduleProgress`. `IWorker.GetAsyncProgress` is marked as obsolete and will issue warning if used.
- Performance: Introduced new CPU backend powered by Unity Burst compiler `CSharpBurst`. The new backend runs in parallel to the main thread and is roughly ~2x faster in Standalone and ~5x faster in the Editor. Prefer to use yield instruction `WaitForCompletion(output)` before accessing values of the output tensor to avoid blocking the main thread.
- Performance: Improved `Conv2d` performance for smaller workloads.
- Performance: Activation fusion is now supported with `Conv2DTrans` and `DepthwiseConv` layers.
- Performance: Made Broadcast ops (`Add`, `Sub,`, `Mul`, `Div`, `Mean`) more efficient.
- ONNX: Fixed `Tranpose` layer import for networks originating from Keras.
- ONNX: Added `LRN` support.
- ONNX: Convolutions now support `dilatation`.
- Fix: Fixed multiple out of bounds reads and writes in compute shaders. Should improve stability on PS4, XB1, Metal and Vulkan platforms.
- Fix: Fixed memory leak in ComputePrecompiled elementwise broadcast ops.
- Fix: Fixed broadcast ops for none unit batches
- Docs: Major documentation overhaul. 

## [0.7.1] - 2020-05-12
- ONNX: Added DepthToSpace and SpaceToDepth support (*thanks to latentspace.co!*).
- ONNX: Added Expand support (*thanks to latentspace.co!*).
- ONNX: Implemented `sizes` input for Resize op, previously only `scales` input was supported (*thanks to latentspace.co!*).
- ONNX: Automatic activation folding for ConvTranspose and DepthwiseConv
- ONNX: Conv dilatation support
- Fix: Fixed issue where second call to `Execute()` would produce incorrect results.
- Fix: Fix ConvTranpose for non-unit strides and outputadjustment input. Also fixed ComputePrecompiled path for non-0 bias.
- Fix: Small shader fix linked to activation fusing on ComputePrecompile backend.
- Performance: Slight improved performance of ConvTranpose.
- Compute: Added GPU implementation for Abs, Ceil, Floor, Reciprocal ops. Previously these operations would fallback to CPU.
- Compute: Added GPU implementation for generic MatMul op. Previously this operation would fallback to CPU.


## [0.7.0] - 2020-04-27
- API/Breaking change: Barracuda namespace renamed to Unity.Barracuda.
- API/Breaking change: Barracuda assembly renamed to Unity.Barracuda. IMPORTANT: You have to update asmdef files that reference Barracuda in your project!
- API: Added 3D LUT support to TensorToRenderTexture.
- ONNX: ImageScaler layer support (internally maps to ScaleBias).
- ONNX: Added support for Double, Int8, Int16, UInt8, UInt16, UInt32, UInt64 and Bool data types. Now all ONNX specified data types except `string` should be supported.
- ONNX: Fixed case when model input is directly passed to Activation layer.
- ONNX: Fixed support for empty null tensors.
- Performance: Reduced temporary memory allocations for CPU inference path.
- Performance: Preparation work to run GPU inference path in NCHW (aka channels-first) layout. NCHW offers more performant layout for number of desktop GPUs. No changes are necessary from client API perspective.
- Performance: InstanceNorm optimizations.
- Performance: VarianceMean, AveragePooling optimizations.
- Performance: Relu, Tanh, Sigmoid, Relu6, Swish activations are automatically fused into other layers whenever possible.
- Performance: Neg, Sqrt, Exp, Log elementwise operations are automatically fused into other layers whenever possible.
- Performance: Added GPU implementation for StridedSlice.
- Performance: ConvTranspose improvements.
- Performance: Inspector UI now loads much faster for large models in the Editor.
- Misc: Added bilinear Upsample support.
- Misc: Added support for dynamic Upsample.
- Misc: Adjusted file structure layout to comply with Unity Package guidelines.
- Misc: Fixed license for 3rd party libraries.
- Misc: Removed ONNX python tools and docker wrapper from the Barracuda distribution. Please use ONNX C# importer by placing ONNX models directly into Assets folders.
- Importer: Bumped .nn & .onnx importer version numbers.


## [0.6.3] - 2020-04-03
- Performance: over 10x performance improvement for 'Tiny Yolo V2' model and over 5x improvement for 'Mobilenet' model when CPU side inference is used.
- Performance: significantly improved speed of Convolutions on CPU by using fast Matrix Multiplication (powered by BLAS when available) and memory efficient version of Im2Col algorithm. Memory consumption stays constant regardless of kernel size!
- Performance: significantly improved speed of Depthwise Convolution on CPU.
- Performance: optimized Relu, Relu6, LeakyRelu, ScaleBias, AvgPool layers on CPU.
- Fix/Performance: BLAS and Burst plugins are not stripped anymore when making IL2CPP builds.
- Fix: no more occasional misleading shader warning "use of potentially uninitialized variable (ReadonlyTensor::SafeGet)"
- Misc: Burst dependency bumped to 1.2.3.


## [0.6.2] - 2020-03-23
- Fix: Prevent deallocation of the user-owned input tensor when passed to Identity or Reshape layer at the beginning of the model. As a side effect of this change Reshape layer now always makes copy of tensor data, this will be optimised in the future.
- Fix: Tensor constructor with ComputeBuffer argument accepts correct stride value now.
- Performance: automatically remove no-operation layers from the model (Identity, Linear activation and Flatten).
- ONNX: Automatically remove unused constants and layers.
- ONNX: Fixed constant folding of Transpose node to work properly with input tensor of any rank. Previously would fail, if tensor rank was different than 4.
- ONNX: Made Pads and Scales arguments more fault tolerant and support default values.
- ONNX: Keep ML-Agents metadata in the model. ML-Agents metadata is stored as unreferenced constants.
- API: ModelBuilder: default axis to -1 for simplicity of use.
- API: ModelBuilder: added default constructor that automatically creates new Model.
- API: Implemented RecurrentState automatic batch size detection based on the model information by passing -1 value as `batchSize` argument in the constructor.
- API: Throw exception when incorrect arguments are passed.
- API: Validate model integrity in `WorkerFactory.Create(...)`.
- UI: Fixed weight totals in the Inspector. Previously shared weights would be counted multiple time blowing up the totals.
- UI: Improved readability by increasing layers' estate when model has no constants and displaying large numbers with separators in the Inspector.
- Tests: Added variety of tests to test Public APIs.
- Cosmetic changes to error and log messages.


## [0.6.1] - 2020-03-03
- Performance: improved inference time for 'Tiny Yolo v2' model on Mali-G71 from 600ms to 190ms.
- Compute: significantly improved precision of InstanceNormalization by using Kahan/Neumaier summation and assumean mean algorithms when calculating variance.
- Fix: convolution performance and occasional result regression by temporarily disabling 64 element-per-thread version in convolutional kernel. This optimized kernel will come back in a latter release.
- Fix: convolutional Winograd kernel crash on iOS devices.
- Fix: convolutional Winograd kernel assert and potentially incorrect results when used with ComputePrecompiled backend.
- ONNX: fixed regression in Upsample opset=9.
- ONNX: fixed incorrect asserts when Reshape node contains -1 in shape definition.
- Rewrote CPU reference implementation of InstanceNormalization to use double precision.
- Tests: added opset=9 models


## [0.6.0] - 2020-02-14
- Performance: more Conv2D optimizations.
- API: introduced WaitForCompletion yield instruction  to simplify asyncronous download of execution results from GPU to CPU.
- ONNX: added optimization pass for non-variable parts of the execution graph by turning them into constants.
- ONNX: fail hard when layer of unknown type is encountered during the model import. Use "Treat Errors as Warnings" flag in asset Inspector, if wish to override this behavior.
- ONNX: added optimization pass for Tensorflow models (exported with tf2onnx) that removes excessive Tranpose layers.
- ONNX: added correction of input layouts when Tensorflow models (exported with tf2onnx) do not follow ONNX specification.
- ONNX: added stripping of trailing ":0" symbols from Tensorflow models (exported with tf2onnx).
- ONNX: added support of Shape op with non-constant inputs. Fixes some cases with Upsample opset=9.
- ONNX: added basic LSTM op support. Requires _h/_c inputs to be passed directly. Tested only on ML Agents models.
- ONNX: added support for GroupNormalization by implementing special path for Shape followed by Reshape layer.
- ONNX: fixed multi-batch PyTorch models with BatchNormalization & InstanceNormalization layers by cutting excessive weights.
- ONNX: show warnings if BatchNormalization & InstanceNormalization weights do not match feature count from the previous layer.
- ONNX: fixed support for tensors with rank below 4 in Tranpose and Slice layers.
- ONNX: fixed incorrect defaults for RandomNormal, RandomUniform, RandomNormalLike and RandomUniformLike layers.
- ONNX: silence Squeeze warning when "seq_length" is used.
- TF: added FusedBatchNormV2 and FusedBatchNormV3 support.
- Compute: asynchronous GPU readback now is enabled for Unity 2018.2 or later. Made several small fixes too.
- Compute: implemented fallback when asynchronous GPU readback is not supported, by waiting for 3 frames to pass.
- Fix: found incorrect Profiler block termination in Compute backend.
- Fix: dangling buffer assert when CPU array buffers are used with fast CPU backend.
- Fix: enabled Google Protobuf for all platforms as ML Agents rely on this package. Fixes building ML Agents on non-desktop platforms.
- UI: separated constants from layers in model inspector UI. Makes it easier to understand the model.
- UI: made model warnings more visible, added icon.
- Docs: detailed that not only copy, but shared tensor data access can happen when using ITensorData.Download() and ITensorData.SharedAccess().

## [0.5.0] - 2020-01-29
- Performance: added small optimizations for Dense layers on GPU.
- API: added worker.CopyOutput() instead of worker.Fetch(). The new name better reflects the functionality.
- API: introduced CreateWorker() convenience function directly on NNModel asset instance. Should make it easier for users instead of looking docs for ModelLoader and then WorkerFactory.
- API: added ModelBuilder.Input() with TensorShape argument.
- API: added Tensor constructor that accepts ComputeBuffer as an argument.
- API: ModelLoader.Load() verbose parameter default value changed from `true` to `false`.
- Examples: updated according with the latest API changes.
- ONNX: C# importer refactoring and cleanup.
- ONNX: added constant node baking.
- ONNX: added override for global inputs, set "sequence_length" to 1.
- ONNX: added handling of :0 in string name from TF->ONNX import, keep unconnected constants.
- ONNX: added Gather support.
- ONNX: added OneHot support.
- ONNX: added Shape support.
- ONNX: added Abs, Ceil, Floor, Round.
- ONNX: improved ONNX asset handling performance in Unity Editor UI.
- ONNX: implemented Upsample support for opset=9!
- ONNX: fixed Upsample support for opset=7,8.
- ONNX: implemented Slice for constant tensors.
- ONNX: implemented Resize op using Upsample2D or AvgPool2D depending, if scale is larger than 1 or not.
- ONNX: implemented arbitrary axis support for Concat.
- ONNX: implemented multiple axes support for Reduce ops.
- Importers: .nn and .onnx importer versions were increased, expect model reimport to happen automatically.
- Fix: fixed Unity 2020.1 support. Unity versions > 2020.1.0a21 should work fine now.
- Fix: moved MatrixUtils and ComputeShaderSingleton to Barracuda namespace.
- Fix: fixed GPU kernel selection for iPhone7 (A10), as it is limited to 224 thread groups.

## [0.4.0] - 2020-01-08
- Now Barracuda ships as a source! At the moment no PRs are accepted, but this policy might change in the future.
- Bumped min supported Unity version to 2018.x.
- Moved Burst BLAS plugin to the Barracuda Core package.
- API: Renamed AddInput() -> SetInput and Peek() -> PeekOutput(), old methods are still available, but will be removed in the future.
- ONNX: added support for importing float16 models. They will be upscaled to float32 during import. No runtime support yet.
- Performance: improved optimal kernel selection for Conv2D.
- Performance: improved activation function performance.
- Fix: kernel compilation errors on consoles.
- Fix: possible out of bounds read in Border2D op.
- Fix: prevent duplicate outputs to be added to the model via ModelBuilder.
- Fix: made CreateWorker arguments `trimOutputs` and `additionalOutputs` more robust against using incorrect layer names that do not actually exist in the model.
- Cleanup: removed unused Compute Kernels.
- Docs: added RecurrentState docs

## [0.3.2] - 2019-11-22
- Implemented PRelu activation
- Performance: improved InstanceNormalization performance on GPU, 40% faster.
- Performance: implemented batch support for InstanceNormalization.
- ONNX: implemented opset9 support for Upsample op. In opset9 `scales` parameter became specified as input tensor instead of attribute.
- ONNX: enabled LogSoftmax, Sqrt, Clip, Reciprocal and Neg ops.
- ONNX: enabled RandomNormal, RandomNormalLike, RandomUniform, RandomUniformLike ops
- Fix: added workaround for buggy Adreno + Android 8 Vulkan drivers.
- Fix: moved Protobuf to the runtime plugins so it can be shared with ML-Agents.
- Fix: compatibility with Unity 2017.4.x.
- ONNX.Py: added globals (meta info) detection.
- ONNX.Py: opset=10 introduced Resize op and deprecated Upsample.
- TF, ONNX.Py: skip default values when printing verbose JSON.

## [0.3.1] - 2019-11-13
- Compute: fixed RenderTexture to Tensor conversion.
- Compute: implemented Loop type of InstanceNormTail kernel, fixes Pix2Pix inputs that are larger than 256x256.
- Compute: fixed dummy texture cache to properly survive domain reload.
- Compute: added support for 2 channels in TensorToTexture.
- Compute: fixed kernel compilation on consoles.
- Compute: disabled compute buffer clearing when data is coming from texture, saves a lot of GC allocations.
- Compute: various small optimizations for Kernel selection and data preparation, saves some GC allocations.
- Docs: added instructions how to export model from TF2.x.

## [0.3.0] - 2019-10-29
- ONNX: Implemented .onnx asset importer as Editor Plugin. Now .onnx files can be added directly to the project as regular assets.
Python script is not needed to import ONNX models anymore. When loading from script reference ONNX asset as `NNModel`. 
- ONNX: Barracuda ONNX importer supports model versions up to `opset-8`.
- Added `Reflect`, `Edge` and `Symmetric` padding implementations.
- TF/ONNX: Improved padding import for TF and ONNX models.
- TF/ONNX: Improved StridedSlice support.
- TF/ONNX: Implemented logic operation support.
- Added `ModelBuilder` utility to make it easier to build `Model` from code.
- UI: Clicking on `*.nn` and `*.onnx` files now will display layers that model contains, right in Unity Editor Inspector.
- Small clean up of public APIs.
- Refactored `Model` fields from array to list.
- Performance: improved heuristics of chosing Convolutional kernels - expect better performance with thin layers (channels < 16).
- Fixed several bugs regarding padding and odd kernel dimensions (3x3) in Conv2dTranspose op.
- Fixed out of bounds reads in `Conv2DKernelKxK_T16x16_R4x4` kernel when tensor `batch*width*heigh` is not modulo of 64 in Compute backend.
- Fixed occasional NaNs in Tanh op by clamping large activation values in Compute & Reference Compute backends.
- Fixed bug in conversion of single channel Texture to Tensor. Bug would cause output 3 times lower than expected.
- Fixed discrepancy in InstanceNormalization op output by using epsilon specified in the model instead of hardcoded one.
- Fixed support for older iOS and Android devices that do not support total group size of 256 or higher. Added work around to skip this kernel.
- Docs: first pass of Barracuda API documentation.



## [0.2.7] - 2019-09-27
- Barracuda is now available in Unity Package Manager
- Adjusted release notes format

## 0.2.6
- Small changes to adhere UPM package structure.

## 0.2.5
- Fixed MaxPool2D to not take padding value into calculations on GPU backends. Now matches TF and C# impl.
- Fixed dense layers that have scalar input.
- Added support for LSTM blocks that have concat as the output. 
- Added support for LSTM nodes in non-root scope.
- Added support for comparison and logical operator in tensorflow and onnx importers.

## 0.2.4
- Switched to 2018.4.3f1 as primary Unity version for testing.
- Fixed ScaleBias scheduling issue with large amounts of data (reproduced with MobileNet @ 16 batch)
- Fixed buffer overrun in ThreadGroup SharedMemory when TRANSPOSE_X and/or SHIFTED_X paths are enabled. This should fix GPU worker issues on Windows.
- Added string cache to minimise string concat generated GC pressure.
- Added small fixes for temp memory allocations, saves ~200B per layer.
- Refactored inner loop workings, to avoid GC allocations for delegates.
- Fixed input handling for layers, now inputs are not regenerated with every execution. Static model tensors are to stay forever until worker is disposed.
- Bumped Burst version to 1.1.1.

## 0.2.3
- Rewritten Dense, Conv and some other ops on GPU. Speedup of 33% in most models with batch=1 and over 100% for batch=16.
- Optimizations: reimplemented InstanceNormalization using pyramid approach for calculating mean and variance.

## 0.2.2
- Added support for --print-supported-ops flag for model converters, now it will print approximate list of supported operations. List of supported ops depends on converter.
- Added Keras converter as part of distribution.
- Now compute shaders are loaded only if GPU worker is requested.
- Fixed bug in MaxPool and AvgPool padding. Issue discovered by Yolo faces network.
- Fixed bug in Transpose convolution support for C# backend.
- Fixed TF model conversion with two LSTM cells.
- Fixed case when strided slice end overflows to zero and thus producing negative range.

## 0.2.1
- TF importer: fixed ResizeNearestNeighbor aka Upsample2D scaling factor detection.
- TF importer: optimized node sorting. Should be faster than 0.2.0.
- TF importer: made detection of actual output node from LSTM/GRU pattern more bullet proof by skipping Const nodes.
- TF importer: improved InstanceNormalization handling.
- TF importer: fixed SquareDifference pattern.
- TF importer: fixed Conv2DBackpropInput (transpose convolution) import. 
- Fixed Conv2D performance regression on some GPUs.
- Fixed TextureAsTensorData.Download() to work properly with InterpretDepthAs.Channels.
- Fixed bug when identity/nop layers would reuse input as an output and later causing premature release of that tensor as part of intermediate data cleanup.
- Added scale + bias to TenstorToRenderTexture interface, usefull for adjusting network output scale + bias on the fly.
- Fixed double Dispose issue when worker gets garbage collected.

## 0.2.0
- Version bumped to 0.2.0 as it brings breaking API changes, for details look below. 
- Significantly reduced temporary memory allocations by introducing internal allocator support. Now memory is re-used between layer execution as much as possible.
- Improved small workload performance on CSharp backend
- Added parallel implementation for multiple activation functions on CSharp backend
- Added `Peek()` function to `IWorker`, it retains object storage in worker's allocator, useful for quick grabbing of output. If you want to preserve content of output tensor between `Execute()` invocations, then use `Fetch()`.
- Fixed ESRGAN model conversion (ONNX importer).
- Fixed Tensor <-> Texture copy for textures/tensors that dimensions are not multiple of 8.
- Added `Summary()` method to `Worker`. Currently returns allocator information.
- Tabs to spaces! Aiming at higher salary (https://stackoverflow.blog/2017/06/15/developers-use-spaces-make-money-use-tabs/).
- Renamed worker type enum members: `CSharp` -> `CSharpRef`, `CSharpFast` -> `CSharp`, `Compute` -> `ComputeRef`, `ComputeFast` -> `Compute`.
- Implemented new optimized `ComputePrecompiled` worker. This worker caches Compute kernels and state beforehand to reduce CPU overhead. 
- Added `ExecuteAsync()` to `IWorker` interface, it returns `IEnumerator`, which enables you to control how many layers to schedule per frame (one iteration == one layer).
- Added `Log` op support on Compute workers.
- Optimized activation functions and ScaleBias by accessing tensor as continuous array. Gained ~2.0ms on 4 batch MobileNet (MBP2016).
- Introduced _Loop version of activations to fight 65535 scheduling limit on D3D11.
- Added .nn as Barracuda model file extension for use in Unity Editor. Also added simple editor importer. Now you can declare serializable fields as NNModel to bind them to .nn asset. ModelLoader.Load() now accepts NNModel as a source.
- Compute: Reduce reference GPU implementation.
- TF importer: Expanded Mean support to mean over channels, implemented Pad (as Border2D), implemented SquaredDifference, added InstanceNormalization and LeakyRelu patterns, StridedSlice implementation.
- TF importer: sort model nodes by dependencies before processing.
- Fixed ComputeBuffer leak when using Compute and ComputePrecompiled backends.
- Made to use Conv2D_L1Cached64_RegisterBlock4x4 more often: improves perf ~2x on Vega 16, and ~30% on Nvidia and Intel.

## 0.1.6
- Added activation type print in verbose mode
- Added fast and parallel CPU implementation for Swish, Relu, Add, Sub, Div, Min, Max, Tanh, Exp
- Removed duplicate profiler blocks for ops
- Improved scheduling on CPU for small batches of data
- Fixed compatibility with Unity 2019.2.x

## 0.1.5
- Added Transpose, MatMul and Indentity layer support for models exported from ONNX.
- Added BasicLSTM layer support for models exported from TF. Limited set of LSTM networks should work now.
- Added DepthwiseConv2D layer support. Most of the networks based on the MobileNet should work now.
- Added OneHot layer support for models exported from TF.
- Added optimized path for Conv2D, Dense and Transpose layers with single batch executions. Performance gain up to 100%.
- Fixed FMA performance issue on Metal GFX platforms.
- Added fast optimized path for Sigmoid and Mul layers on CPU.
- Fixed issue when worker is executed with different batch sizes.
- Added ``pip`` requirements file for Python dependencies, check ``Tools/requirements.txt```.
- Added proof of concept Docker wrappers for running model conversion inside of Docker container. Check ``Tools/docker-tensorflow-to-barracuda.sh`` and ``Tools/docker-onnx-to-barracuda.sh``. Currently it was tested only on Mac host.
- Refactored model importers for easier integration with ML Agents.
- Fixed input shape determination for Keras sequential model.
- Added metadata about input shapes to model. Look for ``Model.GetShapeByName()``.
- Added API to query constant Tensors embedded into network, look for ``Model.GetTensorByName()``.
- Added reference implementations for Selu, Abs, Neg, Ceil, Floor, Clip, Rcp, Log layers.
- Added support for Mean, Square, StridedSlice and Border2D layers.
- Added support for Swish activation, now it is automatically detected in models.
- Fixed Tanh NaN issue when large argument is passed.
- RandomNormal and RandomUniform now supports either embedded shape constant OR previous tensor shape for input.
- Fixed Keras/TF/ONNX FusedBatchNorm/BatchNorm import and now it takes ``epsilon`` into account.
- Now Barracuda will fallback to CSharpFast if compute shaders are not supported on the current platform.
- Improved compute kernel interop on Android.
- Implemented Pix2Pix model (.pict) importer.

## 0.1.4
- Implemented fast Conv2DTrans. Useful for GAN type networks.
- Fixed few ComputeBuffer handling issues.
- Simplified way to pass texture via ``Tensor`` constructor.
- Documentation improvements.
- Added Unity Companion License as part of distribution.
- Fixed boundary checks for Compute Copy/Concat operations.
- Improved profiling experience, now each layer will be reported separately in Unity Profiler.
- Fixed Broadcast layer support in ``ModelAnalyzer``.
- Exp, Pow and other layers are now also implemented in Compute. Improves RL model inference performance on GPU.
- Added platform specific BLAS plugin support. Out of the box Barracuda ships with Apple Accelerate framework support for iOS and macOS.
- Added Burst BLAS plugin, greatly improves performance in Unity Editor where native OS BLAS is not available. It's packaged as separate package and requires to have Burst enabled.
- Improved memory handling, now less GC allocations should be made per inference execution.

## 0.1.3
- Improved Barracuda support for Unity Profiler.
- Cleaned up Barracuda APIs.
- Added direct ``Texture`` input support. Look for ``TextureAsTensorData``. The following types of texture supported as input: ``Texture2D``, ``Texture2DArray``, ``Texture3D``, ``RenderTexture``.
- Added ``Tensor`` to ``RenderTexture`` conversion. Look for ``TensorToRenderTexture``.
- Autoencoder type networks can run completely on GPU now. Data roundtrip via CPU is not necessary anymore.
- Vertical flip is applied when converting between ``Texture`` and ``Tensor`` to match conventionts. To override this behavior look for ``TextureAsTensorData.Flip`` enum.
- Removed direct reference to WebCamTexture, now Barracuda compiles for Console targets.
- Fixed _Conv2DTranspose_ layer support. Now GANs using _Conv2DTranspose_ work properly.
- Added automated test for pix2pix GAN.

## 0.1.2
- Barracuda now is also available as preview package. Look for ``com.unity.barracuda`` in https://staging-packages.unity.com registry.
- Conv2D layers are now *up to 30x faster* with ``CSharpFast`` backend (``ComputeFast`` remains best backend for convolutional networks).
- Added profiler sample for ``Fetch()``.
- Fixed compilation issues on Xbox One.
- TexConv2D support was temporary disabled.
- Barracuda logging now can be configured via static fields of ``Barracuda.D`` class, it allows both disable specific logging levels or just disable stack trace collection (helps with performance when profiling).
- Compute Concat implementation now will fall back to C# implementation instead of throwing exception when unsupported configuration is encountered. 
- Fixed several ``ComputeBuffer`` release issues. 
- Added constructor for ``Tensor`` that allows to pass in data array.
- Improved Flatten handling in TensorFlow models.
- Added helper func ``ModelLoader.LoadFromStreamingAssets``.
- Fixed .meta file packaging.
- Small docs improvements.
- Fixed unnecessary patching of Activation layers in ``ModelLoader``.
- Added output trimming at run-time. See for extra parameters Worker factory.

## 0.1.1
- First internal realease as drop-in package
- Compatibility with ML Agents models: 3DBall, PushBlock, GridWorld, Soccer.

## 0.1.0
- First internal build. Due some bugs encountered wasn't published.

#Contributors
- Renaldas (ReJ) Zioma
- Mantas Puida
- Vladimir Oster
- Aurimas Petrovas
- Martin Sternevald
- Valdemar Bučilko
- Kuba Cupisz
- Povilas Kanapickas
- Paulius Puodžiūnas
- Florent Guinier
- Alexandre Ribard
