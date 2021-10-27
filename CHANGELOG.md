# Changelog
All notable changes to this package will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [2.3.1] - 2021-10-26
### Fixed
- Fixed `CSharpBurst` backend failure on `RandomNormal` and `RandomUniform` ops

## [2.3.0] - 2021-10-15
### Added
- PixelShader worker! Allowing to run on GPU without requiring support for compute shaders. Note: limited to 4D tensors for now.
- ONNX: Added support for `transA` and `transB` parameters for `Gemm` operator.
- ONNX: Added support for multi-axis `Squeeze`/`Unsqueeze`.
- Added support for `RoiAlign`.

### Changed
- Improved Tensor storage handling for Burst backend, should reduce overall memory pressure when used.
- Improved `Transpose` performance when they are not changing memory layout.
- Improved automatic test coverage for consoles.
- Improved `ConvTranspose2D` performance for large spatial kernel.

### Fixed
- Added ability to create tensor from RenderTexture when compute shaders or GPU are not available.

## [2.2.1] - 2021-09-14
### Fixed
- Fixed `CSharpBurst` Random layer failure when model is loaded from an old .nn file

## [2.2.0] - 2021-09-01
### Added
- Added Burst implementation for Random layers, improves scheduling of the whole network computation
- Added scale and bias options for `TensorFromTexture`
- Added `Erf` layer support
- Added `LogSoftmax` axis support

### Changed
- Improved MatMul performance on `CSharpBurst` backend on Desktop platforms
- Improved temp Tensor allocation handling, should reduce GC pressure
- Improved Tensor storage handling for PrecompiledCompute backend, should reduce overall memory pressure when used.
- Exposed flag for GPU worker to take ownership of weights, reduces memory pressure on CPU side, but limits `Model` object re-use.
- Improved convolution performance on Mobiles
- Improved 1x1 convolution performance
- Minimal supported Unity version updated to 2019.4.29f1
- Improved Barracuda installation instructions 

### Fixed
- Fixed broadcasting between rank 4 and rank 2 operands 
- Fixed GPU `Flatten` when channel order is NHWC

## [2.1.0] - 2021-05-17
### Added
- ONNX: Added support for the `metadata_props` field, which is exposed as a <string,string> dictionary via `Model.Metadata`

### Changed
- Improved `Slice` range handling
- Empty tensor buffers are now filled with `-1` instead of the random data, helps to reduce accidental errors
- Greatly reduced memory allocation when using GPU backends (no more CPU mem cache for intermediate tensors). Mostly helps mobiles with unified memory architecture
- Improved `SoftMax` performance
- Improved `LSTM` performance on CPU

### Fixed
- Fixed default axis to `channels` in `ModelBuilder.SoftMax`
- Fixed `Optimize Model` checkbox in model importer UI
- Fixed crash when tensor is created on CPU and later is moved to GPU
- Fixed out of bounds write in NCHW path
- Fixed crash in Tensor finalizers when using `CSharpBurst` backend
- Fixed Gather behavior for rank3 tensors

### Known issues
- There is small regression in ARMv7 performance when running `CSharpBurst` backend. It can be worked around on most of the phones by shipping ARMv8(ARM64) builds

## [2.0.0] - 2021-04-05
### Changed
- Combined verified release of 1.1.x-1.4.x improvements
- Burst updated to 1.6.0
- Now Barracuda requires Unity 2019.4.29f1 or newer 

### Fixed
- Fixed Transpose removal pass for ML-Agents networks

## [1.4.0] - 2021-04-01
### Added
- ONNX: Added support for TF custom ONNX node `MirrorPad`
- ONNX: Added support ops set 11 and 13 for `Pad` operator
- UI: Added a button to open Barracuda imported model in `netron`

### Changed
- Breaking Change: Barracuda now requires Unity 2019.4 LTS or later (was 2018.4 LTS)
- Improved memory usage with `LSTM`

### Fixed  
- Implicitly initialize memories for `LSTM`; Add documentation about usage
- Allocator disposal in GenericWorker
- ML-Agents 1.9 (Release 15) LSTMs (broken in 1.3.3)
- Fixed crash in Tensor cleanup for `CSharpBurst` backend  
- ONNX: Fixed `Resize` when downsampling (need to be by a 1/int factor still) + support for const path for scale.
- ONNX: Fixed `ReduceXXX` ops axis import.

## [1.3.3] - 2021-03-04
### Added
- ONNX: `Round` support

### Changed
- ONNX: Improved `LSTM` support.
- ONNX: better dynamic op shape support
- Performance: Improved Transformer architecture performance on GPU.
- Performance: Better GPU perf for: `MatMul`, `Reduce`, `OneHot`
- Performance: better constant Layer fusing
- Performance: Improved loading times for compute implementation of `Transpose/Transpose8D` ops.
  
### Fixed  
- Fixed `Reduce` op in NCHW mode.
- `CSharpBurst` fixed fencing in Concat, DepthwiseConv2D and PRelu.
- StridedSlice for stride of 2 and odd numbers start index.
- Fixed `CsharpBurst` read fence handling.


## [1.3.2] - 2021-02-22
### Fixed
- numerous small fixes in `CSharpBurst` fences and memory handling.
- small fixes for GPU NCHW path.
- Performance: numerous performance fixes in `CSharpBurst` backend. Transformer like architectures should run faster with this backend.

## [1.3.1] - 2021-01-31
### Changed
- Performance: Improved Transformer (MatMul rank-3 / rank-4) performance on `CSharpBurst` backend.
- Performance: Reduced compute shader loading times for most popular networks.
- Performance: Reduced amount of temp allocations for `Compute`, `ComputePrecompiled` and `CSharpBurst` backends.
- Performance: Improved `StridedSlice`, `Concat` and `Border2D` performance on `CSharpBurst` backend.
  
### Fixed
- ONNX: Fixed import of models with `ArgMin`, `ArgMax`

## [1.3.0] - 2021-01-28
### Added
- ONNX: Added spatial rank 3 support for `Pad` and upsampling via `Resize` or `Upsample`.
- ONNX: Added `LessOrEqual` support.

### Changed
- ONNX: Major ONNX import refactoring, which allows to better track `Tensor` shapes and axes. More models should import correctly.
- ONNX: Improved multidimensional MatMul support.   
- ONNX: Improved 1D/2D `OneHot` handling.
- ONNX: Improved Transformer architecture support.
- ONNX: Improved `Tile` support.
- Improved `Conv3D` op performance on GPU.
- Improved support for existing operations on >4D `Tensors` in general.
- Improved tensor shape display in Model Inspector, now it will show >4D dimensions only if they are actually used.
- Performance: Improved layer fusing and optimization as ONNX post-import step.
- Docs: Improved supported op documentation.
  
### Fixed
- Fix: Fixed axis handling for `Squeeze` op.
- Fix: Fixed `Softmax` negative axis support.
- Fix: Fixed possible name collision in compute shaders responsible for `Texture`<->`Tensor` conversion.

## [1.2.1] - 2020-12-02
### Added
- ONNX: Added `ArgMax`/`ArgMin` support.
- Added `Tensor` constructors for `float[,]` and `float[,,,]`.
  
### Changed
- Now network execution can be restarted with `StartManualSchedule()` without finishing previous execution.
- Now `TextureToTensor` and `TensorToTexture` kernels will initialize faster on Open GL ES 3.1+.  

### Fixed
- Fixed import of 'Gemm' from ONNX when transposing inputs.
- Fixed a performance regression when uploading tensor from CPU to GPU in channel first mode.
- Fixed import of 'reshape-transpose-reshape' pattern from ONNX when multiple channels are present.
- Fixed import of 'reshape' from ONNX when spatial dimension are inserted by target shape.

## [1.2.0] - 2020-10-27
### Added
- Added axis support other than C to `Reduce` ops.
- Added YOLOv3/TinyYOLOv3 support.
- Added Shufflenet/super resolution CNN support.
- Added >4D `Transpose` and `Reshape` support on GPU.
- ONNX: Added `Split` op support.
- ONNX: Added `NonMaxSuppression` op support.
- ONNX: Added axis support to `Softmax`.
- ONNX: Added `ConstantOfShape` op support (constant inputs only).
- ONNX: Added constant as argument support for `Add/Sum/Sub/Mul/Div`.
- ONNX: Added `NonZero` op support.
- Docs: Added multiple missing API docs.
  
### Changed
- Performance: Improved Conv2D execution performance when Winograd kernel is in use.
- Change: Moved some of internal-only intended APIs from `public` to `internal` visibility.

### Fixed
- ONNX: Fixed Rank 1 tensor winding up in C channel rather than N.
- ONNX: Fixes `TopK` implementation to index in default Barracuda format (NHWC).
- ONNX: Fixed improper conversion of Int64, UInt64, Double.
- ONNX: Fixed several issues when importing networks with channels-last input layout and for operation on >4 rank.
- Fixed `Sub/Div` fusing when operation order is important.
- Fixed TensorShape(int[]) constructor when rank is <4.
- Fixed Resample2D op on GPU path.

## [1.1.2] - 2020-10-14
### Fixed
- Fixed matrix multiplication issue in Universal Mac Standalone builds. 

## [1.1.1] - 2020-09-04
### Fixed
- Fixed ONNX input shape detection for NHWC networks.

## [1.1.0] - 2020-08-20
### Added
- ONNX: Added `Conv1D` pad/strides/dilatations support.
- ONNX: Added `MaxPool 1D` support.
- ONNX: Added support for dynamic `MatMul`.
- ONNX: Added `TopK` support.
- Added basic 8D tensor support. Work in progress.

### Changed
- Performance: Linear Layer Fusing for `Add/Mul/Conv/Dense/ScaleBias` ops.
- Performance: Improved `InstanceNorm` performance.
- Performance: Trigonometric activations now could be fused with some layers.
- Now `Transpose` should work in most generic use cases.
- Faster Broadcast ops.
- ONNX: importer moved to the runtime, now it's possible to directly load ONNX at runtime. Look for `ONNXModelConverter`.
- ONNX: Improved trigonometric activation support.
- ONNX: MatMul can be called with dynamic inputs.
  
### Fixed
- Now `NNModelEditor.RenderStaticPreview` should not throw exceptions when Barracuda package is referenced via filesystem on Unity 2018.4.x. 
- Fixed `Concat` behaviour with unusual axis spec.
- Fixed `ConvTranspose` when used with `ComputePrecompiled`.
- Fixed `Concat` when used with `ComputePrecompiled`.
- Fixed issue for some `Squeeze/Unsqueeze` + `ReduceOp` combinations.

## [1.0.4] - 2020-10-07
### Changed
- Performance: improved memory handling while loading model data. This improvement should reduce GC memory pressure.

## [1.0.3] - 2020-09-15
### Fixed
- Fixed matrix multiplication issue in Universal Mac Standalone builds. 

## [1.0.2] - 2020-07-30
### Changed
- Upgraded Burst dependency to 1.3.4.
  
### Fixed
- Docs: Minor fixes. 

## [1.0.1] - 2020-07-02
### Fixed
- Fix in regard to package dependencies for Unity 2020.2 and newer.

## [1.0.0] - 2020-06-01
- First verified release

## [0.8.0] - 2020-05-28
### Added
- ONNX: Added `LRN` support.
- ONNX: Convolutions now support `dilatation`.

### Changed
- Breaking change: API cleanup, number of internal implementation classes marked as `internal`.
- API change: `IWorker.ExecuteAsync` renamed to `IWorker.StartManualSchedule`. `IWorker.ExecuteAsync` is marked as obsolete and will issue warning if used.
- API change: `IWorker.WaitForCompletion` renamed to `IWorker.FlushSchedule`. `IWorker.WaitForCompletion` is marked as obsolete and will issue warning if used.
- API change: `IWorker.GetAsyncProgress` renamed to `IWorker.scheduleProgress`. `IWorker.GetAsyncProgress` is marked as obsolete and will issue warning if used.
- Performance: Introduced new CPU backend powered by Unity Burst compiler `CSharpBurst`. The new backend runs in parallel to the main thread and is roughly ~2x faster in Standalone and ~5x faster in the Editor. Prefer to use yield instruction `WaitForCompletion(output)` before accessing values of the output tensor to avoid blocking the main thread.
- Performance: Improved `Conv2d` performance for smaller workloads.
- Performance: Activation fusion is now supported with `Conv2DTrans` and `DepthwiseConv` layers.
- Performance: Made Broadcast ops (`Add`, `Sub,`, `Mul`, `Div`, `Mean`) more efficient.
- Docs: Major documentation overhaul. 

### Fixed
- ONNX: Fixed `Tranpose` layer import for networks originating from Keras.
- Fixed multiple out of bounds reads and writes in compute shaders. Should improve stability on PS4, XB1, Metal and Vulkan platforms.
- Fixed memory leak in ComputePrecompiled elementwise broadcast ops.
- Fixed broadcast ops for none unit batches

## [0.7.1] - 2020-05-12
### Added
- ONNX: Added DepthToSpace and SpaceToDepth support (*thanks to latentspace.co!*).
- ONNX: Added Expand support (*thanks to latentspace.co!*).
- ONNX: Implemented `sizes` input for Resize op, previously only `scales` input was supported (*thanks to latentspace.co!*).
- ONNX: Automatic activation folding for ConvTranspose and DepthwiseConv
- ONNX: Conv dilatation support
- Compute: Added GPU implementation for Abs, Ceil, Floor, Reciprocal ops. Previously these operations would fallback to CPU.
- Compute: Added GPU implementation for generic MatMul op. Previously this operation would fallback to CPU.

### Changed
- Performance: Slight improved performance of ConvTranpose.

### Fixed
- Fixed issue where second call to `Execute()` would produce incorrect results.
- Fix ConvTranpose for non-unit strides and outputadjustment input. Also fixed ComputePrecompiled path for non-0 bias.
- Small shader fix linked to activation fusing on ComputePrecompile backend.


## [0.7.0] - 2020-04-27
### Added
- API: Added 3D LUT support to TensorToRenderTexture.
- ONNX: ImageScaler layer support (internally maps to ScaleBias).
- ONNX: Added support for Double, Int8, Int16, UInt8, UInt16, UInt32, UInt64 and Bool data types. Now all ONNX specified data types except `string` should be supported.
- Misc: Added bilinear Upsample support.
- Misc: Added support for dynamic Upsample.

### Changed
- API/Breaking change: Barracuda namespace renamed to Unity.Barracuda.
- API/Breaking change: Barracuda assembly renamed to Unity.Barracuda. IMPORTANT: You have to update asmdef files that reference Barracuda in your project!
- Performance: Reduced temporary memory allocations for CPU inference path.
- Performance: Preparation work to run GPU inference path in NCHW (aka channels-first) layout. NCHW offers more performant layout for number of desktop GPUs. No changes are necessary from client API perspective.
- Performance: InstanceNorm optimizations.
- Performance: VarianceMean, AveragePooling optimizations.
- Performance: Relu, Tanh, Sigmoid, Relu6, Swish activations are automatically fused into other layers whenever possible.
- Performance: Neg, Sqrt, Exp, Log elementwise operations are automatically fused into other layers whenever possible.
- Performance: Added GPU implementation for StridedSlice.
- Performance: ConvTranspose improvements.
- Performance: Inspector UI now loads much faster for large models in the Editor.
- Misc: Adjusted file structure layout to comply with Unity Package guidelines.
- Importer: Bumped .nn & .onnx importer version numbers.

### Removed
- Misc: Removed ONNX python tools and docker wrapper from the Barracuda distribution. Please use ONNX C# importer by placing ONNX models directly into Assets folders.

### Fixed
- ONNX: Fixed case when model input is directly passed to Activation layer.
- ONNX: Fixed support for empty null tensors.
- Misc: Fixed license for 3rd party libraries.

## [0.6.3] - 2020-04-03
### Changed
- Performance: over 10x performance improvement for 'Tiny Yolo V2' model and over 5x improvement for 'Mobilenet' model when CPU side inference is used.
- Performance: significantly improved speed of Convolutions on CPU by using fast Matrix Multiplication (powered by BLAS when available) and memory efficient version of Im2Col algorithm. Memory consumption stays constant regardless of kernel size!
- Performance: significantly improved speed of Depthwise Convolution on CPU.
- Performance: optimized Relu, Relu6, LeakyRelu, ScaleBias, AvgPool layers on CPU.
- Misc: Burst dependency bumped to 1.2.3.

### Fixed
- Fix/Performance: BLAS and Burst plugins are not stripped anymore when making IL2CPP builds.
- no more occasional misleading shader warning "use of potentially uninitialized variable (ReadonlyTensor::SafeGet)"


## [0.6.2] - 2020-03-23
### Changed
- Performance: automatically remove no-operation layers from the model (Identity, Linear activation and Flatten).
- ONNX: Automatically remove unused constants and layers.
- ONNX: Made Pads and Scales arguments more fault tolerant and support default values.
- ONNX: Keep ML-Agents metadata in the model. ML-Agents metadata is stored as unreferenced constants.
- API: ModelBuilder: default axis to -1 for simplicity of use.
- API: ModelBuilder: added default constructor that automatically creates new Model.
- API: Implemented RecurrentState automatic batch size detection based on the model information by passing -1 value as `batchSize` argument in the constructor.
- API: Throw exception when incorrect arguments are passed.
- API: Validate model integrity in `WorkerFactory.Create(...)`.
- UI: Improved readability by increasing layers' estate when model has no constants and displaying large numbers with separators in the Inspector.
- Tests: Added variety of tests to test Public APIs.
- Cosmetic changes to error and log messages.

### Fixed
- Prevent deallocation of the user-owned input tensor when passed to Identity or Reshape layer at the beginning of the model. As a side effect of this change Reshape layer now always makes copy of tensor data, this will be optimised in the future.
- Tensor constructor with ComputeBuffer argument accepts correct stride value now.
- ONNX: Fixed constant folding of Transpose node to work properly with input tensor of any rank. Previously would fail, if tensor rank was different than 4.
- UI: Fixed weight totals in the Inspector. Previously shared weights would be counted multiple time blowing up the totals.

## [0.6.1] - 2020-03-03
### Changed
- Performance: improved inference time for 'Tiny Yolo v2' model on Mali-G71 from 600ms to 190ms.
- Compute: significantly improved precision of InstanceNormalization by using Kahan/Neumaier summation and assumean mean algorithms when calculating variance.
- Rewrote CPU reference implementation of InstanceNormalization to use double precision.

### Fixed
- convolution performance and occasional result regression by temporarily disabling 64 element-per-thread version in convolutional kernel. This optimized kernel will come back in a latter release.
- convolutional Winograd kernel crash on iOS devices.
- convolutional Winograd kernel assert and potentially incorrect results when used with ComputePrecompiled backend.
- ONNX: fixed regression in Upsample opset=9.
- ONNX: fixed incorrect asserts when Reshape node contains -1 in shape definition.


## [0.6.0] - 2020-02-14
### Added
- ONNX: added optimization pass for non-variable parts of the execution graph by turning them into constants.
- ONNX: added optimization pass for Tensorflow models (exported with tf2onnx) that removes excessive Tranpose layers.
- ONNX: added correction of input layouts when Tensorflow models (exported with tf2onnx) do not follow ONNX specification.
- ONNX: added stripping of trailing ":0" symbols from Tensorflow models (exported with tf2onnx).
- ONNX: added support of Shape op with non-constant inputs. Fixes some cases with Upsample opset=9.
- ONNX: added basic LSTM op support. Requires _h/_c inputs to be passed directly. Tested only on ML Agents models.
- ONNX: added support for GroupNormalization by implementing special path for Shape followed by Reshape layer.
- TF: added FusedBatchNormV2 and FusedBatchNormV3 support.

### Changed
- Performance: more Conv2D optimizations.
- API: introduced WaitForCompletion yield instruction  to simplify asyncronous download of execution results from GPU to CPU.
- ONNX: fail hard when layer of unknown type is encountered during the model import. Use "Treat Errors as Warnings" flag in asset Inspector, if wish to override this behavior.
- ONNX: show warnings if BatchNormalization & InstanceNormalization weights do not match feature count from the previous layer.
- Compute: asynchronous GPU readback now is enabled for Unity 2018.2 or later. Made several small fixes too.
- Compute: implemented fallback when asynchronous GPU readback is not supported, by waiting for 3 frames to pass.
- UI: separated constants from layers in model inspector UI. Makes it easier to understand the model.
- UI: made model warnings more visible, added icon.
- Docs: detailed that not only copy, but shared tensor data access can happen when using ITensorData.Download() and ITensorData.SharedAccess().

### Fixed
- ONNX: fixed multi-batch PyTorch models with BatchNormalization & InstanceNormalization layers by cutting excessive weights.
- ONNX: fixed support for tensors with rank below 4 in Tranpose and Slice layers.
- ONNX: fixed incorrect defaults for RandomNormal, RandomUniform, RandomNormalLike and RandomUniformLike layers.
- ONNX: silence Squeeze warning when "seq_length" is used.
- Found incorrect Profiler block termination in Compute backend.
- Dangling buffer assert when CPU array buffers are used with fast CPU backend.
- Enabled Google Protobuf for all platforms as ML Agents rely on this package. Fixes building ML Agents on non-desktop platforms.

## [0.5.0] - 2020-01-29
### Added
- API: added worker.CopyOutput() instead of worker.Fetch(). The new name better reflects the functionality.
- API: introduced CreateWorker() convenience function directly on NNModel asset instance. Should make it easier for users instead of looking docs for ModelLoader and then WorkerFactory.
- API: added ModelBuilder.Input() with TensorShape argument.
- API: added Tensor constructor that accepts ComputeBuffer as an argument.
- ONNX: added constant node baking.
- ONNX: added override for global inputs, set "sequence_length" to 1.
- ONNX: added handling of :0 in string name from TF->ONNX import, keep unconnected constants.
- ONNX: added Gather support.
- ONNX: added OneHot support.
- ONNX: added Shape support.
- ONNX: added Abs, Ceil, Floor, Round.

### Changed
- Performance: added small optimizations for Dense layers on GPU.
- API: ModelLoader.Load() verbose parameter default value changed from `true` to `false`.
- Importers: .nn and .onnx importer versions were increased, expect model reimport to happen automatically.
- Examples: updated according with the latest API changes.
- ONNX: C# importer refactoring and cleanup.
- ONNX: improved ONNX asset handling performance in Unity Editor UI.
- ONNX: implemented Upsample support for opset=9!
- ONNX: fixed Upsample support for opset=7,8.
- ONNX: implemented Slice for constant tensors.
- ONNX: implemented Resize op using Upsample2D or AvgPool2D depending, if scale is larger than 1 or not.
- ONNX: implemented arbitrary axis support for Concat.
- ONNX: implemented multiple axes support for Reduce ops.
  
### Fixed
- Fixed Unity 2020.1 support. Unity versions > 2020.1.0a21 should work fine now.
- Moved MatrixUtils and ComputeShaderSingleton to Barracuda namespace.
- Fixed GPU kernel selection for iPhone7 (A10), as it is limited to 224 thread groups.

## [0.4.0] - 2020-01-08
### Added
- ONNX: added support for importing float16 models. They will be upscaled to float32 during import. No runtime support yet.
- Docs: added RecurrentState docs

### Changed
- Now Barracuda ships as a source! At the moment no PRs are accepted, but this policy might change in the future.
- Bumped min supported Unity version to 2018.x.
- Moved Burst BLAS plugin to the Barracuda Core package.
- API: Renamed AddInput() -> SetInput and Peek() -> PeekOutput(), old methods are still available, but will be removed in the future.
- Performance: improved optimal kernel selection for Conv2D.
- Performance: improved activation function performance.
- Cleanup: removed unused Compute Kernels.
  
### Fixed
- Kernel compilation errors on consoles.
- Possible out of bounds read in Border2D op.
- Prevent duplicate outputs to be added to the model via ModelBuilder.
- Made CreateWorker arguments `trimOutputs` and `additionalOutputs` more robust against using incorrect layer names that do not actually exist in the model.

## [0.3.2] - 2019-11-22
### Added
- Implemented PRelu activation
- ONNX: enabled LogSoftmax, Sqrt, Clip, Reciprocal and Neg ops.
- ONNX: enabled RandomNormal, RandomNormalLike, RandomUniform, RandomUniformLike ops
- ONNX.Py: added globals (meta info) detection.
- ONNX.Py: opset=10 introduced Resize op and deprecated Upsample.
  
### Changed
- Performance: improved InstanceNormalization performance on GPU, 40% faster.
- Performance: implemented batch support for InstanceNormalization.
- ONNX: implemented opset9 support for Upsample op. In opset9 `scales` parameter became specified as input tensor instead of attribute.
- TF, ONNX.Py: skip default values when printing verbose JSON.

### Fixed
- Added workaround for buggy Adreno + Android 8 Vulkan drivers.
- Moved Protobuf to the runtime plugins so it can be shared with ML-Agents.
- Compatibility with Unity 2017.4.x.

## [0.3.1] - 2019-11-13
### Added
- Docs: added instructions how to export model from TF2.x.
- Compute: added support for 2 channels in TensorToTexture.

### Fixed
- Compute: fixed RenderTexture to Tensor conversion.
- Compute: implemented Loop type of InstanceNormTail kernel, fixes Pix2Pix inputs that are larger than 256x256.
- Compute: fixed dummy texture cache to properly survive domain reload.
- Compute: fixed kernel compilation on consoles.
- Compute: disabled compute buffer clearing when data is coming from texture, saves a lot of GC allocations.
- Compute: various small optimizations for Kernel selection and data preparation, saves some GC allocations.

## [0.3.0] - 2019-10-29
### Added
- ONNX: Implemented .onnx asset importer as Editor Plugin. Now .onnx files can be added directly to the project as regular assets.
Python script is not needed to import ONNX models anymore. When loading from script reference ONNX asset as `NNModel`. 
- ONNX: Barracuda ONNX importer supports model versions up to `opset-8`.
- Added `Reflect`, `Edge` and `Symmetric` padding implementations.
- Added `ModelBuilder` utility to make it easier to build `Model` from code.
- Docs: first pass of Barracuda API documentation.
  
### Changed
- TF/ONNX: Improved padding import for TF and ONNX models.
- TF/ONNX: Improved StridedSlice support.
- TF/ONNX: Implemented logic operation support.
- UI: Clicking on `*.nn` and `*.onnx` files now will display layers that model contains, right in Unity Editor Inspector.
- Small clean up of public APIs.
- Refactored `Model` fields from array to list.
- Performance: improved heuristics of chosing Convolutional kernels - expect better performance with thin layers (channels < 16).
  
### Fixed
- Fixed several bugs regarding padding and odd kernel dimensions (3x3) in Conv2dTranspose op.
- Fixed out of bounds reads in `Conv2DKernelKxK_T16x16_R4x4` kernel when tensor `batch*width*heigh` is not modulo of 64 in Compute backend.
- Fixed occasional NaNs in Tanh op by clamping large activation values in Compute & Reference Compute backends.
- Fixed bug in conversion of single channel Texture to Tensor. Bug would cause output 3 times lower than expected.
- Fixed discrepancy in InstanceNormalization op output by using epsilon specified in the model instead of hardcoded one.
- Fixed support for older iOS and Android devices that do not support total group size of 256 or higher. Added work around to skip this kernel.


## [0.2.7] - 2019-09-27
### Added
- Barracuda is now available in Unity Package Manager
  
### Changed
- Adjusted release notes format

## 0.2.6
### Changed
- Small changes to adhere UPM package structure.

## 0.2.5
### Added
- Added support for LSTM blocks that have concat as the output. 
- Added support for LSTM nodes in non-root scope.
- Added support for comparison and logical operator in tensorflow and onnx importers.

### Fixed
- Fixed MaxPool2D to not take padding value into calculations on GPU backends. Now matches TF and C# impl.
- Fixed dense layers that have scalar input.

## 0.2.4
### Added
- Added string cache to minimise string concat generated GC pressure.
- Added small fixes for temp memory allocations, saves ~200B per layer.

### Changed
- Switched to 2018.4.3f1 as primary Unity version for testing.
- Bumped Burst version to 1.1.1.
- Refactored inner loop workings, to avoid GC allocations for delegates.
  
### Fixed
- Fixed ScaleBias scheduling issue with large amounts of data (reproduced with MobileNet @ 16 batch)
- Fixed buffer overrun in ThreadGroup SharedMemory when TRANSPOSE_X and/or SHIFTED_X paths are enabled. This should fix GPU worker issues on Windows.
- Fixed input handling for layers, now inputs are not regenerated with every execution. Static model tensors are to stay forever until worker is disposed.


## 0.2.3
### Changed
- Rewritten Dense, Conv and some other ops on GPU. Speedup of 33% in most models with batch=1 and over 100% for batch=16.
- Optimizations: reimplemented InstanceNormalization using pyramid approach for calculating mean and variance.

## 0.2.2
### Added
- Added support for --print-supported-ops flag for model converters, now it will print approximate list of supported operations. List of supported ops depends on converter.
- Added Keras converter as part of distribution.
  
### Changed
- Now compute shaders are loaded only if GPU worker is requested.
  
### Fixed
- Fixed bug in MaxPool and AvgPool padding. Issue discovered by Yolo faces network.
- Fixed bug in Transpose convolution support for C# backend.
- Fixed TF model conversion with two LSTM cells.
- Fixed case when strided slice end overflows to zero and thus producing negative range.

## 0.2.1
### Added
- Added scale + bias to TenstorToRenderTexture interface, usefull for adjusting network output scale + bias on the fly.

### Fixed
- TF importer: fixed ResizeNearestNeighbor aka Upsample2D scaling factor detection.
- TF importer: optimized node sorting. Should be faster than 0.2.0.
- TF importer: made detection of actual output node from LSTM/GRU pattern more bullet proof by skipping Const nodes.
- TF importer: improved InstanceNormalization handling.
- TF importer: fixed SquareDifference pattern.
- TF importer: fixed Conv2DBackpropInput (transpose convolution) import. 
- Fixed Conv2D performance regression on some GPUs.
- Fixed TextureAsTensorData.Download() to work properly with InterpretDepthAs.Channels.
- Fixed bug when identity/nop layers would reuse input as an output and later causing premature release of that tensor as part of intermediate data cleanup.
- Fixed double Dispose issue when worker gets garbage collected.

## 0.2.0
### Added
- Added parallel implementation for multiple activation functions on CSharp backend
- Added `Peek()` function to `IWorker`, it retains object storage in worker's allocator, useful for quick grabbing of output. If you want to preserve content of output tensor between `Execute()` invocations, then use `Fetch()`.
- Added `Summary()` method to `Worker`. Currently returns allocator information.
- Added `ExecuteAsync()` to `IWorker` interface, it returns `IEnumerator`, which enables you to control how many layers to schedule per frame (one iteration == one layer).
- Added `Log` op support on Compute workers.
- Added .nn as Barracuda model file extension for use in Unity Editor. Also added simple editor importer. Now you can declare serializable fields as NNModel to bind them to .nn asset. ModelLoader.Load() now accepts NNModel as a source.
- Compute: Reduce reference GPU implementation.

### Changed
- Version bumped to 0.2.0 as it brings breaking API changes, for details look below. 
- Significantly reduced temporary memory allocations by introducing internal allocator support. Now memory is re-used between layer execution as much as possible.
- Improved small workload performance on CSharp backend
- Tabs to spaces! Aiming at higher salary (https://stackoverflow.blog/2017/06/15/developers-use-spaces-make-money-use-tabs/).
- Renamed worker type enum members: `CSharp` -> `CSharpRef`, `CSharpFast` -> `CSharp`, `Compute` -> `ComputeRef`, `ComputeFast` -> `Compute`.
- Implemented new optimized `ComputePrecompiled` worker. This worker caches Compute kernels and state beforehand to reduce CPU overhead. 
- Optimized activation functions and ScaleBias by accessing tensor as continuous array. Gained ~2.0ms on 4 batch MobileNet (MBP2016).
- Introduced _Loop version of activations to fight 65535 scheduling limit on D3D11.
- TF importer: Expanded Mean support to mean over channels, implemented Pad (as Border2D), implemented SquaredDifference, added InstanceNormalization and LeakyRelu patterns, StridedSlice implementation.
- TF importer: sort model nodes by dependencies before processing.
- Made to use Conv2D_L1Cached64_RegisterBlock4x4 more often: improves perf ~2x on Vega 16, and ~30% on Nvidia and Intel.

### Fixed
- Fixed ESRGAN model conversion (ONNX importer).
- Fixed Tensor <-> Texture copy for textures/tensors that dimensions are not multiple of 8.
- Fixed ComputeBuffer leak when using Compute and ComputePrecompiled backends.


## 0.1.6
### Added
- Added activation type print in verbose mode
- Added fast and parallel CPU implementation for Swish, Relu, Add, Sub, Div, Min, Max, Tanh, Exp
  
### Changed
- Improved scheduling on CPU for small batches of data

### Removed
- Removed duplicate profiler blocks for ops

### Fixed
- Fixed compatibility with Unity 2019.2.x

## 0.1.5
### Added
- Added Transpose, MatMul and Indentity layer support for models exported from ONNX.
- Added BasicLSTM layer support for models exported from TF. Limited set of LSTM networks should work now.
- Added DepthwiseConv2D layer support. Most of the networks based on the MobileNet should work now.
- Added OneHot layer support for models exported from TF.
- Added optimized path for Conv2D, Dense and Transpose layers with single batch executions. Performance gain up to 100%.
- Added fast optimized path for Sigmoid and Mul layers on CPU.
- Added ``pip`` requirements file for Python dependencies, check ``Tools/requirements.txt```.
- Added proof of concept Docker wrappers for running model conversion inside of Docker container. Check ``Tools/docker-tensorflow-to-barracuda.sh`` and ``Tools/docker-onnx-to-barracuda.sh``. Currently it was tested only on Mac host.
- Added metadata about input shapes to model. Look for ``Model.GetShapeByName()``.
- Added API to query constant Tensors embedded into network, look for ``Model.GetTensorByName()``.
- Added reference implementations for Selu, Abs, Neg, Ceil, Floor, Clip, Rcp, Log layers.
- Added support for Mean, Square, StridedSlice and Border2D layers.
- Added support for Swish activation, now it is automatically detected in models.
- RandomNormal and RandomUniform now supports either embedded shape constant OR previous tensor shape for input.
- Implemented Pix2Pix model (.pict) importer.

### Changed
- Refactored model importers for easier integration with ML Agents.
- Now Barracuda will fallback to CSharpFast if compute shaders are not supported on the current platform.
- Improved compute kernel interop on Android.

### Fixed
- Fixed FMA performance issue on Metal GFX platforms.
- Fixed issue when worker is executed with different batch sizes.
- Fixed input shape determination for Keras sequential model.
- Fixed Tanh NaN issue when large argument is passed.
- Fixed Keras/TF/ONNX FusedBatchNorm/BatchNorm import and now it takes ``epsilon`` into account.


## 0.1.4
### Added
- Implemented fast Conv2DTrans. Useful for GAN type networks.
- Added Unity Companion License as part of distribution.
- Added platform specific BLAS plugin support. Out of the box Barracuda ships with Apple Accelerate framework support for iOS and macOS.
- Added Burst BLAS plugin, greatly improves performance in Unity Editor where native OS BLAS is not available. It's packaged as separate package and requires to have Burst enabled.
  
### Changed
- Simplified way to pass texture via ``Tensor`` constructor.
- Improved profiling experience, now each layer will be reported separately in Unity Profiler.
- Exp, Pow and other layers are now also implemented in Compute. Improves RL model inference performance on GPU.
- Improved memory handling, now less GC allocations should be made per inference execution.
- Documentation improvements.

### Fixed
- Fixed few ComputeBuffer handling issues.
- Fixed boundary checks for Compute Copy/Concat operations.
- Fixed Broadcast layer support in ``ModelAnalyzer``.

## 0.1.3
### Added
- Added direct ``Texture`` input support. Look for ``TextureAsTensorData``. The following types of texture supported as input: ``Texture2D``, ``Texture2DArray``, ``Texture3D``, ``RenderTexture``.
- Added ``Tensor`` to ``RenderTexture`` conversion. Look for ``TensorToRenderTexture``.

### Changed
- Improved Barracuda support for Unity Profiler.
- Cleaned up Barracuda APIs.
- Autoencoder type networks can run completely on GPU now. Data roundtrip via CPU is not necessary anymore.
- Vertical flip is applied when converting between ``Texture`` and ``Tensor`` to match conventionts. To override this behavior look for ``TextureAsTensorData.Flip`` enum.

### Removed
- Removed direct reference to WebCamTexture, now Barracuda compiles for Console targets.

### Fixed
- Fixed _Conv2DTranspose_ layer support. Now GANs using _Conv2DTranspose_ work properly.

## 0.1.2
### Added
- Barracuda now is also available as preview package. Look for ``com.unity.barracuda`` in https://staging-packages.unity.com registry.
- Added profiler sample for ``Fetch()``.
- Added constructor for ``Tensor`` that allows to pass in data array.
- Added helper func ``ModelLoader.LoadFromStreamingAssets``.
- Added output trimming at run-time. See for extra parameters Worker factory.

### Changed
- Conv2D layers are now *up to 30x faster* with ``CSharpFast`` backend (``ComputeFast`` remains best backend for convolutional networks).
- TexConv2D support was temporary disabled.
- Barracuda logging now can be configured via static fields of ``Barracuda.D`` class, it allows both disable specific logging levels or just disable stack trace collection (helps with performance when profiling).
- Compute Concat implementation now will fall back to C# implementation instead of throwing exception when unsupported configuration is encountered. 
- Improved Flatten handling in TensorFlow models.
- Small docs improvements.

### Fixed
- Fixed compilation issues on Xbox One.
- Fixed several ``ComputeBuffer`` release issues.
- Fixed .meta file packaging.
- Fixed unnecessary patching of Activation layers in ``ModelLoader``.

## 0.1.1
### Added
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
- Amir Ebrahimi
