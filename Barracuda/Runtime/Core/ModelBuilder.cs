using System;
using System.Collections.Generic;
using System.Linq;
using Unity.Barracuda.Compiler.Passes;
using UnityEngine.Assertions;

namespace Unity.Barracuda
{
    /// <summary>
    /// Class responsible for run-time model building from Neural Net primitives.
    /// </summary>
    public class ModelBuilder
    {
        readonly Model m_Model;

        /// <summary>
        /// Model under construction
        /// </summary>
        public Model model => m_Model;

        /// <summary>
        /// Create a model builder helper to construct the underlying Model.
        /// </summary>
        /// <param name="model">base model to continue building on</param>
        public ModelBuilder(Model model = null)
        {
            if (model == null)
                model = new Model();
            m_Model = model;
        }

        /// <summary>
        /// Add an input to the model
        /// </summary>
        /// <param name="name">input name</param>
        /// <param name="shape">input shape</param>
        /// <param name="rank">input rank</param>
        /// <returns>Input instance</returns>
        public Model.Input Input(string name, Int32[] shape, int rank)
        {
            m_Model.inputs.Add(new Model.Input {name = name, shape = shape, rank = rank});

            return m_Model.inputs.Last();
        }

        /// <summary>
        /// Add an input to the model
        /// </summary>
        /// <param name="name">input name</param>
        /// <param name="shape">input shape</param>
        /// <returns>Input instance</returns>
        public Model.Input Input(string name, TensorShape shape)
        {
            m_Model.inputs.Add(new Model.Input {name = name, shape = shape.ToArray()});

            return m_Model.inputs.Last();
        }

        /// <summary>
        /// Add an input to the model
        /// </summary>
        /// <param name="name">input name</param>
        /// <param name="batch">input batch size</param>
        /// <param name="channels">input channel count</param>
        /// <returns>Input instance</returns>
        public Model.Input Input(string name, Int32 batch, Int32 channels)
        {
            m_Model.inputs.Add(new Model.Input {name = name, shape = new []{batch, 1, 1, channels}});

            return m_Model.inputs.Last();
        }

        /// <summary>
        /// Add an input to the model
        /// </summary>
        /// <param name="name">input name</param>
        /// <param name="batch">input batch size</param>
        /// <param name="height">input height</param>
        /// <param name="width">input width</param>
        /// <param name="channels">input channel count</param>
        /// <returns>Input instance</returns>
        public Model.Input Input(string name, Int32 batch, Int32 height, Int32 width, Int32 channels)
        {
            m_Model.inputs.Add(new Model.Input {name = name, shape = new []{batch, height, width, channels}});

            return m_Model.inputs.Last();
        }

        /// <summary>
        /// Add an output to the model
        /// </summary>
        /// <param name="input">reference object, could be `string`, `Layer` or `Model.Input`</param>
        /// <returns>Output instance</returns>
        public string Output(object input)
        {
            var name = ResolveInput(input);
            if (!m_Model.outputs.Contains(name))
                m_Model.outputs.Add(name);
            return name;
        }

        /// <summary>
        /// Add memory to the model
        /// </summary>
        /// <param name="input">reference input object, could be `string`, `Layer` or `Model.Input`</param>
        /// <param name="output">reference output object, could be `string`, `Layer` or `Model.Input`</param>
        /// <param name="shape">memory shape</param>
        /// <returns>Memory instance</returns>
        public Model.Memory Memory(object input, object output, TensorShape shape)
        {
            m_Model.memories.Add(new Model.Memory {
                shape = shape,
                input = ResolveInput(input),
                output = ResolveInput(output)});

            return m_Model.memories.Last();
        }

        private string ResolveInput(object input)
        {
            if (input == null)
                return null;

            if (input is string)
                return input as string;

            if (input is Layer)
                return (input as Layer).name;

            if (input is Model.Input)
                return ((Model.Input)input).name;

            throw new ArgumentException($"Unsupported input type: {input.GetType()}");
        }

        /// <summary>
        /// Allow to load a tensor from constants.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="tensor">data Tensor</param>
        /// <param name="insertionIndex">insertion index in Layer list</param>
        /// <param name="rank">constant rank</param>
        /// <returns>created Layer instance</returns>
        public Layer Const(string name, Tensor tensor, int insertionIndex = -1, int rank = -1)
        {
            Layer layer = new Layer(name, Layer.Type.Load);
            if (rank >= 0)
                layer.axis = rank;
            layer.datasets = new Layer.DataSet[1];
            layer.datasets[0].name            = name;
            layer.datasets[0].shape           = tensor.shape;
            layer.datasets[0].itemSizeInBytes = 4;
            layer.datasets[0].length          = tensor.shape.length;
            layer.datasets[0].offset          = 0;
            layer.weights                     = new float[tensor.shape.length];
            tensor.ToReadOnlyArray().CopyTo(layer.weights, 0);

            if (insertionIndex < 0 || insertionIndex >= m_Model.layers.Count)
                m_Model.layers.Add(layer);
            else
                m_Model.layers.Insert(insertionIndex, layer);

            return layer;
        }

        /// <summary>
        /// Apply per channel scale and bias.
        /// Scale and bias should be tensors of shape [1,1,1, input.shape[C]]
        ///
        /// Output shape is same as input.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="scale">scale data Tensor</param>
        /// <param name="bias">bias data Tensor</param>
        /// <returns>created Layer instance</returns>
        public Layer ScaleBias(string name, object input, Tensor scale, Tensor bias)
        {
            Layer layer = new Layer(name,Layer.Type.ScaleBias);
            layer.inputs = new [] {ResolveInput(input)};
            layer.datasets = new Layer.DataSet[2];
            layer.datasets[0].name            = $"{name}/S";
            layer.datasets[0].shape           = scale.shape;
            layer.datasets[0].itemSizeInBytes = 4;
            layer.datasets[0].length          = scale.shape.length;
            layer.datasets[0].offset          = 0;
            layer.datasets[1].name            = $"{name}/B";
            layer.datasets[1].shape           = bias.shape;
            layer.datasets[1].itemSizeInBytes = 4;
            layer.datasets[1].length          = bias.shape.length;
            layer.datasets[1].offset          = scale.shape.length;
            layer.weights                     = new float[scale.shape.length + bias.shape.length];

            scale.ToReadOnlyArray().CopyTo(layer.weights, 0);
            bias.ToReadOnlyArray().CopyTo(layer.weights, layer.datasets[1].offset);

            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Apply Local Response Normalization as described in the AlexNet paper
        /// https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
        /// It normalizes over local input regions, local region being defined across channels.
        ///
        /// For an element X[n, h, w, c] in a tensor of shape (N x H x W x C), its region is X[n, h, w, cRange]
        /// with cRange = [max(0, c - floor((size - 1) / 2)), min(C - 1, c + ceil((size - 1) / 2)].
        ///
        /// y = x / Pow( bias + alpha * sum( xOverLocalRange ^ 2 ) / size, beta)
        ///
        /// Output shape is same as input.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="alpha">alpha</param>
        /// <param name="beta">beta</param>
        /// <param name="bias">bias</param>
        /// <param name="size">size</param>
        /// <returns>created Layer instance</returns>
        public Layer LRN(string name, object input, float alpha, float beta, float bias, int size)
        {
            Layer layer = new Layer(name, Layer.Type.LRN);
            layer.inputs = new [] {ResolveInput(input)};
            layer.alpha = alpha;
            layer.beta = beta;
            layer.datasets = new Layer.DataSet[1];
            layer.datasets[0].name            = $"{name}/B";
            layer.datasets[0].shape           = new TensorShape(1,1,1,1);
            layer.datasets[0].itemSizeInBytes = 4;
            layer.datasets[0].length          = 1;
            layer.datasets[0].offset          = 0;
            layer.weights = new float[1];
            layer.weights[0] = bias;
            layer.pool = new int[1];
            layer.pool[0] = size;
            m_Model.layers.Add(layer);

            return layer;
        }


        /// <summary>
        /// Takes a tensor as input and outputs a tensor containing the shape of the input tensor.
        /// Optionally, if an axis is specified, then it will return only that part of the shape.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="axis">axis</param>
        /// <returns>created Layer instance</returns>
        public Layer Shape(string name, object input, int axis = -1)
        {
            var layer = new Layer(name, Layer.Type.Shape);
            layer.inputs = new [] { ResolveInput(input) };
            layer.axis = axis; // If positive, then this will return the specific axis of the shape

            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Carries out instance normalization as described in the paper https://arxiv.org/abs/1607.08022
        /// y = scale * (x - mean) / sqrt(variance + epsilon) + bias, where mean and variance are computed per instance per channel.
        /// Scale and bias should be tensors of shape [1,1,1, input.shape[C]]
        ///
        /// Output shape is same as input.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="scale">scale</param>
        /// <param name="bias">bias</param>
        /// <param name="epsilon">epsilon</param>
        /// <returns>created Layer instance</returns>
        public Layer Normalization(string name, object input, Tensor scale, Tensor bias, float epsilon = 1e-5f)
        {
            Layer layer = new Layer(name, Layer.Type.Normalization);
            layer.inputs = new [] {ResolveInput(input)};
            layer.datasets = new Layer.DataSet[2];
            layer.datasets[0].name            = $"{name}/S";
            layer.datasets[0].shape           = scale.shape;
            layer.datasets[0].itemSizeInBytes = 4;
            layer.datasets[0].length          = scale.shape.length;
            layer.datasets[0].offset          = 0;
            layer.datasets[1].name            = $"{name}/B";
            layer.datasets[1].shape           = bias.shape;
            layer.datasets[1].itemSizeInBytes = 4;
            layer.datasets[1].length          = bias.shape.length;
            layer.datasets[1].offset          = scale.shape.length;
            layer.weights                     = new float[scale.shape.length + bias.shape.length];
            layer.beta                        = epsilon;

            scale.ToReadOnlyArray().CopyTo(layer.weights, 0);
            bias.ToReadOnlyArray().CopyTo(layer.weights, layer.datasets[1].offset);

            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Apply a densely connected layer (aka general matrix multiplication or GEMM)
        /// Bias should be a tensor with (batch == input.shape[H] * input.shape[W] * input.shape[C]) and only one other dimensions of size > 1
        /// Weight should be a tensor with (batch == 1) and (height * width * channels == bias.shape[B] * )
        ///
        /// Output shape is [input.shape[B], 1, 1, Weight.shape[H]*Weight.shape[W]*Weight.shape[C]]
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="weight">weight data Tensor</param>
        /// <param name="bias">bias data Tensor</param>
        /// <returns>created Layer instance</returns>
        public Layer Dense(string name, object input, Tensor weight, Tensor bias)
        {
            Layer layer = new Layer(name, Layer.Type.Dense);
            layer.inputs = new [] {ResolveInput(input)};
            layer.datasets = new Layer.DataSet[2];
            layer.datasets[0].name            = $"{name}/W";
            layer.datasets[0].shape           = weight.shape;
            layer.datasets[0].itemSizeInBytes = 4;
            layer.datasets[0].length          = weight.shape.length;
            layer.datasets[0].offset          = 0;
            layer.datasets[1].name            = $"{name}/B";
            layer.datasets[1].shape           = bias.shape;
            layer.datasets[1].itemSizeInBytes = 4;
            layer.datasets[1].length          = bias.shape.length;
            layer.datasets[1].offset          = weight.shape.length;
            layer.weights                     = new float[weight.shape.length + bias.shape.length];

            weight.ToReadOnlyArray().CopyTo(layer.weights, 0);
            bias.ToReadOnlyArray().CopyTo(layer.weights, layer.datasets[1].offset);

            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Applies matrix multiplication between A and B
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input0">first input node</param>
        /// <param name="input1">second input node</param>
        /// <returns>created Layer instance</returns>
        public Layer MatMul(string name, object input0, object input1)
        {
            var inputs = new[] { input0, input1 };
            Layer layer = new Layer(name, Layer.Type.MatMul);
            layer.inputs = inputs.Select(i => ResolveInput(i)).ToArray();

            m_Model.layers.Add(layer);

            return layer;
        }

        private Layer Conv(string name, Layer.Type convType, object input, Int32[] stride, Int32[] pad, Int32[] outputPad, Tensor kernel, Tensor bias)
        {
            Layer layer = new Layer(name, convType);
            layer.pad = pad;
            layer.stride = stride;
            layer.pool = outputPad;
            layer.inputs = new [] {ResolveInput(input)};
            layer.datasets = new Layer.DataSet[2];
            layer.datasets[0].name            = $"{name}/K";
            layer.datasets[0].shape           = kernel.shape;
            layer.datasets[0].itemSizeInBytes = 4;
            layer.datasets[0].length          = kernel.shape.length;
            layer.datasets[0].offset          = 0;
            layer.datasets[1].name            = $"{name}/B";
            layer.datasets[1].shape           = bias.shape;
            layer.datasets[1].itemSizeInBytes = 4;
            layer.datasets[1].length          = bias.shape.length;
            layer.datasets[1].offset          = kernel.shape.length;
            layer.weights                     = new float[kernel.shape.length + bias.shape.length];

            kernel.ToReadOnlyArray().CopyTo(layer.weights, 0);
            bias.ToReadOnlyArray().CopyTo(layer.weights, layer.datasets[1].offset);

            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Apply a spatial 2D convolution on H and W.
        /// Stride should be of size 2 and format is [W, H].
        /// Pad should be of size 4 and format is [pre W, pre H, post W, post H].
        /// Kernel should be a tensor of shape [kernelHeight, kernelWidth, kernelDepth, kernelCount]
        /// Bias should be a tensor with (batch == 1) and (height * width * channels == kernelCount)
        ///
        /// Output batch is same as input.
        /// Output channel is kernel.kernelCount.
        /// output.shape[H,W] = (input.shape[H,W] + pad[1,0] + pad[3,2] - kernel.shape[1,0]) / stride[1,0] + 1.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="stride">stride</param>
        /// <param name="pad">padding</param>
        /// <param name="kernel">kernel weight data Tensor</param>
        /// <param name="bias">bias data Tensor</param>
        /// <returns>created Layer instance</returns>
        public Layer Conv2D(string name, object input, Int32[] stride, Int32[] pad, Tensor kernel, Tensor bias)
        {
            return Conv(name, Layer.Type.Conv2D, input, stride, pad, new int[0], kernel, bias);
        }

        /// <summary>
        /// Apply a spatial 3D convolution on H, W and D.
        /// Stride should be of size 3 and format is [W, H, D].
        /// Pad should be of size 6 and format is [pre W, pre H, pre D, post W, post H, post D].
        /// Kernel should be a tensor of shape [kernelSpatialHeight, kernelSpatialWidth, kernelSpatialDepth, kernelDepth, kernelCount]
        /// Bias should be a tensor with (batch == 1) and (height * width * channels == kernelCount)
        ///
        /// Output batch is same as input.
        /// Output channel is kernel.kernelCount.
        /// output.shape[D,H,W] = (input.shape[D,H,W] + pad[2,1,0] + pad[5,4,3] - kernel.shape[2,1,0]) / stride[2,1,0] + 1.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="stride">stride</param>
        /// <param name="pad">padding</param>
        /// <param name="kernel">kernel weight data Tensor</param>
        /// <param name="bias">bias data Tensor</param>
        /// <returns>created Layer instance</returns>
        public Layer Conv3D(string name, object input, Int32[] stride, Int32[] pad, Tensor kernel, Tensor bias)
        {
            return Conv(name, Layer.Type.Conv3D, input, stride, pad, new int[0], kernel, bias);
        }

        /// <summary>
        /// Apply a spatial 2D depthwise convolution on H and W.
        /// Stride should be of size 2 and format is [W, H].
        /// Pad should be of size 4 and format is [pre W, pre H, post W, post H].
        /// Kernel should be a tensor of shape [kernelHeight, kernelWidth, kernelDepth, kernelCount]
        /// Thus input must have a channel dimension of 1
        /// Bias should be a tensor with (batch == 1) and (height * width * channels == kernelCount)
        ///
        /// Output batch is same as input.
        /// Output channel is kernel.shape[3].
        /// output.shape[H,W] = (input.shape[H,W] + pad[1,0] + pad[3,2] - kernel.shape[1,0]) / stride[1,0] + 1.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="stride">stride</param>
        /// <param name="pad">padding</param>
        /// <param name="kernel">kernel weight data Tensor</param>
        /// <param name="bias">bias data Tensor</param>
        /// <returns>created Layer instance</returns>
        public Layer DepthwiseConv2D(string name, object input, Int32[] stride, Int32[] pad, Tensor kernel, Tensor bias)
        {
            return Conv(name, Layer.Type.DepthwiseConv2D, input, stride, pad, new int[0], kernel, bias);
        }

        /// <summary>
        /// Apply a spatial 2D transposed convolution on H and W.
        /// Stride should be of size 2 and format is [W, H].
        /// Pad should be of size 4 and format is [pre W, pre H, post W, post H].
        /// Kernel should be a tensor of rank 4 of dimensions [kernelHeight, kernelWidth, kernelDepth, kernelCount]
        /// Bias should be a tensor with (batch == 1) and (height * width * channels == kernelCount)
        /// OutputPad should be of length 0 or 2, format is [W, H].
        /// If OutputPad length is 0 it will be defaulted to:
        ///     OutputPad[W,H] = (input.shape[W,H] * stride[0,1] + pad[0,1] + pad[2,3] - [kernelWidth, kernelHeight]) % stride[0,1]
        ///
        /// Output batch is same as input.
        /// Output channel is kernel.shape[3].
        /// output.shape[H,W] = (input.shape[H,W]-1) * stride[0,1] - (pad[1,0] + pad[3,2]) + [kernelWidth, kernelHeight] + OutputPad[W,H]
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="stride">stride</param>
        /// <param name="pad">padding</param>
        /// <param name="outputPad">output padding</param>
        /// <param name="kernel">kernel weight data Tensor</param>
        /// <param name="bias">bias data Tensor</param>
        /// <returns>created Layer instance</returns>
        public Layer Conv2DTrans(string name, object input, Int32[] stride, Int32[] pad, Int32[] outputPad, Tensor kernel, Tensor bias)
        {
            return Conv(name, Layer.Type.Conv2DTrans, input, stride, pad, outputPad, kernel, bias);
        }

        private Layer Pool(Layer.Type type, string name, object input, Int32[] pool, Int32[] stride, Int32[] pad)
        {
            Layer layer = new Layer(name, type);
            layer.pad = pad;
            layer.stride = stride;
            layer.pool = pool;
            layer.inputs = new [] {ResolveInput(input)};

            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Apply 'average' pooling by downscaling H and W dimension according to `pool`, `stride` and `pad`.
        /// Pool and stride should be of size 2 and format is [W, H].
        /// Pad should be of size 4 and format is [pre W, pre H, post W, post H].
        ///
        /// Output batch and channels dimensions the same as input.
        /// output.shape[H,W] = (input.shape[H,W] + pad[1,0] + pad[3,2] - pool[1,0]) / stride[1,0] + 1.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="pool">pooling</param>
        /// <param name="stride">stride</param>
        /// <param name="pad">padding</param>
        /// <returns>created Layer instance</returns>
        public Layer AvgPool2D(string name, object input, Int32[] pool, Int32[] stride, Int32[] pad)
        {
            return Pool(Layer.Type.AvgPool2D, name, input, pool, stride, pad);
        }

        /// <summary>
        /// Apply 'max' pooling by downscaling H and W dimension according to `pool`, `stride` and `pad`.
        /// Pool and stride should be of size 2 and format is [W, H].
        /// Pad should be of size 4 and format is [pre W, pre H, post W, post H].
        ///
        /// Output batch and channels dimensions the same as input.
        /// output.shape[H,W] = (input.shape[H,W] + pad[1,0] + pad[3,2] - pool[1,0]) / stride[1,0] + 1.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="pool">pooling</param>
        /// <param name="stride">stride</param>
        /// <param name="pad">padding</param>
        /// <returns>created Layer instance</returns>
        public Layer MaxPool2D(string name, object input, Int32[] pool, Int32[] stride, Int32[] pad)
        {
            return Pool(Layer.Type.MaxPool2D, name, input, pool, stride, pad);
        }

        /// <summary>
        /// Apply 'average' pooling by downscaling H and W dimension to [1,1]
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <returns>created Layer instance</returns>
        public Layer GlobalAvgPool2D(string name, object input)
        {
            return Pool(Layer.Type.GlobalAvgPool2D, name, input, new int[0], new int[0], new int[0]);
        }

        /// <summary>
        /// Apply 'max' pooling by downscaling H and W dimension to [1,1]
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <returns>created Layer instance</returns>
        public Layer GlobalMaxPool2D(string name, object input)
        {
            return Pool(Layer.Type.GlobalMaxPool2D, name, input, new int[0], new int[0], new int[0]);
        }

        /// <summary>
        /// Upsample the input tensor by scaling W and H by upsample[0] and upsample[1] respectively.
        /// `bilinear` allow to choose between nearest neighbor or bilinear upsampling.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="upsample">upsampling</param>
        /// <param name="bilinear">use bilinear</param>
        /// <returns>created Layer instance</returns>
        public Layer Upsample2D(string name, object input, Int32[] upsample, bool bilinear)
        {
            Layer layer = new Layer(name, Layer.Type.Upsample2D);
            layer.pool = upsample;
            layer.axis = bilinear ? 1: -1;
            layer.inputs = new [] {ResolveInput(input)};

            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Upsample the input tensor
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="source">source input node</param>
        /// <param name="scale">scale input node</param>
        /// <param name="bilinear">use bilinear</param>
        /// <returns>created Layer instance</returns>
        public Layer Upsample2D(string name, object source, object scale, bool bilinear)
        {
            Layer layer = new Layer(name, Layer.Type.Upsample2D);
            layer.axis = bilinear ? 1: -1;
            layer.inputs = new[] { ResolveInput(source), ResolveInput(scale) };

            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Upsample the input tensor by scaling W,H and D by upsample[0], upsample[1] and upsample[2] respectively.
        /// `trilinear` allow to choose between nearest neighbor or trilinear upsampling.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="upsample">scaling factors array [W,H,D]</param>
        /// <param name="trilinear">trilinear flag</param>
        /// <returns>created Layer instance</returns>
        public Layer Upsample3D(string name, object input, Int32[] upsample, bool trilinear)
        {
            Layer layer = new Layer(name, Layer.Type.Upsample3D);
            layer.pool = upsample;
            layer.axis = trilinear ? 1: -1;
            layer.inputs = new [] {ResolveInput(input)};

            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Upsample the input tensor by scaling W,H and D by scale[0], scale[1] and scale[2] respectively.
        /// `trilinear` allow to choose between nearest neighbor or trilinear upsampling.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="source">input node</param>
        /// <param name="scale">scale Tensor</param>
        /// <param name="trilinear">trilinear flag</param>
        /// <returns>created Layer instance</returns>
        public Layer Upsample3D(string name, object source, object scale, bool trilinear)
        {
            Layer layer = new Layer(name, Layer.Type.Upsample3D);
            layer.axis = trilinear ? 1: -1;
            layer.inputs = new[] { ResolveInput(source), ResolveInput(scale) };

            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Resample2D scales the input tensor to the given resolution (W=size[0], H=size[1]).
        /// `bilinear` allows to choose between nearest neighbour or bilinear sampling.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="size">size</param>
        /// <param name="bilinear">use bilinear</param>
        /// <returns>created Layer instance</returns>
        public Layer Resample2D(string name, object input, Int32[] size, bool bilinear)
        {
            Layer layer = new Layer(name, Layer.Type.Resample2D);
            layer.pool = size;
            layer.axis = bilinear ? 1 : -1;
            layer.inputs = new[] { ResolveInput(input) };

            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Resample2D scales the input tensor to the given resolution (W=size[0], H=size[1]).
        /// `bilinear` allows to choose between nearest neighbour or bilinear sampling.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="size">size tensor</param>
        /// <param name="bilinear">use bilinear</param>
        /// <returns>created Layer instance</returns>
        internal Layer Resample2D(string name, object input, object size, bool bilinear)
        {
            Layer layer = new Layer(name, Layer.Type.Resample2D);
            layer.axis = bilinear ? 1 : -1;
            layer.inputs = new[] { ResolveInput(input), ResolveInput(size) };

            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// DepthToSpace rearranges (permutes) data from depth into blocks of
        /// spatial data. This is the reverse transformation of SpaceToDepth.
        /// More specifically, this op outputs a copy of the input tensor where
        /// values from the depth dimension are moved in spatial blocks to the
        /// height and width dimensions. By default, mode = DCR. In the DCR mode,
        /// elements along the depth dimension from the input tensor are rearranged
        /// in the following order: depth, column, and then row.
        /// In the CRD mode, elements along the depth dimension from the input
        /// tensor are rearranged in the following order: column, row, and depth.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="source">input node</param>
        /// <param name="blocksize">block size</param>
        /// <param name="mode">mode, see `Layer.DepthToSpaceMode`</param>
        /// <returns>created Layer instance</returns>
        public Layer DepthToSpace(string name, object source, int blocksize, string mode)
        {
            Layer layer = new Layer(name, Layer.Type.DepthToSpace);

            layer.pool = new int[] { blocksize, blocksize };
            layer.axis = (int)(Layer.DepthToSpaceMode)Enum.Parse(typeof(Layer.DepthToSpaceMode), mode);
            layer.inputs = new[] { ResolveInput(source) };

            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// SpaceToDepth rearranges blocks of [blocksize, blocksize] spatial data into depth.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="source">input node</param>
        /// <param name="blocksize">block size</param>
        /// <returns>created Layer instance</returns>
        public Layer SpaceToDepth(string name, object source, int blocksize)
        {
            Layer layer = new Layer(name, Layer.Type.SpaceToDepth);

            layer.pool = new int[] { blocksize, blocksize };
            layer.inputs = new[] { ResolveInput(source) };

            m_Model.layers.Add(layer);

            return layer;
        }


        /// <summary>
        /// Apply symbolic shape to input tensor. Symbolic shape can have up to one dimension specified as unknown (value -1).
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="shape">shape</param>
        /// <param name="rank">rank</param>
        /// <returns>created Layer instance</returns>
        public Layer Reshape(string name, object input, int[] shape, int rank = -1)
        {
            Layer layer = new Layer(name, Layer.Type.Reshape);
            layer.pool = shape;
            if (rank >= 0)
                layer.pad = new[] { rank };
            layer.inputs = new [] {ResolveInput(input)};

            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Creates a constant tensor populated with `value` as the same shape of `input`.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="value">value</param>
        /// <returns>created Layer instance</returns>
        public Layer ConstantOfShape(string name, object input, float value)
        {
            Layer layer = new Layer(name, Layer.Type.ConstantOfShape);
            layer.inputs = new[] { ResolveInput(input) };
            layer.alpha = value;

            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Apply shape to the input tensor. Number of elements in the shape must match number of elements in input tensor.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="shape">shape</param>
        /// <returns>created Layer instance</returns>
        public Layer Reshape(string name, object input, TensorShape shape)
        {
            return Reshape(name, input, shape.ToArray());
        }

        /// <summary>
        /// Return a tensor of the shape given as tensor.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="shape">shape</param>
        /// <returns>created Layer instance</returns>
        public Layer Reshape(string name, object input, object shape)
        {
            Layer layer = new Layer(name, Layer.Type.Reshape);
            layer.inputs = new [] {ResolveInput(input), ResolveInput(shape)};
            layer.axis = 1; // Use tensor value as the shape; -1 is legacy for using the shape of input tensor

            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Broadcast the input tensor following the given shape and similar to
        /// numpy.array(input) * numpy.ones(shape). Two corresponding dimension
        /// must have the same value, or the input dimension is 1.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="shape">shape</param>
        /// <returns>created Layer instance</returns>
        public Layer Expand(string name, object input, int[] shape)
        {
            Layer layer = new Layer(name, Layer.Type.Expand);
            layer.inputs = new[] { ResolveInput(input) };
            layer.pool = shape;

            m_Model.layers.Add(layer);

            return layer;
        }
        internal Layer Expand(string name, object input, object shape)
        {
            Layer layer = new Layer(name, Layer.Type.Expand);
            layer.inputs = new[] { ResolveInput(input), ResolveInput(shape) };

            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// From a Tensor of shape [S,R,N,T,D,H,W,C] return a tensor of shape [S,R,N,1,1,1,1,T*D*H*W*C]
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <returns>created Layer instance</returns>
        public Layer Flatten(string name, object input)
        {
            Layer layer = new Layer(name, Layer.Type.Flatten);
            layer.inputs = new [] {ResolveInput(input)};

            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Concatenate a list of tensors into a single tensor. All input tensors must have the same shape, except for the axis to concatenate on.
        /// If axisIs8D==true axis rank is from [S,R,N,T,D,H,W,C] overwise from [N,H,W,C]
        /// `axis` must be superior to -4
        /// `axis` must be inferior to 8 when axisIs8D==true or inferior to 4 if axisIs8D==false
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="inputs">input node</param>
        /// <param name="axis">axis</param>
        /// <param name="axisIs8D">is axis 8D</param>
        /// <returns>created Layer instance</returns>
        public Layer Concat(string name, object[] inputs, int axis = -1, bool axisIs8D=false)
        {
            Layer layer = new Layer(name, Layer.Type.Concat);
            layer.axis = axisIs8D?axis:TensorExtensions.Convert4DTo8DAxis(axis);
            layer.inputs = inputs.Select(i => ResolveInput(i)).ToArray();

            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Produces a slice of the input tensor along all axes.
        /// The following rules apply:
        ///     begin=0, end=0, stride=1: copy the full range of elements from the given axis
        ///     begin=A, end=B, stride=1: copy the range [A, B) (excluding the Bth element) from the given axis
        ///     begin=A, end=B, stride=I: copy every Ith element in the range [A, B) from the given axis
        ///     begin=N, end=N, stride=0: shrink axis to a single Nth element
        /// output.shape[*] = (ends[*] - starts[*]) / max(1, stride[*])
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="starts">starts</param>
        /// <param name="ends">ends</param>
        /// <param name="strides">strides</param>
        /// <returns>created Layer instance</returns>
        public Layer StridedSlice(string name, object input, int[] starts, int[] ends, int[] strides)
        {
            Layer layer = new Layer(name, Layer.Type.StridedSlice);
            layer.inputs = new [] {ResolveInput(input)};
            layer.pad = starts;
            layer.pool = ends;
            layer.stride = strides;

            m_Model.layers.Add(layer);

            return layer;
        }

        internal Layer StridedSlice(string name, object input, int[] starts, int[] ends, int[] strides, int[] axes)
        {
            Layer layer = new Layer(name, Layer.Type.StridedSlice);
            layer.inputs = new[] { ResolveInput(input) };
            layer.pad = starts;
            layer.pool = ends;
            layer.stride = strides;
            layer.axes = axes;

            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Constructs a tensor by repeating the input tensor the number of times given by repeats
        /// For example input = [[1, 2], [3, 4]], repeats = [1, 2], Tile(input, repeats) = [[1, 2, 1, 2], [3, 4, 3, 4]]
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="repeats">tile repeats</param>
        /// <returns>created Layer instance</returns>
        public Layer Tile(string name, object input, int[] repeats)
        {
            Layer layer = new Layer(name, Layer.Type.Tile);
            layer.inputs = new[] { ResolveInput(input) };
            layer.pool = repeats;

            m_Model.layers.Add(layer);

            return layer;
        }
        internal Layer Tile(string name, object input, object repeats)
        {
            Layer layer = new Layer(name, Layer.Type.Tile);
            layer.inputs = new[] { ResolveInput(input), ResolveInput(repeats) };
            //layer.pool = repeats;

            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Make a shallow copy of the input tensor.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <returns>created Layer instance</returns>
        public Layer Copy(string name, object input)
        {
            Layer layer = new Layer(name, Layer.Type.Nop);
            layer.inputs = new [] {ResolveInput(input)};
            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Maps integer to one-hot vector of length equal to depth.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="depth">depth</param>
        /// <param name="on">on value</param>
        /// <param name="off">off value</param>
        /// <returns>created Layer instance</returns>
        public Layer OneHot(string name, object input, int depth, int on, int off)
        {
            Layer layer = new Layer(name, Layer.Type.OneHot);
            layer.inputs = new [] {ResolveInput(input)};
            layer.pool = new int[] { depth };
            layer.alpha = on;
            layer.beta = off;

            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Retrieve the indices for top-K largest or smallest elements along a specified axis.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="k">k</param>
        /// <param name="axis">axis</param>
        /// <param name="largest">largest</param>
        /// <param name="sorted">sorted</param>
        /// <returns>created Layer instance</returns>
        public Layer TopKIndices(string name, object input, object k, int axis, bool largest, bool sorted)
        {
            var layer = new Layer(name, Layer.Type.TopKIndices);
            layer.inputs = new [] {ResolveInput(input), ResolveInput(k)};
            layer.axis = axis;
            layer.pad = new [] { largest ? 1 : 0, sorted ? 1 : 0 };

            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Given the indices for top-K largest or smallest elements along a specified axis, return the values
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="indices">indices node</param>
        /// <param name="axis">axis</param>
        /// <returns>created Layer instance</returns>
        public Layer TopKValues(string name, object input, object indices, int axis)
        {
            var layer = new Layer(name, Layer.Type.TopKValues);
            layer.inputs = new [] {ResolveInput(input), ResolveInput(indices)};
            layer.axis = axis;

            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Returns the indices of the elements that are non-zero
        ///  For example an input tensor of shape(1,2,3,1):
        ///  [0, 2, 3],
        ///  [4, 1, 0]
        ///
        ///  Would return a tensor of shape(2, 1, 1, 4)
        ///  N = 2 as the rank of input tensor is 2.
        ///  C = 4 as there exist 3 non zero value in input tensor.
        ///  [0, 0, 1, 1],
        ///  [1, 2, 0, 1]
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <returns>created Layer instance</returns>
        public Layer NonZero(string name, object input)
        {
            var layer = new Layer(name, Layer.Type.NonZero);
            layer.inputs = new [] {ResolveInput(input) };

            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Transpose
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="permutations">list of axis permutations</param>
        /// <returns>created Layer instance</returns>
        public Layer Transpose(string name, object input, int[] permutations)
        {
            Layer layer = new Layer(name, Layer.Type.Transpose);
            layer.inputs = new[] { ResolveInput(input) };
            layer.pool = permutations;

            m_Model.layers.Add(layer);

            return layer;
        }

        internal Layer Squeeze(string name, object input, int[] axes)
        {
            Layer layer = new Layer(name, Layer.Type.Squeeze);
            layer.inputs = new[] { ResolveInput(input) };
            layer.pool = axes;

            m_Model.layers.Add(layer);

            return layer;
        }

        internal Layer Unsqueeze(string name, object input, int[] axes)
        {
            Layer layer = new Layer(name, Layer.Type.Unsqueeze);
            layer.inputs = new[] { ResolveInput(input) };
            layer.pool = axes;

            m_Model.layers.Add(layer);

            return layer;
        }

        private Layer Activation(Layer.Activation activation, string name, object input)
        {
            Layer layer = new Layer(name, activation);
            layer.inputs = new [] {ResolveInput(input)};

            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// No-op layer
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="rank">input rank</param>
        /// <returns>created Layer instance</returns>
        public Layer Identity(string name, object input, int rank = -1)
        {
            Layer identity = Activation(Layer.Activation.None, name, input);
            if (rank > 0)
                identity.pad = new[] { rank };
            return identity;
        }


        /// <summary>
        /// Element-wise `Relu` activation function: f(x) = max(0, x)
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <returns>created Layer instance</returns>
        public Layer Relu(string name, object input)
        {
            return Activation(Layer.Activation.Relu, name, input);
        }

        /// <summary>
        /// Return the Softmax (normalized exponential) values of the input along provided axis.
        /// Thus output will be of shape of the input.
        /// If axisIs8D==true axis rank is from [S,R,N,T,D,H,W,C] overwise from [N,H,W,C]
        /// `axis` must be superior to -4
        /// `axis` must be inferior to 8 when axisIs8D==true or inferior to 4 if axisIs8D==false
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="axis">axis</param>
        /// <param name="axisIs8D">is axis 8D</param>
        /// <returns>created Layer instance</returns>
        public Layer Softmax(string name, object input, int axis=1, bool axisIs8D=false)
        {
            Layer layer = Activation(Layer.Activation.Softmax, name, input);
            layer.axis = axisIs8D ? axis : TensorExtensions.Convert4DTo8DAxis(axis);
            return layer;
        }

        /// <summary>
        /// Return the logSoftmax (log of normalized exponential) values of the input along flatWidth of the input tensor.
        /// Thus output will be of shape of the input.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <returns>created Layer instance</returns>
        public Layer LogSoftmax(string name, object input)
        {
            return Activation(Layer.Activation.LogSoftmax, name, input);
        }

        /// <summary>
        /// Element-wise `Sqrt` activation function
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <returns>created Layer instance</returns>
        public Layer Sqrt(string name, object input)
        {
            return Activation(Layer.Activation.Sqrt, name, input);
        }

        /// <summary>
        /// Element-wise `Tanh` activation function: f(x) = (1 - e^{-2x})/(1 + e^{-2x})
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <returns>created Layer instance</returns>
        public Layer Tanh(string name, object input)
        {
            return Activation(Layer.Activation.Tanh, name, input);
        }

        /// <summary>
        /// Element-wise `Softplus` activation function: f(x) = ln(e^{x} + 1)
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <returns>created Layer instance</returns>
        public Layer Softplus(string name, object input)
        {
            return Activation(Layer.Activation.Softplus, name, input);
        }

        /// <summary>
        /// Element-wise `Sigmoid` activation function: f(x) = 1/(1 + e^{-x})
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <returns>created Layer instance</returns>
        public Layer Sigmoid(string name, object input)
        {
            return Activation(Layer.Activation.Sigmoid, name, input);
        }

        /// <summary>
        /// Element-wise `Elu` activation function: f(x) = x if x >= 0 else alpha*(e^x - 1)
        /// alpha default is 1.0
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="alpha">alpha</param>
        /// <returns>created Layer instance</returns>
        public Layer Elu(string name, object input, float alpha = 1.0f)
        {
            var layer = Activation(Layer.Activation.Elu, name, input);
            layer.alpha = alpha;
            return layer;
        }

        /// <summary>
        /// Element-wise `Relu6` activation function. f(x) = min(max(x, 0), 6)
        /// see http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <returns>created Layer instance</returns>
        public Layer Relu6(string name, object input)
        {
            return Activation(Layer.Activation.Relu6, name, input);
        }

        /// <summary>
        /// Element-wise `LeakyRelu` activation function: f(x) = x if x >= 0 else alpha * x
        /// alpha default is 0.01
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="alpha">alpha</param>
        /// <returns>created Layer instance</returns>
        public Layer LeakyRelu(string name, object input, float alpha = 0.01f)
        {
            var layer = Activation(Layer.Activation.LeakyRelu, name, input);
            layer.alpha = alpha;
            return layer;
        }

        /// <summary>
        /// Element-wise `Selu` activation function: f(x) = gamma * x if x >= 0 else (alpha * e^x - alpha)
        /// alpha default is 1.67326
        /// gamma default is 1.0507
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="alpha">alpha</param>
        /// <param name="gamma">gamma</param>
        /// <returns>created Layer instance</returns>
        public Layer Selu(string name, object input, float alpha = 1.67326f, float gamma = 1.0507f)
        {
            var layer = Activation(Layer.Activation.Selu, name, input);
            layer.alpha = alpha;
            layer.beta = gamma;
            return layer;
        }

        /// <summary>
        /// Element-wise `PRelu` activation function: f(x) = x if x >= 0 else slope * x
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="slope">slope input node</param>
        /// <returns>created Layer instance</returns>
        public Layer PRelu(string name, object input, object slope)
        {
            object[] inputs = new [] {input, slope};

            Layer layer = new Layer(name, Layer.Activation.PRelu);
            layer.inputs = inputs.Select(i => ResolveInput(i)).ToArray();

            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Element-wise `Swish` activation function. f(x) = sigmoid(x) * x = x/(1 + e^{-x})
        /// see https://arxiv.org/abs/1710.05941
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <returns>created Layer instance</returns>
        public Layer Swish(string name, object input)
        {
            return Activation(Layer.Activation.Swish, name, input);
        }

        /// <summary>
        /// Element-wise `Clip` function that limits values within an interval: f(x, xmin, xmax) = min(max(x, xmin), xmax)
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="min">min</param>
        /// <param name="max">max</param>
        /// <returns>created Layer instance</returns>
        public Layer Clip(string name, object input, float min, float max)
        {
            var layer = Activation(Layer.Activation.Clip, name, input);
            layer.alpha = min;
            layer.beta  = max;

            return layer;
        }

        /// <summary>
        /// Element-wise `Exp` function that calculates exponential of the input: f(x) = e^{x}
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <returns>created Layer instance</returns>
        public Layer Exp(string name, object input)
        {
            return Activation(Layer.Activation.Exp, name, input);
        }

        /// <summary>
        /// Element-wise `Log` function that calculates the natural log of the input: f(x) = log(x)
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <returns>created Layer instance</returns>
        public Layer Log(string name, object input)
        {
            return Activation(Layer.Activation.Log, name, input);
        }

        /// <summary>
        /// Element-wise function that flips the sign of the input: f(x) = -x
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <returns>created Layer instance</returns>
        public Layer Neg(string name, object input)
        {
            return Activation(Layer.Activation.Neg, name, input);
        }

        /// <summary>
        /// Element-wise function that calculates reciprocal of the input: f(x) = 1/x
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <returns>created Layer instance</returns>
        public Layer Reciprocal(string name, object input)
        {
            return Activation(Layer.Activation.Reciprocal, name, input);
        }

        /// <summary>
        /// Element-wise function that calculates absolute values of the input: f(x) = abs(x)
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <returns>created Layer instance</returns>
        public Layer Abs(string name, object input)
        {
            return Activation(Layer.Activation.Abs, name, input);
        }

        /// <summary>
        /// Element-wise function that produces rounding towards the greatest integer less than or equal to the input value: f(x) = ceil(x)
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <returns>created Layer instance</returns>
        public Layer Ceil(string name, object input)
        {
            return Activation(Layer.Activation.Ceil, name, input);
        }

        /// <summary>
        /// Element-wise function that produces rounding towards least integer greater than or equal to the input value: f(x) = floor(x)
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <returns>created Layer instance</returns>
        public Layer Floor(string name, object input)
        {
            return Activation(Layer.Activation.Floor, name, input);
        }

        /// <summary>
        /// Element-wise function that produces rounding of the input value: f(x) = round(x)
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <returns>created Layer instance</returns>
        public Layer Round(string name, object input)
        {
            return Activation(Layer.Activation.Round, name, input);
        }

        /// <summary>
        /// Element-wise `Acos` activation function: f(x) = acos(x)
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <returns>created Layer instance</returns>
        public Layer Acos(string name, object input)
        {
            return Activation(Layer.Activation.Acos, name, input);
        }

        /// <summary>
        /// Element-wise `Acosh` activation function: f(x) = acosh(x)
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <returns>created Layer instance</returns>
        public Layer Acosh(string name, object input)
        {
            return Activation(Layer.Activation.Acosh, name, input);
        }

        /// <summary>
        /// Element-wise `Asin` activation function: f(x) = asin(x)
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <returns>created Layer instance</returns>
        public Layer Asin(string name, object input)
        {
            return Activation(Layer.Activation.Asin, name, input);
        }

        /// <summary>
        /// Element-wise `Asinh` activation function: f(x) = asinh(x)
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <returns>created Layer instance</returns>
        public Layer Asinh(string name, object input)
        {
            return Activation(Layer.Activation.Asinh, name, input);
        }

        /// <summary>
        /// Element-wise `Atan` activation function: f(x) = atan(x)
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <returns>created Layer instance</returns>
        public Layer Atan(string name, object input)
        {
            return Activation(Layer.Activation.Atan, name, input);
        }

        /// <summary>
        /// Element-wise `Atanh` activation function: f(x) = atanh(x)
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <returns>created Layer instance</returns>
        public Layer Atanh(string name, object input)
        {
            return Activation(Layer.Activation.Atanh, name, input);
        }

        /// <summary>
        /// Element-wise `Cos` activation function: f(x) = cos(x)
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <returns>created Layer instance</returns>
        public Layer Cos(string name, object input)
        {
            return Activation(Layer.Activation.Cos, name, input);
        }

        /// <summary>
        /// Element-wise `Cosh` activation function: f(x) = cosh(x)
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <returns>created Layer instance</returns>
        public Layer Cosh(string name, object input)
        {
            return Activation(Layer.Activation.Cosh, name, input);
        }

        /// <summary>
        /// Element-wise `Sin` activation function: f(x) = sin(x)
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <returns>created Layer instance</returns>
        public Layer Sin(string name, object input)
        {
            return Activation(Layer.Activation.Sin, name, input);
        }

        /// <summary>
        /// Element-wise `Sinh` activation function: f(x) = sinh(x)
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <returns>created Layer instance</returns>
        public Layer Sinh(string name, object input)
        {
            return Activation(Layer.Activation.Sinh, name, input);
        }

        /// <summary>
        /// Element-wise `Tan` activation function: f(x) = tan(x)
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <returns>created Layer instance</returns>
        public Layer Tan(string name, object input)
        {
            return Activation(Layer.Activation.Tan, name, input);
        }


        private Layer Broadcast(Layer.Type type, string name, object[] inputs)
        {
            Layer layer = new Layer(name, type);
            layer.inputs = inputs.Select(i => ResolveInput(i)).ToArray();

            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Element-wise `add` of each of the input tensors with multidimensional broadcasting support.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="inputs">input nodes</param>
        /// <returns>created Layer instance</returns>
        public Layer Add(string name, object[] inputs)
        {
            return Broadcast(Layer.Type.Add, name, inputs);
        }

        /// <summary>
        /// Element-wise `sub` of each of the input tensors with multidimensional broadcasting support.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="inputs">input nodes</param>
        /// <returns>created Layer instance</returns>
        public Layer Sub(string name, object[] inputs)
        {
            return Broadcast(Layer.Type.Sub, name, inputs);
        }

        /// <summary>
        /// Element-wise multiplication of each of the input tensors with multidimensional broadcasting support.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="inputs">input nodes</param>
        /// <returns>created Layer instance</returns>
        public Layer Mul(string name, object[] inputs)
        {
            return Broadcast(Layer.Type.Mul, name, inputs);
        }

        /// <summary>
        /// Element-wise division of each of the input tensors with multidimensional broadcasting support.
        /// First element is divided by the 2nd, then result is divided by the third one and so on.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="inputs">input nodes</param>
        /// <returns>created Layer instance</returns>
        public Layer Div(string name, object[] inputs)
        {
            return Broadcast(Layer.Type.Div, name, inputs);
        }

        /// <summary>
        /// Element-wise pow of each of the input tensors with multidimensional broadcasting support.
        /// First element get raised to the pow of the 2nd, then result is raised to the pow of the third one and so on.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="inputs">input nodes</param>
        /// <returns>created Layer instance</returns>
        public Layer Pow(string name, object[] inputs)
        {
            return Broadcast(Layer.Type.Pow, name, inputs);
        }

        /// <summary>
        /// Element-wise `min` of each of the input tensors with multidimensional broadcasting support.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="inputs">input nodes</param>
        /// <returns>created Layer instance</returns>
        public Layer Min(string name, object[] inputs)
        {
            return Broadcast(Layer.Type.Min, name, inputs);
        }

        /// <summary>
        /// Element-wise `max` of each of the input tensors with multidimensional broadcasting support.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="inputs">input nodes</param>
        /// <returns>created Layer instance</returns>
        public Layer Max(string name, object[] inputs)
        {
            return Broadcast(Layer.Type.Max, name, inputs);
        }

        /// <summary>
        /// Element-wise `mean` of each of the input tensors with multidimensional broadcasting support.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="inputs">input nodes</param>
        /// <returns>created Layer instance</returns>
        public Layer Mean(string name, object[] inputs)
        {
            return Broadcast(Layer.Type.Mean, name, inputs);
        }

        /// <summary>
        /// Performs a `greater` logical operation elementwise on the input tensors with multidimensional broadcasting support.
        /// Return 1.0 elementwise if condition is true 0.0 otherwise.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input0">left input node</param>
        /// <param name="input1">right input node</param>
        /// <returns>created Layer instance</returns>
        public Layer Greater(string name, object input0, object input1)
        {
            return Broadcast(Layer.Type.Greater, name, new [] {input0, input1});
        }

        /// <summary>
        /// Performs a `greaterEqual` logical operation elementwise on the input tensors with multidimensional broadcasting support.
        /// Return 1.0 elementwise if condition is true 0.0 otherwise.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input0">left input node</param>
        /// <param name="input1">right input node</param>
        /// <returns>created Layer instance</returns>
        public Layer GreaterEqual(string name, object input0, object input1)
        {
            return Broadcast(Layer.Type.GreaterEqual, name, new [] {input0, input1});
        }

        /// <summary>
        /// Performs a `less` logical operation elementwise on the input tensors with multidimensional broadcasting support.
        /// Return 1.0 elementwise if condition is true 0.0 otherwise.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input0">left input node</param>
        /// <param name="input1">right input node</param>
        /// <returns>created Layer instance</returns>
        public Layer Less(string name, object input0, object input1)
        {
            return Broadcast(Layer.Type.Less, name, new [] {input0, input1});
        }

        /// <summary>
        /// Performs a `less equal` logical operation elementwise on the input tensors with multidimensional broadcasting support.
        /// Return 1.0 elementwise if condition is true 0.0 otherwise.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input0">left input node</param>
        /// <param name="input1">right input node</param>
        /// <returns>created Layer instance</returns>
        public Layer LessEqual(string name, object input0, object input1)
        {
            return Broadcast(Layer.Type.LessEqual, name, new [] {input0, input1});
        }

        /// <summary>
        /// Performs a `equal` logical operation elementwise on the input tensors with multidimensional broadcasting support.
        /// Return 1.0 elementwise if condition is true 0.0 otherwise.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input0">left input node</param>
        /// <param name="input1">right input node</param>
        /// <returns>created Layer instance</returns>
        public Layer Equal(string name, object input0, object input1)
        {
            return Broadcast(Layer.Type.Equal, name, new [] {input0, input1});
        }

        /// <summary>
        /// Performs a `and` logical operation elementwise on the input tensors with multidimensional broadcasting support.
        /// Return 1.0 elementwise if condition is true 0.0 otherwise.
        /// Input is consider false if 0.0 elementwise true otherwise.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input0">left input node</param>
        /// <param name="input1">right input node</param>
        /// <returns>created Layer instance</returns>
        public Layer LogicalAnd(string name, object input0, object input1)
        {
            return Broadcast(Layer.Type.LogicalAnd, name, new [] {input0, input1});
        }

        /// <summary>
        /// Performs a `or` logical operation elementwise on the input tensors with multidimensional broadcasting support.
        /// Return 1.0 elementwise if condition is true 0.0 otherwise.
        /// Input is consider false if 0.0 elementwise true otherwise.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input0">left input node</param>
        /// <param name="input1">right input node</param>
        /// <returns>created Layer instance</returns>
        public Layer LogicalOr(string name, object input0, object input1)
        {
            return Broadcast(Layer.Type.LogicalOr, name, new [] {input0, input1});
        }

        /// <summary>
        /// Performs a `xor` logical operation elementwise on the input tensors with multidimensional broadcasting support.
        /// Return 1.0 elementwise if condition is true 0.0 otherwise.
        /// Input is consider false if 0.0 elementwise true otherwise.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input0">left input node</param>
        /// <param name="input1">right input node</param>
        /// <returns>created Layer instance</returns>
        public Layer LogicalXor(string name, object input0, object input1)
        {
            return Broadcast(Layer.Type.LogicalXor, name, new [] {input0, input1});
        }

        /// <summary>
        /// Performs a `not` logical operation elementwise on the input tensor.
        /// Return 1.0 elementwise if condition is true 0.0 otherwise.
        /// Input is consider false if 0.0 elementwise true otherwise.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <returns>created Layer instance</returns>
        public Layer LogicalNot(string name, object input)
        {
            Layer layer = new Layer(name, Layer.Type.LogicalNot);
            layer.inputs = new[] { ResolveInput(input) };

            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Return elements, either from X or Y, depending on condition (with broadcasting support, based on the shape of the condition)
        /// Return X elementwise if condition is true Y otherwise.
        /// Input is consider false if 0.0 elementwise true otherwise.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="condition">condition</param>
        /// <param name="input1">first input</param>
        /// <param name="input2">second input</param>
        /// <returns>created Layer instance</returns>
        public Layer Where(string name, object condition, object input1, object input2)
        {
            Layer layer = new Layer(name, Layer.Type.Where);
            layer.inputs = new[] { ResolveInput(condition), ResolveInput(input1), ResolveInput(input2) };

            m_Model.layers.Add(layer);

            return layer;
        }

        internal Layer Pad(Layer.Type type, string name, object input, Int32[] pad, float constantValue = 0.0f)
        {
            Layer layer = new Layer(name, type);
            layer.inputs = new[] { ResolveInput(input) };
            layer.beta = constantValue;
            layer.pad = pad;

            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Pads H and W dimension with a given constant value (default to 0).
        /// Pad should be of size 4 and format is [pre W, pre H, post W, post H].
        /// If pad contain negative values H and W dimensions will be cropped instead.
        ///
        /// For example a tensor of shape(1,2,3,1)
        /// [1, 2, 3],
        /// [4, 5, 6]
        ///
        /// With pad [2, 1, 2, 1]
        ///
        /// Result in a tensor of shape(1,4,7,1)
        /// [0, 0, 0, 0, 0, 0, 0],
        /// [0, 0, 1, 2, 3, 0, 0],
        /// [0, 0, 4, 5, 6, 0, 0],
        /// [0, 0, 0, 0, 0, 0, 0]
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="pad">padding</param>
        /// <param name="constantValue">border constant value</param>
        /// <returns>created Layer instance</returns>
        public Layer Border2D(string name, object input, Int32[] pad, float constantValue = 0.0f)
        {
            return Pad(Layer.Type.Border2D, name, input, pad, constantValue);
        }

        /// <summary>
        /// Pads D,H and W dimension with a given constant value (default to 0).
        /// Pad should be of size 6 and format is [pre W, pre H, pre D, post W, post H, post D].
        /// If pad contain negative values H and W dimensions will be cropped instead.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="pad">padding</param>
        /// <param name="constantValue">constant value to use for border</param>
        /// <returns>created Layer instance</returns>
        public Layer Border3D(string name, object input, Int32[] pad, float constantValue = 0.0f)
        {
            return Pad(Layer.Type.Border3D, name, input, pad, constantValue);
        }

        /// <summary>
        /// Pads H and W dimension by repeating the edge values of the input.
        /// Pad should be of size 4 and format is [pre W, pre H, post W, post H].
        ///
        /// For example a tensor of shape(1,2,3,1):
        /// [1, 2, 3],
        /// [4, 5, 6]
        ///
        /// With pad [2, 1, 2, 1]
        ///
        /// Result in a tensor of shape(1,4,7,1)
        /// [1, 1, 1, 2, 3, 3, 3],
        /// [1, 1, 1, 2, 3, 3, 3],
        /// [4, 4, 4, 5, 6, 6, 6],
        /// [4, 4, 4, 5, 6, 6, 6]
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="pad">padding</param>
        /// <returns>created Layer instance</returns>
        public Layer Pad2DEdge(string name, object input, Int32[] pad)
        {
            return Pad(Layer.Type.Pad2DEdge, name, input, pad);
        }

        /// <summary>
        /// Pads H and W dimension by mirroring on the first and last values along those axis.
        /// Pad should be of size 4 and format is [pre W, pre H, post W, post H].
        ///
        /// For example a tensor of shape(1,2,3,1):
        /// [1, 2, 3],
        /// [4, 5, 6]
        ///
        /// With pad [2, 1, 2, 1]
        ///
        /// Result in a tensor of shape(1,4,7,1)
        /// [6, 5, 4, 5, 6, 5, 4],
        /// [3, 2, 1, 2, 3, 2, 1],
        /// [6, 5, 4, 5, 6, 5, 4],
        /// [3, 2, 1, 2, 3, 2, 1]
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="pad">padding</param>
        /// <returns>created Layer instance</returns>
        public Layer Pad2DReflect(string name, object input, Int32[] pad)
        {
            return Pad(Layer.Type.Pad2DReflect, name, input, pad);
        }

        /// <summary>
        /// Pads H and W dimension with symmetric replication along those axis.
        /// Pad should be of size 4 and format is [pre W, pre H, post W, post H].
        ///
        ///  For example a tensor of shape(1,2,3,1):
        ///  [1, 2, 3],
        ///  [4, 5, 6]
        ///
        ///  With pad [2, 1, 2, 1]
        ///
        ///  Result in a tensor of shape(1,4,7,1)
        ///  [2, 1, 1, 2, 3, 3, 2],
        ///  [2, 1, 1, 2, 3, 3, 2],
        ///  [5, 4, 4, 5, 6, 6, 5],
        ///  [5, 4, 4, 5, 6, 6, 5]
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="pad">padding</param>
        /// <returns>created Layer instance</returns>
        public Layer Pad2Symmetric(string name, object input, Int32[] pad)
        {
            return Pad(Layer.Type.Pad2DSymmetric, name, input, pad);
        }

        /// <summary>
        /// Generates a Tensor with random values drawn from a normal distribution.
        /// The shape of the tensor is specified by input tensor
        /// The normal distribution is specified by mean and scale
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="mean">mean</param>
        /// <param name="scale">scale</param>
        /// <param name="seed">seed</param>
        /// <returns>created Layer instance</returns>
        public Layer RandomNormal(string name, object input, float mean, float scale, float seed)
        {
            Assert.IsFalse(input is TensorShape); // TensorShape must be handled by separate RandomNormal(name, shape...) implementation

            Layer layer = new Layer(name, Layer.Type.RandomNormal);
            layer.inputs = new[] { ResolveInput(input) };
            layer.alpha = scale;
            layer.beta = mean;
            layer.pad = new int[1] {(int)seed};
            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Generates a Tensor with random values drawn from a normal distribution.
        /// The shape of the tensor is specified by scale
        /// The normal distribution is specified by mean and scale
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="shape">shape</param>
        /// <param name="mean">mean</param>
        /// <param name="scale">scale</param>
        /// <param name="seed">seed</param>
        /// <returns>created Layer instance</returns>
        public Layer RandomNormal(string name, TensorShape shape, float mean, float scale, float seed)
        {
            Layer layer = new Layer(name, Layer.Type.RandomNormal);
            layer.alpha = scale;
            layer.beta = mean;
            layer.pad = new int[1] {(int)seed};
            layer.pool = shape.ToArray();
            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Generates a Tensor with random values drawn from a uniform distribution.
        /// The shape of the tensor is specified by input tensor
        /// The uniform distribution scale is specified by min and max range
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="min">min</param>
        /// <param name="max">max</param>
        /// <param name="seed">seed</param>
        /// <returns>created Layer instance</returns>
        public Layer RandomUniform(string name, object input, float min, float max, float seed)
        {
            Assert.IsFalse(input is TensorShape); // TensorShape must be handled by separate RandomUniform(name, shape...) implementation

            Layer layer = new Layer(name, Layer.Type.RandomUniform);
            layer.inputs = new[] { ResolveInput(input) };
            layer.alpha = (max-min);
            layer.beta = min;
            layer.pad = new int[1] {(int)seed};
            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Generates a Tensor with random values drawn from a uniform distribution.
        /// The shape of the tensor is specified by shape
        /// The uniform distribution scale is specified by min and max range
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="shape">shape</param>
        /// <param name="min">min</param>
        /// <param name="max">max</param>
        /// <param name="seed">seed</param>
        /// <returns>created Layer instance</returns>
        public Layer RandomUniform(string name, TensorShape shape, float min, float max, float seed)
        {
            Layer layer = new Layer(name, Layer.Type.RandomUniform);
            layer.alpha = (max-min);
            layer.beta = min;
            layer.pad = new int[1] {(int)seed};
            layer.pool = shape.ToArray();
            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Generate a Tensor with random samples drawn from a multinomial distribution according to the probabilities of each of the possible outcomes.
        /// Output batch is same as input.
        /// Output channel is `numberOfSamplesDrawnPerInputChannel`.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="numberOfSamplesDrawnPerInputChannel">number of samples drawn per input channel</param>
        /// <param name="seed">seed</param>
        /// <returns>created Layer instance</returns>
        public Layer Multinomial(string name, object input, int numberOfSamplesDrawnPerInputChannel, float seed)
        {
            Layer layer = new Layer(name, Layer.Type.Multinomial);
            layer.inputs = new[] { ResolveInput(input) };
            layer.pad = new int[1] {(int)seed};
            layer.pool = new int[1] {numberOfSamplesDrawnPerInputChannel};
            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Computes a reduce operation (max/min/mean/prod/sum) of the input tensor's element along the provided axis
        /// If axisIs8D==true axis rank is from [S,R,N,T,D,H,W,C] overwise from [N,H,W,C]
        /// `axis` must be superior to -4
        /// `axis` must be inferior to 8 when axisIs8D==true or inferior to 4 if axisIs8D==false
        /// </summary>
        /// <param name="type">operation type</param>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="axis">axis</param>
        /// <param name="axisIs8D">is axis 8D</param>
        /// <param name="keepDims">is shape rank reduced</param>
        /// <returns>created Layer instance</returns>
        public Layer Reduce(Layer.Type type, string name, object input, int axis = -1, bool axisIs8D=false, int keepDims = 1)
        {
            Layer layer = new Layer(name, type);
            layer.inputs = new[] { ResolveInput(input) };
            layer.axis = axisIs8D?axis:TensorExtensions.Convert4DTo8DAxis(axis);
            layer.alpha = keepDims;
            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Gathers input along provided axis. Swizzling pattern is given by input indices:
        /// If axisIs8D==false
        ///     axis == 0: gatheredData[b, y, x, c] = data[indices[b], y, x, c]
        ///     axis == 1: gatheredData[b, y, x, c] = data[b, indices[y], x, c]
        ///     ...
        /// Else
        ///     axis == 0: gatheredData[s, r, n, t, d, y, x, c] = data[indices[s], r, n, t, d, y, x, c]
        ///     axis == 1: gatheredData[s, r, n, t, d, y, x, c] = data[indices[s], indices[y], n, t, d, y, x, c]
        ///     ...
        /// While in both case
        ///     axis == -1: gatheredData[..., x, c] = data[...x, indices[c]]
        /// `axis` must be superior to -4
        /// `axis` must be inferior to 8 when axisIs8D==true or inferior to 4 if axisIs8D==false
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="indices">indices</param>
        /// <param name="axis">axis</param>
        /// <param name="axisIs8D">is axis 8D</param>
        /// <returns>created Layer instance</returns>
        public Layer Gather(string name, object input, object indices, int axis = -1, bool axisIs8D=false)
        {
            object[] inputs = new[] { input, indices };

            Layer layer = new Layer(name, Layer.Type.Gather);
            layer.inputs = inputs.Select(i => ResolveInput(i)).ToArray();
            layer.axis = axisIs8D?axis:TensorExtensions.Convert4DTo8DAxis(axis);
            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Filter out boxes that have high intersection-over-union (IOU) overlap with previously selected boxes.
        /// Bounding boxes with score less than scoreThreshold are removed.
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="boxes">boxes input node</param>
        /// <param name="scores">scores input node</param>
        /// <param name="maxOutputBoxesPerClass">max output boxes per class input node</param>
        /// <param name="iouThreshold">IOU threshold input node</param>
        /// <param name="scoreThreshold">score input node</param>
        /// <param name="centerPointBox">center point box</param>
        /// <returns>created Layer instance</returns>
        public Layer NonMaxSuppression(string name, object boxes, object scores, object maxOutputBoxesPerClass,
            object iouThreshold, object scoreThreshold, int centerPointBox)
        {
            var layer = new Layer(name, Layer.Type.NonMaxSuppression);

            if (maxOutputBoxesPerClass is float bpc && iouThreshold is float iou && scoreThreshold is float score)
            {
                layer.inputs = new[] { ResolveInput(boxes), ResolveInput(scores) };
                layer.pool = new[] { (int)bpc };
                layer.alpha = iou;
                layer.beta = score;
            }
            else
            {
                layer.inputs = new []
                {
                    ResolveInput(boxes), ResolveInput(scores), ResolveInput(maxOutputBoxesPerClass),
                    ResolveInput(iouThreshold), ResolveInput(scoreThreshold)
                };
            }
            layer.axis = centerPointBox;

            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// LSTM (ML-Agents models only) - requires expansion (see ExpandOpsPass)
        /// </summary>
        /// <param name="name">Layer name</param>
        /// <param name="input">input node</param>
        /// <param name="outputs">output nodes</param>
        /// <param name="W">W data Tensor</param>
        /// <param name="R">R data Tensor</param>
        /// <param name="B">B data Tensor</param>
        /// <param name="hiddenSize">Number of neurons in the hidden layer</param>
        /// <returns>created Layer instance</returns>
        public Layer[] LSTM(string name, object input, string[] outputs, Tensor W, Tensor R, Tensor B, int hiddenSize)
        {
            Layer layer = new Layer(name, Layer.Type.LSTM);
            layer.inputs = new[] { ResolveInput(input) };
            layer.outputs = outputs; // outputs 0-2 are real outputs, output 3 is the resolved input base name, output 4 is the resolved output base name
            layer.pool = new[] { hiddenSize };

            layer.datasets = new Layer.DataSet[3];
            layer.datasets[0].name            = $"{name}/W";
            layer.datasets[0].shape           = W.shape;
            layer.datasets[0].itemSizeInBytes = 4;
            layer.datasets[0].length          = W.shape.length;
            layer.datasets[0].offset          = 0;

            layer.datasets[1].name            = $"{name}/R";
            layer.datasets[1].shape           = R.shape;
            layer.datasets[1].itemSizeInBytes = 4;
            layer.datasets[1].length          = R.shape.length;
            layer.datasets[1].offset          = W.shape.length;

            layer.datasets[2].name            = $"{name}/B";
            layer.datasets[2].shape           = B.shape;
            layer.datasets[2].itemSizeInBytes = 4;
            layer.datasets[2].length          = B.shape.length;
            layer.datasets[2].offset          = W.shape.length + R.shape.length;

            layer.weights                     = new float[W.shape.length + R.shape.length + B.shape.length];

            W.ToReadOnlyArray().CopyTo(layer.weights, 0);
            R.ToReadOnlyArray().CopyTo(layer.weights, layer.datasets[1].offset);
            B.ToReadOnlyArray().CopyTo(layer.weights, layer.datasets[2].offset);

            m_Model.layers.Add(layer);

            Layer layer1 = Identity(outputs[1], layer, rank: 3); // Y_h
            Layer layer2 = Identity(outputs[2], layer, rank: 3); // Y_c

            // LSTM requires expanding
            model.flags |= Model.Flags.NeedsCompilation;

            return new [] { layer, layer1, layer2 };
        }
    }
}
