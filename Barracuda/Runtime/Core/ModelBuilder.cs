using System;
using System.Linq;
using UnityEngine.Assertions;

namespace Unity.Barracuda
{
    public class ModelBuilder
    {
        private readonly Model m_Model;
        public Model model { get { return m_Model; } }

        /// <summary>
        /// Create a model builder helper to construct the underlying Model.
        /// </summary>
        public ModelBuilder(Model model = null)
        {
            if (model == null)
                model = new Model();
            m_Model = model;
        }

        /// <summary>
        /// Add an input to the model
        /// </summary>
        public Model.Input Input(string name, Int32[] shape)
        {
            m_Model.inputs.Add(new Model.Input {name = name, shape = shape});

            return m_Model.inputs.Last();
        }

        /// <summary>
        /// Add an input to the model
        /// </summary>
        public Model.Input Input(string name, TensorShape shape)
        {
            m_Model.inputs.Add(new Model.Input {name = name, shape = shape.ToArray()});

            return m_Model.inputs.Last();
        }

        /// <summary>
        /// Add an input to the model
        /// </summary>
        public Model.Input Input(string name, Int32 batch, Int32 channels)
        {
            m_Model.inputs.Add(new Model.Input {name = name, shape = new []{batch, 1, 1, channels}});

            return m_Model.inputs.Last();
        }

        /// <summary>
        /// Add an input to the model
        /// </summary>
        public Model.Input Input(string name, Int32 batch, Int32 height, Int32 width, Int32 channels)
        {
            m_Model.inputs.Add(new Model.Input {name = name, shape = new []{batch, height, width, channels}});

            return m_Model.inputs.Last();
        }

        /// <summary>
        /// Add an output to the model
        /// </summary>
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
        public Layer Const(string name, Tensor tensor, int insertionIndex = -1)
        {
            Layer layer = new Layer(name, Layer.Type.Load);
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
        /// Carries out instance normalization as described in the paper https://arxiv.org/abs/1607.08022
        /// y = scale * (x - mean) / sqrt(variance + epsilon) + bias, where mean and variance are computed per instance per channel.
        /// Scale and bias should be tensors of shape [1,1,1, input.shape[C]]
        ///
        /// Output shape is same as input.
        /// </summary>
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
        /// Output channel is kernel.shape[3].
        /// output.shape[H,W] = (input.shape[H,W] + pad[1,0] + pad[3,2] - kernel.shape[1,0]) / stride[1,0] + 1.
        /// </summary>
        public Layer Conv2D(string name, object input, Int32[] stride, Int32[] pad, Tensor kernel, Tensor bias)
        {
            return Conv(name, Layer.Type.Conv2D, input, stride, pad, new int[0], kernel, bias);
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
        public Layer MaxPool2D(string name, object input, Int32[] pool, Int32[] stride, Int32[] pad)
        {
            return Pool(Layer.Type.MaxPool2D, name, input, pool, stride, pad);
        }

        /// <summary>
        /// Apply 'average' pooling by downscaling H and W dimension to [1,1]
        /// </summary>
        public Layer GlobalAvgPool2D(string name, object input)
        {
            return Pool(Layer.Type.GlobalAvgPool2D, name, input, new int[0], new int[0], new int[0]);
        }

        /// <summary>
        /// Apply 'max' pooling by downscaling H and W dimension to [1,1]
        /// </summary>
        public Layer GlobalMaxPool2D(string name, object input)
        {
            return Pool(Layer.Type.GlobalMaxPool2D, name, input, new int[0], new int[0], new int[0]);
        }

        /// <summary>
        /// Upsample the input tensor by scaling H and W by upsample[0] and upsample[1] respectively.
        /// `bilinear` allow to choose betwen nearest neighbor or bilinear upsampling.
        /// </summary>
        public Layer Upsample2D(string name, object input, Int32[] upsample, bool bilinear)
        {
            Layer layer = new Layer(name, Layer.Type.Upsample2D);
            layer.pool = upsample;
            layer.axis = bilinear ? 1: -1;
            layer.inputs = new [] {ResolveInput(input)};

            m_Model.layers.Add(layer);

            return layer;
        }

        public Layer Upsample2D(string name, object source, object scale, bool bilinear)
        {
            Layer layer = new Layer(name, Layer.Type.Upsample2D);
            layer.axis = bilinear ? 1: -1;
            layer.inputs = new[] { ResolveInput(source), ResolveInput(scale) };

            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Resample2D scales the input tensor to the given resolution.
        /// `bilinear` allows to choose between nearest neighbour or bilinear sampling.
        /// </summary>
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
        public Layer Reshape(string name, object input, int[] shape)
        {
            Layer layer = new Layer(name, Layer.Type.Reshape);
            layer.pool = shape;
            layer.inputs = new [] {ResolveInput(input)};

            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Apply shape to the input tensor. Number of elements in the shape must match number of elements in input tensor.
        /// </summary>
        public Layer Reshape(string name, object input, TensorShape shape)
        {
            return Reshape(name, input, shape.ToArray());
        }

        /// <summary>
        /// Return a tensor of the shape like another tensor. Both tensors have to have the same number of elements.
        /// </summary>
        public Layer Reshape(string name, object input, object shapeLike)
        {
            Assert.IsFalse(shapeLike is TensorShape); // TensorShape must be handled by separate Reshape(name, input, shape...) implementation

            Layer layer = new Layer(name, Layer.Type.Reshape);
            layer.inputs = new [] {ResolveInput(input), ResolveInput(shapeLike)};

            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Broadcast the input tensor following the given shape and similar to
        /// numpy.array(input) * numpy.ones(shape). Two corresponding dimension
        /// must have the same value, or the input dimension is 1.
        /// </summary>
        public Layer Expand(string name, object input, int[] shape)
        {
            Layer layer = new Layer(name, Layer.Type.Expand);
            layer.inputs = new[] { ResolveInput(input) };
            layer.pool = shape;

            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// From a Tensor of shape [S,R,N,T,D,H,W,C] return a tensor of shape [S,R,N,1,1,1,1,T*D*H*W*C]
        /// </summary>
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
        public Layer Concat(string name, object[] inputs, int axis = -1, bool axisIs8D=false)
        {
            Layer layer = new Layer(name, Layer.Type.Concat);
            layer.axis = axisIs8D?axis:TensorExtensions.NHWCTo8DAxis(axis);
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

        /// <summary>
        /// Make a shallow copy of the input tensor.
        /// </summary>
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
        public Layer TopKValues(string name, object input, object indices, int axis)
        {
            var layer = new Layer(name, Layer.Type.TopKValues);
            layer.inputs = new [] {ResolveInput(input), ResolveInput(indices)};
            layer.axis = axis;

            m_Model.layers.Add(layer);

            return layer;
        }

        /// <summary>
        /// Transpose
        /// </summary>
        public Layer Transpose(string name, object input, int[] permutations)
        {
            Layer layer = new Layer(name, Layer.Type.Transpose);
            layer.inputs = new[] { ResolveInput(input) };
            layer.pool = permutations;

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
        public Layer Identity(string name, object input)
        {
            return Activation(Layer.Activation.None, name, input);
        }


        /// <summary>
        /// Element-wise `Relu` activation function: f(x) = max(0, x)
        /// </summary>
        public Layer Relu(string name, object input)
        {
            return Activation(Layer.Activation.Relu, name, input);
        }

        /// <summary>
        /// Return the softmax (normalized exponential) values of the flatten HWC dimensions of the input.
        /// Thus output will be of shape [input.Batch, input.Height * input.Width * input.Channels]
        /// </summary>
        public Layer Softmax(string name, object input)
        {
            return Activation(Layer.Activation.Softmax, name, input);
        }

        /// <summary>
        /// Return the logsoftmax (normalized exponential) values of the flatten HWC dimensions of the input.
        /// Thus output will be of shape [input.Batch, input.Height * input.Width * input.Channels]
        /// </summary>
        public Layer LogSoftmax(string name, object input)
        {
            return Activation(Layer.Activation.LogSoftmax, name, input);
        }

        /// <summary>
        /// Element-wise `Sqrt` activation function
        /// </summary>
        public Layer Sqrt(string name, object input)
        {
            return Activation(Layer.Activation.Sqrt, name, input);
        }

        /// <summary>
        /// Element-wise `Tanh` activation function: f(x) = (1 - e^{-2x})/(1 + e^{-2x})
        /// </summary>
        public Layer Tanh(string name, object input)
        {
            return Activation(Layer.Activation.Tanh, name, input);
        }

        /// <summary>
        /// Element-wise `Sigmoid` activation function: f(x) = 1/(1 + e^{-x})
        /// </summary>
        public Layer Sigmoid(string name, object input)
        {
            return Activation(Layer.Activation.Sigmoid, name, input);
        }

        /// <summary>
        /// Element-wise `Elu` activation function: f(x) = x if x >= 0 else alpha*(e^x - 1)
        /// alpha default is 1.0
        /// </summary>
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
        public Layer Relu6(string name, object input)
        {
            return Activation(Layer.Activation.Relu6, name, input);
        }

        /// <summary>
        /// Element-wise `LeakyRelu` activation function: f(x) = x if x >= 0 else alpha * x
        /// alpha default is 0.01
        /// </summary>
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
        public Layer Swish(string name, object input)
        {
            return Activation(Layer.Activation.Swish, name, input);
        }

        /// <summary>
        // Element-wise `Clip` function that limits values within an interval: f(x, xmin, xmax) = min(max(x, xmin), xmax)
        /// </summary>
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
        public Layer Exp(string name, object input)
        {
            return Activation(Layer.Activation.Exp, name, input);
        }

        /// <summary>
        /// Element-wise `Log` function that calculates the natural log of the input: f(x) = log(x)
        /// </summary>
        public Layer Log(string name, object input)
        {
            return Activation(Layer.Activation.Log, name, input);
        }

        /// <summary>
        /// Element-wise function that flips the sign of the input: f(x) = -x
        /// </summary>
        public Layer Neg(string name, object input)
        {
            return Activation(Layer.Activation.Neg, name, input);
        }

        /// <summary>
        /// Element-wise function that calculates reciprocal of the input: f(x) = 1/x
        /// </summary>
        public Layer Reciprocal(string name, object input)
        {
            return Activation(Layer.Activation.Reciprocal, name, input);
        }

        /// <summary>
        /// Element-wise function that calculates absolute values of the input: f(x) = abs(x)
        /// </summary>
        public Layer Abs(string name, object input)
        {
            return Activation(Layer.Activation.Abs, name, input);
        }

        /// <summary>
        /// Element-wise function that produces rounding towards the greatest integer less than or equal to the input value: f(x) = ceil(x)
        /// </summary>
        public Layer Ceil(string name, object input)
        {
            return Activation(Layer.Activation.Ceil, name, input);
        }

        /// <summary>
        /// Element-wise function that produces rounding towards least integer greater than or equal to the input value: f(x) = floor(x)
        /// </summary>
        public Layer Floor(string name, object input)
        {
            return Activation(Layer.Activation.Floor, name, input);
        }

        /// <summary>
        /// Element-wise function that produces rounding of the input value: f(x) = round(x)
        /// </summary>
        public Layer Round(string name, object input)
        {
            return Activation(Layer.Activation.Round, name, input);
        }

        /// <summary>
        /// Element-wise `Acos` activation function: f(x) = acos(x)
        /// </summary>
        public Layer Acos(string name, object input)
        {
            return Activation(Layer.Activation.Acos, name, input);
        }

        /// <summary>
        /// Element-wise `Acosh` activation function: f(x) = acosh(x)
        /// </summary>
        public Layer Acosh(string name, object input)
        {
            return Activation(Layer.Activation.Acosh, name, input);
        }

        /// <summary>
        /// Element-wise `Asin` activation function: f(x) = asin(x)
        /// </summary>
        public Layer Asin(string name, object input)
        {
            return Activation(Layer.Activation.Asin, name, input);
        }

        /// <summary>
        /// Element-wise `Asinh` activation function: f(x) = asinh(x)
        /// </summary>
        public Layer Asinh(string name, object input)
        {
            return Activation(Layer.Activation.Asinh, name, input);
        }

        /// <summary>
        /// Element-wise `Atan` activation function: f(x) = atan(x)
        /// </summary>
        public Layer Atan(string name, object input)
        {
            return Activation(Layer.Activation.Atan, name, input);
        }

        /// <summary>
        /// Element-wise `Atanh` activation function: f(x) = atanh(x)
        /// </summary>
        public Layer Atanh(string name, object input)
        {
            return Activation(Layer.Activation.Atanh, name, input);
        }

        /// <summary>
        /// Element-wise `Cos` activation function: f(x) = cos(x)
        /// </summary>
        public Layer Cos(string name, object input)
        {
            return Activation(Layer.Activation.Cos, name, input);
        }

        /// <summary>
        /// Element-wise `Cosh` activation function: f(x) = cosh(x)
        /// </summary>
        public Layer Cosh(string name, object input)
        {
            return Activation(Layer.Activation.Cosh, name, input);
        }

        /// <summary>
        /// Element-wise `Sin` activation function: f(x) = sin(x)
        /// </summary>
        public Layer Sin(string name, object input)
        {
            return Activation(Layer.Activation.Sin, name, input);
        }

        /// <summary>
        /// Element-wise `Sinh` activation function: f(x) = sinh(x)
        /// </summary>
        public Layer Sinh(string name, object input)
        {
            return Activation(Layer.Activation.Sinh, name, input);
        }

        /// <summary>
        /// Element-wise `Tan` activation function: f(x) = tan(x)
        /// </summary>
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
        public Layer Add(string name, object[] inputs)
        {
            return Broadcast(Layer.Type.Add, name, inputs);
        }

        /// <summary>
        /// Element-wise `sub` of each of the input tensors with multidimensional broadcasting support.
        /// </summary>
        public Layer Sub(string name, object[] inputs)
        {
            return Broadcast(Layer.Type.Sub, name, inputs);
        }

        /// <summary>
        /// Element-wise multiplication of each of the input tensors with multidimensional broadcasting support.
        /// </summary>
        public Layer Mul(string name, object[] inputs)
        {
            return Broadcast(Layer.Type.Mul, name, inputs);
        }

        /// <summary>
        /// Element-wise division of each of the input tensors with multidimensional broadcasting support.
        /// First element is divided by the 2nd, then result is divided by the third one and so on.
        /// </summary>
        public Layer Div(string name, object[] inputs)
        {
            return Broadcast(Layer.Type.Div, name, inputs);
        }

        /// <summary>
        /// Element-wise pow of each of the input tensors with multidimensional broadcasting support.
        /// First element get raised to the pow of the 2nd, then result is raised to the pow of the third one and so on.
        /// </summary>
        public Layer Pow(string name, object[] inputs)
        {
            return Broadcast(Layer.Type.Pow, name, inputs);
        }

        /// <summary>
        /// Element-wise `min` of each of the input tensors with multidimensional broadcasting support.
        /// </summary>
        public Layer Min(string name, object[] inputs)
        {
            return Broadcast(Layer.Type.Min, name, inputs);
        }

        /// <summary>
        /// Element-wise `max` of each of the input tensors with multidimensional broadcasting support.
        /// </summary>
        public Layer Max(string name, object[] inputs)
        {
            return Broadcast(Layer.Type.Max, name, inputs);
        }

        /// <summary>
        /// Element-wise `mean` of each of the input tensors with multidimensional broadcasting support.
        /// </summary>
        public Layer Mean(string name, object[] inputs)
        {
            return Broadcast(Layer.Type.Mean, name, inputs);
        }

        /// <summary>
        /// Performs a `greater` logical operation elementwise on the input tensors with multidimensional broadcasting support.
        /// Return 1.0 elementwise if condition is true 0.0 otherwise.
        /// </summary>
        public Layer Greater(string name, object input0, object input1)
        {
            return Broadcast(Layer.Type.Greater, name, new [] {input0, input1});
        }

        /// <summary>
        /// Performs a `greaterEqual` logical operation elementwise on the input tensors with multidimensional broadcasting support.
        /// Return 1.0 elementwise if condition is true 0.0 otherwise.
        /// </summary>
        public Layer GreaterEqual(string name, object input0, object input1)
        {
            return Broadcast(Layer.Type.GreaterEqual, name, new [] {input0, input1});
        }

        /// <summary>
        /// Performs a `less` logical operation elementwise on the input tensors with multidimensional broadcasting support.
        /// Return 1.0 elementwise if condition is true 0.0 otherwise.
        /// </summary>
        public Layer Less(string name, object input0, object input1)
        {
            return Broadcast(Layer.Type.Less, name, new [] {input0, input1});
        }

        /// <summary>
        /// Performs a `less equal` logical operation elementwise on the input tensors with multidimensional broadcasting support.
        /// Return 1.0 elementwise if condition is true 0.0 otherwise.
        /// </summary>
        public Layer LessEqual(string name, object input0, object input1)
        {
            return Broadcast(Layer.Type.LessEqual, name, new [] {input0, input1});
        }

        /// <summary>
        /// Performs a `equal` logical operation elementwise on the input tensors with multidimensional broadcasting support.
        /// Return 1.0 elementwise if condition is true 0.0 otherwise.
        /// </summary>
        public Layer Equal(string name, object input0, object input1)
        {
            return Broadcast(Layer.Type.Equal, name, new [] {input0, input1});
        }

        /// <summary>
        /// Performs a `and` logical operation elementwise on the input tensors with multidimensional broadcasting support.
        /// Return 1.0 elementwise if condition is true 0.0 otherwise.
        /// Input is consider false if 0.0 elementwise true otherwise.
        /// </summary>
        public Layer LogicalAnd(string name, object input0, object input1)
        {
            return Broadcast(Layer.Type.LogicalAnd, name, new [] {input0, input1});
        }

        /// <summary>
        /// Performs a `or` logical operation elementwise on the input tensors with multidimensional broadcasting support.
        /// Return 1.0 elementwise if condition is true 0.0 otherwise.
        /// Input is consider false if 0.0 elementwise true otherwise.
        /// </summary>
        public Layer LogicalOr(string name, object input0, object input1)
        {
            return Broadcast(Layer.Type.LogicalOr, name, new [] {input0, input1});
        }

        /// <summary>
        /// Performs a `xor` logical operation elementwise on the input tensors with multidimensional broadcasting support.
        /// Return 1.0 elementwise if condition is true 0.0 otherwise.
        /// Input is consider false if 0.0 elementwise true otherwise.
        /// </summary>
        public Layer LogicalXor(string name, object input0, object input1)
        {
            return Broadcast(Layer.Type.LogicalXor, name, new [] {input0, input1});
        }

        /// <summary>
        /// Performs a `not` logical operation elementwise on the input tensor.
        /// Return 1.0 elementwise if condition is true 0.0 otherwise.
        /// Input is consider false if 0.0 elementwise true otherwise.
        /// </summary>
        public Layer LogicalNot(string name, object input)
        {
            Layer layer = new Layer(name, Layer.Type.LogicalNot);
            layer.inputs = new[] { ResolveInput(input) };

            m_Model.layers.Add(layer);

            return layer;
        }

        private Layer Pad(Layer.Type type, string name, object input, Int32[] pad, float constantValue = 0.0f)
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
        public Layer Border2D(string name, object input, Int32[] pad, float constantValue = 0.0f)
        {
            return Pad(Layer.Type.Border2D, name, input, pad, constantValue);
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
        public Layer Pad2Symmetric(string name, object input, Int32[] pad)
        {
            return Pad(Layer.Type.Pad2DSymmetric, name, input, pad);
        }

        /// <summary>
        /// Generates a Tensor with random values drawn from a normal distribution.
        /// The shape of the tensor is specified by input tensor
        /// The normal distribution is specified by mean and scale
        /// </summary>
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
        public Layer Multinomial(string name, object input, int numberOfSamplesDrawnPerInputChannel, float seed)
        {
            Layer layer = new Layer(name, Layer.Type.Multinomial);
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
        public Layer Reduce(Layer.Type type, string name, object input, int axis = -1, bool axisIs8D=false)
        {
            Layer layer = new Layer(name, type);
            layer.inputs = new[] { ResolveInput(input) };
            layer.axis = axisIs8D?axis:TensorExtensions.NHWCTo8DAxis(axis);
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
        public Layer Gather(string name, object input, object indices, int axis = -1, bool axisIs8D=false)
        {
            object[] inputs = new[] { input, indices };

            Layer layer = new Layer(name, Layer.Type.Gather);
            layer.inputs = inputs.Select(i => ResolveInput(i)).ToArray();
            layer.axis = axisIs8D?axis:TensorExtensions.NHWCTo8DAxis(axis);
            m_Model.layers.Add(layer);

            return layer;
        }


    }
}
