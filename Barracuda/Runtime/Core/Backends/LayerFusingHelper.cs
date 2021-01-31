using System;
using System.Collections.Generic;
using System.Linq; // ToArray(), ToDictionary()

namespace Unity.Barracuda
{
    internal class LinearLayerFusing
    {
        public static bool IsLayerLinear(Layer layer, Dictionary<string, Layer> constantLayers)
        {
            var constInputs = layer.inputs.Count(x => constantLayers.ContainsKey(x));
            bool allConstInputsButOne = (layer.inputs.Length - constInputs) == 1;

            return layer.type == Layer.Type.Dense ||
                   layer.type == Layer.Type.Conv2D || //TODO Conv3D
                   layer.type == Layer.Type.DepthwiseConv2D ||
                   layer.type == Layer.Type.ScaleBias ||
                   IsLayerLinearMathOp(layer) && allConstInputsButOne;
        }

        public static bool IsLayerLinearMathOp(Layer layer)
        {
            return layer.type == Layer.Type.Add ||
                   layer.type == Layer.Type.Mul;
        }

        public bool AreLayersFusable(Layer l0, Layer l1)
        {
            bool conditions = true;
            if ((l0.type == Layer.Type.DepthwiseConv2D) || (l0.type == Layer.Type.Conv2D) || (l0.type == Layer.Type.ScaleBias) &&
                (l1.type == Layer.Type.Conv2D) || (l1.type == Layer.Type.DepthwiseConv2D))
                conditions = conditions && !l1.pad.Any(x => x != 0); // padding breaks bias merging for non-zero bias
            if (IsLayerLinearMathOp(l0) && (l1.type == Layer.Type.Conv2D))
            {
                if (l0.datasets == null || l0.datasets.Length != 1)
                    return false;
                conditions = conditions && (l0.datasets[0].shape.length == 1) ||
                    (l0.datasets[0].shape.batch == 1 && l0.datasets[0].shape.height == 1 && l0.datasets[0].shape.width == 1 && l0.datasets[0].shape.channels == l1.datasets[0].shape.kernelCount);
            }
            if ((l0.type == Layer.Type.Conv2D) && IsLayerLinearMathOp(l1))
            {
                if (l1.datasets == null || l1.datasets.Length != 1)
                    return false;
                conditions = conditions && (l1.datasets[0].shape.length == 1) ||
                    (l1.datasets[0].shape.batch == 1 && l1.datasets[0].shape.height == 1 && l1.datasets[0].shape.width == 1 && l1.datasets[0].shape.channels == l0.datasets[0].shape.kernelCount);
            }

            return m_LayerFusers.ContainsKey((l0.type, l1.type)) && conditions;
        }

        private readonly BurstCPUOps m_Ops = new BurstCPUOps();

        private readonly Dictionary<(Layer.Type, Layer.Type), Func<Layer, Layer, Layer>> m_LayerFusers =
            new Dictionary<(Layer.Type, Layer.Type), Func<Layer, Layer, Layer>>();

        private void Add((Layer.Type, Layer.Type) layersType, Func<Layer, Layer, Layer> opFuseAction)
        {
            m_LayerFusers.Add(layersType, opFuseAction);
        }
        public LinearLayerFusing()
        {
            Add((Layer.Type.Add, Layer.Type.Add), (l0, l1) =>
            {
                Tensor bias0 = l0.DataSetToTensor(0);
                Tensor bias1 = l1.DataSetToTensor(0);

                TensorShape biasShape = TensorExtensions.MaxShape(new [] { bias0, bias1 });

                Layer lmerged = new Layer(l0.name, l0.type);
                lmerged.inputs = l0.inputs;
                lmerged.datasets = new Layer.DataSet[1];
                lmerged.datasets[0].name = l0.datasets[0].name;
                lmerged.datasets[0].shape = biasShape;
                lmerged.datasets[0].itemSizeInBytes = 4;
                lmerged.datasets[0].length = biasShape.length;
                lmerged.datasets[0].offset = 0;
                lmerged.weights = new float[biasShape.length];
                lmerged.axis = Math.Max(l0.axis, l1.axis);

                Tensor bias = m_Ops.Add(new [] { bias0, bias1 });

                Array.Copy(bias.ToReadOnlyArray(), 0, lmerged.weights, 0, bias.length);

                bias.Dispose();
                bias0.Dispose();
                bias1.Dispose();

                return lmerged;
            });
            Add((Layer.Type.Mul, Layer.Type.Mul), (l0, l1) =>
            {
                Tensor scale0 = l0.DataSetToTensor(0);
                Tensor scale1 = l1.DataSetToTensor(0);

                TensorShape biasShape = TensorExtensions.MaxShape(new[] { scale0, scale1 });

                Layer lmerged = new Layer(l0.name, l0.type);
                lmerged.inputs = l0.inputs;
                lmerged.datasets = new Layer.DataSet[1];
                lmerged.datasets[0].name = l0.datasets[0].name;
                lmerged.datasets[0].shape = biasShape;
                lmerged.datasets[0].itemSizeInBytes = 4;
                lmerged.datasets[0].length = biasShape.length;
                lmerged.datasets[0].offset = 0;
                lmerged.weights = new float[biasShape.length];
                lmerged.axis = Math.Max(l0.axis, l1.axis);

                Tensor bias = m_Ops.Mul(new[] { scale0, scale1 });

                Array.Copy(bias.ToReadOnlyArray(), 0, lmerged.weights, 0, bias.length);

                bias.Dispose();
                scale0.Dispose();
                scale1.Dispose();

                return lmerged;
            });
            Add((Layer.Type.ScaleBias, Layer.Type.ScaleBias), (l0, l1) =>
            {
                Tensor scale0 = l0.DataSetToTensor(0);
                Tensor bias0 = l0.DataSetToTensor(1);

                Tensor scale1 = l1.DataSetToTensor(0);
                Tensor bias1 = l1.DataSetToTensor(1);

                Layer lmerged = new Layer(l0.name, l0.type);
                lmerged.inputs = l0.inputs;
                lmerged.datasets = l0.datasets;
                lmerged.weights = new float[l0.weights.Length];

                // s1*(s0*x + b0)+b1 = s1*s0*x + s1*b0+b1
                Tensor scale = m_Ops.Mul(new [] { scale1, scale0});
                Tensor bias = m_Ops.ScaleBias(bias0, scale1, bias1);

                Array.Copy(scale.ToReadOnlyArray(), 0, lmerged.weights, 0, scale.length);
                Array.Copy(bias.ToReadOnlyArray(), 0, lmerged.weights, scale.length, bias.length);

                scale.Dispose();
                bias.Dispose();
                scale0.Dispose();
                bias0.Dispose();
                scale1.Dispose();
                bias1.Dispose();

                return lmerged;
            });
            Add((Layer.Type.ScaleBias, Layer.Type.Dense), (l0, l1) =>
            {
                Tensor scale0 = l0.DataSetToTensor(0);
                Tensor bias0 = l0.DataSetToTensor(1);

                Tensor weights1 = l1.DataSetToTensor(0);
                Tensor bias1 = l1.DataSetToTensor(1);

                Layer lmerged = new Layer(l0.name, l1.type);
                lmerged.inputs = l0.inputs;
                lmerged.datasets = l1.datasets;
                lmerged.weights = new float[l1.weights.Length];

                // b = W1 x b0 + b1
                Tensor bias = m_Ops.Dense(bias0, weights1, bias1, Layer.FusedActivation.None);

                // W = W1 x s
                Tensor weights = new Tensor(weights1.shape);
                for (int x = 0; x < weights1.flatWidth; ++x)
                    for (int i = 0; i < weights1.flatHeight; ++i)
                    {
                        int c = i % bias0.length;
                        float gamma = scale0[c];

                        float w = weights1[i, x];
                        weights[i, x] = w * gamma;
                    }

                Array.Copy(weights.ToReadOnlyArray(), 0, lmerged.weights, 0, weights.length);
                Array.Copy(bias.ToReadOnlyArray(), 0, lmerged.weights, weights.length, bias.length);

                bias.Dispose();
                weights.Dispose();
                scale0.Dispose();
                bias0.Dispose();
                weights1.Dispose();
                bias1.Dispose();

                return lmerged;
            });
            Add((Layer.Type.Dense, Layer.Type.ScaleBias), (l0, l1) =>
            {
                Tensor weights0 = l0.DataSetToTensor(0);
                Tensor bias0 = l0.DataSetToTensor(1);

                Tensor scale1 = l1.DataSetToTensor(0);
                Tensor bias1 = l1.DataSetToTensor(1);

                Layer lmerged = new Layer(l0.name, l0.type);
                lmerged.inputs = l0.inputs;
                lmerged.datasets = l0.datasets;
                lmerged.weights = new float[l0.weights.Length];

                // w = s1*w0
                Tensor weights = m_Ops.Mul(new [] { scale1, weights0 });
                // b = s1*b0+b1
                Tensor bias = m_Ops.ScaleBias(bias0, scale1, bias1);

                Array.Copy(weights.ToReadOnlyArray(), 0, lmerged.weights, 0, weights.length);
                Array.Copy(bias.ToReadOnlyArray(), 0, lmerged.weights, weights.length, bias.length);

                weights.Dispose();
                bias.Dispose();
                weights0.Dispose();
                bias0.Dispose();
                scale1.Dispose();
                bias1.Dispose();

                return lmerged;
            });
            Add((Layer.Type.Mul, Layer.Type.Conv2D), (l0, l1) =>
            {
                Tensor scale0 = l0.DataSetToTensor(0);

                Tensor kernel1 = l1.DataSetToTensor(0);
                Tensor bias1 = l1.DataSetToTensor(1);

                Layer lmerged = new Layer(l0.name, l1.type);
                lmerged.pad = l1.pad;
                lmerged.stride = l1.stride;
                lmerged.pool = l1.pool;
                lmerged.inputs = l0.inputs;
                lmerged.datasets = l1.datasets;
                lmerged.weights = new float[l1.weights.Length];

                // k = k * s
                Tensor kernel = new Tensor(kernel1.shape);

                for (int y = 0; y < kernel1.kernelHeight; ++y)
                    for (int x = 0; x < kernel1.kernelWidth; ++x)
                        for (int c = 0; c < kernel1.kernelDepth; ++c)
                        {
                            float gamma = scale0[scale0.IndexWithBroadcast(0, 0, 0, c)];
                            for (int k = 0; k < kernel1.kernelCount; ++k)
                            {
                                float w = kernel1[y, x, c, k];
                                kernel[y, x, c, k] = gamma * w;
                            }
                        }


                Array.Copy(kernel.ToReadOnlyArray(), 0, lmerged.weights, 0, kernel.length);
                Array.Copy(bias1.ToReadOnlyArray(), 0, lmerged.weights, kernel.length, bias1.length);

                kernel.Dispose();
                scale0.Dispose();
                kernel1.Dispose();
                bias1.Dispose();

                return lmerged;
            });
            Add((Layer.Type.Conv2D, Layer.Type.Mul), (l0, l1) =>
            {
                Tensor kernel0 = l0.DataSetToTensor(0);
                Tensor bias0 = l0.DataSetToTensor(1);

                Tensor scale1 = l1.DataSetToTensor(0);

                Layer lmerged = new Layer(l0.name, l0.type);
                lmerged.pad = l0.pad;
                lmerged.stride = l0.stride;
                lmerged.pool = l0.pool;
                lmerged.inputs = l0.inputs;
                lmerged.datasets = l0.datasets;
                lmerged.weights = new float[l0.weights.Length];

                // k = s1*k0
                Tensor kernel = m_Ops.Mul(new[] { scale1, kernel0 });
                // b = s1*b0
                Tensor bias = m_Ops.Mul(new[] { scale1, bias0 });

                Array.Copy(kernel.ToReadOnlyArray(), 0, lmerged.weights, 0, kernel.length);
                Array.Copy(bias.ToReadOnlyArray(), 0, lmerged.weights, kernel.length, bias.length);

                kernel.Dispose();
                bias.Dispose();
                kernel0.Dispose();
                bias0.Dispose();
                scale1.Dispose();

                return lmerged;
            });
            Add((Layer.Type.Add, Layer.Type.Conv2D), (l0, l1) =>
            {
                Tensor bias0 = l0.DataSetToTensor(0);

                Tensor kernel1 = l1.DataSetToTensor(0);
                Tensor bias1 = l1.DataSetToTensor(1);

                Layer lmerged = new Layer(l0.name, l1.type);
                lmerged.pad = l1.pad;
                lmerged.stride = l1.stride;
                lmerged.pool = l1.pool;
                lmerged.inputs = l0.inputs;
                lmerged.datasets = l1.datasets;
                lmerged.weights = new float[l1.weights.Length];

                // k = k
                // b = Sum_k[wk * beta] + b
                Tensor bias = new Tensor(bias1.shape, bias1.ToReadOnlyArray());
                for (int y = 0; y < kernel1.kernelHeight; ++y)
                    for (int x = 0; x < kernel1.kernelWidth; ++x)
                        for (int c = 0; c < kernel1.kernelDepth; ++c)
                        {
                            float beta = bias0[bias0.IndexWithBroadcast(0, 0, 0, c)];
                            for (int k = 0; k < kernel1.kernelCount; ++k)
                            {
                                float w = kernel1[y, x, c, k];
                                bias[k] += w * beta;
                            }
                        }


                Array.Copy(kernel1.ToReadOnlyArray(), 0, lmerged.weights, 0, kernel1.length);
                Array.Copy(bias.ToReadOnlyArray(), 0, lmerged.weights, kernel1.length, bias.length);

                bias.Dispose();
                bias0.Dispose();
                kernel1.Dispose();
                bias1.Dispose();

                return lmerged;
            });
            Add((Layer.Type.Conv2D, Layer.Type.Add), (l0, l1) =>
            {
                Tensor kernel0 = l0.DataSetToTensor(0);
                Tensor bias0 = l0.DataSetToTensor(1);

                Tensor bias1 = l1.DataSetToTensor(0);

                Layer lmerged = new Layer(l0.name, l0.type);
                lmerged.pad = l0.pad;
                lmerged.stride = l0.stride;
                lmerged.pool = l0.pool;
                lmerged.inputs = l0.inputs;
                lmerged.datasets = l0.datasets;
                lmerged.weights = new float[l0.weights.Length];

                // b = b0+b1
                Tensor bias = m_Ops.Add( new [] { bias0, bias1 });

                Array.Copy(kernel0.ToReadOnlyArray(), 0, lmerged.weights, 0, kernel0.length);
                Array.Copy(bias.ToReadOnlyArray(), 0, lmerged.weights, kernel0.length, bias.length);

                bias.Dispose();
                kernel0.Dispose();
                bias0.Dispose();
                bias1.Dispose();

                return lmerged;
            });
            Add((Layer.Type.Conv2D, Layer.Type.ScaleBias), (l0, l1) =>
            {
                Tensor kernel0 = l0.DataSetToTensor(0);
                Tensor bias0 = l0.DataSetToTensor(1);

                Tensor scale1 = l1.DataSetToTensor(0);
                Tensor bias1 = l1.DataSetToTensor(1);

                Layer lmerged = new Layer(l0.name, l0.type);
                lmerged.pad = l0.pad;
                lmerged.stride = l0.stride;
                lmerged.pool = l0.pool;
                lmerged.inputs = l0.inputs;
                lmerged.datasets = l0.datasets;
                lmerged.weights = new float[l0.weights.Length];

                // k = s1*k0
                Tensor kernel = m_Ops.Mul(new[] { scale1, kernel0 });
                // b = s1*b0+b1
                Tensor bias = m_Ops.ScaleBias(bias0, scale1, bias1);

                Array.Copy(kernel.ToReadOnlyArray(), 0, lmerged.weights, 0, kernel.length);
                Array.Copy(bias.ToReadOnlyArray(), 0, lmerged.weights, kernel.length, bias.length);

                kernel.Dispose();
                bias.Dispose();
                kernel0.Dispose();
                bias0.Dispose();
                scale1.Dispose();
                bias1.Dispose();

                return lmerged;
            });
            Add((Layer.Type.ScaleBias, Layer.Type.Conv2D), (l0, l1) =>
            {
                Tensor scale0 = l0.DataSetToTensor(0);
                Tensor bias0 = l0.DataSetToTensor(1);

                Tensor kernel1 = l1.DataSetToTensor(0);
                Tensor bias1 = l1.DataSetToTensor(1);

                Layer lmerged = new Layer(l0.name, l1.type);
                lmerged.pad = l1.pad;
                lmerged.stride = l1.stride;
                lmerged.pool = l1.pool;
                lmerged.inputs = l0.inputs;
                lmerged.datasets = l1.datasets;
                lmerged.weights = new float[l1.weights.Length];

                // k = k * s
                Tensor kernel = new Tensor(kernel1.shape);
                // b = Sum_k[wk * beta] + b
                Tensor bias = new Tensor(bias1.shape, bias1.ToReadOnlyArray());
                for (int y = 0; y < kernel1.kernelHeight; ++y)
                    for (int x = 0; x < kernel1.kernelWidth; ++x)
                        for (int c = 0; c < kernel1.kernelDepth; ++c)
                        {
                            float beta = bias0[0, 0, 0, c];
                            float gamma = scale0[0, 0, 0, c];
                            for (int k = 0; k < kernel1.kernelCount; ++k)
                            {
                                float w = kernel1[y, x, c, k];
                                kernel[y, x, c, k] = gamma * w;
                                bias[k] += w * beta;
                            }
                        }

                Array.Copy(kernel.ToReadOnlyArray(), 0, lmerged.weights, 0, kernel.length);
                Array.Copy(bias.ToReadOnlyArray(), 0, lmerged.weights, kernel.length, bias.length);

                kernel.Dispose();
                bias.Dispose();
                scale0.Dispose();
                bias0.Dispose();
                kernel1.Dispose();
                bias1.Dispose();

                return lmerged;
            });
            Add((Layer.Type.DepthwiseConv2D, Layer.Type.ScaleBias), (l0, l1) =>
            {
                Tensor kernel0 = l0.DataSetToTensor(0);
                Tensor bias0 = l0.DataSetToTensor(1);

                Tensor scale1 = l1.DataSetToTensor(0);
                Tensor bias1 = l1.DataSetToTensor(1);

                Layer lmerged = new Layer(l0.name, l0.type);
                lmerged.pad = l0.pad;
                lmerged.stride = l0.stride;
                lmerged.pool = l0.pool;
                lmerged.inputs = l0.inputs;
                lmerged.datasets = l0.datasets;
                lmerged.weights = new float[l0.weights.Length];

                // k = s1*k0
                Tensor kernel = m_Ops.Mul(new[] { scale1, kernel0 });
                // b = s1*b0+b1
                Tensor bias = m_Ops.ScaleBias(bias0, scale1, bias1);

                Array.Copy(kernel.ToReadOnlyArray(), 0, lmerged.weights, 0, kernel.length);
                Array.Copy(bias.ToReadOnlyArray(), 0, lmerged.weights, kernel.length, bias.length);

                kernel.Dispose();
                bias.Dispose();
                kernel0.Dispose();
                bias0.Dispose();
                scale1.Dispose();
                bias1.Dispose();

                return lmerged;
            });
            Add((Layer.Type.ScaleBias, Layer.Type.DepthwiseConv2D), (l0, l1) =>
            {
                Tensor scale0 = l0.DataSetToTensor(0);
                Tensor bias0 = l0.DataSetToTensor(1);

                Tensor kernel1 = l1.DataSetToTensor(0);
                Tensor bias1 = l1.DataSetToTensor(1);

                Layer lmerged = new Layer(l0.name, l1.type);
                lmerged.pad = l1.pad;
                lmerged.stride = l1.stride;
                lmerged.pool = l1.pool;
                lmerged.inputs = l0.inputs;
                lmerged.datasets = l1.datasets;
                lmerged.weights = new float[l1.weights.Length];

                // k = k * s
                Tensor kernel = new Tensor(kernel1.shape);
                // b = Sum_k[wk * beta] + b
                Tensor bias = new Tensor(bias1.shape);
                for (int k = 0; k < kernel1.kernelCount; ++k)
                {
                    float b = bias1[k];

                    float beta = bias0[0, 0, 0, k];
                    float gamma = scale0[0, 0, 0, k];
                    for (int y = 0; y < kernel1.kernelHeight; ++y)
                        for (int x = 0; x < kernel1.kernelWidth; ++x)
                        {
                            float w = kernel1[y, x, 0, k];
                            kernel[y, x, 0, k] = gamma * w;
                            b += w * beta;
                        }

                    bias[k] = b;
                }

                Array.Copy(kernel.ToReadOnlyArray(), 0, lmerged.weights, 0, kernel.length);
                Array.Copy(bias.ToReadOnlyArray(), 0, lmerged.weights, kernel.length, bias.length);

                kernel.Dispose();
                bias.Dispose();
                scale0.Dispose();
                bias0.Dispose();
                kernel1.Dispose();
                bias1.Dispose();

                return lmerged;
            });
            Add((Layer.Type.Dense, Layer.Type.Dense), (l0, l1) =>
            {
                var weights0 = l0.DataSetToTensor(0);
                var bias0 = l0.DataSetToTensor(1);

                var weights1 = l1.DataSetToTensor(0);
                var bias1 = l1.DataSetToTensor(1);

                TensorShape weightsShape = new TensorShape(weights0.shape.flatHeight, weights1.shape.flatWidth);

                Layer lmerged = new Layer(l0.name, l1.type);
                lmerged.inputs = l0.inputs;
                lmerged.datasets = new Layer.DataSet[2];
                lmerged.datasets[0].name = weights0.name;
                lmerged.datasets[0].shape = weightsShape;
                lmerged.datasets[0].itemSizeInBytes = 4;
                lmerged.datasets[0].length = weightsShape.length;
                lmerged.datasets[0].offset = 0;

                lmerged.datasets[1].name = bias0.name;
                lmerged.datasets[1].shape = bias1.shape;
                lmerged.datasets[1].itemSizeInBytes = 4;
                lmerged.datasets[1].length = bias1.length;
                lmerged.datasets[1].offset = 0;
                lmerged.datasets[1].offset = weightsShape.length;
                lmerged.weights = new float[weightsShape.length + bias1.shape.length];

                // W = W1 x W0
                Tensor weights = m_Ops.MatMul(weights0, false, weights1, false);
                // b = W1 x b0 + b1
                Tensor bias = m_Ops.Dense(bias0, weights1, bias1, Layer.FusedActivation.None);

                Array.Copy(weights.ToReadOnlyArray(), 0, lmerged.weights, 0, weights.length);
                Array.Copy(bias.ToReadOnlyArray(), 0, lmerged.weights, weights.length, bias.length);

                weights.Dispose();
                bias.Dispose();
                weights0.Dispose();
                bias0.Dispose();
                weights1.Dispose();
                bias1.Dispose();

                return lmerged;
            });
            Add((Layer.Type.Conv2D, Layer.Type.Conv2D), (l0, l1) =>
            {
                Tensor kernel0 = l0.DataSetToTensor(0);
                Tensor bias0 = l0.DataSetToTensor(1);
                var strides0 = l0.stride;
                var pad0 = l0.pad;

                Tensor kernel1 = l1.DataSetToTensor(0);
                Tensor bias1 = l1.DataSetToTensor(1);
                var strides1 = l1.stride;
                var pad1 = l1.pad;


                // Y = (X * K0 + b0) * K1 +  b1
                //   = (X * K0) * K1 + (b0 * K1 + b1)
                //   = X * (K0 * k1) + (b0 * K1 + b1)
                //   = X * K2 + b2
                // K2 dimensions:
                // kernelDepth and kernelCount:
                // X = [n, . , . , c0], K0 = [ . , . , c0, d0] , K1 = [ . , . , c1, d1]
                //                   => Km = [ x , x , c0, d1]
                // kernelHeight and kernelHeight:
                // Y = (((X + 2*p0 - k0)/s0 + 1) + 2*p1 - k1)/s1 + 1
                //   = ((X + 2*p0 - k0 + s0 + 2*p1*s0 - k1*s0)/s0)/s1 + 1
                //   = (X + 2*p0 - k0 + s0 + 2*p1*s0 - k1*s0) / (s0*s1) + 1
                //   = (X + 2*(p0+p1*s0) - (k0 + k1*s0 - s0)) / (s0*s1) + 1
                // => pad = p0 + p1*s0
                //    kernel = k0 + s0*(k1 - 1)
                //    stride = s0*s1
                TensorShape kernelShape = new TensorShape(kernel0.kernelHeight + (kernel1.kernelHeight - 1) * strides0[0],
                                                          kernel0.kernelWidth  + (kernel1.kernelWidth  - 1) * strides0[1],
                                                          kernel0.kernelDepth, kernel1.kernelCount);

                var pad = new int[4] { pad0[0] + pad1[0] * strides0[0], pad0[1] + pad1[1] * strides0[1],
                                       pad0[2] + pad1[2] * strides0[0], pad0[3] + pad1[3] * strides0[1] };
                var strides = new int[2] { strides0[0] * strides1[0], strides0[1] * strides1[1] };

                TensorShape biasShape = bias1.shape;


                Layer lmerged = new Layer(l0.name, l1.type);
                lmerged.inputs = l0.inputs;
                lmerged.stride = strides;
                lmerged.pad = pad;
                lmerged.datasets = new Layer.DataSet[2];
                lmerged.datasets[0].name = kernel0.name;
                lmerged.datasets[0].shape = kernelShape;
                lmerged.datasets[0].itemSizeInBytes = 4;
                lmerged.datasets[0].length = kernelShape.length;
                lmerged.datasets[0].offset = 0;

                lmerged.datasets[1].name = bias0.name;
                lmerged.datasets[1].shape = biasShape;
                lmerged.datasets[1].itemSizeInBytes = 4;
                lmerged.datasets[1].length = biasShape.length;
                lmerged.datasets[1].offset = 0;
                lmerged.datasets[1].offset = kernelShape.length;
                lmerged.weights = new float[kernelShape.length + biasShape.length];


                Tensor kernel = new Tensor(kernelShape); // 0-filled by default
                // |x0  x1  x3 | x4                |y0 y1| y2              |z0| z1
                // |x5  x6  x7 | x8   * k0 k1  =>  |y3 y4| y5 * l0 l1 =>    z2  z3
                // |x9  x10 x11| x12    k2 k3       y6 y7  y8   l2 l3
                //  x13 x14 x15  x13
                //
                // in order to compute z0, we need to do 2 convolutions
                //
                //    |y0        y1/
                //  | |x0  /x1|  x3/  |
                //  | |x5  /x6|  x7/  |
                //  |  x9   x10   x11 |
                //
                //  |x0  x1| is convolved with K and then * l0
                //  |x5  x6|
                //  /x1  x3/ is convolved with K and then * l1
                //  /x6  x7/
                //
                // by unwrapping the whole process
                // z0 = [x0 * k0 * l0 + x1 * k1 * l0 + ....] + [x1 * k1 * l1 + ....]
                //        l0 * y0-block                           l1 * y1-block
                // resulting conv kernel is the following
                //
                // z0 = | x0 x1  x3  | * | [k0*l0]         [k1*l0 + k1*l1]                  [l2*l1] |
                //      | x5 x6  x7  |   | [k2*l0 + k2*l2] [k3*l0 + k2*l1 + k1*l2 + k0*l3]  [k3*l1 + k3*l3] |
                //      | x9 x10 x11 |   | [k2*l2]         [k2*l0 + k2*l3                   [k3*l3] |
                Tensor kernel0T = m_Ops.Transpose(kernel0, new[] { 2, 0, 1, 3 });
                Tensor emptyB = new Tensor(new TensorShape(1, 1, 1, kernel.kernelCount));
                for (int y1 = 0; y1 < kernel1.kernelHeight; ++y1)
                    for (int x1 = 0; x1 < kernel1.kernelWidth; ++x1)
                    {
                        Tensor kernel1XY = m_Ops.StridedSlice(kernel1, new[] { y1, x1, 0, 0 }, new[] { y1 + 1, x1 + 1, kernel1.kernelDepth, kernel.kernelCount }, new[] { 1, 1, 1, 1 });
                        Tensor kernelk = m_Ops.Conv2D(kernel0T, kernel1XY, emptyB, new[] { 1, 1 }, new[] { 0, 0, 0, 0 }, Layer.FusedActivation.None);

                        for (int y0 = 0; y0 < kernel0.kernelHeight; ++y0)
                            for (int x0 = 0; x0 < kernel0.kernelWidth; ++x0)
                            {
                                int ox = x0 + strides0[0] * x1;
                                int oy = y0 + strides0[1] * y1;
                                for (int c = 0; c < kernel.kernelDepth; ++c)
                                    for (int k = 0; k < kernel.kernelCount; ++k)
                                    {
                                        kernel[oy, ox, c, k] += kernelk[c,y0,x0,k];
                                    }
                            }
                        kernel1XY.Dispose();
                        kernelk.Dispose();
                    }

                // |y0 y1| * l0 l1  + bl = z0
                // |y3 y4|   l2 l3
                // y0 = Sum_k() + bk, y1 = Sum_k() + bk
                // y2 = Sum_k() + bk, y2 = Sum_k() + bk
                //
                // moving b from the convolution process leads
                // z0 = | x0 x1  x3  | * M + bl + l0*bk + l1*bk + l2*bk + l3*bk
                //      | x5 x6  x7  |
                //      | x9 x10 x11 |
                // N.B: as you can see this breaks if there is some amount of zero-padding to the second conv layer
                // because some weights of L will be * 0, essentialy masking out bk
                Tensor bias = new Tensor(biasShape, bias1.ToReadOnlyArray());
                for (int x1 = 0; x1 < kernel1.kernelWidth; ++x1)
                    for (int y1 = 0; y1 < kernel1.kernelHeight; ++y1)
                        for (int c = 0; c < kernel1.kernelDepth; ++c)
                        {
                            float bias0c = bias0[c];
                            for (var k = 0; k < kernel.kernelCount; ++k)
                            {
                                bias[k] += kernel1[y1, x1, c, k] * bias0c;
                            }
                        }

                Array.Copy(kernel.ToReadOnlyArray(), 0, lmerged.weights, 0, kernel.length);
                Array.Copy(bias.ToReadOnlyArray(), 0, lmerged.weights, kernel.length, bias.length);

                kernel0T.Dispose();
                emptyB.Dispose();
                kernel.Dispose();
                bias.Dispose();
                kernel0.Dispose();
                bias0.Dispose();
                kernel1.Dispose();
                bias1.Dispose();

                return lmerged;
            });
        }

        public Layer FuseLayers(Layer l0, Layer l1)
        {
            var fnFuse = m_LayerFusers[(l0.type, l1.type)];
            return fnFuse(l0, l1);
        }
    }

} // namespace Unity.Barracuda
