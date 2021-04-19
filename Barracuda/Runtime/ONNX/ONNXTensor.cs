using Onnx;
using UnityEngine;
using UnityEngine.Profiling;
using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using UnityEngine.Android;
using UnityEngine.Assertions;

[assembly: InternalsVisibleToAttribute("Barracuda.EditorTests")]

namespace Unity.Barracuda.ONNX
{
    // Combines information about ONNX tensor and data read from TensorProto
    internal struct ONNXTensor
    {
        public int[] shape => m_Shape;
        public int rank => shape.Length;
        public Tensor data => m_Data;

        Tensor m_Data;
        int[] m_Shape;

        public ONNXTensor(TensorProto onnxTensor)
        {
            // read shape
            var onnxShape = onnxTensor.Dims.Select(v => v < int.MinValue ? int.MinValue : v > int.MaxValue ? int.MaxValue : (int)v).ToArray();

            if (onnxShape.Any(s => s == 0))
            {
                // empty tensor, not data
                m_Shape = onnxShape;
                m_Data = null;
            }
            else
            {
                // read data
                var shape = ONNXLayout.ConvertShapeToBarracuda(onnxShape, onnxLayout:"?");
                float[] data;
                if ((onnxTensor.RawData != null) && (!onnxTensor.RawData.IsEmpty))
                {
                    var byteArray = new byte[onnxTensor.RawData.Length];
                    onnxTensor.RawData.CopyTo(byteArray, 0);

                    // Double
                    if (onnxTensor.DataType == (int)TensorProto.Types.DataType.Double)
                    {
                        var typedData = new double[shape.length];
                        Assert.IsTrue((sizeof(double) * shape.length) == onnxTensor.RawData.Length);
                        Buffer.BlockCopy(byteArray, 0, typedData, 0, byteArray.Length);
                        data = typedData.Select(v => v < int.MinValue ? (float)int.MinValue : v > int.MaxValue ? (float)int.MaxValue : (float)v).ToArray();
                    }
                    // Float32
                    else if (onnxTensor.DataType == (int)TensorProto.Types.DataType.Float)
                    {
                        data = new float[shape.length];
                        Assert.IsTrue((sizeof(float) * shape.length) == onnxTensor.RawData.Length);
                        Buffer.BlockCopy(byteArray, 0, data, 0, byteArray.Length);
                    }
                    // Float16
                    else if (onnxTensor.DataType == (int)TensorProto.Types.DataType.Float16)
                    {
                        var typedData = new UInt16[shape.length];
                        Assert.IsTrue((sizeof(UInt16) * shape.length) == onnxTensor.RawData.Length);
                        Buffer.BlockCopy(byteArray, 0, typedData, 0, byteArray.Length);
                        data = typedData.Select(v => HalfHelper.HalfToSingle(v)).ToArray();
                    }
                    // Int8
                    else if (onnxTensor.DataType == (int)TensorProto.Types.DataType.Int8)
                    {
                        var typedData = new sbyte[shape.length];
                        Assert.IsTrue((sizeof(sbyte) * shape.length) == onnxTensor.RawData.Length);
                        Buffer.BlockCopy(byteArray, 0, typedData, 0, byteArray.Length);
                        data = typedData.Select(v => (float)v).ToArray();
                    }
                    // Int16
                    else if (onnxTensor.DataType == (int)TensorProto.Types.DataType.Int16)
                    {
                        var typedData = new short[shape.length];
                        Assert.IsTrue((sizeof(short) * shape.length) == onnxTensor.RawData.Length);
                        Buffer.BlockCopy(byteArray, 0, typedData, 0, byteArray.Length);
                        data = typedData.Select(v => (float)v).ToArray();
                    }
                    // Int32
                    else if (onnxTensor.DataType == (int)TensorProto.Types.DataType.Int32)
                    {
                        var typedData = new int[shape.length];
                        Assert.IsTrue((sizeof(int) * shape.length) == onnxTensor.RawData.Length);
                        Buffer.BlockCopy(byteArray, 0, typedData, 0, byteArray.Length);
                        data = typedData.Select(v => (float)v).ToArray();
                    }
                    // Int64
                    else if (onnxTensor.DataType == (int)TensorProto.Types.DataType.Int64)
                    {
                        var typedData = new long[shape.length];
                        Assert.IsTrue((sizeof(long) * shape.length) == onnxTensor.RawData.Length);
                        Buffer.BlockCopy(byteArray, 0, typedData, 0, byteArray.Length);
                        data = typedData.Select(v => v < (long)int.MinValue ? (float)int.MinValue : v > (long)int.MaxValue ? (float)int.MaxValue : (float)v).ToArray();
                    }
                    // UInt8
                    else if (onnxTensor.DataType == (int)TensorProto.Types.DataType.Uint8)
                    {
                        var typedData = new byte[shape.length];
                        Assert.IsTrue((sizeof(byte) * shape.length) == onnxTensor.RawData.Length);
                        Buffer.BlockCopy(byteArray, 0, typedData, 0, byteArray.Length);
                        data = typedData.Select(v => (float)v).ToArray();
                    }
                    // UInt16
                    else if (onnxTensor.DataType == (int)TensorProto.Types.DataType.Uint16)
                    {
                        var typedData = new ushort[shape.length];
                        Assert.IsTrue((sizeof(ushort) * shape.length) == onnxTensor.RawData.Length);
                        Buffer.BlockCopy(byteArray, 0, typedData, 0, byteArray.Length);
                        data = typedData.Select(v => (float)v).ToArray();
                    }
                    // UInt32
                    else if (onnxTensor.DataType == (int)TensorProto.Types.DataType.Uint32)
                    {
                        var typedData = new uint[shape.length];
                        Assert.IsTrue((sizeof(uint) * shape.length) == onnxTensor.RawData.Length);
                        Buffer.BlockCopy(byteArray, 0, typedData, 0, byteArray.Length);
                        data = typedData.Select(v => (float)v).ToArray();
                    }
                    // UInt64
                    else if (onnxTensor.DataType == (int)TensorProto.Types.DataType.Uint64)
                    {
                        var typedData = new ulong[shape.length];
                        Assert.IsTrue((sizeof(ulong) * shape.length) == onnxTensor.RawData.Length);
                        Buffer.BlockCopy(byteArray, 0, typedData, 0, byteArray.Length);
                        data = typedData.Select(v => v > uint.MaxValue ? (float)uint.MaxValue : (float)v).ToArray();
                    }
                    // Bool
                    else if (onnxTensor.DataType == (int)TensorProto.Types.DataType.Bool)
                    {
                        var typedData = new bool[shape.length];
                        Assert.IsTrue((sizeof(bool) * shape.length) == onnxTensor.RawData.Length);
                        Buffer.BlockCopy(byteArray, 0, typedData, 0, byteArray.Length);
                        data = typedData.Select(v => v ? 1.0f : 0.0f).ToArray();
                    }
                    else
                        throw new OnnxLayerImportException($"Tensor data type {(TensorProto.Types.DataType)onnxTensor.DataType} is not supported.");
                }
                // Float32
                else if ((onnxTensor.FloatData != null) && (onnxTensor.FloatData.Count != 0))
                {
                    Assert.IsTrue(shape.length == onnxTensor.FloatData.Count);
                    data = new float[shape.length];
                    onnxTensor.FloatData.CopyTo(data, 0);
                }
                // Int32
                else if ((onnxTensor.Int32Data != null) && (onnxTensor.Int32Data.Count != 0))
                {
                    Assert.IsTrue(shape.length == onnxTensor.Int32Data.Count);
                    data = onnxTensor.Int32Data.Select(v => (float)v).ToArray();
                }
                // Int64
                else if ((onnxTensor.Int64Data != null) && (onnxTensor.Int64Data.Count != 0))
                {
                    Assert.IsTrue(shape.length == onnxTensor.Int64Data.Count);
                    data = onnxTensor.Int64Data.Select(v => v < int.MinValue ? (float)int.MinValue : v > int.MaxValue ? (float)int.MaxValue : (float)v).ToArray();
                }
                else
                {
                    throw new OnnxLayerImportException("Could not read tensor data for constant tensor.");
                }

                m_Data = new Tensor(shape, new SharedArrayTensorData(data));
                m_Shape = onnxShape;
            }
        }

        public ONNXTensor(Tensor data, int[] onnxShape)
        {
            m_Data = data;
            m_Shape = onnxShape;
        }

        public bool IsEmpty()
        {
            return m_Shape.Any(s => s == 0);
        }

        public ONNXTensor Reshape(int[] onnxShape)
        {
            var symbolicShape = ONNXLayout.ConvertSymbolicShapeToBarracuda(onnxShape, "?");
            var reshapedData = m_Data.Reshape(symbolicShape);
            for (var i = 0; i < onnxShape.Length; ++i)
            {
                if (onnxShape[i] < 0)
                    onnxShape[i] = reshapedData.shape[i];
                Assert.IsTrue(onnxShape[i] == reshapedData.shape[i]);
            }
            return new ONNXTensor(reshapedData, onnxShape);
        }

        public ONNXTensor Permute(int[] permutations)
        {
            // transpose both data & shape
            var transposedData = Permute(m_Data, permutations);
            var transposedShape = ONNXLayout.Permute(m_Shape, permutations);
            return new ONNXTensor(transposedData, transposedShape);
        }

        public ONNXTensor NonZero()
        {
            //https://github.com/onnx/onnx/blob/master/docs/Operators.md#NonZero
            //https://numpy.org/doc/stable/reference/generated/numpy.nonzero.html
            //Return the indices of the elements that are non-zero. Iterating row major c style.

            // pad with 1s to visit all elements at least once in the loop.
            int[] paddedONNXShape = new int[] {1, 1, 1, 1, 1, 1, 1, 1};
            for (int d = 0; d < rank; ++d)
                paddedONNXShape[d] = shape[d];

            // collect all non zero item
            List<int[]> nonZeroIndices = new List<int[]>();
            for (var it = new TensorIterator(m_Data.shape); it.IsValid(); it.Next())
            {
                if (Math.Abs(m_Data[it.index]) > Single.Epsilon)
                    nonZeroIndices.Add(new int[] {it.d0,it.d1,it.d2,it.d3,it.d4,it.d5,it.d6,it.d7});
            }

            // store indices in dest tensor
            Tensor result = new Tensor(new TensorShape(rank, nonZeroIndices.Count));
            for(int i = 0; i < nonZeroIndices.Count; ++i)
            {
                for (int d = 0; d < rank; ++d)
                    result[d,i] = nonZeroIndices[i][d];
            }

            return new ONNXTensor(result, new int[] {rank, nonZeroIndices.Count});
        }

        public ONNXTensor SqueezeAll()
        {
            var newShape = m_Shape.Where(x => x > 1).ToArray();
            if (newShape.Length == 0)
                newShape = new[] { 1 };
            return Reshape(newShape);
        }

        public ONNXTensor Squeeze(int[] axes)
        {
            var newShape = m_Shape.ToList();
            foreach (var axis in axes)
            {
                // axis in [-rank,rank-1]
                var axisInRange = axis >= 0 ? axis : 4 + axis;
                if (newShape[axisInRange] == 1)
                    newShape[axisInRange] = -1;
            }
            newShape.RemoveAll(x => x == -1);
            for (int i = newShape.Count; i < 4; i++)
                newShape.Add(1);

            return Reshape(newShape.ToArray());
        }

        public ONNXTensor Unsqueeze(int[] axes)
        {
            var newShape = m_Shape.ToList();
            foreach (var axis in axes)
            {
                // axis in [-rank,rank-1]
                var axisInRange = axis >= 0 ? axis : 4 + axis;
                newShape.Insert(axis, 1);
            }
            return Reshape(newShape.ToArray());
        }

        public ONNXTensor Slice(int[] starts, int[] ends, int[] steps)
        {
            Assert.IsTrue(starts.Length == ends.Length);
            Assert.IsTrue(starts.Length == steps.Length);

            var newShape = new int[starts.Length];
            // handle negative indices, negative steps
            for (var i = 0; i < m_Shape.Length; ++i)
            {
                if (starts[i] < 0)
                    starts[i] = (int)m_Shape[i] + starts[i];
                if (ends[i] < 0)
                    ends[i] = (int)m_Shape[i] + ends[i];
                if (steps[i] == 0)
                {
                    starts[i] = 0;
                    ends[i] = 1;
                    steps[i] = 1;
                }
                ends[i] = Math.Min((int)m_Shape[i], ends[i]);
            }

            // calculate shape for sliced tensor
            for (var i = 0; i < m_Shape.Length; ++i)
                newShape[i] = (ends[i] - starts[i]) / steps[i];

            int[] newONNXShapePadded = new int[] {1, 1, 1, 1, 1, 1, 1, 1};
            for (int d = 0; d < newShape.Length; ++d)
                newONNXShapePadded[d] = newShape[d];
            Tensor result = new Tensor(newONNXShapePadded);

            // pad to the number of the loops - 4
            starts = starts.Concat(Enumerable.Repeat(0, 4 - starts.Length)).ToArray();
            ends   = ends.Concat  (Enumerable.Repeat(1, 4 - ends.Length)).ToArray(); // we need to keep 1, to visit all elements at least once
            steps  = steps.Concat (Enumerable.Repeat(1, 4 - steps.Length)).ToArray();

            for (int b = starts[0], bo = 0; b < ends[0]; b += steps[0], bo++)
                for (int y = starts[1], yo = 0; y < ends[1]; y += steps[1], yo++)
                    for (int x = starts[2], xo = 0; x < ends[2]; x += steps[2], xo++)
                        for (int c = starts[3], co = 0; c < ends[3]; c += steps[3], co++)
                            result[bo, yo, xo, co] = m_Data[b, y, x, c];

            return new ONNXTensor(result, newShape.ToArray());
        }

        public ONNXTensor Gather(int axis, int[] indices)
        {
            //Atm support up to 4D tensors.
            Assert.IsTrue(indices.Length < 5);

            // good explanation can be found here:
            // https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms
            int[] newONNXShape = m_Shape.Select(i => (int)i).ToArray();
            newONNXShape[axis] = indices.Length;

            // pad with 1s to visit all elements at least once in the loop.
            int[] newONNXShapePadded = new int[] {1, 1, 1, 1, 1, 1, 1, 1};
            for (int d = 0; d < newONNXShape.Length; ++d)
                newONNXShapePadded[d] = newONNXShape[d];

            Tensor result = new Tensor(newONNXShapePadded);

            for (int b = 0; b < newONNXShapePadded[0]; ++b)
                for (int y = 0; y < newONNXShapePadded[1]; ++y)
                    for (int x = 0; x < newONNXShapePadded[2]; ++x)
                        for (int c = 0; c < newONNXShapePadded[3]; ++c)
                        {
                            if (axis == 0)
                                result[b, y, x, c, 0, 0, 0, 0] = m_Data[indices[b], y, x, c, 0, 0, 0, 0];
                            else if (axis == 1)
                                result[b, y, x, c, 0, 0, 0, 0] = m_Data[b, indices[y], x, c, 0, 0, 0, 0];
                            else if (axis == 2)
                                result[b, y, x, c, 0, 0, 0, 0] = m_Data[b, y, indices[x], c, 0, 0, 0, 0];
                            else
                                result[b, y, x, c, 0, 0, 0, 0] = m_Data[b, y, x, indices[c], 0, 0, 0, 0];
                        }

            return new ONNXTensor(result, newONNXShape.ToArray());
        }

        public float this[int index]
        {
            get { return m_Data[index]; }
        }

        public Tensor ToBarracuda(string onnxLayout)
        {
            Profiler.BeginSample("ONNXTensor.ToBarracuda");
            if (onnxLayout == "?")
                throw new OnnxLayerImportException("Unknown ONNX layout in not supported when converting constant tensor to Barracuda");

            Assert.IsTrue(m_Shape.All(v => v > 0));
            var permutations = ONNXLayout.AxisPermutationsForMappingONNXLayoutToBarracuda(rank, onnxLayout);
            Assert.IsTrue(rank <= permutations.Length);

            var outTensor = Permute(m_Data, permutations);
            Profiler.EndSample();
            return outTensor;
        }

        internal static ONNXTensor Range(float start, float limit, float delta)
        {
            int nbElements = Mathf.Max((int)Mathf.Ceil((limit - start) / delta), 0);
            Tensor output = new Tensor(nbElements, 1);

            for (int i = 0; i < nbElements; ++i)
            {
                output[i] = start + (i * delta);
            }
            return new ONNXTensor(output, new[] { nbElements });
        }

        internal static Tensor Permute(Tensor inTensor, int[] permutations) // TODO: unify Permute() arguments
        {
            var padPermutationsToBarracudaRank = TensorShape.MaxRank - permutations.Length;
            if (padPermutationsToBarracudaRank > 0)
                permutations = permutations.Concat(Enumerable.Range(permutations.Length, padPermutationsToBarracudaRank)).ToArray();
            Assert.IsTrue(permutations.Length == TensorShape.MaxRank);

            // See: https://stackoverflow.com/a/32034565
            Profiler.BeginSample("ONNXTensor.Permute");
            var outTensor = new Tensor(ONNXLayout.Permute(inTensor.shape.ToArray(), permutations));
            Assert.IsTrue(outTensor.length == inTensor.length);

            // {0, 2, 3, 1} => {0, 3, 1, 2}
            // {2, 3, 1, 0} => {3, 2, 0, 1}
            //              => {find_index(0), find_index(1), find_index(2), find_index(3)}
            var reversePermute = new int[permutations.Length];
            for (var i = 0; i < permutations.Length; ++i)
                reversePermute[i] = Array.IndexOf(permutations, i);

            // outTensor strides
            var tempOutStrides = new int[TensorShape.MaxRank+1];
            tempOutStrides[8] = 1;
            for (int i = 7; i >= 0; --i)
                tempOutStrides[i] = tempOutStrides[i+1] * outTensor.shape[i];

            var outStride = new int[reversePermute.Length];
            for (var i = 0; i < reversePermute.Length; ++i)
                outStride[i] = tempOutStrides[reversePermute[i] + 1];

            for (var it = new TensorIterator(inTensor.shape); it.IsValid(); it.Next())
            {
                float value = inTensor[it.index];

                outTensor[it.d0 * outStride[0] +
                          it.d1 * outStride[1] +
                          it.d2 * outStride[2] +
                          it.d3 * outStride[3] +
                          it.d4 * outStride[4] +
                          it.d5 * outStride[5] +
                          it.d6 * outStride[6] +
                          it.d7 * outStride[7]] = value;
            }

            Profiler.EndSample();
            return outTensor;
        }

        // slow version - kept just for performance comparison and validation
        internal static Tensor PermuteSlow(Tensor readTensor, int[] permutations) // TODO: unify Permute() arguments
        {
            var padPermutationsToBarracudaRank = 8 - permutations.Length;
            if (padPermutationsToBarracudaRank > 0)
                permutations = permutations.Concat(Enumerable.Range(permutations.Length, padPermutationsToBarracudaRank)).ToArray();
            Assert.IsTrue(permutations.Length == 8);

            var outputTensor = new Tensor(ONNXLayout.Permute(readTensor.shape.ToArray(), permutations));
            Assert.IsTrue(outputTensor.length == readTensor.length);

            var inShape = readTensor.shape.ToArray();
            for (var s = 0; s < inShape[0]; ++s)
                for (var n = 0; n < inShape[1]; ++n)
                    for (var i0 = 0; i0 < inShape[2]; ++i0)
                        for (var i1 = 0; i1 < inShape[3]; ++i1)
                            for (var i2 = 0; i2 < inShape[4]; ++i2)
                                for (var h = 0; h < inShape[5]; ++h)
                                    for (var w = 0; w < inShape[6]; ++w)
                                        for (var c = 0; c < inShape[7]; ++c)
                                        {
                                            var it = new int[] {0, s, n, i0, i1, i2, h, w, c}; // prepend with 0 to handle "new axis" -1 value in permutations
                                            var oS  = it[permutations[0] + 1];
                                            var oN  = it[permutations[1] + 1];
                                            var oI0 = it[permutations[2] + 1];
                                            var oI1 = it[permutations[3] + 1];
                                            var oI2 = it[permutations[4] + 1];
                                            var oH  = it[permutations[5] + 1];
                                            var oW  = it[permutations[6] + 1];
                                            var oC  = it[permutations[7] + 1];
                                            outputTensor[oS, oN, oI0, oI1, oI2, oH, oW, oC] = readTensor[s, n, i0, i1, i2, h, w, c];
                                        }

            return outputTensor;
        }
    }

    // Description of the layer's output
    internal struct VariableTensor
    {
        public enum Layout
        {
            Unknown = 0,
            NCHW = 1, ChannelsFirst = NCHW,
            NHWC = 2, ChannelsLast = NHWC,
        };

        public int features;
        public int rank;
        public string productOfShape;
        public Layout layout;
    }

    // Keeps track of constant and variable tensors of the model
    internal class ONNXModelTensors
    {
        internal Dictionary<string, ONNXTensor> constants =
            new Dictionary<string, ONNXTensor>();

        internal Dictionary<string, VariableTensor> variables =
            new Dictionary<string, VariableTensor>();

        public void AddConstant(string name, ONNXTensor onnxTensor)
        {
            if (!onnxTensor.IsEmpty())
            {
                constants[name] = onnxTensor;
                AddVariable(name, onnxTensor);
            }
        }

        public void AddVariable(string nodeId, int features, string productOfShape,
            VariableTensor.Layout layout = VariableTensor.Layout.Unknown)
        {
            variables[nodeId] = new VariableTensor {
                features = features,
                rank = 1,
                productOfShape = productOfShape,
                layout = VariableTensor.Layout.Unknown };
        }
        public void AddVariable(string nodeId, int features = -1, int rank = -1,
            VariableTensor.Layout layout = VariableTensor.Layout.Unknown)
        {
            variables[nodeId] = new VariableTensor {
                features = features,
                rank = rank,
                layout = layout,
                productOfShape = null };
        }
        public void AddVariable(string nodeId, ONNXTensor onnxTensor)
        {
            variables[nodeId] = new VariableTensor {
                features = -1,
                rank = onnxTensor.rank,
                layout = VariableTensor.Layout.Unknown,
                productOfShape = null };
        }
        public void AddVariable(string nodeId, long[] onnxShape, string onnxLayout)
        {
            var onnxRank = onnxShape.Length;
            var permuatations = ONNXLayout.AxisPermutationsForMappingONNXLayoutToBarracuda(onnxRank, onnxLayout);
            var barracudaChannelIndex = permuatations.Length - 1;
            var onnxChannelIndex = permuatations[barracudaChannelIndex];
            var channels = (onnxLayout != "?" && onnxChannelIndex >= 0) ? (int)onnxShape[onnxChannelIndex]: -1;
            var layout = VariableTensor.Layout.Unknown;
            if (onnxLayout == "NCHW")
                layout = VariableTensor.Layout.NCHW;
            else if (onnxLayout == "NHWC")
                layout = VariableTensor.Layout.NHWC;

            variables[nodeId] = new VariableTensor {
                features = channels,
                rank = onnxRank,
                layout = layout,
                productOfShape = null };
        }

        public void CompleteUninitializedFields(ONNXNodeWrapper node)
        {
            Assert.IsTrue(variables.ContainsKey(node.Name));
            var output = variables[node.Name];

            if (output.features == -1)
            {
                if (variables.ContainsKey(node.Input0Optional))
                    output.features = variables[node.Input0Optional].features;
            }
            if (output.rank == -1)
            {
                if (constants.ContainsKey(node.Name))
                    output.rank = constants[node.Name].rank;
                else if (variables.ContainsKey(node.Input0Optional))
                    output.rank = variables[node.Input0Optional].rank;
            }
            if (output.layout == VariableTensor.Layout.Unknown)
            {
                if (variables.ContainsKey(node.Input0Optional))
                    output.layout = variables[node.Input0Optional].layout;
            }
            if (!node.IsTerminatorForProductOfShape && output.productOfShape == null)
            {
                if (variables.ContainsKey(node.Input0Optional))
                    output.productOfShape = variables[node.Input0Optional].productOfShape;
            }

            variables[node.Name] = output;
        }
    }
}
