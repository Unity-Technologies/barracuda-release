using Onnx;
using UnityEngine;
using UnityEngine.Profiling;
using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

[assembly: InternalsVisibleToAttribute("Barracuda.EditorTests")]

namespace Unity.Barracuda
{
    // Combines information about ONNX tensor and data read from TensorProto
    public struct ONNXTensor
    {
        public long[] shape { get { return m_Shape; } }
        public int rank { get { return shape.Length; } }

        private Tensor m_Data;
        private long[] m_Shape;

        public ONNXTensor(TensorProto onnxTensor)
        {
            // read shape
            var onnxShape = onnxTensor.Dims.ToArray();


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
                        Debug.Assert((sizeof(double) * shape.length) == onnxTensor.RawData.Length);
                        Buffer.BlockCopy(byteArray, 0, typedData, 0, byteArray.Length);
                        data = typedData.Select(x => (float)x).ToArray();
                    }
                    // Float32
                    else if (onnxTensor.DataType == (int)TensorProto.Types.DataType.Float)
                    {
                        data = new float[shape.length];
                        Debug.Assert((sizeof(float) * shape.length) == onnxTensor.RawData.Length);
                        Buffer.BlockCopy(byteArray, 0, data, 0, byteArray.Length);
                    }
                    // Float16
                    else if (onnxTensor.DataType == (int)TensorProto.Types.DataType.Float16)
                    {
                        var typedData = new UInt16[shape.length];
                        Debug.Assert((sizeof(UInt16) * shape.length) == onnxTensor.RawData.Length);
                        Buffer.BlockCopy(byteArray, 0, typedData, 0, byteArray.Length);
                        data = typedData.Select(x => HalfHelper.HalfToSingle(x)).ToArray();
                    }
                    // Int8
                    else if (onnxTensor.DataType == (int)TensorProto.Types.DataType.Int8)
                    {
                        var typedData = new sbyte[shape.length];
                        Debug.Assert((sizeof(sbyte) * shape.length) == onnxTensor.RawData.Length);
                        Buffer.BlockCopy(byteArray, 0, typedData, 0, byteArray.Length);
                        data = typedData.Select(x => (float)x).ToArray();
                    }
                    // Int16
                    else if (onnxTensor.DataType == (int)TensorProto.Types.DataType.Int16)
                    {
                        var typedData = new short[shape.length];
                        Debug.Assert((sizeof(short) * shape.length) == onnxTensor.RawData.Length);
                        Buffer.BlockCopy(byteArray, 0, typedData, 0, byteArray.Length);
                        data = typedData.Select(x => (float)x).ToArray();
                    }
                    // Int32
                    else if (onnxTensor.DataType == (int)TensorProto.Types.DataType.Int32)
                    {
                        var typedData = new int[shape.length];
                        Debug.Assert((sizeof(int) * shape.length) == onnxTensor.RawData.Length);
                        Buffer.BlockCopy(byteArray, 0, typedData, 0, byteArray.Length);
                        data = typedData.Select(x => (float)x).ToArray();
                    }
                    // Int64
                    else if (onnxTensor.DataType == (int)TensorProto.Types.DataType.Int64)
                    {
                        var typedData = new long[shape.length];
                        Debug.Assert((sizeof(long) * shape.length) == onnxTensor.RawData.Length);
                        Buffer.BlockCopy(byteArray, 0, typedData, 0, byteArray.Length);
                        data = typedData.Select(x => (float)x).ToArray();
                    }
                    // UInt8
                    else if (onnxTensor.DataType == (int)TensorProto.Types.DataType.Uint8)
                    {
                        var typedData = new byte[shape.length];
                        Debug.Assert((sizeof(byte) * shape.length) == onnxTensor.RawData.Length);
                        Buffer.BlockCopy(byteArray, 0, typedData, 0, byteArray.Length);
                        data = typedData.Select(x => (float)x).ToArray();
                    }
                    // UInt16
                    else if (onnxTensor.DataType == (int)TensorProto.Types.DataType.Uint16)
                    {
                        var typedData = new ushort[shape.length];
                        Debug.Assert((sizeof(ushort) * shape.length) == onnxTensor.RawData.Length);
                        Buffer.BlockCopy(byteArray, 0, typedData, 0, byteArray.Length);
                        data = typedData.Select(x => (float)x).ToArray();
                    }
                    // UInt32
                    else if (onnxTensor.DataType == (int)TensorProto.Types.DataType.Uint32)
                    {
                        var typedData = new uint[shape.length];
                        Debug.Assert((sizeof(uint) * shape.length) == onnxTensor.RawData.Length);
                        Buffer.BlockCopy(byteArray, 0, typedData, 0, byteArray.Length);
                        data = typedData.Select(x => (float)x).ToArray();
                    }
                    // UInt64
                    else if (onnxTensor.DataType == (int)TensorProto.Types.DataType.Uint64)
                    {
                        var typedData = new ulong[shape.length];
                        Debug.Assert((sizeof(ulong) * shape.length) == onnxTensor.RawData.Length);
                        Buffer.BlockCopy(byteArray, 0, typedData, 0, byteArray.Length);
                        data = typedData.Select(x => (float)x).ToArray();
                    }
                    // Bool
                    else if (onnxTensor.DataType == (int)TensorProto.Types.DataType.Bool)
                    {
                        var typedData = new bool[shape.length];
                        Debug.Assert((sizeof(bool) * shape.length) == onnxTensor.RawData.Length);
                        Buffer.BlockCopy(byteArray, 0, typedData, 0, byteArray.Length);
                        data = typedData.Select(x => x ? 1.0f : 0.0f).ToArray();
                    }
                    else
                        throw new OnnxLayerImportException($"Tensor data type {(TensorProto.Types.DataType)onnxTensor.DataType} is not supported.");
                }
                // Float32
                else if ((onnxTensor.FloatData != null) && (onnxTensor.FloatData.Count != 0))
                {
                    Debug.Assert(shape.length == onnxTensor.FloatData.Count);
                    data = new float[shape.length];
                    onnxTensor.FloatData.CopyTo(data, 0);
                }
                // Int32
                else if ((onnxTensor.Int32Data != null) && (onnxTensor.Int32Data.Count != 0))
                {
                    Debug.Assert(shape.length == onnxTensor.Int32Data.Count);
                    data = onnxTensor.Int32Data.Select(x => (float)x).ToArray();
                }
                // Int64
                else if ((onnxTensor.Int64Data != null) && (onnxTensor.Int64Data.Count != 0))
                {
                    Debug.Assert(shape.length == onnxTensor.Int64Data.Count);
                    data = onnxTensor.Int64Data.Select(x => (float)x).ToArray();
                }
                else
                {
                    throw new OnnxLayerImportException("Could not read tensor data for constant tensor.");
                }

                m_Data = new Tensor(shape, new SharedArrayTensorData(data));
                m_Shape = onnxShape;
            }
        }

        public ONNXTensor(Tensor data, long[] onnxShape)
        {
            m_Data = data;
            m_Shape = onnxShape;
        }

        public bool IsEmpty()
        {
            return m_Shape.Any(s => s == 0);
        }

        public ONNXTensor Reshape(long[] onnxShape)
        {
            var symbolicShape = ONNXLayout.ConvertSymbolicShapeToBarracuda(onnxShape, "?");
            var reshapedData = m_Data.Reshape(symbolicShape);
            for (var i = 0; i < onnxShape.Length; ++i)
            {
                if (onnxShape[i] < 0)
                    onnxShape[i] = reshapedData.shape[i];
                Debug.Assert(onnxShape[i] == reshapedData.shape[i]);
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

        public ONNXTensor SqueezeAll()
        {
            var newShape = m_Shape.Where(x => x > 1).ToArray();
            if (newShape.Length == 0)
                newShape = new[] { 1L };
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
            Debug.Assert(starts.Length == ends.Length);
            Debug.Assert(starts.Length == steps.Length);

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

            // caluclate shape for sliced tensor
            for (var i = 0; i < m_Shape.Length; ++i)
                newShape[i] = (ends[i] - starts[i]) / steps[i];

            Tensor result = new Tensor(newShape);

            // pad to the number of the loops - 4
            starts = starts.Concat(Enumerable.Repeat(0, 4 - starts.Length)).ToArray();
            ends   = ends.Concat  (Enumerable.Repeat(1, 4 - ends.Length)).ToArray(); // we need to keep 1, to visit all elements at least once
            steps  = steps.Concat (Enumerable.Repeat(1, 4 - steps.Length)).ToArray();

            for (int b = starts[0], bo = 0; b < ends[0]; b += steps[0], bo++)
                for (int y = starts[1], yo = 0; y < ends[1]; y += steps[1], yo++)
                    for (int x = starts[2], xo = 0; x < ends[2]; x += steps[2], xo++)
                        for (int c = starts[3], co = 0; c < ends[3]; c += steps[3], co++)
                            result[bo, yo, xo, co] = m_Data[b, y, x, c];

            return new ONNXTensor(result, newShape.Select(x => (long)x).ToArray());
        }

        public ONNXTensor Gather(int axis, int[] indices)
        {
            // good explanation can be found here:
            // https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms
            int[] newShape = m_Shape.Select(i => (int)i).ToArray();
            newShape[axis] = indices.Length;

            Tensor result = new Tensor(newShape);

            // pad to the number of the loops - 4
            var newShapeRank4 = newShape.Concat(Enumerable.Repeat(1, 4 - newShape.Length)).ToArray(); // we need to keep 1, to visit all elements at least once

            for (int b = 0; b < newShapeRank4[0]; ++b)
                for (int y = 0; y < newShapeRank4[1]; ++y)
                    for (int x = 0; x < newShapeRank4[2]; ++x)
                        for (int c = 0; c < newShapeRank4[3]; ++c)
                        {
                            if (axis == 0)
                                result[b, y, x, c] = m_Data[indices[b], y, x, c];
                            else if (axis == 1)
                                result[b, y, x, c] = m_Data[b, indices[y], x, c];
                            else if (axis == 2)
                                result[b, y, x, c] = m_Data[b, y, indices[x], c];
                            else
                                result[b, y, x, c] = m_Data[b, y, x, indices[c]];
                        }

            return new ONNXTensor(result, newShape.Select(x => (long)x).ToArray());
        }

        public Tensor ToBarracuda(string onnxLayout)
        {
            Profiler.BeginSample("ONNXTensor.ToBarracuda");
            if (onnxLayout == "?")
                throw new OnnxLayerImportException("Unknown ONNX layout in not supported when converting constant tensor to Barracuda");

            Debug.Assert(m_Shape.All(v => v > 0));
            var permutations = ONNXLayout.AxisPermutationsForMappingONNXLayoutToBarracuda(rank, onnxLayout);
            Debug.Assert(rank <= permutations.Length);

            var outTensor = Permute(m_Data, permutations);
            Profiler.EndSample();
            return outTensor;
        }

        internal static Tensor Permute(Tensor inTensor, int[] permutations) // TODO: unify Permute() arguments
        {
            var padPermutationsToBarracudaRank = 4 - permutations.Length;
            if (padPermutationsToBarracudaRank > 0)
                permutations = permutations.Concat(Enumerable.Range(permutations.Length, padPermutationsToBarracudaRank)).ToArray();
            Debug.Assert(permutations.Length == 4);

            // See: https://stackoverflow.com/a/32034565
            Profiler.BeginSample("ONNXTensor.Permute");
            var outTensor = new Tensor(ONNXLayout.Permute(inTensor.shape.ToArray(), permutations));
            Debug.Assert(outTensor.length == inTensor.length);

            // {0, 2, 3, 1} => {0, 3, 1, 2}
            // {2, 3, 1, 0} => {3, 2, 0, 1}
            //              => {find_index(0), find_index(1), find_index(2), find_index(3)}
            var reversePermute = new int[permutations.Length];
            for (var i = 0; i < permutations.Length; ++i)
                reversePermute[i] = Array.IndexOf(permutations, i);

            // outTensor strides
            var outStrideC   =               outTensor.channels;
            var outStrideWC  = outStrideC  * outTensor.width;
            var outStrideHWC = outStrideWC * outTensor.height;

            var outStride = new int[reversePermute.Length];
            for (var i = 0; i < reversePermute.Length; ++i)
                outStride[i] = new[] {0, outStrideHWC, outStrideWC, outStrideC, 1}[reversePermute[i] + 1];

            // inTensor strides
            var inStrideC   =              inTensor.channels;
            var inStrideWC  = inStrideC  * inTensor.width;
            var inStrideHWC = inStrideWC * inTensor.height;

            var inShape = inTensor.shape.ToArray();
            for (var n = 0; n < inShape[0]; ++n)
                for (var h = 0; h < inShape[1]; ++h)
                    for (var w = 0; w < inShape[2]; ++w)
                        for (var c = 0; c < inShape[3]; ++c)
                        {
                            outTensor[n * outStride[0] + h * outStride[1] + w * outStride[2] + c * outStride[3]] =
                                inTensor[n * inStrideHWC + h * inStrideWC + w * inStrideC + c];
                        }

            Profiler.EndSample();
            return outTensor;
        }

        // slow version - kept just for performance comparison and validation
        internal static Tensor PermuteSlow(Tensor readTensor, int[] permutations) // TODO: unify Permute() arguments
        {
            var outputTensor = new Tensor(ONNXLayout.Permute(readTensor.shape.ToArray(), permutations));
            Debug.Assert(outputTensor.length == readTensor.length);

            var inShape = readTensor.shape.ToArray();
            for (var n = 0; n < inShape[0]; ++n)
                for (var h = 0; h < inShape[1]; ++h)
                    for (var w = 0; w < inShape[2]; ++w)
                        for (var c = 0; c < inShape[3]; ++c)
                        {
                            var it = new int[] {0, n, h, w, c}; // prepend with 0 to handle "new axis" -1 value in permutations
                            var oN = it[permutations[0] + 1];
                            var oH = it[permutations[1] + 1];
                            var oW = it[permutations[2] + 1];
                            var oC = it[permutations[3] + 1];
                            outputTensor[oN, oH, oW, oC] = readTensor[n, h, w, c];
                        }

            return outputTensor;
        }
    }

    // Description of the layer's output
    public struct VariableTensor
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
    public class ONNXModelTensors
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
            Debug.Assert(variables.ContainsKey(node.Name));
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
