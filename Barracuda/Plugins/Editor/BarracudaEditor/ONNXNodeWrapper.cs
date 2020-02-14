using Onnx;
using UnityEngine;
using UnityEditor;
using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

[assembly: InternalsVisibleToAttribute("Barracuda.EditorTests")]

namespace Barracuda
{
    public class ONNXNodeWrapper
    {
        // Layer identification (name and op)
        public string Name {
            get {
                // prefer node.output over the node.name
                return (m_ONNXNode.Output[0].Length > 0) ? m_ONNXNode.Output[0] : m_ONNXNode.Name;
            }
        }
        public string OperatorType { get { return m_ONNXNode.OpType; } }
        public bool IsConstant { get { return OperatorType == "Constant"; } }
        public bool IsTerminatorForProductOfShape { get { return OperatorType == "Reshape"; } }

        // Outputs
        public string[] Outputs { get { return m_ONNXNode.Output.ToArray(); }}

        // Inputs
        public int InputCount { get { return m_ONNXNode.Input.Count;  } }
        public string[] Inputs { get { return m_ONNXNode.Input.ToArray(); } }
        public string Input0 { get { return GetRequiredInput(0); } }
        public string Input1 { get { return GetRequiredInput(1); } }
        public string Input2 { get { return GetRequiredInput(2); } }
        public string Input3 { get { return GetRequiredInput(3); } }
        public string Input4 { get { return GetRequiredInput(4); } }
        public string Input5 { get { return GetRequiredInput(5); } }
        public string Input6 { get { return GetRequiredInput(6); } }
        public string Input0Optional { get { return InputCount > 0 ? GetRequiredInput(0) : ""; } }
        public string Input1Optional { get { return InputCount > 1 ? GetRequiredInput(1) : ""; } }
        public string Input2Optional { get { return InputCount > 2 ? GetRequiredInput(2) : ""; } }
        public string Input3Optional { get { return InputCount > 3 ? GetRequiredInput(3) : ""; } }
        public string Input4Optional { get { return InputCount > 4 ? GetRequiredInput(4) : ""; } }
        public string Input5Optional { get { return InputCount > 5 ? GetRequiredInput(5) : ""; } }
        public string Input6Optional { get { return InputCount > 6 ? GetRequiredInput(6) : ""; } }
        public bool IsInput0Const { get { return IsInputConst(0); } }
        public bool IsInput1Const { get { return IsInputConst(1); } }
        public bool IsInput2Const { get { return IsInputConst(2); } }
        public bool IsInput3Const { get { return IsInputConst(3); } }
        public bool IsInput4Const { get { return IsInputConst(4); } }
        public bool IsInput5Const { get { return IsInputConst(5); } }
        public bool IsInput6Const { get { return IsInputConst(6); } }
        public bool AreAllInputsConst { get {
            for (var i = 0; i < InputCount; ++i)
                if (!IsInputConst(i))
                    return false;
            return true;
        } }

        public int Input0Features { get { return m_ONNXModelTensors.variables[Input0].features; } }
        public int Input1Features { get { return m_ONNXModelTensors.variables[Input1].features; } }
        public int Input2Features { get { return m_ONNXModelTensors.variables[Input2].features; } }
        public int Input3Features { get { return m_ONNXModelTensors.variables[Input3].features; } }
        public int Input4Features { get { return m_ONNXModelTensors.variables[Input4].features; } }
        public int Input5Features { get { return m_ONNXModelTensors.variables[Input5].features; } }
        public int Input6Features { get { return m_ONNXModelTensors.variables[Input6].features; } }
        public int Input0Rank { get { return m_ONNXModelTensors.variables[Input0].rank; } }
        public VariableTensor.Layout Input0Layout { get { return m_ONNXModelTensors.variables[Input0].layout; } }
        public Tensor Input0Constant(string onnxLayout, string name = "X") { return GetRequiredInputAsConstant(Input0, onnxLayout, name); }
        public Tensor Input1Constant(string onnxLayout, string name)       { return GetRequiredInputAsConstant(Input1, onnxLayout, name); }
        public Tensor Input2Constant(string onnxLayout, string name)       { return GetRequiredInputAsConstant(Input2, onnxLayout, name); }
        public Tensor Input3Constant(string onnxLayout, string name)       { return GetRequiredInputAsConstant(Input3, onnxLayout, name); }
        public Tensor Input4Constant(string onnxLayout, string name)       { return GetRequiredInputAsConstant(Input4, onnxLayout, name); }
        public Tensor Input5Constant(string onnxLayout, string name)       { return GetRequiredInputAsConstant(Input5, onnxLayout, name); }
        public Tensor Input6Constant(string onnxLayout, string name)       { return GetRequiredInputAsConstant(Input6, onnxLayout, name); }
        public Tensor Input1ConstantOptional(Tensor defaultValue, string onnxLayout, string name) { try { return GetRequiredInputAsConstant(Input1, onnxLayout, name); } catch (Exception) { return defaultValue; } }
        public Tensor Input2ConstantOptional(Tensor defaultValue, string onnxLayout, string name) { try { return GetRequiredInputAsConstant(Input2, onnxLayout, name); } catch (Exception) { return defaultValue; } }
        public Tensor Input3ConstantOptional(Tensor defaultValue, string onnxLayout, string name) { try { return GetRequiredInputAsConstant(Input3, onnxLayout, name); } catch (Exception) { return defaultValue; } }
        public Tensor Input4ConstantOptional(Tensor defaultValue, string onnxLayout, string name) { try { return GetRequiredInputAsConstant(Input4, onnxLayout, name); } catch (Exception) { return defaultValue; } }
        public Tensor Input1ConstantOptional(TensorShape shape, float defaultValue, string onnxLayout, string name) { try { return GetRequiredInputAsConstant(Input1, onnxLayout, name); } catch (Exception) { return DefaultTensor(shape, defaultValue); } }
        public Tensor Input2ConstantOptional(TensorShape shape, float defaultValue, string onnxLayout, string name) { try { return GetRequiredInputAsConstant(Input2, onnxLayout, name); } catch (Exception) { return DefaultTensor(shape, defaultValue); } }
        public Tensor Input3ConstantOptional(TensorShape shape, float defaultValue, string onnxLayout, string name) { try { return GetRequiredInputAsConstant(Input3, onnxLayout, name); } catch (Exception) { return DefaultTensor(shape, defaultValue); } }
        public Tensor Input4ConstantOptional(TensorShape shape, float defaultValue, string onnxLayout, string name) { try { return GetRequiredInputAsConstant(Input4, onnxLayout, name); } catch (Exception) { return DefaultTensor(shape, defaultValue); } }
        public Tensor Input1ConstantOptional(float defaultValue, string onnxLayout, string name) { return Input1ConstantOptional(new TensorShape(1, 1), defaultValue, onnxLayout, name); }
        public Tensor Input2ConstantOptional(float defaultValue, string onnxLayout, string name) { return Input2ConstantOptional(new TensorShape(1, 1), defaultValue, onnxLayout, name); }
        public Tensor Input3ConstantOptional(float defaultValue, string onnxLayout, string name) { return Input3ConstantOptional(new TensorShape(1, 1), defaultValue, onnxLayout, name); }
        public Tensor Input4ConstantOptional(float defaultValue, string onnxLayout, string name) { return Input4ConstantOptional(new TensorShape(1, 1), defaultValue, onnxLayout, name); }

        // Attributes
        public float Alpha { get { return GetRequiredFloat("alpha"); } }
        public float Beta { get { return GetRequiredFloat("beta"); } }
        public float Gamma { get { return GetRequiredFloat("gamma"); } }
        public float Epsilon { get { return GetRequiredFloat("epsilon"); } }
        public float Mean { get { return GetRequiredFloat("mean"); } }
        public float Scale { get { return GetRequiredFloat("scale"); } }
        public float Seed { get { return GetOptionalFloat("seed", 1337f); } } // seed is always optional and defaults to 'auto generated'
        public ONNXTensor ValueAsTensor { get { return GetRequiredTensor("value"); } }
        public int Axis { get { return GetRequiredInt("axis"); } }
        public int Group { get { return GetRequiredInt("group"); } }
        public long[] Shape { get { return GetRequiredLongArray("shape"); } }
        public int[] Starts { get { return GetRequiredIntArray("starts"); } }
        public int[] Ends { get { return GetRequiredIntArray("ends"); } }
        public int[] Axes { get { return GetRequiredIntArray("axes"); } }
        public int[] KernelShape { get { return GetRequiredIntArray("kernel_shape"); } }
        public int[] Strides { get { return GetOptionalIntArray("strides", new[] {1,1}); } }
        public int[] OutputPadding { get { return GetOptionalIntArray("output_padding", new[] {0,0}); } }
        internal bool SupportsAutoPad { get { return OperatorType != "Pad"; } }
        internal bool SupportsSpatialOnlyPads { get { return OperatorType != "Pad"; } }
        public int[] Pads { get { return GetPads(); } }
        public float[] Scales { get { return GetScales(); } }
        public float AlphaOptional(float defaultValue) { return GetOptionalFloat("alpha", defaultValue); }
        public float BetaOptional(float defaultValue) { return GetOptionalFloat("beta", defaultValue); }
        public float GammaOptional(float defaultValue) { return GetOptionalFloat("gamma", defaultValue); }
        public float EpsilonOptional(float defaultValue=1e-5f) { return GetOptionalFloat("epsilon", defaultValue); }
        public float MeanOptional(float defaultValue=0f) { return GetOptionalFloat("mean", defaultValue); }
        public float ScaleOptional(float defaultValue=1f) { return GetOptionalFloat("scale", defaultValue); }
        public bool TransAOptional(bool defaultValue=false) { return GetOptionalInt("transA", defaultValue?1:0) != 0;}
        public bool TransBOptional(bool defaultValue=false) { return GetOptionalInt("transB", defaultValue?1:0) != 0;}
        public int AxisOptional(int defaultValue) { return GetOptionalInt("axis", defaultValue); }
        public int GroupOptional(int defaultValue=1) { return GetOptionalInt("group", defaultValue); }
        public int[] KernelShapeOptional(int[] defaultValue) { return GetOptionalIntArray("kernel_shape", defaultValue); }
        public int[] AxesOptional(int[] defaultValue) { return GetOptionalIntArray("axes", defaultValue); }
        public float MinOptional(float defaultValue) { return GetOptionalFloat("min", defaultValue); }
        public float MaxOptional(float defaultValue) { return GetOptionalFloat("max", defaultValue); }

        // ---------------------------------------------------------------------------------
        // Implementation
        private NodeProto m_ONNXNode;
        private ONNXModelTensors m_ONNXModelTensors;
        private List<Model.ImporterWarning> m_ImporterWarnings;

        public ONNXNodeWrapper(NodeProto ONNXNode, ONNXModelTensors ONNXModelTensors,
            List<Model.ImporterWarning> importerWarnings)
        {
            m_ONNXNode = ONNXNode;
            m_ONNXModelTensors = ONNXModelTensors;
            m_ImporterWarnings = importerWarnings;
        }

        // Logging helpers
        public void Warn(string message)
        {
            m_ImporterWarnings.Add(new Model.ImporterWarning(Name, message));
            Debug.LogWarning(message);
        }
        public void UnsupportedAttribute(string name)
        {
            AttributeProto attr;
            if (TryFindAttribute(name, out attr))
                Warn($"Unsupported attribute {name}, node {Name} of type {OperatorType}. Value will be ignored.");
        }
        public void UnsupportedAttribute(string name, int defaultValue)
        {
            if (GetOptionalInt(name, defaultValue) != defaultValue)
                Warn($"Unsupported attribute {name}, node {Name} of type {OperatorType}. Value will be ignored and defaulted to {defaultValue}.");
        }
        public void UnsupportedAttribute(string name, float defaultValue)
        {
            if (GetOptionalFloat(name, defaultValue) != defaultValue)
                Warn($"Unsupported attribute {name}, node {Name} of type {OperatorType}. Value will be ignored and defaulted to {defaultValue}.");
        }
        public void UnsupportedAttribute(string name, string defaultValue)
        {
            if (GetOptionalString(name, defaultValue) != defaultValue)
                Warn($"Unsupported attribute {name}, node {Name} of type {OperatorType}. Value will be ignored and defaulted to {defaultValue}.");
        }
        public void UnsupportedAttribute(string name, int[] defaultValue)
        {
            var valueArray = GetOptionalIntArray(name, defaultValue);
            if (!Enumerable.SequenceEqual(valueArray, defaultValue))
                Warn($"Unsupported attribute {name}, node {Name} of type {OperatorType}. Value will be ignored and defaulted to [{string.Join(", ", defaultValue)}].");
        }
        public void UnsupportedAttribute(string name, Func<int, bool> predicate, int[] defaultValue)
        {
            var valueArray = GetOptionalIntArray(name, defaultValue);
            if (!Enumerable.All(valueArray, predicate))
                Warn($"Unsupported attribute {name}, node {Name} of type {OperatorType}. Value will be ignored and defaulted to [{string.Join(", ", defaultValue)}].");
        }
        public void IgnoredAttribute(string name, string reasonToIgnore)
        {
        }

        // Input helpers
        internal string GetRequiredInput(int inputIndex)
        {
            if ((inputIndex >= m_ONNXNode.Input.Count) || (m_ONNXNode.Input[inputIndex] == ""))
                throw new OnnxLayerImportException($"required Input {inputIndex} was not found.");

            return m_ONNXNode.Input[inputIndex];
        }
        internal Tensor GetRequiredInputAsConstant(string input, string onnxLayout, string onnxName)
        {
            if (input == "")
                throw new OnnxLayerImportException("Input value is marked as required, but it is missing in the model.");

            ONNXTensor onnxTensor;
            if (!m_ONNXModelTensors.constants.TryGetValue(input, out onnxTensor))
                throw new OnnxLayerImportException(
                    $"Currently only constant tensors are supported for `{onnxName}` input in node of type {OperatorType}. Instead {Name}.{onnxName} is pointing to non constant node {input}.");

            return onnxTensor.ToBarracuda(onnxLayout);
        }
        internal bool IsInputConst(int inputIndex)
        {
            var input = GetRequiredInput(inputIndex);
            return m_ONNXModelTensors.constants.ContainsKey(input);
        }

        // Attribute helpers
        internal bool TryFindAttribute(string name, out AttributeProto attr)
        {
            return TryFindAttribute(name, AttributeProto.Types.AttributeType.Undefined, out attr);
        }
        internal bool TryFindAttribute(string name, AttributeProto.Types.AttributeType type, out AttributeProto attr)
        {
            const AttributeProto.Types.AttributeType undefined = AttributeProto.Types.AttributeType.Undefined;
            var attributes = m_ONNXNode.Attribute;
            for (var i = 0; i < attributes.Count; ++i)
            {
                attr = attributes[i];
                if (attr.Name == name && (attr.Type == type || attr.Type == undefined || type == undefined))
                    return true;
            }
            attr = null;
            return false;
        }
        internal AttributeProto FindAttribute(string name, AttributeProto.Types.AttributeType type = AttributeProto.Types.AttributeType.Undefined)
        {
            AttributeProto attr = null;
            if (TryFindAttribute(name, type, out attr))
                return attr;

            throw new OnnxLayerImportException($"Couldn't find attribute {name} of type {type}");
        }
        public float GetOptionalFloat(string name, float defaultValue)
        {
            try { return GetRequiredFloat(name); }
            catch (OnnxLayerImportException) { return defaultValue; }
        }
        public float GetRequiredFloat(string name)
        {
            return FindAttribute(name, AttributeProto.Types.AttributeType.Float).F;
        }
        public float[] GetOptionalFloatArray(string name, float[] defaultValue)
        {
            try { return GetRequiredFloatArray(name); }
            catch (OnnxLayerImportException) { return defaultValue; }
        }
        public float[] GetRequiredFloatArray(string name)
        {
            var attribute = FindAttribute(name,AttributeProto.Types.AttributeType.Floats);
            return attribute.Floats.ToArray();//.Select(x => (float)x).ToArray();
        }
        public ONNXTensor GetRequiredTensor(string name)
        {
            var tensorProto = FindAttribute(name, AttributeProto.Types.AttributeType.Tensor).T;
            return new ONNXTensor(tensorProto);
        }
        public int GetOptionalInt(string name, int defaultValue)
        {
            try { return GetRequiredInt(name); }
            catch (OnnxLayerImportException) { return defaultValue; }
        }
        public int GetRequiredInt(string name)
        {
            return (int)FindAttribute(name, AttributeProto.Types.AttributeType.Int).I;
        }
        public int[] GetOptionalIntArray(string name, int[] defaultValue)
        {
            try { return GetRequiredIntArray(name); }
            catch (OnnxLayerImportException) { return defaultValue; }
        }
        public int[] GetRequiredIntArray(string name)
        {
            var attribute = FindAttribute(name,AttributeProto.Types.AttributeType.Ints);
            return attribute.Ints.Select(x => (int)x).ToArray();
        }
        public long[] GetOptionalLongArray(string name, long[] defaultValue)
        {
            try { return GetRequiredLongArray(name); }
            catch (OnnxLayerImportException) { return defaultValue; }
        }
        public long[] GetRequiredLongArray(string name)
        {
            var attribute = FindAttribute(name,AttributeProto.Types.AttributeType.Ints);
            return attribute.Ints.ToArray();
        }
        public string GetOptionalString(string name, string defaultValue)
        {
            try { return GetRequiredString(name); }
            catch (OnnxLayerImportException) { return defaultValue; }
        }
        public string GetRequiredString(string name)
        {
            var raw = FindAttribute(name, AttributeProto.Types.AttributeType.String).S;
            return raw.ToStringUtf8();
        }

        // Complex attribute helpers
        private int[] GetPads()
        {
            var noPadding = new[] {0,0,0,0};
            if (SupportsAutoPad)
            {
                // known_paddings = {
                //     'VALID' : [0,0,0,0],
                //     'SAME_UPPER'  : [-1],
                //     'SAME_LOWER'  : [-2],
                // }
                var autoPad = GetOptionalString("auto_pad", "NOTSET");
                if (autoPad == "VALID")
                    return noPadding;
                else if (autoPad == "SAME_UPPER")
                    return new[] { -1 };
                else if (autoPad == "SAME_LOWER")
                    return new[] { -2 };
                else {} // TODO: Assert NOTSET
            }

            // NOTE: ONNX has pad layout of [z, y, x ...] while Barracuda is opposite [x, y, z ...]
            var pads = GetOptionalIntArray("pads", noPadding);
            if (SupportsSpatialOnlyPads)
            {
                // See: https://github.com/onnx/onnx/blob/master/docs/Operators.md#AveragePool
                // Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0.
                // The value represent the number of pixels added to the beginning and end part of the corresponding axis.
                // `pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...],
                // where xi_begin the number of pixels added at the beginning of axis `i` and xi_end,
                // the number of pixels added at the end of axis `i`.

                switch (pads.Length)
                {
                    case 2: return new [] { pads[0], 0, pads[1], 0 }; // 1D WW => W_W_
                    case 4: return new [] { pads[1], pads[0], pads[3], pads[2] }; // 2D HWHW => WHWH
                    case 6: Warn("3D pads are not supported yet!");
                        return new [] { pads[2], pads[1], pads[0], pads[3], pads[4], pads[5] }; // TODO: 3D DHWDHW => WHDWHD
                    default:
                        throw new OnnxLayerImportException(
                            $"Attribute pads of unsupported length {pads.Length} in {Name} ot fype {OperatorType}.");
                }
            }

            // See: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Pad
            // `pads` should be a 1D tensor of shape [2 * input_rank].
            // `pads` format should be: [x1_begin, x2_begin,...,x1_end, x2_end,...],
            // where xi_begin is the number of pad values added at the beginning of axis `i` and xi_end,
            // the number of pad values added at the end of axis `i

            Debug.Assert(pads.Length % 2 == 0);
            long[] onnxStarts = new long[pads.Length / 2];//TODO make the semantic diff between Permute(int[] and Permute(long[] more clear.
            long[] onnxEnds = new long[pads.Length / 2];
            for(int i=0; i < onnxStarts.Length; ++i)
            {
                onnxStarts[i] = (long)pads[i];
                onnxEnds[i] = (long)pads[i + onnxStarts.Length];
            }
            var starts = ONNXLayout.Permute(onnxStarts, "NCHW");
            var ends = ONNXLayout.Permute(onnxEnds, "NCHW");
            if ((starts[0] != 0) || (starts[3] != 0) || (ends[0] != 0) || (ends[3] != 0))
                Warn($"Unsupported padding, only H and W padding are supported. Value will be ignored and defaulted to 0.");

            return new int[] { starts[2], starts[1], ends[2], ends[1] };
        }

        private float[] GetScales()
        {
            var scales = GetOptionalFloatArray("scales", new float[0]);
            if (scales.Length > 0)
                return scales;
            return new[] {
                1, // N
                1, // C
                GetRequiredFloat("height_scale"),
                GetRequiredFloat("width_scale")
            };
        }

        public Tensor DefaultTensor(TensorShape tensorShape, float defaultValue)
        {
            var shape = tensorShape;
            var data = Enumerable.Repeat(defaultValue, tensorShape.length).ToArray();
            return new Tensor(shape, data);
        }
    }
}
