using System;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

namespace Unity.Barracuda.Compiler.Passes
{
    partial class NCHWToNHWCPass
    {
        void CorrectOutputLayoutToMatchNHWCLayout(ref Model nhwc)
        {
            var inputShapesNHWC = new Dictionary<string, TensorShape>();
            foreach (var i in nhwc.inputs)
            {
                inputShapesNHWC.Add(i.name, new TensorShape(i.shape));
            }

            IDictionary<string, TensorShape?> shapesByNameNHWC;
            ModelAnalyzer.ListTemporaryTensorShapes(nhwc, inputShapesNHWC, out shapesByNameNHWC);

            foreach (var o in nhwc.outputs)
            {
                if (!(shapesByNameNHWC.ContainsKey(o) && shapesByNameNHWC[o] != null))
                    continue;
                if (!(m_ShapesByName.ContainsKey(o) && m_ShapesByName[o] != null))
                    continue;

                var outputShapeNHWC = shapesByNameNHWC[o].Value;
                var outputShapeNHWCList = new List<int> { outputShapeNHWC.sequenceLength, outputShapeNHWC.numberOfDirections, outputShapeNHWC.batch, outputShapeNHWC.extraDimension, outputShapeNHWC.depth, outputShapeNHWC.height, outputShapeNHWC.width, outputShapeNHWC.channels };
                // check that outputShapeNHWC matches the NCHW shape
                var outputShape = m_ShapesByName[o].Value;
                var outputShapeONNX = IRShapeInferenceHelper.ShapeInference.ShapeToOnnxLayout(outputShape, m_RanksByName[o].Value).ToArray();
                var outputShapeList = IRShapeInferenceHelper.ShapeInference.BarracudaLayoutToTensorShapeLayout(outputShapeONNX).ToList();

                if (outputShapeNHWCList.SequenceEqual(outputShapeList))
                    continue;

                var permutations = new List<int>();
                for (int i = 0; i < 8; i++)
                {
                    for (int j = 0; j < 8; j++)
                        if (outputShapeList[j] == outputShapeNHWCList[i] && !permutations.Contains(j))
                            permutations.Add(j);
                }

                // insert transpose to match layout
                string transposedName = $"transpose_{o}_ToMatchNHWCLayout";

                for (int l = 0; l < nhwc.layers.Count; l++)
                {
                    Layer layer = nhwc.layers[l];
                    int index = Array.IndexOf(layer.inputs, o);
                    if (index != -1)
                        nhwc.layers[l].inputs[index] = transposedName;

                    if (layer.name == o)
                        nhwc.layers[l].name = transposedName;
                }

                Layer transposedOutput = new Layer(o, Layer.Type.Transpose);
                transposedOutput.inputs = new[] { transposedName };
                transposedOutput.pool = permutations.ToArray();
                
                nhwc.layers.Add(transposedOutput);
            }
        }
    }
}
