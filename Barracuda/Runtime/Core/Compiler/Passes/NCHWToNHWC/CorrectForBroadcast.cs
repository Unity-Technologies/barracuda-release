using System;
using System.Collections.Generic;
using UnityEngine;

namespace Unity.Barracuda.Compiler.Passes
{
    partial class NCHWToNHWCPass
    {
        void CorrectConstantsForBroadCast(ref Model nhwc)
        {
            List<Layer> correctedConstants = new List<Layer>();
            for (int l = 0; l < nhwc.layers.Count; l++)
            {
                Layer layer = nhwc.layers[l];
                for (int i = 0; i < layer.inputs.Length; i++)
                {
                    var input = layer.inputs[i];

                    if (!ModelAnalyzer.IsLayerBroacastable(layer))
                        continue;

                    if (!m_RanksByName.ContainsKey(input) || !m_RanksByName.ContainsKey(layer.name))
                        continue;

                    Layer inputLayer;
                    bool found = ModelAnalyzer.FindLayerByName(nhwc, input, out inputLayer);
                    if (!found)
                        continue;

                    if (!ModelOptimizer.IsLayerConstant(inputLayer))
                        continue;

                    if (m_RanksByName[input] < 1 || m_RanksByName[input] == m_RanksByName[layer.name])
                        continue;
                    if (inputLayer.weights.Length == 1)
                        continue;

                    if (m_RanksByName[input] > m_RanksByName[layer.name])
                        throw new Exception($"constant must be lower rank than input for broadcast to work, TODO add transpose before input");

                    Layer correctedConstLayer = new Layer("c_" + inputLayer.name + "For_" + layer.name, Layer.Type.Load);

                    // transpose dataset
                    correctedConstLayer.datasets = new Layer.DataSet[1];
                    Array.Copy(inputLayer.datasets, correctedConstLayer.datasets, inputLayer.datasets.Length);
                    correctedConstLayer.datasets[0].name = correctedConstLayer.name;


                    correctedConstLayer.weights = new float[inputLayer.weights.Length];

                    var X = inputLayer.DataSetToTensor(0);

                    int[] permutations = new[] { 0, 1, 2, 3 };
                    var rank = m_RanksByName[layer.name].Value;

                    switch (rank)
                    {
                        case 2:
                            // ONNX: 5,7 + 7
                            // Barracuda: 5,_,_,7 + 7,_,_,- => _,_,_,7
                            permutations = new[] { 1, 2, 3, 0 };
                            break;
                        case 3:
                            // ONNX: 5,7,3 + 3
                            // Barracuda: 5,_,3,7 + 3,_,_,_  => _,_,3,_
                            if (m_RanksByName[input] == 1)
                                permutations = new[] { 1, 2, 0, 3 };

                            // ONNX: 5,7,3 + 7,3
                            // Barracuda: 5,_,3,7 + 7,_,_,3 => _,_,3,7
                            else if (m_RanksByName[input] == 2)
                                permutations = new[] { 1, 2, 3, 0 };

                            break;
                        case 4:
                            // ONNX: 2,5,7,3 + 3
                            // Barracuda: 2,7,3,5 + 3,_,_,_  => _,_,3,_
                            if (m_RanksByName[input] == 1)
                                permutations = new[] { 1, 2, 0, 3 };

                            // ONNX: 2,5,7,3 + 7,3
                            // Barracuda: 2,7,3,5 + 7,_,_,3  => _,7,3,_
                            else if (m_RanksByName[input] == 2)
                                permutations = new[] { 2, 0, 1, 3 };

                            // ONNX: 2,5,7,3 + 5,7,3
                            // Barracuda: 2,7,3,5 + 5,_,3,7  => _,7,3,5
                            else if (m_RanksByName[input] == 3)
                                permutations = new[] { 1, 3, 2, 0 };
                            break;
                    }

                    if (m_isModelExportedFromNCHW && (m_layersChannelOrder[layer.name] == LayoutTransposeRemovalHelper.ChannelsOrder.NHWC))
                    {
                        switch (rank)
                        {
                            case 2:
                                // ONNX: 5,7 + 7
                                // Barracuda: 5,_,_,7 + 7,_,_,- => _,_,_,7
                                permutations = new[] { 1, 2, 3, 0 };
                                break;
                            case 3:
                                // ONNX: 5,7,3 + 3
                                // Barracuda: 5,_,7,3 + 3,_,_,_  => _,_,_,3
                                if (m_RanksByName[input] == 1)
                                    permutations = new[] { 1, 2, 3, 0 };

                                // ONNX: 5,7,3 + 7,3
                                // Barracuda: 5,_,7,3 + 7,_,_,3 => _,_,7,3
                                else if (m_RanksByName[input] == 2)
                                    permutations = new[] { 2, 3, 0, 1 };

                                break;
                            case 4:
                                // ONNX: 2,5,7,3 + 3
                                // Barracuda: 2,5,7,3 + 3,_,_,_  => _,_,_,3
                                if (m_RanksByName[input] == 1)
                                    permutations = new[] { 1, 2, 3, 0 };

                                // ONNX: 2,5,7,3 + 7,3
                                // Barracuda: 2,5,7,3 + 7,_,_,3  => _,_,7,3,
                                else if (m_RanksByName[input] == 2)
                                    permutations = new[] { 2, 3, 0, 1 };

                                // ONNX: 2,5,7,3 + 5,7,3
                                // Barracuda: 2,5,7,3 + 5,_,7,3  => _,5,7,3
                                else if (m_RanksByName[input] == 3)
                                    permutations = new[] { 1, 0, 2, 3 };
                                break;
                        }
                    }

                    var O = m_Ops.Transpose(X, permutations);
                    correctedConstLayer.ApplyTensorToDataSet(O, 0);

                    correctedConstants.Add(correctedConstLayer);
                    layer.inputs[i] = correctedConstLayer.name;
                }

                nhwc.layers[l] = layer;
            }

            foreach (var l in correctedConstants)
            {
                nhwc.layers.Insert(0, l);
            }
        }
    }
}
