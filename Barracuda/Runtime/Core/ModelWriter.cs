using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Threading;

using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Profiling;

namespace Unity.Barracuda {

    /// <summary>
    /// Serializes model to binary stream
    /// </summary>
    public class ModelWriter
    {
        /// <summary>
        /// Save model to file
        /// </summary>
        /// <param name="fileName">file name</param>
        /// <param name="model">`Model`</param>
        /// <param name="verbose">verbose flag</param>
        public static void Save(string fileName, Model model, bool verbose = false)
        {
            BinaryWriter writer = new BinaryWriter(File.Open(fileName, FileMode.Create));
            Save(writer, model, verbose);
            writer.Close();
        }

        /// <summary>
        /// Save model to file
        /// </summary>
        /// <param name="writer">`BinaryWriter`</param>
        /// <param name="model">`Model`</param>
        /// <param name="verbose">verbose flag</param>
        public static void Save(BinaryWriter writer, Model model, bool verbose = false)
        {
            Profiler.BeginSample("Barracuda.ModelWriter.Save");

            writer.Write((long)Model.Version);

            writer.Write(model.inputs.Count);
            for (var i = 0; i < model.inputs.Count; ++i)
            {
                WriteString(writer, model.inputs[i].name);
                WriteInt32Array(writer, model.inputs[i].shape);
            }
            WriteStringArray(writer, model.outputs);

            writer.Write(model.memories.Count);
            for (var m = 0; m < model.memories.Count; ++m)
            {
                WriteInt32Array(writer, model.memories[m].shape.ToArray());
                WriteString(writer, model.memories[m].input);
                WriteString(writer, model.memories[m].output);
            }

            // Write layers
            long offsetFromModelStartToLayer = 0;
            writer.Write(model.layers.Count);
            for (var l = 0; l < model.layers.Count; ++l)
            {
                Layer layer         = model.layers[l];
                WriteString(writer, layer.name);
                writer.Write((Int32)layer.type);
                writer.Write((Int32)layer.activation);
                writer.Write(0); //dummy 0 size array
                writer.Write(0); //dummy 0 size array
                WriteInt32Array(writer, layer.pad);
                WriteInt32Array(writer, layer.stride);
                WriteInt32Array(writer, layer.pool);
                writer.Write(layer.axis);
                writer.Write(layer.alpha);
                writer.Write(layer.beta);
                writer.Write(0); //dummy 0 size array

                WriteStringArray(writer, layer.inputs);

                long offsetFromLayerStart = 0;
                writer.Write(layer.datasets.Length);
                for (var i = 0; i < layer.datasets.Length; ++i)
                {
                    WriteString(writer, layer.datasets[i].name);
                    WriteInt32Array(writer, layer.datasets[i].shape.ToArray());
                    // Recalculate all offsets to be global inside the model
                    // this way weights can be stored in one block at the end of the file
                    Assert.AreEqual(offsetFromLayerStart, layer.datasets[i].offset - layer.datasets[0].offset);
                    writer.Write(offsetFromModelStartToLayer + offsetFromLayerStart);
                    writer.Write(layer.datasets[i].itemSizeInBytes);
                    writer.Write(layer.datasets[i].length);
                    offsetFromLayerStart += layer.datasets[i].length;
                }
                offsetFromModelStartToLayer += offsetFromLayerStart;

                if (verbose)
                    D.Log("layer " + l + ", " + layer.name + " type: " + layer.type.ToString() +
                        ((layer.activation != Layer.Activation.None) ? " activation " + layer.activation : "") +
                    " tensors: " + layer.datasets.Length +
                        " inputs: " + String.Join(",", layer.inputs));

                if (verbose)
                    foreach (var t in layer.datasets)
                        D.Log("        Tensor: " + t.shape + " offset: " + t.offset + " len: " + t.length);
            }

            //Version 20 introduce weights type but full model need to be in the same type. Per layer no supported yet.
            Assert.IsTrue(model.layers.Count >= 0);
            var weightsDataType = model.layers[0].weights.Type;
            var sizeOfDataItem = BarracudaArray.DataItemSize(weightsDataType);
            writer.Write((int)weightsDataType);

            //Pad to 4 bytes
            long writerCurrentPosition = writer.BaseStream.Position;
            long paddingForAlignment = Model.WeightsAlignment - (writerCurrentPosition % Model.WeightsAlignment);
            writer.Write(new byte[paddingForAlignment]);

            // Write tensor data
            for (var l = 0; l < model.layers.Count; ++l)
            {
                for (var d = 0; d < model.layers[l].datasets.Length; ++d)
                {
                    Assert.AreEqual(weightsDataType, model.layers[0].weights.Type);
                    byte[] dst = new byte[model.layers[l].datasets[d].length * sizeOfDataItem];
                    BarracudaArray.BlockCopy(model.layers[l].weights, (int)(model.layers[l].datasets[d].offset * sizeOfDataItem), dst, 0, dst.Length);
                    writer.Write(dst);
                }
            }

            WriteString(writer, model.IrSource);
            WriteString(writer, model.IrVersion);
            WriteString(writer, model.ProducerName);
            int numWarnings = model.Warnings.Count;
            writer.Write(numWarnings);
            for (var i = 0; i < numWarnings; ++i)
            {
                WriteString(writer, model.Warnings[i].LayerName);
                WriteString(writer, model.Warnings[i].Message);
            }

            int numMetadataProps = model.Metadata.Count;
            writer.Write(numMetadataProps);
            foreach (KeyValuePair<string, string> kvp in model.Metadata)
            {
                WriteString(writer, kvp.Key);
                WriteString(writer, kvp.Value);
            }

            Profiler.EndSample();
        }



        static void WriteInt32Array(BinaryWriter writer, Int32[] arr)
        {
            writer.Write(arr.Length);
            for (var i = 0; i < arr.Length; ++i)
                writer.Write(arr[i]);
        }

        static void WriteString(BinaryWriter writer, string str)
        {
            writer.Write(str.Length);
            writer.Write(str.ToCharArray());
        }

        static void WriteStringArray(BinaryWriter writer, string[] strArray)
        {
            writer.Write(strArray.Length);
            foreach(string str in strArray)
                WriteString(writer, str);
        }

        static void WriteStringArray(BinaryWriter writer, List<string> strArray)
        {
            writer.Write(strArray.Count);
            foreach(string str in strArray)
                WriteString(writer, str);
        }
    }
} // namespace Unity.Barracuda
