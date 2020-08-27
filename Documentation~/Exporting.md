# Exporting your model to ONNX format

To use your trained neural network in Unity, you need to export it to the [ONNX](https://onnx.ai/) format. 

ONNX (Open Neural Network Exchange) is an open format for ML models. It allows you to easily interchange models between various ML frameworks and tools. 

You can export a neural network from the following Deep Learning APIs:
*  Pytorch
*  Tensorflow
*  Keras

For a list of the ONNX operators that Barracuda supports, see [Supported operators](SupportedOperators.md).

## Pytorch

It is easy to export a `Pytorch` model to ONNX because it is built into the API. The [Pytorch documentation](https://pytorch.org/docs/stable/onnx.html) provides a good example on how to perform this conversion.

This is a simplified example:

```Python
# network
net = ...

# Input to the model
x = torch.randn(1, 3, 256, 256)

# Export the model
torch.onnx.export(net,                       # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "example.onnx",            # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=9,           # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['X'],       # the model's input names
                  output_names = ['Y']       # the model's output names
                  )
```


## TensorFlow

Exporting a TensorFlow neural network to ONNX takes a bit longer than with Pytorch, but it is still straightforward.

Install [tf2onnx](https://github.com/onnx/tensorflow-onnx).

These tutorials provide end-to-end examples:  
- [Jupyter notebook tutorial](https://github.com/onnx/tutorials/blob/master/tutorials/TensorflowToOnnx-1.ipynb)
- [Blog post on saving, loading and inferencing from TensorFlow frozen graph](https://leimao.github.io/blog/Save-Load-Inference-From-TF-Frozen-Graph)

This is a simplified example:
* First save your TensorFlow to [`.pd` format](https://www.tensorflow.org/guide/saved_model).  

```Python
# network
net = ...

# Export the model
tf.saved_model.save(net, "saved_model")
# or
tf.train.write_graph(self.sess.graph_def, directory,
                     'saved_model.pb', as_text=False)
```

* Second, convert the `.pb` file to `.onnx` with `tf2onnx`. 

```Python
# load saved network
graph_def = tf.compat.v1.GraphDef()
with open(modelPathIn, 'rb') as f:
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name='')


# optimize and save to ONNX
# Note: tf appends :0 to layer names
inputs[:] = [i+":0" for i in inputs]
outputs[:] = [o+":0" for o in outputs]

# optional step, but helpful to facilitate readability and import to Barracuda
newGraphModel_Optimized = tf2onnx.tfonnx.tf_optimize(inputs, outputs, graph_def)

# saving the model
tf.compat.v1.reset_default_graph()
tf.import_graph_def(newGraphModel_Optimized, name='')

with tf.compat.v1.Session() as sess:
    # inputs_as_nchw are optional, but with ONNX in NCHW and Tensorflow in NHWC format, it is best to add this option
    g = tf2onnx.tfonnx.process_tf_graph(sess.graph,input_names=inputs, output_names=outputs, inputs_as_nchw=inputs)

    model_proto = g.make_model(modelPathOut)
    checker = onnx.checker.check_model(model_proto)

    tf2onnx.utils.save_onnx_model("./", "saved_model", feed_dict={}, model_proto=model_proto)


# validate onnxruntime
if(args.validate_onnx_runtime):
    print("validating onnx runtime")
    import onnxruntime as rt
    sess = rt.InferenceSession("saved_model.onnx")
```

Alternatively, you can use the [command line](https://github.com/onnx/tensorflow-onnx#cli-reference) as follows:
```Bash
python -m tf2onnx.convert --graphdef model.pb --inputs=input:0 --outputs=output:0 --output model.onnx
```

Note that the flag `inputs_as_nchw` is optional, but with ONNX in `NCHW` and Tensorflow in `NHWC` format, it is best to add this option.

## Keras
To export a Keras neural network to ONNX you need [keras2onnx](https://github.com/onnx/keras-onnx).

These two tutorials provide end-to-end examples:  
- [Blog post on converting Keras model to ONNX](https://medium.com/analytics-vidhya/how-to-convert-your-keras-model-to-onnx-8d8b092c4e4f)
- [Keras ONNX Github site](https://github.com/onnx/keras-onnx)

Keras provides a Keras to ONNX format converter as a Python API. You must write a script to perform the conversion itself. See the Keras tutorials above for this API conversion script. The following code is an extract from that script: 

```python
# network
net = ...

# convert model to ONNX
onnx_model = keras2onnx.convert_keras(net,         # keras model
                         name="example",           # the converted ONNX model internal name                     
                         target_opset=9,           # the ONNX version to export the model to
                         channel_first_inputs=None # which inputs to transpose from NHWC to NCHW
                         )

onnx.save_model(onnx_model, "example.onnx")
```

Note that the flag `inputs_as_nchw` is optional, but with ONNX in `NCHW` and Keras in `NHWC` format, it is best to add this option.


## Notes
* In most cases, use ONNX `opset=9`, because it has wider coverage in Barracuda.
* When exporting from TensorFlow or Keras, use `TF-1` instead of `TF-2`.
