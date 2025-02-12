<div style="background: #ffe2d7; padding: 16px; border-radius: 4px; margin: 16px 0;">
    The Barracuda package is deprecated. Use <a href="https://docs.unity3d.com/Packages/com.unity.sentis@latest/index.html">the Sentis package</a> instead.
</div>

# Visualizing Model

It can be useful to visualize a model both before and after it is imported into Barracuda.

[Netron](https://github.com/lutzroeder/netron) is a popular viewer for neural network models, it support both ONNX and Barracuda file format (and many others). 



As a consequence once Netron is [installed](https://github.com/lutzroeder/netron/releases) one can open those files natively:

* To open the ONNX model: `double click` or press `Enter` on the asset in the Project Explorer.

* To open the Barracuda model: press the `Open imported NN model as temp file` button in the Inspector.

  ![Assets](images/ModelOpenVisualization.png)



Here in Netron both the ONNX model (on the left) and the Barracuda model (on the right) have been opened side by side.

![Assets](images/ModelDiffVisualization.png)

