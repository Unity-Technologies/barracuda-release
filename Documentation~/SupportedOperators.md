# Supported ONNX operators

Barracuda currently supports the following [ONNX operators](https://github.com/onnx/onnx/blob/master/docs/Operators.md) and parameters. If an operator is not  on the list and you need it, please create a ticket on the [Unity ML-Agents GitHub](https://github.com/Unity-Technologies/ml-agents/issues).

### Operations
* <a href="#Constant">Constant</a>
* Reshape
* Shape
* <a href="#Unsqueeze">Unsqueeze</a>
* <a href="#Squeeze">Squeeze</a>
* Flatten
* Concat
* Slice
* Gather
* <a href="#OneHot">OneHot</a>
* LSTM
* Add
* Sum
* Sub
* Mul
* Div
* Pow
* Min
* Max
* Mean
* Greater
* Equal
* Or
* And
* Not
* Xor
* Pad
* <a href="#AveragePool">AveragePool</a>
* <a href="#MaxPool">MaxPool</a>
* GlobalAveragePool
* GlobalMaxPool
* Upsample
* <a href="#Resize">Resize</a>
* Transpose
* <a href="#Gemm">Gemm</a>
* MatMul
* Conv
* <a href="#ConvTranspose">ConvTranspose</a>
* BatchNormalization
* ImageScaler
* InstanceNormalization
* RandomNormal
* RandomNormalLike
* RandomUniform
* RandomUniformLike
* Multinomial
* ReduceMax
* ReduceMean
* ReduceMin
* ReduceProd
* ReduceSum
* Identity
* Cast
* Dropout
  

### Activations
* Relu
* <a href="#Softmax">Softmax</a>
* <a href="#LogSoftmax">LogSoftmax</a>
* Tanh
* Sqrt
* Sigmoid
* Elu
* LeakyRelu
* Selu
* PRelu
* Exp
* Log
* Reciprocal
* Abs
* Neg
* Ceil
* Floor
* Clip

<br/>

---
<br/>

#### <a name="Constant">**Constant**</a>
<dt><tt>sparse_value</tt> : not supported</dt>
<br/>

#### <a name="Unsqueeze">**Unsqueeze**</a>
<dt><tt>axis <= 1 </tt> : not supported</dt>
<br/>

#### <a name="Squeeze">**Squeeze**</a>
<dt><tt>axis <= 1 </tt> : not supported</dt>
<br/>

#### <a name="OneHot">**OneHot**</a>
<dt><tt>axis</tt> : not supported</dt>
<br/>

#### <a name="AveragePool">**AveragePool**</a>
<dt><tt>ceil_mode</tt> : not supported</dt>
<dt><tt>count_include_pad</tt> : not supported</dt>
<br/>

#### <a name="MaxPool">**MaxPool**</a>
<dt><tt>ceil_mode</tt> : not supported</dt>
<dt><tt>dilations</tt> : not supported</dt>
<dt><tt>storage_order</tt> : not supported</dt>
<br/>

#### <a name="Resize">**Resize**</a>
<dt><tt>opset-11</tt> : not supported</dt>
=>
<dt><tt>coordinate_transformation_mode</tt> : not supported</dt>
<dt><tt>cubic_coeff_a</tt> : not supported, default to -0.75</dt>
<dt><tt>exclude_outside</tt> : not supported, default to 0</dt>
<dt><tt>extrapolation_value</tt> : not supported, default to 0</dt>
<dt><tt>nearest_mode</tt> : not supported</dt>
<br/>

#### <a name="Gemm">**Gemm**</a>
<dt><tt>alpha</tt> : not supported, default to 1</dt>
<dt><tt>beta</tt> : not supported, default to 1</dt>
<dt><tt>transA</tt> : not supported, default to 0</dt>
<br/>


#### <a name="ConvTranspose">**ConvTranspose**</a>
<dt><tt>dilations</tt> : not supported, default to {1,1}</dt>
<dt><tt>group</tt> : not supported, default to 1</dt>
<dt><tt>output_shape</tt> : not supported, default to [0]</dt>
<br/>

#### <a name="Softmax">**Softmax**</a>
<dt><tt>axis</tt> : not supported</dt>
<br/>

#### <a name="LogSoftmax">**LogSoftmax**
<dt><tt>axis</tt> : not supported</dt>
<br/>
