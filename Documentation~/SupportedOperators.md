# Supported ONNX operators

Barracuda currently supports the following [ONNX operators](https://github.com/onnx/onnx/blob/master/docs/Operators.md) and parameters. If an operator is not  on the list and you need it, please create a ticket on the [Unity Barracuda GitHub](https://github.com/Unity-Technologies/barracuda-release/issues).

### Operations
* <a href="#Add">Add</a>
* <a href="#And">And</a>
* <a href="#ArgMax">ArgMax</a>
* <a href="#ArgMin">ArgMin</a>
* <a href="#AveragePool">AveragePool</a>
* <a href="#BatchNormalization">BatchNormalization</a>
* <a href="#Cast">Cast</a>
* <a href="#Concat">Concat</a>
* <a href="#Constant">Constant</a>
* <a href="#ConstantOfShape">ConstantOfShape</a>
* <a href="#Conv">Conv</a>
* <a href="#ConvTranspose">ConvTranspose</a>
* <a href="#DepthToSpace">DepthToSpace</a>
* <a href="#Div">Div</a>
* <a href="#Dropout">Dropout</a>
* <a href="#Equal">Equal</a>
* <a href="#Expand">Expand</a>
* <a href="#Flatten">Flatten</a>
* <a href="#Gather">Gather</a>
* <a href="#Gemm">Gemm</a>
* <a href="#GlobalAveragePool">GlobalAveragePool</a>
* <a href="#GlobalMaxPool">GlobalMaxPool</a>
* <a href="#Greater">Greater</a>
* <a href="#Identity">Identity</a>
* <a href="#ImageScaler">ImageScaler</a>
* <a href="#InstanceNormalization">InstanceNormalization</a>
* <a href="#Less">Less</a>
* <a href="#LessOrEqual">LessOrEqual</a>
* <a href="#LRN">LRN</a>
* <a href="#LSTM">LSTM  (ML-Agents models only)</a>
* <a href="#MatMul">MatMul</a>
* <a href="#Max">Max</a>
* <a href="#MaxPool">MaxPool</a>
* <a href="#Mean">Mean</a>
* <a href="#Min">Min</a>
* <a href="#Mul">Mul</a>
* <a href="#Multinomial">Multinomial</a>
* <a href="#NonMaxSuppression">NonMaxSuppression</a>
* <a href="#NonZero">NonZero</a>
* <a href="#Not">Not</a>
* <a href="#OneHot">OneHot</a>
* <a href="#Or">Or</a>
* <a href="#Pad">Pad</a>
* <a href="#Pow">Pow</a>
* <a href="#RandomNormal">RandomNormal</a>
* <a href="#RandomNormalLike">RandomNormalLike</a>
* <a href="#RandomUniform">RandomUniform</a>
* <a href="#RandomUniformLike">RandomUniformLike</a>
* <a href="#ReduceMax">ReduceMax</a>
* <a href="#ReduceMean">ReduceMean</a>
* <a href="#ReduceMin">ReduceMin</a>
* <a href="#ReduceProd">ReduceProd</a>
* <a href="#ReduceSum">ReduceSum</a>
* <a href="#Reshape">Reshape</a>
* <a href="#Resize">Resize</a>
* <a href="#Shape">Shape</a>
* <a href="#Slice">Slice</a>
* <a href="#SpaceToDepth">SpaceToDepth</a>
* <a href="#Split">Split</a>
* <a href="#Squeeze">Squeeze</a>
* <a href="#Sub">Sub</a>
* <a href="#Sum">Sum</a>
* <a href="#Tile">Tile</a>
* <a href="#TopK">TopK</a>
* <a href="#Transpose">Transpose</a>
* <a href="#Unsqueeze">Unsqueeze</a>
* <a href="#Upsample">Upsample</a>
* <a href="#Where">Where</a>
* <a href="#Xor">Xor</a>


### Activations
* <a href="#Abs">Abs</a>
* <a href="#Acos">Acos</a>
* <a href="#Acosh">Acosh</a>
* <a href="#Asin">Asin</a>
* <a href="#Asinh">Asinh</a>
* <a href="#Atan">Atan</a>
* <a href="#Atanh">Atanh</a>
* <a href="#Ceil">Ceil</a>
* <a href="#Clip">Clip</a>
* <a href="#Cos">Cos</a>
* <a href="#Cosh">Cosh</a>
* <a href="#Elu">Elu</a>
* <a href="#Exp">Exp</a>
* <a href="#Floor">Floor</a>
* <a href="#LeakyRelu">LeakyRelu</a>
* <a href="#Log">Log</a>
* <a href="#LogSoftmax">LogSoftmax</a>
* <a href="#Neg">Neg</a>
* <a href="#PRelu">PRelu </a>
* <a href="#Reciprocal">Reciprocal</a>
* <a href="#Relu">Relu</a>
* <a href="#Round">Round</a>
* <a href="#Selu">Selu</a>
* <a href="#Sigmoid">Sigmoid</a>
* <a href="#Sin">Sin</a>
* <a href="#Sinh">Sinh</a>
* <a href="#Softmax">Softmax</a>
* <a href="#Sqrt">Sqrt</a>
* <a href="#Tan">Tan</a>
* <a href="#Tanh">Tanh</a>




### Operations details
##### <a name="Add">Add</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Add)<br>
* Maps to Barracuda op: <tt>Add</tt><br>


##### <a name="And">And</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#And)<br>
* Maps to Barracuda op: <tt>LogicalAnd</tt><br>

##### <a name="ArgMax">ArgMax</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#ArgMax)<br>
* Unsupported attribute: <tt>select_last_index</tt><br>
* Maps to Barracuda op: <tt>Reduce</tt><br>


##### <a name="ArgMin">ArgMin</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#ArgMin)<br>
* Unsupported attribute: <tt>select_last_index</tt><br>
* Maps to Barracuda op: <tt>Reduce</tt><br>


##### <a name="AveragePool">AveragePool</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#AveragePool)<br>
* Unsupported attribute: <tt>ceil_mode, count_include_pad</tt><br>
* Maps to Barracuda op: <tt>AvgPool2D</tt><br>
* Notes: No spatial 3D support.<br>


##### <a name="BatchNormalization">BatchNormalization</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#BatchNormalization)<br>
* Maps to Barracuda op: <tt>ScaleBias</tt><br>


##### <a name="Cast">Cast</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Cast)<br>
* Maps to Barracuda op: <tt>Identity</tt><br>
* Notes: No-op during inference.<br>


##### <a name="Concat">Concat</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Concat)<br>
* Maps to Barracuda op: <tt>Concat</tt><br>
* Notes: GPU path support up to 4D, will fallback to CPU if tensors have more dimensions.<br>


##### <a name="Constant">Constant</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Constant)<br>
* Unsupported attribute: <tt>sparse_value</tt><br>
* Maps to Barracuda op: <tt>Const</tt><br>


##### <a name="ConstantOfShape">ConstantOfShape</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#ConstantOfShape)<br>
* Maps to Barracuda op: <tt>Const</tt><br>


##### <a name="Conv">Conv</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv)<br>
* Maps to Barracuda op: <tt>DepthwiseConv2D, Conv2D, Conv3D</tt><br>
* Notes: Depthwise convolution 3D not supported.<br>


##### <a name="ConvTranspose">ConvTranspose</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#ConvTranspose)<br>
* Unsupported attribute: <tt>dilatations, group, output_shape</tt><br>
* Maps to Barracuda op: <tt>Conv2DTrans</tt><br>
* Notes: No spatial 3D support.<br>


##### <a name="DepthToSpace">DepthToSpace</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#DepthToSpace)<br>
* Maps to Barracuda op: <tt>DepthToSpace</tt><br>


##### <a name="Div">Div</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Div)<br>
* Maps to Barracuda op: <tt>Div</tt><br>


##### <a name="Dropout">Dropout</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Dropout)<br>
* Maps to Barracuda op: <tt>Identity</tt><br>
* Notes: No-op during inference.<br>


##### <a name="Equal">Equal</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Equal)<br>
* Maps to Barracuda op: <tt>Equal</tt><br>


##### <a name="Expand">Expand</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Expand)<br>
* Maps to Barracuda op: <tt>Expand</tt><br>


##### <a name="Flatten">Flatten</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Flatten)<br>
* Unsupported attribute: <tt>axis</tt><br>
* Maps to Barracuda op: <tt>Flatten</tt><br>


##### <a name="Gather">Gather</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gather)<br>
* Maps to Barracuda op: <tt>Gather</tt><br>


##### <a name="Gemm">Gemm</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gemm)<br>
* Unsupported attribute: <tt>alpha, beta, transA</tt><br>
* Maps to Barracuda op: <tt>Dense</tt><br>


##### <a name="GlobalAveragePool">GlobalAveragePool</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#GlobalAveragePool)<br>
* Maps to Barracuda op: <tt>GlobalAvgPool2D</tt><br>
* Notes: No spatial 3D support.<br>


##### <a name="GlobalMaxPool">GlobalMaxPool</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#GlobalMaxPool)<br>
* Maps to Barracuda op: <tt>GlobalMaxPool2D</tt><br>
* Notes: No spatial 3D support.<br>


##### <a name="Greater">Greater</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Greater)<br>
* Maps to Barracuda op: <tt>Greater</tt><br>


##### <a name="Identity">Identity</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Identity)<br>
* Maps to Barracuda op: <tt>Identity</tt><br>
* Notes: No-op during inference.<br>


##### <a name="ImageScaler">ImageScaler</a>
* Maps to Barracuda op: <tt>ScaleBias</tt><br>
* Notes: Was removed from recent ONNX versions.


##### <a name="InstanceNormalization">InstanceNormalization</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#InstanceNormalization)<br>
* Maps to Barracuda op: <tt>Normalization</tt><br>
* Notes: Support up to 4D.<br>


##### <a name="Less">Less</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Less)<br>
* Maps to Barracuda op: <tt>Less</tt><br>


##### <a name="LessOrEqual">LessOrEqual</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#LessOrEqual)<br>
* Maps to Barracuda op: <tt>LessEqual</tt><br>


##### <a name="LRN">LRN</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#LRN)<br>
* Maps to Barracuda op: <tt>LRN</tt><br>


##### <a name="LSTM">LSTM</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#LSTM)<br>
* Unsupported attribute: <tt>activation_alpha</tt><br>
* Unsupported attribute: <tt>activation_beta</tt><br>
* Unsupported attribute: <tt>activations</tt><br>
* Unsupported attribute: <tt>clip</tt><br>
* Unsupported attribute: <tt>direction</tt><br>
* Unsupported attribute: <tt>input_forget</tt><br>
* Unsupported attribute: <tt>activation_beta</tt><br>
* Maps to Barracuda op: <tt>LSTM</tt><br>
* Notes: Only ML-Agents models are supported.<br>See additional information about [execution](ModelExecution.md#LSTM-Execution).<br>


##### <a name="MatMul">MatMul</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#MatMul)<br>
* Maps to Barracuda op: <tt>MatMul, Dense</tt><br>


##### <a name="Max">Max</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Max)<br>
* Maps to Barracuda op: <tt>Max</tt><br>


##### <a name="MaxPool">MaxPool</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#MaxPool)<br>
* Unsupported attribute: <tt>ceil_mode, dilatations, storage_order</tt><br>
* Maps to Barracuda op: <tt>MaxPool2D</tt><br>
* Notes: No spatial 3D support.<br>


##### <a name="Mean">Mean</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Mean)<br>
* Maps to Barracuda op: <tt>Mean</tt><br>


##### <a name="Min">Min</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Min)<br>
* Maps to Barracuda op: <tt>Min</tt><br>


##### <a name="Mul">Mul</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Mul)<br>
* Maps to Barracuda op: <tt>Mul</tt><br>


##### <a name="Multinomial">Multinomial</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Multinomial)<br>
* Maps to Barracuda op: <tt>Multinomial</tt><br>


##### <a name="NonMaxSuppression">NonMaxSuppression</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#NonMaxSuppression)<br>
* Maps to Barracuda op: <tt>NonMaxSuppression</tt><br>


##### <a name="NonZero">NonZero</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#NonZero)<br>
* Maps to Barracuda op: <tt>NonZero</tt><br>
* Notes: Known bug for non const inputs with at least one dimension of size 1.<br>


##### <a name="Not">Not</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Not)<br>
* Maps to Barracuda op: <tt>LogicalNot</tt><br>


##### <a name="OneHot">OneHot</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#OneHot)<br>
* Unsupported attribute: <tt>axis</tt><br>
* Maps to Barracuda op: <tt>Onehot</tt><br>
* Notes: Support up to 6D.<br>


##### <a name="Or">Or</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Or)<br>
* Maps to Barracuda op: <tt>LogicalOr</tt><br>


##### <a name="Pad">Pad</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Pad)<br>
* Maps to Barracuda op: <tt>Border2D, Border3D, Pad2DReflect, Pad2DEdge</tt><br>
* Notes: Only 'constant' mode have spatial 3D support.<br>


##### <a name="Pow">Pow</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Pow)<br>
* Maps to Barracuda op: <tt>Pow</tt><br>


##### <a name="RandomNormal">RandomNormal</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#RandomNormal)<br>
* Maps to Barracuda op: <tt>RandomNormal</tt><br>


##### <a name="RandomNormalLike">RandomNormalLike</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#RandomNormalLike)<br>
* Maps to Barracuda op: <tt>RandomNormal</tt><br>


##### <a name="RandomUniform">RandomUniform</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#RandomUniform)<br>
* Maps to Barracuda op: <tt>RandomUniform</tt><br>


##### <a name="RandomUniformLike">RandomUniformLike</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#RandomUniformLike)<br>
* Maps to Barracuda op: <tt>RandomUniform</tt><br>


##### <a name="ReduceMax">ReduceMax</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceMax)<br>
* Maps to Barracuda op: <tt>ReduceMax</tt><br>


##### <a name="ReduceMean">ReduceMean</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceMean)<br>
* Maps to Barracuda op: <tt>ReduceMean</tt><br>


##### <a name="ReduceMin">ReduceMin</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceMin)<br>
* Maps to Barracuda op: <tt>ReduceMin</tt><br>


##### <a name="ReduceProd">ReduceProd</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceProd)<br>
* Maps to Barracuda op: <tt>ReduceProd</tt><br>


##### <a name="ReduceSum">ReduceSum</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceSum)<br>
* Maps to Barracuda op: <tt>ReduceSum</tt><br>


##### <a name="Reshape">Reshape</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reshape)<br>
* Maps to Barracuda op: <tt>Reshape</tt><br>


##### <a name="Resize">Resize</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Resize)<br>
* Unsupported attribute: <tt>coordinate_transformation_mode, cubic_coeff_a, exclude_outside, extrapolation_value, nearest_mode</tt><br>
* Maps to Barracuda op: <tt>Resample2D, Upsample2D, Upsample3D, AvgPool2D</tt><br>
* Notes: No spatial 3D downsampling support.<br>


##### <a name="Shape">Shape</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Shape)<br>
* Maps to Barracuda op: <tt>Const</tt><br>
* Notes: Only support constant shapes.<br>


##### <a name="Slice">Slice</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Slice)<br>
* Maps to Barracuda op: <tt>StridedSlice</tt><br>


##### <a name="SpaceToDepth">SpaceToDepth</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#SpaceToDepth)<br>
* Maps to Barracuda op: <tt>SpaceToDepth</tt><br>


##### <a name="Split">Split</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Split)<br>
* Maps to Barracuda op: <tt>StridedSlice</tt><br>


##### <a name="Squeeze">Squeeze</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Squeeze)<br>
* Maps to Barracuda op: <tt>Transpose</tt><br>


##### <a name="Sub">Sub</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sub)<br>
* Maps to Barracuda op: <tt>Sub</tt><br>


##### <a name="Sum">Sum</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sum)<br>
* Maps to Barracuda op: <tt>Sum</tt><br>


##### <a name="Tile">Tile</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Tile)<br>
* Maps to Barracuda op: <tt>Tile</tt><br>


##### <a name="TopK">TopK</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#TopK)<br>
* Maps to Barracuda op: <tt>TopKIndices,TopKValues</tt><br>
* Notes: Support up to 4D.<br>


##### <a name="Transpose">Transpose</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Transpose)<br>
* Maps to Barracuda op: <tt>Transpose</tt><br>


##### <a name="Unsqueeze">Unsqueeze</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Unsqueeze)<br>
* Maps to Barracuda op: <tt>Transpose</tt><br>


##### <a name="Upsample">Upsample</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Upsample)<br>
* Maps to Barracuda op: <tt>Upsample2D, Upsample3D, AvgPool2D</tt><br>
* Notes: No spatial 3D downsampling support.<br>


##### <a name="Where">Where</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Where)<br>
* Maps to Barracuda op: <tt>Where</tt><br>


##### <a name="Xor">Xor</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Xor)<br>
* Maps to Barracuda op: <tt>LogicalXor</tt><br>



### Activations details

##### <a name="Abs">Abs</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Abs)<br>
* Maps to Barracuda op: <tt>Abs</tt><br>


##### <a name="Acos">Acos</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Acos)<br>
* Maps to Barracuda op: <tt>Acos</tt><br>


##### <a name="Acosh">Acosh</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Acosh)<br>
* Maps to Barracuda op: <tt>Acosh</tt><br>


##### <a name="Asin">Asin</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Asin)<br>
* Maps to Barracuda op: <tt>Asin</tt><br>


##### <a name="Asinh">Asinh</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Asinh)<br>
* Maps to Barracuda op: <tt>Asinh</tt><br>


##### <a name="Atan">Atan</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Atan)<br>
* Maps to Barracuda op: <tt>Atan</tt><br>


##### <a name="Atanh">Atanh</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Atanh)<br>
* Maps to Barracuda op: <tt>Atanh</tt><br>


##### <a name="Ceil">Ceil</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Ceil)<br>
* Maps to Barracuda op: <tt>Ceil</tt><br>


##### <a name="Clip">Clip</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Clip)<br>
* Maps to Barracuda op: <tt>Clip</tt><br>


##### <a name="Cos">Cos</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Cos)<br>
* Maps to Barracuda op: <tt>Cos</tt><br>


##### <a name="Cosh">Cosh</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Cosh)<br>
* Maps to Barracuda op: <tt>Cosh</tt><br>


##### <a name="Elu">Elu</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Elu)<br>
* Maps to Barracuda op: <tt>Elu</tt><br>


##### <a name="Exp">Exp</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Exp)<br>
* Maps to Barracuda op: <tt>Exp</tt><br>


##### <a name="Floor">Floor</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Floor)<br>
* Maps to Barracuda op: <tt>Floor</tt><br>


##### <a name="LeakyRelu">LeakyRelu</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#LeakyRelu)<br>
* Maps to Barracuda op: <tt>LeykyRelu</tt><br>


##### <a name="Log">Log</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Log)<br>
* Maps to Barracuda op: <tt>Log</tt><br>


##### <a name="LogSoftmax">LogSoftmax</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#LogSoftmax)<br>
* Unsupported attribute: <tt>axis</tt><br>
* Maps to Barracuda op: <tt>LogSoftmax</tt><br>
* Notes: Support up to 6D.<br>


##### <a name="Neg">Neg</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Neg)<br>
* Maps to Barracuda op: <tt>Neg</tt><br>


##### <a name="PRelu">PRelu </a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#PRelu )<br>
* Maps to Barracuda op: <tt>PRelu</tt><br>


##### <a name="Reciprocal">Reciprocal</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reciprocal)<br>
* Maps to Barracuda op: <tt>Reciprocal</tt><br>


##### <a name="Relu">Relu</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Relu)<br>
* Maps to Barracuda op: <tt>Relu</tt><br>


##### <a name="Round">Round</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Round)<br>
* Maps to Barracuda op: <tt>Round</tt><br>


##### <a name="Selu">Selu</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Selu)<br>
* Maps to Barracuda op: <tt>Selu</tt><br>


##### <a name="Sigmoid">Sigmoid</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sigmoid)<br>
* Maps to Barracuda op: <tt>Sigmoid</tt><br>


##### <a name="Sin">Sin</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sin)<br>
* Maps to Barracuda op: <tt>Sin</tt><br>


##### <a name="Sinh">Sinh</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sinh)<br>
* Maps to Barracuda op: <tt>Sinh</tt><br>


##### <a name="Softmax">Softmax</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Softmax)<br>
* Maps to Barracuda op: <tt>Softmax</tt><br>
* Notes: Support up to 6D.<br>


##### <a name="Sqrt">Sqrt</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sqrt)<br>
* Maps to Barracuda op: <tt>Sqrt</tt><br>


##### <a name="Tan">Tan</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Tan)<br>
* Maps to Barracuda op: <tt>Tan</tt><br>


##### <a name="Tanh">Tanh</a>
* [ONNX specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Tanh)<br>
* Maps to Barracuda op: <tt>Tanh</tt><br>


