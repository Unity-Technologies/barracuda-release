using System;
using System.Collections.Generic;

namespace Unity.Barracuda {

/// <summary>
/// Interfaces for backend implementers
/// see ModelBuilder.cs for detail on layers.
/// </summary>
public interface IOps
{
    /// <summary>
    /// Matrix multiplication o = `x` ⨯ `y`
    /// </summary>
    /// <param name="x">left Tensor</param>
    /// <param name="xTranspose">transposed `x` flag</param>
    /// <param name="y">right Tensor</param>
    /// <param name="yTranspose">transposed `y` flag</param>
    /// <returns>output Tensor</returns>
    Tensor MatMul(Tensor x, bool xTranspose, Tensor y, bool yTranspose);// @TODO: consider MatMulAdd instead

    /// <summary>
    /// Multidimensional Matrix multiplication o = `x` ⨯ `y`
    /// </summary>
    /// <param name="x">left Tensor</param>
    /// <param name="rankX">rank of `x`</param>
    /// <param name="y">right Tensor</param>
    /// <param name="rankY">rank of `y`</param>
    /// <returns>output Tensor</returns>
    Tensor MatMul(Tensor x, int rankX, Tensor y, int rankY);

    /// <summary>
    /// Dense layer (matrix multiplication) o = `x` ⨯ `w` + `b`
    /// </summary>
    /// <param name="x">x argument</param>
    /// <param name="w">w argument</param>
    /// <param name="b">bias argument</param>
    /// <param name="fusedActivation">fused activation type</param>
    /// <returns>output Tensor</returns>
    Tensor Dense(Tensor x, Tensor w, Tensor b, Layer.FusedActivation fusedActivation);

    /// <summary>
    /// rank3 Dense layer (matrix multiplication) o = `x` ⨯ `w` + `b`
    /// O: N,_,W,C / X: N,_,W,C / W:N,_,_,C / B:N,_,_,_
    /// </summary>
    /// <param name="x">x argument (rank3)</param>
    /// <param name="w">w argument (rank2)</param>
    /// <param name="b">bias argument (rank1)</param>
    /// <param name="fusedActivation">fused activation type</param>
    /// <returns>output Tensor</returns>
    Tensor Dense3(Tensor x, Tensor w, Tensor b);


    /// <summary>
    /// 2D convolution
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="k">kernel</param>
    /// <param name="b">bias</param>
    /// <param name="stride">stride</param>
    /// <param name="pad">padding</param>
    /// <param name="fusedActivation">fused activation type</param>
    /// <returns>output Tensor</returns>
    Tensor Conv2D(Tensor x, Tensor k, Tensor b, int[] stride, int[] pad, Layer.FusedActivation fusedActivation);

    /// <summary>
    /// 3D convolution
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="k">kernel</param>
    /// <param name="b">bias</param>
    /// <param name="stride">stride</param>
    /// <param name="pad">padding</param>
    /// <param name="fusedActivation">fused activation type</param>
    /// <returns>output Tensor</returns>
    Tensor Conv3D(Tensor x, Tensor k, Tensor b, int[] stride, int[] pad, Layer.FusedActivation fusedActivation);

    /// <summary>
    /// Depthwise 2D convolution
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="k">kernel</param>
    /// <param name="b">bias</param>
    /// <param name="stride">stride</param>
    /// <param name="pad">padding</param>
    /// <param name="fusedActivation">fused activation type</param>
    /// <returns>output Tensor</returns>
    Tensor DepthwiseConv2D(Tensor x, Tensor k, Tensor b, int[] stride, int[] pad, Layer.FusedActivation fusedActivation);

    /// <summary>
    /// Transpose 2D convolution
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="k">kernel</param>
    /// <param name="b">bias</param>
    /// <param name="stride">stride</param>
    /// <param name="pad">padding</param>
    /// <param name="outputAdjustment">output adjustments</param>
    /// <param name="fusedActivation">fused activation type</param>
    /// <returns>output Tensor</returns>
    Tensor Conv2DTrans(Tensor x, Tensor k, Tensor b, int[] stride, int[] pad, int[] outputAdjustment, Layer.FusedActivation fusedActivation);

    /// <summary>
    /// Upsample 2D
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="scale">scale</param>
    /// <param name="bilinear">bilinear flag</param>
    /// <returns>output Tensor</returns>
    Tensor Upsample2D(Tensor x, int[] scale, bool bilinear);

    /// <summary>
    /// Upsample 3D
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="scale">scale</param>
    /// <param name="trilinear">trilinear flag</param>
    /// <returns>output Tensor</returns>
    Tensor Upsample3D(Tensor x, int[] scale, bool trilinear);

    /// <summary>
    /// Resample 2D
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="size">size</param>
    /// <param name="bilinear">bilinear flag</param>
    /// <returns>output Tensor</returns>
    Tensor Resample2D(Tensor x, int[] size, bool bilinear);

    /// <summary>
    /// Depth to space
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="scale">scale</param>
    /// <param name="mode">mode</param>
    /// <returns>output Tensor</returns>
    Tensor DepthToSpace(Tensor x, int[] scale, Layer.DepthToSpaceMode mode);

    /// <summary>
    /// Space to depth
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="scale">scale</param>
    /// <returns>output Tensor</returns>
    Tensor SpaceToDepth(Tensor x, int[] scale);

    /// <summary>
    /// 2D max pooling
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="pool">pooling</param>
    /// <param name="stride">stride</param>
    /// <param name="pad">padding</param>
    /// <returns>output Tensor</returns>
    Tensor MaxPool2D(Tensor x, int[] pool, int[] stride, int[] pad);

    /// <summary>
    /// 2D average pooling
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="pool">pooling</param>
    /// <param name="stride">stride</param>
    /// <param name="pad">padding</param>
    /// <returns>output Tensor</returns>
    Tensor AvgPool2D(Tensor x, int[] pool, int[] stride, int[] pad);

    /// <summary>
    /// 2D global max pooling
    /// </summary>
    /// <param name="x">input</param>
    /// <returns>output Tensor</returns>
    Tensor GlobalMaxPool2D(Tensor x); // @TODO: consider, if it should be just a special case of MaxPool2D with {pool=X.width/height, stride=1}

    /// <summary>
    /// 2D global average pooling
    /// </summary>
    /// <param name="x">input</param>
    /// <returns>output Tensor</returns>
    Tensor GlobalAvgPool2D(Tensor x);

    /// <summary>
    /// 2D global average variance pooling
    /// </summary>
    /// <param name="x">input</param>
    /// <returns>output Tensor</returns>
    Tensor GlobalAvgVariancePool2D(Tensor x);

    /// <summary>
    /// 2D border padding
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="pad">padding</param>
    /// <param name="borderValue">border value</param>
    /// <returns>output Tensor</returns>
    Tensor Border2D(Tensor x, int[] pad, float borderValue);

    /// <summary>
    /// 3D border padding
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="pad">padding</param>
    /// <param name="borderValue">border value</param>
    /// <returns>output Tensor</returns>
    Tensor Border3D(Tensor x, int[] pad, float borderValue);

    /// <summary>
    /// Reflection padding
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="pad">padding</param>
    /// <returns>output Tensor</returns>
    Tensor Pad2DReflect(Tensor x, int[] pad);

    /// <summary>
    /// Symmetric padding
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="pad">padding</param>
    /// <returns>output Tensor</returns>
    Tensor Pad2DSymmetric(Tensor x, int[] pad);

    /// <summary>
    /// Edge padding
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="pad">padding</param>
    /// <returns>output Tensor</returns>
    Tensor Pad2DEdge(Tensor x, int[] pad);

    /// <summary>
    /// Scale bias o = s * x + b, element wise
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="s">scale</param>
    /// <param name="b">bias</param>
    /// <returns>output Tensor</returns>
    Tensor ScaleBias(Tensor x, Tensor s, Tensor b);

    /// <summary>
    /// Normalization
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="s">scale</param>
    /// <param name="b">bias</param>
    /// <param name="pool">pooling</param>
    /// <param name="axis">axis</param>
    /// <param name="epsilon">threshold</param>
    /// <param name="fusedActivation">fused activation type</param>
    /// <returns>output Tensor</returns>
    Tensor Normalization(Tensor x, Tensor s, Tensor b, int pool, int axis, float epsilon, Layer.FusedActivation fusedActivation);

    /// <summary>
    /// LRN (Local Response Normalization)
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="alpha">alpha</param>
    /// <param name="beta">beta</param>
    /// <param name="bias">bias</param>
    /// <param name="size">size</param>
    /// <returns>output Tensor</returns>
    Tensor LRN(Tensor x, float alpha, float beta, float bias, int size);

    /// <summary>
    /// Dropout
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="alpha">alpha</param>
    /// <returns>output Tensor</returns>
    Tensor Dropout(Tensor x, float alpha);

    /// <summary>
    /// Normal random distribution
    /// </summary>
    /// <param name="s">shape</param>
    /// <param name="mean">mean</param>
    /// <param name="scale">scale</param>
    /// <param name="seed">seed</param>
    /// <returns>output Tensor</returns>
    Tensor RandomNormal(TensorShape s, float mean, float scale, int seed);

    /// <summary>
    /// Uniform random distribution
    /// </summary>
    /// <param name="s">shape</param>
    /// <param name="mean">mean</param>
    /// <param name="scale">scale</param>
    /// <param name="seed">seed</param>
    /// <returns>output Tensor</returns>
    Tensor RandomUniform(TensorShape s, float mean, float scale, int seed);

    /// <summary>
    /// Multinomial random distribution
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="count">count</param>
    /// <param name="seed">seed</param>
    /// <returns>output Tensor</returns>
    Tensor Multinomial(Tensor x, int count, int seed);

    /// <summary>
    /// One hot
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="depth">output depth</param>
    /// <param name="onValue">on value</param>
    /// <param name="offValue">off value</param>
    /// <returns>output Tensor</returns>
    Tensor OneHot(Tensor x, int depth, float onValue, float offValue);

    /// <summary>
    /// Top K indices
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="k">k</param>
    /// <param name="axis">axis</param>
    /// <param name="largest">largest flag</param>
    /// <param name="sorted">sorted flag</param>
    /// <returns>output Tensor</returns>
    Tensor TopKIndices(Tensor x, int k, int axis, bool largest, bool sorted);

    /// <summary>
    /// Top K values
    /// </summary>
    /// <param name="X">input</param>
    /// <param name="I">indices</param>
    /// <param name="axis">axis</param>
    /// <returns>output Tensor</returns>
    Tensor TopKValues(Tensor X, Tensor I, int axis);

    /// <summary>
    /// Indices for non zero values
    /// </summary>
    /// <param name="X">input</param>
    /// <returns>output Tensor</returns>
    Tensor NonZero(Tensor X);

    /// <summary>
    /// ReLU
    /// </summary>
    /// <param name="x">input</param>
    /// <returns>output Tensor</returns>
    Tensor Relu(Tensor x);

    /// <summary>
    /// Softmax
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="axis">axis</param>
    /// <returns>output Tensor</returns>
    Tensor Softmax(Tensor x, int axis=1);

    /// <summary>
    /// LogSoftmax
    /// </summary>
    /// <param name="x">input</param>
    /// <returns>output Tensor</returns>
    Tensor LogSoftmax(Tensor x);

    /// <summary>
    /// Tanh
    /// </summary>
    /// <param name="x">input</param>
    /// <returns>output Tensor</returns>
    Tensor Tanh(Tensor x);

    /// <summary>
    /// Softplus
    /// </summary>
    /// <param name="x">input</param>
    /// <returns>output Tensor</returns>
    Tensor Softplus(Tensor x);

    /// <summary>
    /// Sigmoid
    /// </summary>
    /// <param name="x">input</param>
    /// <returns>output Tensor</returns>
    Tensor Sigmoid(Tensor x);

    /// <summary>
    /// ELU
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="alpha">alpha</param>
    /// <returns>output Tensor</returns>
    Tensor Elu(Tensor x, float alpha);

    /// <summary>
    /// ReLU capped to 6
    /// </summary>
    /// <param name="x">input</param>
    /// <returns>output Tensor</returns>
    Tensor Relu6(Tensor x);

    /// <summary>
    /// Leaky ReLU
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="alpha">alpha</param>
    /// <returns>output Tensor</returns>
    Tensor LeakyRelu(Tensor x, float alpha);

    /// <summary>
    /// SELU
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="alpha">alpha</param>
    /// <param name="gamma">gamma</param>
    /// <returns>output Tensor</returns>
    Tensor Selu(Tensor x, float alpha, float gamma);

    /// <summary>
    /// PReLU
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="alpha">alpha</param>
    /// <returns>output Tensor</returns>
    Tensor PRelu(Tensor x, Tensor alpha);

    /// <summary>
    /// Swish
    /// </summary>
    /// <param name="x">input</param>
    /// <returns>output Tensor</returns>
    Tensor Swish(Tensor x);

    /// <summary>
    /// Abs
    /// </summary>
    /// <param name="x">input</param>
    /// <returns>output Tensor</returns>
    Tensor Abs(Tensor x);

    /// <summary>
    /// Neg
    /// </summary>
    /// <param name="x">input</param>
    /// <returns>output Tensor</returns>
    Tensor Neg(Tensor x);

    /// <summary>
    /// Ceil
    /// </summary>
    /// <param name="x">input</param>
    /// <returns>output Tensor</returns>
    Tensor Ceil(Tensor x);

    /// <summary>
    /// Clip
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="min">min value</param>
    /// <param name="max">max value</param>
    /// <returns>output Tensor</returns>
    Tensor Clip(Tensor x, float min, float max);

    /// <summary>
    /// Floor
    /// </summary>
    /// <param name="x">input</param>
    /// <returns>output Tensor</returns>
    Tensor Floor(Tensor x);

    /// <summary>
    /// Round to nearest integer. In case of halfs, round to nearest even integer
    /// </summary>
    /// <param name="x">input</param>
    /// <returns>output Tensor</returns>
    Tensor Round(Tensor x);

    /// <summary>
    /// Reciprocal (1/x)
    /// </summary>
    /// <param name="x">input</param>
    /// <returns>output Tensor</returns>
    Tensor Reciprocal(Tensor x);

    /// <summary>
    /// Power
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="alpha">alpha</param>
    /// <returns>output Tensor</returns>
    Tensor Pow(Tensor x, float alpha);

    /// <summary>
    /// Exponent e^x
    /// </summary>
    /// <param name="x">input</param>
    /// <returns>output Tensor</returns>
    Tensor Exp(Tensor x);

    /// <summary>
    /// Log
    /// </summary>
    /// <param name="x">input</param>
    /// <returns>output Tensor</returns>
    Tensor Log(Tensor x);

    /// <summary>
    /// Sqrt
    /// </summary>
    /// <param name="x">input</param>
    /// <returns>output Tensor</returns>
    Tensor Sqrt(Tensor x);

    /// <summary>
    /// Acos
    /// </summary>
    /// <param name="x">input</param>
    /// <returns>output Tensor</returns>
    Tensor Acos(Tensor x);

    /// <summary>
    /// Acosh
    /// </summary>
    /// <param name="x">input</param>
    /// <returns>output Tensor</returns>
    Tensor Acosh(Tensor x);

    /// <summary>
    /// Asin
    /// </summary>
    /// <param name="x">input</param>
    /// <returns>output Tensor</returns>
    Tensor Asin(Tensor x);

    /// <summary>
    /// Asinh
    /// </summary>
    /// <param name="x">input</param>
    /// <returns>output Tensor</returns>
    Tensor Asinh(Tensor x);

    /// <summary>
    /// Atan
    /// </summary>
    /// <param name="x">input</param>
    /// <returns>output Tensor</returns>
    Tensor Atan(Tensor x);

    /// <summary>
    /// Atanh
    /// </summary>
    /// <param name="x">input</param>
    /// <returns>output Tensor</returns>
    Tensor Atanh(Tensor x);

    /// <summary>
    /// Cos
    /// </summary>
    /// <param name="x">input</param>
    /// <returns>output Tensor</returns>
    Tensor Cos(Tensor x);

    /// <summary>
    /// Cosh
    /// </summary>
    /// <param name="x">input</param>
    /// <returns>output Tensor</returns>
    Tensor Cosh(Tensor x);

    /// <summary>
    /// Sin
    /// </summary>
    /// <param name="x">input</param>
    /// <returns>output Tensor</returns>
    Tensor Sin(Tensor x);

    /// <summary>
    /// Sinh
    /// </summary>
    /// <param name="x">input</param>
    /// <returns>output Tensor</returns>
    Tensor Sinh(Tensor x);

    /// <summary>
    /// Tan
    /// </summary>
    /// <param name="x">input</param>
    /// <returns>output Tensor</returns>
    Tensor Tan(Tensor x);

    /// <summary>
    /// Add `tensors` together
    /// </summary>
    /// <param name="tensors">input tensors</param>
    /// <returns>output Tensor</returns>
    Tensor Add(Tensor[] tensors);


    /// <summary>
    /// Subtract tensors o = tensors[0] - tensors[1] - ... - tensors[N-1]
    /// </summary>
    /// <param name="tensors">input tensors</param>
    /// <returns>output Tensor</returns>
    Tensor Sub(Tensor[] tensors);

    /// <summary>
    /// Multiply tensors together
    /// </summary>
    /// <param name="tensors">input tensors</param>
    /// <returns>output Tensor</returns>
    Tensor Mul(Tensor[] tensors);

    /// <summary>
    /// Divide tensors o = tensors[0] / tensors[1] / ... / tensors[N-1]
    /// </summary>
    /// <param name="tensors">input tensors</param>
    /// <returns>output Tensor</returns>
    Tensor Div(Tensor[] tensors);

    /// <summary>
    /// Raise tensors to the power o =tensors[0] ^ tensors[1] ^ ... ^ tensors[N-1]
    /// </summary>
    /// <param name="tensors">input tensors</param>
    /// <returns>output Tensor</returns>
    Tensor Pow(Tensor[] tensors);

    /// <summary>
    /// Min
    /// </summary>
    /// <param name="tensors">input tensors</param>
    /// <returns>output Tensor</returns>
    Tensor Min(Tensor[] tensors);

    /// <summary>
    /// Max
    /// </summary>
    /// <param name="tensors">input tensors</param>
    /// <returns>output Tensor</returns>
    Tensor Max(Tensor[] tensors);

    /// <summary>
    /// Mean
    /// </summary>
    /// <param name="tensors">input tensors</param>
    /// <returns>output Tensor</returns>
    Tensor Mean(Tensor[] tensors);

    /// <summary>
    /// Reduce with max
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="axis">axis</param>
    /// <returns>output Tensor</returns>
    Tensor ReduceMax(Tensor x, int axis);

    /// <summary>
    /// Reduce with mean
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="axis">axis</param>
    /// <returns>output Tensor</returns>
    Tensor ReduceMean(Tensor x, int axis);

    /// <summary>
    /// Reduce with min
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="axis">axis</param>
    /// <returns>output Tensor</returns>
    Tensor ReduceMin(Tensor x, int axis);

    /// <summary>
    /// Reduce with product
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="axis">axis</param>
    /// <returns>output Tensor</returns>
    Tensor ReduceProd(Tensor x, int axis);

    /// <summary>
    /// Reduce with sum
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="axis">axis</param>
    /// <returns>output Tensor</returns>
    Tensor ReduceSum(Tensor x, int axis);

    /// <summary>
    /// ArgMax
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="axis">axis</param>
    /// <returns>output Tensor</returns>
    Tensor ArgMax(Tensor x, int axis);

    /// <summary>
    /// ArgMax
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="axis">axis</param>
    /// <returns>output Tensor</returns>
    Tensor ArgMin(Tensor x, int axis);

    /// <summary>
    /// Greater
    /// </summary>
    /// <param name="a">left Tensor</param>
    /// <param name="b">right Tensor</param>
    /// <returns>Tensor with `true` where a &gt; b</returns>
    Tensor Greater(Tensor a, Tensor b);

    /// <summary>
    /// Greater or equal
    /// </summary>
    /// <param name="a">left Tensor</param>
    /// <param name="b">right Tensor</param>
    /// <returns>Tensor with `true` where a &gt;= b</returns>
    Tensor GreaterEqual(Tensor a, Tensor b);

    /// <summary>
    /// Less
    /// </summary>
    /// <param name="a">left Tensor</param>
    /// <param name="b">right Tensor</param>
    /// <returns>Tensor with `true` where a &lt; b</returns>
    Tensor Less(Tensor a, Tensor b);

    /// <summary>
    /// Less or equal
    /// </summary>
    /// <param name="a">left Tensor</param>
    /// <param name="b">right Tensor</param>
    /// <returns>Tensor with `true` where a &lt; b</returns>
    Tensor LessEqual(Tensor a, Tensor b);

    /// <summary>
    /// Equal
    /// </summary>
    /// <param name="a">left Tensor</param>
    /// <param name="b">right Tensor</param>
    /// <returns>Tensor with `true` where a == b</returns>
    Tensor Equal(Tensor a, Tensor b);

    /// <summary>
    /// Or
    /// </summary>
    /// <param name="a">left Tensor</param>
    /// <param name="b">right Tensor</param>
    /// <returns>Tensor with `true` where a || b</returns>
    Tensor LogicalOr(Tensor a, Tensor b);

    /// <summary>
    /// And
    /// </summary>
    /// <param name="a">left Tensor</param>
    /// <param name="b">right Tensor</param>
    /// <returns>Tensor with `true` where a &amp;&amp; b</returns>
    Tensor LogicalAnd(Tensor a, Tensor b);

    /// <summary>
    /// Xor
    /// </summary>
    /// <param name="a">left Tensor</param>
    /// <param name="b">right Tensor</param>
    /// <returns>Tensor with `true` where a xor b</returns>
    Tensor LogicalXor(Tensor a, Tensor b);

    /// <summary>
    /// Not
    /// </summary>
    /// <param name="x">input</param>
    /// <returns>Tensor with !x values</returns>
    Tensor LogicalNot(Tensor x);

    /// <summary>
    /// Where
    /// </summary>
    /// <param name="c">Tensor c</param>
    /// <param name="a">Tensor a</param>
    /// <param name="b">Tensor b</param>
    /// <returns>Tensor with values `c` ? `a` : `b`</returns>
    Tensor Where(Tensor c, Tensor a, Tensor b);

    /// <summary>
    /// Sign
    /// </summary>
    /// <param name="x">input</param>
    /// <returns>Tensor with 1 if x &gt; 0 -1 if &lt; 0 and 0 if == 0 values</returns>
    Tensor Sign(Tensor x);

    /// <summary>
    /// Flatten
    /// </summary>
    /// <param name="x">input</param>
    /// <returns>output Tensor</returns>
    Tensor Flatten(Tensor x);

    /// <summary>
    /// Reshape
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="shape">new shape</param>
    /// <returns>output Tensor</returns>
    Tensor Reshape(Tensor x, TensorShape shape);

    /// <summary>
    /// Expand
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="shape">new shape</param>
    /// <returns>output Tensor</returns>
    Tensor Expand(Tensor x, TensorShape shape);

    /// <summary>
    /// Transpose matrix
    /// </summary>
    /// <param name="x">input</param>
    /// <returns>output Tensor</returns>
    Tensor Transpose(Tensor x);

    /// <summary>
    /// Transpose according to permutations
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="permutations">new axis order</param>
    /// <returns>output Tensor</returns>
    Tensor Transpose(Tensor x, int[] permutations);

    /// <summary>
    /// Concatenate `tensors` across `axis`
    /// </summary>
    /// <param name="tensors">input tensors</param>
    /// <param name="axis">axis</param>
    /// <returns>output Tensor</returns>
    Tensor Concat(Tensor[] tensors, int axis);

    /// <summary>
    /// Strided slice
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="starts4Dor8D"></param>
    /// <param name="ends4Dor8D"></param>
    /// <param name="strides4Dor8D">stride</param>
    /// <returns>output Tensor</returns>
    Tensor StridedSlice(Tensor x, int[] starts4Dor8D, int[] ends4Dor8D, int[] strides4Dor8D);

    /// <summary>
    /// Tile
    /// </summary>
    /// <param name="x">input</param>
    /// <param name="repeats">repetition counts</param>
    /// <returns>output Tensor</returns>
    Tensor Tile(Tensor x, int[] repeats);

    /// <summary>
    /// Gather
    /// </summary>
    /// <param name="tensors">input tensors</param>
    /// <param name="axis">axis</param>
    /// <returns>output Tensor</returns>
    Tensor Gather(Tensor[] tensors, int axis);

    /// <summary>
    /// Non max suppression tensors[0] - boxes, tensors[1] - scores
    /// </summary>
    /// <param name="tensors"></param>
    /// <param name="maxOutputBoxesPerClass">max output boxes per class</param>
    /// <param name="iouThreshold">IOU (Intersection Over Union) threshold</param>
    /// <param name="scoreThreshold">score threshold</param>
    /// <param name="centerPointBox">center point box</param>
    /// <returns>output Tensor</returns>
    Tensor NonMaxSuppression(Tensor[] tensors, int maxOutputBoxesPerClass, float iouThreshold, float scoreThreshold, int centerPointBox);

    /// <summary>
    /// LSTM
    /// </summary>
    /// <param name="X">The input sequences packed into one 3-D tensor.</param>
    /// <param name="W">W parameter weight matrix for input, output, forget, and cell gates - W[iofc]</param>
    /// <param name="R">R recurrence weight matrix for input, output, forget, and cell gates - R[iofc]</param>
    /// <param name="Wb">W bias vectors for input, output, forget, and cell gates - Wb[iofc]</param>
    /// <param name="Rb">R bias vectors for input, output, forget, and cell gates - Rb[iofc]</param>
    /// <param name="hidden">Initial value of the hidden</param>
    /// <param name="cell">Initial value of the cell</param>
    /// <returns>[Y (concatenated intermediate values of the hidden), Y_h (final hidden), Y_c (final cell)]</returns>
    Tensor[] LSTM(Tensor X, Tensor[] W, Tensor[] R, Tensor[] Wb, Tensor[] Rb, Tensor hidden, Tensor cell);

    /// <summary>
    /// Shape of the `input`
    /// </summary>
    /// <param name="X">input</param>
    /// <param name="axis">axis</param>
    /// <returns>output Tensor</returns>
    Tensor Shape(Tensor X, int axis = -1);

    /// <summary>
    /// Creates a constant of shape `input`
    /// </summary>
    /// <param name="X">input shape</param>
    /// <param name="value">value</param>
    /// <returns>output Tensor</returns>
    Tensor ConstantOfShape(TensorShape X, float value = 0.0f);

    /// <summary>
    /// Copy
    /// </summary>
    /// <param name="x">input</param>
    /// <returns>output Tensor</returns>
    Tensor Copy(Tensor x);

    /// <summary>
    /// Prepares tensor for use
    /// </summary>
    /// <param name="x">input</param>
    /// <returns>Tensor</returns>
    Tensor Prepare(Tensor x);

    /// <summary>
    /// Reset internal allocator
    /// </summary>
    /// <param name="keepCachedMemory">keep cached memory flag</param>
    void ResetAllocator(bool keepCachedMemory = true);
}

/// <summary>
/// Interfaces for model compiler
/// </summary>
internal interface IModelCompiler
{
    /// <summary>
    /// Prepare model for execution, allocating required intermediate tensors
    /// </summary>
    /// <param name="model">model</param>
    /// <param name="inputShapes">input shapes</param>
    void PrepareModel(Model model, IDictionary<string, TensorShape> inputShapes);

    /// <summary>
    /// Prepare for layer execution
    /// </summary>
    /// <param name="layer">layer</param>
    /// <param name="inputs">inputs</param>
    void PreExecuteLayer(Layer layer, Tensor[] inputs);
}

/// <summary>
/// Interfaces for variables
/// </summary>
public interface IVars : IDisposable
{
    /// <summary>
    /// Set input
    /// </summary>
    /// <param name="name">name</param>
    /// <param name="x">input</param>
    void SetInput(string name, Tensor x);

    /// <summary>
    /// Prepare storage
    /// </summary>
    /// <param name="model">model</param>
    /// <param name="optionalOpsToPrepareTensors">`IOps` to prepare tensors</param>
    /// <param name="optionalInputShapes">input shapes dictionary</param>
    void PrepareStorage(Model model, IOps optionalOpsToPrepareTensors = null, IDictionary<string, TensorShape> optionalInputShapes = null);

    /// <summary>
    /// Gather layer inputs
    /// </summary>
    /// <param name="forLayer">layer</param>
    /// <returns>all input tensors</returns>
    Tensor[] GatherInputs(Layer forLayer);

    /// <summary>
    /// Prepare storage for layer
    /// </summary>
    /// <param name="forLayer">layer</param>
    void PrepareStorage(Layer forLayer);

    /// <summary>
    /// Dispose storage that can be deleted after layer
    /// </summary>
    /// <param name="forLayer">layer</param>
    void DisposeAfterLayer(Layer forLayer);

    /// <summary>
    /// Store `result` for layer
    /// </summary>
    /// <param name="fromLayer">layer</param>
    /// <param name="result">Tensor to store</param>
    void Store(Layer fromLayer, Tensor result);

    /// <summary>
    /// Peek output
    /// </summary>
    /// <param name="name">name</param>
    /// <returns>Tensor</returns>
    Tensor PeekOutput(string name);

    /// <summary>
    /// Peek constants
    /// </summary>
    /// <param name="layerName">layer name</param>
    /// <returns>Tensor array</returns>
    Tensor[] PeekConstants(string layerName);

    /// <summary>
    /// Get allocator
    /// </summary>
    /// <returns>current `ITensorAllocator`</returns>
    ITensorAllocator GetAllocator();
}

/// <summary>
/// Interfaces for tensor allocator
/// </summary>
public interface ITensorAllocator : IDisposable
{
    /// <summary>
    /// Allocate
    /// </summary>
    /// <param name="shape">shape</param>
    /// <returns>allocated Tensor</returns>
    Tensor Alloc(TensorShape shape);

    /// <summary>
    /// Allocate with existing `ITensorData` buffer
    /// </summary>
    /// <param name="shape">shape</param>
    /// <param name="buffer">buffer</param>
    /// <returns>allocated Tensor</returns>
    Tensor Alloc(TensorShape shape, ITensorData buffer);

    // MoveToDevice() callback is called from the following Tensor methods:
    // UploadToDevice(), AttachToDevice() and DetachFromDevice()
    /// <summary>
    /// Move Tensor to device
    /// </summary>
    /// <param name="x">Tensor</param>
    /// <param name="newBuffer">new buffer</param>
    /// <param name="oldBuffer">old buffer</param>
    /// <param name="disposeDetachedBufferHint">dispose detached buffer hint</param>
    void MoveToDevice(Tensor x, ITensorData newBuffer, ITensorData oldBuffer, bool disposeDetachedBufferHint);

    // NOTE: Release() should be ready to handle edge-case situation when
    //  externally created new Tensor instance is passed with
    //  ITensorData (tensorOnDevice) that is already owned by the allocator
    /// <summary>
    /// Release Tensor
    /// </summary>
    /// <param name="x">Tensor</param>
    /// <param name="calledFromTensorDispose">called from tensor dispose flag</param>
    void Release(Tensor x, bool calledFromTensorDispose);

    /// <summary>
    /// Waive ownership
    /// </summary>
    /// <param name="x">Tensor</param>
    void WaiveOwnership(Tensor x);

    /// <summary>
    /// Reset allocator
    /// </summary>
    /// <param name="keepCachedMemory">keep cached memory flag</param>
    void Reset(bool keepCachedMemory); // end-of-frame
}

} // namespace Unity.Barracuda
