using System;
using System.Linq;
using Onnx;
using UnityEngine;

namespace Unity.Barracuda.ONNX {

static class ONNXExtensions
{
    public static int[] AsInts(this TensorShapeProto shape)
    {
        return shape.Dim.Select(v => v.DimValue < int.MinValue ? int.MinValue : v.DimValue > int.MaxValue ? int.MaxValue : (int)v.DimValue).ToArray();
    }
}

}
