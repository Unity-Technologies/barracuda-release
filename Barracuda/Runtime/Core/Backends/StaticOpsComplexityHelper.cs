using System;
using UnityEngine;
using System.Collections.Generic;

namespace Unity.Barracuda {


internal class StaticLayerOppComplexity
{
    private readonly Dictionary<Layer.Type, Func<Layer, long>> m_layerComplexityStats =
        new Dictionary<Layer.Type, Func<Layer, long>>();

    private void Add(Layer.Type layerType, Func<Layer, long> opStats)
    {
        m_layerComplexityStats.Add(layerType, opStats);
    }

    public StaticLayerOppComplexity()
    {
        Add((Layer.Type.Add), (l) =>
        {
            return l.datasets.Length;
        });
        Add((Layer.Type.Mul), (l) =>
        {
            return l.datasets.Length;
        });
        Add((Layer.Type.ScaleBias), (l) =>
        {
            return 2L;
        });
        Add((Layer.Type.Dense), (l) =>
        {
            var W = l.datasets[0].shape;
            return (long)W.flatHeight * (long)W.flatWidth * 2L;
        });
        Add((Layer.Type.Conv2D), (l) =>
        {
            var K = l.datasets[0].shape;
            long n = (long)K.kernelDepth;
            long k = (long)K.kernelWidth * (long)K.kernelHeight * (long)K.channels;
            return n * k * 2L;
        });
        Add((Layer.Type.Conv3D), (l) =>
        {
            var K = l.datasets[0].shape;
            long n = (long)K.kernelDepth;
            long k = (long)K.kernelSpatialDepth * K.kernelWidth * (long)K.kernelHeight * (long)K.channels;
            return n * k * 2L;
        });
        Add((Layer.Type.DepthwiseConv2D), (l) =>
        {
            var K = l.datasets[0].shape;
            long n = (long)K.kernelDepth;
            long k = (long)K.kernelWidth * (long)K.kernelHeight;
            return n * k * 2L;
        });
    }

    public long LayerComplextity(Layer l)
    {
        var fnComplexity = m_layerComplexityStats[l.type];
        return fnComplexity(l);
    }
}


} // namespace Unity.Barracuda
