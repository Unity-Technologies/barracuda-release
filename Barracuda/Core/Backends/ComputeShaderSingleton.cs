using System.Collections.Generic;
using UnityEngine;
using Barracuda;

public sealed class ComputeShaderSingleton
{
    public readonly ComputeShader referenceKernels;
    public readonly ComputeShader[] kernels;

    private static readonly ComputeShaderSingleton instance = new ComputeShaderSingleton ();

    private ComputeShaderSingleton ()
    {
        referenceKernels = LoadIf(ComputeInfo.supportsCompute, "BarracudaReferenceImpl");
        
        List<ComputeShader> kernelsList = new List<ComputeShader>();

        LoadIf(ComputeInfo.supportsCompute, "Generic", kernelsList);
        LoadIf(ComputeInfo.supportsCompute, "Activation", kernelsList);
        LoadIf(ComputeInfo.supportsCompute, "Broadcast", kernelsList);
        LoadIf(ComputeInfo.supportsCompute, "Pool", kernelsList);
        LoadIf(ComputeInfo.supportsCompute, "Pad", kernelsList);
        LoadIf(ComputeInfo.supportsCompute, "Dense", kernelsList);
        LoadIf(ComputeInfo.supportsCompute, "DenseFP16", kernelsList);
        LoadIf(ComputeInfo.supportsCompute, "Conv", kernelsList);

        kernels = kernelsList.ToArray();
    }

    public static ComputeShaderSingleton Instance {
        get { return instance; }
    }

    public static ComputeShader LoadIf(bool condition, string fileName)
    {
        if (condition)
            return (ComputeShader)Resources.Load(fileName);

        return null;
    }
    
    public static void LoadIf(bool condition, string fileName, List<ComputeShader> list)
    {
        ComputeShader shader = LoadIf(condition, fileName);
        
        if (shader)
            list.Add(shader);
    }

    public bool supported { get { return SystemInfo.supportsComputeShaders; } }
}
