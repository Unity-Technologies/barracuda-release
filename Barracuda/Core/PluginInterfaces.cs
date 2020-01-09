using System;
using System.Collections.Generic;
using UnityEngine;

namespace Barracuda
{
    public interface BLASPlugin
    {
        bool IsCurrentPlatformSupported();
        unsafe void SGEMM(float* Ap, int AN, int AM,
            float* Bp, int BN, int BM,
            float* Cp, int CN, int CM, int bs,
            bool transposeA = false, bool transposeB = false);
    }

    public class BLASPluginFactory
    {
        public static BLASPlugin CreateBLASPlugin()
        {
            BLASPlugin blas = null;
            
            // TODO make plugins discoverable via custom attributes
            Stack<string> plugins = new Stack<string>();
            plugins.Push(typeof(CSharpBLAS).FullName);
            plugins.Push("BurstBLAS");

            if (Application.platform == RuntimePlatform.IPhonePlayer)
                plugins.Push("iOSBLAS");
            else if (Application.platform == RuntimePlatform.OSXPlayer || Application.platform == RuntimePlatform.OSXEditor)
                plugins.Push("MacBLAS");

            while (plugins.Count > 0)
            {
                var candidate = plugins.Pop();
                D.Log($"Probing {candidate}");
                foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
                {
                    var t = assembly.GetType(candidate);
                    if (t != null)
                    {
                        try
                        {
                            var inst = Activator.CreateInstance(t) as BLASPlugin;

                            if (inst != null && inst.IsCurrentPlatformSupported())
                            {
                                blas = inst;
                            }
                        }
                        catch (Exception e)
                        {
                            D.LogWarning($"Failed to load {t} with exception {e}");
                            break;
                        }  
                    }
                }

                // Found working candidate
                if (blas != null)
                    break;
            }

            return blas;
        }
    }
}