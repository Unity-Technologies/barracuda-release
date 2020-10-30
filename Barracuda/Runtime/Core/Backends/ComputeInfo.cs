using UnityEngine;
using UnityEngine.Rendering;

namespace Unity.Barracuda
{
    /// <summary>
    /// GPU compute info
    /// </summary>
    public class ComputeInfo
    {
        /// <summary>
        /// Channel order enum
        /// </summary>
        public enum ChannelsOrder
        {
            /// <summary>
            /// Channels last
            /// </summary>
            NHWC,

            /// <summary>
            /// Channels first
            /// </summary>
            NCHW
        }

        /// <summary>
        /// GPU supports shared memory
        /// </summary>
        public static bool supportsComputeSharedMemory = true;

        /// <summary>
        /// GPU supports Dense 32x32 kernels
        /// </summary>
        public static bool supportsDense32x32 = true;

        /// <summary>
        /// GPU supports Dense 64x64 kernels
        /// </summary>
        public static bool supportsDense64x64 = true;

        /// <summary>
        /// GPU supports compute
        /// </summary>
        public static bool supportsCompute = true;

        /// <summary>
        /// Max compute work group size supported by GPU
        /// </summary>
        public static uint maxComputeWorkGroupSize = 1024;

        /// <summary>
        /// GPU vendor
        /// </summary>
        public static string graphicsDeviceVendor = "";

        /// <summary>
        /// EXPERIMENTAL: Select Channel order of the compute backends.
        /// Production code should stick to default (NHWC) for now.
        /// </summary>
        public static ChannelsOrder channelsOrder = ChannelsOrder.NHWC;

        /// <summary>
        /// Static constructor, initializes and caches data
        /// </summary>
        static ComputeInfo()
        {
            string[] args = System.Environment.GetCommandLineArgs ();
            for (int i = 0; i < args.Length; i++) {
                if (args [i] == "-barracuda-compute-use-nchw")
                {
                    channelsOrder = ChannelsOrder.NCHW;
                }
            }

            supportsCompute = SystemInfo.supportsComputeShaders;

            graphicsDeviceVendor = SystemInfo.graphicsDeviceVendor;

            // TODO switch to SystemInfo.maxComputeWorkGroupSize when we bump min spec to 2019.3
            if (Application.platform == RuntimePlatform.Android)
            {
                maxComputeWorkGroupSize = (SystemInfo.graphicsDeviceType == GraphicsDeviceType.Vulkan) ? 256u : 128u;

                var gpuName = SystemInfo.graphicsDeviceName ?? "";
                var osName = SystemInfo.operatingSystem ?? "";

                // Known issue with Adreno Vulkan drivers on Android 8.x
                if (gpuName.Contains("Adreno") && osName.StartsWith("Android OS 8") &&
                    SystemInfo.graphicsDeviceType == GraphicsDeviceType.Vulkan)
                    maxComputeWorkGroupSize = 128u;
            }
            else if (Application.platform == RuntimePlatform.IPhonePlayer || Application.platform == RuntimePlatform.tvOS)
            {
                var gpuName = SystemInfo.graphicsDeviceName;
                if (gpuName != null && gpuName.StartsWith("Apple A"))
                {
                    int gpuNumber = 0, idx = "Apple A".Length;
                    while (idx < gpuName.Length && '0' <= gpuName[idx] && gpuName[idx] <= '9')
                    {
                        gpuNumber = gpuNumber * 10 + gpuName[idx++] - '0';
                    }

                    // TODO check on lower end iOS devices
                    maxComputeWorkGroupSize = (gpuNumber <= 10) ? 224u : 256u;
                }
                else
                {
                    maxComputeWorkGroupSize = 256u;
                }
            }
        }
}
}
