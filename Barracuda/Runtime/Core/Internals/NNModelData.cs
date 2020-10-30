using UnityEngine;

namespace Unity.Barracuda
{
    /// <summary>
    /// Barracuda `Model` data storage
    /// </summary>
    public class NNModelData : ScriptableObject
    {
        /// <summary>
        /// `Model` byte stream
        /// </summary>
        [HideInInspector]
        public byte[] Value;
    }
}
