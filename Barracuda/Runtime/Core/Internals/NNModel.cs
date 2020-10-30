using UnityEngine;

namespace Unity.Barracuda
{
    /// <summary>
    /// Barracuda Model asset
    /// </summary>
    public class NNModel : ScriptableObject
    {
        /// <summary>
        /// Model data
        /// </summary>
        [HideInInspector]
        public NNModelData modelData;
    }
}
