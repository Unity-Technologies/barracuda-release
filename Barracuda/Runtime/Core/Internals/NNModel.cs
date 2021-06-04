using System;
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

        [NonSerialized]
        Model m_Model;

        [NonSerialized]
        float m_LastLoaded;

        internal Model GetDeserializedModel(bool verbose = false, bool skipWeights = true)
        {
            if (m_Model == null)
            {
                m_Model = ModelLoader.Load(this, verbose, skipWeights);
                m_LastLoaded = Time.realtimeSinceStartup;
            }

            return m_Model;
        }

        void OnEnable()
        {
            // Used for detecting re-serialized models (e.g. adjusting import settings in the editor)
            // Force a reload on next access
            if (Time.realtimeSinceStartup >= m_LastLoaded)
                m_Model = null;
        }
    }
}
