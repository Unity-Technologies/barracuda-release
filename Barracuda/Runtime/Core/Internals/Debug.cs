#define BARRACUDA_LOG_ENABLED

using System;
using UnityEngine;
using Object = UnityEngine.Object;

namespace Unity.Barracuda
{
    /// <summary>
    /// Barracuda debug logging utility
    /// </summary>
    public class D
    {
        /// <summary>
        /// Warning stack trace collection enabling flag
        /// </summary>
        public static bool warningStackTraceEnabled = Application.isEditor;

        /// <summary>
        /// Error stack trace collection enabling flag
        /// </summary>
        public static bool errorStackTraceEnabled = true;

        /// <summary>
        /// Debug log stack trace collection enabling flag
        /// </summary>
        public static bool logStackTraceEnabled = false;

        /// <summary>
        /// Warning logging enabled flag
        /// </summary>
        public static bool warningEnabled = true;

        /// <summary>
        /// Error logging enabled flag
        /// </summary>
        public static bool errorEnabled = true;

        /// <summary>
        /// Debug logging enabled flag
        /// </summary>
        public static bool logEnabled = true;

#if BARRACUDA_LOG_ENABLED

        /// <summary>
        /// Log warning
        /// </summary>
        /// <param name="message">message</param>
        public static void LogWarning(object message)
        {
            if (!warningEnabled)
                return;

            if (!warningStackTraceEnabled)
            {
                try
                {
                    var oldConfig = Application.GetStackTraceLogType(LogType.Warning);
                    Application.SetStackTraceLogType(LogType.Warning, StackTraceLogType.None);
                    UnityEngine.Debug.LogWarning(message);
                    Application.SetStackTraceLogType(LogType.Warning, oldConfig);
                }
                catch (Exception)
                {
                    UnityEngine.Debug.LogWarning(message);
                }

            }
            else
            {
                UnityEngine.Debug.LogWarning(message);
            }
        }

        /// <summary>
        /// Log warning
        /// </summary>
        /// <param name="message">message</param>
        /// <param name="context">context</param>
        public static void LogWarning(object message, Object context)
        {
            if (!warningEnabled)
                return;

            if (!warningStackTraceEnabled)
            {
                try
                {
                    var oldConfig = Application.GetStackTraceLogType(LogType.Warning);
                    Application.SetStackTraceLogType(LogType.Warning, StackTraceLogType.None);
                    UnityEngine.Debug.LogWarning(message, context);
                    Application.SetStackTraceLogType(LogType.Warning, oldConfig);
                }
                catch (Exception)
                {
                    UnityEngine.Debug.LogWarning(message, context);
                }
            }
            else
            {
                UnityEngine.Debug.LogWarning(message, context);
            }
        }

        /// <summary>
        /// Log error
        /// </summary>
        /// <param name="message">message</param>
        public static void LogError(object message)
        {
            if (!errorEnabled)
                return;

            if (!errorStackTraceEnabled)
            {
                try
                {
                    var oldConfig = Application.GetStackTraceLogType(LogType.Warning);
                    Application.SetStackTraceLogType(LogType.Error, StackTraceLogType.None);
                    UnityEngine.Debug.LogError(message);
                    Application.SetStackTraceLogType(LogType.Error, oldConfig);
                }
                catch (Exception)
                {
                    UnityEngine.Debug.LogError(message);
                }
            }
            else
            {
                UnityEngine.Debug.LogError(message);
            }
        }

        /// <summary>
        /// Log error
        /// </summary>
        /// <param name="message">message</param>
        /// <param name="context">context</param>
        public static void LogError(object message, Object context)
        {
            if (!errorEnabled)
                return;

            if (!errorStackTraceEnabled)
            {
                try
                {
                    var oldConfig = Application.GetStackTraceLogType(LogType.Warning);
                    Application.SetStackTraceLogType(LogType.Error, StackTraceLogType.None);
                    UnityEngine.Debug.LogError(message, context);
                    Application.SetStackTraceLogType(LogType.Error, oldConfig);
                }
                catch (Exception)
                {
                    UnityEngine.Debug.LogError(message, context);
                }
            }
            else
            {
                UnityEngine.Debug.LogError(message, context);
            }
        }

        /// <summary>
        /// Log debug info
        /// </summary>
        /// <param name="message">message</param>
        public static void Log(object message)
        {
            if (!logEnabled)
                return;

            if (!logStackTraceEnabled)
            {
                try
                {
                    var oldConfig = Application.GetStackTraceLogType(LogType.Warning);
                    Application.SetStackTraceLogType(LogType.Log, StackTraceLogType.None);
                    UnityEngine.Debug.Log(message);
                    Application.SetStackTraceLogType(LogType.Log, oldConfig);
                }
                catch (Exception)
                {
                    UnityEngine.Debug.Log(message);
                }
            }
            else
            {
                UnityEngine.Debug.Log(message);
            }
        }

        /// <summary>
        /// Log debug info
        /// </summary>
        /// <param name="message">message</param>
        /// <param name="context">context</param>
        public static void Log(object message, Object context)
        {
            if (!logEnabled)
                return;

            if (!logStackTraceEnabled)
            {
                try
                {
                    var oldConfig = Application.GetStackTraceLogType(LogType.Warning);
                    Application.SetStackTraceLogType(LogType.Log, StackTraceLogType.None);
                    UnityEngine.Debug.Log(message, context);
                    Application.SetStackTraceLogType(LogType.Log, oldConfig);
                }
                catch (Exception)
                {
                    UnityEngine.Debug.Log(message, context);
                }
            }
            else
            {
                UnityEngine.Debug.Log(message, context);
            }
        }
#else
        public static void LogWarning(object message)
        {

        }

        public static void LogWarning(object message, Object context)
        {

        }

        public static void LogError(object message)
        {

        }

        public static void LogError(object message, Object context)
        {

        }

        public static void Log(object message)
        {

        }

        public static void Log(object message, Object context)
        {

        }
#endif
    }

    internal class Debug : D
    {

    }
}
