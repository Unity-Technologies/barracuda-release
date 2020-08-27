#define BARRACUDA_LOG_ENABLED

using UnityEngine;

namespace Unity.Barracuda
{
    public class D
    {
        public static bool warningStackTraceEnabled = Application.isEditor;
        public static bool errorStackTraceEnabled = true;
        public static bool logStackTraceEnabled = false;

        public static bool warningEnabled = true;
        public static bool errorEnabled = true;
        public static bool logEnabled = true;

#if BARRACUDA_LOG_ENABLED
        public static void LogWarning(object message)
        {
            if (!warningEnabled)
                return;

            if (!warningStackTraceEnabled)
            {
                var oldConfig = Application.GetStackTraceLogType(LogType.Warning);
                Application.SetStackTraceLogType(LogType.Warning, StackTraceLogType.None);
                UnityEngine.Debug.LogWarning(message);
                Application.SetStackTraceLogType(LogType.Warning, oldConfig);
            }
            else
            {
                UnityEngine.Debug.LogWarning(message);
            }
        }

        public static void LogWarning(object message, Object context)
        {
            if (!warningEnabled)
                return;

            if (!warningStackTraceEnabled)
            {
                var oldConfig = Application.GetStackTraceLogType(LogType.Warning);
                Application.SetStackTraceLogType(LogType.Warning, StackTraceLogType.None);
                UnityEngine.Debug.LogWarning(message, context);
                Application.SetStackTraceLogType(LogType.Warning, oldConfig);
            }
            else
            {
                UnityEngine.Debug.LogWarning(message, context);
            }
        }

        public static void LogError(object message)
        {
            if (!errorEnabled)
                return;

            if (!errorStackTraceEnabled)
            {
                var oldConfig = Application.GetStackTraceLogType(LogType.Warning);
                Application.SetStackTraceLogType(LogType.Error, StackTraceLogType.None);
                UnityEngine.Debug.LogError(message);
                Application.SetStackTraceLogType(LogType.Error, oldConfig);
            }
            else
            {
                UnityEngine.Debug.LogError(message);
            }
        }

        public static void LogError(object message, Object context)
        {
            if (!errorEnabled)
                return;

            if (!errorStackTraceEnabled)
            {
                var oldConfig = Application.GetStackTraceLogType(LogType.Warning);
                Application.SetStackTraceLogType(LogType.Error, StackTraceLogType.None);
                UnityEngine.Debug.LogError(message, context);
                Application.SetStackTraceLogType(LogType.Error, oldConfig);
            }
            else
            {
                UnityEngine.Debug.LogError(message, context);
            }
        }

        public static void Log(object message)
        {
            if (!logEnabled)
                return;

            if (!logStackTraceEnabled)
            {
                var oldConfig = Application.GetStackTraceLogType(LogType.Warning);
                Application.SetStackTraceLogType(LogType.Log, StackTraceLogType.None);
                UnityEngine.Debug.Log(message);
                Application.SetStackTraceLogType(LogType.Log, oldConfig);
            }
            else
            {
                UnityEngine.Debug.Log(message);
            }
        }

        public static void Log(object message, Object context)
        {
            if (!logEnabled)
                return;

            if (!logStackTraceEnabled)
            {
                var oldConfig = Application.GetStackTraceLogType(LogType.Warning);
                Application.SetStackTraceLogType(LogType.Log, StackTraceLogType.None);
                UnityEngine.Debug.Log(message, context);
                Application.SetStackTraceLogType(LogType.Log, oldConfig);
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
