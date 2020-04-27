namespace Unity.Barracuda {

public class CompareOpsUtils
{
    public enum LogLevel
    {
        Warning,
        Error
    }

    static public void CheckSame(Tensor X, Tensor Y, Layer.Type type, LogLevel logLevel, float epsilon=0.0001f, params Tensor[] inputs)
    {
        CheckSame(X, Y, type.ToString(), logLevel, epsilon, inputs);
    }

    static public void CheckSame(Tensor X, Tensor Y, string opName, LogLevel logLevel, float epsilon=0.0001f, params Tensor[] inputs)
    {
        if (!X.Approximately(Y, epsilon))
        {
            if (logLevel == LogLevel.Error)
            {
                string mainLogMessage = $"Tensors not equal after {opName}, epsilon {epsilon}";
                D.LogError(mainLogMessage);
            }
            else
            {
                string mainLogMessage = $"Tensors not equal after {opName} max error: {X.MaxDifference(Y)}";
                D.LogWarning(mainLogMessage);

                D.Log("First: " + X.shape);
                D.Log("Second:" + Y.shape);

                X.PrintDataPart(X.channels * X.width * 2);
                Y.PrintDataPart(Y.channels * Y.width * 2);

                for (var i = 0; i < inputs.Length; i++)
                {
                    inputs[i].PrintDataPart(32, "input_" + i);
                }
            }


        }
        if (X.tensorOnDevice != Y.tensorOnDevice)
            Y.Dispose();
    }

    static public void CheckApproximately(Tensor X, Tensor Y, int count, float epsilon, Layer.Type type, LogLevel logLevel)
    {
        CheckApproximately(X, Y, count, epsilon, type.ToString(), logLevel);
    }

    static public void CheckApproximately(Tensor X, Tensor Y, int count, float epsilon, string opName, LogLevel logLevel)
    {
        if (!X.Approximately(Y, epsilon, count))
        {
            string mainLogMessage = $"Tensors not equal after {opName}";
            if (logLevel == LogLevel.Error)
                D.LogError(mainLogMessage);
            else
                D.LogWarning(mainLogMessage);

            D.Log("First: " + X.shape);
            D.Log("Second:" + Y.shape);

            if (count < 0)
                count = X.channels * X.width * 2;
            X.PrintDataPart(count);
            Y.PrintDataPart(count);
        }
        if (X.tensorOnDevice != Y.tensorOnDevice)
            Y.Dispose();
    }
}


} // namespace Unity.Barracuda
