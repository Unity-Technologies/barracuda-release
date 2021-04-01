using System;
using System.Collections.Generic;

namespace Unity.Barracuda
{

/// <summary>
/// Utility class to help with disposing tensors automatically:
/// Example usage:
/// using (var td = new TensorScope())
/// {
///      TensorScope.F _ = td._; // Function pointer to have less "visual noise" when making use of this
///      var t1 = _(m_Ops.<Op>(...));
///      var t2 = _(m_Ops.<Op>(...));
///      var t3 = _(m_Ops.<Op>(...));
///      ...
/// }
///
/// or alternatively it can depend on another tensor being disposed
///
/// var td = new TensorScope();
/// {
///      TensorScope.F _ = td._; // Function pointer to have less "visual noise" when making use of this
///      var t1 = _(m_Ops.<Op>(...));
///      var t2 = _(m_Ops.<Op>(...));
///      var t3 = _(m_Ops.<Op>(...));g
///      ...
/// }
/// O = m_Ops.<Op>(...);
/// td.DependentOn(O);
/// </summary>
class TensorScope : IDisposable
{
    public delegate Tensor F(Tensor tensor);
    HashSet<Tensor> m_Tensors = new HashSet<Tensor>();
    Tensor m_DependentOnTensor;

    public Tensor _(Tensor tensor)
    {
        m_Tensors.Add(tensor);
        return tensor;
    }

    public bool Remove(Tensor tensor)
    {
        return m_Tensors.Remove(tensor);
    }

    public void DependentOn(Tensor tensor)
    {
        Tensor.tensorDisposed -= DependentDispose; // Prevents multiple subscribes
        m_DependentOnTensor = tensor;
        Tensor.tensorDisposed += DependentDispose;
    }

    void DependentDispose(Tensor tensor)
    {
        if (m_DependentOnTensor == tensor)
        {
            m_DependentOnTensor = null;
            Tensor.tensorDisposed -= DependentDispose;
            Dispose();
        }
    }

    public void Dispose()
    {
        foreach (Tensor t in m_Tensors)
            t.Dispose();
        m_Tensors.Clear();
        m_DependentOnTensor = null;
    }
}

}
