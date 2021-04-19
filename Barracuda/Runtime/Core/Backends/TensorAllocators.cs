using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq; // ToList()

using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Profiling;

namespace Unity.Barracuda {


internal class TensorOperatorNewAllocator : ITensorAllocator
{
    private List<Tensor> m_AllocatedTensors = new List<Tensor>();
    private HashSet<ITensorData> m_AllocatedBuffers = new HashSet<ITensorData>();

    public TensorOperatorNewAllocator()
    {
    }

    ~TensorOperatorNewAllocator()
    {
        Dispose();
    }

    public virtual Tensor Alloc(TensorShape shape)
    {
        var newTensor = new Tensor(shape, this);
        newTensor.name = "untitled";
        m_AllocatedTensors.Add(newTensor);
        return newTensor;
    }

    public virtual Tensor Alloc(TensorShape shape, ITensorData buffer)
    {
        var newTensor = new Tensor(shape, buffer, this);
        newTensor.name = "untitled";
        m_AllocatedTensors.Add(newTensor);
        m_AllocatedBuffers.Add(buffer);
        return newTensor;
    }

    public virtual void Release(Tensor tensor, bool calledFromTensorDispose)
    {
    }

    public virtual void MoveToDevice(Tensor tensor, ITensorData newBuffer, ITensorData oldBuffer, bool disposeDetachedBufferHint)
    {
        if (newBuffer != null)
            m_AllocatedBuffers.Add(newBuffer);
    }

    public virtual void Reset(bool keepCachedMemory)
    {
        Dispose();
    }

    public virtual void WaiveOwnership(Tensor tensor)
    {
        tensor.DetachFromDevice();
        m_AllocatedTensors.Remove(tensor);
        m_AllocatedBuffers.Remove(tensor.tensorOnDevice);
    }

    public virtual void Dispose()
    {
        foreach (var tensor in m_AllocatedTensors)
            tensor.Dispose();
        foreach (var buf in m_AllocatedBuffers)
            buf.Dispose();
        m_AllocatedTensors.Clear();
        m_AllocatedBuffers.Clear();
    }

    public long busyBytes
    { get {
        long bytes = 0;
        foreach(var tensor in m_AllocatedTensors)
            bytes += tensor.length * sizeof(float);
        return bytes;
    } }
    public long freeBytes
    { get {
        return 0;
    } }
    public long totalBytes
    { get {
        return busyBytes + freeBytes;
    } }
    public override string ToString()
    {
        return "Total allocated: " + totalBytes;
    }
}

// @TODO: reduce code duplication between TensorCachingByShapeAllocator and TensorCachingAllocator
internal class TensorCachingByShapeAllocator : ITensorAllocator
{
    struct Entry
    {
        public TensorShape shape;
        public ITensorData buffer;
    }
    // multi-value Dictionary<TensorShape, Entry*> implemented via
    // pair of m_FreeTensorByShape and m_FreeTensors
    private Dictionary<TensorShape, LinkedListNode<Entry>> m_FreeBufferByShape = new Dictionary<TensorShape, LinkedListNode<Entry>>();
    private LinkedList<Entry> m_FreeBuffers = new LinkedList<Entry>();
    private Dictionary<Tensor, ITensorData> m_BusyTensors = new Dictionary<Tensor, ITensorData>();
    private Dictionary<ITensorData, int> m_SharedBuffers = new Dictionary<ITensorData, int>();

    public TensorCachingByShapeAllocator()
    {
    }

    ~TensorCachingByShapeAllocator()
    {
        Dispose();
    }

    protected void AddRef(ITensorData buffer)
    {
        if (buffer == null)
            return;

        var sharedBufferCount = 0;
        m_SharedBuffers.TryGetValue(buffer, out sharedBufferCount);
        m_SharedBuffers[buffer] = sharedBufferCount + 1;
    }

    protected void DecRef(ITensorData buffer, Action<ITensorData> onLastRef = null)
    {
        if (buffer == null)
            return;

        Assert.IsTrue(m_SharedBuffers.ContainsKey(buffer));
        Assert.IsTrue(m_SharedBuffers[buffer] > 0);
        if (--m_SharedBuffers[buffer] > 0)
            return;

        m_SharedBuffers.Remove(buffer);

        if (onLastRef != null)
            onLastRef(buffer);
    }

    protected void AdoptFreeBuffer(TensorShape shape, ITensorData buffer)
    {
        // code below automatically covers handles edge-case (2)
        // by adopting tensor's with the new ITensorData into m_FreeTensors/m_FreeTensorByShape
        var newEntry = new Entry { shape = shape, buffer = buffer };
        LinkedListNode<Entry> node;
        if (m_FreeBufferByShape.TryGetValue(newEntry.shape, out node))
        {
            m_FreeBuffers.AddAfter(node, newEntry);
        }
        else
        {
            var newNode = m_FreeBuffers.AddLast(newEntry);
            m_FreeBufferByShape.Add(newEntry.shape, newNode);
        }
    }

    public virtual Tensor Alloc(TensorShape shape)
    {
        Profiler.BeginSample("Barracuda.ShapeAllocator.Alloc");
        var name = "untitled";

        LinkedListNode<Entry> node;
        if (m_FreeBufferByShape.TryGetValue(shape, out node))
        {
            Assert.AreEqual(node.Value.shape, shape);

            // advance dictionary to the next Tensor with the same shape, if available
            if (node.Next != null && node.Next.Value.shape == shape)
                m_FreeBufferByShape[shape] = node.Next;
            else
                m_FreeBufferByShape.Remove(shape);

            var buffer = node.Value.buffer;
            buffer?.Reserve(shape.length);

            var tensor = new Tensor(shape, buffer, this); // @TODO: reuse Tensor instances
            tensor.name = name;

            m_FreeBuffers.Remove(node);
            m_BusyTensors.Add(tensor, buffer);
            AddRef(buffer);

            Assert.AreEqual(tensor.shape, shape);
            Profiler.EndSample();
            return tensor;
        }

        var newTensor = new Tensor(shape, this);
        newTensor.name = name;
        m_BusyTensors.Add(newTensor, newTensor.tensorOnDevice);
        AddRef(newTensor.tensorOnDevice);

        Profiler.EndSample();
        return newTensor;
    }

    public virtual Tensor Alloc(TensorShape shape, ITensorData buffer)
    {
        Profiler.BeginSample("Barracuda.ShapeAllocator.Alloc");
        var name = "untitled";

        var tensor = new Tensor(shape, buffer, this); // @TODO: reuse Tensor instances
        tensor.name = name;
        m_BusyTensors.Add(tensor, buffer);
        AddRef(buffer);

        Profiler.EndSample();
        return tensor;
    }

    public virtual void Release(Tensor tensor, bool calledFromTensorDispose)
    {
        Profiler.BeginSample("Barracuda.ShapeAllocator.Release");
        Assert.AreEqual(tensor.allocator, this);

        var detachedBuffer = tensor.Invalidate(); // calls MoveToDevice(newBuffer=null)

        if (!m_BusyTensors.ContainsKey(tensor))
        {
            if (detachedBuffer == null)
                return;

            foreach (var freeEntry in m_FreeBuffers)
                if (freeEntry.buffer == detachedBuffer)
                    return;

            // some operations can create new Tensor and reassign ITensorData to it
            foreach (var busyEntry in m_BusyTensors)
                if (busyEntry.Value == detachedBuffer)
                    return; // we have at least another instance ITensorData in m_BusyTensors, nothing to realease
        }

        Assert.IsTrue(m_BusyTensors.ContainsKey(tensor));
        m_BusyTensors.Remove(tensor);
        Profiler.EndSample();
    }

    public virtual void MoveToDevice(Tensor tensor, ITensorData newBuffer, ITensorData oldBuffer, bool disposeDetachedBufferHint)
    {
        if (newBuffer == oldBuffer)
            return;

        Assert.AreEqual(tensor.allocator, this);
        Assert.IsTrue(m_BusyTensors.ContainsKey(tensor));
        m_BusyTensors[tensor] = newBuffer;

        AddRef(newBuffer);
        DecRef(oldBuffer,
            (freeBuffer) => {
                if (disposeDetachedBufferHint)
                    freeBuffer.Dispose();
                else
                    AdoptFreeBuffer(tensor.shape, freeBuffer);
                });
    }

    public virtual void Cast(Tensor tensor, ITensorData newBuffer, ITensorData oldBuffer)
    {
        if (newBuffer == oldBuffer)
            return;

        Assert.AreEqual(tensor.allocator, this);
        Assert.IsTrue(m_BusyTensors.ContainsKey(tensor));
        m_BusyTensors[tensor] = newBuffer;

        AddRef(newBuffer);
        DecRef(oldBuffer);
    }

    public virtual void Reset(bool keepCachedMemory)
    {
        Profiler.BeginSample("Barracuda.ShapeAllocator.Reset");

        if (!keepCachedMemory)
            Dispose();

        foreach(var tensor in m_BusyTensors.Keys.ToList())
            Release(tensor, false);

        Assert.AreEqual(m_BusyTensors.Count, 0);
        Assert.AreEqual(m_SharedBuffers.Count, 0);

        Profiler.EndSample();
    }

    public virtual void WaiveOwnership(Tensor tensor)
    {
        Assert.AreEqual(tensor.allocator, this);
        Assert.IsTrue(m_BusyTensors.ContainsKey(tensor));
        m_BusyTensors.Remove(tensor);

        var buffer = tensor.tensorOnDevice;
        if (buffer == null)
            return;

        Profiler.BeginSample("Barracuda.ShapeAllocator.WaiveOwnership");

        int sharedCount = 0;
        m_SharedBuffers.TryGetValue(buffer, out sharedCount);
        if (sharedCount > 1)
        {
            var patchBusyTensors = new List<Tensor>();
            foreach (var busyEntry in m_BusyTensors)
                if (busyEntry.Value == buffer)
                    patchBusyTensors.Add(busyEntry.Key);

            Assert.AreEqual(sharedCount - 1, patchBusyTensors.Count);

            foreach (var busyTensor in patchBusyTensors)
            {
                Assert.AreEqual(m_BusyTensors[busyTensor], buffer);

                var oldBuffer = busyTensor.DetachFromDevice(false);
                var newBuffer = busyTensor.tensorOnDevice;
                Assert.IsTrue(oldBuffer == buffer);
                Assert.IsTrue(newBuffer != buffer);
                m_BusyTensors[busyTensor] = newBuffer;
                AddRef(newBuffer);
            }
        }

        // Assert no references to tensor are left owned by allocator
        Assert.IsTrue(m_SharedBuffers[buffer] == 1);
        m_SharedBuffers.Remove(buffer);
        foreach (var freeEntry in m_FreeBuffers)
        {
            Assert.IsTrue(freeEntry.buffer != buffer);
        }
        foreach(var busyEntry in m_BusyTensors)
        {
            Assert.IsTrue(busyEntry.Key != tensor);
            Assert.IsTrue(busyEntry.Value != buffer);
        }

        Profiler.EndSample();
    }

    public virtual void Dispose()
    {
        m_FreeBufferByShape.Clear();
        foreach(var tensor in m_BusyTensors.Keys.ToList())
            Release(tensor, false);
        foreach (var entry in m_FreeBuffers)
            entry.buffer?.Dispose();

        m_BusyTensors.Clear();
        m_FreeBuffers.Clear();
        m_SharedBuffers.Clear();
    }

    public long busyBytes
    { get {
        long bytes = 0;
        foreach(var tensor in m_BusyTensors.Keys)
            bytes += tensor.length * sizeof(float);
        return bytes;
    } }
    public long freeBytes
    { get {
        long bytes = 0;
        foreach(var entry in m_FreeBuffers)
            bytes += entry.shape.length * sizeof(float);
        return bytes;
    } }
    public long totalBytes
    { get {
        return busyBytes + freeBytes;
    } }
    public override string ToString()
    {
        return "Total allocated: " + totalBytes + " busy: " + busyBytes;
    }
}

/// <summary>
/// Caching `Tensor` allocator
/// </summary>
public class TensorCachingAllocator : ITensorAllocator
{
    struct Entry
    {
        public int size;
        public ITensorData buffer;
        public bool free;
    }
    // Sorted by size array of ITensorData
    private List<Entry> m_AllocatedBuffers = new List<Entry>();
    private Dictionary<Tensor, ITensorData> m_BusyTensors = new Dictionary<Tensor, ITensorData>();
    private Dictionary<ITensorData, int> m_SharedBuffers = new Dictionary<ITensorData, int>();

    private Action<ITensorData> disposeAllocatedBufferDelegate;
    private Action<ITensorData> adoptFreeBufferDelegate;

    /// <summary>
    /// Create `TensorCachingAllocator`
    /// </summary>
    public TensorCachingAllocator()
    {
        disposeAllocatedBufferDelegate = DisposeAllocatedBuffer;
        adoptFreeBufferDelegate = AdoptFreeBuffer;
    }

    /// <summary>
    /// Finalizer
    /// </summary>
    ~TensorCachingAllocator()
    {
        Dispose();
    }

    static internal int GetAllocationMaxCount(Tensor tensor)
    {
        return (tensor.tensorOnDevice != null) ?
            tensor.tensorOnDevice.maxCapacity:
            tensor.length;
    }

    internal void AddRef(ITensorData buffer)
    {
        if (buffer == null)
            return;

        var sharedBufferCount = 0;
        m_SharedBuffers.TryGetValue(buffer, out sharedBufferCount);
        m_SharedBuffers[buffer] = sharedBufferCount + 1;
    }

    internal void DecRef(ITensorData buffer, Action<ITensorData> onLastRef = null)
    {
        if (buffer == null)
            return;

        Assert.IsTrue(m_SharedBuffers.ContainsKey(buffer));
        Assert.IsTrue(m_SharedBuffers[buffer] > 0);
        if (--m_SharedBuffers[buffer] > 0)
            return;

        m_SharedBuffers.Remove(buffer);

        if (onLastRef != null)
            onLastRef(buffer);
    }

    internal void AdoptFreeBuffer(ITensorData buffer)
    {
        // insert into the sorted array
        var size = buffer.maxCapacity;
        var newEntry = new Entry { size = size, buffer = buffer, free = true };
        bool found = false;
        for (int i = 0; !found && i < m_AllocatedBuffers.Count; ++i)
        {
            var entry = m_AllocatedBuffers[i];
            if (buffer == entry.buffer)
            {
                Assert.IsTrue(!entry.free);
                entry.free = true;
                m_AllocatedBuffers[i] = entry;
                Assert.IsTrue(m_AllocatedBuffers[i].free);
                found = true;
            }
            if (size < entry.size)
            {
                m_AllocatedBuffers.Insert(i, newEntry);
                Assert.IsTrue(m_AllocatedBuffers[i].size < m_AllocatedBuffers[i + 1].size);
                found = true;
            }
        }

        if (!found)
            m_AllocatedBuffers.Add(newEntry);
    }

    internal void DisposeAllocatedBuffer(ITensorData buffer)
    {
        for (int i = m_AllocatedBuffers.Count - 1; i >= 0; i--)
            if (m_AllocatedBuffers[i].buffer == buffer)
                m_AllocatedBuffers.RemoveAt(i);
        buffer.Dispose();
    }

    /// <inheritdoc/>
    public virtual Tensor Alloc(TensorShape shape)
    {
        Profiler.BeginSample("Barracuda.SizeAllocator.Alloc");
        var name = "untitled";

        for (int i = 0; i < m_AllocatedBuffers.Count; ++i)
        {
            var entry = m_AllocatedBuffers[i];
            if (entry.size >= shape.length && entry.free)
            {
                entry.free = false;
                m_AllocatedBuffers[i] = entry;

                var buffer = entry.buffer;
                buffer?.Reserve(shape.length);

                var tensor = new Tensor(shape, buffer, this); // @TODO: reuse Tensor instances
                tensor.name = name;

                m_BusyTensors.Add(tensor, tensor.tensorOnDevice);
                AddRef(tensor.tensorOnDevice);

                Profiler.EndSample();
                return tensor;
            }
        }


        var newTensor = new Tensor(shape, this);
        newTensor.name = name;
        m_BusyTensors.Add(newTensor, newTensor.tensorOnDevice);
        AddRef(newTensor.tensorOnDevice);

        Profiler.EndSample();
        return newTensor;
    }

    /// <inheritdoc/>
    public virtual Tensor Alloc(TensorShape shape, ITensorData buffer)
    {
        Profiler.BeginSample("Barracuda.SizeAllocator.Alloc");
        var name = "untitled";

        var tensor = new Tensor(shape, buffer, this); // @TODO: reuse Tensor instances
        tensor.name = name;
        m_BusyTensors.Add(tensor, tensor.tensorOnDevice);
        AddRef(tensor.tensorOnDevice);

        Profiler.EndSample();
        return tensor;
    }

    /// <inheritdoc/>
    public virtual void Release(Tensor tensor, bool calledFromTensorDispose)
    {
        Profiler.BeginSample("Barracuda.SizeAllocator.Release");
        Assert.AreEqual(tensor.allocator, this);

        var detachedBuffer = tensor.Invalidate(); // calls MoveToDevice(newBuffer=null)

        if (!m_BusyTensors.ContainsKey(tensor))
        {
            if (detachedBuffer == null)
                return;

            foreach (var entry in m_AllocatedBuffers)
                if (entry.buffer == detachedBuffer && entry.free)
                    return;

            // some operations can create new Tensor and reassign ITensorData to it
            foreach (var busyEntry in m_BusyTensors)
                if (busyEntry.Value == detachedBuffer)
                    return; // we have original ITensorData in m_BusyTensors, nothing to realease
        }

        Assert.IsTrue(m_BusyTensors.ContainsKey(tensor));
        m_BusyTensors.Remove(tensor);

        Profiler.EndSample();
    }

    /// <inheritdoc/>
    public virtual void MoveToDevice(Tensor tensor, ITensorData newBuffer, ITensorData oldBuffer, bool disposeDetachedBufferHint)
    {
        if (newBuffer == oldBuffer)
            return;

        Assert.AreEqual(tensor.allocator, this);
        Assert.IsTrue(m_BusyTensors.ContainsKey(tensor));
        m_BusyTensors[tensor] = newBuffer;

        AddRef(newBuffer);

        if (disposeDetachedBufferHint)
            DecRef(oldBuffer, disposeAllocatedBufferDelegate);
        else
            DecRef(oldBuffer, adoptFreeBufferDelegate);
    }

    /// <summary>
    /// Replace old buffer with new buffer
    /// </summary>
    /// <param name="tensor">owning `Tensor`</param>
    /// <param name="newBuffer">new buffer</param>
    /// <param name="oldBuffer">old buffer</param>
    public virtual void Cast(Tensor tensor, ITensorData newBuffer, ITensorData oldBuffer)
    {
        if (newBuffer == oldBuffer)
            return;

        Assert.AreEqual(tensor.allocator, this);
        Assert.IsTrue(m_BusyTensors.ContainsKey(tensor));
        m_BusyTensors[tensor] = newBuffer;

        AddRef(newBuffer);
        DecRef(oldBuffer);
    }

    /// <inheritdoc/>
    public virtual void Reset(bool keepCachedMemory)
    {
        Profiler.BeginSample("Barracuda.SizeAllocator.Reset");

        if (!keepCachedMemory)
            Dispose();

        foreach(var tensor in m_BusyTensors.Keys.ToList())
            Release(tensor, false);

        Assert.AreEqual(m_BusyTensors.Count, 0);
        Assert.AreEqual(m_SharedBuffers.Count, 0);

        foreach(var buf in m_AllocatedBuffers)
            Assert.IsTrue(buf.free);

        Profiler.EndSample();
    }

    /// <inheritdoc/>
    public virtual void WaiveOwnership(Tensor tensor)
    {
        Assert.AreEqual(tensor.allocator, this);
        Assert.IsTrue(m_BusyTensors.ContainsKey(tensor));
        m_BusyTensors.Remove(tensor);

        var buffer = tensor.tensorOnDevice;
        if (buffer == null)
            return;

        Profiler.BeginSample("Barracuda.SizeAllocator.WaiveOwnership");

        int sharedCount = 0;
        m_SharedBuffers.TryGetValue(buffer, out sharedCount);
        if (sharedCount > 1)
        {
            var patchBusyTensors = new List<Tensor>();
            foreach (var busyEntry in m_BusyTensors)
                if (busyEntry.Value == buffer)
                    patchBusyTensors.Add(busyEntry.Key);

            Assert.AreEqual(sharedCount - 1, patchBusyTensors.Count);

            foreach (var busyTensor in patchBusyTensors)
            {
                Assert.AreEqual(m_BusyTensors[busyTensor], buffer);

                var oldBuffer = busyTensor.DetachFromDevice(false);
                var newBuffer = busyTensor.tensorOnDevice;
                Assert.IsTrue(oldBuffer == buffer);
                Assert.IsTrue(newBuffer != buffer);
                m_BusyTensors[busyTensor] = newBuffer;
                AddRef(newBuffer);
            }
        }

        // Assert no references to tensor are left owned by allocator
        Assert.IsTrue(m_SharedBuffers[buffer] == 1);
        m_SharedBuffers.Remove(buffer);
        foreach (var freeEntry in m_AllocatedBuffers)
        {
            Assert.IsTrue(freeEntry.buffer != buffer);
        }
        foreach(var busyEntry in m_BusyTensors)
        {
            Assert.IsTrue(busyEntry.Key != tensor);
            Assert.IsTrue(busyEntry.Value != buffer);
        }

        Profiler.EndSample();
    }

    /// <summary>
    /// Dispose all allocated buffers
    /// </summary>
    public virtual void Dispose()
    {
        foreach(var tensor in m_BusyTensors.Keys.ToList())
            Release(tensor, false);
        foreach (var entry in m_AllocatedBuffers)
            entry.buffer?.Dispose();

        m_BusyTensors.Clear();
        m_AllocatedBuffers.Clear();
        m_SharedBuffers.Clear();
    }

    /// <summary>
    /// Busy bytes
    /// </summary>
    public long busyBytes
    { get {
        long bytes = 0;
        foreach(var tensor in m_BusyTensors.Keys)
            bytes += GetAllocationMaxCount(tensor)  * sizeof(float);
        return bytes;
    } }

    /// <summary>
    /// Free bytes
    /// </summary>
    public long freeBytes
    { get {
        long bytes = 0;
        foreach(var entry in m_AllocatedBuffers)
            if (entry.free)
                bytes += entry.size * sizeof(float);
        return bytes;
    } }

    /// <summary>
    /// Total bytes
    /// </summary>
    public long totalBytes
    { get {
        return busyBytes + freeBytes;
    } }

    /// <summary>
    /// Summary
    /// </summary>
    /// <returns>summary</returns>
    public override string ToString()
    {
        return "Total allocated: " + totalBytes + " busy: " + busyBytes;
    }
}

} // namespace Unity.Barracuda
