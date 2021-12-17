using System;
using System.Runtime.ConstrainedExecution;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;

namespace Unity.Barracuda
{

///see https://referencesource.microsoft.com/#mscorlib/system/runtime/interopservices/safehandle.cs
internal class NativeMemorySafeHandle : SafeHandle
{
    public readonly Allocator m_AllocatorLabel;

    [ReliabilityContract(Consistency.WillNotCorruptState, Cer.MayFail)]
    public unsafe NativeMemorySafeHandle(long size, int alignment, Allocator allocator) : base(IntPtr.Zero, true)
    {
        m_AllocatorLabel = allocator;
        if (size > 0)
            SetHandle((IntPtr)UnsafeUtility.Malloc(size, alignment, allocator));
    }

    public override bool IsInvalid {
        get { return handle == IntPtr.Zero; }
    }

    [ReliabilityContract(Consistency.WillNotCorruptState, Cer.Success)]
    protected override unsafe bool ReleaseHandle()
    {
        UnsafeUtility.Free((void*)handle, m_AllocatorLabel);
        return true;
    }
}

internal class PinnedMemorySafeHandle : SafeHandle
{
    private readonly GCHandle m_GCHandle;

    [ReliabilityContract(Consistency.WillNotCorruptState, Cer.MayFail)]
    public PinnedMemorySafeHandle(object managedObject) : base(IntPtr.Zero, true)
    {
        m_GCHandle = GCHandle.Alloc(managedObject, GCHandleType.Pinned);
        IntPtr pinnedPtr = m_GCHandle.AddrOfPinnedObject();
        SetHandle(pinnedPtr);
    }

    public override bool IsInvalid {
        get { return handle == IntPtr.Zero; }
    }

    [ReliabilityContract(Consistency.WillNotCorruptState, Cer.Success)]
    protected override bool ReleaseHandle()
    {
        m_GCHandle.Free();
        return true;
    }
}

/// <summary>
/// A BarracudaArrayFromManagedArray exposes a buffer of managed memory as if it was native memory (by pinning it).
/// </summary>
public class BarracudaArrayFromManagedArray : BarracudaArray
{
    private readonly int m_PinnedMemoryByteOffset;

    public BarracudaArrayFromManagedArray(float[] srcData, int srcOffset = 0) : this(srcData, srcOffset, sizeof(float), DataType.Float, srcData.Length-srcOffset)
    {
    }

    public BarracudaArrayFromManagedArray(byte[] srcData, int srcOffset, DataType destType, int numDestElement) : this(srcData, srcOffset, sizeof(byte), destType, numDestElement)
    {
    }

    private unsafe BarracudaArrayFromManagedArray(Array srcData, int srcElementOffset, int srcElementSize, DataType destElementType, int numDestElement) : base(new PinnedMemorySafeHandle(srcData), destElementType, numDestElement)
    {
        m_PinnedMemoryByteOffset = srcElementSize * srcElementOffset;

        //Safety checks
        int requiredAlignment = DataAlignmentSize(destElementType);
        int srcLenghtInByte = (srcData.Length - srcElementOffset) * srcElementSize;
        int dstLenghtInByte = numDestElement * DataItemSize(destElementType);
        IntPtr pinnedPtrWithOffset = (IntPtr) base.RawPtr + m_PinnedMemoryByteOffset;
        if (srcElementOffset > srcData.Length)
            throw new ArgumentOutOfRangeException(nameof (srcElementOffset), "SrcElementOffset must be <= srcData.Length");
        if (dstLenghtInByte > srcLenghtInByte)
            throw new ArgumentOutOfRangeException(nameof (numDestElement), "NumDestElement too big for srcData and srcElementOffset");

        if (pinnedPtrWithOffset.ToInt64() % requiredAlignment != 0)
            throw new InvalidOperationException($"The BarracudaArrayFromManagedArray source ptr (including offset) need to be aligned on {requiredAlignment} bytes for the data to be express as {destElementType}.");

        var neededSrcPaddedLengthInByte = LengthWithPaddingForGPUCopy(destElementType, numDestElement) * DataItemSize(destElementType);
        if (srcLenghtInByte < neededSrcPaddedLengthInByte)
            throw new InvalidOperationException($"The BarracudaArrayFromManagedArray source ptr (including offset) is to small to account for extra padding needing for type {destElementType}.");
    }

    public override unsafe void* RawPtr => (byte*) base.RawPtr + m_PinnedMemoryByteOffset;
}

public enum DataType
{
    Float,
    Half
}

/// <summary>
/// A BarracudaArray exposes a buffer of native memory to managed code.
/// </summary>
public class BarracudaArray : IDisposable
{
    protected readonly SafeHandle m_SafeHandle;
    private readonly Allocator m_Allocator;
    private readonly int m_Length;
    private readonly DataType m_DataType;

    #region helpers
    public static int DataItemSize(DataType dataType)
    {
        if (dataType == DataType.Float)
            return  UnsafeUtility.SizeOf<float>();
        if (dataType == DataType.Half)
            return  UnsafeUtility.SizeOf<half>();

        throw new NotImplementedException($"Type {dataType} not supported.");
    }

    public static int DataAlignmentSize(DataType dataType)
    {
        if (dataType == DataType.Float)
            return  UnsafeUtility.AlignOf<float>();
        if (dataType == DataType.Half)
            return  UnsafeUtility.AlignOf<uint>();

        throw new NotImplementedException($"Type {dataType} not supported.");
    }

    public static int LengthWithPaddingForGPUCopy(DataType dataType, int length)
    {
        if (dataType == DataType.Float)
            return  length;
        if (dataType == DataType.Half)
            return length + (length % 2);

        throw new NotImplementedException($"Type {dataType} not supported.");
    }

    private void CheckElementAccess(DataType dataType, long index)
    {
        //Disabled by default for performance reasons.
        #if ENABLE_BARRACUDA_DEBUG
        if (Disposed)
            throw new InvalidOperationException("The BarracudaArray was disposed.");
        if (index <0 || index >= m_Length)
            throw new IndexOutOfRangeException($"Accessing BarracudaArray of length {m_Length} at index {index}, data type is {m_DataType}.");
        if (dataType != m_DataType)
            throw new InvalidOperationException($"Accessing BarracudaArray of data type {m_DataType} as if it was {dataType}.");
        #endif
    }
    #endregion

    protected BarracudaArray(SafeHandle safeHandle, DataType dataType, int dataLength)
    {
        m_DataType = dataType;
        m_Length = dataLength;
        m_SafeHandle = safeHandle;
        m_Allocator = Allocator.Persistent;
    }

    public BarracudaArray(int length, DataType dataType = DataType.Float, Allocator allocator = Allocator.Persistent)
    {
        if (!UnsafeUtility.IsValidAllocator(allocator))
            throw new InvalidOperationException("The BarracudaArray should use a valid allocator.");
        if (length < 0)
            throw new ArgumentOutOfRangeException(nameof (length), "Length must be >= 0");

        m_DataType = dataType;
        m_Length = length;
        m_SafeHandle = new NativeMemorySafeHandle(LengthWithPaddingForGPUCopy(m_DataType, m_Length) * DataItemSize(dataType), DataAlignmentSize(dataType), allocator);
        m_Allocator = allocator;
    }

    public unsafe void ZeroMemory()
    {
        var numByteToClear = LengthWithPaddingForGPUCopy(m_DataType, m_Length) * DataItemSize(m_DataType);
        UnsafeUtility.MemClear(RawPtr, numByteToClear);
    }

    public virtual void Dispose()
    {
        m_SafeHandle.Dispose();
    }

    #region properties
    public DataType Type => m_DataType;

    public int SizeOfType => DataItemSize(m_DataType);

    public int Length => m_Length;
    public long LongLength => m_Length;

    public virtual unsafe void* RawPtr
    {
        get
        {
            if (Disposed)
                throw new InvalidOperationException("The BarracudaArray was disposed.");
            return (void*)m_SafeHandle.DangerousGetHandle();
        }
    }

    public bool Disposed => m_SafeHandle.IsClosed;

    #endregion

    #region indexers and single access accessor

    public unsafe float* AddressAt(long index)
    {
        Assert.AreEqual(DataType.Float, m_DataType);
        return (float*) RawPtr + index;
    }

    public unsafe half* HalfAddressAt(long index)
    {
        Assert.AreEqual(DataType.Half, m_DataType);
        return (half*) RawPtr + index;
    }

    public unsafe void* RawAddressAt(long index)
    {
        if (m_DataType == DataType.Half)
            return HalfAddressAt(index);
        else
            return AddressAt(index);
    }

    public float this[long index]
    {
        get => this[(int)index];
        set => this[(int)index] = value;
    }
    public float this[int index]
    {
        get
        {
            switch (m_DataType)
            {
                case DataType.Float:
                    return GetFloat(index);
                default:
                    return GetHalf(index);
            }
        }
        set
        {
            switch (m_DataType)
            {
                case DataType.Float:
                    SetFloat(index, value);
                    break;
                default:
                    SetHalf(index, (half) value);
                    break;
            }
        }
    }

    public unsafe float GetFloat(int index)
    {
        CheckElementAccess(DataType.Float, index);
        return UnsafeUtility.ReadArrayElement<float>(RawPtr, index);
    }
    public unsafe half GetHalf(int index)
    {
        CheckElementAccess(DataType.Half, index);
        return UnsafeUtility.ReadArrayElement<half>(RawPtr, index);
    }
    public unsafe void SetFloat(int index, float value)
    {
        CheckElementAccess(DataType.Float, index);
        UnsafeUtility.WriteArrayElement<float>(RawPtr, index, value);
    }
    public unsafe void SetHalf(int index, half value)
    {
        CheckElementAccess(DataType.Half, index);
        UnsafeUtility.WriteArrayElement<half>(RawPtr, index, value);
    }
    #endregion

    #region copy to other memory containers
    public void UploadToComputeBuffer(ComputeBuffer buffer)
    {
        UploadToComputeBuffer(buffer, 0, 0, m_Length);
    }

    public unsafe void UploadToComputeBuffer(ComputeBuffer buffer, int elementStartIndex, int computeBufferStartIndex, int numElementToCopy)
    {
        if (numElementToCopy == 0)
            return;
        if (m_DataType == DataType.Float)
        {
            NativeArray<float> nativeArray = NativeArrayUnsafeUtility.ConvertExistingDataToNativeArray<float>(RawPtr, m_Length, m_Allocator);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            NativeArrayUnsafeUtility.SetAtomicSafetyHandle(ref nativeArray, AtomicSafetyHandle.Create());
#endif
            buffer.SetData(nativeArray, elementStartIndex, computeBufferStartIndex, numElementToCopy);
        }
        else if (m_DataType == DataType.Half)
        {
            if (elementStartIndex % 2 == 1 || computeBufferStartIndex % 2 == 1)
                throw new ArgumentException($"For half buffer type nativeBufferStartIndex and computeBufferStartIndex should be modulo of 2.");

            numElementToCopy += numElementToCopy % 2;

            int uintBufferViewLength = LengthWithPaddingForGPUCopy(m_DataType, m_Length) / 2;
            NativeArray<uint> nativeArray = NativeArrayUnsafeUtility.ConvertExistingDataToNativeArray<uint>(RawPtr, uintBufferViewLength, m_Allocator);
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            NativeArrayUnsafeUtility.SetAtomicSafetyHandle(ref nativeArray, AtomicSafetyHandle.Create());
#endif
            //TODO fp16 should computeBufferStartIndex be expressed in half or uint? For now in half
            buffer.SetData(nativeArray, elementStartIndex/2, computeBufferStartIndex/2, numElementToCopy/2);
        }
        else
        {
            throw new NotImplementedException($"Type {m_DataType} not supported.");
        }
    }

    /// <summary>
    /// Warning, this return a copy! Do not use to modify a BarracudaArray
    /// </summary>
    public static implicit operator float[](BarracudaArray barracudaArray)
    {
        var floatArray = new float[barracudaArray.Length];
        Copy(barracudaArray, 0, floatArray, 0, barracudaArray.Length);
        return floatArray;
    }

    public void CopyTo(BarracudaArray dst, int dstOffset)
    {
        Copy(this, 0, dst, dstOffset, Length);
    }

    public void CopyTo(BarracudaArray dst, long dstOffset)
    {
        Copy(this, 0, dst, (int)dstOffset, Length);
    }

    public static void Copy(BarracudaArray sourceArray, BarracudaArray destinationArray, int length = -1)
    {
        Copy(sourceArray, 0, destinationArray, 0, length);
    }

    public static void Copy(float[] sourceArray, BarracudaArray destinationArray, int length = -1)
    {
        Copy(sourceArray, 0, destinationArray, 0, length);
    }

    public static unsafe void Copy(
        BarracudaArray sourceArray,
        int sourceIndex,
        BarracudaArray destinationArray,
        int destinationIndex,
        int length)
    {
        if (length < 0)
            length = sourceArray.Length;
        if (length == 0)
            return;
        if (sourceIndex+length > sourceArray.Length)
            throw new ArgumentException($"Cannot copy {length} element from sourceIndex {sourceIndex} and Barracuda array of length {sourceArray.Length}.");
        if (destinationIndex+length > destinationArray.Length)
            throw new ArgumentException($"Cannot copy {length} element to sourceIndex {destinationIndex} and Barracuda array of length {destinationArray.Length}.");

        //Same type we can do a memcopy
        if (sourceArray.m_DataType == destinationArray.m_DataType)
        {
            int itemSize = DataItemSize(sourceArray.m_DataType);
            void* srcPtr = (byte*)sourceArray.RawPtr + sourceIndex * itemSize;
            void* dstPtr = (byte*)destinationArray.RawPtr + destinationIndex * itemSize;
            UnsafeUtility.MemCpy(dstPtr, srcPtr, length * itemSize);
        }
        else//different type, we need to iterate and cast
        {
            for (var i=0; i < length; ++i)
            {
                //this will use float as intermediate/common representation
                destinationArray[destinationIndex+i] = sourceArray[sourceIndex+i];
            }
        }
    }

    public static unsafe void Copy(
        BarracudaArray sourceArray,
        int sourceIndex,
        float[] destinationArray,
        int destinationIndex,
        int length)
    {
        if (length < 0)
            length = sourceArray.Length;
        if (length == 0)
            return;
        if (sourceIndex+length > sourceArray.Length)
            throw new ArgumentException($"Cannot copy {length} element from sourceIndex {sourceIndex} and Barracuda array of length {sourceArray.Length}.");
        if (destinationIndex+length > destinationArray.Length)
            throw new ArgumentException($"Cannot copy {length} element to sourceIndex {destinationIndex} and array of length {destinationArray.Length}.");

        //Same type we can do a memcopy
        if (sourceArray.m_DataType == DataType.Float)
        {
            fixed (void* dstPtr = &destinationArray[destinationIndex])
            {
                int itemSize = DataItemSize(sourceArray.m_DataType);
                void* srcPtr = (byte*)sourceArray.RawPtr + sourceIndex * itemSize;
                UnsafeUtility.MemCpy(dstPtr, srcPtr, length * itemSize);
            }
        }
        else//different type, we need to iterate and cast
        {
            for (var i=0; i < length; ++i)
            {
                //this will use float as intermediate/common representation
                destinationArray[destinationIndex+i] = sourceArray[sourceIndex+i];
            }
        }
    }

    public static unsafe void BlockCopy(
        BarracudaArray sourceArray,
        int sourceByteOffset,
        byte[] destinationArray,
        int destinationByteOffset,
        int lengthInBytes)
    {
        int itemSize = sourceArray.SizeOfType;
        int srcLengthBytes = sourceArray.Length * itemSize;

        if (lengthInBytes == 0)
            return;
        if (lengthInBytes < 0)
            lengthInBytes = srcLengthBytes;

        if (sourceByteOffset+lengthInBytes > srcLengthBytes)
            throw new ArgumentException($"Cannot copy {lengthInBytes} bytes from sourceByteOffset {sourceByteOffset} and BarracudaArray of {srcLengthBytes} num bytes.");
        if (destinationByteOffset+lengthInBytes > destinationArray.Length)
            throw new ArgumentException($"Cannot copy {lengthInBytes} bytes to destinationByteOffset {destinationByteOffset} and byte[] array of {destinationArray.Length} num bytes.");

        fixed (void* dstPtr = &destinationArray[destinationByteOffset])
        {
            void* srcPtr = (byte*)sourceArray.RawPtr + sourceByteOffset;
            UnsafeUtility.MemCpy(dstPtr, srcPtr, lengthInBytes);
        }
    }

    public static unsafe void BlockCopy(
        byte[] sourceArray,
        int sourceByteOffset,
        BarracudaArray destinationArray,
        int destinationByteOffset,
        int lengthInBytes)
    {
        if (lengthInBytes == 0)
            return;
        if (lengthInBytes < 0)
            lengthInBytes = sourceArray.Length;

        if (sourceByteOffset+lengthInBytes > sourceArray.Length)
            throw new ArgumentException($"Cannot copy {lengthInBytes} bytes from sourceByteOffset {sourceByteOffset} and byte[] array of {sourceArray.Length} num bytes.");
        var fullDestPaddedSizeInByte = LengthWithPaddingForGPUCopy(destinationArray.Type, destinationArray.Length) * DataItemSize(destinationArray.Type);
        if (destinationByteOffset+lengthInBytes > fullDestPaddedSizeInByte)
            throw new ArgumentException($"Cannot copy {lengthInBytes} bytes to destinationByteOffset {destinationByteOffset} and byte[] array of {destinationArray.Length} num bytes.");

        void* dstPtr = (byte*)destinationArray.RawPtr + destinationByteOffset;
        fixed (void* srcPtr = &sourceArray[sourceByteOffset])
        {
            UnsafeUtility.MemCpy(dstPtr, srcPtr, lengthInBytes);
        }
    }

    public static void Copy(
        float[] sourceArray,
        int sourceIndex,
        BarracudaArray destinationArray,
        long destinationIndex,
        int length)
    {
        Copy(sourceArray, sourceIndex, destinationArray, (int)destinationIndex, length);
    }

    public static unsafe void Copy(
        float[] sourceArray,
        int sourceIndex,
        BarracudaArray destinationArray,
        int destinationIndex,
        int length)
    {
        if (length < 0)
            length = sourceArray.Length;
        if (length == 0)
            return;
        if (sourceIndex+length > sourceArray.Length)
            throw new ArgumentException($"Cannot copy {length} element from sourceIndex {sourceIndex} and Barracuda array of length {sourceArray.Length}.");
        if (destinationIndex+length > destinationArray.Length)
            throw new ArgumentException($"Cannot copy {length} element to sourceIndex {destinationIndex} and Barracuda array of length {destinationArray.Length}.");

        //Same type we can do a memcopy
        if (destinationArray.m_DataType == DataType.Float)
        {
            fixed (void* srcPtr = &sourceArray[sourceIndex])
            {
                int itemSize = DataItemSize(destinationArray.m_DataType);
                void* dstPtr = (byte*)destinationArray.RawPtr + destinationIndex * itemSize;
                UnsafeUtility.MemCpy(dstPtr, srcPtr, length * itemSize);
            }
        }
        else//different type, we need to iterate and cast
        {
            for (var i=0; i < length; ++i)
            {
                //this will use float as intermediate/common representation
                destinationArray[destinationIndex+i] = sourceArray[sourceIndex+i];
            }
        }
    }
    #endregion
}

static class BarracudaArrayExtensionHelper
{
    public static void CopyToBarracudaArray(this float[] sourceArray, BarracudaArray destinationArray, int destinationIndex)
    {
        BarracudaArray.Copy(sourceArray, 0, destinationArray, destinationIndex, sourceArray.Length);
    }

    public static void CopyToBarracudaArray(this float[] sourceArray, BarracudaArray destinationArray, long destinationIndex)
    {
        BarracudaArray.Copy(sourceArray, 0, destinationArray, (int)destinationIndex, sourceArray.Length);
    }
}


} // namespace Barracuda
