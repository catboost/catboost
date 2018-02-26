#pragma once

#include <catboost/cuda/cuda_util/kernel/fill.cuh>
#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/libs/helpers/exception.h>

namespace NKernelHost {
    template <class T>
    class TFillBufferKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<T> Buffer;
        T Value;

    public:
        TFillBufferKernel() = default;

        TFillBufferKernel(TCudaBufferPtr<T> buffer,
                          T value)
            : Buffer(buffer)
            , Value(value)
        {
        }

        Y_SAVELOAD_DEFINE(Buffer, Value);

        void Run(const TCudaStream& stream) const {
            NKernel::FillBuffer(Buffer.Get(), Value, Buffer.Size(), stream.GetStream());
        }
    };

    template <class T>
    class TMakeSequenceKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<T> Buffer;
        T Offset;

    public:
        TMakeSequenceKernel() = default;

        TMakeSequenceKernel(TCudaBufferPtr<T> ptr,
                            T offset = 0)
            : Buffer(ptr)
            , Offset(offset)
        {
        }

        Y_SAVELOAD_DEFINE(Buffer, Offset);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE(Buffer.ObjectCount() == Buffer.Size(), "MakeSequence expects single-object buffer " << Buffer.ObjectCount() << " " << Buffer.Size() << " " << Buffer.GetColumnCount() << " " << Buffer.ColumnSize());
            NKernel::MakeSequence(Offset, Buffer.Get(), Buffer.Size(), stream.GetStream());
        }
    };

    template <class T>
    class TInversePermutationKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const T> Order;
        TCudaBufferPtr<T> InverseOrder;

    public:
        TInversePermutationKernel() = default;

        TInversePermutationKernel(TCudaBufferPtr<const T> order,
                                  TCudaBufferPtr<T> inverseOrder)
            : Order(order)
            , InverseOrder(inverseOrder)
        {
        }

        Y_SAVELOAD_DEFINE(Order, InverseOrder);

        void Run(const TCudaStream& stream) const {
            NKernel::InversePermutation(Order.Get(), InverseOrder.Get(), Order.Size(), stream.GetStream());
        }
    };
}

template <typename T, class TMapping>
inline void FillBuffer(TCudaBuffer<T, TMapping>& buffer,
                       T value,
                       ui32 streamId = 0) {
    using TKernel = NKernelHost::TFillBufferKernel<T>;
    LaunchKernels<TKernel>(buffer.NonEmptyDevices(), streamId, buffer, value);
}

template <typename T, class TMapping>
inline void MakeSequence(TCudaBuffer<T, TMapping>& buffer, ui32 stream = 0) {
    using TKernel = NKernelHost::TMakeSequenceKernel<T>;
    LaunchKernels<TKernel>(buffer.NonEmptyDevices(), stream, buffer);
}

template <typename T, class TMapping>
inline void MakeSequenceWithOffset(TCudaBuffer<T, TMapping>& buffer,
                                   const NCudaLib::TDistributedObject<T>& offset,
                                   ui32 stream = 0) {
    using TKernel = NKernelHost::TMakeSequenceKernel<T>;
    LaunchKernels<TKernel>(buffer.NonEmptyDevices(), stream, buffer, offset);
}

template <class TUi32, class TMapping>
inline void InversePermutation(const TCudaBuffer<TUi32, TMapping>& order,
                               TCudaBuffer<ui32, TMapping>& inverseOrder,
                               ui32 streamId = 0) {
    using TKernel = NKernelHost::TInversePermutationKernel<ui32>;
    LaunchKernels<TKernel>(order.NonEmptyDevices(), streamId, order, inverseOrder);
}
