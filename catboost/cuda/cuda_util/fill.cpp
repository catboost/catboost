#include "fill.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_util/kernel/fill.cuh>
#include <catboost/libs/helpers/exception.h>

#include <util/generic/va_args.h>

using NCudaLib::TMirrorMapping;
using NCudaLib::TSingleMapping;
using NCudaLib::TStripeMapping;
using NKernelHost::TCudaBufferPtr;
using NKernelHost::TCudaStream;
using NKernelHost::TStatelessKernel;

// FillBuffer

namespace {
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
            NKernel::FillBuffer(Buffer.Get(), Value, Buffer.Size(), static_cast<ui32>(Buffer.GetColumnCount()), Buffer.AlignedColumnSize(), stream.GetStream());
        }
    };
}

template <typename T, typename TMapping>
static void FillBufferImpl(
    TCudaBuffer<T, TMapping>& buffer,
    std::remove_const_t<T> value,
    ui32 streamId) {
    using TKernel = TFillBufferKernel<T>;
    LaunchKernels<TKernel>(buffer.NonEmptyDevices(), streamId, buffer, value);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, TMapping)        \
    template <>                                    \
    void FillBuffer<T, TMapping>(                  \
        TCudaBuffer<T, TMapping> & buffer,         \
        std::remove_const_t<T> value,              \
        ui32 streamId) {                           \
        ::FillBufferImpl(buffer, value, streamId); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (char, TMirrorMapping),
    (i32, TMirrorMapping),
    (ui32, TMirrorMapping),
    (i64, TMirrorMapping),
    (ui64, TMirrorMapping),
    (float, TMirrorMapping),
    (double, TMirrorMapping),
    (char, TSingleMapping),
    (i32, TSingleMapping),
    (ui32, TSingleMapping),
    (i64, TSingleMapping),
    (ui64, TSingleMapping),
    (float, TSingleMapping),
    (double, TSingleMapping),
    (bool, TStripeMapping),
    (char, TStripeMapping),
    (i32, TStripeMapping),
    (ui32, TStripeMapping),
    (i64, TStripeMapping),
    (ui64, TStripeMapping),
    (float, TStripeMapping),
    (double, TStripeMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// MakeSequence, MakeSequenceWithOffset, MakeSequenceGlobal

// MakeSequence

namespace {
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
            CB_ENSURE(Buffer.ObjectCount() == Buffer.Size(), "MakeSequence expects single-object buffer " << Buffer.ObjectCount() << " " << Buffer.Size());
            NKernel::MakeSequence(Offset, Buffer.Get(), Buffer.Size(), stream.GetStream());
        }
    };
}

template <typename T, typename TMapping>
static void MakeSequenceImpl(TCudaBuffer<T, TMapping>& buffer, ui32 stream) {
    using TKernel = TMakeSequenceKernel<T>;
    LaunchKernels<TKernel>(buffer.NonEmptyDevices(), stream, buffer);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, TMapping)                            \
    template <>                                                        \
    void MakeSequence(TCudaBuffer<T, TMapping>& buffer, ui32 stream) { \
        ::MakeSequenceImpl(buffer, stream);                            \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (int, TMirrorMapping),
    (ui32, TMirrorMapping),
    (ui64, TMirrorMapping),
    (int, TSingleMapping),
    (ui32, TSingleMapping),
    (ui64, TSingleMapping),
    (int, TStripeMapping),
    (ui32, TStripeMapping),
    (ui64, TStripeMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// MakeSequenceWithOffset

template <typename T, typename TMapping>
static void MakeSequenceWithOffsetImpl(
    TCudaBuffer<T, TMapping>& buffer,
    const NCudaLib::TDistributedObject<T>& offset,
    ui32 stream) {
    using TKernel = TMakeSequenceKernel<T>;
    LaunchKernels<TKernel>(buffer.NonEmptyDevices(), stream, buffer, offset);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, TMapping)                   \
    template <>                                               \
    void MakeSequenceWithOffset<T, TMapping>(                 \
        TCudaBuffer<T, TMapping> & buffer,                    \
        const NCudaLib::TDistributedObject<T>& offset,        \
        ui32 stream) {                                        \
        ::MakeSequenceWithOffsetImpl(buffer, offset, stream); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (int, TMirrorMapping),
    (ui32, TMirrorMapping),
    (ui64, TMirrorMapping),
    (int, TSingleMapping),
    (ui32, TSingleMapping),
    (ui64, TSingleMapping),
    (int, TStripeMapping),
    (ui32, TStripeMapping),
    (ui64, TStripeMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// MakeSequenceGlobal

template <typename T>
static void MakeSequenceGlobalImpl(
    TCudaBuffer<T, NCudaLib::TStripeMapping>& buffer,
    ui32 stream) {
    auto offset = CreateDistributedObject<T>(0);
    for (ui32 dev = 0; dev < offset.DeviceCount(); ++dev) {
        offset.Set(dev, buffer.GetMapping().DeviceSlice(dev).Left);
    }

    using TKernel = TMakeSequenceKernel<T>;
    LaunchKernels<TKernel>(buffer.NonEmptyDevices(), stream, buffer, offset);
}

#define Y_CATBOOST_CUDA_F_IMPL(T)                          \
    template <>                                            \
    void MakeSequenceGlobal<T>(                            \
        TCudaBuffer<T, NCudaLib::TStripeMapping> & buffer, \
        ui32 stream) {                                     \
        ::MakeSequenceGlobalImpl(buffer, stream);          \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL,
    int,
    ui32,
    ui64);

#undef Y_CATBOOST_CUDA_F_IMPL

// InversePermutation

namespace {
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

template <typename T, typename TMapping>
static void InversePermutationImpl(
    const TCudaBuffer<T, TMapping>& order,
    TCudaBuffer<ui32, TMapping>& inverseOrder,
    ui32 streamId) {
    using TKernel = TInversePermutationKernel<ui32>;
    LaunchKernels<TKernel>(order.NonEmptyDevices(), streamId, order, inverseOrder);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, TMapping)                      \
    template <>                                                  \
    void InversePermutation<T, TMapping>(                        \
        const TCudaBuffer<T, TMapping>& order,                   \
        TCudaBuffer<ui32, TMapping>& inverseOrder,               \
        ui32 streamId) {                                         \
        ::InversePermutationImpl(order, inverseOrder, streamId); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (ui32, TMirrorMapping),
    (ui32, TSingleMapping),
    (ui32, TStripeMapping),
    (const ui32, TMirrorMapping),
    (const ui32, TSingleMapping),
    (const ui32, TStripeMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

// register kernels

namespace NCudaLib {
    REGISTER_KERNEL_TEMPLATE(0x001001, TFillBufferKernel, char);
    REGISTER_KERNEL_TEMPLATE(0x001002, TFillBufferKernel, i32);
    REGISTER_KERNEL_TEMPLATE(0x001003, TFillBufferKernel, ui32);
    REGISTER_KERNEL_TEMPLATE(0x001013, TFillBufferKernel, i64);
    REGISTER_KERNEL_TEMPLATE(0x001005, TFillBufferKernel, ui64);
    REGISTER_KERNEL_TEMPLATE(0x001011, TFillBufferKernel, bool);
    REGISTER_KERNEL_TEMPLATE(0x001000, TFillBufferKernel, float);
    REGISTER_KERNEL_TEMPLATE(0x001012, TFillBufferKernel, double);

    REGISTER_KERNEL_TEMPLATE(0x001006, TMakeSequenceKernel, int);
    REGISTER_KERNEL_TEMPLATE(0x001007, TMakeSequenceKernel, ui32);
    REGISTER_KERNEL_TEMPLATE(0x001010, TMakeSequenceKernel, ui64);

    REGISTER_KERNEL_TEMPLATE(0x001008, TInversePermutationKernel, ui32);
}
