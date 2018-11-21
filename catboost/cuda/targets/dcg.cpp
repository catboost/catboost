#include "dcg.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/targets/kernel/dcg.cuh>

using NCudaLib::TMirrorMapping;
using NCudaLib::TSingleMapping;
using NCudaLib::TStripeMapping;
using NKernelHost::TCudaBufferPtr;
using NKernelHost::TCudaStream;
using NKernelHost::TStatelessKernel;

namespace {
    template <typename I, typename T>
    class TDcgDecayKernel : public TStatelessKernel {
    private:
        TCudaBufferPtr<const I> BiasedOffsets_;
        TCudaBufferPtr<T> Decay_;

    public:
        Y_SAVELOAD_DEFINE(BiasedOffsets_, Decay_);

        TDcgDecayKernel() = default;
        TDcgDecayKernel(
            TCudaBufferPtr<const I> biasedOffsets,
            TCudaBufferPtr<T> decay)
            : BiasedOffsets_(biasedOffsets)
            , Decay_(decay)
        {
            Y_ASSERT(BiasedOffsets_.Size() == Decay_.Size());
        }

        void Run(const TCudaStream& stream) const {
            NKernel::MakeDcgDecay(BiasedOffsets_.Get(), Decay_.Get(), BiasedOffsets_.Size(), stream);
        }
    };
}

template <typename I, typename T, typename TMapping>
static void MakeDcgDecayImpl(
    const TCudaBuffer<I, TMapping>& biasedOffsets,
    TCudaBuffer<T, TMapping>& decay,
    ui32 stream)
{
    using TKernel = TDcgDecayKernel<I, T>;
    LaunchKernels<TKernel>(biasedOffsets.NonEmptyDevices(), stream, biasedOffsets, decay);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(I, T, TMapping)                       \
        template <>                                                  \
        void NCatboostCuda::NDetail::MakeDcgDecay<I, T, TMapping>(   \
            const NCudaLib::TCudaBuffer<I, TMapping>& biasedOffsets, \
            NCudaLib::TCudaBuffer<T, TMapping>& decay,               \
            ui32 stream)                                             \
{                                                                    \
    ::MakeDcgDecayImpl(biasedOffsets, decay, stream);                \
}

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (ui32, float, TMirrorMapping),
    (ui32, float, TSingleMapping),
    (ui32, float, TStripeMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

namespace {
    template <typename I, typename T>
    class TDcgExponentialDecayKernel : public TStatelessKernel {
    private:
        TCudaBufferPtr<const I> BiasedOffsets_;
        T Base_ = 0;
        TCudaBufferPtr<T> Decay_;

    public:
        Y_SAVELOAD_DEFINE(BiasedOffsets_, Base_, Decay_);

        TDcgExponentialDecayKernel() = default;
        TDcgExponentialDecayKernel(
            TCudaBufferPtr<const I> biasedOffsets,
            T base,
            TCudaBufferPtr<T> decay)
            : BiasedOffsets_(biasedOffsets)
            , Base_(base)
            , Decay_(decay)
        {
            Y_ASSERT(BiasedOffsets_.Size() == Decay_.Size());
        }

        void Run(const TCudaStream& stream) const {
            NKernel::MakeDcgExponentialDecay(BiasedOffsets_.Get(), Decay_.Get(), BiasedOffsets_.Size(), Base_, stream);
        }
    };
}

template <typename I, typename T, typename TMapping>
static void MakeDcgExponentialDecayImpl(
    const TCudaBuffer<I, TMapping>& biasedOffsets,
    T base,
    TCudaBuffer<T, TMapping>& decay,
    ui32 stream)
{
    using TKernel = TDcgExponentialDecayKernel<I, T>;
    LaunchKernels<TKernel>(biasedOffsets.NonEmptyDevices(), stream, biasedOffsets, base, decay);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(I, T, TMapping)                                \
        template <>                                                           \
        void NCatboostCuda::NDetail::MakeDcgExponentialDecay<I, T, TMapping>( \
            const NCudaLib::TCudaBuffer<I, TMapping>& biasedOffsets,          \
            T base,                                                           \
            NCudaLib::TCudaBuffer<T, TMapping>& decay,                        \
            ui32 stream)                                                      \
{                                                                             \
    ::MakeDcgExponentialDecayImpl(biasedOffsets, base, decay, stream);        \
}

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (ui32, float, TMirrorMapping),
    (const ui32, float, TMirrorMapping),
    (ui32, float, TSingleMapping),
    (const ui32, float, TSingleMapping),
    (ui32, float, TStripeMapping),
    (const ui32, float, TStripeMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

namespace NCudaLib {
    REGISTER_KERNEL_TEMPLATE_2(0x110016, TDcgDecayKernel, ui32, float);

    REGISTER_KERNEL_TEMPLATE_2(0x110017, TDcgExponentialDecayKernel, ui32, float);
}
