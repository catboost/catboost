#include "filter.h"

#include "reduce.h"
#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/cuda_util/kernel/filter.cuh>

#include <util/generic/cast.h>

using NCudaLib::TMirrorMapping;
using NCudaLib::TSingleMapping;
using NCudaLib::TStripeMapping;
using NKernelHost::TCudaBufferPtr;
using NKernelHost::TCudaStream;
using NKernelHost::TStatelessKernel;

// NonZeroFilter

namespace {
    template <typename TStatus>
    class TFilterKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> Weights;
        TCudaBufferPtr<TStatus> Result;

    public:
        TFilterKernel() = default;

        TFilterKernel(TCudaBufferPtr<const float> weights,
                      TCudaBufferPtr<TStatus> result)
            : Weights(weights)
            , Result(result)
        {
        }

        Y_SAVELOAD_DEFINE(Weights, Result);

        void Run(const TCudaStream& stream) const {
            NKernel::Filter(Weights.Get(), Weights.Size(), Result.Get(), stream.GetStream());
        }
    };
}

template <typename TMapping, class TStatus>
static void NonZeroFilterImpl(
    const TCudaBuffer<float, TMapping>& weights,
    TCudaBuffer<TStatus, TMapping>& status,
    ui32 stream) {
    using TKernel = TFilterKernel<TStatus>;
    LaunchKernels<TKernel>(weights.NonEmptyDevices(), stream, weights, status);
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(TMapping, TStatus)     \
    template <>                                       \
    void NonZeroFilter<TMapping, TStatus>(            \
        const TCudaBuffer<float, TMapping>& weights,  \
        TCudaBuffer<TStatus, TMapping>& status,       \
        ui32 stream) {                                \
        ::NonZeroFilterImpl(weights, status, stream); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (TMirrorMapping, ui32),
    (TSingleMapping, ui32),
    (TStripeMapping, ui32),
    (TMirrorMapping, ui64),
    (TSingleMapping, ui64),
    (TStripeMapping, ui64));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY

namespace NCudaLib {
    REGISTER_KERNEL_TEMPLATE(0xFF1F01, TFilterKernel, ui32);
    REGISTER_KERNEL_TEMPLATE(0xFF1F02, TFilterKernel, ui64);
}
