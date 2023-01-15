#include "filter.h"

#include "reduce.h"
#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/cuda_util/kernel/filter.cuh>

using NCudaLib::TMirrorMapping;
using NCudaLib::TSingleMapping;
using NCudaLib::TStripeMapping;
using NKernelHost::TCudaBufferPtr;
using NKernelHost::TCudaStream;
using NKernelHost::TStatelessKernel;

// NonZeroFilter

namespace {
    class TFilterKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> Weights;
        TCudaBufferPtr<ui32> Result;

    public:
        TFilterKernel() = default;

        TFilterKernel(TCudaBufferPtr<const float> weights,
                      TCudaBufferPtr<ui32> result)
            : Weights(weights)
            , Result(result)
        {
        }

        Y_SAVELOAD_DEFINE(Weights, Result);

        void Run(const TCudaStream& stream) const {
            NKernel::Filter(Weights.Get(), (const ui32)Weights.Size(), Result.Get(), stream.GetStream());
        }
    };
}

template <typename TMapping>
static void NonZeroFilterImpl(
    const TCudaBuffer<float, TMapping>& weights,
    TCudaBuffer<ui32, TMapping>& status,
    ui32 stream) {
    using TKernel = TFilterKernel;
    LaunchKernels<TKernel>(weights.NonEmptyDevices(), stream, weights, status);
}

#define Y_CATBOOST_CUDA_F_IMPL(TMapping)              \
    template <>                                       \
    void NonZeroFilter<TMapping>(                     \
        const TCudaBuffer<float, TMapping>& weights,  \
        TCudaBuffer<ui32, TMapping>& status,          \
        ui32 stream) {                                \
        ::NonZeroFilterImpl(weights, status, stream); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL,
    TMirrorMapping,
    TSingleMapping,
    TStripeMapping);

#undef Y_CATBOOST_CUDA_F_IMPL

namespace NCudaLib {
    REGISTER_KERNEL(0xFF1F01, TFilterKernel);
}
