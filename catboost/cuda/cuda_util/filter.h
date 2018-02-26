#pragma once

#include "reduce.h"
#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/cuda_util/kernel/filter.cuh>

namespace NKernelHost {
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

template <class TMapping>
inline void NonZeroFilter(const TCudaBuffer<float, TMapping>& weights,
                          TCudaBuffer<ui32, TMapping>& status,
                          ui32 stream) {
    using TKernel = NKernelHost::TFilterKernel;
    LaunchKernels<TKernel>(weights.NonEmptyDevices(), stream, weights, status);
}

template <class TMapping>
inline TCudaBuffer<ui32, TMapping> NonZeroSizes(const TCudaBuffer<float, TMapping>& weights,
                                                ui32 stream = 0) {
    TCudaBuffer<ui32, TMapping> status;
    status.Reset(weights.GetMapping());
    NonZeroFilter(weights, status, stream);

    TCudaBuffer<ui32, TMapping> result;
    auto resultMapping = status.GetMapping().Transform([&](const TSlice&) {
        return 1;
    });
    result.Reset(resultMapping);
    ReduceVector(status, result, EOperatorType::Sum, stream);
    return result;
}
