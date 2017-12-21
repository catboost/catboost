#pragma once

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
inline void BootstrapPointsFilter(const TCudaBuffer<float, TMapping>& weights,
                                  TCudaBuffer<ui32, TMapping>& status,
                                  ui64 stream) {
    using TKernel = NKernelHost::TFilterKernel;
    LaunchKernels<TKernel>(weights.NonEmptyDevices(), stream, weights, status);
}
