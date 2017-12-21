#pragma once

#include <catboost/cuda/cuda_util/kernel/bootstrap.cuh>
#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>

namespace NKernelHost {
    class TPoissonBootstrapKernel: public TStatelessKernel {
    private:
        float Lambda;
        TCudaBufferPtr<ui64> Seeds;
        TCudaBufferPtr<float> Weights;

    public:
        TPoissonBootstrapKernel() = default;

        TPoissonBootstrapKernel(float lambda,
                                TCudaBufferPtr<ui64> seeds,
                                TCudaBufferPtr<float> weights)
            : Lambda(lambda)
            , Seeds(seeds)
            , Weights(weights)
        {
        }

        void Run(const TCudaStream& stream) const {
            CB_ENSURE(Seeds.Size() % 256 == 0);
            NKernel::PoissonBootstrap(Lambda, Seeds.Get(), Seeds.Size(), Weights.Get(), Weights.Size(), stream.GetStream());
        }

        Y_SAVELOAD_DEFINE(Lambda, Seeds, Weights);
    };

    class TUniformBootstrapKernel: public TStatelessKernel {
    private:
        float SampleRate;
        TCudaBufferPtr<ui64> Seeds;
        TCudaBufferPtr<float> Weights;

    public:
        TUniformBootstrapKernel() = default;

        TUniformBootstrapKernel(float sampleRate,
                                TCudaBufferPtr<ui64> seeds,
                                TCudaBufferPtr<float> weights)
            : SampleRate(sampleRate)
            , Seeds(seeds)
            , Weights(weights)
        {
            Y_ASSERT(sampleRate > 0);
            Y_ASSERT(sampleRate <= 1.0f);
        }

        Y_SAVELOAD_DEFINE(SampleRate, Seeds, Weights);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE(Seeds.Size() % 256 == 0);
            NKernel::UniformBootstrap(SampleRate, Seeds.Get(), Seeds.Size(), Weights.Get(), Weights.Size(), stream.GetStream());
        }
    };

    class TBayesianBootstrapKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<ui64> Seeds;
        TCudaBufferPtr<float> Weights;
        float Temperature;

    public:
        TBayesianBootstrapKernel() = default;

        TBayesianBootstrapKernel(TCudaBufferPtr<ui64> seeds,
                                 TCudaBufferPtr<float> weights,
                                 float temperature)
            : Seeds(seeds)
            , Weights(weights)
            , Temperature(temperature)
        {
        }

        Y_SAVELOAD_DEFINE(Seeds, Weights, Temperature);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE(Seeds.Size() % 256 == 0);

            NKernel::BayesianBootstrap(Seeds.Get(), Seeds.Size(), Weights.Get(), Weights.Size(), Temperature, stream.GetStream());
        }
    };
}

template <class TMapping>
inline void PoissonBootstrap(TCudaBuffer<ui64, TMapping>& seeds,
                             TCudaBuffer<float, TMapping>& weights,
                             float lambda,
                             ui32 stream = 0) {
    using TKernel = NKernelHost::TPoissonBootstrapKernel;
    LaunchKernels<TKernel>(weights.NonEmptyDevices(), stream, lambda, seeds, weights);
}

template <class TMapping>
inline void UniformBootstrap(TCudaBuffer<ui64, TMapping>& seeds,
                             TCudaBuffer<float, TMapping>& weights,
                             float takenFraction = 0.5,
                             ui32 stream = 0) {
    using TKernel = NKernelHost::TUniformBootstrapKernel;
    LaunchKernels<TKernel>(weights.NonEmptyDevices(), stream, takenFraction, seeds, weights);
}

template <class TMapping>
inline void BayesianBootstrap(TCudaBuffer<ui64, TMapping>& seeds,
                              TCudaBuffer<float, TMapping>& weights,
                              float temperature,
                              ui32 stream = 0) {
    using TKernel = NKernelHost::TBayesianBootstrapKernel;
    LaunchKernels<TKernel>(weights.NonEmptyDevices(), stream, seeds, weights, temperature);
}
