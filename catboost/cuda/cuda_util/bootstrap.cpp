#include "bootstrap.h"

#include <catboost/cuda/cuda_util/kernel/bootstrap.cuh>
#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>

using NCudaLib::TMirrorMapping;
using NCudaLib::TSingleMapping;
using NCudaLib::TStripeMapping;
using NKernelHost::TCudaBufferPtr;
using NKernelHost::TCudaStream;
using NKernelHost::TStatelessKernel;

// PoissonBootstrap

namespace {
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
}

template <typename TMapping>
static void PoissonBootstrapImpl(
    TCudaBuffer<ui64, TMapping>& seeds,
    TCudaBuffer<float, TMapping>& weights,
    float lambda,
    ui32 stream) {
    using TKernel = TPoissonBootstrapKernel;
    LaunchKernels<TKernel>(weights.NonEmptyDevices(), stream, lambda, seeds, weights);
}

#define Y_CATBOOST_CUDA_F_IMPL(TMapping)                        \
    template <>                                                 \
    void PoissonBootstrap<TMapping>(                            \
        TCudaBuffer<ui64, TMapping> & seeds,                    \
        TCudaBuffer<float, TMapping> & weights,                 \
        float lambda,                                           \
        ui32 stream) {                                          \
        ::PoissonBootstrapImpl(seeds, weights, lambda, stream); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL,
    TMirrorMapping,
    TSingleMapping,
    TStripeMapping);

#undef Y_CATBOOST_CUDA_F_IMPL

// UniformBootstrap

namespace {
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
}

template <typename TMapping>
static void UniformBootstrapImpl(
    TCudaBuffer<ui64, TMapping>& seeds,
    TCudaBuffer<float, TMapping>& weights,
    float takenFraction,
    ui32 stream) {
    using TKernel = TUniformBootstrapKernel;
    LaunchKernels<TKernel>(weights.NonEmptyDevices(), stream, takenFraction, seeds, weights);
}

#define Y_CATBOOST_CUDA_F_IMPL(TMapping)                               \
    template <>                                                        \
    void UniformBootstrap<TMapping>(                                   \
        TCudaBuffer<ui64, TMapping> & seeds,                           \
        TCudaBuffer<float, TMapping> & weights,                        \
        float takenFraction,                                           \
        ui32 stream) {                                                 \
        ::UniformBootstrapImpl(seeds, weights, takenFraction, stream); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL,
    TMirrorMapping,
    TSingleMapping,
    TStripeMapping);

#undef Y_CATBOOST_CUDA_F_IMPL

// BayesianBootstrap

namespace {
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

template <typename TMapping>
static void BayesianBootstrapImpl(
    TCudaBuffer<ui64, TMapping>& seeds,
    TCudaBuffer<float, TMapping>& weights,
    float temperature,
    ui32 stream) {
    using TKernel = TBayesianBootstrapKernel;
    LaunchKernels<TKernel>(weights.NonEmptyDevices(), stream, seeds, weights, temperature);
}

#define Y_CATBOOST_CUDA_F_IMPL(TMapping)                              \
    template <>                                                       \
    void BayesianBootstrap<TMapping>(                                 \
        TCudaBuffer<ui64, TMapping> & seeds,                          \
        TCudaBuffer<float, TMapping> & weights,                       \
        float temperature,                                            \
        ui32 stream) {                                                \
        ::BayesianBootstrapImpl(seeds, weights, temperature, stream); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL,
    TMirrorMapping,
    TSingleMapping,
    TStripeMapping);

#undef Y_CATBOOST_CUDA_F_IMPL

namespace NCudaLib {
    REGISTER_KERNEL(0x0AAA00, TPoissonBootstrapKernel)
    REGISTER_KERNEL(0x0AAA01, TUniformBootstrapKernel)
    REGISTER_KERNEL(0x0AAA02, TBayesianBootstrapKernel)
}
