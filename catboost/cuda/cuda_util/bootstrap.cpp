#include "bootstrap.h"

#include <catboost/cuda/cuda_util/kernel/bootstrap.cuh>
#include <catboost/cuda/cuda_util/kernel/mvs.cuh>
#include <catboost/cuda/cuda_lib/helpers.h>
#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>

#include <util/generic/ptr.h>

using NCudaLib::TMirrorMapping;
using NCudaLib::TSingleMapping;
using NCudaLib::TStripeMapping;
using NKernelHost::IMemoryManager;
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



//CalculateMvsThreshold

namespace {
    class TCalculateMvsThresholdKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<float> Candidates;
        float TakenFraction;
        TCudaBufferPtr<float> Threshold;

    public:
        TCalculateMvsThresholdKernel() = default;

        TCalculateMvsThresholdKernel(
                            TCudaBufferPtr<float> candidates,
                            float takenFraction,
                            TCudaBufferPtr<float> threshold
                            )
            : Candidates(candidates)
            , TakenFraction(takenFraction)
            , Threshold(threshold)
        {
            Y_ASSERT(takenFraction > 0);
            Y_ASSERT(takenFraction <= 1);
        }

        Y_SAVELOAD_DEFINE(Candidates, TakenFraction, Threshold);

        void Run(const TCudaStream& stream) const {
            NKernel::CalculateMvsThreshold(TakenFraction, Candidates.Get(), Candidates.Size(), Threshold.Get(), stream.GetStream());
        }
    };
}


template <typename TMapping>
static TVector<float> CalculateMvsThresholdImpl(
    TCudaBuffer<float, TMapping>& candidates,
    float takenFraction,
    ui32 stream) {

    auto dst = TSingleBuffer<float>::Create(TSingleMapping(0, NHelpers::CeilDivide(candidates.GetObjectsSlice().Size(), (1 << 13))));
    using TKernel = TCalculateMvsThresholdKernel;
    LaunchKernels<TKernel>(candidates.NonEmptyDevices(), stream, candidates, takenFraction, dst);

    TVector<float> result;
    dst.Read(result, stream);

    return result;
}

#define Y_CATBOOST_CUDA_F_IMPL(TMapping)                                       \
    template <>                                                                \
    TVector<float> CalculateMvsThreshold<TMapping>(                            \
        NCudaLib::TCudaBuffer<float, TMapping>& candidates,                    \
        float takenFraction,                                                   \
        ui32 stream) {                                                         \
        return ::CalculateMvsThresholdImpl(candidates, takenFraction, stream); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL,
    TMirrorMapping,
    TSingleMapping,
    TStripeMapping);

#undef Y_CATBOOST_CUDA_F_IMPL

//MvsBootstrapRadixSort
namespace {
    class TMvsBootstrapRadixSortKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<ui64> Seeds;
        TCudaBufferPtr<float> Weights;
        TCudaBufferPtr<const float> Ders;
        float TakenFraction;
        float Lambda;

    public:

        TMvsBootstrapRadixSortKernel() = default;

        TMvsBootstrapRadixSortKernel(
            TCudaBufferPtr<ui64> seeds,
            TCudaBufferPtr<float> weights,
            TCudaBufferPtr<const float> ders,
            float takenFraction,
            float lambda)
            : Seeds(seeds)
            , Weights(weights)
            , Ders(ders)
            , TakenFraction(takenFraction)
            , Lambda(lambda)
        {
            Y_ASSERT(Weights.Size() == Ders.Size());
        }

        void Run(const TCudaStream& stream
            ) const {
            CB_ENSURE(Seeds.Size() % 256 == 0);

            NKernel::MvsBootstrapRadixSort(
                TakenFraction, Lambda,
                Weights.Get(),
                Ders.Get(),
                Ders.Size(),
                Seeds.Get(), Seeds.Size(),
                stream.GetStream());
        }

        Y_SAVELOAD_DEFINE(Seeds, Weights, Ders, TakenFraction, Lambda);
    };
}

template <typename TMapping>
static void MvsBootstrapRadixSortImpl(
    TCudaBuffer<ui64, TMapping>& seeds,
    TCudaBuffer<float, TMapping>& weights,
    const TCudaBuffer<float, TMapping>& ders,
    float takenFraction,
    float lambda,
    ui32 stream) {

    using TKernel = TMvsBootstrapRadixSortKernel;
    LaunchKernels<TKernel>(weights.NonEmptyDevices(), stream, seeds, weights, ders, takenFraction, lambda);
}

#define Y_CATBOOST_CUDA_F_IMPL(TMapping)                                            \
    template <>                                                                     \
    void MvsBootstrapRadixSort<TMapping>(                                           \
        TCudaBuffer<ui64, TMapping> & seeds,                                        \
        TCudaBuffer<float, TMapping> & weights,                                     \
        const NCudaLib::TCudaBuffer<float, TMapping>& ders,                         \
        float takenFraction,                                                        \
        float lambda,                                                               \
        ui32 stream) {                                                              \
        ::MvsBootstrapRadixSortImpl(seeds, weights, ders, takenFraction, lambda, stream); \
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
    REGISTER_KERNEL(0x0AAA03, TCalculateMvsThresholdKernel)
    REGISTER_KERNEL(0x0AAA04, TMvsBootstrapRadixSortKernel)
}
