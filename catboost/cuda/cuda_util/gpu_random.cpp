#include "gpu_random.h"

#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_util/kernel/random.cuh>
#include <catboost/libs/helpers/cpu_random.h>

using NCudaLib::TMirrorMapping;
using NCudaLib::TSingleMapping;
using NCudaLib::TStripeMapping;
using NKernelHost::TCudaBufferPtr;
using NKernelHost::TCudaStream;
using NKernelHost::TStatelessKernel;

// PoissonRand

namespace {
    class TPoissonKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<ui64> Seeds;
        TCudaBufferPtr<const float> Alphas;
        TCudaBufferPtr<int> Result;

    public:
        TPoissonKernel() = default;

        TPoissonKernel(TCudaBufferPtr<ui64> seeds,
                       TCudaBufferPtr<const float> alphas,
                       TCudaBufferPtr<int> result)
            : Seeds(seeds)
            , Alphas(alphas)
            , Result(result)
        {
        }

        void Run(const TCudaStream& stream) const {
            Y_ASSERT(Result.Size() < (static_cast<ui64>(1) << 32));
            Y_ASSERT(Result.Size() == Result.ObjectCount());
            NKernel::PoissonRand(Seeds.Get(), Result.Size(), Alphas.Get(), Result.Get(), stream.GetStream());
        }

        Y_SAVELOAD_DEFINE(Seeds, Alphas, Result);
    };
}

template <typename TMapping>
static void PoissonRandImpl(
    TCudaBuffer<ui64, TMapping>& seeds,
    const TCudaBuffer<float, TMapping>& alphas,
    TCudaBuffer<int, TMapping>& result,
    ui64 streamId) {
    using TKernel = TPoissonKernel;
    LaunchKernels<TKernel>(result.NonEmptyDevices(), streamId, seeds, alphas, result);
}

#define Y_CATBOOST_CUDA_F_IMPL(TMapping)                    \
    template <>                                             \
    void PoissonRand<TMapping>(                             \
        TCudaBuffer<ui64, TMapping> & seeds,                \
        const TCudaBuffer<float, TMapping>& alphas,         \
        TCudaBuffer<int, TMapping>& result,                 \
        ui64 streamId) {                                    \
        ::PoissonRandImpl(seeds, alphas, result, streamId); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL,
    TMirrorMapping,
    TSingleMapping,
    TStripeMapping);

#undef Y_CATBOOST_CUDA_F_IMPL

// GaussianRand

namespace {
    class TGaussianRandKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<ui64> Seeds;
        TCudaBufferPtr<float> Result;

    public:
        TGaussianRandKernel() = default;

        TGaussianRandKernel(TCudaBufferPtr<ui64> seeds,
                            TCudaBufferPtr<float> result)
            : Seeds(seeds)
            , Result(result)
        {
        }

        Y_SAVELOAD_DEFINE(Seeds, Result);

        void Run(const TCudaStream& stream) const {
            Y_ASSERT(Result.Size() < (static_cast<ui64>(1) << 32));
            Y_ASSERT(Result.Size() == Result.ObjectCount());
            NKernel::GaussianRand(Seeds.Get(), Result.Size(), Result.Get(), stream.GetStream());
        }
    };
}

template <typename TMapping>
static void GaussianRandImpl(
    TCudaBuffer<ui64, TMapping>& seeds,
    TCudaBuffer<float, TMapping>& result,
    ui64 streamId) {
    using TKernel = TGaussianRandKernel;
    LaunchKernels<TKernel>(result.NonEmptyDevices(), streamId, seeds, result);
}

#define Y_CATBOOST_CUDA_F_IMPL(TMapping)             \
    template <>                                      \
    void GaussianRand<TMapping>(                     \
        TCudaBuffer<ui64, TMapping> & seeds,         \
        TCudaBuffer<float, TMapping> & result,       \
        ui64 streamId) {                             \
        ::GaussianRandImpl(seeds, result, streamId); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL,
    TMirrorMapping,
    TSingleMapping,
    TStripeMapping);

#undef Y_CATBOOST_CUDA_F_IMPL

// UniformRand

namespace {
    class TUniformRandKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<ui64> Seeds;
        TCudaBufferPtr<float> Result;

    public:
        TUniformRandKernel() = default;

        TUniformRandKernel(TCudaBufferPtr<ui64> seeds,
                           TCudaBufferPtr<float> result)
            : Seeds(seeds)
            , Result(result)
        {
        }

        Y_SAVELOAD_DEFINE(Seeds, Result);

        void Run(const TCudaStream& stream) const {
            Y_ASSERT(Result.Size() < (static_cast<ui64>(1) << 32));
            Y_ASSERT(Result.Size() == Result.ObjectCount());
            NKernel::UniformRand(Seeds.Get(), Result.Size(), Result.Get(), stream.GetStream());
        }
    };
}

template <typename TMapping>
static void UniformRandImpl(
    TCudaBuffer<ui64, TMapping>& seeds,
    TCudaBuffer<float, TMapping>& result,
    ui64 streamId) {
    using TKernel = TUniformRandKernel;
    LaunchKernels<TKernel>(result.NonEmptyDevices(), streamId, seeds, result);
}

#define Y_CATBOOST_CUDA_F_IMPL(TMapping)            \
    template <>                                     \
    void UniformRand(                               \
        TCudaBuffer<ui64, TMapping>& seeds,         \
        TCudaBuffer<float, TMapping>& result,       \
        ui64 streamId) {                            \
        ::UniformRandImpl(seeds, result, streamId); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL,
    TMirrorMapping,
    TSingleMapping,
    TStripeMapping);

#undef Y_CATBOOST_CUDA_F_IMPL

// GenerateSeedsOnGpu

namespace {
    class TGenerateSeeds: public TStatelessKernel {
    private:
        TCudaBufferPtr<ui64> Seeds;
        ui64 BaseSeed;

    public:
        TGenerateSeeds() = default;

        TGenerateSeeds(TCudaBufferPtr<ui64> seeds,
                       ui64 baseSeed)
            : Seeds(seeds)
            , BaseSeed(baseSeed)
        {
        }

        Y_SAVELOAD_DEFINE(Seeds, BaseSeed);

        void Run(const TCudaStream& stream) const {
            NKernel::GenerateSeeds(BaseSeed, Seeds.Get(), Seeds.Size(), stream.GetStream());
        }
    };
}

template <typename TMapping>
static void GenerateSeedsOnGpuImpl(
    const NCudaLib::TDistributedObject<ui64>& baseSeed,
    TCudaBuffer<ui64, TMapping>& seeds,
    ui64 streamId) {
    using TKernel = TGenerateSeeds;
    LaunchKernels<TKernel>(seeds.NonEmptyDevices(), streamId, seeds, baseSeed);
}

#define Y_CATBOOST_CUDA_F_IMPL(TMapping)                     \
    template <>                                              \
    void GenerateSeedsOnGpu<TMapping>(                       \
        const NCudaLib::TDistributedObject<ui64>& baseSeed,  \
        TCudaBuffer<ui64, TMapping>& seeds,                  \
        ui64 streamId) {                                     \
        ::GenerateSeedsOnGpuImpl(baseSeed, seeds, streamId); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL,
    TMirrorMapping,
    TSingleMapping,
    TStripeMapping);

#undef Y_CATBOOST_CUDA_F_IMPL

// TGpuAwareRandom::CreateSeedsBuffer

template <typename TMapping>
static TCudaBuffer<ui64, TMapping> CreateSeedsBufferImpl(ui32 maxCountPerDevice) {
    auto maxSeedCount = CreateDistributedObject<ui64>(maxCountPerDevice);
    auto mapping = CreateMapping<TMapping>(maxSeedCount);
    return TCudaBuffer<ui64, TMapping>::Create(mapping);
}

#define Y_CATBOOST_CUDA_F_IMPL(TMapping)                                                               \
    template <>                                                                                        \
    TCudaBuffer<ui64, TMapping> TGpuAwareRandom::CreateSeedsBuffer<TMapping>(ui32 maxCountPerDevice) { \
        return ::CreateSeedsBufferImpl<TMapping>(maxCountPerDevice);                                   \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL,
    TMirrorMapping,
    TSingleMapping,
    TStripeMapping);

#undef Y_CATBOOST_CUDA_F_IMPL

// TGpuAwareRandom::FillSeeds

#define Y_CATBOOST_CUDA_F_IMPL(TMapping)                                                          \
    template <>                                                                                   \
    void TGpuAwareRandom::FillSeeds<TMapping>(NCudaLib::TCudaBuffer<ui64, TMapping> * seedsPtr) { \
        auto& seeds = *seedsPtr;                                                                  \
        TVector<ui64> seedsCpu(seeds.GetObjectsSlice().Size());                                   \
        for (ui32 i = 0; i < seeds.GetObjectsSlice().Size(); ++i) {                               \
            seedsCpu[i] = NextUniformL();                                                         \
        }                                                                                         \
        seeds.Write(seedsCpu);                                                                    \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL,
    TMirrorMapping,
    TSingleMapping,
    TStripeMapping);

#undef Y_CATBOOST_CUDA_F_IMPL

// TGpuAwareRandom::CreateSeeds

#define Y_CATBOOST_CUDA_F_IMPL(TMapping)                                                                                  \
    template <>                                                                                                           \
    NCudaLib::TCudaBuffer<ui64, TMapping> TGpuAwareRandom::CreateSeeds<TMapping>(ui64 baseSeed, ui32 maxCountPerDevice) { \
        TRandom random(baseSeed);                                                                                         \
        auto buffer = CreateSeedsBuffer<TMapping>(maxCountPerDevice);                                                     \
        FillSeeds(&buffer);                                                                                               \
        return buffer;                                                                                                    \
    };

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL,
    TMirrorMapping,
    TSingleMapping,
    TStripeMapping);

#undef Y_CATBOOST_CUDA_F_IMPL

// TGpuAwareRandom::GetGpuSeeds

#define Y_CATBOOST_CUDA_F_IMPL(TMapping)                                                        \
    template <>                                                                                 \
    NCudaLib::TCudaBuffer<ui64, TMapping>& TGpuAwareRandom::GetGpuSeeds<TMapping>() {           \
        std::type_index index(typeid(TMapping));                                                \
        return CacheHolder.Cache(*this, index, [&]() -> NCudaLib::TCudaBuffer<ui64, TMapping> { \
            return CreateSeeds<TMapping>(NextUniformL());                                       \
        });                                                                                     \
    };

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL,
    TMirrorMapping,
    TSingleMapping,
    TStripeMapping);

#undef Y_CATBOOST_CUDA_F_IMPL

// register kernels

namespace NCudaLib {
    REGISTER_KERNEL(0xADD000, TPoissonKernel)
    REGISTER_KERNEL(0xADD001, TUniformRandKernel)
    REGISTER_KERNEL(0xADD002, TGaussianRandKernel)
    REGISTER_KERNEL(0xADD003, TGenerateSeeds)
}
