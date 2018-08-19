#pragma once

#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_util/kernel/random.cuh>
#include <catboost/libs/helpers/cpu_random.h>

namespace NKernelHost {
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

        Y_SAVELOAD_DEFINE(Seeds, Alphas, Result)
    };

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

class TGpuAwareRandom: public TRandom, public TGuidHolder {
public:
    explicit TGpuAwareRandom(ui64 seed = 0)
        : TRandom(seed)
    {
    }

    template <class TMapping>
    NCudaLib::TCudaBuffer<ui64, TMapping>& GetGpuSeeds() {
        std::type_index index(typeid(TMapping));
        return CacheHolder.Cache(*this, index, [&]() -> NCudaLib::TCudaBuffer<ui64, TMapping> {
            return CreateSeeds<TMapping>(NextUniformL());
        });
    };

private:
    template <class TMapping>
    static NCudaLib::TCudaBuffer<ui64, TMapping> CreateSeedsBuffer(ui32 maxCountPerDevice) {
        NCudaLib::TDistributedObject<ui64> maxSeedCount = CreateDistributedObject<ui64>(maxCountPerDevice);
        auto mapping = CreateMapping<TMapping>(maxSeedCount);
        return TCudaBuffer<ui64, TMapping>::Create(mapping);
    }

    template <class TMapping>
    inline void FillSeeds(NCudaLib::TCudaBuffer<ui64, TMapping>* seedsPtr) {
        auto& seeds = *seedsPtr;
        TVector<ui64> seedsCpu(seeds.GetObjectsSlice().Size());
        for (ui32 i = 0; i < seeds.GetObjectsSlice().Size(); ++i) {
            seedsCpu[i] = NextUniformL();
        }
        seeds.Write(seedsCpu);
    }

    template <class TMapping>
    inline NCudaLib::TCudaBuffer<ui64, TMapping> CreateSeeds(ui64 baseSeed, ui32 maxCountPerDevice = 256 * 256) {
        TRandom random(baseSeed);
        auto buffer = CreateSeedsBuffer<TMapping>(maxCountPerDevice);
        FillSeeds(&buffer);
        return buffer;
    };

private:
    TScopedCacheHolder CacheHolder;
};

template <class TMapping>
inline void PoissonRand(TCudaBuffer<ui64, TMapping>& seeds, const TCudaBuffer<float, TMapping>& alphas,
                        TCudaBuffer<int, TMapping>& result, ui64 streamId = 0) {
    using TKernel = NKernelHost::TPoissonKernel;
    LaunchKernels<TKernel>(result.NonEmptyDevices(), streamId, seeds, alphas, result);
}

template <class TMapping>
inline void GaussianRand(TCudaBuffer<ui64, TMapping>& seeds, TCudaBuffer<float, TMapping>& result, ui64 streamId = 0) {
    using TKernel = NKernelHost::TGaussianRandKernel;
    LaunchKernels<TKernel>(result.NonEmptyDevices(), streamId, seeds, result);
}

template <class TMapping>
inline void UniformRand(TCudaBuffer<ui64, TMapping>& seeds, TCudaBuffer<float, TMapping>& result, ui64 streamId = 0) {
    using TKernel = NKernelHost::TUniformRandKernel;
    LaunchKernels<TKernel>(result.NonEmptyDevices(), streamId, seeds, result);
}

template <class TMapping>
inline void GenerateSeedsOnGpu(const NCudaLib::TDistributedObject<ui64>& baseSeed,
                               TCudaBuffer<ui64, TMapping>& seeds, ui64 streamId = 0) {
    using TKernel = NKernelHost::TGenerateSeeds;
    LaunchKernels<TKernel>(seeds.NonEmptyDevices(), streamId, seeds, baseSeed);
}
