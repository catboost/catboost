#pragma once

#include <catboost/cuda/cuda_lib/cache.h>
#include <catboost/cuda/cuda_lib/fwd.h>
#include <catboost/libs/helpers/cpu_random.h>

class TGpuAwareRandom: public TRandom, public TGuidHolder {
public:
    explicit TGpuAwareRandom(ui64 seed = 0)
        : TRandom(seed)
    {
    }

    template <class TMapping>
    NCudaLib::TCudaBuffer<ui64, TMapping>& GetGpuSeeds();

private:
    template <class TMapping>
    static NCudaLib::TCudaBuffer<ui64, TMapping> CreateSeedsBuffer(ui32 maxCountPerDevice);

    template <class TMapping>
    void FillSeeds(NCudaLib::TCudaBuffer<ui64, TMapping>* seedsPtr);

    template <class TMapping>
    NCudaLib::TCudaBuffer<ui64, TMapping> CreateSeeds(ui64 baseSeed, ui32 maxCountPerDevice = 256 * 256);

private:
    TScopedCacheHolder CacheHolder;
};

template <class TMapping>
void PoissonRand(
    NCudaLib::TCudaBuffer<ui64, TMapping>& seeds,
    const NCudaLib::TCudaBuffer<float, TMapping>& alphas,
    NCudaLib::TCudaBuffer<int, TMapping>& result,
    ui64 streamId = 0);

template <class TMapping>
void GaussianRand(
    NCudaLib::TCudaBuffer<ui64, TMapping>& seeds,
    NCudaLib::TCudaBuffer<float, TMapping>& result,
    ui64 streamId = 0);

template <class TMapping>
void UniformRand(
    NCudaLib::TCudaBuffer<ui64, TMapping>& seeds,
    NCudaLib::TCudaBuffer<float, TMapping>& result,
    ui64 streamId = 0);

template <class TMapping>
void GenerateSeedsOnGpu(
    const NCudaLib::TDistributedObject<ui64>& baseSeed,
    NCudaLib::TCudaBuffer<ui64, TMapping>& seeds,
    ui64 streamId = 0);
