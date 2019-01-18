#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/gpu_data/feature_parallel_dataset.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/gpu_data/oblivious_tree_bin_builder.h>

namespace NCatboostCuda {
    template <class TProp>
    inline TProp TakeBest(const TProp& first, const TProp& second) {
        return first < second
                   ? first
                   : second;
    }

    inline TBestSplitProperties TakeBest(TBestSplitProperties first,
                                         TBestSplitProperties second,
                                         TBestSplitProperties third) {
        return TakeBest(TakeBest(first, second), third);
    }

    TBinarySplit ToSplit(const TBinarizedFeaturesManager& manager,
                         const TBestSplitProperties& props);

    bool HasPermutationDependentSplit(const TObliviousTreeStructure& structure,
                                      const TBinarizedFeaturesManager& featuresManager);

    void CacheBinsForModel(TScopedCacheHolder& cacheHolder,
                           const TBinarizedFeaturesManager& featuresManager,
                           const TFeatureParallelDataSet& dataSet,
                           const TObliviousTreeStructure& structure,
                           TMirrorBuffer<ui32>&& bins);

    const TMirrorBuffer<ui32>& GetBinsForModel(TScopedCacheHolder& cacheHolder,
                                               const TBinarizedFeaturesManager& featuresManager,
                                               const TFeatureParallelDataSet& dataSet,
                                               const TObliviousTreeStructure& structure);

    TString SplitConditionToString(const TBinarizedFeaturesManager& featuresManager, const TBinarySplit& split);

    TString SplitConditionToString(
        const TBinarizedFeaturesManager& featuresManager,
        const TBinarySplit& split,
        ESplitValue value);

    void PrintBestScore(const TBinarizedFeaturesManager& featuresManager,
                        const TBinarySplit& bestSplit,
                        double score,
                        ui32 depth);

    inline ui32 ReverseBits(int u, int nBits) {
        ui32 v = u;
        v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1);
        v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2);
        v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4);
        v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8);
        v = (v >> 16) | (v << 16);
        v >>= (32 - nBits);
        return v;
    }

    inline int GetOddBits(int val) {
        int mask = (val & 0xAAAAAAAA) >> 1;
        int r = 0;
        r |= (mask & 1);
        r |= (mask & 4) >> 1;
        r |= (mask & 16) >> 2;
        r |= (mask & 64) >> 3;
        r |= (mask & 256) >> 4;
        r |= (mask & 1024) >> 5;
        r |= (mask & 4096) >> 6;
        r |= (mask & 16384) >> 7;
        return r;
    }

    inline int GetEvenBits(int val) {
        int mask = (val & 0x55555555);
        int c = 0;
        c |= (mask & 1);
        c |= (mask & 4) >> 1;
        c |= (mask & 16) >> 2;
        c |= (mask & 64) >> 3;
        c |= (mask & 256) >> 4;
        c |= (mask & 1024) >> 5;
        c |= (mask & 4096) >> 6;
        c |= (mask & 16384) >> 7;
        return c;
    }

    inline int MergeBits(int x, int y) {
        int res = 0;
        res |= (x & 1) | ((y & 1) << 1);
        res |= ((x & 2) << 1) | ((y & 2) << 2);
        res |= ((x & 4) << 2) | ((y & 4) << 3);
        res |= ((x & 8) << 3) | ((y & 8) << 4);
        res |= ((x & 16) << 4) | ((y & 16) << 5);
        res |= ((x & 32) << 5) | ((y & 32) << 6);
        res |= ((x & 64) << 6) | ((y & 64) << 7);
        res |= ((x & 128) << 7) | ((y & 128) << 8);
        return res;
    }
}
