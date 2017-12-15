#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/gpu_data/fold_based_dataset.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/gpu_data/oblivious_tree_bin_builder.h>
#include <catboost/cuda/models/add_bin_values.h>
#include <catboost/cuda/targets/target_base.h>

namespace NCatboostCuda {
    inline TBestSplitProperties TakeBest(TBestSplitProperties first, TBestSplitProperties second) {
        return ((first.Score < second.Score || (first.Score == second.Score && (first.FeatureId < second.FeatureId)))
                    ? first
                    : second);
    }

    inline TBestSplitProperties TakeBest(TBestSplitProperties first,
                                         TBestSplitProperties second,
                                         TBestSplitProperties third) {
        return TakeBest(TakeBest(first, second), third);
    }

    template <class TDataSet>
    inline void CacheBinsForModel(TScopedCacheHolder& cacheHolder,
                                  const TDataSet& dataSet,
                                  const TObliviousTreeStructure& structure,
                                  TMirrorBuffer<ui32>&& bins) {
        cacheHolder.CacheOnly(dataSet, structure, [&]() -> TMirrorBuffer<ui32> {
            TMirrorBuffer<ui32> cachedBins = std::move(bins);
            return cachedBins;
        });
    }

    template <class TDataSet>
    inline const TMirrorBuffer<ui32>& GetBinsForModel(TScopedCacheHolder& cacheHolder,
                                                      const TBinarizedFeaturesManager& featuresManager,
                                                      const TDataSet& dataSet,
                                                      const TObliviousTreeStructure& structure) {
        return cacheHolder.Cache(dataSet, structure, [&]() -> TMirrorBuffer<ui32> {
            const bool hasHistory = dataSet.HasCtrHistoryDataSet();
            TMirrorBuffer<ui32> learnBins;
            TMirrorBuffer<ui32> testBins;

            if (hasHistory) {
                learnBins = TMirrorBuffer<ui32>::Create(dataSet.LinkedHistoryForCtr().GetDocumentsMapping());
                testBins = TMirrorBuffer<ui32>::Create(dataSet.GetDocumentsMapping());
            } else {
                learnBins = TMirrorBuffer<ui32>::Create(dataSet.GetDocumentsMapping());
            }

            {
                TTreeUpdater<TDataSet> builder(cacheHolder,
                                               featuresManager,
                                               dataSet.GetCtrTargets(),
                                               hasHistory ? dataSet.LinkedHistoryForCtr() : dataSet,
                                               learnBins,
                                               hasHistory ? &dataSet : nullptr,
                                               hasHistory ? &testBins : nullptr);

                for (auto& split : structure.Splits) {
                    builder.AddSplit(split);
                }
            }

            if (hasHistory) {
                cacheHolder.CacheOnly(dataSet.LinkedHistoryForCtr(), structure, [&]() -> TMirrorBuffer<ui32> {
                    return std::move(learnBins);
                });
            }
            return hasHistory ? std::move(testBins) : std::move(learnBins);
        });
    }
}
