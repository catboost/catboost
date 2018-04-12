#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/gpu_data/feature_parallel_dataset.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/gpu_data/oblivious_tree_bin_builder.h>
#include <catboost/cuda/models/add_bin_values.h>
#include <catboost/cuda/targets/target_func.h>

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

    inline TBinarySplit ToSplit(const TBinarizedFeaturesManager& manager,
                                const TBestSplitProperties& props) {
        TBinarySplit bestSplit;
        bestSplit.FeatureId = props.FeatureId;
        bestSplit.BinIdx = props.BinId;
        //We need to adjust binIdx. Float arithmetic could generate empty bin splits for ctrs
        if (manager.IsCat(props.FeatureId)) {
            bestSplit.SplitType = EBinSplitType::TakeBin;
            bestSplit.BinIdx = Min<ui32>(manager.GetBinCount(bestSplit.FeatureId), bestSplit.BinIdx);
        } else {
            bestSplit.SplitType = EBinSplitType::TakeGreater;
            bestSplit.BinIdx = Min<ui32>(manager.GetBorders(bestSplit.FeatureId).size() - 1, bestSplit.BinIdx);
        }
        return bestSplit;
    }

    inline bool HasPermutationDependentSplit(const TObliviousTreeStructure& structure,
                                             const TBinarizedFeaturesManager& featuresManager) {
        for (const auto& split : structure.Splits) {
            if (featuresManager.IsCtr(split.FeatureId)) {
                auto ctr = featuresManager.GetCtr(split.FeatureId);
                if (featuresManager.IsPermutationDependent(ctr)) {
                    return true;
                }
            }
        }
        return false;
    }

    template <class TDataSet>
    inline void CacheBinsForModel(TScopedCacheHolder& cacheHolder,
                                  const TBinarizedFeaturesManager& featuresManager,
                                  const TDataSet& dataSet,
                                  const TObliviousTreeStructure& structure,
                                  TMirrorBuffer<ui32>&& bins) {
        bool hasPermutationCtrs = HasPermutationDependentSplit(structure, featuresManager);
        const auto& scope = hasPermutationCtrs ? dataSet.GetPermutationDependentScope() : dataSet.GetPermutationIndependentScope();
        cacheHolder.CacheOnly(scope, structure, [&]() -> TMirrorBuffer<ui32> {
            TMirrorBuffer<ui32> cachedBins = std::move(bins);
            return cachedBins;
        });
    }

    template <class TDataSet>
    inline const TMirrorBuffer<ui32>& GetBinsForModel(TScopedCacheHolder& cacheHolder,
                                                      const TBinarizedFeaturesManager& featuresManager,
                                                      const TDataSet& dataSet,
                                                      const TObliviousTreeStructure& structure) {
        bool hasPermutationCtrs = HasPermutationDependentSplit(structure, featuresManager);
        const auto& scope = hasPermutationCtrs ? dataSet.GetPermutationDependentScope() : dataSet.GetPermutationIndependentScope();
        return cacheHolder.Cache(scope, structure, [&]() -> TMirrorBuffer<ui32> {
            const bool hasHistory = dataSet.HasCtrHistoryDataSet();
            TMirrorBuffer<ui32> learnBins;
            TMirrorBuffer<ui32> testBins;

            if (hasHistory) {
                learnBins = TMirrorBuffer<ui32>::Create(dataSet.LinkedHistoryForCtr().GetSamplesMapping());
                testBins = TMirrorBuffer<ui32>::Create(dataSet.GetSamplesMapping());
            } else {
                learnBins = TMirrorBuffer<ui32>::Create(dataSet.GetSamplesMapping());
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

    inline void PrintBestScore(const TBinarizedFeaturesManager& featuresManager,
                               const TBinarySplit& bestSplit,
                               double score,
                               ui32 depth) {
        TString splitTypeMessage;

        if (bestSplit.SplitType == EBinSplitType::TakeBin) {
            splitTypeMessage = "TakeBin";
        } else {
            splitTypeMessage = TStringBuilder() << ">" << featuresManager.GetBorders(bestSplit.FeatureId)[bestSplit.BinIdx];
        }

        MATRIXNET_INFO_LOG
            << "Best split for depth " << depth << ": " << bestSplit.FeatureId << " / " << bestSplit.BinIdx << " ("
            << splitTypeMessage << ")"
            << " with score " << score;
        if (featuresManager.IsCtr(bestSplit.FeatureId)) {
            MATRIXNET_INFO_LOG
                << " tensor : " << featuresManager.GetCtr(bestSplit.FeatureId).FeatureTensor << "  (ctr type "
                << featuresManager.GetCtr(bestSplit.FeatureId).Configuration.Type << ")";
        }
        MATRIXNET_INFO_LOG << Endl;
    }

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
