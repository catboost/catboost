#include "score_calcer.h"
#include "calc_score_cache.h"
#include "index_calcer.h"
#include "split.h"
#include <catboost/libs/options/defaults_helper.h>

#include <type_traits>

int GetSplitCount(const TVector<int>& splitsCount,
                         const TVector<TVector<int>>& oneHotValues,
                         const TSplitCandidate& split) {
    if (split.Type == ESplitType::OnlineCtr) {
        return split.Ctr.BorderCount;
    } else if (split.Type == ESplitType::FloatFeature) {
        return splitsCount[split.FeatureIdx];
    } else {
        Y_ASSERT(split.Type == ESplitType::OneHotFeature);
        return oneHotValues[split.FeatureIdx].ysize();
    }
}

template<typename TFullIndexType, typename TIsCaching>
static TVector<TScoreBin> CalcScoreImpl(const TIsCaching& isCaching,
        const TVector<TFullIndexType>& singleIdx,
        const TCalcScoreFold& fold,
        bool isPlainMode,
        float l2Regularizer,
        ESplitType splitType,
        const TStatsIndexer& indexer,
        int depth,
        int splitStatsCount,
        TBucketStats* splitStats) {
    Y_ASSERT(!isCaching || depth > 0);
    const int approxDimension = fold.GetApproxDimension();
    const int leafCount = 1 << depth;
    TVector<TScoreBin> scoreBins(indexer.BucketCount);
    for (int bodyTailIdx = 0; bodyTailIdx < fold.GetBodyTailCount(); ++bodyTailIdx) {
        const auto& bt = fold.BodyTailArr[bodyTailIdx];
        for (int dim = 0; dim < approxDimension; ++dim) {
            TBucketStats* stats = splitStats + (bodyTailIdx * approxDimension + dim) * splitStatsCount;
            CalcStatsKernel(isCaching, singleIdx, fold, isPlainMode, indexer, depth, bt, dim, stats);
            if (isPlainMode) {
                UpdateScoreBin(stats, leafCount, indexer, splitType, l2Regularizer, /*isPlainMode=*/std::true_type(), &scoreBins);
            } else {
                UpdateScoreBin(stats, leafCount, indexer, splitType, l2Regularizer, /*isPlainMode=*/std::false_type(), &scoreBins);
            }
        }
    }
    return scoreBins;
}

TVector<TScoreBin> CalcScore(const TAllFeatures& af,
                          const TVector<int>& splitsCount,
                          const std::tuple<const TOnlineCTRHash&, const TOnlineCTRHash&>& allCtrs,
                          const TCalcScoreFold& fold,
                          const TCalcScoreFold& prevLevelData,
                          const NCatboostOptions::TCatBoostOptions& fitParams,
                          const TSplitCandidate& split,
                          int depth,
                          TBucketStatsCache* statsFromPrevTree) {
    const int bucketCount = GetSplitCount(splitsCount, af.OneHotValues, split) + 1;
    const TStatsIndexer indexer(bucketCount);
    const int bucketIndexBits = GetValueBitCount(bucketCount) + depth + 1;

    decltype(auto) SelectCalcScoreImpl = [&] (auto isCaching, const TCalcScoreFold& fold, int splitStatsCount, auto* splitStats) {
        const bool isPlainMode = IsPlainMode(fitParams.BoostingOptions->BoostingType);
        const float l2Regularizer = static_cast<const float>(fitParams.ObliviousTreeOptions->L2Reg);
        if (bucketIndexBits <= 8) {
            TVector<ui8> singleIdx;
            BuildSingleIndex(fold, af, allCtrs, split, indexer, &singleIdx);
            return CalcScoreImpl(isCaching, singleIdx, fold, isPlainMode, l2Regularizer, split.Type, indexer, depth, splitStatsCount, GetDataPtr(*splitStats));
        } else if (bucketIndexBits <= 16) {
            TVector<ui16> singleIdx;
            BuildSingleIndex(fold, af, allCtrs, split, indexer, &singleIdx);
            return CalcScoreImpl(isCaching, singleIdx, fold, isPlainMode, l2Regularizer, split.Type, indexer, depth, splitStatsCount, GetDataPtr(*splitStats));
        } else if (bucketIndexBits <= 32) {
            TVector<ui32> singleIdx;
            BuildSingleIndex(fold, af, allCtrs, split, indexer, &singleIdx);
            return CalcScoreImpl(isCaching, singleIdx, fold, isPlainMode, l2Regularizer, split.Type, indexer, depth, splitStatsCount, GetDataPtr(*splitStats));
        }
        CB_ENSURE(false, "too deep or too much splitsCount for score calculation");
    };
    const auto& treeOptions = fitParams.ObliviousTreeOptions.Get();
    if (!IsSamplingPerTree(treeOptions)) {
        TVector<TBucketStats> scratchSplitStats;
        const int splitStatsCount = indexer.CalcSize(depth);
        const int statsCount = splitStatsCount;
        scratchSplitStats.yresize(statsCount);
        return SelectCalcScoreImpl(/*isCaching*/ std::false_type(), fold, /*splitStatsCount*/ 0, &scratchSplitStats);
    } else {
        const int splitStatsCount = indexer.CalcSize(treeOptions.MaxDepth);
        const int statsCount = fold.GetBodyTailCount() * fold.GetApproxDimension() * splitStatsCount;
        bool areStatsDirty;
        TVector<TBucketStats, TPoolAllocator>& splitStats = statsFromPrevTree->GetStats(split, statsCount, &areStatsDirty); // thread-safe access
        if (depth == 0 || areStatsDirty) {
            return SelectCalcScoreImpl(/*isCaching*/ std::false_type(), fold, splitStatsCount, &splitStats);
        } else {
            return SelectCalcScoreImpl(/*isCaching*/ std::true_type(), prevLevelData, splitStatsCount, &splitStats);
        }
    }
    CB_ENSURE(false, "too deep or too much splitsCount for score calculation");
}
