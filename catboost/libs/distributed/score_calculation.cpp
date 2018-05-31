#include "score_calculation.h"

#include <catboost/libs/algo/score_calcer.h>

using NCatboostDistributed::TStats3D;

template<typename TFullIndexType, typename TIsCaching>
static void CalcStatsImpl(const TIsCaching& isCaching,
        const TVector<TFullIndexType>& singleIdx,
        const TCalcScoreFold& fold,
        bool isPlainMode,
        const TStatsIndexer& indexer,
        int depth,
        int splitStatsCount,
        TBucketStats* splitStats) {
    Y_ASSERT(!isCaching || depth > 0);
    const int approxDimension = fold.GetApproxDimension();
    for (int bodyTailIdx = 0; bodyTailIdx < fold.GetBodyTailCount(); ++bodyTailIdx) {
        const auto& bt = fold.BodyTailArr[bodyTailIdx];
        for (int dim = 0; dim < approxDimension; ++dim) {
            TBucketStats* stats = splitStats + (bodyTailIdx * approxDimension + dim) * splitStatsCount;
            CalcStatsKernel(isCaching, singleIdx, fold, isPlainMode, indexer, depth, bt, dim, stats);
        }
    }
}

TStats3D CalcStats3D(const TAllFeatures& af,
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

    decltype(auto) SelectCalcStatsImpl = [&] (auto isCaching, const TCalcScoreFold& fold, int splitStatsCount, auto* splitStats) {
        const bool isPlainMode = IsPlainMode(fitParams.BoostingOptions->BoostingType);
        if (bucketIndexBits <= 8) {
            TVector<ui8> singleIdx;
            BuildSingleIndex(fold, af, allCtrs, split, indexer, &singleIdx);
            CalcStatsImpl(isCaching, singleIdx, fold, isPlainMode, indexer, depth, splitStatsCount, GetDataPtr(*splitStats));
        } else if (bucketIndexBits <= 16) {
            TVector<ui16> singleIdx;
            BuildSingleIndex(fold, af, allCtrs, split, indexer, &singleIdx);
            CalcStatsImpl(isCaching, singleIdx, fold, isPlainMode, indexer, depth, splitStatsCount, GetDataPtr(*splitStats));
        } else if (bucketIndexBits <= 32) {
            TVector<ui32> singleIdx;
            BuildSingleIndex(fold, af, allCtrs, split, indexer, &singleIdx);
            CalcStatsImpl(isCaching, singleIdx, fold, isPlainMode, indexer, depth, splitStatsCount, GetDataPtr(*splitStats));
        } else {
            CB_ENSURE(false, "too deep or too much splitsCount for score calculation");
        }
    };
    const auto& treeOptions = fitParams.ObliviousTreeOptions.Get();
    if (!IsSamplingPerTree(treeOptions)) {
        TVector<TBucketStats> scratchSplitStats;
        const int splitStatsCount = indexer.CalcSize(depth);
        const int statsCount = fold.GetBodyTailCount() * fold.GetApproxDimension() * splitStatsCount;
        scratchSplitStats.yresize(statsCount);
        SelectCalcStatsImpl(/*isCaching*/ std::false_type(), fold, splitStatsCount, &scratchSplitStats);
        return TStats3D(scratchSplitStats, bucketCount, 1U << depth);
    } else {
        const int splitStatsCount = indexer.CalcSize(treeOptions.MaxDepth);
        const int statsCount = fold.GetBodyTailCount() * fold.GetApproxDimension() * splitStatsCount;
        bool areStatsDirty;
        TVector<TBucketStats, TPoolAllocator>& splitStats = statsFromPrevTree->GetStats(split, statsCount, &areStatsDirty); // thread-safe access
        if (depth == 0 || areStatsDirty) {
            SelectCalcStatsImpl(/*isCaching*/ std::false_type(), fold, splitStatsCount, &splitStats);
        } else {
            SelectCalcStatsImpl(/*isCaching*/ std::true_type(), prevLevelData, splitStatsCount, &splitStats);
        }
        return TStats3D(TVector<TBucketStats>(splitStats.begin(), splitStats.end()), bucketCount, 1U << treeOptions.MaxDepth);
    }
    CB_ENSURE(false, "too deep or too much splitsCount for score calculation");
}

TVector<TScoreBin> GetScoreBins(const TStats3D& stats, ESplitType splitType, int depth, const NCatboostOptions::TCatBoostOptions& fitParams) {
    const TVector<TBucketStats>& bucketStats = stats.Stats;
    const int splitStatsCount = stats.BucketCount * stats.MaxLeafCount;
    const int bucketCount = stats.BucketCount;
    const float l2Regularizer = static_cast<const float>(fitParams.ObliviousTreeOptions->L2Reg);
    const bool isPlainMode = IsPlainMode(fitParams.BoostingOptions->BoostingType);
    const int leafCount = 1 << depth;
    const TStatsIndexer indexer(bucketCount);
    TVector<TScoreBin> scoreBin(bucketCount);
    for (int statsIdx = 0; statsIdx * splitStatsCount < bucketStats.ysize(); ++statsIdx) {
        const TBucketStats* stats = GetDataPtr(bucketStats) + statsIdx * splitStatsCount;
        if (isPlainMode) {
            UpdateScoreBin(stats, leafCount, indexer, splitType, l2Regularizer, /*isPlainMode=*/std::true_type(), &scoreBin);
        } else {
            UpdateScoreBin(stats, leafCount, indexer, splitType, l2Regularizer, /*isPlainMode=*/std::false_type(), &scoreBin);
        }
    }
    return scoreBin;
}
