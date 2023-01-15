#pragma once

#include "online_ctr.h"
#include "score_calcers.h"
#include "tensor_search_helpers.h"

#include <catboost/libs/data/objects.h>
#include <catboost/private/libs/algo_helpers/scoring_helpers.h>
#include <catboost/private/libs/data_types/pair.h>

class TStatsForSubtractionTrick {
public:
    TStatsForSubtractionTrick(
        TArrayRef<TBucketStats> statsRef,
        TArrayRef<TBucketStats> parentStatsRef,
        TArrayRef<TBucketStats> siblingStatsRef,
        int maxBucketCount,
        int maxSplitEnsamples)
        : StatsRef(statsRef)
        , ParentStatsRef(parentStatsRef)
        , SiblingStatsRef(siblingStatsRef)
        , MaxBucketCount(maxBucketCount)
        , MaxSplitEnsembles(maxSplitEnsamples)
    {
    }
    TStatsForSubtractionTrick()
        : StatsRef(nullptr, (size_t)0)
        , ParentStatsRef(nullptr, (size_t)0)
        , SiblingStatsRef(nullptr, (size_t)0)
        , MaxBucketCount(0)
        , MaxSplitEnsembles(0)
    {
    }
    TStatsForSubtractionTrick MakeSlice(int idx, size_t statsSize) const {
        TArrayRef<TBucketStats> statsRefSlice(StatsRef.data() + idx * statsSize, statsSize);
        TArrayRef<TBucketStats> parentStatsRefSlice(ParentStatsRef.data() + idx * statsSize, statsSize);
        TArrayRef<TBucketStats> siblingStatsRefSlice(SiblingStatsRef.data() + idx * statsSize, statsSize);
        TArrayRef<TBucketStats> emptyRef;
        TStatsForSubtractionTrick statsForSubtractionTrickSlice(StatsRef.data() != nullptr ? statsRefSlice : emptyRef,
                                                                ParentStatsRef.data() != nullptr ? parentStatsRefSlice : emptyRef,
                                                                SiblingStatsRef.data() != nullptr ? siblingStatsRefSlice : emptyRef,
                                                                MaxBucketCount, MaxSplitEnsembles);
        return statsForSubtractionTrickSlice;
    }
    TArrayRef<TBucketStats> GetStatsRef() const {
        return StatsRef;
    }
    TArrayRef<TBucketStats> GetParentStatsRef() const {
        return ParentStatsRef;
    }
    TArrayRef<TBucketStats> GetSiblingStatsRef() const {
        return SiblingStatsRef;
    }
    int GetMaxBucketCount() const {
        return MaxBucketCount;
    }
    int GetMaxSplitEnsembles() const {
        return MaxSplitEnsembles;
    }
private:
    TArrayRef<TBucketStats> StatsRef;
    TArrayRef<TBucketStats> ParentStatsRef;
    TArrayRef<TBucketStats> SiblingStatsRef;
    int MaxBucketCount = 0;
    int MaxSplitEnsembles = 0;
};

class TCalcScoreFold;
class TFold;

namespace NCatboostOptions {
    class TCatBoostOptions;
}

namespace NCB {
    class TQuantizedObjectsDataProvider;
}

bool IsLeafwiseScoringApplicable(const NCatboostOptions::TCatBoostOptions& params);

TVector<TVector<double>> CalcScoresForOneCandidate(
    const NCB::TQuantizedObjectsDataProvider& data,
    const TCandidatesInfoList& candidate,
    const TCalcScoreFold& fold,
    const TFold& initialFold,
    const TVector<TIndexType>& leafs,
    const TStatsForSubtractionTrick& statsForSubtractionTrick,
    TLearnContext* ctx
);

double CalcScoreWithoutSplit(int leaf, const TFold& fold, const TLearnContext& ctx);

template <typename TGetBucketStats, typename TUpdateSplitScore>
inline void CalcScoresForLeaf(
    const TSplitEnsembleSpec& splitEnsembleSpec,
    ui32 oneHotMaxSize,
    int bucketCount,
    const TGetBucketStats& getBucketStats, // int bucketIdx -> const TBucketStats&
    const TUpdateSplitScore& updateSplitScore // trueStats, falseStats, int splitIdx -> void
) {
    switch(splitEnsembleSpec.Type) {
        case ESplitEnsembleType::OneFeature:
        {
            auto splitType = splitEnsembleSpec.OneSplitType;

            TBucketStats allStats{0, 0, 0, 0};

            for (int bucketIdx = 0; bucketIdx < bucketCount; ++bucketIdx) {
                const TBucketStats& leafStats = getBucketStats(bucketIdx);
                allStats.Add(leafStats);
            }

            if (splitType == ESplitType::OnlineCtr ||
                splitType == ESplitType::FloatFeature ||
                splitType == ESplitType::EstimatedFeature)
            {
                TBucketStats trueStats = allStats;
                TBucketStats falseStats{0, 0, 0, 0};
                for (int splitIdx = 0; splitIdx < bucketCount - 1; ++splitIdx) {
                    falseStats.Add(getBucketStats(splitIdx));
                    trueStats.Remove(getBucketStats(splitIdx));

                    updateSplitScore(trueStats, falseStats, splitIdx);
                }
            } else {
                Y_ASSERT(splitType == ESplitType::OneHotFeature);
                TBucketStats falseStats = allStats;
                for (int bucketIdx = 0; bucketIdx < bucketCount; ++bucketIdx) {
                    const TBucketStats& trueStats = getBucketStats(bucketIdx);
                    falseStats.Remove(trueStats);
                    updateSplitScore(trueStats, falseStats, bucketIdx);
                    falseStats.Add(trueStats);
                }
            }
        }
            break;
        case ESplitEnsembleType::BinarySplits:
        {
            int binaryFeaturesCount = (int)GetValueBitCount(bucketCount - 1);
            for (int binFeatureIdx = 0; binFeatureIdx < binaryFeaturesCount; ++binFeatureIdx) {
                TBucketStats trueStats{0, 0, 0, 0};
                TBucketStats falseStats{0, 0, 0, 0};

                for (int bucketIdx = 0; bucketIdx < bucketCount; ++bucketIdx) {
                    auto& dstStats = ((bucketIdx >> binFeatureIdx) & 1) ? trueStats : falseStats;
                    dstStats.Add(getBucketStats(bucketIdx));
                }

                updateSplitScore(trueStats, falseStats, binFeatureIdx);
            }
        }
            break;
        case ESplitEnsembleType::ExclusiveBundle:
        {
            const auto& exclusiveFeaturesBundle = splitEnsembleSpec.ExclusiveFeaturesBundle;

            TVector<TBucketStats> bundlePartsStats(exclusiveFeaturesBundle.Parts.size());

            TBucketStats allStats = getBucketStats(bucketCount - 1);

            for (auto bundlePartIdx : xrange(exclusiveFeaturesBundle.Parts.size())) {
                auto& bundlePartStats = bundlePartsStats[bundlePartIdx];
                bundlePartStats = TBucketStats{0, 0, 0, 0};

                for (int bucketIdx : exclusiveFeaturesBundle.Parts[bundlePartIdx].Bounds.Iter()) {
                    bundlePartStats.Add(getBucketStats(bucketIdx));
                }
                allStats.Add(bundlePartStats);
            }

            int binsBegin = 0;
            for (auto bundlePartIdx : xrange(exclusiveFeaturesBundle.Parts.size())) {
                if (!UseForCalcScores(exclusiveFeaturesBundle.Parts[bundlePartIdx], oneHotMaxSize)) {
                    continue;
                }

                const auto& bundlePart = exclusiveFeaturesBundle.Parts[bundlePartIdx];
                auto binBounds = bundlePart.Bounds;

                if (bundlePart.FeatureType == EFeatureType::Float) {
                    TBucketStats falseStats = allStats;
                    TBucketStats trueStats = bundlePartsStats[bundlePartIdx];
                    falseStats.Remove(bundlePartsStats[bundlePartIdx]);

                    for (int splitIdx = 0; splitIdx < static_cast<int>(binBounds.GetSize()); ++splitIdx) {
                        if (splitIdx != 0) {
                            const auto& statsPart = getBucketStats(static_cast<int>(binBounds.Begin + splitIdx - 1));
                            falseStats.Add(statsPart);
                            trueStats.Remove(statsPart);
                        }

                        updateSplitScore(trueStats, falseStats, binsBegin + splitIdx);
                    }
                    binsBegin += binBounds.GetSize();
                } else {
                    Y_ASSERT(bundlePart.FeatureType == EFeatureType::Categorical);
                    Y_ASSERT((binBounds.GetSize() + 1) <= oneHotMaxSize);

                    /* for binary features split on value 0 is the same as split on value 1
                     * so don't double the calculations,
                     * it also maintains compatibility with binary packed binary categorical features
                     * where value 1 is always assumed
                     */
                    if (binBounds.GetSize() > 1) {
                        TBucketStats trueStats = allStats;
                        trueStats.Remove(bundlePartsStats[bundlePartIdx]);

                        updateSplitScore(trueStats, bundlePartsStats[bundlePartIdx], binsBegin);
                    }

                    for (int binIdx = 0; binIdx < static_cast<int>(binBounds.GetSize()); ++binIdx) {
                        const auto& statsPart = getBucketStats(static_cast<int>(binBounds.Begin + binIdx));

                        TBucketStats falseStats = allStats;
                        falseStats.Remove(statsPart);

                        updateSplitScore(statsPart, falseStats, binsBegin + binIdx + 1);
                    }

                    binsBegin += binBounds.GetSize() + 1;
                }
            }
        }
            break;
        case ESplitEnsembleType::FeaturesGroup:
        {
            int splitIdxOffset = 0;
            int partStatsOffset = 0;
            for (const auto& part : splitEnsembleSpec.FeaturesGroup.Parts) {
                TBucketStats allStats{0, 0, 0, 0};
                for (int statsIndex = partStatsOffset; statsIndex < partStatsOffset + static_cast<int>(part.BucketCount); ++statsIndex) {
                    allStats.Add(getBucketStats(statsIndex));
                }
                TBucketStats trueStats{0, 0, 0, 0};
                TBucketStats falseStats{0, 0, 0, 0};
                trueStats = allStats;
                for (int splitIdx = 0, statsIndex = partStatsOffset; splitIdx < static_cast<int>(part.BucketCount) - 1; ++splitIdx, ++statsIndex) {
                    const TBucketStats& stats = getBucketStats(statsIndex);
                    falseStats.Add(stats);
                    trueStats.Remove(stats);
                    updateSplitScore(trueStats, falseStats, splitIdxOffset + splitIdx);
                }
                splitIdxOffset += part.BucketCount - 1;
                partStatsOffset += part.BucketCount;
            }
        }
            break;
    }
}

