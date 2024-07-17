#pragma once

#include "calc_score_cache.h"
#include "index_calcer.h"
#include "score_calcers.h"
#include "split.h"

#include <catboost/libs/data/feature_grouping.h>
#include <catboost/libs/data/packed_binary_features.h>
#include <catboost/private/libs/index_range/index_range.h>

#include <library/cpp/binsaver/bin_saver.h>
#include <library/cpp/containers/2d_array/2d_array.h>


struct TBucketPairWeightStatistics {
    double SmallerBorderWeightSum = 0.0; // The weight sum of pair elements with smaller border.
    double GreaterBorderRightWeightSum = 0.0; // The weight sum of pair elements with greater border.

public:
    SAVELOAD(SmallerBorderWeightSum, GreaterBorderRightWeightSum);

    void Add(const TBucketPairWeightStatistics& rhs) {
        SmallerBorderWeightSum += rhs.SmallerBorderWeightSum;
        GreaterBorderRightWeightSum += rhs.GreaterBorderRightWeightSum;
    }
};


struct TPairwiseStats {
    TVector<TVector<double>> DerSums; // [leafCount][bucketCount]

    /* statsCount is
     *  For SplitCandidates:         bucketCount
     *  For Binary packs:            binaryFeaturesCount * 2 (binIdx)
     *  For ExclusiveFeaturesBundle: bucketCount for all used features
     *  For FeaturesGroup:           bucketCount for all grouped features
     */
    TArray2D<TVector<TBucketPairWeightStatistics>> PairWeightStatistics; // [leafCount][leafCount][statsCount]

    TSplitEnsembleSpec SplitEnsembleSpec;

public:
    SAVELOAD(DerSums, PairWeightStatistics, SplitEnsembleSpec);

    void Add(const TPairwiseStats& rhs);
};


// TGetBucketFunc is of type ui32(ui32 docId)
template <class TGetBucketFunc>
inline TVector<TVector<double>> ComputeDerSums(
    TConstArrayRef<double> weightedDerivativesData,
    int leafCount,
    int bucketCount,
    const TVector<TIndexType>& leafIndices,
    TGetBucketFunc getBucketFunc,
    NCB::TIndexRange<int> docIndexRange
) {
    TVector<TVector<double>> derSums(leafCount, TVector<double>(bucketCount));

    for (int docId : docIndexRange.Iter()) {
        const ui32 leafIndex = leafIndices[docId];
        const ui32 bucketIndex = getBucketFunc((ui32)docId);
        derSums[leafIndex][bucketIndex] += weightedDerivativesData[docId];
    }
    return derSums;
}

// TGetBucketFunc is of type ui32(ui32 docId)
template <class TGetBucketFunc>
inline TArray2D<TVector<TBucketPairWeightStatistics>> ComputePairWeightStatistics(
    const TFlatPairsInfo& pairs,
    int leafCount,
    int bucketCount,
    const TVector<TIndexType>& leafIndices,
    TGetBucketFunc getBucketFunc,
    NCB::TIndexRange<int> pairIndexRange
) {
    TArray2D<TVector<TBucketPairWeightStatistics>> weightSums(leafCount, leafCount);
    weightSums.FillEvery(TVector<TBucketPairWeightStatistics>(bucketCount));
    for (size_t pairIdx : pairIndexRange.Iter()) {
        const auto winnerIdx = pairs[pairIdx].WinnerId;
        const auto loserIdx = pairs[pairIdx].LoserId;
        if (winnerIdx == loserIdx) {
            continue;
        }
        const size_t winnerBucketId = getBucketFunc(winnerIdx);
        const auto winnerLeafId = leafIndices[winnerIdx];
        const size_t loserBucketId = getBucketFunc(loserIdx);
        const auto loserLeafId = leafIndices[loserIdx];
        const float weight = pairs[pairIdx].Weight;
        if (winnerBucketId > loserBucketId) {
            weightSums[loserLeafId][winnerLeafId][loserBucketId].SmallerBorderWeightSum -= weight;
            weightSums[loserLeafId][winnerLeafId][winnerBucketId].GreaterBorderRightWeightSum -= weight;
        } else {
            weightSums[winnerLeafId][loserLeafId][winnerBucketId].SmallerBorderWeightSum -= weight;
            weightSums[winnerLeafId][loserLeafId][loserBucketId].GreaterBorderRightWeightSum -= weight;
        }
    }

    return weightSums;
}

// TGetBinaryFeaturesPack is of type TBinaryFeaturesPack(ui32 docId)
template <class TGetBinaryFeaturesPack>
inline TArray2D<TVector<TBucketPairWeightStatistics>> ComputePairWeightStatisticsForBinaryFeaturesPacks(
    const TFlatPairsInfo& pairs,
    int leafCount,
    int bucketCount,
    const TVector<TIndexType>& leafIndices,
    TGetBinaryFeaturesPack getBinaryFeaturesPack,
    NCB::TIndexRange<int> pairIndexRange
) {
    const int binaryFeaturesCount = (int)GetValueBitCount(bucketCount - 1);

    TArray2D<TVector<TBucketPairWeightStatistics>> weightSums(leafCount, leafCount);
    weightSums.FillEvery(TVector<TBucketPairWeightStatistics>(2 * binaryFeaturesCount));
    for (size_t pairIdx : pairIndexRange.Iter()) {
        const auto winnerIdx = pairs[pairIdx].WinnerId;
        const auto loserIdx = pairs[pairIdx].LoserId;
        if (winnerIdx == loserIdx) {
            continue;
        }
        const NCB::TBinaryFeaturesPack winnerFeaturesPack = getBinaryFeaturesPack(winnerIdx);
        const auto winnerLeafId = leafIndices[winnerIdx];
        const NCB::TBinaryFeaturesPack loserFeaturesPack = getBinaryFeaturesPack(loserIdx);
        const auto loserLeafId = leafIndices[loserIdx];
        const float weight = pairs[pairIdx].Weight;

        for (auto bitIndex : xrange<NCB::TBinaryFeaturesPack>(binaryFeaturesCount)) {
            auto winnerBit = (winnerFeaturesPack >> bitIndex) & 1;
            auto loserBit = (loserFeaturesPack >> bitIndex) & 1;

            if (winnerBit > loserBit) {
                weightSums[loserLeafId][winnerLeafId][2 * bitIndex].SmallerBorderWeightSum -= weight;
                weightSums[loserLeafId][winnerLeafId][2 * bitIndex + 1].GreaterBorderRightWeightSum -= weight;
            } else {
                auto winnerBucketId = 2 * bitIndex + winnerBit;
                weightSums[winnerLeafId][loserLeafId][winnerBucketId].SmallerBorderWeightSum -= weight;
                auto loserBucketId = 2 * bitIndex + loserBit;
                weightSums[winnerLeafId][loserLeafId][loserBucketId].GreaterBorderRightWeightSum -= weight;
            }
        }
    }

    return weightSums;
}


// TGetExclusiveFeaturesBundleValue is of type TBundle(ui32 docId)
template <class TGetExclusiveFeaturesBundleValue>
inline TArray2D<TVector<TBucketPairWeightStatistics>> ComputePairWeightStatisticsForExclusiveFeaturesBundle(
    ui32 oneHotMaxSize,
    const TFlatPairsInfo& pairs,
    int leafCount,
    const TVector<TIndexType>& leafIndices,
    const NCB::TExclusiveFeaturesBundle& exclusiveFeaturesBundle,
    TGetExclusiveFeaturesBundleValue getExclusiveFeaturesBundleValue,
    NCB::TIndexRange<int> pairIndexRange
) {
    size_t totalBucketCount = 0;
    TVector<bool> calcStatsForBundlePart; // don't calc for cat features that are not one hot
    calcStatsForBundlePart.yresize(exclusiveFeaturesBundle.Parts.size());

    for (auto bundlePartIdx : xrange(exclusiveFeaturesBundle.Parts.size())) {
        const auto& bundlePart = exclusiveFeaturesBundle.Parts[bundlePartIdx];
        if (UseForCalcScores(bundlePart, oneHotMaxSize)) {
            totalBucketCount += bundlePart.Bounds.GetSize() + 1;
            calcStatsForBundlePart[bundlePartIdx] = true;
        } else {
            calcStatsForBundlePart[bundlePartIdx] = false;
        }
    }

    TArray2D<TVector<TBucketPairWeightStatistics>> weightSums(leafCount, leafCount);
    weightSums.FillEvery(TVector<TBucketPairWeightStatistics>(totalBucketCount));
    for (size_t pairIdx : pairIndexRange.Iter()) {
        const auto winnerIdx = pairs[pairIdx].WinnerId;
        const auto loserIdx = pairs[pairIdx].LoserId;
        if (winnerIdx == loserIdx) {
            continue;
        }
        const ui32 winnerBundleValue = getExclusiveFeaturesBundleValue(winnerIdx);
        const auto winnerLeafId = leafIndices[winnerIdx];
        const ui32 loserBundleValue = getExclusiveFeaturesBundleValue(loserIdx);
        const auto loserLeafId = leafIndices[loserIdx];
        const float weight = pairs[pairIdx].Weight;

        ui32 bucketOffset = 0;
        for (auto bundlePartIdx : xrange(exclusiveFeaturesBundle.Parts.size())) {
            if (!calcStatsForBundlePart[bundlePartIdx]) {
                continue;
            }

            NCB::TBoundsInBundle boundsInBundle = exclusiveFeaturesBundle.Parts[bundlePartIdx].Bounds;

            auto winnerBucketId = NCB::GetBinFromBundle<ui32>(winnerBundleValue, boundsInBundle);
            auto loserBucketId = NCB::GetBinFromBundle<ui32>(loserBundleValue, boundsInBundle);

            if (winnerBucketId > loserBucketId) {
                weightSums[loserLeafId][winnerLeafId][bucketOffset + loserBucketId].SmallerBorderWeightSum
                    -= weight;
                weightSums[loserLeafId][winnerLeafId][bucketOffset + winnerBucketId].GreaterBorderRightWeightSum
                    -= weight;
            } else {
                weightSums[winnerLeafId][loserLeafId][bucketOffset + winnerBucketId].SmallerBorderWeightSum
                    -= weight;
                weightSums[winnerLeafId][loserLeafId][bucketOffset + loserBucketId].GreaterBorderRightWeightSum
                    -= weight;
            }

            bucketOffset += boundsInBundle.GetSize() + 1;
        }
    }

    return weightSums;
}


// TGetFeaturesGroupValue is of type TGroupValue(ui32 docId)
template <class TGetFeaturesGroupValue>
inline TArray2D<TVector<TBucketPairWeightStatistics>> ComputePairWeightStatisticsForFeaturesGroup(
    const TFlatPairsInfo& pairs,
    int leafCount,
    const TVector<TIndexType>& leafIndices,
    const NCB::TFeaturesGroup& featuresGroup,
    TGetFeaturesGroupValue getFeaturesGroupValue,
    NCB::TIndexRange<int> pairIndexRange
) {
    TArray2D<TVector<TBucketPairWeightStatistics>> weightSums(leafCount, leafCount);
    weightSums.FillEvery(TVector<TBucketPairWeightStatistics>(featuresGroup.TotalBucketCount));
    for (size_t pairIdx : pairIndexRange.Iter()) {
        const auto winnerIdx = pairs[pairIdx].WinnerId;
        const auto loserIdx = pairs[pairIdx].LoserId;
        if (winnerIdx == loserIdx) {
            continue;
        }
        const auto winnerGroupValue = getFeaturesGroupValue(winnerIdx);
        const auto winnerLeafId = leafIndices[winnerIdx];
        const auto loserGroupValue = getFeaturesGroupValue(loserIdx);
        const auto loserLeafId = leafIndices[loserIdx];
        const float weight = pairs[pairIdx].Weight;

        ui32 bucketOffset = 0;
        for (auto partIdx : xrange(featuresGroup.Parts.size())) {
            auto winnerBucketId = NCB::GetPartValueFromGroup(winnerGroupValue, partIdx);
            auto loserBucketId = NCB::GetPartValueFromGroup(loserGroupValue, partIdx);

            if (winnerBucketId > loserBucketId) {
                weightSums[loserLeafId][winnerLeafId][bucketOffset + loserBucketId].SmallerBorderWeightSum
                    -= weight;
                weightSums[loserLeafId][winnerLeafId][bucketOffset + winnerBucketId].GreaterBorderRightWeightSum
                    -= weight;
            } else {
                weightSums[winnerLeafId][loserLeafId][bucketOffset + winnerBucketId].SmallerBorderWeightSum
                    -= weight;
                weightSums[winnerLeafId][loserLeafId][bucketOffset + loserBucketId].GreaterBorderRightWeightSum
                    -= weight;
            }

            bucketOffset += featuresGroup.Parts[partIdx].BucketCount;
        }
    }

    return weightSums;
}


template <typename TBucket, typename TGetBucketFunc>
inline void ComputePairwiseStats(
    ESplitEnsembleType splitEnsembleType,
    TConstArrayRef<double> weightedDerivatives,
    const TFlatPairsInfo& pairs,
    int leafCount,
    int bucketCount,
    ui32 oneHotMaxSize,
    const TVector<TIndexType>& leafIndices,

    // used only if SplitEnsembleType == ESplitEnsembleType::ExclusiveBundle
    TMaybe<const NCB::TExclusiveFeaturesBundle*> exclusiveFeaturesBundle,
    // used only if SplitEnsembleType == ESplitEnsembleType::FeaturesGroup
    TMaybe<const NCB::TFeaturesGroup*> featuresGroup,
    NCB::TIndexRange<int> docIndexRange,
    NCB::TIndexRange<int> pairIndexRange,
    TGetBucketFunc&& getBucketFunc,
    TPairwiseStats* output
) {
    output->DerSums = ComputeDerSums(
        weightedDerivatives,
        leafCount,
        bucketCount,
        leafIndices,
        getBucketFunc,
        docIndexRange
    );

    switch (splitEnsembleType) {
        case ESplitEnsembleType::OneFeature:
            output->PairWeightStatistics = ComputePairWeightStatistics(
                pairs,
                leafCount,
                bucketCount,
                leafIndices,
                getBucketFunc,
                pairIndexRange
            );
            break;
        case ESplitEnsembleType::BinarySplits:
            output->PairWeightStatistics = ComputePairWeightStatisticsForBinaryFeaturesPacks(
                pairs,
                leafCount,
                bucketCount,
                leafIndices,
                getBucketFunc,
                pairIndexRange
            );
            break;
        case ESplitEnsembleType::ExclusiveBundle:
            output->PairWeightStatistics = ComputePairWeightStatisticsForExclusiveFeaturesBundle(
                oneHotMaxSize,
                pairs,
                leafCount,
                leafIndices,
                **exclusiveFeaturesBundle,
                getBucketFunc,
                pairIndexRange
            );
            break;
        case ESplitEnsembleType::FeaturesGroup:
            output->PairWeightStatistics = ComputePairWeightStatisticsForFeaturesGroup(
                pairs,
                leafCount,
                leafIndices,
                **featuresGroup,
                getBucketFunc,
                pairIndexRange
            );
            break;
    }
}


template <class TColumn>
inline void ComputePairwiseStats(
    const TCalcScoreFold& fold,
    TConstArrayRef<double> weightedDerivatives,
    const TFlatPairsInfo& pairs,
    int leafCount,
    int bucketCount,
    ui32 oneHotMaxSize,

    // used only if splitEnsembleType == ESplitEnsembleType::ExclusiveBundle
    TMaybe<const NCB::TExclusiveFeaturesBundle*> exclusiveFeaturesBundle,
    // used only if SplitEnsembleType == ESplitEnsembleType::FeaturesGroup
    TMaybe<const NCB::TFeaturesGroup*> featuresGroup,
    const TColumn& column,
    bool isEstimatedData,
    bool isOnlineEstimatedData,
    NCB::TIndexRange<int> docIndexRange,
    NCB::TIndexRange<int> pairIndexRange,
    TPairwiseStats* output
) {
    ESplitEnsembleType splitEnsembleType;
    const auto featureValuesType = column.GetType();
    if (featureValuesType == NCB::EFeatureValuesType::BinaryPack) {
        splitEnsembleType = ESplitEnsembleType::BinarySplits;
    } else if (featureValuesType == NCB::EFeatureValuesType::ExclusiveFeatureBundle) {
        splitEnsembleType = ESplitEnsembleType::ExclusiveBundle;
    } else if (featureValuesType == NCB::EFeatureValuesType::FeaturesGroup) {
        splitEnsembleType = ESplitEnsembleType::FeaturesGroup;
    } else {
        splitEnsembleType = ESplitEnsembleType::OneFeature;
    }

    using TDenseColumnData = NCB::TCompressedValuesHolderImpl<TColumn>;

    if (const auto* denseColumnData = dynamic_cast<const TDenseColumnData*>(&column)) {
        const ui32* bucketIndexing = nullptr;
        if (isEstimatedData) {
            if (isOnlineEstimatedData) {
                bucketIndexing = fold.IndexInFold.data();
            } else {
                bucketIndexing = fold.GetLearnPermutationOfflineEstimatedFeaturesSubset().data();
            }
        } else {
            bucketIndexing = fold.LearnPermutationFeaturesSubset.Get<NCB::TIndexedSubset<ui32>>().data();
        }

        const TCompressedArray& compressedArray = *denseColumnData->GetCompressedData().GetSrc();

        compressedArray.DispatchBitsPerKeyToDataType(
            "ComputePairwiseStats",
            [&] (const auto* bucketSrcData) {
                ComputePairwiseStats<decltype(*bucketSrcData)>(
                    splitEnsembleType,
                    weightedDerivatives,
                    pairs,
                    leafCount,
                    bucketCount,
                    oneHotMaxSize,
                    fold.Indices,
                    exclusiveFeaturesBundle,
                    featuresGroup,
                    docIndexRange,
                    pairIndexRange,
                    [bucketSrcData, bucketIndexing](ui32 docIdx) {
                        return bucketSrcData[bucketIndexing[docIdx]];
                    },
                    output);
            });
    } else {
        CB_ENSURE_INTERNAL(false, "ComputePairwiseStats: unsupported column type");
    }
}

class TPairwiseScoreCalcer final : public IScoreCalcer {
public:
    void SetSplitsCount(int splitsCount) override {
        IScoreCalcer::SetSplitsCount(splitsCount);
        Scores.resize(splitsCount);
    }

    void CalculateScore(
        int splitIdx,
        TConstArrayRef<double> avrg,
        TConstArrayRef<double> sumDer,
        const TArray2D<double>& sumWeights);

    TVector<double> GetScores() const override {
        return Scores;
    }

private:
    TVector<double> Scores;
};

void CalculatePairwiseScore(
    const TPairwiseStats& pairwiseStats,
    int bucketCount,
    float l2DiagReg,
    float pairwiseBucketWeightPriorReg,
    ui32 oneHotMaxSize,
    TPairwiseScoreCalcer* scoreCalcer
);
