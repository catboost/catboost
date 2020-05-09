#include "scoring.h"

#include "calc_score_cache.h"
#include "fold.h"
#include "index_calcer.h"
#include "leafwise_scoring.h"
#include "pairwise_scoring.h"
#include "split.h"
#include "monotonic_constraint_utils.h"
#include "tensor_search_helpers.h"

#include <catboost/libs/data/objects.h>
#include <catboost/libs/helpers/map_merge.h>
#include <catboost/private/libs/algo_helpers/online_predictor.h>
#include <catboost/private/libs/algo_helpers/scoring_helpers.h>
#include <catboost/private/libs/data_types/pair.h>
#include <catboost/private/libs/index_range/index_range.h>
#include <catboost/private/libs/options/catboost_options.h>

#include <library/cpp/threading/local_executor/local_executor.h>
#include <library/cpp/dot_product/dot_product.h>

#include <util/generic/array_ref.h>

#include <type_traits>


using namespace NCB;


namespace {

    // Statistics (sums for score calculation) are stored in an array.
    // This class helps navigating in this array.
    struct TStatsIndexer {
    public:
        const int BucketCount;

    public:
        explicit TStatsIndexer(int bucketCount)
            : BucketCount(bucketCount) {
        }

        int CalcSize(int depth) const {
            return (1U << depth) * BucketCount;
        }

        int GetIndex(int leafIndex, int bucketIndex) const {
            return BucketCount * leafIndex + bucketIndex;
        }
    };
}


// Helper function for calculating index of leaf for each document given a new split.
// Calculates indices when a permutation is given.
template <typename TBucketIndexType, typename TFullIndexType>
inline static void SetSingleIndex(
    const TCalcScoreFold& fold,
    const TStatsIndexer& indexer,
    TBucketIndexType* bucketIndex,
    const ui32* bucketIndexing, // can be nullptr for simple case, use bucketBeginOffset instead then
    const int bucketBeginOffset,
    const int permBlockSize,
    NCB::TIndexRange<int> docIndexRange, // aligned by permutation blocks in docPermutation
    TVector<TFullIndexType>* singleIdx // already of proper size
) {
    const int docCount = fold.GetDocCount();
    const TIndexType* indices = GetDataPtr(fold.Indices);
    const TArrayRef<TFullIndexType> singleIdxRef(*singleIdx);

    if (bucketIndexing == nullptr) {
        for (int doc : docIndexRange.Iter()) {
            singleIdxRef[doc] = indexer.GetIndex(indices[doc], bucketIndex[bucketBeginOffset + doc]);
        }
    } else if (permBlockSize > 1) {
        const int blockCount = (docCount + permBlockSize - 1) / permBlockSize;
        Y_ASSERT(
            (static_cast<int>(bucketIndexing[0]) / permBlockSize + 1 == blockCount)
            || (static_cast<int>(bucketIndexing[0]) + permBlockSize - 1
            == static_cast<int>(bucketIndexing[permBlockSize - 1]))
        );
        int blockStart = docIndexRange.Begin;
        while (blockStart < docIndexRange.End) {
            const int blockIdx = static_cast<int>(bucketIndexing[blockStart]) / permBlockSize;
            const int nextBlockStart = Min(
                blockStart + (blockIdx + 1 == blockCount ? docCount - blockIdx * permBlockSize : permBlockSize),
                docIndexRange.End
            );
            const int originalBlockIdx = static_cast<int>(bucketIndexing[blockStart]);
            for (int doc = blockStart; doc < nextBlockStart; ++doc) {
                const int originalDocIdx = originalBlockIdx + doc - blockStart;
                singleIdxRef[doc] = indexer.GetIndex(indices[doc], bucketIndex[originalDocIdx]);
            }
            blockStart = nextBlockStart;
        }
    } else {
        for (int doc : docIndexRange.Iter()) {
            const ui32 originalDocIdx = bucketIndexing[doc];
            singleIdxRef[doc] = indexer.GetIndex(indices[doc], bucketIndex[originalDocIdx]);
        }
    }
}


static void GetIndexingParams(
    const TCalcScoreFold& fold,
    bool isEstimatedData,
    bool isOnlineData,
    const ui32** objectIndexing,
    int* beginOffset,
    int* permutationBlockSize
) {
    if (isOnlineData) {
        const bool simpleIndexing = fold.OnlineDataPermutationBlockSize == fold.GetDocCount();
        *objectIndexing = simpleIndexing ? nullptr : GetDataPtr(fold.IndexInFold);
        *beginOffset = 0;
        *permutationBlockSize = fold.OnlineDataPermutationBlockSize;
    } else if (isEstimatedData) {
        *objectIndexing = fold.GetLearnPermutationOfflineEstimatedFeaturesSubset().data();
        *beginOffset = 0;
        *permutationBlockSize = FoldPermutationBlockSizeNotSet;
    } else {
        const bool simpleIndexing = fold.MainDataPermutationBlockSize == fold.GetDocCount();
        *objectIndexing = simpleIndexing
            ? nullptr
            : fold.LearnPermutationFeaturesSubset.Get<TIndexedSubset<ui32>>().data();
        *beginOffset = simpleIndexing ? fold.FeaturesSubsetBegin : 0;
        *permutationBlockSize = fold.MainDataPermutationBlockSize;
    }
}


template <class TColumn, typename TFullIndexType>
inline static void BuildSingleIndex(
    const TCalcScoreFold& fold,
    const TColumn& column,
    bool isEstimatedData,
    bool isOnlineData,
    const TStatsIndexer& indexer,
    NCB::TIndexRange<int> docIndexRange,
    TVector<TFullIndexType>* singleIdx // already of proper size
) {
    if (const auto* denseColumnData
            = dynamic_cast<const TCompressedValuesHolderImpl<TColumn>*>(&column))
    {
        const ui32* objectIndexing;
        int beginOffset;
        int permutationBlockSize;
        GetIndexingParams(
            fold,
            isEstimatedData,
            isOnlineData,
            &objectIndexing,
            &beginOffset,
            &permutationBlockSize
        );

        const TCompressedArray& compressedArray = *denseColumnData->GetCompressedData().GetSrc();

        compressedArray.DispatchBitsPerKeyToDataType(
            "BuildSingleIndex",
            [&] (const auto* histogram) {
                SetSingleIndex(
                    fold,
                    indexer,
                    histogram,
                    objectIndexing,
                    beginOffset,
                    permutationBlockSize,
                    docIndexRange,
                    singleIdx
                );
            }
        );
    } else {
        Y_FAIL("BuildSingleIndex: unexpected column type");
    }
}


// Calculate index of leaf for each document given a new split ensemble.
template <typename TFullIndexType>
inline static void BuildSingleIndex(
    const TCalcScoreFold& fold,
    const TQuantizedForCPUObjectsDataProvider& objectsDataProvider,
    const std::tuple<const TOnlineCTRHash&, const TOnlineCTRHash&>& allCtrs,
    const TSplitEnsemble& splitEnsemble,
    const TStatsIndexer& indexer,
    NCB::TIndexRange<int> docIndexRange,
    TVector<TFullIndexType>* singleIdx // already of proper size
) {
    if (splitEnsemble.IsSplitOfType(ESplitType::OnlineCtr)) {
        const TCtr& ctr = splitEnsemble.SplitCandidate.Ctr;
        const ui32* objectIndexing;
        int beginOffset;
        int permutationBlockSize;
        GetIndexingParams(
            fold,
            /*isEstimatedData*/ false,
            /*isOnlineData*/true,
            &objectIndexing,
            &beginOffset,
            &permutationBlockSize
        );

        SetSingleIndex(
            fold,
            indexer,
            GetCtr(allCtrs, ctr.Projection).Feature[ctr.CtrIdx][ctr.TargetBorderIdx][ctr.PriorIdx].data(),
            objectIndexing,
            beginOffset,
            permutationBlockSize,
            docIndexRange,
            singleIdx
        );
    } else {
        auto buildSingleIndexFunc = [&] (const auto& column) {
            BuildSingleIndex(
                fold,
                column,
                splitEnsemble.IsEstimated,
                splitEnsemble.IsOnlineEstimated,
                indexer,
                docIndexRange,
                singleIdx
            );
        };

        switch (splitEnsemble.Type) {
            case ESplitEnsembleType::OneFeature:
                {
                    const auto& splitCandidate = splitEnsemble.SplitCandidate;
                    if ((splitCandidate.Type == ESplitType::FloatFeature) ||
                        (splitCandidate.Type == ESplitType::EstimatedFeature))
                    {
                        buildSingleIndexFunc(
                            **objectsDataProvider.GetNonPackedFloatFeature((ui32)splitCandidate.FeatureIdx)
                        );
                    } else {
                        Y_ASSERT(splitCandidate.Type == ESplitType::OneHotFeature);
                        buildSingleIndexFunc(
                           **objectsDataProvider.GetNonPackedCatFeature((ui32)splitCandidate.FeatureIdx)
                        );
                    }
                }
                break;
            case ESplitEnsembleType::BinarySplits:
                buildSingleIndexFunc(
                    objectsDataProvider.GetBinaryFeaturesPack(splitEnsemble.BinarySplitsPackRef.PackIdx)
                );
                break;
            case ESplitEnsembleType::ExclusiveBundle:
                buildSingleIndexFunc(
                    objectsDataProvider.GetExclusiveFeaturesBundle(
                        splitEnsemble.ExclusiveFeaturesBundleRef.BundleIdx
                    )
                );
                break;
            case ESplitEnsembleType::FeaturesGroup:
                // FeaturesGroups are implemented only in leafwise scoring now
                Y_UNREACHABLE();
                break;
        }
    }
}


// Update bootstraped sums on docIndexRange in a bucket
template <typename TFullIndexType>
inline static void UpdateWeighted(
    const TVector<TFullIndexType>& singleIdx,
    const double* weightedDer,
    const float* sampleWeights,
    NCB::TIndexRange<int> docIndexRange,
    TBucketStats* stats
) {
    for (int doc : docIndexRange.Iter()) {
        TBucketStats& leafStats = stats[singleIdx[doc]];
        leafStats.SumWeightedDelta += weightedDer[doc];
        leafStats.SumWeight += sampleWeights[doc];
    }
}


// Update not bootstraped sums on docIndexRange in a bucket
template <typename TFullIndexType>
inline static void UpdateDeltaCount(
    const TVector<TFullIndexType>& singleIdx,
    const double* derivatives,
    const float* learnWeights,
    NCB::TIndexRange<int> docIndexRange,
    TBucketStats* stats
) {
    if (learnWeights == nullptr) {
        for (int doc : docIndexRange.Iter()) {
            TBucketStats& leafStats = stats[singleIdx[doc]];
            leafStats.SumDelta += derivatives[doc];
            leafStats.Count += 1;
        }
    } else {
        for (int doc : docIndexRange.Iter()) {
            TBucketStats& leafStats = stats[singleIdx[doc]];
            leafStats.SumDelta += derivatives[doc];
            leafStats.Count += learnWeights[doc];
        }
    }
}


template <typename TFullIndexType>
inline static void CalcStatsKernel(
    bool isCaching,
    const TVector<TFullIndexType>& singleIdx,
    const TCalcScoreFold& fold,
    bool isPlainMode,
    const TStatsIndexer& indexer,
    int depth,
    const TCalcScoreFold::TBodyTail& bt,
    int dim,
    NCB::TIndexRange<int> docIndexRange,
    TBucketStats* stats
) {
    Y_ASSERT(!isCaching || depth > 0);
    if (isCaching) {
        Fill(
            stats + indexer.CalcSize(depth - 1),
            stats + indexer.CalcSize(depth),
            TBucketStats{0, 0, 0, 0}
        );
    } else {
        Fill(stats, stats + indexer.CalcSize(depth), TBucketStats{0, 0, 0, 0});
    }

    if (bt.TailFinish > docIndexRange.Begin) {
        const bool hasPairwiseWeights = !bt.PairwiseWeights.empty();
        const float* weightsData = hasPairwiseWeights ?
            GetDataPtr(bt.PairwiseWeights) : GetDataPtr(fold.LearnWeights);
        const float* sampleWeightsData = hasPairwiseWeights ?
            GetDataPtr(bt.SamplePairwiseWeights) : GetDataPtr(fold.SampleWeights);

        int tailFinishInRange = Min((int)bt.TailFinish, docIndexRange.End);

        if (isPlainMode) {
            UpdateWeighted(
                singleIdx,
                GetDataPtr(bt.SampleWeightedDerivatives[dim]),
                sampleWeightsData,
                NCB::TIndexRange<int>(docIndexRange.Begin, tailFinishInRange),
                stats
            );
        } else {
            if (bt.BodyFinish > docIndexRange.Begin) {
                UpdateDeltaCount(
                    singleIdx,
                    GetDataPtr(bt.WeightedDerivatives[dim]),
                    weightsData,
                    NCB::TIndexRange<int>(docIndexRange.Begin, Min((int)bt.BodyFinish, docIndexRange.End)),
                    stats
                );
            }
            if (tailFinishInRange > bt.BodyFinish) {
                UpdateWeighted(
                    singleIdx,
                    GetDataPtr(bt.SampleWeightedDerivatives[dim]),
                    sampleWeightsData,
                    NCB::TIndexRange<int>(Max((int)bt.BodyFinish, docIndexRange.Begin), tailFinishInRange),
                    stats
                );
            }
        }
    }
}

inline static void FixUpStats(
    int depth,
    const TStatsIndexer& indexer,
    bool selectedSplitValue,
    TBucketStats* stats
) {
    const int halfOfStats = indexer.CalcSize(depth - 1);
    if (selectedSplitValue == true) {
        for (int statIdx = 0; statIdx < halfOfStats; ++statIdx) {
            stats[statIdx].Remove(stats[statIdx + halfOfStats]);
        }
    } else {
        for (int statIdx = 0; statIdx < halfOfStats; ++statIdx) {
            stats[statIdx].Remove(stats[statIdx + halfOfStats]);
            DoSwap(stats[statIdx], stats[statIdx + halfOfStats]);
        }
    }
}


template <typename TFullIndexType, typename TIsCaching>
static void CalcStatsImpl(
    const TCalcScoreFold& fold,
    const TQuantizedForCPUObjectsDataProvider& objectsDataProvider,
    const TFlatPairsInfo& pairs,
    const std::tuple<const TOnlineCTRHash&, const TOnlineCTRHash&>& allCtrs,
    const TSplitEnsemble& splitEnsemble,
    const TStatsIndexer& indexer,
    const TIsCaching& /*isCaching*/,
    bool /*isPlainMode*/,
    ui32 oneHotMaxSize,
    int depth,
    int /*splitStatsCount*/,
    NPar::TLocalExecutor* localExecutor,
    TPairwiseStats* stats
) {
    const int approxDimension = fold.GetApproxDimension();
    const int leafCount = 1 << depth;

    Y_ASSERT(approxDimension == 1 && fold.GetBodyTailCount() == 1);

    const int docCount = fold.GetDocCount();
    auto weightedDerivativesData = MakeArrayRef(
        fold.BodyTailArr[0].WeightedDerivatives[0].data(),
        docCount
    );
    const auto blockCount = fold.GetCalcStatsIndexRanges().RangesCount();
    const auto docPart = CeilDiv(docCount, blockCount);

    const auto pairCount = pairs.ysize();
    const auto pairPart = CeilDiv(pairCount, blockCount);

    NCB::MapMerge(
        localExecutor,
        fold.GetCalcStatsIndexRanges(),
        /*mapFunc*/[&](NCB::TIndexRange<int> partIndexRange, TPairwiseStats* output) {
            Y_ASSERT(!partIndexRange.Empty());

            auto docIndexRange = NCB::TIndexRange<int>(
                Min(docCount, docPart * partIndexRange.Begin),
                Min(docCount, docPart * partIndexRange.End)
            );

            auto pairIndexRange = NCB::TIndexRange<int>(
                Min(pairCount, pairPart * partIndexRange.Begin),
                Min(pairCount, pairPart * partIndexRange.End)
            );

            auto computePairwiseStats = [&] (
                const auto& column,
                TMaybe<const TExclusiveFeaturesBundle*> exclusiveFeaturesBundle = Nothing(),
                TMaybe<const TFeaturesGroup*> featuresGroup = Nothing()
            ) {
                ComputePairwiseStats(
                    fold,
                    weightedDerivativesData,
                    pairs,
                    leafCount,
                    indexer.BucketCount,
                    oneHotMaxSize,
                    exclusiveFeaturesBundle,
                    featuresGroup,
                    column,
                    splitEnsemble.IsEstimated,
                    splitEnsemble.IsOnlineEstimated,
                    docIndexRange,
                    pairIndexRange,
                    output);
            };

            switch (splitEnsemble.Type) {
                case ESplitEnsembleType::OneFeature:
                    {
                        const auto& splitCandidate = splitEnsemble.SplitCandidate;
                        output->SplitEnsembleSpec = TSplitEnsembleSpec::OneSplit(splitCandidate.Type);

                        switch (splitCandidate.Type) {
                            case ESplitType::OnlineCtr:
                                {
                                    const TCtr& ctr = splitCandidate.Ctr;
                                    TConstArrayRef<ui8> buckets =
                                        GetCtr(allCtrs, ctr.Projection)
                                            .Feature[ctr.CtrIdx][ctr.TargetBorderIdx][ctr.PriorIdx];

                                    ComputePairwiseStats<ui8>(
                                        ESplitEnsembleType::OneFeature,
                                        weightedDerivativesData,
                                        pairs,
                                        leafCount,
                                        indexer.BucketCount,
                                        oneHotMaxSize,
                                        fold.Indices,
                                        /*exclusiveFeaturesBundle*/ Nothing(),
                                        /*featuresGroup*/ Nothing(),
                                        docIndexRange,
                                        pairIndexRange,
                                        [buckets](ui32 docIdx) { return buckets[docIdx]; },
                                        output);
                                }
                                break;
                            case ESplitType::FloatFeature:
                            case ESplitType::EstimatedFeature:
                                computePairwiseStats(
                                    **objectsDataProvider.GetNonPackedFloatFeature(
                                        (ui32)splitCandidate.FeatureIdx
                                    ));
                                break;
                            case ESplitType::OneHotFeature:
                                computePairwiseStats(
                                    **objectsDataProvider.GetNonPackedCatFeature(
                                        (ui32)splitCandidate.FeatureIdx
                                    ));
                                break;
                        }
                    }
                    break;
                case ESplitEnsembleType::BinarySplits:
                    output->SplitEnsembleSpec = TSplitEnsembleSpec::BinarySplitsPack();
                    computePairwiseStats(
                        objectsDataProvider.GetBinaryFeaturesPack(splitEnsemble.BinarySplitsPackRef.PackIdx)
                    );
                    break;
                case ESplitEnsembleType::ExclusiveBundle:
                    {
                        const auto bundleIdx = splitEnsemble.ExclusiveFeaturesBundleRef.BundleIdx;
                        const auto* bundleMetaData =
                            &(objectsDataProvider.GetExclusiveFeatureBundlesMetaData()[bundleIdx]);
                        output->SplitEnsembleSpec = TSplitEnsembleSpec::ExclusiveFeatureBundle(*bundleMetaData);

                        computePairwiseStats(
                            objectsDataProvider.GetExclusiveFeaturesBundle(bundleIdx),
                            bundleMetaData);
                    }
                    break;
                case ESplitEnsembleType::FeaturesGroup:
                    // FeaturesGroups are implemented only in leafwise scoring now
                    Y_UNREACHABLE();
                    break;
            }
        },
        /*mergeFunc*/[&](TPairwiseStats* output, TVector<TPairwiseStats>&& addVector) {
            for (const auto& addItem : addVector) {
                output->Add(addItem);
            }
        },
        stats
    );
}


template <typename TFullIndexType, typename TIsCaching>
static void CalcStatsImpl(
    const TCalcScoreFold& fold,
    const TQuantizedForCPUObjectsDataProvider& objectsDataProvider,
    const TFlatPairsInfo& /*pairs*/,
    const std::tuple<const TOnlineCTRHash&, const TOnlineCTRHash&>& allCtrs,
    const TSplitEnsemble& splitEnsemble,
    const TStatsIndexer& indexer,
    const TIsCaching& isCaching,
    bool isPlainMode,
    ui32 /*oneHotMaxSize*/,
    int depth,
    int splitStatsCount,
    NPar::TLocalExecutor* localExecutor,
    TBucketStatsRefOptionalHolder* stats
) {
    Y_ASSERT(!isCaching || depth > 0);

    const int docCount = fold.GetDocCount();

    TVector<TFullIndexType> singleIdx;
    singleIdx.yresize(docCount);

    const int statsCount = fold.GetBodyTailCount() * fold.GetApproxDimension() * splitStatsCount;
    const int filledSplitStatsCount = indexer.CalcSize(depth);

    // bodyFunc must accept (bodyTailIdx, dim, bucketStatsArrayBegin) params
    auto forEachBodyTailAndApproxDimension = [&](auto bodyFunc) {
        const int approxDimension = fold.GetApproxDimension();
        for (int bodyTailIdx : xrange(fold.GetBodyTailCount())) {
            for (int dim : xrange(approxDimension)) {
                bodyFunc(bodyTailIdx, dim, (bodyTailIdx * approxDimension + dim) * splitStatsCount);
            }
        }
    };

    NCB::MapMerge(
        localExecutor,
        fold.GetCalcStatsIndexRanges(),
        /*mapFunc*/[&](NCB::TIndexRange<int> indexRange, TBucketStatsRefOptionalHolder* output) {
            NCB::TIndexRange<int> docIndexRange = fold.HasQueryInfo() ?
                NCB::TIndexRange<int>(
                    fold.LearnQueriesInfo[indexRange.Begin].Begin,
                    (indexRange.End == 0) ? 0 : fold.LearnQueriesInfo[indexRange.End - 1].End
                )
                : indexRange;

            BuildSingleIndex(
                fold,
                objectsDataProvider,
                allCtrs,
                splitEnsemble,
                indexer,
                docIndexRange,
                &singleIdx
            );

            if (output->NonInited()) {
                (*output) = TBucketStatsRefOptionalHolder(statsCount);
            } else {
                Y_ASSERT(docIndexRange.Begin == 0);
            }

            forEachBodyTailAndApproxDimension(
                [&](int bodyTailIdx, int dim, int bucketStatsArrayBegin) {
                    TBucketStats* statsSubset = output->GetData().data() + bucketStatsArrayBegin;
                    CalcStatsKernel(
                        isCaching && (indexRange.Begin == 0),
                        singleIdx,
                        fold,
                        isPlainMode,
                        indexer,
                        depth,
                        fold.BodyTailArr[bodyTailIdx],
                        dim,
                        docIndexRange,
                        statsSubset
                    );
                }
            );
        },
        /*mergeFunc*/[&](
            TBucketStatsRefOptionalHolder* output,
            TVector<TBucketStatsRefOptionalHolder>&& addVector
        ) {
            forEachBodyTailAndApproxDimension(
                [&](int /*bodyTailIdx*/, int /*dim*/, int bucketStatsArrayBegin) {
                    TBucketStats* outputStatsSubset =
                        output->GetData().data() + bucketStatsArrayBegin;

                    for (const auto& addItem : addVector) {
                        const TBucketStats* addStatsSubset =
                            addItem.GetData().data() + bucketStatsArrayBegin;
                        for (size_t i : xrange(filledSplitStatsCount)) {
                            (outputStatsSubset + i)->Add(*(addStatsSubset + i));
                        }
                    }
                }
            );
        },
        stats
    );

    if (isCaching) {
        forEachBodyTailAndApproxDimension(
            [&](int /*bodyTailIdx*/, int /*dim*/, int bucketStatsArrayBegin) {
                TBucketStats* statsSubset = stats->GetData().data() + bucketStatsArrayBegin;
                FixUpStats(depth, indexer, fold.SmallestSplitSideValue, statsSubset);
            }
        );
    }
}


inline void UpdateSplitScore(
    bool isPlainMode,
    const TBucketStats& trueStats,
    const TBucketStats& falseStats,
    int splitIdx,
    IPointwiseScoreCalcer* scoreCalcer
) {
    if (isPlainMode) {
        scoreCalcer->AddLeafPlain(splitIdx, falseStats, trueStats);
    } else {
        scoreCalcer->AddLeafOrdered(splitIdx, falseStats, trueStats);
    }
}


/* This function calculates resulting sums for each split given statistics that are calculated for each bucket
 * of the histogram.
 */
template <typename TIsPlainMode, typename THaveMonotonicConstraints>
inline static void UpdateScores(
    const TBucketStats* stats,
    int leafCount,
    const TStatsIndexer& indexer,
    const TSplitEnsembleSpec& splitEnsembleSpec,
    double scaledL2Regularizer,
    TIsPlainMode isPlainMode,
    THaveMonotonicConstraints haveMonotonicConstraints,
    ui32 oneHotMaxSize,
    const TVector<int>& currTreeMonotonicConstraints,
    const TVector<int>& candidateSplitMonotonicConstraints,
    IPointwiseScoreCalcer* scoreCalcer
) {
    Y_ASSERT(haveMonotonicConstraints == !candidateSplitMonotonicConstraints.empty());
    // Used only if monotonic constraints are non trivial.
    TVector<TVector<double>> leafDeltas;
    TVector<TVector<double>> bodyLeafWeights;
    TVector<TVector<double>> tailLeafSumWeightedDers;
    TVector<TVector<double>> tailLeafWeights;
    TVector<int> leafsProcessed;
    if (haveMonotonicConstraints) {
        /* In this case updateSplitScoreClosure simply stores relevant statistics for every leaf and
         * every evaluated split. Then monotonization applies to leaf values and split score is calculated.
         * This implies unnecessary memory usage.
         */
        for (TVector<TVector<double>>* vec : {
            &leafDeltas, &bodyLeafWeights, &tailLeafSumWeightedDers, &tailLeafWeights
        }) {
            vec->resize(scoreCalcer->GetSplitsCount());
            for (auto& perLeafStats : *vec) {
                perLeafStats.resize(2 * leafCount);
            }
        }
        leafsProcessed.resize(scoreCalcer->GetSplitsCount());
    }
    const auto updateSplitScoreClosure = [&] (
        const TBucketStats& trueStats,
        const TBucketStats& falseStats,
        int splitIdx
    ) {
        if (!haveMonotonicConstraints) {
            UpdateSplitScore(
                isPlainMode,
                trueStats,
                falseStats,
                splitIdx,
                scoreCalcer
            );
        } else {
            auto currLeafId = leafsProcessed[splitIdx];
            Y_ASSERT(currLeafId < leafCount);
            for (const auto leafStats : {&falseStats, &trueStats}) {
                double bodyLeafWeight = 0.0;
                if (isPlainMode) {
                    bodyLeafWeight = leafStats->SumWeight;
                    leafDeltas[splitIdx][currLeafId] = CalcAverage(
                        leafStats->SumWeightedDelta,
                        bodyLeafWeight,
                        scaledL2Regularizer
                    );
                } else {
                    // compute leaf value using statistics of current BodyTail body:
                    bodyLeafWeight = leafStats->Count;
                    leafDeltas[splitIdx][currLeafId] = CalcAverage(
                        leafStats->SumDelta,
                        bodyLeafWeight,
                        scaledL2Regularizer
                    );
                }
                /* Note that the following lines perform a reduction from isotonic regression with
                 * l2-regularization to usual isotonic regression with properly modified weights and values.
                 */
                bodyLeafWeights[splitIdx][currLeafId] = bodyLeafWeight + scaledL2Regularizer;
                tailLeafWeights[splitIdx][currLeafId] = leafStats->SumWeight;
                tailLeafSumWeightedDers[splitIdx][currLeafId] = leafStats->SumWeightedDelta;
                currLeafId += leafCount;
            }
            leafsProcessed[splitIdx] += 1;
        }
    };

    for (int leaf = 0; leaf < leafCount; ++leaf) {
        const auto& getBucketStats = [stats, leaf, indexer] (int bucketIdx) {
            return stats[indexer.GetIndex(leaf, bucketIdx)];
        };
        CalcScoresForLeaf(
            splitEnsembleSpec,
            oneHotMaxSize,
            indexer.BucketCount,
            getBucketStats,
            updateSplitScoreClosure);
    }

    if (haveMonotonicConstraints) {
        Y_ASSERT(AllOf(leafDeltas, [=] (const auto& vec) {
            return vec.size() == SafeIntegerCast<size_t>(2 * leafCount);
        }));
        const THashSet<int> possibleNewSplitConstraints(
            candidateSplitMonotonicConstraints.begin(), candidateSplitMonotonicConstraints.end()
        );
        THashMap<int, TVector<TVector<ui32>>> possibleLeafIndexOrders;
        auto monotonicConstraints = currTreeMonotonicConstraints;
        for (int newSplitMonotonicConstraint : possibleNewSplitConstraints) {
            monotonicConstraints.push_back(newSplitMonotonicConstraint);
            possibleLeafIndexOrders[newSplitMonotonicConstraint] = BuildMonotonicLinearOrdersOnLeafs(
                monotonicConstraints
            );
            monotonicConstraints.pop_back();
        }
        for (int splitIdx : xrange(scoreCalcer->GetSplitsCount())) {
            const auto& indexOrder = possibleLeafIndexOrders[candidateSplitMonotonicConstraints[splitIdx]];
            for (const auto& monotonicSubtreeIndexOrder : indexOrder) {
                CalcOneDimensionalIsotonicRegression(
                    leafDeltas[splitIdx],
                    bodyLeafWeights[splitIdx],
                    monotonicSubtreeIndexOrder,
                    &leafDeltas[splitIdx]
                );
            }
            for (int leafIndex = 0; leafIndex < 2 * leafCount; ++leafIndex) {
                const double leafDelta = leafDeltas[splitIdx][leafIndex];
                const TBucketStats leafStats {
                    tailLeafSumWeightedDers[splitIdx][leafIndex],
                    tailLeafWeights[splitIdx][leafIndex],
                    0, // it is unused in following call
                    0  // it is unused in following call
                };
                scoreCalcer->AddLeaf(splitIdx, leafDelta, leafStats);
            }
        }
    }
}


static void CalculateNonPairwiseScore(
    const TCalcScoreFold& fold,
    const TFold& initialFold,
    const TSplitEnsembleSpec& splitEnsembleSpec,
    bool isPlainMode,
    const int leafCount,
    const float l2Regularizer,
    const ui32 oneHotMaxSize,
    const TStatsIndexer& indexer,
    const TBucketStats* splitStats,
    int splitStatsCount,
    const TVector<int>& currTreeMonotonicConstraints,
    const TVector<int>& candidateSplitMonotonicConstraints,
    IPointwiseScoreCalcer* scoreCalcer
) {
    const int approxDimension = fold.GetApproxDimension();
    const bool haveMonotonicConstraints = !candidateSplitMonotonicConstraints.empty();

    for (int bodyTailIdx = 0; bodyTailIdx < fold.GetBodyTailCount(); ++bodyTailIdx) {
        const double sumAllWeights = initialFold.BodyTailArr[bodyTailIdx].BodySumWeight;
        const int docCount = initialFold.BodyTailArr[bodyTailIdx].BodyFinish;
        const double scaledL2Regularizer = l2Regularizer * (sumAllWeights / docCount);
        scoreCalcer->SetL2Regularizer(scaledL2Regularizer);
        const auto updateScores = [&] (auto isPlainMode, auto haveMonotonicConstraints, const auto* stats) {
            UpdateScores(
                stats,
                leafCount,
                indexer,
                splitEnsembleSpec,
                scaledL2Regularizer,
                isPlainMode,
                haveMonotonicConstraints,
                oneHotMaxSize,
                currTreeMonotonicConstraints,
                candidateSplitMonotonicConstraints,
                scoreCalcer
            );
        };
        for (int dim = 0; dim < approxDimension; ++dim) {
            const TBucketStats* stats = splitStats
                + (bodyTailIdx * approxDimension + dim) * splitStatsCount;
            if (isPlainMode && haveMonotonicConstraints) {
                updateScores(std::true_type(), std::true_type(), stats);
            } else if (isPlainMode && !haveMonotonicConstraints) {
                updateScores(std::true_type(), std::false_type(), stats);
            } else if (!isPlainMode && haveMonotonicConstraints) {
                updateScores(std::false_type(), std::true_type(), stats);
            } else {
                updateScores(std::false_type(), std::false_type(), stats);
            }
        }
    }
}


void CalcStatsAndScores(
    const TQuantizedForCPUObjectsDataProvider& objectsDataProvider,
    const std::tuple<const TOnlineCTRHash&, const TOnlineCTRHash&>& allCtrs,
    const TCalcScoreFold& fold,
    const TCalcScoreFold& prevLevelData,
    const TFold* initialFold,
    const TFlatPairsInfo& pairs,
    const NCatboostOptions::TCatBoostOptions& fitParams,
    const TCandidateInfo& candidateInfo,
    int depth,
    bool useTreeLevelCaching,
    const TVector<int>& currTreeMonotonicConstraints,
    const TMap<ui32, int>& monotonicConstraints,
    NPar::TLocalExecutor* localExecutor,
    TBucketStatsCache* statsFromPrevTree,
    TStats3D* stats3d,
    TPairwiseStats* pairwiseStats,
    IScoreCalcer* scoreCalcer
) {
    CB_ENSURE(
        stats3d || pairwiseStats || scoreCalcer,
        "stats3d, pairwiseStats, and scoreCalcer are empty - nothing to calculate"
    );
    CB_ENSURE(!scoreCalcer || initialFold, "initialFold must be non-nullptr for scores calculation");

    const auto& splitEnsemble = candidateInfo.SplitEnsemble;
    const bool isPairwiseScoring = IsPairwiseScoring(fitParams.LossFunctionDescription->GetLossFunction());

    const int bucketCount = GetBucketCount(
        splitEnsemble,
        *objectsDataProvider.GetQuantizedFeaturesInfo(),
        objectsDataProvider.GetPackedBinaryFeaturesSize(),
        objectsDataProvider.GetExclusiveFeatureBundlesMetaData(),
        objectsDataProvider.GetFeaturesGroupsMetaData()
    );
    const TStatsIndexer indexer(bucketCount);
    const int fullIndexBitCount = depth + GetValueBitCount(bucketCount - 1);
    const bool isPlainMode = IsPlainMode(fitParams.BoostingOptions->BoostingType);

    const float l2Regularizer = static_cast<const float>(fitParams.ObliviousTreeOptions->L2Reg);
    const ui32 oneHotMaxSize = fitParams.CatFeatureParams.Get().OneHotMaxSize.Get();

    decltype(auto) selectCalcStatsImpl = [&] (
        auto isCaching,
        const TCalcScoreFold& fold,
        int splitStatsCount,
        auto* stats
    ) {
        if (fullIndexBitCount <= 8) {
            CalcStatsImpl<ui8>(
                fold,
                objectsDataProvider,
                pairs,
                allCtrs,
                splitEnsemble,
                indexer,
                isCaching,
                isPlainMode,
                oneHotMaxSize,
                depth,
                splitStatsCount,
                localExecutor,
                stats
            );
        } else if (fullIndexBitCount <= 16) {
            CalcStatsImpl<ui16>(
                fold,
                objectsDataProvider,
                pairs,
                allCtrs,
                splitEnsemble,
                indexer,
                isCaching,
                isPlainMode,
                oneHotMaxSize,
                depth,
                splitStatsCount,
                localExecutor,
                stats
            );
        } else if (fullIndexBitCount <= 32) {
            CalcStatsImpl<ui32>(
                fold,
                objectsDataProvider,
                pairs,
                allCtrs,
                splitEnsemble,
                indexer,
                isCaching,
                isPlainMode,
                oneHotMaxSize,
                depth,
                splitStatsCount,
                localExecutor,
                stats
            );
        }
    };

    // Pairwise scoring doesn't use statistics from previous tree level
    if (isPairwiseScoring) {
        CB_ENSURE(!stats3d, "Pairwise scoring is incompatible with stats3d calculation");

        TPairwiseStats localPairwiseStats;
        if (pairwiseStats == nullptr) {
            pairwiseStats = &localPairwiseStats;
        }
        pairwiseStats->SplitEnsembleSpec = TSplitEnsembleSpec(
            splitEnsemble,
            objectsDataProvider.GetExclusiveFeatureBundlesMetaData(),
            objectsDataProvider.GetFeaturesGroupsMetaData()
        );

        selectCalcStatsImpl(/*isCaching*/ std::false_type(), fold, /*splitStatsCount*/0, pairwiseStats);

        if (scoreCalcer) {
            const float pairwiseBucketWeightPriorReg =
                static_cast<const float>(fitParams.ObliviousTreeOptions->PairwiseNonDiagReg);
            CalculatePairwiseScore(
                *pairwiseStats,
                bucketCount,
                l2Regularizer,
                pairwiseBucketWeightPriorReg,
                oneHotMaxSize,
                dynamic_cast<TPairwiseScoreCalcer*>(scoreCalcer)
            );
        }
    } else {
        CB_ENSURE(!pairwiseStats, "Per-object scoring is incompatible with pairwiseStats calculation");
        TBucketStatsRefOptionalHolder extOrInSplitStats;
        int splitStatsCount = 0;

        const auto& treeOptions = fitParams.ObliviousTreeOptions.Get();

        if (!useTreeLevelCaching) {
            splitStatsCount = indexer.CalcSize(depth);
            const int statsCount =
                fold.GetBodyTailCount() * fold.GetApproxDimension() * splitStatsCount;

            if (stats3d != nullptr) {
                stats3d->Stats.yresize(statsCount);
                stats3d->BucketCount = bucketCount;
                stats3d->MaxLeafCount = 1U << depth;
                stats3d->SplitEnsembleSpec = TSplitEnsembleSpec(
                    splitEnsemble,
                    objectsDataProvider.GetExclusiveFeatureBundlesMetaData(),
                    objectsDataProvider.GetFeaturesGroupsMetaData()
                );

                extOrInSplitStats = TBucketStatsRefOptionalHolder(stats3d->Stats);
            }
            selectCalcStatsImpl(
                /*isCaching*/ std::false_type(),
                fold,
                splitStatsCount,
                &extOrInSplitStats
            );
        } else {
            splitStatsCount = indexer.CalcSize(treeOptions.MaxDepth);
            bool areStatsDirty;

            // thread-safe access
            TVector<TBucketStats, TPoolAllocator>& splitStatsFromCache =
                statsFromPrevTree->GetStats(splitEnsemble, splitStatsCount, &areStatsDirty);
            extOrInSplitStats = TBucketStatsRefOptionalHolder(splitStatsFromCache);
            if (depth == 0 || areStatsDirty) {
                selectCalcStatsImpl(
                    /*isCaching*/ std::false_type(),
                    fold,
                    splitStatsCount,
                    &extOrInSplitStats
                );
            } else {
                selectCalcStatsImpl(
                    /*isCaching*/ std::true_type(),
                    prevLevelData,
                    splitStatsCount,
                    &extOrInSplitStats
                );
            }
            if (stats3d) {
                TBucketStatsCache::GetStatsInUse(fold.GetBodyTailCount() * fold.GetApproxDimension(),
                    splitStatsCount,
                    indexer.CalcSize(depth),
                    splitStatsFromCache
                ).swap(stats3d->Stats);
                stats3d->BucketCount = bucketCount;
                stats3d->MaxLeafCount = 1U << depth;
                stats3d->SplitEnsembleSpec = TSplitEnsembleSpec(
                    splitEnsemble,
                    objectsDataProvider.GetExclusiveFeatureBundlesMetaData(),
                    objectsDataProvider.GetFeaturesGroupsMetaData()
                );
            }
        }
        if (scoreCalcer) {
            const int leafCount = 1 << depth;
            TSplitEnsembleSpec splitEnsembleSpec(
                splitEnsemble,
                objectsDataProvider.GetExclusiveFeatureBundlesMetaData(),
                objectsDataProvider.GetFeaturesGroupsMetaData()
            );
            const int candidateSplitCount = CalcSplitsCount(
                splitEnsembleSpec, indexer.BucketCount, oneHotMaxSize
            );
            scoreCalcer->SetSplitsCount(candidateSplitCount);

            TVector<int> candidateSplitMonotonicConstraints;
            if (!monotonicConstraints.empty()) {
                candidateSplitMonotonicConstraints.resize(candidateSplitCount, 0);
                for (int splitIdx : xrange(candidateSplitCount)) {
                    const auto split = candidateInfo.GetSplit(
                        splitIdx, objectsDataProvider, oneHotMaxSize
                    );
                    if (split.Type == ESplitType::FloatFeature) {
                        Y_ASSERT(split.FeatureIdx >= 0);
                        if (monotonicConstraints.contains(split.FeatureIdx)) {
                            candidateSplitMonotonicConstraints[splitIdx] =
                                monotonicConstraints.at(split.FeatureIdx);
                        }
                    }
                }
            }

            CalculateNonPairwiseScore(
                fold,
                *initialFold,
                splitEnsembleSpec,
                isPlainMode,
                leafCount,
                l2Regularizer,
                oneHotMaxSize,
                indexer,
                extOrInSplitStats.GetData().data(),
                splitStatsCount,
                currTreeMonotonicConstraints,
                candidateSplitMonotonicConstraints,
                dynamic_cast<IPointwiseScoreCalcer*>(scoreCalcer)
            );
        }
    }
}

TVector<double> GetScores(
    const TStats3D& stats3d,
    int depth,
    double sumAllWeights,
    int allDocCount,
    const NCatboostOptions::TCatBoostOptions& fitParams
) {
    const TVector<TBucketStats>& bucketStats = stats3d.Stats;
    const int splitStatsCount = stats3d.BucketCount * stats3d.MaxLeafCount;
    const int bucketCount = stats3d.BucketCount;
    const float l2Regularizer = static_cast<const float>(fitParams.ObliviousTreeOptions->L2Reg);
    const ui32 oneHotMaxSize = fitParams.CatFeatureParams.Get().OneHotMaxSize.Get();
    const int leafCount = 1 << depth;
    const TStatsIndexer indexer(bucketCount);

    auto scoreCalcer = MakePointwiseScoreCalcer(fitParams.ObliviousTreeOptions->ScoreFunction);
    scoreCalcer->SetSplitsCount(CalcSplitsCount(stats3d.SplitEnsembleSpec, bucketCount, oneHotMaxSize));

    const double scaledL2Regularizer = l2Regularizer * (sumAllWeights / allDocCount);
    scoreCalcer->SetL2Regularizer(scaledL2Regularizer);
    for (int statsIdx = 0; statsIdx * splitStatsCount < bucketStats.ysize(); ++statsIdx) {
        const TBucketStats* stats = GetDataPtr(bucketStats) + statsIdx * splitStatsCount;
        UpdateScores(
            stats,
            leafCount,
            indexer,
            stats3d.SplitEnsembleSpec,
            scaledL2Regularizer,
            /*isPlainMode=*/std::true_type(),
            /*haveMonotonicConstraints*/std::false_type(),
            oneHotMaxSize,
            /*currTreeMonotonicConstraints*/{},
            /*candidateSplitMonotonicConstraints*/{},
            scoreCalcer.Get()
        );
    }
    return scoreCalcer->GetScores();
}
