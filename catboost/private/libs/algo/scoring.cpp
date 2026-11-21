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
#include <catboost/libs/helpers/dispatch_generic_lambda.h>
#include <catboost/private/libs/algo_helpers/online_predictor.h>
#include <catboost/private/libs/algo_helpers/scoring_helpers.h>
#include <catboost/private/libs/data_types/pair.h>
#include <catboost/private/libs/index_range/index_range.h>
#include <catboost/private/libs/options/catboost_options.h>
#include <catboost/private/libs/options/restrictions.h>

#include <library/cpp/containers/stack_vector/stack_vec.h>
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
        const int Depth;
        const TIndexType* const LeafIndices;
        const char* const QuantizedValues;
        const size_t BitsPerValue;
        const ui32* const ObjectIndices; // may be nullptr
        const int ObjectOffset; // use if ObjectIndices == nullptr

    public:
        explicit TStatsIndexer(int bucketCount)
        : BucketCount(bucketCount)
        , Depth(0)
        , LeafIndices(nullptr)
        , QuantizedValues(nullptr)
        , BitsPerValue(0)
        , ObjectIndices(nullptr)
        , ObjectOffset(0)
        {
        }

        TStatsIndexer(
            int bucketCount,
            int depth,
            const TIndexType* leafIndices,
            const char* quantizedValues,
            size_t bitsPerValue,
            const ui32* objectIndices,
            int objectOffset)
        : BucketCount(bucketCount)
        , Depth(depth)
        , LeafIndices(leafIndices)
        , QuantizedValues(quantizedValues)
        , BitsPerValue(bitsPerValue)
        , ObjectIndices(objectIndices)
        , ObjectOffset(objectOffset)
        {
            Y_ASSERT(LeafIndices && QuantizedValues);
        }

        int CalcSize(int depth) const {
            return (1U << depth) * BucketCount;
        }

        int GetIndex(int leafIndex, int bucketIndex) const {
            Y_ASSERT(!LeafIndices && !ObjectIndices && !QuantizedValues);
            return BucketCount * leafIndex + bucketIndex;
        }

        template <bool isOneNodeTree, typename TQuantType>
        int GetIndex(int obj, const TQuantType* quantizedValues) const {
            Y_ASSERT(LeafIndices && QuantizedValues);
            const auto objectIdx = ObjectIndices ? ObjectIndices[obj] : ObjectOffset + obj;
            const auto quantizedValue = quantizedValues[objectIdx];
            if (isOneNodeTree) {
                return quantizedValue;
            }
            const auto leafIndex = LeafIndices[obj];
            return BucketCount * leafIndex + quantizedValue;
        }
    };
}


template <class TColumn>
static void GetBitsPerValueAndRawPtr(const TColumn& column, size_t* bitsPerValue, const char** rawPtr) {
    const auto* denseColumnData = dynamic_cast<const TCompressedValuesHolderImpl<TColumn>*>(&column);
    CB_ENSURE_INTERNAL(denseColumnData, "GetBitsPerValueAndRawPtr: unexpected column type");
    *bitsPerValue = denseColumnData->GetCompressedData().GetSrc()->GetBitsPerKey();
    *rawPtr = denseColumnData->GetCompressedData().GetSrc()->GetRawPtr();
}


template <class TFunc>
static void DispatchByBitsPerValue(TFunc func, size_t bitsPerValue, const char* data) {
    switch (bitsPerValue) {
        case 8:
            func((const ui8*)data);
            break;
        case 16:
            func((const ui16*)data);
            break;
        case 32:
            func((const ui32*)data);
            break;
        default:
            CB_ENSURE_INTERNAL(false, "Unsupported bitsPerValue " << bitsPerValue);
    }
}


static void GetIndexingParams(
    const TCalcScoreFold& fold,
    const TSplitEnsemble& splitEnsemble,
    const ui32** objectIndexing,
    int* beginOffset
) {
    bool isEstimatedData;
    bool isOnlineData;
    if (splitEnsemble.IsSplitOfType(ESplitType::OnlineCtr)) {
        isEstimatedData = false;
        isOnlineData = true;
    } else {
        isEstimatedData = splitEnsemble.IsEstimated;
        isOnlineData = splitEnsemble.IsOnlineEstimated;
    }

    if (isOnlineData) {
        const bool simpleIndexing = fold.OnlineDataPermutationBlockSize == fold.GetDocCount();
        *objectIndexing = simpleIndexing ? nullptr : GetDataPtr(fold.IndexInFold);
        *beginOffset = 0;
    } else if (isEstimatedData) {
        *objectIndexing = fold.GetLearnPermutationOfflineEstimatedFeaturesSubset().data();
        *beginOffset = 0;
    } else {
        const bool simpleIndexing = fold.MainDataPermutationBlockSize == fold.GetDocCount();
        *objectIndexing = simpleIndexing
            ? nullptr
            : fold.LearnPermutationFeaturesSubset.Get<TIndexedSubset<ui32>>().data();
        *beginOffset = simpleIndexing ? fold.FeaturesSubsetBegin : 0;
    }
}

static void GetBitsPerValueAndRawPtr(
    const TQuantizedObjectsDataProvider& objectsDataProvider,
    const std::tuple<const TOnlineCtrBase&, const TOnlineCtrBase&>& allCtrs,
    const TSplitEnsemble& splitEnsemble,
    size_t* bitsPerValue,
    const char** rawPtr
) {
    if (splitEnsemble.IsSplitOfType(ESplitType::OnlineCtr)) {
        const TCtr& ctr = splitEnsemble.SplitCandidate.Ctr;
        *bitsPerValue = 8;
        *rawPtr = (const char*)GetCtr(allCtrs, ctr.Projection).GetData(ctr, /*datasetIdx*/ 0).data();
    } else {
        switch (splitEnsemble.Type) {
            case ESplitEnsembleType::OneFeature: {
                const auto& splitCandidate = splitEnsemble.SplitCandidate;
                const auto featureIdx = (ui32)splitCandidate.FeatureIdx;
                if (EqualToOneOf(splitCandidate.Type, ESplitType::FloatFeature, ESplitType::EstimatedFeature)) {
                    GetBitsPerValueAndRawPtr(**objectsDataProvider.GetNonPackedFloatFeature(featureIdx), bitsPerValue, rawPtr);
                } else {
                    Y_ASSERT(splitCandidate.Type == ESplitType::OneHotFeature);
                    GetBitsPerValueAndRawPtr(**objectsDataProvider.GetNonPackedCatFeature(featureIdx), bitsPerValue, rawPtr);
                }
                break;
            }
            case ESplitEnsembleType::BinarySplits: {
                const auto packIdx = splitEnsemble.BinarySplitsPackRef.PackIdx;
                GetBitsPerValueAndRawPtr(objectsDataProvider.GetBinaryFeaturesPack(packIdx), bitsPerValue, rawPtr);
                break;
            }
            case ESplitEnsembleType::ExclusiveBundle: {
                const auto bundleIdx = splitEnsemble.ExclusiveFeaturesBundleRef.BundleIdx;
                GetBitsPerValueAndRawPtr(objectsDataProvider.GetExclusiveFeaturesBundle(bundleIdx), bitsPerValue, rawPtr);
                break;
            }
            case ESplitEnsembleType::FeaturesGroup:
                CB_ENSURE_INTERNAL(false, "FeaturesGroups are implemented only in leafwise scoring");
        }
    }
}


// Update bootstraped sums on docIndexRange in a bucket
inline static void UpdateWeighted(
    const TStatsIndexer& indexer,
    const double* weightedDer,
    const float* sampleWeights,
    NCB::TIndexRange<int> docIndexRange,
    TBucketStats* stats
) {
    DispatchByBitsPerValue(
        [=] (const auto* quantizedValues) {
            DispatchGenericLambda(
                [=] (auto isOneNode) {
                    for (int doc : docIndexRange.Iter()) {
                        auto& leafStats0 = stats[indexer.GetIndex<isOneNode>(doc, quantizedValues)];
                        leafStats0.SumWeightedDelta += weightedDer[doc];
                        leafStats0.SumWeight += sampleWeights[doc];
                    }
                },
                indexer.Depth == 0);
        },
        indexer.BitsPerValue,
        indexer.QuantizedValues);
}


// Same as UpdateWeighted for two indexers that differ only in quantized values and bucket count,
// objects data is loaded once for both
inline static void UpdateWeightedForTwo(
    const TStatsIndexer& indexer0,
    const TStatsIndexer& indexer1,
    const double* weightedDer,
    const float* sampleWeights,
    NCB::TIndexRange<int> docIndexRange,
    TBucketStats* stats0,
    TBucketStats* stats1
) {
    Y_ASSERT(indexer0.Depth == indexer1.Depth);
    Y_ASSERT(indexer0.BitsPerValue == indexer1.BitsPerValue);
    Y_ASSERT(indexer0.LeafIndices == indexer1.LeafIndices);
    Y_ASSERT(indexer0.ObjectIndices == indexer1.ObjectIndices && indexer0.ObjectOffset == indexer1.ObjectOffset);
    DispatchByBitsPerValue(
        [=] (const auto* quantizedValues0) {
            const auto* quantizedValues1 = reinterpret_cast<decltype(quantizedValues0)>(indexer1.QuantizedValues);
            DispatchGenericLambda(
                [=] (auto isOneNode) {
                    for (int doc : docIndexRange.Iter()) {
                        const double weightedDerOfDoc = weightedDer[doc];
                        const float sampleWeight = sampleWeights[doc];
                        const auto objectIdx = indexer0.ObjectIndices
                            ? indexer0.ObjectIndices[doc]
                            : indexer0.ObjectOffset + doc;
                        const TIndexType leafIndex = isOneNode ? 0 : indexer0.LeafIndices[doc];
                        auto& leafStats0 = stats0[indexer0.BucketCount * leafIndex + quantizedValues0[objectIdx]];
                        leafStats0.SumWeightedDelta += weightedDerOfDoc;
                        leafStats0.SumWeight += sampleWeight;
                        auto& leafStats1 = stats1[indexer1.BucketCount * leafIndex + quantizedValues1[objectIdx]];
                        leafStats1.SumWeightedDelta += weightedDerOfDoc;
                        leafStats1.SumWeight += sampleWeight;
                    }
                },
                indexer0.Depth == 0);
        },
        indexer0.BitsPerValue,
        indexer0.QuantizedValues);
}


// Update not bootstraped sums on docIndexRange in a bucket
inline static void UpdateDeltaCount(
    const TStatsIndexer& indexer,
    const double* derivatives,
    const float* learnWeights,
    NCB::TIndexRange<int> docIndexRange,
    TBucketStats* stats
) {
    DispatchByBitsPerValue(
        [=] (const auto* quantizedValues) {
            DispatchGenericLambda(
                [=] (auto haveWeights, auto isOneNode) {
                    for (int doc : docIndexRange.Iter()) {
                        auto& leafStats = stats[indexer.GetIndex<isOneNode>(doc, quantizedValues)];
                        leafStats.SumDelta += derivatives[doc];
                        leafStats.Count += haveWeights ? learnWeights[doc] : 1;
                    }
                },
                learnWeights != nullptr, indexer.Depth == 0);
        },
        indexer.BitsPerValue,
        indexer.QuantizedValues);
}


inline static void CalcStatsKernel(
    bool isCaching,
    const TCalcScoreFold& fold,
    bool isPlainMode,
    const TStatsIndexer& indexer,
    int depth,
    const TCalcScoreFold::TBodyTail& bt,
    int dim,
    NCB::TIndexRange<int> docIndexRange,
    TBucketStats* stats,
    // optional second candidate (only in plain mode), its sums are updated in the same pass over objects
    const TStatsIndexer* secondIndexer = nullptr,
    TBucketStats* secondStats = nullptr
) {
    Y_ASSERT(!isCaching || depth > 0);
    Y_ASSERT(!secondIndexer || isPlainMode);
    const auto fillStats = [&] (const TStatsIndexer& candidateIndexer, TBucketStats* candidateStats) {
        if (isCaching) {
            Fill(
                candidateStats + candidateIndexer.CalcSize(depth - 1),
                candidateStats + candidateIndexer.CalcSize(depth),
                TBucketStats{0, 0, 0, 0}
            );
        } else {
            Fill(candidateStats, candidateStats + candidateIndexer.CalcSize(depth), TBucketStats{0, 0, 0, 0});
        }
    };
    fillStats(indexer, stats);
    if (secondIndexer) {
        fillStats(*secondIndexer, secondStats);
    }

    if (bt.TailFinish > docIndexRange.Begin) {
        const bool hasPairwiseWeights = !bt.PairwiseWeights.empty();
        const float* weightsData = hasPairwiseWeights ?
            GetDataPtr(bt.PairwiseWeights) : GetDataPtr(fold.LearnWeights);
        const float* sampleWeightsData = hasPairwiseWeights ?
            GetDataPtr(bt.SamplePairwiseWeights) : GetDataPtr(fold.SampleWeights);

        int tailFinishInRange = Min((int)bt.TailFinish, docIndexRange.End);

        if (isPlainMode && secondIndexer) {
            UpdateWeightedForTwo(
                indexer,
                *secondIndexer,
                GetDataPtr(bt.SampleWeightedDerivatives[dim]),
                sampleWeightsData,
                NCB::TIndexRange<int>(docIndexRange.Begin, tailFinishInRange),
                stats,
                secondStats
            );
        } else if (isPlainMode) {
            UpdateWeighted(
                indexer,
                GetDataPtr(bt.SampleWeightedDerivatives[dim]),
                sampleWeightsData,
                NCB::TIndexRange<int>(docIndexRange.Begin, tailFinishInRange),
                stats
            );
        } else {
            if (bt.BodyFinish > docIndexRange.Begin) {
                UpdateDeltaCount(
                    indexer,
                    GetDataPtr(bt.WeightedDerivatives[dim]),
                    weightsData,
                    NCB::TIndexRange<int>(docIndexRange.Begin, Min((int)bt.BodyFinish, docIndexRange.End)),
                    stats
                );
            }
            if (tailFinishInRange > bt.BodyFinish) {
                UpdateWeighted(
                    indexer,
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


static void CalcStatsPairwise(
    const TCalcScoreFold& fold,
    const TQuantizedObjectsDataProvider& objectsDataProvider,
    const TFlatPairsInfo& pairs,
    const std::tuple<const TOnlineCtrBase&, const TOnlineCtrBase&>& allCtrs,
    const TSplitEnsemble& splitEnsemble,
    int bucketCount,
    ui32 oneHotMaxSize,
    int depth,
    NPar::ILocalExecutor* localExecutor,
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
                    bucketCount,
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
                                        GetCtr(allCtrs, ctr.Projection).GetData(ctr, /*datasetIdx*/ 0);

                                    ComputePairwiseStats<ui8>(
                                        ESplitEnsembleType::OneFeature,
                                        weightedDerivativesData,
                                        pairs,
                                        leafCount,
                                        bucketCount,
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
                    CB_ENSURE(false, "FeaturesGroups are implemented only in leafwise scoring now");
                default:
                    CB_ENSURE(false, "Unexpected split ensemble type");
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


// Stats for one or two (only in plain mode) candidates with the same objects and leaves indexing,
// bucket sums of two candidates are updated in one pass over objects
template <typename TIsCaching>
static void CalcStatsPointwise(
    const TCalcScoreFold& fold,
    TConstArrayRef<TStatsIndexer> indexers,
    const TIsCaching& isCaching,
    bool isPlainMode,
    int depth,
    TConstArrayRef<int> splitStatsCounts,
    NPar::ILocalExecutor* localExecutor,
    TConstArrayRef<TBucketStatsRefOptionalHolder*> stats
) {
    Y_ASSERT(!isCaching || depth > 0);
    const size_t candidateCount = indexers.size();
    Y_ASSERT(candidateCount == 1 || (candidateCount == 2 && isPlainMode));

    // bodyFunc must accept (bodyTailIdx, dim, bodyTailAndDimIdx) params,
    // stats for them begin at bodyTailAndDimIdx * splitStatsCount
    auto forEachBodyTailAndApproxDimension = [&](auto bodyFunc) {
        const int approxDimension = fold.GetApproxDimension();
        for (int bodyTailIdx : xrange(fold.GetBodyTailCount())) {
            for (int dim : xrange(approxDimension)) {
                bodyFunc(bodyTailIdx, dim, bodyTailIdx * approxDimension + dim);
            }
        }
    };

    using TCandidatesStats = TStackVec<TBucketStatsRefOptionalHolder, 2>;
    TCandidatesStats candidatesStats;
    for (auto* candidateStats : stats) {
        candidatesStats.push_back(std::move(*candidateStats));
    }

    NCB::MapMerge(
        localExecutor,
        fold.GetCalcStatsIndexRanges(),
        /*mapFunc*/[&](NCB::TIndexRange<int> indexRange, TCandidatesStats* output) {
            NCB::TIndexRange<int> docIndexRange = fold.HasQueryInfo() ?
                NCB::TIndexRange<int>(
                    fold.LearnQueriesInfo[indexRange.Begin].Begin,
                    (indexRange.End == 0) ? 0 : fold.LearnQueriesInfo[indexRange.End - 1].End
                )
                : indexRange;

            output->resize(candidateCount);
            for (auto candidateIdx : xrange(candidateCount)) {
                if ((*output)[candidateIdx].NonInited()) {
                    const int statsCount =
                        fold.GetBodyTailCount() * fold.GetApproxDimension() * splitStatsCounts[candidateIdx];
                    (*output)[candidateIdx] = TBucketStatsRefOptionalHolder(statsCount);
                } else {
                    Y_ASSERT(docIndexRange.Begin == 0);
                }
            }

            forEachBodyTailAndApproxDimension(
                [&](int bodyTailIdx, int dim, int bodyTailAndDimIdx) {
                    const auto getStatsSubset = [&] (size_t candidateIdx) {
                        return (*output)[candidateIdx].GetData().data()
                            + bodyTailAndDimIdx * splitStatsCounts[candidateIdx];
                    };
                    CalcStatsKernel(
                        isCaching && (indexRange.Begin == 0),
                        fold,
                        isPlainMode,
                        indexers[0],
                        depth,
                        fold.BodyTailArr[bodyTailIdx],
                        dim,
                        docIndexRange,
                        getStatsSubset(0),
                        (candidateCount == 2) ? &indexers[1] : nullptr,
                        (candidateCount == 2) ? getStatsSubset(1) : nullptr
                    );
                }
            );
        },
        /*mergeFunc*/[&](
            TCandidatesStats* output,
            TVector<TCandidatesStats>&& addVector
        ) {
            for (auto candidateIdx : xrange(candidateCount)) {
                const int splitStatsCount = splitStatsCounts[candidateIdx];
                const int filledSplitStatsCount = indexers[candidateIdx].CalcSize(depth);
                forEachBodyTailAndApproxDimension(
                    [&](int /*bodyTailIdx*/, int /*dim*/, int bodyTailAndDimIdx) {
                        TBucketStats* outputStatsSubset =
                            (*output)[candidateIdx].GetData().data() + bodyTailAndDimIdx * splitStatsCount;

                        for (const auto& addItem : addVector) {
                            const TBucketStats* addStatsSubset =
                                addItem[candidateIdx].GetData().data() + bodyTailAndDimIdx * splitStatsCount;
                            for (size_t i : xrange(filledSplitStatsCount)) {
                                (outputStatsSubset + i)->Add(*(addStatsSubset + i));
                            }
                        }
                    }
                );
            }
        },
        &candidatesStats
    );

    for (auto candidateIdx : xrange(candidateCount)) {
        *stats[candidateIdx] = std::move(candidatesStats[candidateIdx]);
        if (isCaching) {
            forEachBodyTailAndApproxDimension(
                [&](int /*bodyTailIdx*/, int /*dim*/, int bodyTailAndDimIdx) {
                    TBucketStats* statsSubset = stats[candidateIdx]->GetData().data()
                        + bodyTailAndDimIdx * splitStatsCounts[candidateIdx];
                    FixUpStats(depth, indexers[candidateIdx], fold.SmallestSplitSideValue, statsSubset);
                }
            );
        }
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
        const auto getBucketStats = [stats, leaf, indexer] (int bucketIdx) {
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

    DispatchGenericLambda(
        [&] (auto isPlainMode, auto haveMonotonicConstraints) {
            for (int bodyTailIdx = 0; bodyTailIdx < fold.GetBodyTailCount(); ++bodyTailIdx) {
                const double sumAllWeights = initialFold.BodyTailArr[bodyTailIdx].BodySumWeight;
                const int docCount = initialFold.BodyTailArr[bodyTailIdx].BodyFinish;
                const double scaledL2Regularizer = l2Regularizer * (sumAllWeights / docCount);
                scoreCalcer->SetL2Regularizer(scaledL2Regularizer);
                for (int dim = 0; dim < approxDimension; ++dim) {
                    const TBucketStats* stats = splitStats
                        + (bodyTailIdx * approxDimension + dim) * splitStatsCount;
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
                        scoreCalcer);
                }
             }
        },
        isPlainMode, haveMonotonicConstraints);
}


// Pointwise scoring for one or two candidates (see CalcStatsAndScoresForTwoCandidates)
static void CalcPointwiseStatsAndScores(
    const TQuantizedObjectsDataProvider& objectsDataProvider,
    const std::tuple<const TOnlineCtrBase&, const TOnlineCtrBase&>& allCtrs,
    const TCalcScoreFold& fold,
    const TCalcScoreFold& prevLevelData,
    const TFold* initialFold,
    const NCatboostOptions::TCatBoostOptions& fitParams,
    TConstArrayRef<const TCandidateInfo*> candidateInfos,
    int depth,
    bool useTreeLevelCaching,
    const TVector<int>& currTreeMonotonicConstraints,
    const TMap<ui32, int>& monotonicConstraints,
    NPar::ILocalExecutor* localExecutor,
    TBucketStatsCache* statsFromPrevTree,
    TStats3D* stats3d, // can be non-nullptr only for one candidate
    TConstArrayRef<IScoreCalcer*> scoreCalcers
) {
    const size_t candidateCount = candidateInfos.size();
    Y_ASSERT(scoreCalcers.size() == candidateCount);
    Y_ASSERT(candidateCount == 1 || (candidateCount == 2 && !stats3d));

    const float l2Regularizer = static_cast<const float>(fitParams.ObliviousTreeOptions->L2Reg);
    const ui32 oneHotMaxSize = fitParams.CatFeatureParams.Get().OneHotMaxSize.Get();
    const bool isPlainMode = IsPlainMode(fitParams.BoostingOptions->BoostingType);

    const auto& treeOptions = fitParams.ObliviousTreeOptions.Get();

    TStackVec<int, 2> bucketCounts;
    TStackVec<int, 2> splitStatsCounts;
    TStackVec<TBucketStatsRefOptionalHolder, 2> extOrInSplitStats;
    TStackVec<TVector<TBucketStats, TPoolAllocator>*, 2> splitStatsFromCache;
    TStackVec<bool, 2> isCaching;
    TStackVec<TStatsIndexer, 2> indexers;

    for (const auto* candidateInfo : candidateInfos) {
        const auto& splitEnsemble = candidateInfo->SplitEnsemble;
        const int bucketCount = GetBucketCount(
            splitEnsemble,
            *objectsDataProvider.GetQuantizedFeaturesInfo(),
            objectsDataProvider.GetPackedBinaryFeaturesSize(),
            objectsDataProvider.GetExclusiveFeatureBundlesMetaData(),
            objectsDataProvider.GetFeaturesGroupsMetaData()
        );
        bucketCounts.push_back(bucketCount);
        extOrInSplitStats.emplace_back();
        splitStatsFromCache.push_back(nullptr);
        isCaching.push_back(false);

        if (!useTreeLevelCaching) {
            splitStatsCounts.push_back((ui64(1) << depth) * bucketCount);
            const int statsCount =
                fold.GetBodyTailCount() * fold.GetApproxDimension() * splitStatsCounts.back();

            if (stats3d != nullptr) {
                stats3d->Stats.yresize(statsCount);
                stats3d->BucketCount = bucketCount;
                stats3d->MaxLeafCount = 1U << depth;
                stats3d->SplitEnsembleSpec = TSplitEnsembleSpec(
                    splitEnsemble,
                    objectsDataProvider.GetExclusiveFeatureBundlesMetaData(),
                    objectsDataProvider.GetFeaturesGroupsMetaData()
                );

                extOrInSplitStats.back() = TBucketStatsRefOptionalHolder(stats3d->Stats);
            }
        } else {
            splitStatsCounts.push_back((ui64(1) << treeOptions.MaxDepth) * bucketCount);
            bool areStatsDirty;

            // thread-safe access
            splitStatsFromCache.back() =
                &statsFromPrevTree->GetStats(splitEnsemble, splitStatsCounts.back(), &areStatsDirty);
            extOrInSplitStats.back() = TBucketStatsRefOptionalHolder(*splitStatsFromCache.back());
            isCaching.back() = (depth != 0) && !areStatsDirty;
        }

        size_t bitsPerValue;
        const char* rawPtr;
        GetBitsPerValueAndRawPtr(
            objectsDataProvider,
            allCtrs,
            splitEnsemble,
            &bitsPerValue,
            &rawPtr);

        const TCalcScoreFold& statsFold = isCaching.back() ? prevLevelData : fold;
        const ui32* objectIndexing;
        int beginOffset;
        GetIndexingParams(
            statsFold,
            splitEnsemble,
            &objectIndexing,
            &beginOffset);

        indexers.emplace_back(
            bucketCount,
            depth,
            GetDataPtr(statsFold.Indices),
            rawPtr,
            bitsPerValue,
            objectIndexing,
            beginOffset);
    }

    // candidates [begin, end) are calculated together
    const auto calcStatsPointwise = [&] (size_t begin, size_t end) {
        TStackVec<TBucketStatsRefOptionalHolder*, 2> stats;
        for (auto candidateIdx : xrange(begin, end)) {
            stats.push_back(&extOrInSplitStats[candidateIdx]);
        }
        const auto calc = [&] (auto isCachingConst) {
            CalcStatsPointwise(
                isCachingConst ? prevLevelData : fold,
                MakeArrayRef(indexers).Slice(begin, end - begin),
                isCachingConst,
                isPlainMode,
                depth,
                MakeArrayRef(splitStatsCounts).Slice(begin, end - begin),
                localExecutor,
                stats);
        };
        if (isCaching[begin]) {
            calc(std::true_type());
        } else {
            calc(std::false_type());
        }
    };
    const bool calcInOnePass = (candidateCount == 2)
        && isPlainMode
        && (isCaching[0] == isCaching[1])
        && (indexers[0].BitsPerValue == indexers[1].BitsPerValue)
        && (indexers[0].LeafIndices == indexers[1].LeafIndices)
        && (indexers[0].ObjectIndices == indexers[1].ObjectIndices)
        && (indexers[0].ObjectOffset == indexers[1].ObjectOffset);
    if (calcInOnePass) {
        calcStatsPointwise(0, 2);
    } else {
        for (auto candidateIdx : xrange(candidateCount)) {
            calcStatsPointwise(candidateIdx, candidateIdx + 1);
        }
    }

    for (auto candidateIdx : xrange(candidateCount)) {
        const TCandidateInfo& candidateInfo = *candidateInfos[candidateIdx];
        const auto& splitEnsemble = candidateInfo.SplitEnsemble;
        const int bucketCount = bucketCounts[candidateIdx];
        const int splitStatsCount = splitStatsCounts[candidateIdx];
        if (useTreeLevelCaching && stats3d) {
            TBucketStatsCache::GetStatsInUse(
                fold.GetBodyTailCount() * fold.GetApproxDimension(),
                splitStatsCount,
                (ui64(1) << depth) * bucketCount,
                *splitStatsFromCache[candidateIdx]
            ).swap(stats3d->Stats);
            stats3d->BucketCount = bucketCount;
            stats3d->MaxLeafCount = 1U << depth;
            stats3d->SplitEnsembleSpec = TSplitEnsembleSpec(
                splitEnsemble,
                objectsDataProvider.GetExclusiveFeatureBundlesMetaData(),
                objectsDataProvider.GetFeaturesGroupsMetaData()
            );
        }
        if (IScoreCalcer* scoreCalcer = scoreCalcers[candidateIdx]) {
            const int leafCount = 1 << depth;
            TSplitEnsembleSpec splitEnsembleSpec(
                splitEnsemble,
                objectsDataProvider.GetExclusiveFeatureBundlesMetaData(),
                objectsDataProvider.GetFeaturesGroupsMetaData()
            );
            const int candidateSplitCount = CalcSplitsCount(
                splitEnsembleSpec, bucketCount, oneHotMaxSize
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
                TStatsIndexer(bucketCount),
                extOrInSplitStats[candidateIdx].GetData().data(),
                splitStatsCount,
                currTreeMonotonicConstraints,
                candidateSplitMonotonicConstraints,
                dynamic_cast<IPointwiseScoreCalcer*>(scoreCalcer)
            );
        }
    }
}

void CalcStatsAndScores(
    const TQuantizedObjectsDataProvider& objectsDataProvider,
    const std::tuple<const TOnlineCtrBase&, const TOnlineCtrBase&>& allCtrs,
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
    NPar::ILocalExecutor* localExecutor,
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

    const float l2Regularizer = static_cast<const float>(fitParams.ObliviousTreeOptions->L2Reg);
    const ui32 oneHotMaxSize = fitParams.CatFeatureParams.Get().OneHotMaxSize.Get();

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

        CalcStatsPairwise(
            fold,
            objectsDataProvider,
            pairs,
            allCtrs,
            splitEnsemble,
            bucketCount,
            oneHotMaxSize,
            depth,
            localExecutor,
            pairwiseStats);

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

        CalcPointwiseStatsAndScores(
            objectsDataProvider,
            allCtrs,
            fold,
            prevLevelData,
            initialFold,
            fitParams,
            {&candidateInfo},
            depth,
            useTreeLevelCaching,
            currTreeMonotonicConstraints,
            monotonicConstraints,
            localExecutor,
            statsFromPrevTree,
            stats3d,
            {scoreCalcer});
    }
}


void CalcStatsAndScoresForTwoCandidates(
    const TQuantizedObjectsDataProvider& objectsDataProvider,
    const std::tuple<const TOnlineCtrBase&, const TOnlineCtrBase&>& allCtrs,
    const TCalcScoreFold& fold,
    const TCalcScoreFold& prevLevelData,
    const TFold* initialFold,
    const TFlatPairsInfo& pairs,
    const NCatboostOptions::TCatBoostOptions& fitParams,
    const TCandidateInfo& candidateInfo0,
    const TCandidateInfo& candidateInfo1,
    int depth,
    bool useTreeLevelCaching,
    const TVector<int>& currTreeMonotonicConstraints,
    const TMap<ui32, int>& monotonicConstraints,
    NPar::ILocalExecutor* localExecutor,
    TBucketStatsCache* statsFromPrevTree,
    IScoreCalcer* scoreCalcer0,
    IScoreCalcer* scoreCalcer1
) {
    if (IsPairwiseScoring(fitParams.LossFunctionDescription->GetLossFunction())) {
        for (auto [candidateInfo, scoreCalcer] : {
            std::make_pair(&candidateInfo0, scoreCalcer0),
            std::make_pair(&candidateInfo1, scoreCalcer1)
        }) {
            CalcStatsAndScores(
                objectsDataProvider,
                allCtrs,
                fold,
                prevLevelData,
                initialFold,
                pairs,
                fitParams,
                *candidateInfo,
                depth,
                useTreeLevelCaching,
                currTreeMonotonicConstraints,
                monotonicConstraints,
                localExecutor,
                statsFromPrevTree,
                /*stats3d*/ nullptr,
                /*pairwiseStats*/ nullptr,
                scoreCalcer);
        }
        return;
    }
    CB_ENSURE(initialFold, "initialFold must be non-nullptr for scores calculation");

    CalcPointwiseStatsAndScores(
        objectsDataProvider,
        allCtrs,
        fold,
        prevLevelData,
        initialFold,
        fitParams,
        {&candidateInfo0, &candidateInfo1},
        depth,
        useTreeLevelCaching,
        currTreeMonotonicConstraints,
        monotonicConstraints,
        localExecutor,
        statsFromPrevTree,
        /*stats3d*/ nullptr,
        {scoreCalcer0, scoreCalcer1});
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
