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


template <typename TIsCaching>
static void CalcStatsPointwise(
    const TCalcScoreFold& fold,
    const TStatsIndexer& indexer,
    const TIsCaching& isCaching,
    bool isPlainMode,
    int depth,
    int splitStatsCount,
    NPar::ILocalExecutor* localExecutor,
    TBucketStatsRefOptionalHolder* stats
) {
    Y_ASSERT(!isCaching || depth > 0);

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

        size_t bitsPerValue;
        const char* rawPtr;
        GetBitsPerValueAndRawPtr(
            objectsDataProvider,
            allCtrs,
            splitEnsemble,
            &bitsPerValue,
            &rawPtr);

        const bool isPlainMode = IsPlainMode(fitParams.BoostingOptions->BoostingType);

        const auto calcStatsPointwise = [&] (
            auto isCaching,
            const TCalcScoreFold& fold,
            int splitStatsCount,
            auto* stats
        ) {
            const ui32* objectIndexing;
            int beginOffset;
            GetIndexingParams(
                fold,
                splitEnsemble,
                &objectIndexing,
                &beginOffset);

            const TStatsIndexer indexer(
                bucketCount,
                depth,
                GetDataPtr(fold.Indices),
                rawPtr,
                bitsPerValue,
                objectIndexing,
                beginOffset);

            CalcStatsPointwise(
                fold,
                indexer,
                isCaching,
                isPlainMode,
                depth,
                splitStatsCount,
                localExecutor,
                stats);
        };

        TBucketStatsRefOptionalHolder extOrInSplitStats;
        int splitStatsCount = 0;

        const auto& treeOptions = fitParams.ObliviousTreeOptions.Get();

        if (!useTreeLevelCaching) {
            splitStatsCount = (ui64(1) << depth) * bucketCount;
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
            calcStatsPointwise(
                /*isCaching*/ std::false_type(),
                fold,
                splitStatsCount,
                &extOrInSplitStats
            );
        } else {
            splitStatsCount = (ui64(1) << treeOptions.MaxDepth) * bucketCount;
            bool areStatsDirty;

            // thread-safe access
            TVector<TBucketStats, TPoolAllocator>& splitStatsFromCache =
                statsFromPrevTree->GetStats(splitEnsemble, splitStatsCount, &areStatsDirty);
            extOrInSplitStats = TBucketStatsRefOptionalHolder(splitStatsFromCache);
            if (depth == 0 || areStatsDirty) {
                calcStatsPointwise(
                    /*isCaching*/ std::false_type(),
                    fold,
                    splitStatsCount,
                    &extOrInSplitStats
                );
            } else {
                calcStatsPointwise(
                    /*isCaching*/ std::true_type(),
                    prevLevelData,
                    splitStatsCount,
                    &extOrInSplitStats
                );
            }
            if (stats3d) {
                TBucketStatsCache::GetStatsInUse(
                    fold.GetBodyTailCount() * fold.GetApproxDimension(),
                    splitStatsCount,
                    (ui64(1) << depth) * bucketCount,
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
