#include "leafwise_scoring.h"

#include <catboost/libs/data/columns.h>
#include <catboost/libs/helpers/parallel_tasks.h>
#include <catboost/private/libs/algo_helpers/scoring_helpers.h>

// TODO(ilyzhin) sampling with groups
// TODO(ilyzhin) queries

using namespace NCB;


bool IsLeafwiseScoringApplicable(const NCatboostOptions::TCatBoostOptions& params) {
    // TODO(ilyzhin) support these constraints
    return (params.BoostingOptions->BoostingType == EBoostingType::Plain
           && !IsPairwiseScoring(params.LossFunctionDescription->GetLossFunction())
           && params.SystemOptions->IsSingleHost()
           && params.ObliviousTreeOptions->MonotoneConstraints.Get().empty()
           && params.DataProcessingOptions->DevLeafwiseScoring)
           || params.ObliviousTreeOptions->GrowPolicy == EGrowPolicy::Lossguide
           || params.ObliviousTreeOptions->GrowPolicy == EGrowPolicy::Depthwise;
}


// Helper function for calculating index of leaf for each document given a new split.
// Calculates indices when a permutation is given.
template <typename TColumnType, typename TBucketIndexType>
inline static void SetBucketIndex(
    const TColumnType* column,
    const ui32* bucketIndexing, // can be nullptr for simple case, use bucketBeginOffset instead then
    const int bucketBeginOffset,
    TIndexRange<ui32> docIndexRange,
    int groupSize,
    const TVector<ui32>& groupPartsBucketsOffsets,
    TVector<TBucketIndexType>* bucketIdx // already of proper size
) {
    const TArrayRef<TBucketIndexType> bucketIdxRef(*bucketIdx);

    if (groupSize == 1) {
        if (bucketIndexing == nullptr) {
            for (auto doc : docIndexRange.Iter()) {
                bucketIdxRef[doc] = column[bucketBeginOffset + doc];
            }
        } else {
            for (auto doc : docIndexRange.Iter()) {
                const ui32 originalDocIdx = bucketIndexing[doc];
                bucketIdxRef[doc] = column[originalDocIdx];
            }
        }
    } else {
        int pos = docIndexRange.Begin * groupSize;
        if (bucketIndexing == nullptr) {
            for (auto doc : docIndexRange.Iter()) {
                for (auto partIdx : xrange(groupSize)) {
                    bucketIdxRef[pos++] = groupPartsBucketsOffsets[partIdx] +
                        GetPartValueFromGroup(column[bucketBeginOffset + doc], partIdx);
                }
            }
        } else {
            for (auto doc : docIndexRange.Iter()) {
                const ui32 originalDocIdx = bucketIndexing[doc];
                for (auto partIdx : xrange(groupSize)) {
                    bucketIdxRef[pos++] = groupPartsBucketsOffsets[partIdx] +
                        GetPartValueFromGroup(column[originalDocIdx], partIdx);
                }
            }
        }
    }

}

static void GetIndexingParams(
    const TCalcScoreFold& fold,
    bool isEstimatedData,
    bool isOnlineData,
    const ui32** objectIndexing,
    int* beginOffset
) {
    // Simple indexing possible only at first level now
    if (isOnlineData) {
        const bool simpleIndexing = (fold.LeavesBounds.ysize() == 1)
            && (fold.OnlineDataPermutationBlockSize == fold.GetDocCount());
        *objectIndexing = simpleIndexing ? nullptr : GetDataPtr(fold.IndexInFold);
        *beginOffset = 0;
    } else if (isEstimatedData) {
        *objectIndexing = fold.GetLearnPermutationOfflineEstimatedFeaturesSubset().data();
        *beginOffset = 0;
    } else {
        const bool simpleIndexing = (fold.LeavesBounds.size() == 1) &&
            (fold.MainDataPermutationBlockSize == fold.GetDocCount());
        *objectIndexing = simpleIndexing
            ? nullptr
            : fold.LearnPermutationFeaturesSubset.Get<TIndexedSubset<ui32>>().data();
        *beginOffset = simpleIndexing ? fold.FeaturesSubsetBegin : 0;
    }
}


template <class TColumn, typename TBucketIndexType>
inline static void ExtractBucketIndex(
    const TCalcScoreFold& fold,
    const TColumn& column,
    bool isEstimatedData,
    bool isOnlineData,
    TIndexRange<ui32> docIndexRange,
    int groupSize,
    const TVector<ui32>& groupPartsBucketsOffsets,
    TVector<TBucketIndexType>* bucketIdx // already of proper size
) {
    if (const auto* denseColumnData
        = dynamic_cast<const TCompressedValuesHolderImpl<TColumn>*>(&column))
    {
        const ui32* objectIndexing;
        int beginOffset;
        GetIndexingParams(fold, isEstimatedData, isOnlineData, &objectIndexing, &beginOffset);

        const TCompressedArray& compressedArray = *denseColumnData->GetCompressedData().GetSrc();

        compressedArray.DispatchBitsPerKeyToDataType(
            "ExtractBucketIndex",
            [&] (const auto* columnData) {
                SetBucketIndex(
                    columnData,
                    objectIndexing,
                    beginOffset,
                    docIndexRange,
                    groupSize,
                    groupPartsBucketsOffsets,
                    bucketIdx
                );
            }
        );
    } else {
        Y_FAIL("ExtractBucketIndex: unexpected column type");
    }
}


// Calculate index of leaf for each document given a new split ensemble.
template <typename TBucketIndexType>
inline static void ExtractBucketIndex(
    const TCalcScoreFold& fold,
    const TQuantizedForCPUObjectsDataProvider& objectsDataProvider,
    const std::tuple<const TOnlineCTRHash&, const TOnlineCTRHash&>& allCtrs,
    const TSplitEnsemble& splitEnsemble,
    TIndexRange<ui32> docIndexRange,
    int groupSize,
    const TVector<ui32>& groupPartsBucketsOffsets,
    TVector<TBucketIndexType>* bucketIdx // already of proper size
) {
    if (splitEnsemble.IsSplitOfType(ESplitType::OnlineCtr)) {
        const TCtr& ctr = splitEnsemble.SplitCandidate.Ctr;
        const ui32* objectIndexing;
        int beginOffset;
        GetIndexingParams(
            fold,
            /*isEstimatedData*/ false,
            /*isOnlineData*/true,
            &objectIndexing,
            &beginOffset
        );
        SetBucketIndex(
            GetCtr(allCtrs, ctr.Projection).Feature[ctr.CtrIdx][ctr.TargetBorderIdx][ctr.PriorIdx].data(),
            objectIndexing,
            beginOffset,
            docIndexRange,
            groupSize,
            groupPartsBucketsOffsets,
            bucketIdx
        );
    } else {
        auto extractBucketIndexFunc = [&] (const auto& column) {
            ExtractBucketIndex(
                fold,
                column,
                splitEnsemble.IsEstimated,
                splitEnsemble.IsOnlineEstimated,
                docIndexRange,
                groupSize,
                groupPartsBucketsOffsets,
                bucketIdx
            );
        };

        switch (splitEnsemble.Type) {
            case ESplitEnsembleType::OneFeature:
            {
                const auto& splitCandidate = splitEnsemble.SplitCandidate;
                if ((splitCandidate.Type == ESplitType::FloatFeature) ||
                    (splitCandidate.Type == ESplitType::EstimatedFeature))
                {
                    extractBucketIndexFunc(
                        **objectsDataProvider.GetNonPackedFloatFeature((ui32)splitCandidate.FeatureIdx)
                    );
                } else {
                    Y_ASSERT(splitCandidate.Type == ESplitType::OneHotFeature);
                    extractBucketIndexFunc(
                        **objectsDataProvider.GetNonPackedCatFeature((ui32)splitCandidate.FeatureIdx)
                    );
                }
            }
                break;
            case ESplitEnsembleType::BinarySplits:
                extractBucketIndexFunc(
                    objectsDataProvider.GetBinaryFeaturesPack(splitEnsemble.BinarySplitsPackRef.PackIdx)
                );
                break;
            case ESplitEnsembleType::ExclusiveBundle:
                extractBucketIndexFunc(
                    objectsDataProvider.GetExclusiveFeaturesBundle(
                        splitEnsemble.ExclusiveFeaturesBundleRef.BundleIdx
                    )
                );
                break;
            case ESplitEnsembleType::FeaturesGroup:
                extractBucketIndexFunc(
                    objectsDataProvider.GetFeaturesGroup(
                        splitEnsemble.FeaturesGroupRef.GroupIdx
                    )
                );
        }
    }
}

template <typename TBucketIndexType>
inline void UpdateWeighted(
    const TVector<TBucketIndexType>& bucketIdx,
    const double* weightedDer,
    const float* sampleWeights,
    TIndexRange<ui32> docIndexRange,
    int indicesPerDoc,
    TBucketStats* stats
) {
    int pos = docIndexRange.Begin * indicesPerDoc;
    for (auto doc : docIndexRange.Iter()) {
        for (auto idx : xrange(indicesPerDoc)) {
            Y_UNUSED(idx);
            TBucketStats& leafStats = stats[bucketIdx[pos++]];
            leafStats.SumWeightedDelta += weightedDer[doc];
            leafStats.SumWeight += sampleWeights[doc];
        }
    }
}

template <typename TBucketIndexType>
inline void CalcStatsKernel(
    const TCalcScoreFold& fold,
    const TCalcScoreFold::TBodyTail& bt,
    int dim,
    int bucketCount,
    TIndexRange<ui32> docIndexRange,
    const TVector<TBucketIndexType>& bucketIdx, // has size = docs * indicesPerDoc
    int indicesPerDoc,
    TBucketStats* stats
) {
    Fill(stats, stats + bucketCount, TBucketStats{0, 0, 0, 0});

    UpdateWeighted(
        bucketIdx,
        GetDataPtr(bt.SampleWeightedDerivatives[dim]),
        GetDataPtr(fold.SampleWeights),
        docIndexRange,
        indicesPerDoc,
        stats
    );
}

template <typename TBucketIndexType, typename TScoreCalcer>
static void CalcScoresForSubCandidate(
    const NCB::TQuantizedForCPUObjectsDataProvider& objectsDataProvider,
    const TCandidateInfo& candidateInfo,
    int bucketCount,
    const TCalcScoreFold& fold,
    const TFold& initialFold,
    const TVector<TIndexType>& leafs,
    TLearnContext* ctx,
    TScoreCalcer* scoreCalcer
) {
    Y_ASSERT(fold.GetBodyTailCount() == 1);

    const int approxDimension = fold.GetApproxDimension();
    const ui32 oneHotMaxSize = ctx->Params.CatFeatureParams.Get().OneHotMaxSize.Get();
    const TSplitEnsembleSpec splitEnsembleSpec(
        candidateInfo.SplitEnsemble,
        objectsDataProvider.GetExclusiveFeatureBundlesMetaData(),
        objectsDataProvider.GetFeaturesGroupsMetaData());

    TVector<TBucketIndexType> bucketIdx;
    int groupSize = 1;
    TVector<ui32> bucketOffsets(1);
    if (candidateInfo.SplitEnsemble.Type == ESplitEnsembleType::FeaturesGroup) {
        const auto groupIdx = candidateInfo.SplitEnsemble.FeaturesGroupRef.GroupIdx;
        groupSize = objectsDataProvider.GetFeaturesGroupMetaData(groupIdx).Parts.ysize();
        bucketOffsets = objectsDataProvider.GetFeaturesGroupMetaData(groupIdx).BucketOffsets;
    }
    bucketIdx.yresize(fold.GetDocCount() * groupSize);

    auto extractBucketIndex = [&] (TIndexRange<ui32> docIndexRange) {
        ExtractBucketIndex(
            fold,
            objectsDataProvider,
            initialFold.GetAllCtrs(),
            candidateInfo.SplitEnsemble,
            docIndexRange,
            groupSize,
            bucketOffsets,
            &bucketIdx
        );
    };

    auto calcStats = [&] (TIndexRange<ui32> docIndexRange, int dim, TArrayRef<TBucketStats> stats) {
        CalcStatsKernel(
            fold,
            fold.BodyTailArr[0],
            dim,
            bucketCount,
            docIndexRange,
            bucketIdx,
            groupSize,
            GetDataPtr(stats)
        );
    };

    const auto updateSplitScoreClosure = [scoreCalcer] (
        const TBucketStats& trueStats,
        const TBucketStats& falseStats,
        int splitIdx
    ) {
        scoreCalcer->AddLeafPlain(splitIdx, falseStats, trueStats);
    };

    auto calcScores = [&] (TConstArrayRef<TBucketStats> stats) {
        const auto& getBucketStats = [stats] (int bucketIdx) {
            return stats[bucketIdx];
        };
        CalcScoresForLeaf(
            splitEnsembleSpec,
            oneHotMaxSize,
            bucketCount,
            getBucketStats,
            updateSplitScoreClosure);
    };

    // TODO(ilyzhin) make caching for nonsymmetric trees
    if (!ctx->UseTreeLevelCaching() || ctx->Params.ObliviousTreeOptions->GrowPolicy != EGrowPolicy::SymmetricTree) {
        TVector<TBucketStats> stats;
        stats.yresize(bucketCount);

        for (auto leaf : leafs) {
            const auto leafBounds = fold.LeavesBounds[leaf];
            if (leafBounds.Empty()) {
                continue;
            }

            extractBucketIndex(leafBounds);
            for (int dim : xrange(approxDimension)) {
                calcStats(leafBounds, dim, stats);
                calcScores(stats);
            }
        }
    } else { /* UseTreeLevelCaching */
        bool areStatsDirty;
        int maxStatsCount = bucketCount * (1 << ctx->Params.ObliviousTreeOptions->MaxDepth);
        TVector<TBucketStats, TPoolAllocator>& stats =
            ctx->PrevTreeLevelStats.GetStats(candidateInfo.SplitEnsemble, maxStatsCount, &areStatsDirty);

        if (fold.LeavesBounds.size() == 1 || areStatsDirty) {
            extractBucketIndex(TIndexRange<ui32>(0, fold.GetDocCount()));

            for (auto leaf : xrange(fold.LeavesBounds.size())) {
                const auto leafBounds = fold.LeavesBounds[leaf];
                if (leafBounds.Empty()) {
                    auto beginPtr = GetDataPtr(stats, bucketCount * (leaf * approxDimension));
                    auto endPtr = GetDataPtr(stats, bucketCount * ((leaf+1) * approxDimension));
                    Fill(beginPtr, endPtr, TBucketStats{0, 0, 0, 0});
                    continue;
                }

                for (int dim : xrange(approxDimension)) {
                    TArrayRef statsRef(
                        GetDataPtr(stats, bucketCount * (leaf * approxDimension + dim)),
                        bucketCount);
                    calcStats(leafBounds, dim, statsRef);
                    calcScores(statsRef);
                }
            }
        } else {
            for (auto idx : xrange(0, static_cast<int>(fold.LeavesCount / 2))) {
                auto leftLeaf = idx;
                auto rightLeaf = idx + fold.LeavesCount / 2;
                ui32 leftLeafSize = fold.LeavesBounds[leftLeaf].GetSize();
                ui32 rightLeafSize = fold.LeavesBounds[rightLeaf].GetSize();

                auto smallIndexRange = leftLeafSize < rightLeafSize
                    ? fold.LeavesBounds[leftLeaf]
                    : fold.LeavesBounds[rightLeaf];
                extractBucketIndex(smallIndexRange);
                for (int dim : xrange(approxDimension)) {
                    TArrayRef leftStatsRef(
                        GetDataPtr(stats, bucketCount * (leftLeaf * approxDimension + dim)),
                        bucketCount);
                    TArrayRef rightStatsRef(
                        GetDataPtr(stats, bucketCount * (rightLeaf * approxDimension + dim)),
                        bucketCount);
                    calcStats(smallIndexRange, dim, rightStatsRef);
                    calcScores(rightStatsRef);
                    for (auto bucket : xrange(bucketCount)) {
                        leftStatsRef[bucket].Remove(rightStatsRef[bucket]);
                    }
                    calcScores(leftStatsRef);
                    if (leftLeafSize < rightLeafSize) {
                        for (auto bucket : xrange(bucketCount)) {
                            std::swap(leftStatsRef[bucket], rightStatsRef[bucket]);
                        }
                    }
                }
            }
        }
    }
}

template <typename TScoreCalcer>
static TVector<TVector<double>> CalcScoresForOneCandidateImpl(
    const NCB::TQuantizedForCPUObjectsDataProvider& objectsDataProvider,
    const TCandidatesInfoList& candidate,
    const TCalcScoreFold& fold,
    const TFold& initialFold,
    const TVector<TIndexType>& leafs,
    TLearnContext* ctx
) {
    TVector<TVector<double>> scores(candidate.Candidates.size());

    ctx->LocalExecutor->ExecRange(
        [&](int subCandId) {
            const auto& candidateInfo = candidate.Candidates[subCandId];
            const auto& splitEnsemble = candidateInfo.SplitEnsemble;

            if (splitEnsemble.IsSplitOfType(ESplitType::OnlineCtr)) {
                const auto& proj = splitEnsemble.SplitCandidate.Ctr.Projection;
                Y_ASSERT(!initialFold.GetCtr(proj).Feature.empty());
            }

            const int bucketCount = GetBucketCount(
                splitEnsemble,
                *objectsDataProvider.GetQuantizedFeaturesInfo(),
                objectsDataProvider.GetPackedBinaryFeaturesSize(),
                objectsDataProvider.GetExclusiveFeatureBundlesMetaData(),
                objectsDataProvider.GetFeaturesGroupsMetaData()
            );
            const int bucketIndexBitCount = GetValueBitCount(bucketCount - 1);
            TSplitEnsembleSpec splitEnsembleSpec(
                splitEnsemble,
                objectsDataProvider.GetExclusiveFeatureBundlesMetaData(),
                objectsDataProvider.GetFeaturesGroupsMetaData()
            );
            const ui32 oneHotMaxSize = ctx->Params.CatFeatureParams.Get().OneHotMaxSize.Get();
            const int candidateSplitCount = CalcSplitsCount(
                splitEnsembleSpec, bucketCount, oneHotMaxSize
            );

            TScoreCalcer scoreCalcer;
            scoreCalcer.SetSplitsCount(candidateSplitCount);
            const double sumAllWeights = initialFold.BodyTailArr[0].BodySumWeight;
            const int docCount = initialFold.BodyTailArr[0].BodyFinish;
            const float l2Regularizer = static_cast<const float>(ctx->Params.ObliviousTreeOptions->L2Reg);
            const double scaledL2Regularizer = l2Regularizer * (sumAllWeights / docCount);
            scoreCalcer.SetL2Regularizer(scaledL2Regularizer);
            if (bucketIndexBitCount <= 8) {
                CalcScoresForSubCandidate<ui8>(
                    objectsDataProvider,
                    candidateInfo,
                    bucketCount,
                    fold,
                    initialFold,
                    leafs,
                    ctx,
                    &scoreCalcer);
            } else if (bucketIndexBitCount <= 16) {
                CalcScoresForSubCandidate<ui16>(
                    objectsDataProvider,
                    candidateInfo,
                    bucketCount,
                    fold,
                    initialFold,
                    leafs,
                    ctx,
                    &scoreCalcer);
            } else {
                Y_UNREACHABLE();
            }

            scores[subCandId] = scoreCalcer.GetScores();
        },
        0,
        candidate.Candidates.ysize(),
        NPar::TLocalExecutor::WAIT_COMPLETE);

    return scores;
}

TVector<TVector<double>> CalcScoresForOneCandidate(
    const NCB::TQuantizedForCPUObjectsDataProvider& data,
    const TCandidatesInfoList& candidate,
    const TCalcScoreFold& fold,
    const TFold& initialFold,
    const TVector<TIndexType>& leafs,
    TLearnContext* ctx
) {
    const auto scoreFunction = ctx->Params.ObliviousTreeOptions->ScoreFunction;
    if (scoreFunction == EScoreFunction::Cosine) {
        return CalcScoresForOneCandidateImpl<TCosineScoreCalcer>(
            data,
            candidate,
            fold,
            initialFold,
            leafs,
            ctx);
    } else if (scoreFunction == EScoreFunction::L2) {
        return CalcScoresForOneCandidateImpl<TL2ScoreCalcer>(
            data,
            candidate,
            fold,
            initialFold,
            leafs,
            ctx);
    } else {
        CB_ENSURE(false, "Error: score function for CPU should be Cosine or L2");
    }
}


double CalcScoreWithoutSplit(int leaf, const TFold& fold, const TLearnContext& ctx) {
    // TODO(ilyzhin) make parallel
    const auto& leafBounds = ctx.SampledDocs.LeavesBounds[leaf];
    double sumWeightedDerivatives = 0;
    const auto& ders = ctx.SampledDocs.BodyTailArr[0].SampleWeightedDerivatives;
    for (auto dim : xrange(ctx.LearnProgress->ApproxDimension)) {
        sumWeightedDerivatives += Accumulate(
            ders[dim].begin() + leafBounds.Begin,
            ders[dim].begin() + leafBounds.End,
            0.0);
    }
    double sumWeights = Accumulate(
        ctx.SampledDocs.SampleWeights.begin() + leafBounds.Begin,
        ctx.SampledDocs.SampleWeights.begin() + leafBounds.End,
        0.0);
    TBucketStats stats {sumWeightedDerivatives, sumWeights, 0, 0};

    const double sumAllWeights = fold.BodyTailArr[0].BodySumWeight;
    const int docCount = fold.BodyTailArr[0].BodyFinish;
    const double scaledL2Regularizer = ScaleL2Reg(ctx.Params.ObliviousTreeOptions->L2Reg, sumAllWeights, docCount);

    auto scoreCalcer = MakePointwiseScoreCalcer(ctx.Params.ObliviousTreeOptions->ScoreFunction);
    scoreCalcer->SetL2Regularizer(scaledL2Regularizer);
    scoreCalcer->SetSplitsCount(1);
    scoreCalcer->AddLeafPlain(0, stats, {});
    return scoreCalcer->GetScores()[0];
}
