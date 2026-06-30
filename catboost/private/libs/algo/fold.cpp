#include "fold.h"

#include "approx_updater_helpers.h"
#include "estimated_features.h"
#include "helpers.h"

#include <catboost/private/libs/data_types/groupid.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/permutation.h>
#include <catboost/libs/helpers/query_info_helper.h>
#include <catboost/libs/helpers/restorable_rng.h>
#include <catboost/libs/logging/logging.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/cast.h>


using namespace NCB;


static ui32 UpdateSize(
    ui32 size,
    const TVector<TQueryInfo>& queryInfo,
    const TVector<ui32>& queryIndices,
    ui32 learnSampleCount
) {
    size = Min(size, learnSampleCount);
    if (!queryInfo.empty()) {
        size = queryInfo[queryIndices[size - 1]].End;
    }
    return size;
}

static ui32 SelectMinBatchSize(ui32 learnSampleCount) {
    return learnSampleCount > 500 ? Min<ui32>(100, learnSampleCount / 50) : 1;
}

static double SelectTailSize(ui32 oldSize, double multiplier) {
    return ceil(oldSize * multiplier);
}

static void InitPermutationData(
    const NCB::TTrainingDataProvider& learnData,
    bool shuffle,
    ui32 permuteBlockSize,
    TRestorableFastRng64* rand,
    TFold* fold
) {
    const ui32 learnSampleCount = learnData.GetObjectCount();
    const auto& featuresArraySubsetIndexing = learnData.ObjectsData->GetFeaturesArraySubsetIndexing();

    TMaybe<ui32> consecutiveSubsetBegin = featuresArraySubsetIndexing.GetConsecutiveSubsetBegin();
    if (shuffle) {
        if (consecutiveSubsetBegin) {
            fold->PermutationBlockSize = learnData.ObjectsGrouping->IsTrivial() ? permuteBlockSize : 1;
            fold->FeaturesSubsetBegin = *consecutiveSubsetBegin;
        } else {
            fold->PermutationBlockSize = 1;
        }
        fold->LearnPermutation = Shuffle(learnData.ObjectsGrouping, fold->PermutationBlockSize, rand);
        fold->LearnPermutationFeaturesSubset = Compose(
            featuresArraySubsetIndexing,
            fold->LearnPermutation->GetObjectsIndexing()
        );
    } else {
        if (consecutiveSubsetBegin) {
            fold->PermutationBlockSize = learnSampleCount;
            fold->FeaturesSubsetBegin = *consecutiveSubsetBegin;
        } else {
            fold->PermutationBlockSize = 1;
        }

        // implementation requires permutation vectors to exist even if they are not shuffled
        TIndexedSubset<ui32> learnPermutation;
        learnPermutation.yresize(learnSampleCount);
        std::iota(learnPermutation.begin(), learnPermutation.end(), 0);

        fold->LearnPermutation = TObjectsGroupingSubset(
            learnData.ObjectsGrouping,
            TArraySubsetIndexing<ui32>(TFullSubset<ui32>(learnData.ObjectsGrouping->GetGroupCount())),
            EObjectsOrder::Ordered,
            MakeMaybe<TFeaturesArraySubsetIndexing>(std::move(learnPermutation)),
            EObjectsOrder::Ordered
        );

        TIndexedSubset<ui32> learnPermutationFeaturesSubset;
        learnPermutationFeaturesSubset.yresize(learnSampleCount);
        featuresArraySubsetIndexing.ForEach(
            [&] (ui32 idx, ui32 srcIdx) { learnPermutationFeaturesSubset[idx] = srcIdx; }
        );
        fold->LearnPermutationFeaturesSubset = TFeaturesArraySubsetIndexing(
            std::move(learnPermutationFeaturesSubset)
        );
    }
}


TFold TFold::BuildDynamicFold(
    const NCB::TTrainingDataProviders& data,
    const TVector<TTargetClassifier>& targetClassifiers,
    bool shuffle,
    ui32 permuteBlockSize,
    int approxDimension,
    double multiplier,
    bool storeExpApproxes,
    bool hasPairwiseWeights,
    const TMaybe<TVector<double>>& startingApprox,
    const NCatboostOptions::TBinarizationOptions& onlineEstimatedFeaturesQuantizationOptions,
    TQuantizedFeaturesInfoPtr onlineEstimatedFeaturesQuantizedInfo,
    TRestorableFastRng64* rand,
    NPar::ILocalExecutor* localExecutor
) {
    const NCB::TTrainingDataProvider& learnData = *data.Learn;

    const ui32 learnSampleCount = learnData.GetObjectCount();

    TFold ff;
    ff.SampleWeights.resize(learnSampleCount, 1);

    InitPermutationData(learnData, shuffle, permuteBlockSize, rand, &ff);

    ff.AssignTarget(learnData.TargetData->GetTarget(), targetClassifiers, localExecutor);
    ff.SetWeights(GetWeights(*learnData.TargetData), learnSampleCount);

    TVector<ui32> queryIndices;

    auto maybeGroupInfos = learnData.TargetData->GetGroupInfo();
    if (maybeGroupInfos) {
        if (shuffle) {
            GetGroupInfosSubset(*maybeGroupInfos, *ff.LearnPermutation, localExecutor, &ff.LearnQueriesInfo);
        } else {
            ff.LearnQueriesInfo.insert(
                ff.LearnQueriesInfo.end(),
                maybeGroupInfos->begin(),
                maybeGroupInfos->end()
            );
        }
        queryIndices = GetQueryIndicesForDocs(ff.LearnQueriesInfo, learnSampleCount);
    }

    TVector<float> pairwiseWeights;
    if (hasPairwiseWeights) {
        pairwiseWeights.resize(learnSampleCount);
        CalcPairwiseWeights(ff.LearnQueriesInfo, ff.LearnQueriesInfo.ysize(), &pairwiseWeights);
    }

    TMaybeData<TConstArrayRef<TConstArrayRef<float>>> baseline = learnData.TargetData->GetBaseline();

    ui32 leftPartLen = UpdateSize(
        SelectMinBatchSize(learnSampleCount),
        ff.LearnQueriesInfo,
        queryIndices,
        learnSampleCount
    );
    while (ff.BodyTailArr.empty() || leftPartLen < learnSampleCount) {
        int bodyFinish = (int)leftPartLen;
        int tailFinish = (int) UpdateSize(
            SelectTailSize(leftPartLen, multiplier),
            ff.LearnQueriesInfo,
            queryIndices,
            learnSampleCount
        );
        int bodyQueryFinish = 0;
        int tailQueryFinish = 0;
        if (maybeGroupInfos) {
            bodyQueryFinish = queryIndices[bodyFinish - 1] + 1;
            tailQueryFinish = queryIndices[tailFinish - 1] + 1;
        }
        double bodySumWeight = ff.GetLearnWeights().empty()
            ? bodyFinish
            : Accumulate(ff.GetLearnWeights().begin(), ff.GetLearnWeights().begin() + bodyFinish, (double)0.0);

        TFold::TBodyTail bt(bodyQueryFinish, tailQueryFinish, bodyFinish, tailFinish, bodySumWeight);
        InitApproxes(bt.TailFinish, startingApprox, approxDimension, storeExpApproxes,  &(bt.Approx));

        if (baseline) {
            InitApproxFromBaseline(
                bt.TailFinish,
                *baseline,
                ff.GetLearnPermutationArray(),
                storeExpApproxes,
                &bt.Approx
            );
        }
        AllocateRank2(approxDimension, bt.TailFinish, bt.WeightedDerivatives);
        ResizeRank2(approxDimension, bt.TailFinish, bt.SampleWeightedDerivatives);
        if (hasPairwiseWeights) {
            bt.PairwiseWeights.insert(
                bt.PairwiseWeights.begin(),
                pairwiseWeights.begin(),
                pairwiseWeights.begin() + bt.TailFinish
            );
            bt.SamplePairwiseWeights.resize(bt.TailFinish);
        }
        ff.BodyTailArr.emplace_back(std::move(bt));
        leftPartLen = (ui32)bt.TailFinish;
    }

    ff.InitOnlineEstimatedFeatures(
        onlineEstimatedFeaturesQuantizationOptions,
        std::move(onlineEstimatedFeaturesQuantizedInfo),
        data,
        localExecutor,
        rand
    );

    ff.InitOnlineCtrs(data);

    return ff;
}

void TFold::SetWeights(TConstArrayRef<float> weights, ui32 learnSampleCount) {
    if (!weights.empty()) {
        AssignPermuted(weights, &LearnWeights);
        SumWeight = Accumulate(weights.begin(), weights.end(), (double)0.0);
    } else {
        SumWeight = learnSampleCount;
    }
}

TFold TFold::BuildPlainFold(
    const NCB::TTrainingDataProviders& data,
    const TVector<TTargetClassifier>& targetClassifiers,
    bool shuffle,
    ui32 permuteBlockSize,
    int approxDimension,
    bool storeExpApproxes,
    bool hasPairwiseWeights,
    const TMaybe<TVector<double>>& startingApprox,
    const NCatboostOptions::TBinarizationOptions& onlineEstimatedFeaturesQuantizationOptions,
    TQuantizedFeaturesInfoPtr onlineEstimatedFeaturesQuantizedInfo,
    TIntrusivePtr<TPrecomputedOnlineCtr> precomputedSingleOnlineCtrs,
    TRestorableFastRng64* rand,
    NPar::ILocalExecutor* localExecutor
) {
    const NCB::TTrainingDataProvider& learnData = *data.Learn;

    const ui32 learnSampleCount = learnData.GetObjectCount();

    TFold ff;

    InitPermutationData(learnData, shuffle, permuteBlockSize, rand, &ff);

    if (learnSampleCount) {
        ff.SampleWeights.resize(learnSampleCount, 1);

        ff.AssignTarget(learnData.TargetData->GetTarget(), targetClassifiers, localExecutor);
        ff.SetWeights(GetWeights(*learnData.TargetData), learnSampleCount);

        auto maybeGroupInfos = learnData.TargetData->GetGroupInfo();
        int groupCountAsInt = 0;
        if (maybeGroupInfos) {
            if (shuffle) {
                GetGroupInfosSubset(*maybeGroupInfos, *ff.LearnPermutation, localExecutor, &ff.LearnQueriesInfo);
            } else {
                ff.LearnQueriesInfo.insert(
                    ff.LearnQueriesInfo.end(),
                    maybeGroupInfos->begin(),
                    maybeGroupInfos->end()
                );
            }
            groupCountAsInt = SafeIntegerCast<int>(maybeGroupInfos->size());
        }

        const int learnSampleCountAsInt = SafeIntegerCast<int>(learnSampleCount);

        TFold::TBodyTail bt(
            groupCountAsInt,
            groupCountAsInt,
            learnSampleCountAsInt,
            learnSampleCountAsInt,
            ff.GetSumWeight()
        );

        InitApproxes(learnSampleCount, startingApprox, approxDimension, storeExpApproxes, &(bt.Approx));
        AllocateRank2(approxDimension, learnSampleCount, bt.WeightedDerivatives);
        ResizeRank2(approxDimension, learnSampleCount, bt.SampleWeightedDerivatives);
        if (hasPairwiseWeights) {
            bt.PairwiseWeights.resize(learnSampleCount);
            CalcPairwiseWeights(ff.LearnQueriesInfo, bt.TailQueryFinish, &bt.PairwiseWeights);
            bt.SamplePairwiseWeights.resize(learnSampleCount);
        }

        TMaybeData<TConstArrayRef<TConstArrayRef<float>>> baseline = learnData.TargetData->GetBaseline();
        if (baseline) {
            InitApproxFromBaseline(
                learnSampleCount,
                *baseline,
                ff.GetLearnPermutationArray(),
                storeExpApproxes,
                &bt.Approx
            );
        }
        ff.BodyTailArr.emplace_back(std::move(bt));
    }

    ff.InitOnlineEstimatedFeatures(
        onlineEstimatedFeaturesQuantizationOptions,
        std::move(onlineEstimatedFeaturesQuantizedInfo),
        data,
        localExecutor,
        rand
    );

    ff.InitOnlineCtrs(data, precomputedSingleOnlineCtrs);

    return ff;
}


void TFold::DropEmptyCTRs() {
    TVector<TProjection> emptyProjections;
    if (OwnedOnlineSingleCtrs) {
        OwnedOnlineSingleCtrs->DropEmptyData();
    }
    if (OwnedOnlineCtrs) {
        OwnedOnlineCtrs->DropEmptyData();
    }
}

void TFold::AssignTarget(
    TMaybeData<TConstArrayRef<TConstArrayRef<float>>> target,
    const TVector<TTargetClassifier>& targetClassifiers,
    NPar::ILocalExecutor* localExecutor
) {
    ui32 learnSampleCount = GetLearnSampleCount();
    if (target && target->size() > 0) {
        LearnTarget.yresize(target->size());
        NPar::ParallelFor(
            *localExecutor,
            0,
            target->size(),
            [&] (ui32 targetIdx) {
                LearnTarget[targetIdx] = NCB::GetSubset<float>((*target)[targetIdx], LearnPermutation->GetObjectsIndexing());
            }
        );
    } else {
        // TODO(akhropov): make this field optional
        LearnTarget = TVector<TVector<float>>{TVector<float>(learnSampleCount, 0.0f)};
    }

    int ctrCount = targetClassifiers.ysize();
    AllocateRank2(ctrCount, learnSampleCount, LearnTargetClass);
    TargetClassesCount.resize(ctrCount);
    for (int ctrIdx = 0; ctrIdx < ctrCount; ++ctrIdx) {
        NPar::ParallelFor(
            *localExecutor,
            0,
            learnSampleCount,
            [&] (ui32 z) {
                auto targetId = targetClassifiers[ctrIdx].GetTargetId();
                LearnTargetClass[ctrIdx][z] = targetClassifiers[ctrIdx].GetTargetClass(LearnTarget[targetId][z]);
            }
        );
        TargetClassesCount[ctrIdx] = targetClassifiers[ctrIdx].GetClassesCount();
    }
}

void TFold::SaveApproxes(IOutputStream* s) const {
    const ui64 bodyTailCount = BodyTailArr.size();
    ::Save(s, bodyTailCount);
    for (ui64 i = 0; i < bodyTailCount; ++i) {
        ::Save(s, BodyTailArr[i].Approx);
    }
}

void TFold::LoadApproxes(IInputStream* s) {
    ui64 bodyTailCount;
    ::Load(s, bodyTailCount);
    BodyTailArr.resize(bodyTailCount);
    for (ui64 i = 0; i < bodyTailCount; ++i) {
        ::Load(s, BodyTailArr[i].Approx);
    }
}

void TFold::InitOnlineEstimatedFeatures(
    const NCatboostOptions::TBinarizationOptions& quantizationOptions,
    TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
    const NCB::TTrainingDataProviders& data,
    NPar::ILocalExecutor* localExecutor,
    TRestorableFastRng64* rand
) {
    OnlineEstimatedFeatures = CreateEstimatedFeaturesData(
        quantizationOptions,
        /*maxSubsetSizeForBuildBordersAlgorithms*/ 100000,
        std::move(quantizedFeaturesInfo),
        data,
        data.FeatureEstimators,
        GetLearnPermutationArray(),
        localExecutor,
        rand
    );
}

void TFold::InitOnlineCtrs(
    const NCB::TTrainingDataProviders& data,
    TIntrusivePtr<TPrecomputedOnlineCtr> precomputedSingleOnlineCtrs
) {
    TVector<TIndexRange<size_t>> datasetsObjectRanges;
    size_t offset = 0;
    datasetsObjectRanges.push_back(TIndexRange<size_t>(0, data.Learn->GetObjectCount()));
    offset += data.Learn->GetObjectCount();
    for (const auto& test : data.Test) {
        size_t size = test->GetObjectCount();
        datasetsObjectRanges.push_back(TIndexRange<size_t>(offset, offset + size));
        offset += size;
    }

    if (precomputedSingleOnlineCtrs) {
        CATBOOST_DEBUG_LOG << "Fold: Use precomputed online single ctrs\n";

        OnlineSingleCtrs = std::move(precomputedSingleOnlineCtrs);
        OwnedOnlineSingleCtrs = nullptr;
    } else {
        CATBOOST_DEBUG_LOG << "Fold: Use owned online single ctrs\n";

        OwnedOnlineSingleCtrs = new TOwnedOnlineCtr();
        OnlineSingleCtrs.Reset(OwnedOnlineSingleCtrs);
        OwnedOnlineSingleCtrs->DatasetsObjectRanges = datasetsObjectRanges;
    }

    OwnedOnlineCtrs = new TOwnedOnlineCtr();
    OnlineCtrs.Reset(OwnedOnlineCtrs);
    OwnedOnlineCtrs->DatasetsObjectRanges = std::move(datasetsObjectRanges);
}
