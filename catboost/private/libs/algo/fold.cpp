#include "fold.h"

#include "approx_updater_helpers.h"
#include "estimated_features.h"
#include "helpers.h"

#include <catboost/private/libs/data_types/groupid.h>
#include <catboost/libs/helpers/permutation.h>
#include <catboost/libs/helpers/query_info_helper.h>
#include <catboost/libs/helpers/restorable_rng.h>

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
    TMaybe<double> startingApprox,
    const NCatboostOptions::TBinarizationOptions& onlineEstimatedFeaturesQuantizationOptions,
    TQuantizedFeaturesInfoPtr onlineEstimatedFeaturesQuantizedInfo,
    TRestorableFastRng64* rand,
    NPar::TLocalExecutor* localExecutor
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
        bt.Approx.resize(
            approxDimension,
            TVector<double>(
                bt.TailFinish,
                startingApprox ? ExpApproxIf(storeExpApproxes, *startingApprox) : GetNeutralApprox(storeExpApproxes)
            )
        );
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
        AllocateRank2(approxDimension, bt.TailFinish, bt.SampleWeightedDerivatives);
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
    TMaybe<double> startingApprox,
    const NCatboostOptions::TBinarizationOptions& onlineEstimatedFeaturesQuantizationOptions,
    TQuantizedFeaturesInfoPtr onlineEstimatedFeaturesQuantizedInfo,
    TRestorableFastRng64* rand,
    NPar::TLocalExecutor* localExecutor
) {
    const NCB::TTrainingDataProvider& learnData = *data.Learn;

    const ui32 learnSampleCount = learnData.GetObjectCount();

    TFold ff;
    ff.SampleWeights.resize(learnSampleCount, 1);

    InitPermutationData(learnData, shuffle, permuteBlockSize, rand, &ff);

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

    bt.Approx.resize(approxDimension,
        TVector<double>(
            learnSampleCount,
            startingApprox ? ExpApproxIf(storeExpApproxes, *startingApprox) : GetNeutralApprox(storeExpApproxes)));
    AllocateRank2(approxDimension, learnSampleCount, bt.WeightedDerivatives);
    AllocateRank2(approxDimension, learnSampleCount, bt.SampleWeightedDerivatives);
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

    ff.InitOnlineEstimatedFeatures(
        onlineEstimatedFeaturesQuantizationOptions,
        std::move(onlineEstimatedFeaturesQuantizedInfo),
        data,
        localExecutor,
        rand
    );

    return ff;
}


void TFold::DropEmptyCTRs() {
    TVector<TProjection> emptyProjections;
    for (auto& projCtr : OnlineSingleCtrs) {
        if (projCtr.second.Feature.empty()) {
            emptyProjections.emplace_back(projCtr.first);
        }
    }
    for (auto& projCtr : OnlineCTR) {
        if (projCtr.second.Feature.empty()) {
            emptyProjections.emplace_back(projCtr.first);
        }
    }
    for (const auto& proj : emptyProjections) {
        GetCtrs(proj).erase(proj);
    }
}

void TFold::AssignTarget(
    TMaybeData<TConstArrayRef<TConstArrayRef<float>>> target,
    const TVector<TTargetClassifier>& targetClassifiers,
    NPar::TLocalExecutor* localExecutor
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
                LearnTargetClass[ctrIdx][z] = targetClassifiers[ctrIdx].GetTargetClass(LearnTarget[0][z]);
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
    NPar::TLocalExecutor* localExecutor,
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
