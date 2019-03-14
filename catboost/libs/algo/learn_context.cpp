#include "calc_score_cache.h"
#include "learn_context.h"
#include "error_functions.h"
#include "online_ctr.h"

#include <catboost/libs/distributed/master.h>
#include <catboost/libs/helpers/progress_helper.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/options/defaults_helper.h>

#include <library/digest/crc32c/crc32c.h>
#include <library/digest/md5/md5.h>

#include <util/generic/algorithm.h>
#include <util/generic/guid.h>
#include <util/generic/xrange.h>
#include <util/folder/path.h>
#include <util/system/fs.h>
#include <util/stream/file.h>


using namespace NCB;



TLearnContext::~TLearnContext() {
    if (Params.SystemOptions->IsMaster()) {
        FinalizeMaster(this);
    }
}

void TLearnContext::OutputMeta() {
    auto losses = CreateMetrics(
            Params.LossFunctionDescription,
            Params.MetricOptions,
            EvalMetricDescriptor,
            LearnProgress.ApproxDimension
    );

    CreateMetaFile(Files, OutputOptions, GetConstPointers(losses), Params.BoostingOptions->IterationCount);
}

static bool IsPermutationNeeded(bool hasTime, bool hasCtrs, bool isOrderedBoosting, bool isAveragingFold) {
    if (hasTime) {
        return false;
    }
    if (hasCtrs) {
        return true;
    }
    return isOrderedBoosting && !isAveragingFold;
}

static int CountLearningFolds(int permutationCount, bool isPermutationNeededForLearning) {
    return isPermutationNeededForLearning ? Max<ui32>(1, permutationCount - 1) : 1;
}

void TLearnContext::InitContext(const TTrainingForCPUDataProviders& data) {
    LearnProgress.EnableSaveLoadApprox = Params.SystemOptions->IsSingleHost();
    LearnProgress.PoolCheckSum = data.Learn->ObjectsData->CalcFeaturesCheckSum(LocalExecutor);
    for (const auto& testData : data.Test) {
        LearnProgress.PoolCheckSum += testData->ObjectsData->CalcFeaturesCheckSum(LocalExecutor);
    }

    auto lossFunction = Params.LossFunctionDescription->GetLossFunction();
    const bool hasCtrs =
        data.Learn->ObjectsData->GetQuantizedFeaturesInfo()
            ->CalcMaxCategoricalFeaturesUniqueValuesCountOnLearn()
          > Params.CatFeatureParams->OneHotMaxSize.Get();

    const bool hasTime = Params.DataProcessingOptions->HasTimeFlag
        || (data.Learn->ObjectsData->GetOrder() == EObjectsOrder::Ordered);
    const bool isOrderedBoosting = !IsPlainMode(Params.BoostingOptions->BoostingType);
    const bool isLearnFoldPermuted = IsPermutationNeeded(hasTime, hasCtrs, isOrderedBoosting, /*isAveragingFold*/ false);
    const int learningFoldCount = CountLearningFolds(Params.BoostingOptions->PermutationCount, isLearnFoldPermuted);

    LearnProgress.Folds.reserve(learningFoldCount);
    UpdateCtrsTargetBordersOption(lossFunction, LearnProgress.ApproxDimension, &Params.CatFeatureParams.Get());

    CtrsHelper.InitCtrHelper(Params.CatFeatureParams,
                             *Layout,
                             data.Learn->TargetData->GetTarget(),
                             lossFunction,
                             ObjectiveDescriptor,
                             Params.DataProcessingOptions->AllowConstLabel);

    //Todo(noxoomo): check and init
    const auto& boostingOptions = Params.BoostingOptions.Get();
    const ui32 learnSampleCount = data.Learn->GetObjectCount();
    ui32 foldPermutationBlockSize = boostingOptions.PermutationBlockSize;
    if (foldPermutationBlockSize == FoldPermutationBlockSizeNotSet) {
        foldPermutationBlockSize = DefaultFoldPermutationBlockSize(learnSampleCount);
    }
    if (!isLearnFoldPermuted) {
        foldPermutationBlockSize = learnSampleCount;
    }
    const auto storeExpApproxes = IsStoreExpApprox(Params.LossFunctionDescription->GetLossFunction());
    const bool hasPairwiseWeights = UsesPairsForCalculation(Params.LossFunctionDescription->GetLossFunction());

    if (IsPlainMode(Params.BoostingOptions->BoostingType)) {
        for (int foldIdx = 0; foldIdx < learningFoldCount; ++foldIdx) {
            LearnProgress.Folds.emplace_back(
                TFold::BuildPlainFold(
                    *data.Learn,
                    CtrsHelper.GetTargetClassifiers(),
                    foldIdx != 0,
                    (Params.SystemOptions->IsSingleHost() ? foldPermutationBlockSize : learnSampleCount),
                    LearnProgress.ApproxDimension,
                    storeExpApproxes,
                    hasPairwiseWeights,
                    Rand,
                    LocalExecutor
                )
            );
        }
    } else {
        for (int foldIdx = 0; foldIdx < learningFoldCount; ++foldIdx) {
            LearnProgress.Folds.emplace_back(
                TFold::BuildDynamicFold(
                    *data.Learn,
                    CtrsHelper.GetTargetClassifiers(),
                    foldIdx != 0,
                    foldPermutationBlockSize,
                    LearnProgress.ApproxDimension,
                    boostingOptions.FoldLenMultiplier,
                    storeExpApproxes,
                    hasPairwiseWeights,
                    Rand,
                    LocalExecutor
                )
            );
        }
    }

    const bool isAverageFoldPermuted = IsPermutationNeeded(hasTime, hasCtrs, isOrderedBoosting, /*isAveragingFold*/ true);
    LearnProgress.AveragingFold = TFold::BuildPlainFold(
        *data.Learn,
        CtrsHelper.GetTargetClassifiers(),
        isAverageFoldPermuted,
        /*permuteBlockSize=*/ (Params.SystemOptions->IsSingleHost() ? foldPermutationBlockSize : learnSampleCount),
        LearnProgress.ApproxDimension,
        storeExpApproxes,
        hasPairwiseWeights,
        Rand,
        LocalExecutor
    );

    LearnProgress.AvrgApprox.resize(LearnProgress.ApproxDimension, TVector<double>(learnSampleCount));
    TMaybeData<TConstArrayRef<TConstArrayRef<float>>> learnBaseline = data.Learn->TargetData->GetBaseline();
    if (learnBaseline) {
        AssignRank2<float>(*learnBaseline, &LearnProgress.AvrgApprox);
    }
    ResizeRank2(data.Test.size(), LearnProgress.ApproxDimension, LearnProgress.TestApprox);
    for (size_t testIdx = 0; testIdx < data.Test.size(); ++testIdx) {
        const auto* testData = data.Test[testIdx].Get();
        if (testData == nullptr || testData->GetObjectCount() == 0) {
            continue;
        }
        TMaybeData<TConstArrayRef<TConstArrayRef<float>>> testBaseline = testData->TargetData->GetBaseline();
        if (!testBaseline) {
            for (auto& approxDim : LearnProgress.TestApprox[testIdx]) {
                approxDim.resize(testData->GetObjectCount());
            }
        } else {
            AssignRank2<float>(*testBaseline, &LearnProgress.TestApprox[testIdx]);
        }
    }

    const ui32 maxBodyTailCount = Max(1, GetMaxBodyTailCount(LearnProgress.Folds));
    UseTreeLevelCachingFlag = NeedToUseTreeLevelCaching(Params, maxBodyTailCount, LearnProgress.ApproxDimension);
}

void TLearnContext::SaveProgress() {
    if (!OutputOptions.SaveSnapshot()) {
        return;
    }
    TProgressHelper(ToString(ETaskType::CPU)).Write(Files.SnapshotFile, [&](IOutputStream* out) {
        ::SaveMany(out, Rand, LearnProgress, Profile.DumpProfileInfo());
    });
}

bool TLearnContext::TryLoadProgress() {
    if (!OutputOptions.SaveSnapshot() || !NFs::Exists(Files.SnapshotFile)) {
        return false;
    }
    try {
        TProgressHelper(ToString(ETaskType::CPU)).CheckedLoad(Files.SnapshotFile, [&](TIFStream* in)
        {
            // use progress copy to avoid partial deserialization of corrupted progress file
            TLearnProgress learnProgressRestored = LearnProgress;
            TProfileInfoData ProfileRestored;

            // fail here does nothing with real LearnProgress
            ::LoadMany(in, Rand, learnProgressRestored, ProfileRestored);

            const bool paramsCompatible = NCatboostOptions::IsParamsCompatible(
                learnProgressRestored.SerializedTrainParams,
                LearnProgress.SerializedTrainParams);
            CATBOOST_DEBUG_LOG
                << LabeledOutput(learnProgressRestored.SerializedTrainParams) << ' '
                << LabeledOutput(LearnProgress.SerializedTrainParams) << Endl;
            CB_ENSURE(paramsCompatible, "Saved model's params are different from current model's params");

            const bool poolCompatible = (learnProgressRestored.PoolCheckSum == LearnProgress.PoolCheckSum);
            CB_ENSURE(
                poolCompatible,
                "Current pool differs from the original pool "
                LabeledOutput(learnProgressRestored.PoolCheckSum, LearnProgress.PoolCheckSum));

            LearnProgress = std::move(learnProgressRestored);
            Profile.InitProfileInfo(std::move(ProfileRestored));
            LearnProgress.SerializedTrainParams = ToString(Params); // substitute real
            CATBOOST_INFO_LOG << "Loaded progress file containing " << LearnProgress.TreeStruct.size() << " trees" << Endl;
        });
        return true;
    } catch(const TCatBoostException&) {
        throw;
    } catch (...) {
        CATBOOST_WARNING_LOG << "Can't load progress from snapshot file: " << Files.SnapshotFile << " exception: "
                            << CurrentExceptionMessage() << Endl;
        return false;
    }
}

void TLearnProgress::Save(IOutputStream* s) const {
    ::Save(s, SerializedTrainParams);
    ::Save(s, EnableSaveLoadApprox);
    if (EnableSaveLoadApprox) {
        ui64 foldCount = Folds.size();
        ::Save(s, foldCount);
        for (ui64 i = 0; i < foldCount; ++i) {
            Folds[i].SaveApproxes(s);
        }
        AveragingFold.SaveApproxes(s);
        ::SaveMany(s, AvrgApprox);
    }
    ::SaveMany(s,
        TestApprox,
        BestTestApprox,
        CatFeatures,
        FloatFeatures,
        ApproxDimension,
        TreeStruct,
        TreeStats,
        LeafValues,
        MetricsAndTimeHistory,
        UsedCtrSplits,
        PoolCheckSum);
}

void TLearnProgress::Load(IInputStream* s) {
    ::Load(s, SerializedTrainParams);
    bool enableSaveLoadApprox;
    ::Load(s, enableSaveLoadApprox);
    CB_ENSURE(enableSaveLoadApprox == EnableSaveLoadApprox, "Cannot load progress from file");
    if (EnableSaveLoadApprox) {
        ui64 foldCount;
        ::Load(s, foldCount);
        CB_ENSURE(foldCount == Folds.size(), "Cannot load progress from file");
        for (ui64 i = 0; i < foldCount; ++i) {
            Folds[i].LoadApproxes(s);
        }
        AveragingFold.LoadApproxes(s);
        ::Load(s, AvrgApprox);
    }
    ::LoadMany(s,
               TestApprox,
               BestTestApprox,
               CatFeatures,
               FloatFeatures,
               ApproxDimension,
               TreeStruct,
               TreeStats,
               LeafValues,
               MetricsAndTimeHistory,
               UsedCtrSplits,
               PoolCheckSum);
}

bool TLearnContext::UseTreeLevelCaching() const {
    return UseTreeLevelCachingFlag;
}

bool NeedToUseTreeLevelCaching(
    const NCatboostOptions::TCatBoostOptions& params,
    ui32 maxBodyTailCount,
    ui32 approxDimension) {

    const ui32 maxLeafCount = 1 << params.ObliviousTreeOptions->MaxDepth;
    // TODO(nikitxskv): Pairwise scoring doesn't use statistics from previous tree level. Need to fix it.
    return (
        IsSamplingPerTree(params.ObliviousTreeOptions) &&
        !IsPairwiseScoring(params.LossFunctionDescription->GetLossFunction()) &&
        maxLeafCount * approxDimension * maxBodyTailCount < 64 * 1 * 10);
}
