#include "calc_score_cache.h"
#include "learn_context.h"
#include "error_functions.h"

#include <catboost/libs/distributed/master.h>
#include <catboost/libs/helpers/progress_helper.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/options/defaults_helper.h>

#include <library/digest/crc32c/crc32c.h>
#include <library/digest/md5/md5.h>

#include <util/folder/path.h>
#include <util/generic/guid.h>
#include <util/stream/file.h>
#include <util/stream/labeled.h>
#include <util/system/fs.h>

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

template <typename T>
static ui32 CalcMatrixCheckSum(ui32 init, const TVector<TVector<T>>& matrix) {
    ui32 checkSum = init;
    for (const auto& row : matrix) {
        checkSum = Crc32cExtend(checkSum, row.data(), row.size() * sizeof(T));
    }
    return checkSum;
}

static ui32 CalcFeaturesCheckSum(const TAllFeatures& allFeatures) {
    ui32 checkSum = 0;
    checkSum = CalcMatrixCheckSum(checkSum, allFeatures.FloatHistograms);
    checkSum = CalcMatrixCheckSum(checkSum, allFeatures.CatFeaturesRemapped);
    checkSum = CalcMatrixCheckSum(checkSum, allFeatures.OneHotValues);
    return checkSum;
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

void TLearnContext::InitContext(const TDataset& learnData, const TDatasetPtrs& testDataPtrs) {
    LearnProgress.EnableSaveLoadApprox = Params.SystemOptions->IsSingleHost();
    LearnProgress.PoolCheckSum = CalcFeaturesCheckSum(learnData.AllFeatures);
    for (const TDataset* testData : testDataPtrs) {
        LearnProgress.PoolCheckSum += CalcFeaturesCheckSum(testData->AllFeatures);
    }

    auto lossFunction = Params.LossFunctionDescription->GetLossFunction();
    const bool hasCtrs = !IsCategoricalFeaturesEmpty(learnData.AllFeatures);
    const bool hasTime = Params.DataProcessingOptions->HasTimeFlag;
    const bool isOrderedBoosting = !IsPlainMode(Params.BoostingOptions->BoostingType);
    const bool isLearnFoldPermuted = IsPermutationNeeded(hasTime, hasCtrs, isOrderedBoosting, /*isAveragingFold*/ false);
    const int learningFoldCount = CountLearningFolds(Params.BoostingOptions->PermutationCount, isLearnFoldPermuted);

    LearnProgress.Folds.reserve(learningFoldCount);
    UpdateCtrsTargetBordersOption(lossFunction, LearnProgress.ApproxDimension, &Params.CatFeatureParams.Get());

    CtrsHelper.InitCtrHelper(Params.CatFeatureParams,
                             Layout,
                             learnData.Target,
                             lossFunction,
                             ObjectiveDescriptor,
                             Params.DataProcessingOptions->AllowConstLabel);

    //Todo(noxoomo): check and init
    const auto& boostingOptions = Params.BoostingOptions.Get();
    ui32 foldPermutationBlockSize = boostingOptions.PermutationBlockSize;
    if (foldPermutationBlockSize == FoldPermutationBlockSizeNotSet) {
        foldPermutationBlockSize = DefaultFoldPermutationBlockSize(learnData.GetSampleCount());
    }
    if (!isLearnFoldPermuted) {
        foldPermutationBlockSize = learnData.GetSampleCount();
    }
    const auto storeExpApproxes = IsStoreExpApprox(Params.LossFunctionDescription->GetLossFunction());
    const bool hasPairwiseWeights = IsPairwiseError(Params.LossFunctionDescription->GetLossFunction());

    if (IsPlainMode(Params.BoostingOptions->BoostingType)) {
        for (int foldIdx = 0; foldIdx < learningFoldCount; ++foldIdx) {
            LearnProgress.Folds.emplace_back(
                TFold::BuildPlainFold(
                    learnData,
                    CtrsHelper.GetTargetClassifiers(),
                    foldIdx != 0,
                    (Params.SystemOptions->IsSingleHost() ? foldPermutationBlockSize : learnData.GetSampleCount()),
                    LearnProgress.ApproxDimension,
                    storeExpApproxes,
                    hasPairwiseWeights,
                    Rand
                )
            );
        }
    } else {
        for (int foldIdx = 0; foldIdx < learningFoldCount; ++foldIdx) {
            LearnProgress.Folds.emplace_back(
                TFold::BuildDynamicFold(
                    learnData,
                    CtrsHelper.GetTargetClassifiers(),
                    foldIdx != 0,
                    foldPermutationBlockSize,
                    LearnProgress.ApproxDimension,
                    boostingOptions.FoldLenMultiplier,
                    storeExpApproxes,
                    hasPairwiseWeights,
                    Rand
                )
            );
        }
    }

    const bool isAverageFoldPermuted = IsPermutationNeeded(hasTime, hasCtrs, isOrderedBoosting, /*isAveragingFold*/ true);
    LearnProgress.AveragingFold = TFold::BuildPlainFold(
        learnData,
        CtrsHelper.GetTargetClassifiers(),
        isAverageFoldPermuted,
        /*permuteBlockSize=*/ (Params.SystemOptions->IsSingleHost() ? foldPermutationBlockSize : learnData.GetSampleCount()),
        LearnProgress.ApproxDimension,
        storeExpApproxes,
        hasPairwiseWeights,
        Rand
    );

    LearnProgress.AvrgApprox.resize(LearnProgress.ApproxDimension, TVector<double>(learnData.GetSampleCount()));
    if (!learnData.Baseline.empty()) {
        LearnProgress.AvrgApprox = learnData.Baseline;
    }
    ResizeRank2(testDataPtrs.size(), LearnProgress.ApproxDimension, LearnProgress.TestApprox);
    for (size_t testIdx = 0; testIdx < testDataPtrs.size(); ++testIdx) {
        const auto* testData = testDataPtrs[testIdx];
        if (testData == nullptr || testData->GetSampleCount() == 0) {
            continue;
        }
        if (testData->Baseline.empty()) {
            for (auto& approxDim : LearnProgress.TestApprox[testIdx]) {
                approxDim.resize(testData->GetSampleCount());
            }
        } else {
            LearnProgress.TestApprox[testIdx] = testData->Baseline;
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
    } catch(const TCatboostException&) {
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


void LoadFoldApproxesFromScratch(IInputStream* s, TFold* fold) {
    ui64 bodyTailCount;
    ::Load(s, bodyTailCount);
    fold->BodyTailArr.resize(bodyTailCount, TFold::TBodyTail(0, 0, 0, 0, 0));
    for (ui64 i = 0; i < bodyTailCount; ++i) {
        ::Load(s, fold->BodyTailArr[i].Approx);
    }
}


void TLearnProgress::Load(IInputStream* s, bool isFromScratch) {
    ::Load(s, SerializedTrainParams);
    bool enableSaveLoadApprox;
    ::Load(s, enableSaveLoadApprox);
    if (isFromScratch) {
        EnableSaveLoadApprox = enableSaveLoadApprox;
    } else {
        CB_ENSURE(enableSaveLoadApprox == EnableSaveLoadApprox, "Cannot load progress from file");
    }
    if (EnableSaveLoadApprox) {
        ui64 foldCount;
        ::Load(s, foldCount);
        if (isFromScratch) {
            Folds.resize(foldCount);
            for (ui64 i = 0; i < foldCount; ++i) {
                LoadFoldApproxesFromScratch(s, &Folds[i]);
            }
            LoadFoldApproxesFromScratch(s, &AveragingFold);
        } else {
            CB_ENSURE(foldCount == Folds.size(), "Cannot load progress from file");
            for (ui64 i = 0; i < foldCount; ++i) {
                Folds[i].LoadApproxes(s);
            }
            AveragingFold.LoadApproxes(s);
        }
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
