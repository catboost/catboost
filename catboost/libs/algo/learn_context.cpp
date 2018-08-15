#include "learn_context.h"
#include "error_functions.h"

#include <catboost/libs/distributed/master.h>
#include <catboost/libs/helpers/progress_helper.h>
#include <catboost/libs/options/defaults_helper.h>

#include <library/digest/crc32c/crc32c.h>
#include <library/digest/md5/md5.h>

#include <util/generic/guid.h>
#include <util/folder/path.h>
#include <util/system/fs.h>
#include <util/stream/file.h>




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

static bool IsCategoricalFeaturesEmpty(const TAllFeatures& allFeatures) {
    for (int i = 0; i < allFeatures.CatFeaturesRemapped.ysize(); ++i) {
        if (!allFeatures.IsOneHot[i] && !allFeatures.CatFeaturesRemapped[i].empty()) {
            return false;
        }
    }
    return true;
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

void TLearnContext::InitContext(const TDataset& learnData, const TDatasetPtrs& testDataPtrs) {
    LearnProgress.PoolCheckSum = CalcFeaturesCheckSum(learnData.AllFeatures);
    for (const TDataset* testData : testDataPtrs) {
        LearnProgress.PoolCheckSum += CalcFeaturesCheckSum(testData->AllFeatures);
    }

    auto lossFunction = Params.LossFunctionDescription->GetLossFunction();
    int foldCount = Max<ui32>(Params.BoostingOptions->PermutationCount - 1, 1);
    const bool noCtrs = IsCategoricalFeaturesEmpty(learnData.AllFeatures);
    if (Params.BoostingOptions->BoostingType == EBoostingType::Plain && noCtrs) {
        foldCount = 1;
    }
    LearnProgress.Folds.reserve(foldCount);
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
    if (IsPlainMode(Params.BoostingOptions->BoostingType) && noCtrs) {
        foldPermutationBlockSize = learnData.GetSampleCount();
    }
    const auto storeExpApproxes = IsStoreExpApprox(Params.LossFunctionDescription->GetLossFunction());
    const bool hasPairwiseWeights = IsPairwiseError(Params.LossFunctionDescription->GetLossFunction());

    if (IsPlainMode(Params.BoostingOptions->BoostingType)) {
        for (int foldIdx = 0; foldIdx < foldCount; ++foldIdx) {
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
        for (int foldIdx = 0; foldIdx < foldCount; ++foldIdx) {
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

    LearnProgress.AveragingFold = TFold::BuildPlainFold(
        learnData,
        CtrsHelper.GetTargetClassifiers(),
        !(Params.DataProcessingOptions->HasTimeFlag),
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
}

void TLearnContext::SaveProgress() {
    if (!OutputOptions.SaveSnapshot()) {
        return;
    }
    TProgressHelper(ToString(ETaskType::CPU)).Write(Files.SnapshotFile, [&](IOutputStream* out) {
        ::SaveMany(out, Rand, LearnProgress, Profile.DumpProfileInfo());
    });
}

static bool IsParamsCompatible(const TString& firstSerializedParams, const TString& secondSerializedParams) {
    //TODO:(noxoomo, nikitxskv): i don't think this way of checking compatible is good. We should parse params and comprare fields that are essential, not all
    const TVector<TString> paramsToIgnore = {
        "system_options",
        "flat_params",
        "metadata"
    };
    const TVector<TString> boostingParamsToIgnore = {
        "iterations",
        "learning_rate",
    };
    NJson::TJsonValue firstParams, secondParams;
    ReadJsonTree(firstSerializedParams, &firstParams);
    ReadJsonTree(secondSerializedParams, &secondParams);

    for (const auto& paramName : paramsToIgnore) {
        firstParams.EraseValue(paramName);
        secondParams.EraseValue(paramName);
    }
    for (const auto& paramName : boostingParamsToIgnore) {
        firstParams["boosting_options"].EraseValue(paramName);
        secondParams["boosting_options"].EraseValue(paramName);
    }
    return firstParams == secondParams;
}

bool TLearnContext::TryLoadProgress() {
    if (!OutputOptions.SaveSnapshot() || !NFs::Exists(Files.SnapshotFile)) {
        return false;
    }
    try {
        TProgressHelper(ToString(ETaskType::CPU)).CheckedLoad(Files.SnapshotFile, [&](TIFStream* in)
        {
            TLearnProgress LearnProgressRestored = LearnProgress; // use progress copy to avoid partial deserialization of corrupted progress file
            TProfileInfoData ProfileRestored;
            ::LoadMany(in, Rand, LearnProgressRestored, ProfileRestored); // fail here does nothing with real LearnProgress
            const bool paramsCompatible = IsParamsCompatible(LearnProgressRestored.SerializedTrainParams, LearnProgress.SerializedTrainParams);
            CB_ENSURE(paramsCompatible, "Saved model's params are different from current model's params");
            const bool poolCompatible = (LearnProgressRestored.PoolCheckSum == LearnProgress.PoolCheckSum);
            CB_ENSURE(poolCompatible, "Current pool differs from the original pool");
            LearnProgress = std::move(LearnProgressRestored);
            Profile.InitProfileInfo(std::move(ProfileRestored));
            LearnProgress.SerializedTrainParams = ToString(Params); // substitute real
            MATRIXNET_INFO_LOG << "Loaded progress file containing " <<  LearnProgress.TreeStruct.size() << " trees" << Endl;
        });
        return true;
    } catch(const TCatboostException&) {
        throw;
    } catch (...) {
        MATRIXNET_WARNING_LOG << "Can't load progress from snapshot file: " << Files.SnapshotFile << " exception: "
                            << CurrentExceptionMessage() << Endl;
        return false;
    }
}

void TLearnProgress::Save(IOutputStream* s) const {
    ::Save(s, SerializedTrainParams);
    ui64 foldCount = Folds.size();
    ::Save(s, foldCount);
    for (ui64 i = 0; i < foldCount; ++i) {
        Folds[i].SaveApproxes(s);
    }
    AveragingFold.SaveApproxes(s);
    ::SaveMany(s,
               AvrgApprox,
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
    ui64 foldCount;
    ::Load(s, foldCount);
    CB_ENSURE(foldCount == Folds.size(), "Cannot load progress from file");
    for (ui64 i = 0; i < foldCount; ++i) {
        Folds[i].LoadApproxes(s);
    }
    AveragingFold.LoadApproxes(s);
    ::LoadMany(s, AvrgApprox,
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

