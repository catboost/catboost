#include "learn_context.h"
#include "error_functions.h"

#include <catboost/libs/distributed/master.h>
#include <catboost/libs/helpers/progress_helper.h>
#include <catboost/libs/options/defaults_helper.h>

#include <library/digest/md5/md5.h>

#include <util/generic/guid.h>
#include <util/folder/path.h>
#include <util/system/fs.h>
#include <util/stream/file.h>

TString TOutputFiles::AlignFilePath(const TString& baseDir, const TString& fileName, const TString& namePrefix) {
    TFsPath filePath(fileName);
    if (filePath.IsAbsolute()) {
        return JoinFsPaths(filePath.Dirname(), namePrefix + filePath.Basename());
    }
    return JoinFsPaths(baseDir, namePrefix + fileName);
}


void TOutputFiles::InitializeFiles(const NCatboostOptions::TOutputFilesOptions& params, const TString& namesPrefix) {
    if (!params.AllowWriteFiles()) {
        Y_ASSERT(TimeLeftLogFile.empty());
        Y_ASSERT(LearnErrorLogFile.empty());
        Y_ASSERT(TestErrorLogFile.empty());
        Y_ASSERT(MetaFile.empty());
        Y_ASSERT(SnapshotFile.empty());
        return;
    }

    const auto& trainDir = params.GetTrainDir();
    TFsPath trainDirPath(trainDir);
    if (!trainDir.empty() && !trainDirPath.Exists()) {
        trainDirPath.MkDir();
    }
    NamesPrefix = namesPrefix;
    CB_ENSURE(!params.GetTimeLeftLogFilename().empty(), "empty time_left filename");
    TimeLeftLogFile = TOutputFiles::AlignFilePath(trainDir, params.GetTimeLeftLogFilename(), NamesPrefix);

    CB_ENSURE(!params.GetLearnErrorFilename().empty(), "empty learn_error filename");
    LearnErrorLogFile = TOutputFiles::AlignFilePath(trainDir, params.GetLearnErrorFilename(), NamesPrefix);
    if (params.GetTestErrorFilename()) {
        TestErrorLogFile = TOutputFiles::AlignFilePath(trainDir, params.GetTestErrorFilename(), NamesPrefix);
    }
    if (params.SaveSnapshot()) {
        SnapshotFile = TOutputFiles::AlignFilePath(trainDir, params.GetSnapshotFilename(), NamesPrefix);
    }
    const TString& metaFileFilename = params.GetMetaFileFilename();
    CB_ENSURE(!metaFileFilename.empty(), "empty meta filename");
    MetaFile = TOutputFiles::AlignFilePath(trainDir, metaFileFilename, NamesPrefix);

    const TString& jsonLogFilename = params.GetJsonLogFilename();
    CB_ENSURE(!jsonLogFilename.empty(), "empty json_log filename");
    JsonLogFile = TOutputFiles::AlignFilePath(trainDir, jsonLogFilename, "");

    const TString& profileLogFilename = params.GetProfileLogFilename();
    CB_ENSURE(!profileLogFilename.empty(), "empty profile_log filename");
    ProfileLogFile = TOutputFiles::AlignFilePath(trainDir, profileLogFilename, "");
}

TString FilePathForMeta(const TString& filename, const TString& namePrefix) {
    TFsPath filePath(filename);
    if (filePath.IsAbsolute()) {
        return JoinFsPaths(filePath.Dirname(), namePrefix + filePath.Basename());
    }
    return JoinFsPaths(namePrefix + filename);
}


TLearnContext::~TLearnContext() {
    if (Params.SystemOptions->IsMaster()) {
        FinalizeMaster(this);
    }
}

void TLearnContext::OutputMeta() {
    if (Files.MetaFile.empty()) {
        return;
    }

    TOFStream meta(Files.MetaFile);
    meta << "name\t" << OutputOptions.GetName() << Endl;
    meta << "iterCount\t" << Params.BoostingOptions->IterationCount << Endl;

    //output log files path relative to trainDirectory
    meta << "learnErrorLog\t" << FilePathForMeta(OutputOptions.GetLearnErrorFilename(), Files.NamesPrefix) << Endl;
    if (!Files.TestErrorLogFile.empty()) {
        meta << "testErrorLog\t" << FilePathForMeta(OutputOptions.GetTestErrorFilename(), Files.NamesPrefix) << Endl;
    }
    meta << "timeLeft\t" << FilePathForMeta(OutputOptions.GetTimeLeftLogFilename(), Files.NamesPrefix) << Endl;
    auto losses = CreateMetrics(
        Params.LossFunctionDescription,
        Params.MetricOptions,
        EvalMetricDescriptor,
        LearnProgress.ApproxDimension
    );

    for (const auto& loss : losses) {
        EMetricBestValue bestValueType;
        float bestValue;
        loss->GetBestValue(&bestValueType, &bestValue);
        TString bestValueString;
        if (bestValueType == EMetricBestValue::Max) {
            bestValueString = "max";
        } else {
            bestValueString = "min";
        }
        meta << "loss\t" << loss->GetDescription() << "\t" << bestValueString << Endl;
    }
}

static bool IsCategoricalFeaturesEmpty(const TAllFeatures& allFeatures) {
    for (int i = 0; i < allFeatures.CatFeaturesRemapped.ysize(); ++i) {
        if (!allFeatures.IsOneHot[i] && !allFeatures.CatFeaturesRemapped[i].empty()) {
            return false;
        }
    }
    return true;
}

void TLearnContext::InitContext(const TDataset& learnData, const TDataset* testData) {
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
                             ObjectiveDescriptor);

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
                BuildPlainFold(
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
                BuildDynamicFold(
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

    LearnProgress.AveragingFold = BuildPlainFold(
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
    if (testData) {
        LearnProgress.TestApprox.resize(LearnProgress.ApproxDimension, TVector<double>(testData->GetSampleCount()));
        if (!testData->Baseline.empty()) {
            LearnProgress.TestApprox = testData->Baseline;
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
        "flat_params"
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
            CB_ENSURE(IsParamsCompatible(LearnProgressRestored.SerializedTrainParams, LearnProgress.SerializedTrainParams), "Saved model's Params are different from current model's params");
            CB_ENSURE(LearnProgressRestored.PoolCheckSum == LearnProgress.PoolCheckSum, "Current pool differs from the original pool");
            LearnProgress = std::move(LearnProgressRestored);
            Profile.InitProfileInfo(std::move(ProfileRestored));
            LearnProgress.SerializedTrainParams = ToString(Params); // substitute real
            MATRIXNET_INFO_LOG << "Loaded progress file containing " <<  LearnProgress.TreeStruct.size() << " trees" << Endl;
        });
        return true;
    } catch (...) {
        MATRIXNET_WARNING_LOG << "Can't load progress from file: " << Files.SnapshotFile << " exception: " << CurrentExceptionMessage() << Endl;
        return false;
    }
}

void TLearnProgress::Save(IOutputStream* s) const {
    ui64 foldCount = Folds.size();
    ::Save(s, foldCount);
    for (ui64 i = 0; i < foldCount; ++i) {
        Folds[i].SaveApproxes(s);
    }
    AveragingFold.SaveApproxes(s);
    ::SaveMany(s,
               AvrgApprox,
               TestApprox,
               CatFeatures,
               FloatFeatures,
               ApproxDimension,
               SerializedTrainParams,
               TreeStruct,
               TreeStats,
               LeafValues,
               LearnErrorsHistory,
               TestErrorsHistory,
               TimeHistory,
               UsedCtrSplits,
               PoolCheckSum);
}

void TLearnProgress::Load(IInputStream* s) {
    ui64 foldCount;
    ::Load(s, foldCount);
    CB_ENSURE(foldCount == Folds.size(), "Cannot load progress from file");
    for (ui64 i = 0; i < foldCount; ++i) {
        Folds[i].LoadApproxes(s);
    }
    AveragingFold.LoadApproxes(s);
    ::LoadMany(s, AvrgApprox,
               TestApprox,
               CatFeatures,
               FloatFeatures,
               ApproxDimension,
               SerializedTrainParams,
               TreeStruct,
               TreeStats,
               LeafValues,
               LearnErrorsHistory,
               TestErrorsHistory,
               TimeHistory,
               UsedCtrSplits,
               PoolCheckSum);
}

NJson::TJsonValue GetJsonMeta(
    int iterationCount,
    const TString& optionalExperimentName,
    const TVector<const IMetric*>& metrics,
    const TVector<TString>& learnSetNames,
    const TVector<TString>& testSetNames,
    ELaunchMode launchMode
) {
    NJson::TJsonValue meta;
    meta["iteration_count"] = iterationCount;
    meta["name"] = optionalExperimentName;

    meta.InsertValue("learn_sets", NJson::JSON_ARRAY);
    for (auto& name : learnSetNames) {
        meta["learn_sets"].AppendValue(name);
    }

    meta.InsertValue("test_sets", NJson::JSON_ARRAY);
    for (auto& name : testSetNames) {
        meta["test_sets"].AppendValue(name);
    }

    meta.InsertValue("learn_metrics", NJson::JSON_ARRAY);
    meta.InsertValue("test_metrics", NJson::JSON_ARRAY);
    for (const auto& loss : metrics) {
        NJson::TJsonValue metricJson;
        metricJson.InsertValue("name", loss->GetDescription());

        EMetricBestValue bestValueType;
        float bestValue;
        loss->GetBestValue(&bestValueType, &bestValue);
        TString bestValueString;
        if (bestValueType != EMetricBestValue::FixedValue) {
            metricJson.InsertValue("best_value", ToString(bestValueType));
        } else {
            metricJson.InsertValue("best_value", bestValue);
        }

        if (!learnSetNames.empty()) {
            meta["learn_metrics"].AppendValue(metricJson);
        }
        if (!testSetNames.empty()) {
            meta["test_metrics"].AppendValue(metricJson);
        }
    }

    meta.InsertValue("launch_mode", ToString<ELaunchMode>(launchMode));
    return meta;
}
