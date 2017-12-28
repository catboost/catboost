#include "learn_context.h"
#include "error_functions.h"

#include <catboost/libs/helpers/progress_helper.h>

#include <library/digest/md5/md5.h>

#include <util/generic/guid.h>
#include <util/folder/path.h>
#include <util/system/fs.h>
#include <util/stream/file.h>
#include <catboost/libs/options/defaults_helper.h>

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
}

TString FilePathForMeta(const TString& filename, const TString& namePrefix) {
    TFsPath filePath(filename);
    if (filePath.IsAbsolute()) {
        return JoinFsPaths(filePath.Dirname(), namePrefix + filePath.Basename());
    }
    return JoinFsPaths(namePrefix + filename);
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
    auto losses = CreateMetrics(Params.MetricOptions->EvalMetric, EvalMetricDescriptor,
                                Params.MetricOptions->CustomMetrics,
                                LearnProgress.ApproxDimension);

    for (const auto& loss : losses) {
        meta << "loss\t" << loss->GetDescription() << "\t" << (loss->IsMaxOptimal() ? "max" : "min") << Endl;
    }
}


void TLearnContext::InitData(const TTrainData& data) {
    auto lossFunction = Params.LossFunctionDescription->GetLossFunction();
    const auto sampleCount = data.GetSampleCount();
    const auto isQuerywiseError = IsQuerywiseError(lossFunction);
    const int foldCount = Max<ui32>(Params.BoostingOptions->PermutationCount - 1, 1);
    LearnProgress.Folds.reserve(foldCount);
    UpdateCtrsTargetBordersOption(lossFunction, LearnProgress.ApproxDimension, &Params.CatFeatureParams.Get());

    CtrsHelper.Init(Params.CatFeatureParams,
                    Layout,
                    data.Target,
                    static_cast<ui32>(data.LearnSampleCount),
                    lossFunction,
                    ObjectiveDescriptor);

    //Todo(noxoomo): check and init
    const auto& boostingOptions = Params.BoostingOptions.Get();
    ui32 foldPermutationBlockSize = boostingOptions.PermutationBlockSize;
    if (foldPermutationBlockSize == FoldPermutationBlockSizeNotSet) {
        foldPermutationBlockSize = DefaultFoldPermutationBlockSize(data.LearnSampleCount);
    }
    const auto storeExpApproxes = IsStoreExpApprox(Params);

    for (int foldIdx = 0; foldIdx < foldCount; ++foldIdx) {
        LearnProgress.Folds.emplace_back(
            BuildLearnFold(
                data,
                CtrsHelper.GetTargetClassifiers(),
                foldIdx != 0,
                foldPermutationBlockSize,
                LearnProgress.ApproxDimension,
                boostingOptions.FoldLenMultiplier,
                storeExpApproxes,
                isQuerywiseError,
                Rand));
    }

    LearnProgress.AveragingFold = BuildAveragingFold(
        data,
        CtrsHelper.GetTargetClassifiers(),
        !(Params.DataProcessingOptions->HasTimeFlag|| isQuerywiseError),
        LearnProgress.ApproxDimension,
        storeExpApproxes,
        isQuerywiseError,
        Rand
    );

    LearnProgress.AvrgApprox.resize(LearnProgress.ApproxDimension, TVector<double>(sampleCount));
    if (!data.Baseline.empty()) {
        LearnProgress.AvrgApprox = data.Baseline;
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
    NJson::TJsonValue firstParams, secondParams;
    ReadJsonTree(firstSerializedParams, &firstParams);
    ReadJsonTree(secondSerializedParams, &secondParams);

    firstParams = firstParams["flat_params"];
    secondParams = secondParams["flat_params"];

    firstParams.EraseValue("iterations");
    firstParams.EraseValue("learning_rate");
    secondParams.EraseValue("iterations");
    secondParams.EraseValue("learning_rate");
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
               CatFeatures,
               FloatFeatures,
               ApproxDimension,
               SerializedTrainParams,
               TreeStruct,
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
               CatFeatures,
               FloatFeatures,
               ApproxDimension,
               SerializedTrainParams,
               TreeStruct,
               LeafValues,
               LearnErrorsHistory,
               TestErrorsHistory,
               TimeHistory,
               UsedCtrSplits,
               PoolCheckSum);
}
