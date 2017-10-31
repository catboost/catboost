#include "error_functions.h"
#include "learn_context.h"
#include "metric.h"

#include <catboost/libs/helpers/progress_helper.h>
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

void TOutputFiles::InitializeFiles(const TFitParams& params, const TString& namesPrefix) {
    if (!params.AllowWritingFiles) {
        Y_ASSERT(TimeLeftLogFile.empty());
        Y_ASSERT(LearnErrorLogFile.empty());
        Y_ASSERT(TestErrorLogFile.empty());
        Y_ASSERT(MetaFile.empty());
        Y_ASSERT(SnapshotFile.empty());
        return;
    }

    TFsPath trainDirPath(params.TrainDir);
    if (!params.TrainDir.empty() && !trainDirPath.Exists()) {
        trainDirPath.MkDir();
    }
    NamesPrefix = namesPrefix;
    CB_ENSURE(!params.TimeLeftLog.empty(), "empty time_left filename");
    TimeLeftLogFile = TOutputFiles::AlignFilePath(params.TrainDir, params.TimeLeftLog, NamesPrefix);

    CB_ENSURE(!params.LearnErrorLog.empty(), "empty learn_error filename");
    LearnErrorLogFile = TOutputFiles::AlignFilePath(params.TrainDir, params.LearnErrorLog, NamesPrefix);
    if (params.TestErrorLog) {
        TestErrorLogFile = TOutputFiles::AlignFilePath(params.TrainDir, params.TestErrorLog, NamesPrefix);
    }
    if (params.SaveSnapshot) {
        SnapshotFile = TOutputFiles::AlignFilePath(params.TrainDir, params.SnapshotFileName, NamesPrefix);
    }
    CB_ENSURE(!params.MetaFileName.empty(), "empty meta filename");
    MetaFile = TOutputFiles::AlignFilePath(params.TrainDir, params.MetaFileName, NamesPrefix);
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
    meta << "name\t" << Params.Name << Endl;
    meta << "iterCount\t" << Params.Iterations << Endl;

    //output log files path relative to trainDirectory
    meta << "learnErrorLog\t" << FilePathForMeta(Params.LearnErrorLog, Files.NamesPrefix) << Endl;
    if (!Files.TestErrorLogFile.empty()) {
        meta << "testErrorLog\t" << FilePathForMeta(Params.TestErrorLog, Files.NamesPrefix) << Endl;
    }
    meta << "timeLeft\t" << FilePathForMeta(Params.TimeLeftLog, Files.NamesPrefix) << Endl;
    auto losses = CreateMetrics(Params, LearnProgress.Model.ApproxDimension);
    for (const auto& loss : losses) {
        meta << "loss\t" << loss->GetDescription() << "\t" << (loss->IsMaxOptimal() ? "max" : "min") << Endl;
    }
}

void TLearnContext::InitData(const TTrainData& data) {
    for (const auto& ctr : Params.CtrParams.Ctrs) {
        int targetBorderCount = 0;
        if (ctr.CtrType != ECtrType::Counter) {
            if (IsMultiClassError(Params.LossFunction)) {
                targetBorderCount = LearnProgress.Model.ApproxDimension - 1;
            } else {
                targetBorderCount = ctr.TargetBorderCount;
            }
        }

        LearnProgress.Model.TargetClassifiers.emplace_back(
            BuildTargetClassifier(
                data.Target,
                data.LearnSampleCount,
                Params.LossFunction,
                Params.ObjectiveDescriptor,
                targetBorderCount,
                ctr.TargetBorderType));
    }

    const auto sampleCount = data.GetSampleCount();
    const int foldCount = Params.HasTime || IsQuerywiseError(Params.LossFunction) ? 1 : 3;
    LearnProgress.Folds.reserve(foldCount);

    int foldPermutationBlockSize = Params.FoldPermutationBlockSize;
    if (foldPermutationBlockSize == ParameterNotSet) {
        foldPermutationBlockSize = Min(256, data.LearnSampleCount / 1000 + 1);
    }

    for (int foldIdx = 0; foldIdx < foldCount; ++foldIdx) {
        LearnProgress.Folds.emplace_back(
            BuildLearnFold(
                data,
                LearnProgress.Model.TargetClassifiers,
                foldIdx != 0,
                foldPermutationBlockSize,
                LearnProgress.Model.ApproxDimension,
                Params.FoldLenMultiplier,
                Params.StoreExpApprox,
                IsQuerywiseError(Params.LossFunction),
                Rand));
    }

    LearnProgress.AveragingFold = BuildAveragingFold(data,
                                                     LearnProgress.Model.TargetClassifiers,
                                                     !(Params.HasTime || IsQuerywiseError(Params.LossFunction)),
                                                     LearnProgress.Model.ApproxDimension,
                                                     Params.StoreExpApprox,
                                                     Rand);

    LearnProgress.AvrgApprox.resize(LearnProgress.Model.ApproxDimension, yvector<double>(sampleCount));
    if (!data.Baseline.empty()) {
        LearnProgress.AvrgApprox = data.Baseline;
    }
}

//TODO: to enum
static const TString CpuProgressLabel = "CPU";

void TLearnContext::SaveProgress() {
    if (!Params.SaveSnapshot) {
        return;
    }
    TProgressHelper(CpuProgressLabel).Write(Files.SnapshotFile, [&](IOutputStream* out) {
        ::SaveMany(out, Rand, LearnProgress, Profile.DumpProfileInfo());
    });
}

bool TLearnContext::TryLoadProgress() {
    if (!Params.SaveSnapshot || !NFs::Exists(Files.SnapshotFile)) {
        return false;
    }
    try {
        TProgressHelper(CpuProgressLabel).CheckedLoad(Files.SnapshotFile, [&](TIFStream* in)
        {
            TLearnProgress LearnProgressRestored = LearnProgress; // use progress copy to avoid partial deserialization of corrupted progress file
            TProfileInfoData ProfileRestored;
            ::LoadMany(in, Rand, LearnProgressRestored, ProfileRestored); // fail here does nothing with real LearnProgress
            LearnProgress = std::move(LearnProgressRestored);
            Profile.InitProfileInfo(std::move(ProfileRestored));
            LearnProgress.Model.ModelInfo["params"] = ToString(ResultingParams); // substitute real
            MATRIXNET_INFO_LOG << "Loaded progress file containing " <<  LearnProgress.Model.TreeStruct.size() << " trees" << Endl;
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
    ::SaveMany(s, AvrgApprox, Model, LearnErrorsHistory, TestErrorsHistory);
}

void TLearnProgress::Load(IInputStream* s) {
    ui64 foldCount;
    ::Load(s, foldCount);
    CB_ENSURE(foldCount == Folds.size(), "Cannot load progress from file");
    for (ui64 i = 0; i < foldCount; ++i) {
        Folds[i].LoadApproxes(s);
    }
    AveragingFold.LoadApproxes(s);
    ::LoadMany(s, AvrgApprox, Model, LearnErrorsHistory, TestErrorsHistory);
}
