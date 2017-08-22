#include "learn_context.h"
#include "metric.h"

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

void TLearnContext::OutputMeta(int approxDimension) {
    TOFStream meta(Files.MetaFile);
    meta << "name\t" << Params.Name << Endl;
    meta << "iterCount\t" << Params.Iterations << Endl;

    //output log files path relative to trainDirectory
    meta << "learnErrorLog\t" << FilePathForMeta(Params.LearnErrorLog, Files.NamesPrefix) << Endl;
    if (!Files.TestErrorLogFile.empty()) {
        meta << "testErrorLog\t" << FilePathForMeta(Params.TestErrorLog, Files.NamesPrefix) << Endl;
    }
    meta << "timeLeft\t" << FilePathForMeta(Params.TimeLeftLog, Files.NamesPrefix) << Endl;
    auto losses = CreateMetrics(Params, approxDimension);
    for (const auto& loss : losses) {
        meta << "loss\t" << loss->GetDescription() << "\t" << (loss->IsMaxOptimal() ? "max" : "min") << Endl;
    }
}

void TLearnContext::InitData(const TTrainData& data, int approxDimension) {
    for (const auto& ctr : Params.CtrParams.Ctrs) {
        int targetBorderCount = 0;
        if (ctr.CtrType != ECtrType::Counter) {
            if (IsMultiClassError(Params.LossFunction)) {
                targetBorderCount = approxDimension - 1;
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
    const int foldCount = Params.HasTime ? 1 : 3;
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
                approxDimension,
                Params.FoldLenMultiplier,
                Rand));
    }

    LearnProgress.AveragingFold = BuildAveragingFold(data,
                                                     LearnProgress.Model.TargetClassifiers,
                                                     !Params.HasTime,
                                                     approxDimension,
                                                     Rand);

    LearnProgress.AvrgApprox.resize(approxDimension, yvector<double>(sampleCount));
    if (!data.Baseline[0].empty()) {
        for (int dim = 0; dim < approxDimension; ++dim) {
            for (int docId = 0; docId < sampleCount; ++docId) {
                LearnProgress.AvrgApprox[dim][docId] = data.Baseline[docId][dim];
            }
        }
    }
}

namespace {
    class TMD5Output : public IOutputStream {
    public:
        explicit inline TMD5Output(IOutputStream* slave) noexcept
            : Slave_(slave) {
        }

        inline const char* Sum(char* buf) {
            return MD5Sum_.End(buf);
        }

    private:
        void DoWrite(const void* buf, size_t len) override {
            Slave_->Write(buf, len);
            MD5Sum_.Update(buf, len);
        }

        /* Note that default implementation of DoSkip works perfectly fine here as
         * it's implemented in terms of DoRead. */

    private:
        IOutputStream* Slave_;
        MD5 MD5Sum_;
    };
} // namespace

void TLearnContext::SaveProgress() {
    if (!Params.SaveSnapshot) {
        return;
    }
    // actually not so safe, but better then overwriting
    TFsPath myFileName(Files.SnapshotFile);
    TString path = myFileName.Dirname();

    TString tempName = JoinFsPaths(path, CreateGuidAsString()) + ".tmp";
    try {
        {
            TOFStream out(tempName);
            TMD5Output md5out(&out);
            ::SaveMany(&md5out, Rand, LearnProgress);
            char md5buf[33];
            MATRIXNET_INFO_LOG << "Saved progress (md5sum: " << md5out.Sum(md5buf) << " )" << Endl;
        }
        NFs::Rename(tempName, Files.SnapshotFile);
    } catch (...) {
        MATRIXNET_WARNING_LOG << "Can't save progress to file, got exception: " << CurrentExceptionMessage() << Endl;
        NFs::Remove(tempName);
    }
}

void TLearnContext::LoadProgress() {
    if (!Params.SaveSnapshot || !NFs::Exists(Files.SnapshotFile)) {
        return;
    }
    try {
        TIFStream in(Files.SnapshotFile);
        TLearnProgress LearnProgressRestored = LearnProgress; // use progress copy to avoid partial deserialization of corrupted progress file
        ::LoadMany(&in, Rand, LearnProgressRestored); // fail here does nothing with real LearnProgress
        LearnProgress = std::move(LearnProgressRestored);
        LearnProgress.Model.ModelInfo["params"] = ToString(ResultingParams); // substitute real
        Profile.SetInitIterations(LearnProgress.Model.TreeStruct.ysize());
    } catch (...) {
        MATRIXNET_WARNING_LOG << "Can't load progress from file: " << Files.SnapshotFile << " exception: " << CurrentExceptionMessage() << Endl;
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
    CB_ENSURE(foldCount == Folds.size());
    for (ui64 i = 0; i < foldCount; ++i) {
        Folds[i].LoadApproxes(s);
    }
    AveragingFold.LoadApproxes(s);
    ::LoadMany(s, AvrgApprox, Model, LearnErrorsHistory, TestErrorsHistory);
}
