#pragma once

#include "option.h"
#include "json_helper.h"
#include <util/system/types.h>
#include <util/folder/path.h>

namespace NCatboostOptions {

    class TOutputFilesOptions {
    public:
        explicit TOutputFilesOptions(ETaskType taskType)
            : ResultModelPath("result_model_file", "model.bin")
            , UseBestModel("use_best_model", false)
            , TrainDir("train_dir", "catboost_info")
            , Name("name", "experiment")
            , MetaFile("meta", "meta.tsv")
            , JsonLogPath("json_log", "catboost_training.json")
            , ProfileLogPath("profile_log", "catboost_profile.log")
            , LearnErrorLogPath("learn_error_log", "learn_error.tsv")
            , ModelFormats("model_format", {EModelType::CatboostBinary})
            , TestErrorLogPath("test_error_log", "test_error.tsv")
            , TimeLeftLog("time_left_log", "time_left.tsv")
            , SnapshotPath("snapshot_file", "experiment.cbsnapshot")
            , SaveSnapshotFlag("save_snapshot", false)
            , AllowWriteFilesFlag("allow_writing_files", true)
            , SnapshotSaveIntervalSeconds("snapshot_save_interval_secs", 10 * 60, taskType)
            , MetricPeriod("metric_period", 1)
            , PredictionTypes("prediction_type", {EPredictionType::RawFormulaVal}, taskType)
            , EvalFileName("eval_file_name", "", taskType)
            , OutputColumns("output_columns", {"DocId", "RawFormulaVal", "Label"}, taskType)
            , FstrRegularFileName("fstr_regular_file", "", taskType)
            , FstrInternalFileName("fstr_internal_file", "", taskType)

        {
            SnapshotSaveIntervalSeconds.ChangeLoadUnimplementedPolicy(ELoadUnimplementedPolicy::SkipWithWarning);
        }

        TOption<TString> ResultModelPath;
        TOption<bool> UseBestModel;

        const TString& GetTrainDir() const {
            return TrainDir.Get();
        }

        TString CreateLearnErrorLogFullPath() const {
            return GetFullPath(LearnErrorLogPath.Get());
        }

        TString CreateTestErrorLogFullPath() const {
            return GetFullPath(TestErrorLogPath.Get());
        }

        TString CreateTimeLeftLogFullPath() const {
            return GetFullPath(TimeLeftLog.Get());
        }

        TString CreateMetaFileFullPath() const {
            return GetFullPath(MetaFile.Get());
        }

        TString CreateJsonLogFullPath() const {
            return GetFullPath(JsonLogPath.Get());
        }

        TString CreateProfileLogFullPath() const {
            return GetFullPath(ProfileLogPath.Get());
        }

        TString CreateResultModelFullPath() const {
            return GetFullPath(ResultModelPath.Get());
        }

        TString CreateSnapshotFullPath() const {
            return GetFullPath(SnapshotPath.Get());
        }
        //local
        const TString& GetLearnErrorFilename() const {
            return LearnErrorLogPath.Get();
        }

        const TString& GetTestErrorFilename() const {
            return TestErrorLogPath.Get();
        }

        const TString& GetTimeLeftLogFilename() const {
            return TimeLeftLog.Get();
        }

        const TString& GetMetaFileFilename() const {
            return MetaFile.Get();
        }

        const TVector<EModelType>& GetModelFormats() const {
            return ModelFormats.Get();
        }

        const TString& GetJsonLogFilename() const {
            return JsonLogPath.Get();
        }

        const TString& GetProfileLogFilename() const {
            return ProfileLogPath.Get();
        }

        const TString& GetResultModelFilename() const {
            return ResultModelPath.Get();
        }

        const TString& GetSnapshotFilename() const {
            return SnapshotPath.Get();
        }

        bool ShrinkModelToBestIteration() const {
            return UseBestModel.Get();
        }

        const TString& GetName() const {
            return Name.Get();
        }

        const TVector<EPredictionType>& GetPredictionTypes() const {
            return PredictionTypes.Get();
        }

        const TVector<TString>& GetOutputColumns() const {
            return OutputColumns.Get();
        }

        bool AllowWriteFiles() const {
            return AllowWriteFilesFlag.Get();
        }

        bool SaveSnapshot() const {
            return SaveSnapshotFlag.Get();
        }

        ui64 GetSnapshotSaveInterval() const {
            return SnapshotSaveIntervalSeconds.Get();
        }

        int GetMetricPeriod() const {
            return MetricPeriod.Get();
        }

        TString CreateFstrRegularFullPath() const {
            return GetFullPath(FstrRegularFileName.Get());
        }

        TString CreateFstrIternalFullPath() const {
            return GetFullPath(FstrInternalFileName.Get());
        }

        TString CreateEvalFullPath() const {
            return GetFullPath(EvalFileName.Get());
        }

        bool operator==(const TOutputFilesOptions& rhs) const {
            return std::tie(TrainDir, Name, MetaFile, JsonLogPath, ProfileLogPath, LearnErrorLogPath, TestErrorLogPath, TimeLeftLog, ResultModelPath,
                            SnapshotPath, ModelFormats, SaveSnapshotFlag, AllowWriteFilesFlag, UseBestModel, SnapshotSaveIntervalSeconds,
                            EvalFileName, FstrRegularFileName, FstrInternalFileName) ==
                   std::tie(rhs.TrainDir, rhs.Name, rhs.MetaFile, rhs.JsonLogPath, rhs.ProfileLogPath, rhs.LearnErrorLogPath, rhs.TestErrorLogPath,
                            rhs.TimeLeftLog, rhs.ResultModelPath, rhs.SnapshotPath, rhs.ModelFormats, rhs.SaveSnapshotFlag,
                            rhs.AllowWriteFilesFlag, rhs.UseBestModel, rhs.SnapshotSaveIntervalSeconds,
                            rhs.EvalFileName, rhs.FstrRegularFileName, rhs.FstrInternalFileName);
        }

        bool operator!=(const TOutputFilesOptions& rhs) const {
            return !(rhs == *this);
        }

        void Load(const NJson::TJsonValue& options) {
            CheckedLoad(options,
                        &TrainDir, &Name, &MetaFile, &JsonLogPath, &ProfileLogPath, &LearnErrorLogPath, &TestErrorLogPath, &TimeLeftLog,
                        &ResultModelPath,
                        &SnapshotPath, &ModelFormats, &SaveSnapshotFlag, &AllowWriteFilesFlag, &UseBestModel, &SnapshotSaveIntervalSeconds,
                        &EvalFileName, &OutputColumns, &FstrRegularFileName, &FstrInternalFileName, &MetricPeriod, &PredictionTypes);
            Validate();
        }

        void Save(NJson::TJsonValue* options) const {
            SaveFields(options,
                       TrainDir, Name, MetaFile, JsonLogPath, ProfileLogPath, LearnErrorLogPath, TestErrorLogPath, TimeLeftLog, ResultModelPath,
                       SnapshotPath, ModelFormats, SaveSnapshotFlag, AllowWriteFilesFlag, UseBestModel, SnapshotSaveIntervalSeconds,
                       EvalFileName, OutputColumns, FstrRegularFileName, FstrInternalFileName, MetricPeriod, PredictionTypes);
        }

        void Validate() const {
            if (!AllowWriteFilesFlag.Get()) {
                CB_ENSURE(!SaveSnapshotFlag.Get(),
                          "allow_writing_files is set to False, and save_snapshot is set to True.");
            }
        }


    private:
        TString GetFullPath(const TString& fileName) const {
            if (fileName.empty()) {
                return "";
            }

            TFsPath filePath(fileName);
            const TString& trainDirStr = TrainDir.Get();
            if (trainDirStr.Empty() || filePath.IsAbsolute()) {
                return fileName;
            } else {
                return JoinFsPaths(trainDirStr, filePath);
            }
        }

    private:
        TOption<TString> TrainDir;
        TOption<TString> Name;
        TOption<TString> MetaFile;
        TOption<TString> JsonLogPath;
        TOption<TString> ProfileLogPath;
        TOption<TString> LearnErrorLogPath;
        TOption<TVector<EModelType>> ModelFormats;
        TOption<TString> TestErrorLogPath;
        TOption<TString> TimeLeftLog;
        TOption<TString> SnapshotPath;
        TOption<bool> SaveSnapshotFlag;
        TOption<bool> AllowWriteFilesFlag;

        TGpuOnlyOption<ui64> SnapshotSaveIntervalSeconds;
        TOption<int> MetricPeriod;

        TCpuOnlyOption<TVector<EPredictionType>> PredictionTypes;
        TCpuOnlyOption<TString> EvalFileName;
        TCpuOnlyOption<TVector<TString>> OutputColumns;
        TCpuOnlyOption<TString> FstrRegularFileName;
        TCpuOnlyOption<TString> FstrInternalFileName;
    };

}
