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
            , FinalCtrComputationMode("final_ctr_computation_mode", EFinalCtrComputationMode::Default)
            , EvalFileName("eval_file_name", "")
            , FstrRegularFileName("fstr_regular_file", "")
            , FstrInternalFileName("fstr_internal_file", "")
            , TrainingOptionsFileName("training_options_file", "")
            , SnapshotSaveIntervalSeconds("snapshot_save_interval_secs", 10 * 60, taskType)
            , OutputBordersFileName("output_borders", "", taskType)
            , VerbosePeriod("verbose", 1)
            , MetricPeriod("metric_period", 1)
            , PredictionTypes("prediction_type", {EPredictionType::RawFormulaVal}, taskType)
            , OutputColumns("output_columns", {"DocId", "RawFormulaVal", "Label"}, taskType) {
            SnapshotSaveIntervalSeconds.ChangeLoadUnimplementedPolicy(ELoadUnimplementedPolicy::SkipWithWarning);
            OutputBordersFileName.ChangeLoadUnimplementedPolicy(ELoadUnimplementedPolicy::SkipWithWarning);
        }

        TOption<TString> ResultModelPath;
        TOption<bool> UseBestModel;

        const TString& GetTrainDir() const {
            return TrainDir.Get();
        }

        TString CreateResultModelFullPath() const {
            return GetFullPath(ResultModelPath.Get());
        }

        TString CreateSnapshotFullPath() const {
            return GetFullPath(SnapshotPath.Get());
        }

        TString CreateOutputBordersFullPath() const {
            return GetFullPath(OutputBordersFileName.Get());
        }

        bool NeedSaveBorders() const {
            return OutputBordersFileName.IsSet();
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

        EFinalCtrComputationMode GetFinalCtrComputationMode() const {
            return FinalCtrComputationMode.Get();
        }

        bool SaveSnapshot() const {
            return SaveSnapshotFlag.Get();
        }

        ui64 GetSnapshotSaveInterval() const {
            return SnapshotSaveIntervalSeconds.Get();
        }

        int GetVerbosePeriod() const {
            return VerbosePeriod.Get();
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

        TString CreateTrainingOptionsFullPath() const {
            return GetFullPath(TrainingOptionsFileName.Get());
        }

        TString CreateEvalFullPath() const {
            return GetFullPath(EvalFileName.Get());
        }

        bool operator==(const TOutputFilesOptions& rhs) const {
            return std::tie(
                TrainDir, Name, MetaFile, JsonLogPath, ProfileLogPath, LearnErrorLogPath, TestErrorLogPath,
                TimeLeftLog, ResultModelPath, SnapshotPath, ModelFormats, SaveSnapshotFlag,
                AllowWriteFilesFlag, FinalCtrComputationMode, UseBestModel, SnapshotSaveIntervalSeconds,
                EvalFileName, FstrRegularFileName, FstrInternalFileName, TrainingOptionsFileName,
                OutputBordersFileName
            ) == std::tie(
                rhs.TrainDir, rhs.Name, rhs.MetaFile, rhs.JsonLogPath, rhs.ProfileLogPath,
                rhs.LearnErrorLogPath, rhs.TestErrorLogPath, rhs.TimeLeftLog, rhs.ResultModelPath,
                rhs.SnapshotPath, rhs.ModelFormats, rhs.SaveSnapshotFlag, rhs.AllowWriteFilesFlag,
                rhs.FinalCtrComputationMode, rhs.UseBestModel, rhs.SnapshotSaveIntervalSeconds,
                rhs.EvalFileName, rhs.FstrRegularFileName, rhs.FstrInternalFileName,
                rhs.TrainingOptionsFileName, rhs.OutputBordersFileName
            );
        }

        bool operator!=(const TOutputFilesOptions& rhs) const {
            return !(rhs == *this);
        }

        void Load(const NJson::TJsonValue& options) {
            CheckedLoad(
                options,
                &TrainDir, &Name, &MetaFile, &JsonLogPath, &ProfileLogPath, &LearnErrorLogPath,
                &TestErrorLogPath, &TimeLeftLog, &ResultModelPath, &SnapshotPath, &ModelFormats,
                &SaveSnapshotFlag, &AllowWriteFilesFlag, &FinalCtrComputationMode, &UseBestModel,
                &SnapshotSaveIntervalSeconds, &EvalFileName, &OutputColumns, &FstrRegularFileName,
                &FstrInternalFileName, &TrainingOptionsFileName, &MetricPeriod, &VerbosePeriod,
                &PredictionTypes, &OutputBordersFileName
            );
            if (!VerbosePeriod.IsSet()) {
                VerbosePeriod.Set(MetricPeriod.Get());
            }
            Validate();
        }

        void Save(NJson::TJsonValue* options) const {
            SaveFields(
                options,
                TrainDir, Name, MetaFile, JsonLogPath, ProfileLogPath, LearnErrorLogPath, TestErrorLogPath,
                TimeLeftLog, ResultModelPath, SnapshotPath, ModelFormats, SaveSnapshotFlag,
                AllowWriteFilesFlag, FinalCtrComputationMode, UseBestModel, SnapshotSaveIntervalSeconds,
                EvalFileName, OutputColumns, FstrRegularFileName, FstrInternalFileName,
                TrainingOptionsFileName, MetricPeriod, VerbosePeriod, PredictionTypes, OutputBordersFileName
            );
        }

        void Validate() const {
            if (!AllowWriteFilesFlag.Get()) {
                CB_ENSURE(!SaveSnapshotFlag.Get(),
                    "allow_writing_files is set to False, and save_snapshot is set to True.");
            }
            CB_ENSURE(MetricPeriod.Get() != 0 && (VerbosePeriod.Get() % MetricPeriod.Get() == 0),
                "verbose should be a multiple of metric_period");
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
        TOption<EFinalCtrComputationMode> FinalCtrComputationMode;
        TOption<TString> EvalFileName;
        TOption<TString> FstrRegularFileName;
        TOption<TString> FstrInternalFileName;
        TOption<TString> TrainingOptionsFileName;

        TGpuOnlyOption<ui64> SnapshotSaveIntervalSeconds;
        TGpuOnlyOption<TString> OutputBordersFileName;
        TOption<int> VerbosePeriod;
        TOption<int> MetricPeriod;

        TCpuOnlyOption<TVector<EPredictionType>> PredictionTypes;
        TCpuOnlyOption<TVector<TString>> OutputColumns;
    };

}
