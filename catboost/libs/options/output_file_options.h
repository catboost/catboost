#pragma once

#include "option.h"
#include "json_helper.h"

#include <util/folder/path.h>
#include <util/generic/algorithm.h>
#include <util/string/split.h>
#include <util/system/types.h>


namespace NCatboostOptions {
    TString GetModelExtensionFromType(const EModelType& modelType);
    bool TryGetModelTypeFromExtension(const TString& modelExtension, EModelType& modelType);

    EModelType DefineModelFormat(const TString& modelPath);

    void AddExtension(const TString& extension, TString* modelFileName);

    class TOutputFilesOptions {
    public:
        explicit TOutputFilesOptions(ETaskType taskType)
            : ResultModelPath("result_model_file", "model")
            , UseBestModel("use_best_model", false)
            , BestModelMinTrees("best_model_min_trees", 1)
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
            , SnapshotSaveIntervalSeconds("snapshot_interval", 10 * 60)
            , OutputBordersFileName("output_borders", "", taskType)
            , VerbosePeriod("verbose", 1)
            , MetricPeriod("metric_period", 1)
            , PredictionTypes("prediction_type", {EPredictionType::RawFormulaVal}, taskType)
            , OutputColumns("output_columns", {"DocId", "RawFormulaVal", "Label"})
            , RocOutputPath("roc_file", "") {
        }

        TOption<TString> ResultModelPath;
        TOption<bool> UseBestModel;
        TOption<int> BestModelMinTrees;

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

        bool ExportRequiresStaticCtrProvider() const {
            return AnyOf(
                GetModelFormats().cbegin(),
                GetModelFormats().cend(),
                [](EModelType format) {
                    return format == EModelType::Python || format == EModelType::CPP || format == EModelType::json;
                }
            );
        }

        bool AddFileFormatExtension() const {
            return GetModelFormats().size() > 1 || !ResultModelPath.IsSet();
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

        TString GetRocOutputPath() const {
            return GetFullPath(RocOutputPath.Get());
        }

        bool operator==(const TOutputFilesOptions& rhs) const {
            return std::tie(
                TrainDir, Name, MetaFile, JsonLogPath, ProfileLogPath, LearnErrorLogPath, TestErrorLogPath,
                TimeLeftLog, ResultModelPath, SnapshotPath, ModelFormats, SaveSnapshotFlag,
                AllowWriteFilesFlag, FinalCtrComputationMode, UseBestModel, BestModelMinTrees,
                SnapshotSaveIntervalSeconds, EvalFileName, FstrRegularFileName, FstrInternalFileName,
                TrainingOptionsFileName, OutputBordersFileName, RocOutputPath
            ) == std::tie(
                rhs.TrainDir, rhs.Name, rhs.MetaFile, rhs.JsonLogPath, rhs.ProfileLogPath,
                rhs.LearnErrorLogPath, rhs.TestErrorLogPath, rhs.TimeLeftLog, rhs.ResultModelPath,
                rhs.SnapshotPath, rhs.ModelFormats, rhs.SaveSnapshotFlag, rhs.AllowWriteFilesFlag,
                rhs.FinalCtrComputationMode, rhs.UseBestModel, rhs.BestModelMinTrees,
                rhs.SnapshotSaveIntervalSeconds, rhs.EvalFileName, rhs.FstrRegularFileName,
                rhs.FstrInternalFileName, rhs.TrainingOptionsFileName, rhs.OutputBordersFileName,
                rhs.RocOutputPath
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
                &BestModelMinTrees, &SnapshotSaveIntervalSeconds, &EvalFileName, &OutputColumns,
                &FstrRegularFileName, &FstrInternalFileName, &TrainingOptionsFileName, &MetricPeriod,
                &VerbosePeriod, &PredictionTypes, &OutputBordersFileName, &RocOutputPath
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
                AllowWriteFilesFlag, FinalCtrComputationMode, UseBestModel, BestModelMinTrees,
                SnapshotSaveIntervalSeconds, EvalFileName, OutputColumns, FstrRegularFileName,
                FstrInternalFileName, TrainingOptionsFileName, MetricPeriod, VerbosePeriod, PredictionTypes,
                OutputBordersFileName, RocOutputPath
            );
        }

        void Validate() const {
            if (AnyOf(
                    GetModelFormats().cbegin(),
                    GetModelFormats().cend(),
                    [](EModelType format) {
                        return format == EModelType::Python || format == EModelType::CPP;
                    })) {
                CB_ENSURE(GetFinalCtrComputationMode() == EFinalCtrComputationMode::Default,
                    "allow final ctr calculation to save model in CPP or Python format");
            }
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

        TOption<ui64> SnapshotSaveIntervalSeconds;
        TGpuOnlyOption<TString> OutputBordersFileName;
        TOption<int> VerbosePeriod;
        TOption<int> MetricPeriod;

        TCpuOnlyOption<TVector<EPredictionType>> PredictionTypes;
        TOption<TVector<TString>> OutputColumns;
        TOption<TString> RocOutputPath;
    };

}
