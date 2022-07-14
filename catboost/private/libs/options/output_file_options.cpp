#include "output_file_options.h"
#include "json_helper.h"

#include <util/folder/path.h>
#include <util/generic/algorithm.h>
#include <util/string/split.h>

TString NCatboostOptions::GetModelExtensionFromType(const EModelType modelType) {
    switch(modelType) {
        case EModelType::CatboostBinary:
            return "bin";
        case EModelType::AppleCoreML:
            return "coreml";
        case EModelType::Json:
            return "json";
        case EModelType::Cpp:
            return "cpp";
        case EModelType::Python:
            return "py";
        case EModelType::Onnx:
            return "onnx";
        case EModelType::Pmml:
            return "pmml";
        case EModelType::CPUSnapshot:
            return "cbsnapshot";
        default:
            CB_ENSURE(false, "Unexpected model type");
    }
    Y_UNREACHABLE();
}

bool NCatboostOptions::TryGetModelTypeFromExtension(const TStringBuf modelExtension, EModelType& modelType) {
    if (modelExtension == "bin") {
        modelType = EModelType::CatboostBinary;
    } else if (modelExtension == "coreml") {
        modelType = EModelType::AppleCoreML;
    } else if (modelExtension =="json") {
        modelType = EModelType::Json;
    } else if(modelExtension == "cpp") {
        modelType = EModelType::Cpp;
    } else if (modelExtension == "py") {
        modelType = EModelType::Python;
    } else if (modelExtension == "onnx") {
        modelType = EModelType::Onnx;
    } else if (modelExtension == "pmml") {
        modelType = EModelType::Pmml;
    } else if (modelExtension == "cbsnapshot") {
        modelType = EModelType::CPUSnapshot;
    } else {
        return false;
    }
    return true;
}

EModelType NCatboostOptions::DefineModelFormat(TStringBuf modelPath) {
    EModelType modelType;
    TVector<TString> tokens = StringSplitter(modelPath).Split('.').SkipEmpty().ToList<TString>();
    if (tokens.size() > 1) {
        if (NCatboostOptions::TryGetModelTypeFromExtension(tokens.back(), modelType)) {
            return modelType;
        }
    }
    return EModelType::CatboostBinary;
}

TString NCatboostOptions::AddExtension(const EModelType& format, const TString& modelFileName, bool addExtension) {
    auto extension = NCatboostOptions::GetModelExtensionFromType(format);
    if (addExtension && !modelFileName.EndsWith("." + extension)) {
        return modelFileName + "." + extension;
    }
    return modelFileName;
}

NCatboostOptions::TOutputFilesOptions::TOutputFilesOptions()
    : ResultModelPath("result_model_file", "model")
    , UseBestModel("use_best_model", false)
    , BestModelMinTrees("best_model_min_trees", 1)
    , TrainDir("train_dir", "catboost_info")
    , Name("name", "experiment")
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
    , FinalFeatureCalcerComputationMode("final_feature_calcer_computation_mode", EFinalFeatureCalcersComputationMode::Default)
    , EvalFileName("eval_file_name", "")
    , FstrRegularFileName("fstr_regular_file", "")
    , FstrInternalFileName("fstr_internal_file", "")
    , FstrType("fstr_type", EFstrType::FeatureImportance)
    , TrainingOptionsFileName("training_options_file", "")
    , SnapshotSaveIntervalSeconds("snapshot_interval", 10 * 60)
    , OutputBordersFileName("output_borders", "")
    , VerbosePeriod("verbose", 1)
    , MetricPeriod("metric_period", 1)
    , PredictionTypes("prediction_type", {EPredictionType::RawFormulaVal})
    , OutputColumns("output_columns", {"SampleId", "RawFormulaVal", "Label"})
    , RocOutputPath("roc_file", "") {
}

const TString& NCatboostOptions::TOutputFilesOptions::GetTrainDir() const {
    return TrainDir.Get();
}

TString NCatboostOptions::TOutputFilesOptions::CreateResultModelFullPath() const {
    return GetFullPath(ResultModelPath.Get());
}

TString NCatboostOptions::TOutputFilesOptions::CreateSnapshotFullPath() const {
    return GetFullPath(SnapshotPath.Get());
}

TString NCatboostOptions::TOutputFilesOptions::CreateOutputBordersFullPath() const {
    return GetFullPath(OutputBordersFileName.Get());
}

bool NCatboostOptions::TOutputFilesOptions::NeedSaveBorders() const {
    return OutputBordersFileName.IsSet();
}
//local
const TString& NCatboostOptions::TOutputFilesOptions::GetLearnErrorFilename() const {
    return LearnErrorLogPath.Get();
}

const TString& NCatboostOptions::TOutputFilesOptions::GetTestErrorFilename() const {
    return TestErrorLogPath.Get();
}

const TString& NCatboostOptions::TOutputFilesOptions::GetTimeLeftLogFilename() const {
    return TimeLeftLog.Get();
}

const TVector<EModelType>& NCatboostOptions::TOutputFilesOptions::GetModelFormats() const {
    return ModelFormats.Get();
}

bool NCatboostOptions::TOutputFilesOptions::ExportRequiresStaticCtrProvider() const {
    return AnyOf(
            GetModelFormats().cbegin(),
            GetModelFormats().cend(),
            [](EModelType format) {
            return format == EModelType::Python || format == EModelType::Cpp || format == EModelType::Json;
            }
            );
}

bool NCatboostOptions::TOutputFilesOptions::AddFileFormatExtension() const {
    return GetModelFormats().size() > 1 || !ResultModelPath.IsSet();
}

const TString& NCatboostOptions::TOutputFilesOptions::GetJsonLogFilename() const {
    return JsonLogPath.Get();
}

const TString& NCatboostOptions::TOutputFilesOptions::GetProfileLogFilename() const {
    return ProfileLogPath.Get();
}

const TString& NCatboostOptions::TOutputFilesOptions::GetResultModelFilename() const {
    return ResultModelPath.Get();
}

const TString& NCatboostOptions::TOutputFilesOptions::GetSnapshotFilename() const {
    return SnapshotPath.Get();
}

bool NCatboostOptions::TOutputFilesOptions::ShrinkModelToBestIteration() const {
    return UseBestModel.Get();
}

const TString& NCatboostOptions::TOutputFilesOptions::GetName() const {
    return Name.Get();
}

const TVector<EPredictionType>& NCatboostOptions::TOutputFilesOptions::GetPredictionTypes() const {
    return PredictionTypes.Get();
}

const TVector<TString> NCatboostOptions::TOutputFilesOptions::GetOutputColumns(bool datasetHasLabels) const {
    if (!OutputColumns.IsSet()) {
        TVector<TString> result{"SampleId"};
        if (!PredictionTypes.IsSet()) {
            result.emplace_back("RawFormulaVal");
        } else {
            for (const auto& predictionType : PredictionTypes.Get()) {
                result.emplace_back(ToString(predictionType));
            }
        }
        if (datasetHasLabels) {
            result.emplace_back("Label");
        }
        return result;
    } else {
        if (!PredictionTypes.IsSet()) {
            return OutputColumns;
        }
        TVector<TString> result(OutputColumns);
        for (const auto& predictionType : PredictionTypes.Get()) {
            const auto column = ToString(predictionType);
            if (Count(result, column) == 0) {
                result.emplace_back(column);
            }
        }
        return result;
    }
}

bool NCatboostOptions::TOutputFilesOptions::AllowWriteFiles() const {
    return AllowWriteFilesFlag.Get();
}

EFinalCtrComputationMode NCatboostOptions::TOutputFilesOptions::GetFinalCtrComputationMode() const {
    return FinalCtrComputationMode.Get();
}

bool NCatboostOptions::TOutputFilesOptions::SaveSnapshot() const {
    return SaveSnapshotFlag.Get();
}

ui64 NCatboostOptions::TOutputFilesOptions::GetSnapshotSaveInterval() const {
    return SnapshotSaveIntervalSeconds.Get();
}

int NCatboostOptions::TOutputFilesOptions::GetVerbosePeriod() const {
    return VerbosePeriod.IsSet() ? VerbosePeriod.Get() : MetricPeriod.IsSet() ? MetricPeriod.Get() : VerbosePeriod.Get();
}

int NCatboostOptions::TOutputFilesOptions::GetMetricPeriod() const {
    return MetricPeriod.Get();
}

TString NCatboostOptions::TOutputFilesOptions::CreateFstrRegularFullPath() const {
    return GetFullPath(FstrRegularFileName.Get());
}

TString NCatboostOptions::TOutputFilesOptions::CreateFstrIternalFullPath() const {
    return GetFullPath(FstrInternalFileName.Get());
}

EFstrType NCatboostOptions::TOutputFilesOptions::GetFstrType() const {
    return FstrType.Get();
}

bool NCatboostOptions::TOutputFilesOptions::IsFstrTypeSet() const {
    return FstrType.IsSet();
}

TString NCatboostOptions::TOutputFilesOptions::CreateTrainingOptionsFullPath() const {
    return GetFullPath(TrainingOptionsFileName.Get());
}

TString NCatboostOptions::TOutputFilesOptions::CreateEvalFullPath() const {
    return GetFullPath(EvalFileName.Get());
}

TString NCatboostOptions::TOutputFilesOptions::GetRocOutputPath() const {
    return GetFullPath(RocOutputPath.Get());
}

bool NCatboostOptions::TOutputFilesOptions::operator==(const TOutputFilesOptions& rhs) const {
    return std::tie(
            TrainDir, Name, JsonLogPath, ProfileLogPath, LearnErrorLogPath, TestErrorLogPath,
            TimeLeftLog, ResultModelPath, SnapshotPath, ModelFormats, SaveSnapshotFlag,
            AllowWriteFilesFlag, FinalCtrComputationMode, FinalFeatureCalcerComputationMode, UseBestModel, BestModelMinTrees,
            SnapshotSaveIntervalSeconds, EvalFileName, FstrRegularFileName, FstrInternalFileName, FstrType,
            TrainingOptionsFileName, OutputBordersFileName, RocOutputPath
            ) == std::tie(
                rhs.TrainDir, rhs.Name, rhs.JsonLogPath, rhs.ProfileLogPath,
                rhs.LearnErrorLogPath, rhs.TestErrorLogPath, rhs.TimeLeftLog, rhs.ResultModelPath,
                rhs.SnapshotPath, rhs.ModelFormats, rhs.SaveSnapshotFlag, rhs.AllowWriteFilesFlag,
                rhs.FinalCtrComputationMode, rhs.FinalFeatureCalcerComputationMode, rhs.UseBestModel, rhs.BestModelMinTrees,
                rhs.SnapshotSaveIntervalSeconds, rhs.EvalFileName, rhs.FstrRegularFileName,
                rhs.FstrInternalFileName, rhs.FstrType, rhs.TrainingOptionsFileName, rhs.OutputBordersFileName,
                rhs.RocOutputPath
                );
}

bool NCatboostOptions::TOutputFilesOptions::operator!=(const TOutputFilesOptions& rhs) const {
    return !(rhs == *this);
}

void NCatboostOptions::TOutputFilesOptions::Load(const NJson::TJsonValue& options) {
    CheckedLoad(
            options,
            &TrainDir, &Name, &JsonLogPath, &ProfileLogPath, &LearnErrorLogPath,
            &TestErrorLogPath, &TimeLeftLog, &ResultModelPath, &SnapshotPath, &ModelFormats,
            &SaveSnapshotFlag, &AllowWriteFilesFlag, &FinalCtrComputationMode, &FinalFeatureCalcerComputationMode,
            &UseBestModel, &BestModelMinTrees, &SnapshotSaveIntervalSeconds, &EvalFileName, &OutputColumns,
            &FstrRegularFileName, &FstrInternalFileName, &FstrType, &TrainingOptionsFileName, &MetricPeriod,
            &VerbosePeriod, &PredictionTypes, &OutputBordersFileName, &RocOutputPath
            );
    if (!VerbosePeriod.IsSet() || VerbosePeriod.Get() == 1) {
        VerbosePeriod.Set(MetricPeriod.Get());
    }
    Validate();
}

void NCatboostOptions::TOutputFilesOptions::Save(NJson::TJsonValue* options) const {
    SaveFields(
            options,
            TrainDir, Name, JsonLogPath, ProfileLogPath, LearnErrorLogPath, TestErrorLogPath,
            TimeLeftLog, ResultModelPath, SnapshotPath, ModelFormats, SaveSnapshotFlag,
            AllowWriteFilesFlag, FinalCtrComputationMode, FinalFeatureCalcerComputationMode, UseBestModel,
            BestModelMinTrees, SnapshotSaveIntervalSeconds, EvalFileName, OutputColumns, FstrRegularFileName,
            FstrInternalFileName, FstrType, TrainingOptionsFileName, MetricPeriod, VerbosePeriod, PredictionTypes,
            OutputBordersFileName, RocOutputPath
            );
}

void NCatboostOptions::TOutputFilesOptions::Validate() const {
    if (AnyOf(
                GetModelFormats().cbegin(),
                GetModelFormats().cend(),
                [](EModelType format) {
                return format == EModelType::Python || format == EModelType::Cpp;
                })) {
        CB_ENSURE(GetFinalCtrComputationMode() == EFinalCtrComputationMode::Default,
                "allow final ctr calculation to save model in CPP or Python format");
    }
    if (!AllowWriteFilesFlag.Get()) {
        CB_ENSURE(!SaveSnapshotFlag.Get(),
                "allow_writing_files is set to False, and save_snapshot is set to True.");
    }
    CB_ENSURE(GetVerbosePeriod() >= 0, "Verbose period should be nonnegative.");
    CB_ENSURE(GetMetricPeriod() > 0, "Metric period should be positive.");
    CB_ENSURE(GetVerbosePeriod() % GetMetricPeriod() == 0,
        "verbose should be a multiple of metric_period, got " <<
        GetVerbosePeriod() << " vs " << GetMetricPeriod());

    EFstrCalculatedInFitType fstrType;
    CB_ENSURE(TryFromString<EFstrCalculatedInFitType>(ToString(FstrType.Get()), fstrType),
        "Unsupported fstr type " << FstrType.Get());
    for (auto predictionType : PredictionTypes.Get()) {
        CB_ENSURE(!IsUncertaintyPredictionType(predictionType), "Unsupported prediction type " << predictionType);
    }
}

TString NCatboostOptions::TOutputFilesOptions::GetFullPath(const TString& fileName) const {
    if (fileName.empty()) {
        return {};
    }

    TFsPath filePath(fileName);
    const TString& trainDirStr = TrainDir.Get();
    if (trainDirStr.empty() || filePath.IsAbsolute()) {
        return fileName;
    } else {
        return JoinFsPaths(trainDirStr, filePath);
    }
}

EFinalFeatureCalcersComputationMode NCatboostOptions::TOutputFilesOptions::GetFinalFeatureCalcerComputationMode() const {
    return FinalFeatureCalcerComputationMode;
}
