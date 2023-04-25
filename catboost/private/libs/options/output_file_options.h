#pragma once

#include "enums.h"
#include "option.h"
#include "unimplemented_aware_option.h"

#include <catboost/libs/helpers/exception.h>

#include <util/generic/vector.h>
#include <util/system/types.h>

namespace NJson {
    class TJsonValue;
}

namespace NCatboostOptions {
    TString GetModelExtensionFromType(EModelType modelType);
    bool TryGetModelTypeFromExtension(TStringBuf modelExtension, EModelType& modelType);

    EModelType DefineModelFormat(TStringBuf modelPath);

    TString AddExtension(const EModelType& format, const TString& modelFileName, bool addExtension = true);

    class TOutputFilesOptions {
    public:
        explicit TOutputFilesOptions();

        TOption<TString> ResultModelPath;
        TOption<bool> UseBestModel;
        TOption<int> BestModelMinTrees;

        const TString& GetTrainDir() const;

        TString CreateResultModelFullPath() const;

        TString CreateSnapshotFullPath() const;

        TString CreateOutputBordersFullPath() const;

        bool NeedSaveBorders() const;

        //local
        const TString& GetLearnErrorFilename() const;

        const TString& GetTestErrorFilename() const;

        const TString& GetTimeLeftLogFilename() const;

        const TVector<EModelType>& GetModelFormats() const;

        bool ExportRequiresStaticCtrProvider() const;

        bool AddFileFormatExtension() const;

        const TString& GetJsonLogFilename() const;

        const TString& GetProfileLogFilename() const;

        const TString& GetResultModelFilename() const;

        const TString& GetSnapshotFilename() const;

        bool ShrinkModelToBestIteration() const;

        const TString& GetName() const;

        const TVector<EPredictionType>& GetPredictionTypes() const;

        // default depends on whether dataset has target or not
        const TVector<TString> GetOutputColumns(bool datasetHasLabels) const;

        bool AllowWriteFiles() const;

        EFinalCtrComputationMode GetFinalCtrComputationMode() const;

        EFinalFeatureCalcersComputationMode GetFinalFeatureCalcerComputationMode() const;

        bool SaveSnapshot() const;

        ui64 GetSnapshotSaveInterval() const;

        int GetVerbosePeriod() const;

        int GetMetricPeriod() const;

        TString CreateFstrRegularFullPath() const;

        TString CreateFstrIternalFullPath() const;

        EFstrType GetFstrType() const;

        bool IsFstrTypeSet() const;

        TString CreateTrainingOptionsFullPath() const;

        TString CreateEvalFullPath() const;

        TString GetRocOutputPath() const;

        void SetAllowWriteFiles(bool flag) {
            if (!flag) {
                CB_ENSURE(!SaveSnapshot(), "Can't disable writing files because saving snapshots is enabled");
            }
            AllowWriteFilesFlag.Set(flag);
        }

        void SetSaveSnapshotFlag(bool flag) {
            if (flag) {
                CB_ENSURE(AllowWriteFiles(), "Can't enable saving snapshots because writing files is disabled");
            }
            SaveSnapshotFlag.Set(flag);
        }

        void SetMetricPeriod(ui32 period) {
            MetricPeriod.Set(period);
        }

        bool IsMetricPeriodSet() const {
            return MetricPeriod.IsSet();
        }

        void SetTrainDir(const TString& trainDir) {
            TrainDir.Set(trainDir);
        }

        void SetSnapshotFilename(const TString& filename) {
            SnapshotPath.Set(filename);
        }


        bool operator==(const TOutputFilesOptions& rhs) const;
        bool operator!=(const TOutputFilesOptions& rhs) const;

        void Save(NJson::TJsonValue* options) const;
        void Load(const NJson::TJsonValue& options);

        void Validate() const;

    private:
        TString GetFullPath(const TString& fileName) const;

    private:
        TOption<TString> TrainDir;
        TOption<TString> Name;
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
        TOption<EFinalFeatureCalcersComputationMode> FinalFeatureCalcerComputationMode;
        TOption<TString> EvalFileName;
        TOption<TString> FstrRegularFileName;
        TOption<TString> FstrInternalFileName;
        TOption<EFstrType> FstrType;
        TOption<TString> TrainingOptionsFileName;

        TOption<ui64> SnapshotSaveIntervalSeconds;
        TOption<TString> OutputBordersFileName;
        TOption<int> VerbosePeriod;
        TOption<int> MetricPeriod;

        TOption<TVector<EPredictionType>> PredictionTypes;
        TOption<TVector<TString>> OutputColumns;
        TOption<TString> RocOutputPath;
    };
}
