#pragma once

#include "learning_rate.h"
#include <catboost/libs/overfitting_detector/overfitting_detector.h>
#include <catboost/cuda/targets/target_base.h>
#include <catboost/cuda/targets/mse.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/models/additive_model.h>
#include <catboost/cuda/gpu_data/fold_based_dataset.h>
#include <catboost/cuda/gpu_data/fold_based_dataset_builder.h>
#include <catboost/cuda/targets/target_options.h>

class TOutputFilesOptions {
public:
    const TString& GetLearnErrorLogPath() const {
        return LearnErrorLogPath;
    }

    const TString& GetTestErrorLogPath() const {
        return TestErrorLogPath;
    }

    const TString& GetTimeLeftLog() const {
        return TimeLeftLog;
    }

    const TString& GetMetaFile() const {
        return MetaFile;
    }

    const TString& GetName() const {
        return Name;
    }

    const TString& GetResultModelPath() const {
        return ResultModelPath;
    }

    template <class TConfig>
    friend class TOptionsBinder;

private:
    TString Name = "experiment";
    TString LearnErrorLogPath = "learn_error.tsv";
    TString TestErrorLogPath = "test_error.tsv";
    TString TimeLeftLog = "time_left.tsv";
    TString MetaFile = "meta.tsv";
    TString ResultModelPath = "catboost.bin";
};

class TOverfittingDetectorOptions {
public:
    float GetAutoStopPval() const {
        return AutoStopPValue;
    }

    EOverfittingDetectorType GetDetectorType() const {
        return OverfittingDetectorType;
    }

    ui32 GetIterationsWait() const {
        return IterationsWait;
    }

    inline THolder<IOverfittingDetector> CreateOverfittingDetector(bool maxIsOptimal) const {
        switch (OverfittingDetectorType) {
            case EOverfittingDetectorType::IncToDec: {
                return MakeHolder<TOverfittingDetectorIncToDec>(maxIsOptimal, AutoStopPValue, IterationsWait, true);
            }
            case EOverfittingDetectorType::Iter: {
                return MakeHolder<TOverfittingDetectorIncToDec>(maxIsOptimal, 1.0, IterationsWait, true);
            }
            case EOverfittingDetectorType::Wilcoxon: {
                return MakeHolder<TOverfittingDetectorWilcoxon>(maxIsOptimal, AutoStopPValue, IterationsWait, true);
            }
        }
    }

    template <class TConfig>
    friend class TOptionsBinder;

private:
    float AutoStopPValue = 0;
    EOverfittingDetectorType OverfittingDetectorType = EOverfittingDetectorType::IncToDec;
    int IterationsWait = 20;
};

class TBoostingOptions {
public:
    ui32 GetPermutationCount() const {
        return HasTimeFlag ? 1 : PermutationCount;
    }

    void SetPermutationCount(ui32 count) {
        PermutationCount = count;
    }

    double GetGrowthRate() const {
        return GrowthRate;
    }

    bool DisableDontLookAhead() const {
        return DisableDontLookAheadFlag;
    }

    ui32 GetPermutationBlockSize() const {
        return PermutationBlockSize;
    }

    bool UseCpuRamForCatFeaturesDataSet() const {
        return UseCpuRamForCatFeaturesFlag;
    }

    TLearningRate GetLearningRate() const {
        return TLearningRate(Regularization);
    }

    bool UseBestModel() const {
        return UseBestModelFlag;
    }

    bool IsCalcScores() const {
        return CalcScores;
    }
    ui32 GetIterationCount() const {
        return IterationCount;
    }

    ui32 GetMinFoldSize() const {
        return MinFoldSize;
    }

    const TOverfittingDetectorOptions& GetOverfittingDetectorOptions() const {
        return OverfittingDetectorOptions;
    }

    double GetRandomStrength() const {
        return RandomStrength;
    }

    bool HasTime() const {
        return HasTimeFlag;
    }

    template <class TConfig>
    friend class TOptionsBinder;

private:
    ui32 PermutationCount = 4;
    bool HasTimeFlag = false;
    double GrowthRate = 2.0;
    bool DisableDontLookAheadFlag = false;
    ui32 PermutationBlockSize = 1;
    bool UseCpuRamForCatFeaturesFlag = false;
    ui32 IterationCount = 1000;
    ui32 MinFoldSize = 1024;
    double RandomStrength = 1.0;
    double Regularization = 0.5;
    bool CalcScores = true;
    bool UseBestModelFlag;
    TOverfittingDetectorOptions OverfittingDetectorOptions;
};
