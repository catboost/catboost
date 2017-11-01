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

namespace NCatboostCuda
{
    enum EBoostingType {
        Dynamic,
        Plain
    };

    class TSnapshotOptions {
    public:
        bool IsSnapshotEnabled() const {
            return !Path.empty();
        }

        const TString& GetSnapshotPath() const {
            return Path;
        }

        ui32 TimeBetweenWritesSec() const {
            return SaveInterval;

        }

        template<class TConfig>
        friend
        class TOptionsBinder;

    private:
        TString Path = "";
        bool Enabled = false;
        ui32 SaveInterval = 10 * 60; //every 10 minutes
    };

    class TOutputFilesOptions
    {
    public:
        TString GetLearnErrorLogPath() const
        {
            return GetPath(LearnErrorLogPath);
        }

        TString GetTestErrorLogPath() const
        {
            return GetPath(TestErrorLogPath);
        }

        TString GetTimeLeftLog() const
        {
            return GetPath(TimeLeftLog);
        }

        TString GetMetaFile() const
        {
            return GetPath(MetaFile);
        }

        const TString& GetName() const
        {
            return Name;
        }

        const TString& GetResultModelPath() const
        {
            return ResultModelPath;
        }

        template<class TConfig>
        friend
        class TOptionsBinder;

        template<class TConfig>
        friend
        class TOptionsJsonConverter;

    private:

        TString GetPath(const TString& fileName) const
        {
            TFsPath filePath(fileName);
            if (TrainDir.Empty() || filePath.IsAbsolute())
            {
                return fileName;
            } else
            {
                return JoinFsPaths(TrainDir, filePath);
            }
        }

    private:
        TString Name = "experiment";
        TString TrainDir = "";
        TString LearnErrorLogPath = "learn_error.tsv";
        TString TestErrorLogPath = "test_error.tsv";
        TString TimeLeftLog = "time_left.tsv";
        TString MetaFile = "meta.tsv";
        TString ResultModelPath = "catboost.bin";
    };

    class TOverfittingDetectorOptions
    {
    public:
        float GetAutoStopPval() const
        {
            return AutoStopPValue;
        }

        EOverfittingDetectorType GetDetectorType() const
        {
            return OverfittingDetectorType;
        }

        ui32 GetIterationsWait() const
        {
            return IterationsWait;
        }

        inline THolder<IOverfittingDetector> CreateOverfittingDetector(bool maxIsOptimal) const
        {
            switch (OverfittingDetectorType)
            {
                case EOverfittingDetectorType::IncToDec:
                {
                    return MakeHolder<TOverfittingDetectorIncToDec>(maxIsOptimal, AutoStopPValue, IterationsWait, true);
                }
                case EOverfittingDetectorType::Iter:
                {
                    return MakeHolder<TOverfittingDetectorIncToDec>(maxIsOptimal, 1.0, IterationsWait, true);
                }
                case EOverfittingDetectorType::Wilcoxon:
                {
                    return MakeHolder<TOverfittingDetectorWilcoxon>(maxIsOptimal, AutoStopPValue, IterationsWait, true);
                }
                default: {
                    Y_VERIFY(false, "Unknown OD type");
                }
            }
        }

        template<class TConfig>
        friend
        class TOptionsBinder;

        template<class TConfig>
        friend
        class TOptionsJsonConverter;

    private:
        float AutoStopPValue = 0;
        EOverfittingDetectorType OverfittingDetectorType = EOverfittingDetectorType::IncToDec;
        int IterationsWait = 20;
    };

    class TBoostingOptions
    {
    public:
        ui32 GetPermutationCount() const
        {
            return HasTimeFlag ? 1 : PermutationCount;
        }

        void SetPermutationCount(ui32 count)
        {
            PermutationCount = count;
        }

        double GetGrowthRate() const
        {
            return GrowthRate;
        }

        EBoostingType GetBoostingType() const {
            return BoostingType;
        }

        ui32 GetPermutationBlockSize() const
        {
            return PermutationBlockSize;
        }

        bool UseCpuRamForCatFeaturesDataSet() const
        {
            return UseCpuRamForCatFeaturesFlag;
        }

        TLearningRate GetLearningRate() const
        {
            return TLearningRate(Regularization);
        }

        bool UseBestModel() const
        {
            return UseBestModelFlag;
        }

        bool IsCalcScores() const
        {
            return CalcScores;
        }

        ui32 GetIterationCount() const
        {
            return IterationCount;
        }

        ui32 GetMinFoldSize() const
        {
            return MinFoldSize;
        }

        const TOverfittingDetectorOptions& GetOverfittingDetectorOptions() const
        {
            return OverfittingDetectorOptions;
        }

        double GetRandomStrength() const
        {
            return RandomStrength;
        }

        bool HasTime() const
        {
            return HasTimeFlag;
        }

        int GetPrintPeriod() const
        {
            return PrintPeriod;
        }

        template<class TConfig>
        friend
        class TOptionsBinder;

        template<class TConfig>
        friend
        class TOptionsJsonConverter;

    private:
        ui32 PermutationCount = 4;
        bool HasTimeFlag = false;
        double GrowthRate = 2.0;
        EBoostingType BoostingType = EBoostingType::Dynamic;
        ui32 PermutationBlockSize = 32;
        bool UseCpuRamForCatFeaturesFlag = false;
        ui32 IterationCount = 1000;
        ui32 MinFoldSize = 100;
        double RandomStrength = 1.0;
        double Regularization = 0.5;
        bool CalcScores = true;
        bool UseBestModelFlag;
        int PrintPeriod = 1;
        TOverfittingDetectorOptions OverfittingDetectorOptions;
    };
}
