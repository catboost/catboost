#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/loggers/catboost_logger_helpers.h>
#include <catboost/private/libs/options/catboost_options.h>
#include <catboost/private/libs/options/enums.h>
#include <catboost/private/libs/options/output_file_options.h>
#include <catboost/libs/train_lib/train_model.h>

#include <library/cpp/object_factory/object_factory.h>
#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/ptr.h>

namespace NCatboostMlx {

    // Top-level MLX trainer, registered into CatBoost's TTrainerFactory as ETaskType::GPU
    // on darwin-arm64 builds (where CUDA is unavailable).
    class TMLXModelTrainer : public IModelTrainer {
    public:
        void TrainModel(
            const TTrainModelInternalOptions& internalOptions,
            const NCatboostOptions::TCatBoostOptions& catboostOptions,
            const NCatboostOptions::TOutputFilesOptions& outputOptions,
            const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
            const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
            NCB::TTrainingDataProviders trainingData,
            TMaybe<NCB::TPrecomputedOnlineCtrData> precomputedSingleOnlineCtrDataForSingleFold,
            const TLabelConverter& labelConverter,
            ITrainingCallbacks* trainingCallbacks,
            ICustomCallbacks* customCallbacks,
            TMaybe<TFullModel*> initModel,
            THolder<TLearnProgress> initLearnProgress,
            NCB::TDataProviders initModelApplyCompatiblePools,
            NPar::ILocalExecutor* localExecutor,
            const TMaybe<TRestorableFastRng64*> rand,
            TFullModel* dstModel,
            const TVector<TEvalResult*>& evalResultPtrs,
            TMetricsAndTimeLeftHistory* metricsAndTimeHistory,
            THolder<TLearnProgress>* dstLearnProgress
        ) const override;

        void ModelBasedEval(
            const NCatboostOptions::TCatBoostOptions& catboostOptions,
            const NCatboostOptions::TOutputFilesOptions& outputOptions,
            NCB::TTrainingDataProviders trainingData,
            const TLabelConverter& labelConverter,
            NPar::ILocalExecutor* localExecutor
        ) const override;
    };

}  // namespace NCatboostMlx
