#pragma once

#include <catboost/cuda/cuda_util/gpu_random.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/models/additive_model.h>
#include <catboost/cuda/models/oblivious_model.h>

#include <catboost/libs/data_new/data_provider.h>
#include <catboost/libs/loggers/catboost_logger_helpers.h>
#include <catboost/libs/options/catboost_options.h>
#include <catboost/libs/options/enums.h>
#include <catboost/libs/options/output_file_options.h>
#include <catboost/libs/train_lib/train_model.h>

#include <library/object_factory/object_factory.h>

#include <util/generic/ptr.h>


namespace NCatboostCuda {

    class IGpuTrainer {
    public:
        virtual THolder<TAdditiveModel<TObliviousTreeModel>> TrainModel(
            TBinarizedFeaturesManager& featureManager,
            const NCatboostOptions::TCatBoostOptions& catBoostOptions,
            const NCatboostOptions::TOutputFilesOptions& outputOptions,
            const NCB::TTrainingDataProvider& learn,
            const NCB::TTrainingDataProvider* test,
            TGpuAwareRandom& random,
            ui32 approxDimension,
            const TMaybe<TOnEndIterationCallback>& onEndIterationCallback,
            TVector<TVector<double>>* testMultiApprox, // [dim][objectIdx]
            TMetricsAndTimeLeftHistory* metricsAndTimeHistory) const = 0;

        virtual ~IGpuTrainer() = default;
    };

    using TGpuTrainerFactory = NObjectFactory::TParametrizedObjectFactory<IGpuTrainer, ELossFunction>;

}
