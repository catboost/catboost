#pragma once

#include <catboost/cuda/cuda_util/gpu_random.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/models/additive_model.h>
#include <catboost/cuda/models/non_symmetric_tree.h>
#include <catboost/cuda/models/oblivious_model.h>

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/loggers/catboost_logger_helpers.h>
#include <catboost/private/libs/options/catboost_options.h>
#include <catboost/private/libs/options/enums.h>
#include <catboost/private/libs/options/output_file_options.h>
#include <catboost/libs/train_lib/train_model.h>

#include <library/cpp/object_factory/object_factory.h>
#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/ptr.h>

namespace NCatboostCuda {
    using TGpuTrainResult = std::variant<
        THolder<TAdditiveModel<TObliviousTreeModel>>,
        THolder<TAdditiveModel<TNonSymmetricTree>>
    >;

    class IGpuTrainer {
    public:
        virtual TGpuTrainResult TrainModel(
            TBinarizedFeaturesManager& featureManager,
            const TTrainModelInternalOptions& internalOptions,
            const NCatboostOptions::TCatBoostOptions& catBoostOptions,
            const NCatboostOptions::TOutputFilesOptions& outputOptions,
            const NCB::TTrainingDataProvider& learn,
            const NCB::TTrainingDataProvider* test,
            const NCB::TFeatureEstimators& featureEstimators,
            const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
            const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
            TGpuAwareRandom& random,
            ui32 approxDimension,
            ITrainingCallbacks* trainingCallbacks,
            NPar::ILocalExecutor* localExecutor,
            TVector<TVector<double>>* testMultiApprox, // [dim][objectIdx]
            TMetricsAndTimeLeftHistory* metricsAndTimeHistory) const = 0;

        virtual void ModelBasedEval(
            TBinarizedFeaturesManager& featureManager,
            const NCatboostOptions::TCatBoostOptions& catBoostOptions,
            const NCatboostOptions::TOutputFilesOptions& outputOptions,
            const NCB::TTrainingDataProvider& learn,
            const NCB::TTrainingDataProvider& test,
            TGpuAwareRandom& random,
            ui32 approxDimension,
            NPar::ILocalExecutor* localExecutor) const = 0;

        virtual ~IGpuTrainer() = default;
    };

    struct TGpuTrainerFactoryKey {
        ELossFunction Loss;
        EGrowPolicy GrowPolicy;

        TGpuTrainerFactoryKey(ELossFunction loss, EGrowPolicy policy)
            : Loss(loss)
            , GrowPolicy(policy)
        {
        }

        bool operator==(const TGpuTrainerFactoryKey& rhs) const {
            return std::tie(Loss, GrowPolicy) == std::tie(rhs.Loss, rhs.GrowPolicy);
        }

        bool operator!=(const TGpuTrainerFactoryKey& rhs) const {
            return !(rhs == *this);
        }

        ui64 GetHash() const {
            return MultiHash(Loss, GrowPolicy);
        }

        bool operator<(const TGpuTrainerFactoryKey& key) const {
            return GetHash() < key.GetHash();
        }
    };

    inline TGpuTrainerFactoryKey GetTrainerFactoryKey(ELossFunction loss, EGrowPolicy policy = EGrowPolicy::SymmetricTree) {
        return TGpuTrainerFactoryKey(loss, policy);
    }

    inline TGpuTrainerFactoryKey GetTrainerFactoryKey(const NCatboostOptions::TCatBoostOptions& options) {
        return TGpuTrainerFactoryKey(options.LossFunctionDescription->GetLossFunction(),
                                     options.ObliviousTreeOptions->GrowPolicy);
    }

    using TGpuTrainerFactory = NObjectFactory::TParametrizedObjectFactory<IGpuTrainer, TGpuTrainerFactoryKey>;

}

template <>
struct THash<NCatboostCuda::TGpuTrainerFactoryKey> {
    inline size_t operator()(const NCatboostCuda::TGpuTrainerFactoryKey& key) const {
        return key.GetHash();
    }
};
