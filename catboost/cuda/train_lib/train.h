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
#include <library/threading/local_executor/local_executor.h>

#include <util/generic/ptr.h>

namespace NCatboostCuda {
    class IGpuTrainer {
    public:
        virtual THolder<TAdditiveModel<TObliviousTreeModel>> TrainModel(
            TBinarizedFeaturesManager& featureManager,
            const TTrainModelInternalOptions& internalOptions,
            const NCatboostOptions::TCatBoostOptions& catBoostOptions,
            const NCatboostOptions::TOutputFilesOptions& outputOptions,
            const NCB::TTrainingDataProvider& learn,
            const NCB::TTrainingDataProvider* test,
            TGpuAwareRandom& random,
            ui32 approxDimension,
            const TMaybe<TOnEndIterationCallback>& onEndIterationCallback,
            NPar::TLocalExecutor* localExecutor,
            TVector<TVector<double>>* testMultiApprox, // [dim][objectIdx]
            TMetricsAndTimeLeftHistory* metricsAndTimeHistory) const = 0;

        virtual ~IGpuTrainer() = default;
    };

    struct TGpuTrainerFactoryKey {
        ELossFunction Loss;
        EGrowingPolicy GrowingPolicy;

        TGpuTrainerFactoryKey(ELossFunction loss, EGrowingPolicy policy)
            : Loss(loss)
            , GrowingPolicy(policy)
        {
        }

        bool operator==(const TGpuTrainerFactoryKey& rhs) const {
            return std::tie(Loss, GrowingPolicy) == std::tie(rhs.Loss, rhs.GrowingPolicy);
        }

        bool operator!=(const TGpuTrainerFactoryKey& rhs) const {
            return !(rhs == *this);
        }

        ui64 GetHash() const {
            return MultiHash(Loss, GrowingPolicy);
        }

        bool operator<(const TGpuTrainerFactoryKey& key) const {
            return GetHash() < key.GetHash();
        }
    };

    inline TGpuTrainerFactoryKey GetTrainerFactoryKey(ELossFunction loss, EGrowingPolicy policy = EGrowingPolicy::ObliviousTree) {
        return TGpuTrainerFactoryKey(loss, policy);
    }

    inline TGpuTrainerFactoryKey GetTrainerFactoryKey(const NCatboostOptions::TCatBoostOptions& options) {
        return TGpuTrainerFactoryKey(options.LossFunctionDescription->GetLossFunction(),
                                     options.ObliviousTreeOptions->GrowingPolicy);
    }

    using TGpuTrainerFactory = NObjectFactory::TParametrizedObjectFactory<IGpuTrainer, TGpuTrainerFactoryKey>;

}

template <>
struct THash<NCatboostCuda::TGpuTrainerFactoryKey> {
    inline size_t operator()(const NCatboostCuda::TGpuTrainerFactoryKey& key) const {
        return key.GetHash();
    }
};
