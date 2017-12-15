#pragma once

#include <catboost/libs/model/model.h>
#include <catboost/libs/options/catboost_options.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/train_lib/train_model.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/data/data_provider.h>
#include <catboost/cuda/models/additive_model.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <util/system/fs.h>

namespace NCatboostCuda {
    class IGpuTrainer {
    public:
        virtual THolder<TAdditiveModel<TObliviousTreeModel>> TrainModel(
            TBinarizedFeaturesManager& featureManager,
            const NCatboostOptions::TCatBoostOptions& catBoostOptions,
            const NCatboostOptions::TOutputFilesOptions& outputOptions,
            const TDataProvider& learn,
            const TDataProvider* test,
            TRandom& random,
            bool storeCatFeaturesInPinnedMemory) const = 0;

        virtual ~IGpuTrainer() = default;
    };

    using TGpuTrainerFactory = NObjectFactory::TParametrizedObjectFactory<IGpuTrainer, ELossFunction>;

    TFullModel TrainModel(const NCatboostOptions::TCatBoostOptions& trainCatBoostOptions,
                          const NCatboostOptions::TOutputFilesOptions& outputOptions,
                          const TDataProvider& dataProvider,
                          const TDataProvider* testProvider,
                          TBinarizedFeaturesManager& featuresManager);

    void TrainModel(const NJson::TJsonValue& params,
                    const NCatboostOptions::TOutputFilesOptions& outputOptions,
                    TPool& learnPool,
                    const TPool& testPool,
                    TFullModel* model);

    void TrainModel(const NCatboostOptions::TPoolLoadParams& poolLoadOptions,
                    const NCatboostOptions::TOutputFilesOptions& outputOptions,
                    const NJson::TJsonValue& jsonOptions);

}
