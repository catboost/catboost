#pragma once

#include "helpers.h"
#include "oblivious_tree_structure_searcher.h"
#include "add_oblivious_tree_model_feature_parallel.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/gpu_data/feature_parallel_dataset.h>
#include <catboost/cuda/methods/leaves_estimation/oblivious_tree_leaves_estimator.h>
#include <catboost/cuda/models/additive_model.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/gpu_data/bootstrap.h>
#include <catboost/cuda/targets/target_func.h>
#include <catboost/private/libs/options/catboost_options.h>
#include <catboost/private/libs/options/boosting_options.h>

namespace NCatboostCuda {
    class TFeatureParallelPointwiseObliviousTree {
    public:
        using TResultModel = TObliviousTreeModel;
        using TWeakModelStructure = TObliviousTreeStructure;

        TFeatureParallelPointwiseObliviousTree(TBinarizedFeaturesManager& featuresManager,
                                               const NCatboostOptions::TBoostingOptions& boostingOptions,
                                               const NCatboostOptions::TCatBoostOptions& config,
                                               TGpuAwareRandom& random,
                                               bool makeZeroAverage = false)
            : FeaturesManager(featuresManager)
            , BoostingOptions(boostingOptions)
            , TreeConfig(config.ObliviousTreeOptions)
            , LossDescription(config.LossFunctionDescription.Get())
            , MakeZeroAverage(makeZeroAverage)
            , Random(random)
        {
        }

        template <class TDataSet>
        TFeatureParallelPointwiseObliviousTree& CacheStructure(TScopedCacheHolder& cacheHolder,
                                                               const TObliviousTreeStructure& model,
                                                               const TDataSet& dataSet) {
            const auto& bins = GetBinsForModel(cacheHolder, FeaturesManager, dataSet, model);
            Y_UNUSED(bins);
            return *this;
        }

        template <class TTarget,
                  class TDataSet>
        TFeatureParallelObliviousTreeSearcher CreateStructureSearcher(TScopedCacheHolder& cache,
                                                                      const TDataSet& dataSet,
                                                                      const TAdditiveModel<TResultModel>& result) {
            if (Bootstrap == nullptr) {
                const NCatboostOptions::TBootstrapConfig& bootstrapConfig = TreeConfig.BootstrapConfig;
                Bootstrap = MakeHolder<TBootstrap<NCudaLib::TMirrorMapping>>(bootstrapConfig, result.GetL1LeavesSum());
            } else {
                Bootstrap->Reset(result.GetL1LeavesSum());
            }
            CB_ENSURE(Bootstrap);

            return TFeatureParallelObliviousTreeSearcher(cache,
                                                         FeaturesManager,
                                                         BoostingOptions,
                                                         dataSet,
                                                         *Bootstrap,
                                                         TreeConfig,
                                                         Random);
        }

        TObliviousTreeLeavesEstimator CreateEstimator() {
            return TObliviousTreeLeavesEstimator(FeaturesManager,
                                                 CreateLeavesEstimationConfig(TreeConfig,
                                                                              MakeZeroAverage,
                                                                              LossDescription,
                                                                              BoostingOptions),
                                                 Random);
        }

        template <class TDataSet>
        TAddObliviousTreeFeatureParallel CreateAddModelValue(const TObliviousTreeStructure& structure,
                                                             TScopedCacheHolder& cache) {
            return TAddObliviousTreeFeatureParallel(cache,
                                                    FeaturesManager,
                                                    structure);
        }

    private:
        THolder<TBootstrap<NCudaLib::TMirrorMapping>> Bootstrap;
        TBinarizedFeaturesManager& FeaturesManager;
        const NCatboostOptions::TBoostingOptions& BoostingOptions;
        const NCatboostOptions::TObliviousTreeLearnerOptions& TreeConfig;
        const NCatboostOptions::TLossDescription& LossDescription;
        bool MakeZeroAverage = false;
        TGpuAwareRandom& Random;
    };
}
