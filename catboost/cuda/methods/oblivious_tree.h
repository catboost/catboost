#pragma once

#include "bootstrap.h"
#include "oblivious_tree_options.h"
#include "helpers.h"
#include "oblivious_tree_structure_searcher.h"
#include "oblivious_tree_leaves_estimator.h"
#include "add_oblivious_tree_model.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/gpu_data/fold_based_dataset.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/gpu_data/oblivious_tree_bin_builder.h>
#include <catboost/cuda/models/add_bin_values.h>
#include <catboost/cuda/targets/target_base.h>

class TObliviousTree {
public:
    using TResultModel = TObliviousTreeModel;
    using TWeakModelStructure = TObliviousTreeStructure;

    TObliviousTree(TBinarizedFeaturesManager& featuresManager,
                   const TObliviousTreeLearnerOptions& config)
        : FeaturesManager(featuresManager)
        , TreeConfig(config)
    {
    }

    template <class TDataSet>
    TObliviousTree& CacheStructure(TScopedCacheHolder& cacheHolder,
                                   const TObliviousTreeStructure& model,
                                   const TDataSet& dataSet) {
        const auto& bins = GetBinsForModel(cacheHolder, FeaturesManager, dataSet, model);
        Y_UNUSED(bins);
        return *this;
    }

    template <class TTarget,
              class TDataSet>
    TObliviousTreeStructureSearcher<TTarget, TDataSet> CreateStructureSearcher(TScopedCacheHolder& cache,
                                                                               const TDataSet& dataSet) {
        if (Bootstrap == nullptr) {
            auto& bootstrapConfig = TreeConfig.GetBootstrapConfig();
            Bootstrap = MakeHolder<TBootstrap<NCudaLib::TMirrorMapping>>(dataSet.GetTarget().GetMapping(),
                                                                         bootstrapConfig);
        }
        CB_ENSURE(Bootstrap);

        return TObliviousTreeStructureSearcher<TTarget, TDataSet>(cache,
                                                                  FeaturesManager,
                                                                  dataSet,
                                                                  *Bootstrap,
                                                                  TreeConfig);
    }

    template <template <class TMapping, class> class TTarget, class TDataSet>
    TObliviousTreeLeavesEstimator<TTarget, TDataSet> CreateEstimator(const TObliviousTreeStructure& structure,
                                                                     TScopedCacheHolder& cache) {
        return TObliviousTreeLeavesEstimator<TTarget, TDataSet>(structure,
                                                                FeaturesManager,
                                                                cache,
                                                                TreeConfig.IsUseNewton(),
                                                                TreeConfig.GetL2Reg(),
                                                                TreeConfig.GetLeavesEstimationIters(),
                                                                TreeConfig.IsNormalize(),
                                                                TreeConfig.AddRidgeToTargetFunction());
    }

    template <class TDataSet>
    TAddModelValue<TObliviousTreeModel, TDataSet> CreateAddModelValue(const TObliviousTreeStructure& structure,
                                                                      TScopedCacheHolder& cache) {
        return TAddModelValue<TObliviousTreeModel, TDataSet>(cache,
                                                             FeaturesManager,
                                                             structure);
    }

private:
    THolder<TBootstrap<NCudaLib::TMirrorMapping>> Bootstrap;
    TBinarizedFeaturesManager& FeaturesManager;
    TObliviousTreeLearnerOptions TreeConfig;
};
