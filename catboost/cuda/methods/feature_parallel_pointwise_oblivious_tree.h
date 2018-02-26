#pragma once

#include "bootstrap.h"
#include "helpers.h"
#include "oblivious_tree_structure_searcher.h"
#include "oblivious_tree_leaves_estimator.h"
#include "add_oblivious_tree_model_feature_parallel.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/gpu_data/feature_parallel_dataset.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/gpu_data/oblivious_tree_bin_builder.h>
#include <catboost/cuda/models/add_bin_values.h>
#include <catboost/cuda/targets/target_func.h>
#include <catboost/libs/options/catboost_options.h>

namespace NCatboostCuda {
    class TFeatureParallelPointwiseObliviousTree {
    public:
        using TResultModel = TObliviousTreeModel;
        using TWeakModelStructure = TObliviousTreeStructure;

        TFeatureParallelPointwiseObliviousTree(TBinarizedFeaturesManager& featuresManager,
                                               const NCatboostOptions::TCatBoostOptions& config,
                                               bool makeZeroAverage = false)
            : FeaturesManager(featuresManager)
            , TreeConfig(config.ObliviousTreeOptions)
            , RandomSeed(config.RandomSeed)
            , MakeZeroAverage(makeZeroAverage)
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
        TFeatureParallelObliviousTreeSearcher<TTarget, TDataSet> CreateStructureSearcher(TScopedCacheHolder& cache,
                                                                                         const TDataSet& dataSet) {
            if (Bootstrap == nullptr) {
                const NCatboostOptions::TBootstrapConfig& bootstrapConfig = TreeConfig.BootstrapConfig;
                Bootstrap = MakeHolder<TBootstrap<NCudaLib::TMirrorMapping>>(bootstrapConfig,
                                                                             RandomSeed);
            }
            CB_ENSURE(Bootstrap);

            return TFeatureParallelObliviousTreeSearcher<TTarget, TDataSet>(cache,
                                                                            FeaturesManager,
                                                                            dataSet,
                                                                            *Bootstrap,
                                                                            TreeConfig);
        }

        TObliviousTreeLeavesEstimator CreateEstimator() {
            return TObliviousTreeLeavesEstimator(FeaturesManager,
                                                 TLeavesEstimationConfig(TreeConfig.LeavesEstimationMethod == ELeavesEstimation::Newton,
                                                                         TreeConfig.L2Reg,
                                                                         TreeConfig.LeavesEstimationIterations,
                                                                         1e-20,
                                                                         TreeConfig.FoldSizeLossNormalization,
                                                                         TreeConfig.AddRidgeToTargetFunctionFlag,
                                                                         MakeZeroAverage));
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
        const NCatboostOptions::TObliviousTreeLearnerOptions& TreeConfig;
        ui64 RandomSeed;
        bool MakeZeroAverage = false;
    };
}
