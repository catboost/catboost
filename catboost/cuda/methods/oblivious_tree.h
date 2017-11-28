#pragma once

#include "bootstrap.h"
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

namespace NCatboostCuda
{
    class TObliviousTree
    {
    public:
        using TResultModel = TObliviousTreeModel;
        using TWeakModelStructure = TObliviousTreeStructure;

        TObliviousTree(TBinarizedFeaturesManager& featuresManager,
                       const NCatboostOptions::TObliviousTreeLearnerOptions& config,
                       ui64 randomSeed = 0)
                : FeaturesManager(featuresManager)
                  , TreeConfig(config)
                , RandomSeed(randomSeed)
        {
        }

        template<class TDataSet>
        TObliviousTree& CacheStructure(TScopedCacheHolder& cacheHolder,
                                       const TObliviousTreeStructure& model,
                                       const TDataSet& dataSet)
        {
            const auto& bins = GetBinsForModel(cacheHolder, FeaturesManager, dataSet, model);
            Y_UNUSED(bins);
            return *this;
        }

        template<class TTarget,
                class TDataSet>
        TObliviousTreeStructureSearcher<TTarget, TDataSet> CreateStructureSearcher(TScopedCacheHolder& cache,
                                                                                   const TDataSet& dataSet)
        {
            if (Bootstrap == nullptr)
            {
                const NCatboostOptions::TBootstrapConfig& bootstrapConfig = TreeConfig.BootstrapConfig;
                Bootstrap = MakeHolder<TBootstrap<NCudaLib::TMirrorMapping>>(dataSet.GetTarget().GetMapping(),
                                                                             bootstrapConfig,
                                                                             RandomSeed);
            }
            CB_ENSURE(Bootstrap);

            return TObliviousTreeStructureSearcher<TTarget, TDataSet>(cache,
                                                                      FeaturesManager,
                                                                      dataSet,
                                                                      *Bootstrap,
                                                                      TreeConfig);
        }

        template<template<class TMapping, class> class TTarget, class TDataSet>
        TObliviousTreeLeavesEstimator<TTarget, TDataSet> CreateEstimator(const TObliviousTreeStructure& structure,
                                                                         TScopedCacheHolder& cache)
        {
            return TObliviousTreeLeavesEstimator<TTarget, TDataSet>(structure,
                                                                    FeaturesManager,
                                                                    cache,
                                                                    TreeConfig.LeavesEstimationMethod == ELeavesEstimation::Newton,
                                                                    TreeConfig.L2Reg,
                                                                    TreeConfig.LeavesEstimationIterations,
                                                                    TreeConfig.FoldSizeLossNormalization,
                                                                    TreeConfig.AddRidgeToTargetFunctionFlag);
        }

        template<class TDataSet>
        TAddModelValue<TObliviousTreeModel, TDataSet> CreateAddModelValue(const TObliviousTreeStructure& structure,
                                                                          TScopedCacheHolder& cache)
        {
            return TAddModelValue<TObliviousTreeModel, TDataSet>(cache,
                                                                 FeaturesManager,
                                                                 structure);
        }

    private:
        THolder<TBootstrap<NCudaLib::TMirrorMapping>> Bootstrap;
        TBinarizedFeaturesManager& FeaturesManager;
        const NCatboostOptions::TObliviousTreeLearnerOptions& TreeConfig;
        ui64 RandomSeed;
    };
}

