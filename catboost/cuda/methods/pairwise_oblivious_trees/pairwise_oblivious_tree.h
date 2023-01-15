#pragma once

#include "pairwise_structure_searcher.h"

#include <catboost/cuda/models/additive_model.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/gpu_data/doc_parallel_dataset.h>
#include <catboost/private/libs/options/catboost_options.h>
#include <catboost/cuda/gpu_data/bootstrap.h>
#include <catboost/cuda/methods/leaves_estimation/oblivious_tree_leaves_estimator.h>
#include <catboost/cuda/methods/leaves_estimation/doc_parallel_leaves_estimator.h>
#include <catboost/private/libs/options/boosting_options.h>

namespace NCatboostCuda {
    class TPairwiseObliviousTree {
    public:
        using TResultModel = TObliviousTreeModel;
        using TWeakModelStructure = TObliviousTreeStructure;
        using TDataSet = TDocParallelDataSet;

        TPairwiseObliviousTree(const TBinarizedFeaturesManager& featuresManager,
                               const NCatboostOptions::TBoostingOptions& boostingOptions,
                               const NCatboostOptions::TCatBoostOptions& config,
                               TGpuAwareRandom& random,
                               bool zeroAverage)
            : FeaturesManager(featuresManager)
            , BoostingOptions(boostingOptions)
            , TreeConfig(config.ObliviousTreeOptions)
            , LossDescription(config.LossFunctionDescription.Get())
            , Seed(config.RandomSeed)
            , ZeroAverage(zeroAverage)
            , Random(random)
        {
        }

        bool NeedEstimation() const {
            return TreeConfig.LeavesEstimationMethod != ELeavesEstimation::Simple;
        }

        template <class TTarget,
                  class TDataSet>
        TPairwiseObliviousTreeSearcher CreateStructureSearcher(double, const TAdditiveModel<TResultModel>& /*result*/) {
            return TPairwiseObliviousTreeSearcher(FeaturesManager,
                                                  TreeConfig);
        }

        TDocParallelLeavesEstimator CreateEstimator() {
            CB_ENSURE(NeedEstimation());
            return TDocParallelLeavesEstimator(CreateLeavesEstimationConfig(TreeConfig,
                                                                            ZeroAverage,
                                                                            LossDescription,
                                                                            BoostingOptions),
                                               Random);
        }

        template <class TDataSet>
        TAddModelDocParallel<TObliviousTreeModel> CreateAddModelValue(bool useStreams = false) {
            return TAddModelDocParallel<TObliviousTreeModel>(useStreams);
        }

    private:
        const TBinarizedFeaturesManager& FeaturesManager;
        const NCatboostOptions::TBoostingOptions& BoostingOptions;
        const NCatboostOptions::TObliviousTreeLearnerOptions& TreeConfig;
        const NCatboostOptions::TLossDescription& LossDescription;
        ui64 Seed;
        bool ZeroAverage;
        TGpuAwareRandom& Random;
    };
}
