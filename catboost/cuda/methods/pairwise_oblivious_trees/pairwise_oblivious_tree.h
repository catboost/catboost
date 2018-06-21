#pragma once

#include "non_diagonal_leaves_estimator.h"
#include "pairwise_structure_searcher.h"

#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/gpu_data/doc_parallel_dataset.h>
#include <catboost/libs/options/catboost_options.h>
#include <catboost/cuda/gpu_data/bootstrap.h>
#include <catboost/cuda/methods/leaves_estimation/oblivious_tree_leaves_estimator.h>

namespace NCatboostCuda {
    class TPairwiseObliviousTree {
    public:
        using TResultModel = TObliviousTreeModel;
        using TWeakModelStructure = TObliviousTreeStructure;
        using TDataSet = TDocParallelDataSet;

        TPairwiseObliviousTree(const TBinarizedFeaturesManager& featuresManager,
                               const NCatboostOptions::TCatBoostOptions& config,
                               bool zeroAverage)
            : FeaturesManager(featuresManager)
            , TreeConfig(config.ObliviousTreeOptions)
            , Seed(config.RandomSeed)
            , ZeroAverage(zeroAverage)
        {
        }

        bool NeedEstimation() const {
            return TreeConfig.LeavesEstimationMethod != ELeavesEstimation::Simple;
        }

        template <class TTarget,
                  class TDataSet>
        TPairwiseObliviousTreeSearcher CreateStructureSearcher(double) {
            return TPairwiseObliviousTreeSearcher(FeaturesManager,
                                                  TreeConfig);
        }

        TPairwiseObliviousTreeLeavesEstimator CreateEstimator() {
            CB_ENSURE(NeedEstimation());
            return TPairwiseObliviousTreeLeavesEstimator(CreateLeavesEstimationConfig(TreeConfig,
                                                                                      ZeroAverage));
        }

        template <class TDataSet>
        TAddDocParallelObliviousTree CreateAddModelValue(bool useStreams = false) {
            return TAddDocParallelObliviousTree(useStreams);
        }

    private:
        const TBinarizedFeaturesManager& FeaturesManager;
        const NCatboostOptions::TObliviousTreeLearnerOptions& TreeConfig;
        ui64 Seed;
        bool ZeroAverage;
    };
}
