#pragma once

#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/gpu_data/doc_parallel_dataset.h>
#include <catboost/libs/options/catboost_options.h>
#include <catboost/cuda/methods/bootstrap.h>
#include <catboost/cuda/methods/oblivious_tree_leaves_estimator.h>
#include <catboost/cuda/methods/pairwise_oblivious_tree_leaves_estimator.h>
#include "pairwise_structure_searcher.h"

namespace NCatboostCuda {
    class TPairwiseObliviousTree {
    public:
        using TResultModel = TObliviousTreeModel;
        using TWeakModelStructure = TObliviousTreeStructure;
        using TDataSet = TDocParallelDataSet;

        TPairwiseObliviousTree(const TBinarizedFeaturesManager& featuresManager,
                               const NCatboostOptions::TCatBoostOptions& config,
                               bool makeZeroAverage = true)
            : FeaturesManager(featuresManager)
            , TreeConfig(config.ObliviousTreeOptions)
            , Seed(config.RandomSeed)
            , MakeZeroAverage(makeZeroAverage)
        {
        }

        bool NeedEstimation() const {
            return TreeConfig.LeavesEstimationMethod != ELeavesEstimation::Simple;
        }

        template <class TTarget,
                  class TDataSet>
        TPairwiseObliviousTreeSearcher<TTarget, TDataSet> CreateStructureSearcher(double) {
            if (Bootstrap == nullptr) {
                Bootstrap.Reset(new TBootstrap<NCudaLib::TStripeMapping>(TreeConfig.BootstrapConfig,
                                                                         Seed));
            }

            return TPairwiseObliviousTreeSearcher<TTarget, TDataSet>(FeaturesManager,
                                                                     TreeConfig,
                                                                     *Bootstrap);
        }

        TPairwiseObliviousTreeLeavesEstimator CreateEstimator() {
            CB_ENSURE(NeedEstimation());
            return {};
        }

        template <class TDataSet>
        TAddModelValue<TObliviousTreeModel, TDataSet> CreateAddModelValue(bool useStreams = false) {
            return TAddModelValue<TObliviousTreeModel, TDataSet>(useStreams);
        }

    private:
        const TBinarizedFeaturesManager& FeaturesManager;
        const NCatboostOptions::TObliviousTreeLearnerOptions& TreeConfig;
        THolder<TBootstrap<NCudaLib::TStripeMapping>> Bootstrap;
        ui64 Seed;
        bool MakeZeroAverage = false;
    };
}
