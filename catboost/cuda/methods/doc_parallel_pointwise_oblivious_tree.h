#pragma once

#include "helpers.h"
#include "oblivious_tree_doc_parallel_structure_searcher.h"
#include "catboost/cuda/models/add_oblivious_tree_model_doc_parallel.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/methods/leaves_estimation/leaves_estimation_config.h>
#include <catboost/cuda/methods/leaves_estimation/oblivious_tree_leaves_estimator.h>
#include <catboost/private/libs/options/catboost_options.h>

namespace NCatboostCuda {
    class TDocParallelObliviousTree {
    public:
        using TResultModel = TObliviousTreeModel;
        using TWeakModelStructure = TObliviousTreeStructure;
        using TDataSet = TDocParallelDataSet;

        TDocParallelObliviousTree(const TBinarizedFeaturesManager& featuresManager,
                                  const NCatboostOptions::TCatBoostOptions& config,
                                  bool makeZeroAverage = false)
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
        TDocParallelObliviousTreeSearcher CreateStructureSearcher(double mult) {
            if (Bootstrap == nullptr) {
                Bootstrap.Reset(new TBootstrap<NCudaLib::TStripeMapping>(TreeConfig.BootstrapConfig));
            }

            return TDocParallelObliviousTreeSearcher(FeaturesManager,
                                                     TreeConfig,
                                                     *Bootstrap,
                                                     mult);
        }

        TObliviousTreeLeavesEstimator CreateEstimator() {
            CB_ENSURE(NeedEstimation());
            return TObliviousTreeLeavesEstimator(FeaturesManager,
                                                 CreateLeavesEstimationConfig(TreeConfig, MakeZeroAverage));
        }

        template <class TDataSet>
        TAddModelDocParallel<TObliviousTreeModel> CreateAddModelValue(bool useStreams = false) {
            return TAddModelDocParallel<TObliviousTreeModel>(useStreams);
        }

    private:
        const TBinarizedFeaturesManager& FeaturesManager;
        const NCatboostOptions::TObliviousTreeLearnerOptions& TreeConfig;
        THolder<TBootstrap<NCudaLib::TStripeMapping>> Bootstrap;
        ui64 Seed;
        bool MakeZeroAverage = false;
    };
}
