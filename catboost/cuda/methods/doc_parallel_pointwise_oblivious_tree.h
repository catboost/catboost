#pragma once

#include "helpers.h"
#include "oblivious_tree_doc_parallel_structure_searcher.h"
#include "catboost/cuda/models/add_oblivious_tree_model_doc_parallel.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/models/additive_model.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/methods/leaves_estimation/leaves_estimation_config.h>
#include <catboost/cuda/methods/leaves_estimation/oblivious_tree_leaves_estimator.h>
#include <catboost/private/libs/options/catboost_options.h>
#include <catboost/private/libs/options/boosting_options.h>

namespace NCatboostCuda {
    class TDocParallelObliviousTree {
    public:
        using TResultModel = TObliviousTreeModel;
        using TWeakModelStructure = TObliviousTreeStructure;
        using TDataSet = TDocParallelDataSet;

        TDocParallelObliviousTree(const TBinarizedFeaturesManager& featuresManager,
                                  const NCatboostOptions::TBoostingOptions& boostingOptions,
                                  const NCatboostOptions::TCatBoostOptions& config,
                                  TGpuAwareRandom& random,
                                  bool makeZeroAverage = false)
            : FeaturesManager(featuresManager)
            , BoostingOptions(boostingOptions)
            , TreeConfig(config.ObliviousTreeOptions)
            , LossDescription(config.LossFunctionDescription.Get())
            , Seed(config.RandomSeed)
            , MakeZeroAverage(makeZeroAverage)
            , Random(random)
        {
        }

        bool NeedEstimation() const {
            return TreeConfig.LeavesEstimationMethod != ELeavesEstimation::Simple;
        }

        template <class TTarget,
                  class TDataSet>
        TDocParallelObliviousTreeSearcher CreateStructureSearcher(double mult, const TAdditiveModel<TResultModel>& result) {
            if (Bootstrap == nullptr) {
                Bootstrap.Reset(new TBootstrap<NCudaLib::TStripeMapping>(TreeConfig.BootstrapConfig, result.GetL1LeavesSum()));
            } else {
                Bootstrap->Reset(result.GetL1LeavesSum());
            }

            return TDocParallelObliviousTreeSearcher(FeaturesManager,
                                                     BoostingOptions,
                                                     TreeConfig,
                                                     *Bootstrap,
                                                     mult,
                                                     Random);
        }

        TObliviousTreeLeavesEstimator CreateEstimator() {
            CB_ENSURE(NeedEstimation());
            return TObliviousTreeLeavesEstimator(FeaturesManager,
                                                 CreateLeavesEstimationConfig(TreeConfig,
                                                                              MakeZeroAverage,
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
        THolder<TBootstrap<NCudaLib::TStripeMapping>> Bootstrap;
        ui64 Seed;
        bool MakeZeroAverage = false;
        TGpuAwareRandom& Random;
    };
}
