#pragma once

#include "bootstrap.h"
#include "helpers.h"
#include "oblivious_tree_doc_parallel_structure_searcher.h"
#include "catboost/cuda/models/add_oblivious_tree_model_doc_parallel.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/gpu_data/feature_parallel_dataset.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/gpu_data/oblivious_tree_bin_builder.h>
#include <catboost/cuda/models/add_bin_values.h>
#include <catboost/cuda/targets/target_func.h>

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
        TDocParallelObliviousTreeSearcher<TTarget, TDataSet> CreateStructureSearcher(double mult) {
            if (Bootstrap == nullptr) {
                Bootstrap.Reset(new TBootstrap<NCudaLib::TStripeMapping>(TreeConfig.BootstrapConfig, Seed));
            }

            return TDocParallelObliviousTreeSearcher<TTarget, TDataSet>(FeaturesManager,
                                                                        TreeConfig,
                                                                        *Bootstrap,
                                                                        mult);
        }

        TObliviousTreeLeavesEstimator CreateEstimator() {
            CB_ENSURE(NeedEstimation());
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
