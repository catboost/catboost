#pragma once

#include "helpers.h"

#include <catboost/cuda/models/add_oblivious_tree_model_doc_parallel.h>
#include <catboost/cuda/models/add_region_doc_parallel.h>
#include <catboost/cuda/models/add_non_symmetric_tree_doc_parallel.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/models/region_model.h>
#include <catboost/cuda/models/non_symmetric_tree.h>
#include <catboost/cuda/methods/leaves_estimation/leaves_estimation_config.h>
#include <catboost/cuda/methods/greedy_subsets_searcher/structure_searcher_template.h>
#include <catboost/cuda/methods/leaves_estimation/doc_parallel_leaves_estimator.h>
#include <catboost/private/libs/options/catboost_options.h>

namespace NCatboostCuda {
    inline TTreeStructureSearcherOptions MakeStructureSearcherOptions(const NCatboostOptions::TObliviousTreeLearnerOptions& config) {
        TTreeStructureSearcherOptions options;
        options.ScoreFunction = config.ScoreFunction;
        options.BootstrapOptions = config.BootstrapConfig;
        options.MaxDepth = config.MaxDepth;
        options.MinLeafSize = config.MinDataInLeaf;
        options.L2Reg = config.L2Reg;
        options.Policy = config.GrowPolicy;
        if (config.GrowPolicy == EGrowPolicy::Region) {
            options.MaxLeaves = config.MaxDepth + 1;
        } else {
            options.MaxLeaves = config.MaxLeaves;
        }

        options.RandomStrength = config.RandomStrength;
        return options;
    }

    template <class TTreeModel>
    class TGreedySubsetsSearcher {
    public:
        using TResultModel = TTreeModel;
        using TDataSet = TDocParallelDataSet;

        TGreedySubsetsSearcher(const TBinarizedFeaturesManager& featuresManager,
                               const NCatboostOptions::TCatBoostOptions& config,
                               bool makeZeroAverage = false)
            : FeaturesManager(featuresManager)
            , TreeConfig(config.ObliviousTreeOptions)
            , StructureSearcherOptions(MakeStructureSearcherOptions(config.ObliviousTreeOptions))
            , MakeZeroAverage(makeZeroAverage)
            , ZeroLastBinInMulticlassHack(config.LossFunctionDescription->GetLossFunction() == ELossFunction::MultiClass)
        {
        }

        bool NeedEstimation() const {
            return TreeConfig.LeavesEstimationMethod != ELeavesEstimation::Simple;
        }

        template <class TTarget,
                  class TDataSet>
        TGreedyTreeLikeStructureSearcher<TTreeModel> CreateStructureSearcher(double randomStrengthMult) {
            TTreeStructureSearcherOptions options = StructureSearcherOptions;
            options.RandomStrength *= randomStrengthMult;
            return TGreedyTreeLikeStructureSearcher<TTreeModel>(FeaturesManager, options);
        }

        TDocParallelLeavesEstimator CreateEstimator() {
            CB_ENSURE(NeedEstimation());
            return TDocParallelLeavesEstimator(CreateLeavesEstimationConfig(TreeConfig, MakeZeroAverage));
        }

        template <class TDataSet>
        TAddModelDocParallel<TTreeModel> CreateAddModelValue(bool useStreams = false) {
            return TAddModelDocParallel<TTreeModel>(useStreams);
        }

    private:
        const TBinarizedFeaturesManager& FeaturesManager;
        const NCatboostOptions::TObliviousTreeLearnerOptions& TreeConfig;
        TTreeStructureSearcherOptions StructureSearcherOptions;
        bool MakeZeroAverage = false;
        bool ZeroLastBinInMulticlassHack = false;
    };
}
