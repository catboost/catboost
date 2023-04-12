#pragma once

#include "helpers.h"

#include <catboost/cuda/models/add_oblivious_tree_model_doc_parallel.h>
#include <catboost/cuda/models/add_region_doc_parallel.h>
#include <catboost/cuda/models/add_non_symmetric_tree_doc_parallel.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/models/additive_model.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/models/region_model.h>
#include <catboost/cuda/models/non_symmetric_tree.h>
#include <catboost/cuda/methods/leaves_estimation/leaves_estimation_config.h>
#include <catboost/cuda/methods/greedy_subsets_searcher/structure_searcher_template.h>
#include <catboost/cuda/methods/leaves_estimation/doc_parallel_leaves_estimator.h>
#include <catboost/private/libs/options/catboost_options.h>
#include <catboost/private/libs/options/boosting_options.h>

namespace NCatboostCuda {
    inline TTreeStructureSearcherOptions MakeStructureSearcherOptions(
        const NCatboostOptions::TObliviousTreeLearnerOptions& config,
        ui32 featureCount
    ) {
        TTreeStructureSearcherOptions options;
        options.ScoreFunction = config.ScoreFunction;
        options.BootstrapOptions = config.BootstrapConfig;
        options.MaxDepth = config.MaxDepth;
        options.MinLeafSize = config.MinDataInLeaf;
        options.L2Reg = config.L2Reg;
        options.Policy = config.GrowPolicy;
        options.FixedBinarySplits = config.FixedBinarySplits;
        options.FeatureWeights = NCatboostOptions::ExpandFeatureWeights(config.FeaturePenalties.Get(), featureCount);
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
                               const NCatboostOptions::TBoostingOptions& boostingOptions,
                               const NCatboostOptions::TCatBoostOptions& config,
                               TGpuAwareRandom& random,
                               bool makeZeroAverage = false)
            : FeaturesManager(featuresManager)
            , BoostingOptions(boostingOptions)
            , TreeConfig(config.ObliviousTreeOptions)
            , LossDescription(config.LossFunctionDescription.Get())
            , StructureSearcherOptions(MakeStructureSearcherOptions(config.ObliviousTreeOptions, featuresManager.GetFeatureCount()))
            , MakeZeroAverage(makeZeroAverage)
            , ZeroLastBinInMulticlassHack(config.LossFunctionDescription->GetLossFunction() == ELossFunction::MultiClass)
            , Random(random)
        {
        }

        bool NeedEstimation() const {
            return TreeConfig.LeavesEstimationMethod != ELeavesEstimation::Simple;
        }

        template <class TTarget,
                  class TDataSet>
        TGreedyTreeLikeStructureSearcher<TTreeModel> CreateStructureSearcher(double randomStrengthMult, const TAdditiveModel<TResultModel>& /*result*/) {
            TTreeStructureSearcherOptions options = StructureSearcherOptions;
            Y_ASSERT(options.BootstrapOptions.GetBootstrapType() != EBootstrapType::MVS);
            options.RandomStrength *= randomStrengthMult;
            return TGreedyTreeLikeStructureSearcher<TTreeModel>(FeaturesManager, options);
        }

        TDocParallelLeavesEstimator CreateEstimator() {
            CB_ENSURE(NeedEstimation());
            return TDocParallelLeavesEstimator(CreateLeavesEstimationConfig(TreeConfig,
                                                                            MakeZeroAverage,
                                                                            LossDescription,
                                                                            BoostingOptions),
                                               Random);
        }

        template <class TDataSet>
        TAddModelDocParallel<TTreeModel> CreateAddModelValue(bool useStreams = false) {
            return TAddModelDocParallel<TTreeModel>(useStreams);
        }

    private:
        const TBinarizedFeaturesManager& FeaturesManager;
        const NCatboostOptions::TBoostingOptions& BoostingOptions;
        const NCatboostOptions::TObliviousTreeLearnerOptions& TreeConfig;
        const NCatboostOptions::TLossDescription& LossDescription;
        TTreeStructureSearcherOptions StructureSearcherOptions;
        bool MakeZeroAverage = false;
        bool ZeroLastBinInMulticlassHack = false;
        TGpuAwareRandom& Random;
    };
}
