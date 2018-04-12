#pragma once

#include "pairwise_optimization_subsets.h"
#include "pairwise_scores_calcer.h"

#include <catboost/cuda/cuda_lib/mapping.h>
#include <catboost/libs/options/oblivious_tree_options.h>
#include <catboost/cuda/methods/bootstrap.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/methods/pointiwise_optimization_subsets.h>

namespace NCatboostCuda {
    template <class TTarget,
              class TDataSet>
    class TPairwiseObliviousTreeSearcher {
    public:
        using TVec = typename TTarget::TVec;
        using TSampelsMapping = NCudaLib::TStripeMapping;

        TPairwiseObliviousTreeSearcher(const TBinarizedFeaturesManager& featuresManager,
                                       const NCatboostOptions::TObliviousTreeLearnerOptions& learnerOptions,
                                       TBootstrap<NCudaLib::TStripeMapping>& bootstrap)
            : FeaturesManager(featuresManager)
            , TreeConfig(learnerOptions)
            , Bootstrap(bootstrap)
        {
        }

        TObliviousTreeModel Fit(const TDataSet& dataSet,
                                const TTarget& objective) {
            auto& random = objective.GetRandom();

            TPairwiseOptimizationSubsets subsets(ComputeWeakTarget(objective),
                                                 TreeConfig.MaxDepth);

            using TScoreCalcer = TPairwiseScoreCalcer;
            using TScoreCalcerPtr = THolder<TScoreCalcer>;

            TScoreCalcerPtr featuresScoreCalcer;
            TScoreCalcerPtr simpleCtrScoreCalcer;

            if (dataSet.HasFeatures()) {
                featuresScoreCalcer = new TScoreCalcer(dataSet.GetFeatures(),
                                                       TreeConfig,
                                                       subsets);
            }

            if (dataSet.HasPermutationDependentFeatures()) {
                simpleCtrScoreCalcer = new TScoreCalcer(dataSet.GetPermutationFeatures(),
                                                        TreeConfig,
                                                        subsets);
            }

            TObliviousTreeStructure structure;
            TVector<float> leaves;
            auto& profiler = NCudaLib::GetCudaManager().GetProfiler();

            for (ui32 depth = 0; depth < TreeConfig.MaxDepth; ++depth) {
                TBinarySplit bestSplit;
                {
                    auto guard = profiler.Profile(TStringBuilder() << "Compute best splits " << depth);
                    {
                        if (featuresScoreCalcer) {
                            featuresScoreCalcer->Compute();
                        }
                        if (simpleCtrScoreCalcer) {
                            simpleCtrScoreCalcer->Compute();
                        }
                    }
                }
                NCudaLib::GetCudaManager().Barrier();

                TBestSplitResult bestSplitProp;
                const bool needLeavesEstimation = TreeConfig.LeavesEstimationMethod == ELeavesEstimation::Simple;
                const bool readSolution = needLeavesEstimation && ((depth + 1) == TreeConfig.MaxDepth);

                if (featuresScoreCalcer) {
                    bestSplitProp = TakeBest(bestSplitProp,
                                             featuresScoreCalcer->FindOptimalSplit(readSolution));
                }

                if (simpleCtrScoreCalcer) {
                    bestSplitProp = TakeBest(bestSplitProp,
                                             simpleCtrScoreCalcer->FindOptimalSplit(readSolution));
                }

                CB_ENSURE(bestSplitProp.BestSplit.FeatureId != static_cast<ui32>(-1),
                          TStringBuilder() << "Error: something went wrong, best split is NaN with score" << bestSplitProp.BestSplit.Score);

                bestSplit = ToSplit(FeaturesManager, bestSplitProp.BestSplit);
                PrintBestScore(FeaturesManager, bestSplit, bestSplitProp.BestSplit.Score, depth);

                if (((depth + 1) != TreeConfig.MaxDepth)) {
                    auto guard = profiler.Profile(TStringBuilder() << "Update subsets");
                    subsets.Split(dataSet.GetCompressedIndex().GetStorage(),
                                  dataSet.GetTCFeature(bestSplit.FeatureId),
                                  bestSplit.BinIdx);
                }
                structure.Splits.push_back(bestSplit);

                if (((depth + 1) == TreeConfig.MaxDepth) && needLeavesEstimation) {
                    CB_ENSURE(bestSplitProp.Solution, "Solution should not be nullptr");
                    leaves = *bestSplitProp.Solution;
                    //we should swap last level one hot, because splits in solver are inverse
                    FixOneHot(structure.Splits, &leaves);
                } else {
                    leaves.resize(1ULL << structure.Splits.size(), 0.0f);
                }
            }
            return TObliviousTreeModel(std::move(structure),
                                       leaves);
        }

    private:

        void FixOneHot(const TVector<TBinarySplit>& splits, TVector<float>* leavesPtr) {
            auto& leaves = *leavesPtr;
            if (splits.back().SplitType != EBinSplitType::TakeBin) {
                return;
            }
            ui32 prevPartCount = 1 << (splits.size() - 1);
            for (ui32 leaf = 0; leaf < prevPartCount; ++leaf) {
                ui32 rightLeaf = leaf | prevPartCount;
                float tmp = leaves[leaf];
                leaves[leaf] = leaves[rightLeaf];
                leaves[rightLeaf] = tmp;
            }
            CB_ENSURE(false);
        }

        TPairwiseTarget ComputeWeakTarget(const TTarget& objective) {
            TPairwiseTarget target;
            auto& profiler = NCudaLib::GetProfiler();
            auto guard = profiler.Profile("Build randomized pairwise target");
            objective.ComputePairwiseTarget(Bootstrap,
                                            &target);
            return target;
        }

    private:
        const TBinarizedFeaturesManager& FeaturesManager;
        const NCatboostOptions::TObliviousTreeLearnerOptions& TreeConfig;
        TBootstrap<NCudaLib::TStripeMapping>& Bootstrap;
    };

}
