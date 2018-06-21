#pragma once

#include "pairwise_optimization_subsets.h"
#include "pairwise_scores_calcer.h"

#include <catboost/cuda/cuda_lib/mapping.h>
#include <catboost/libs/options/oblivious_tree_options.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/methods/pointiwise_optimization_subsets.h>
#include <catboost/cuda/gpu_data/bootstrap.h>

namespace NCatboostCuda {
    template <class TTarget,
              class TDataSet>
    class TPairwiseObliviousTreeSearcher {
    public:
        using TVec = typename TTarget::TVec;
        using TSampelsMapping = NCudaLib::TStripeMapping;

        TPairwiseObliviousTreeSearcher(const TBinarizedFeaturesManager& featuresManager,
                                       const NCatboostOptions::TObliviousTreeLearnerOptions& learnerOptions)
            : FeaturesManager(featuresManager)
            , TreeConfig(learnerOptions)
        {
        }

        TObliviousTreeModel Fit(const TDataSet& dataSet,
                                const TTarget& objective) {
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
            Y_VERIFY(featuresScoreCalcer != nullptr || simpleCtrScoreCalcer != nullptr);

            TObliviousTreeStructure structure;
            TVector<float> leaves;
            TVector<float> weights;
            auto& profiler = NCudaLib::GetCudaManager().GetProfiler();

            for (ui32 depth = 0; depth < TreeConfig.MaxDepth; ++depth) {
                NCudaLib::GetCudaManager().Barrier();

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

                if (((depth + 1) == TreeConfig.MaxDepth)) {
                    if (needLeavesEstimation) {
                        CB_ENSURE(bestSplitProp.Solution, "Solution should not be nullptr");
                        leaves = *bestSplitProp.Solution;
                        //we should swap last level one hot, because splits in solver are inverse
                        FixSolutionLeaveValuesLayout(structure.Splits, &leaves);
                    } else {
                        leaves.resize(1ULL << structure.Splits.size(), 0.0f);
                    }
                }
            }
            //TODO(noxoomo): support for weigths in pairwise mode
            weights.resize(leaves.size());
            return TObliviousTreeModel(std::move(structure),
                                       leaves,
                                       weights);
        }

    private:
        void FixSolutionLeaveValuesLayout(const TVector<TBinarySplit>& splits, TVector<float>* leavesPtr) {
            auto& solution = *leavesPtr;
            ui32 depth = IntLog2(solution.size());
            CB_ENSURE(depth > 0);
            const ui32 prevDepth = depth - 1;

            TVector<float> fixedLeaves(solution.size());
            const bool isLastLevelOneHot = splits.back().SplitType == EBinSplitType::TakeBin;
            const ui32 lastDepthBit = 1 << prevDepth;

            for (ui32 leaf = 0; leaf < (1 << prevDepth); ++leaf) {
                const ui32 solutionLeafLeft = 2 * leaf;
                const ui32 solutionLeafRight = 2 * leaf + 1;
                ui32 modelLeafLeft = leaf;
                ui32 modelLeafRight = leaf | lastDepthBit;
                if (isLastLevelOneHot) {
                    const ui32 tmp = modelLeafLeft;
                    modelLeafLeft = modelLeafRight;
                    modelLeafRight = tmp;
                }
                fixedLeaves[modelLeafLeft] = solution[solutionLeafLeft];
                fixedLeaves[modelLeafRight] = solution[solutionLeafRight];
            }
            solution.swap(fixedLeaves);
        }

        TNonDiagQuerywiseTargetDers ComputeWeakTarget(const TTarget& objective) {
            TNonDiagQuerywiseTargetDers target;
            auto& profiler = NCudaLib::GetProfiler();
            auto guard = profiler.Profile("Build randomized pairwise target");

            const bool isGradient = TreeConfig.ScoreFunction == EScoreFunction::L2;
            objective.ComputeStochasticDerivatives(TreeConfig.BootstrapConfig.Get(),
                                                   isGradient,
                                                   &target);

            Y_VERIFY(target.PairDer2OrWeights.GetObjectsSlice() == target.Pairs.GetObjectsSlice());
            MATRIXNET_DEBUG_LOG << "Pairs count " << target.PairDer2OrWeights.GetObjectsSlice().Size() << Endl;
            MATRIXNET_DEBUG_LOG << "Doc count " << target.Docs.GetObjectsSlice().Size() << Endl;
            return target;
        }

    private:
        const TBinarizedFeaturesManager& FeaturesManager;
        const NCatboostOptions::TObliviousTreeLearnerOptions& TreeConfig;
    };

}
