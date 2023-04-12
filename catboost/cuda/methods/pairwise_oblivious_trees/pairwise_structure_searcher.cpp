#include "pairwise_structure_searcher.h"
#include "pairwise_scores_calcer.h"

#include <catboost/libs/helpers/math_utils.h>

namespace NCatboostCuda {
    void TPairwiseObliviousTreeSearcher::FixSolutionLeavesValuesLayout(const TVector<TBinarySplit>& splits,
                                                                       TVector<float>* leavesPtr,
                                                                       TVector<double>* weightsPtr) {
        auto& solution = *leavesPtr;
        ui32 depth = NCB::IntLog2(solution.size());
        CB_ENSURE(depth > 0);
        const ui32 prevDepth = depth - 1;

        TVector<float> fixedLeaves(solution.size());
        TVector<double> fixedWeights(solution.size());
        const bool isLastLevelOneHot = splits.back().SplitType == EBinSplitType::TakeBin;
        const ui32 lastDepthBit = 1U << prevDepth;

        for (ui32 leaf = 0; leaf < (1U << prevDepth); ++leaf) {
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

            fixedWeights[modelLeafLeft] = (*weightsPtr)[solutionLeafLeft];
            fixedWeights[modelLeafRight] = (*weightsPtr)[solutionLeafRight];
        }
        solution.swap(fixedLeaves);
        weightsPtr->swap(fixedWeights);
    }

    TNonDiagQuerywiseTargetDers TPairwiseObliviousTreeSearcher::ComputeWeakTarget(const IPairwiseTargetWrapper& objective) {
        TNonDiagQuerywiseTargetDers target;
        auto& profiler = NCudaLib::GetProfiler();
        auto guard = profiler.Profile("Build randomized pairwise target");

        const bool isGradient = TreeConfig.ScoreFunction == EScoreFunction::L2;
        objective.ComputeStochasticDerivatives(TreeConfig.BootstrapConfig.Get(),
                                               isGradient,
                                               &target);

        CB_ENSURE(
            target.PairDer2OrWeights.GetObjectsSlice() == target.Pairs.GetObjectsSlice(),
            "Slices of pairs and pair weight/derivatives should have same size");
        CATBOOST_DEBUG_LOG << "Pairs count " << target.PairDer2OrWeights.GetObjectsSlice().Size() << Endl;
        CATBOOST_DEBUG_LOG << "Doc count " << target.Docs.GetObjectsSlice().Size() << Endl;
        return target;
    }

    TObliviousTreeModel TPairwiseObliviousTreeSearcher::FitImpl(const TPairwiseObliviousTreeSearcher::TDataSet& dataSet,
                                                                const IPairwiseTargetWrapper& objective) {
        TPairwiseOptimizationSubsets subsets(ComputeWeakTarget(objective),
                                             TreeConfig.MaxDepth);

        using TScoreCalcer = TPairwiseScoreCalcer;
        using TScoreCalcerPtr = THolder<TScoreCalcer>;

        TScoreCalcerPtr featuresScoreCalcer;
        TScoreCalcerPtr simpleCtrScoreCalcer;

        if (dataSet.HasFeatures()) {
            featuresScoreCalcer = MakeHolder<TScoreCalcer>(dataSet.GetFeatures(),
                                                   TreeConfig,
                                                   subsets,
                                                   objective.GetRandom());
        }

        if (dataSet.HasPermutationDependentFeatures()) {
            simpleCtrScoreCalcer = MakeHolder<TScoreCalcer>(dataSet.GetPermutationFeatures(),
                                                    TreeConfig,
                                                    subsets,
                                                    objective.GetRandom());
        }
        CB_ENSURE(featuresScoreCalcer != nullptr || simpleCtrScoreCalcer != nullptr, "Need a score calcer");

        TObliviousTreeStructure structure;
        TVector<float> leaves;
        TVector<double> weights;
        auto& profiler = NCudaLib::GetCudaManager().GetProfiler();
        double scoreBeforeSplit = 0;

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
                                         featuresScoreCalcer->FindOptimalSplit(readSolution, scoreBeforeSplit));
            }

            if (simpleCtrScoreCalcer) {
                bestSplitProp = TakeBest(bestSplitProp,
                                         simpleCtrScoreCalcer->FindOptimalSplit(readSolution, scoreBeforeSplit));
            }

            CB_ENSURE(bestSplitProp.BestSplit.FeatureId != static_cast<ui32>(-1),
                      TStringBuilder() << "Error: something went wrong, best split is NaN with score" << bestSplitProp.BestSplit.Score);

            bestSplit = ToSplit(FeaturesManager, bestSplitProp.BestSplit);
            PrintBestScore(FeaturesManager, bestSplit, bestSplitProp.BestSplit.Score, depth);
            scoreBeforeSplit = bestSplitProp.BestSplit.Score;

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
                    weights.resize(bestSplitProp.MatrixDiag->size());
                    for (ui32 i = 0; i < weights.size(); ++i) {
                        weights[i] = (*bestSplitProp.MatrixDiag)[i];
                    }
                    //we should swap last level one hot, because splits in solver are inverse
                    FixSolutionLeavesValuesLayout(structure.Splits, &leaves, &weights);
                } else {
                    leaves.resize(1ULL << structure.Splits.size(), 0.0f);
                    weights.resize(1ULL << structure.Splits.size(), 0.0f);
                }
            }
        }
        return TObliviousTreeModel(std::move(structure),
                                   leaves,
                                   weights,
                                   1);
    }
}
