#include "oblivious_tree_doc_parallel_structure_searcher.h"

#include "helpers.h"
#include "pointwise_scores_calcer.h"
#include "random_score_helper.h"
#include "update_feature_weights.h"

#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/all_reduce.h>
#include <catboost/cuda/methods/langevin_utils.h>

namespace NCatboostCuda {
    TObliviousTreeModel
    TDocParallelObliviousTreeSearcher::FitImpl(const TDocParallelObliviousTreeSearcher::TDataSet& dataSet,
                                               const IStripeTargetWrapper& objective) {
        auto& random = objective.GetRandom();

        TWeakTarget target;
        TStripeBuffer<ui32> observations;
        TStripeBuffer<ui32> groupedByBinObservations;
        double scoreStdDevMult = 0;

        ComputeWeakTarget(objective,
                          &scoreStdDevMult,
                          &target,
                          &observations);

        groupedByBinObservations = TStripeBuffer<ui32>::CopyMapping(observations);

        auto subsets = TSubsetsHelper<NCudaLib::TStripeMapping>::CreateSubsets(TreeConfig.MaxDepth,
                                                                               target);

        using TScoreCalcer = TScoresCalcerOnCompressedDataSet<TDocParallelLayout>;
        using TScoreCalcerPtr = THolder<TScoreCalcer>;
        TScoreCalcerPtr featuresScoreCalcer;
        TScoreCalcerPtr simpleCtrScoreCalcer;

        if (dataSet.HasFeatures()) {
            featuresScoreCalcer = MakeHolder<TScoreCalcer>(dataSet.GetFeatures(),
                                                   TreeConfig,
                                                   1,
                                                   true);
        }
        if (dataSet.HasPermutationDependentFeatures()) {
            simpleCtrScoreCalcer = MakeHolder<TScoreCalcer>(dataSet.GetPermutationFeatures(),
                                                    TreeConfig,
                                                    1,
                                                    true);
        }

        TObliviousTreeStructure structure;
        TVector<float> leaves;
        TVector<double> weights;
        auto& profiler = NCudaLib::GetCudaManager().GetProfiler();

        TMirrorBuffer<float> catFeatureWeights;

        const auto featureCount = FeaturesManager.GetFeatureCount();
        const auto& featureWeightsCpu = ExpandFeatureWeights(TreeConfig.FeaturePenalties.Get(), featureCount);
        TMirrorBuffer<float> featureWeights = TMirrorBuffer<float>::Create(NCudaLib::TMirrorMapping(featureCount));
        featureWeights.Write(featureWeightsCpu);
        double scoreBeforeSplit = 0.0;

        for (ui32 depth = 0; depth < TreeConfig.MaxDepth; ++depth) {
            {
                auto guard = profiler.Profile("Gather observation indices");
                Gather(groupedByBinObservations, observations, subsets.Indices);
            }
            const TStripeBuffer<TPartitionStatistics>& partitionsStatsStriped = subsets.PartitionStats;
            TMirrorBuffer<TPartitionStatistics> reducedPartStats;
            NCudaLib::AllReduceThroughMaster(partitionsStatsStriped, reducedPartStats);
            //all reduce through master will implicitly sync
            //                manager.WaitComplete();

            UpdateFeatureWeightsForBestSplits(FeaturesManager, TreeConfig.ModelSizeReg, catFeatureWeights);

            TBinarySplit bestSplit;
            {
                auto guard = profiler.Profile(TStringBuilder() << "Compute best splits " << depth);
                {
                    if (featuresScoreCalcer) {
                        featuresScoreCalcer->SubmitCompute(subsets,
                                                           groupedByBinObservations);
                    }
                    if (simpleCtrScoreCalcer) {
                        simpleCtrScoreCalcer->SubmitCompute(subsets,
                                                            groupedByBinObservations);
                    }
                }
                {
                    if (featuresScoreCalcer) {
                        featuresScoreCalcer->ComputeOptimalSplit(reducedPartStats.AsConstBuf(),
                                                                 catFeatureWeights.AsConstBuf(),
                                                                 featureWeights.AsConstBuf(),
                                                                 scoreBeforeSplit,
                                                                 scoreStdDevMult,
                                                                 random.NextUniformL());
                    }
                    if (simpleCtrScoreCalcer) {
                        simpleCtrScoreCalcer->ComputeOptimalSplit(reducedPartStats.AsConstBuf(),
                                                                  catFeatureWeights.AsConstBuf(),
                                                                  featureWeights.AsConstBuf(),
                                                                  scoreBeforeSplit,
                                                                  scoreStdDevMult,
                                                                  random.NextUniformL());
                    }
                }
            }

            TBestSplitProperties bestSplitProp = {static_cast<ui32>(-1),
                                                  0,
                                                  std::numeric_limits<float>::infinity(),
                                                  std::numeric_limits<float>::infinity()};

            if (featuresScoreCalcer) {
                bestSplitProp = TakeBest(bestSplitProp, featuresScoreCalcer->ReadOptimalSplit());
            }

            if (simpleCtrScoreCalcer) {
                bestSplitProp = TakeBest(bestSplitProp, simpleCtrScoreCalcer->ReadOptimalSplit());
            }
            scoreBeforeSplit = bestSplitProp.Score;

            CB_ENSURE(bestSplitProp.FeatureId != static_cast<ui32>(-1),
                      TStringBuilder() << "Error: something went wrong, best split is NaN with score"
                                       << bestSplitProp.Score);

            bestSplit = ToSplit(FeaturesManager, bestSplitProp);
            PrintBestScore(FeaturesManager, bestSplit, bestSplitProp.Score, depth);
            if (FeaturesManager.IsCtr(bestSplitProp.FeatureId)) {
                FeaturesManager.AddUsedCtr(bestSplitProp.FeatureId);
            }

            const bool needLeavesEstimation = TreeConfig.LeavesEstimationMethod == ELeavesEstimation::Simple;

            if (structure.HasSplit(bestSplit)) {
                leaves = ReadAndEstimateLeaves(reducedPartStats);
                break;
            }

            if (((depth + 1) != TreeConfig.MaxDepth) || needLeavesEstimation) {
                auto guard = profiler.Profile(TStringBuilder() << "Update subsets");
                TSubsetsHelper<NCudaLib::TStripeMapping>::Split(target,
                                                                dataSet.GetCompressedIndex().GetStorage(),
                                                                groupedByBinObservations,
                                                                dataSet.GetTCFeature(bestSplit.FeatureId),
                                                                bestSplit.BinIdx,
                                                                &subsets);
            }
            structure.Splits.push_back(bestSplit);

            if (((depth + 1) == TreeConfig.MaxDepth) && needLeavesEstimation) {
                auto partitionsStats = ReadReduce(subsets.PartitionStats);
                leaves = EstimateLeaves(partitionsStats);
                weights = ExtractWeights(partitionsStats);
            } else {
                leaves.resize(1ULL << structure.Splits.size(), 0.0f);
                weights.resize(1ULL << structure.Splits.size(), 0.0f);
            }
        }
        CB_ENSURE((1ULL << structure.Splits.size()) == leaves.size(), (1ULL << structure.Splits.size()) << " " << leaves.size());
        return TObliviousTreeModel(std::move(structure),
                                   leaves,
                                   weights,
                                   1);
    }

    TVector<float> TDocParallelObliviousTreeSearcher::ReadAndEstimateLeaves(
        const TCudaBuffer<TPartitionStatistics, NCudaLib::TMirrorMapping>& parts) {
        TVector<TPartitionStatistics> statCpu;
        parts.Read(statCpu);
        return EstimateLeaves(statCpu);
    }

    TVector<float> TDocParallelObliviousTreeSearcher::EstimateLeaves(const TVector<TPartitionStatistics>& statCpu) {
        TVector<float> result;
        for (ui32 i = 0; i < statCpu.size(); ++i) {
            const float mu = statCpu[i].Count > 0 ? statCpu[i].Sum / (statCpu[i].Weight + TreeConfig.L2Reg) : 0;
            result.push_back(mu);
        }
        return result;
    }

    TVector<double> TDocParallelObliviousTreeSearcher::ExtractWeights(const TVector<TPartitionStatistics>& statCpu) {
        TVector<double> result;
        for (ui32 i = 0; i < statCpu.size(); ++i) {
            result.push_back(statCpu[i].Weight);
        }
        return result;
    }

    void TDocParallelObliviousTreeSearcher::ComputeWeakTarget(const IStripeTargetWrapper& objective, double* scoreStdDev,
                                                              TDocParallelObliviousTreeSearcher::TWeakTarget* target,
                                                              TStripeBuffer<ui32>* indices) {
        auto& profiler = NCudaLib::GetProfiler();
        auto guard = profiler.Profile("Build tree search target (gradient)");
        const bool secondOrder = IsSecondOrderScoreFunction(TreeConfig.ScoreFunction);

        target->WeightedTarget.Reset(objective.GetTarget().GetSamplesMapping());
        target->Weights.Reset(objective.GetTarget().GetSamplesMapping());

        if (secondOrder) {
            objective.NewtonAtZero(target->WeightedTarget,
                                   target->Weights);
        } else {
            objective.GradientAtZero(target->WeightedTarget,
                                     target->Weights);
        }

        (*scoreStdDev) = ComputeScoreStdDev(ModelLengthMultiplier,
                                            TreeConfig.RandomStrength,
                                            *target);

        indices->Reset(target->WeightedTarget.GetMapping());
        objective.GetTarget().WriteIndices(*indices);
        {
            auto bootstrapGuard = profiler.Profile("Bootstrap target");
            Bootstrap.BootstrapAndFilter(objective.GetRandom(),
                                         target->WeightedTarget,
                                         target->Weights,
                                         *indices);

            if (BoostingOptions.Langevin) {
                auto &seeds = Random.GetGpuSeeds<NCudaLib::TStripeMapping>();
                AddLangevinNoise(seeds,
                                 &(target->WeightedTarget),
                                 BoostingOptions.DiffusionTemperature,
                                 BoostingOptions.LearningRate);
            }
        }
    }
}
