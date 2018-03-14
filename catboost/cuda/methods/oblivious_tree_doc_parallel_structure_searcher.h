#pragma once

#include "pointwise_scores_calcer.h"
#include "bootstrap.h"
#include "helpers.h"
#include "tree_ctrs.h"
#include "tree_ctr_datasets_visitor.h"
#include "weak_target_helpers.h"
#include "pointiwise_optimization_subsets.h"
#include "feature_parallel_pointwise_oblivious_tree.h"
#include "random_score_helper.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/all_reduce.h>
#include <catboost/cuda/gpu_data/feature_parallel_dataset.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/gpu_data/oblivious_tree_bin_builder.h>
#include <catboost/cuda/models/add_bin_values.h>
#include <catboost/cuda/targets/target_func.h>
#include <catboost/cuda/cuda_util/run_stream_parallel_jobs.h>
#include <catboost/libs/options/oblivious_tree_options.h>

namespace NCatboostCuda {
    template <class TTarget,
              class TDataSet>
    class TDocParallelObliviousTreeSearcher {
    public:
        using TVec = typename TTarget::TVec;
        using TSampelsMapping = NCudaLib::TStripeMapping;
        using TWeakTarget = TL2Target<NCudaLib::TStripeMapping>;

        TDocParallelObliviousTreeSearcher(const TBinarizedFeaturesManager& featuresManager,
                                          const NCatboostOptions::TObliviousTreeLearnerOptions& learnerOptions,
                                          TBootstrap<NCudaLib::TStripeMapping>& bootstrap,
                                          double randomStrengthMult)
            : FeaturesManager(featuresManager)
            , TreeConfig(learnerOptions)
            , Bootstrap(bootstrap)
            , ModelLengthMultiplier(randomStrengthMult)
        {
        }

        TObliviousTreeModel Fit(const TDataSet& dataSet,
                                const TTarget& objective) {
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
                featuresScoreCalcer = new TScoreCalcer(dataSet.GetFeatures(),
                                                       TreeConfig,
                                                       1,
                                                       true);
            }
            if (dataSet.HasPermutationDependentFeatures()) {
                simpleCtrScoreCalcer = new TScoreCalcer(dataSet.GetPermutationFeatures(),
                                                        TreeConfig,
                                                        1,
                                                        true);
            }

            TObliviousTreeStructure structure;
            TVector<float> leaves;
            auto& profiler = NCudaLib::GetCudaManager().GetProfiler();

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
                            featuresScoreCalcer->ComputeOptimalSplit(reducedPartStats,
                                                                     scoreStdDevMult,
                                                                     random.NextUniformL());
                        }
                        if (simpleCtrScoreCalcer) {
                            simpleCtrScoreCalcer->ComputeOptimalSplit(reducedPartStats,
                                                                      scoreStdDevMult,
                                                                      random.NextUniformL());
                        }
                    }
                }

                TBestSplitProperties bestSplitProp = {static_cast<ui32>(-1),
                                                      0,
                                                      std::numeric_limits<float>::infinity()};

                if (featuresScoreCalcer) {
                    bestSplitProp = TakeBest(bestSplitProp, featuresScoreCalcer->ReadOptimalSplit());
                }

                if (simpleCtrScoreCalcer) {
                    bestSplitProp = TakeBest(bestSplitProp, simpleCtrScoreCalcer->ReadOptimalSplit());
                }

                CB_ENSURE(bestSplitProp.FeatureId != static_cast<ui32>(-1),
                          TStringBuilder() << "Error: something went wrong, best split is NaN with score"
                                           << bestSplitProp.Score);

                bestSplit = ToSplit(FeaturesManager, bestSplitProp);
                PrintBestScore(FeaturesManager, bestSplit, bestSplitProp.Score, depth);

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
                } else {
                    leaves.resize(1 << structure.Splits.size(), 0.0f);
                }
            }
            CB_ENSURE((1 << structure.Splits.size()) == leaves.size(), (1 << structure.Splits.size()) << " " << leaves.size());
            return TObliviousTreeModel(std::move(structure),
                                       leaves);
        }

    private:
        TVector<float> ReadAndEstimateLeaves(const TCudaBuffer<TPartitionStatistics, NCudaLib::TMirrorMapping>& parts) {
            TVector<TPartitionStatistics> statCpu;
            parts.Read(statCpu);
            return EstimateLeaves(statCpu);
        }

        TVector<float> EstimateLeaves(const TVector<TPartitionStatistics>& statCpu) {
            TVector<float> result;
            for (ui32 i = 0; i < statCpu.size(); ++i) {
                const float mu = statCpu[i].Count > 0 ? statCpu[i].Sum / (statCpu[i].Weight + TreeConfig.L2Reg) : 0;
                result.push_back(mu);
            }
            return result;
        }

        void ComputeWeakTarget(const TTarget& objective,
                               double* scoreStdDev,
                               TWeakTarget* target,
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
                Bootstrap.BootstrapAndFilter(target->WeightedTarget,
                                             target->Weights,
                                             *indices);
            }
        }

    private:
        const TBinarizedFeaturesManager& FeaturesManager;
        const NCatboostOptions::TObliviousTreeLearnerOptions& TreeConfig;
        TBootstrap<NCudaLib::TStripeMapping>& Bootstrap;
        double ModelLengthMultiplier = 0.0;
    };
}
