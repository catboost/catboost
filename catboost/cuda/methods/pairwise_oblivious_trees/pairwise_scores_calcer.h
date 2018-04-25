#pragma once

#include "pairwise_score_calcer_for_policy.h"
#include <catboost/cuda/gpu_data/compressed_index.h>
#include <catboost/libs/options/oblivious_tree_options.h>
#include <catboost/cuda/methods/helpers.h>
#include <catboost/cuda/methods/pairwise_kernels.h>

namespace NCatboostCuda {
    inline THolder<TComputePairwiseScoresHelper> CreateScoreHelper(EFeaturesGroupingPolicy policy,
                                                                   const TCompressedDataSet<TDocParallelLayout>& dataSet,
                                                                   const TPairwiseOptimizationSubsets& subsets,
                                                                   const NCatboostOptions::TObliviousTreeLearnerOptions& treeConfig) {
        return new TComputePairwiseScoresHelper(policy,
                                                dataSet,
                                                subsets,
                                                treeConfig.MaxDepth,
                                                treeConfig.L2Reg,
                                                treeConfig.PairwiseNonDiagReg);
    };

    struct TBestSplitResult {
        TBestSplitProperties BestSplit;
        TAtomicSharedPtr<TVector<float>> Solution;
        bool operator<(const TBestSplitResult& other) const {
            return BestSplit.operator<(other.BestSplit);
        }
    };

    class TPairwiseScoreCalcer {
    public:
        using TLayoutPolicy = TDocParallelLayout;
        using TSamplesMapping = typename TLayoutPolicy::TSamplesMapping;

        TPairwiseScoreCalcer(const TCompressedDataSet<TLayoutPolicy>& features,
                             const NCatboostOptions::TObliviousTreeLearnerOptions& treeConfig,
                             const TPairwiseOptimizationSubsets& subsets,
                             bool storeSolverTempResults = false)
            : Features(features)
            , Subsets(subsets)
            , TreeConfig(treeConfig)
            , StoreTempResults(storeSolverTempResults)
        {
            for (auto policy : GetAllGroupingPolicies()) {
                if (Features.GetGridSize(policy)) {
                    Helpers[policy] = CreateScoreHelper(policy,
                                                        Features,
                                                        Subsets,
                                                        TreeConfig);
                }
            }
        }

        bool HasHelperForPolicy(EFeaturesGroupingPolicy policy) const {
            return Helpers.has(policy);
        }

        const TComputePairwiseScoresHelper& GetHelperForPolicy(EFeaturesGroupingPolicy policy) const {
            CB_ENSURE(Helpers.has(policy));
            return *Helpers.at(policy);
        }

        const TBinaryFeatureSplitResults& GetResultsForPolicy(EFeaturesGroupingPolicy policy) const {
            CB_ENSURE(Helpers.has(policy));
            return *Solutions.at(policy);
        }

        void Compute() {
            TScopedCacheHolder cacheHolder;

            for (auto& helper : Helpers) {
                Solutions[helper.first] = new TBinaryFeatureSplitResults;

                if (StoreTempResults) {
                    Solutions[helper.first]->LinearSystems = new TStripeBuffer<float>;
                    Solutions[helper.first]->SqrtMatrices = new TStripeBuffer<float>;
                }
            }

            for (auto& helper : Helpers) {
                helper.second->Compute(cacheHolder,
                                       Solutions[helper.first].Get());
            }
        }

        TBestSplitResult FindOptimalSplit(bool needBestSolution) {
            //first: write to one vector on remote side
            //second: read results in one operations
            //reduce latency for mulithost learning
            TStripeBuffer<TBestSplitPropertiesWithIndex> bestSplits;
            TVector<EFeaturesGroupingPolicy> policies;

            for (auto& helper : Helpers) {
                policies.push_back(helper.first);
            }

            const auto policyCount = policies.size();
            Y_VERIFY(policyCount);
            bestSplits.Reset(NCudaLib::TStripeMapping::RepeatOnAllDevices(policyCount));

            for (ui32 i = 0; i < policies.size(); ++i) {
                EFeaturesGroupingPolicy policy = policies[i];
                auto bestScoreSlice = NCudaLib::ParallelStripeView(bestSplits,
                                                                   TSlice(i, (i + 1)));
                SelectOptimalSplit(Solutions[policy]->Scores,
                                   Solutions[policy]->BinFeatures,
                                   bestScoreSlice);
            }

            TVector<TBestSplitPropertiesWithIndex> bestSplitCpu;
            bestSplits.Read(bestSplitCpu);
            const auto deviceCount = NCudaLib::GetCudaManager().GetDeviceCount();
            TVector<TBestSplitPropertiesWithIndex> bestForPolicies(policyCount);
            CB_ENSURE(bestSplitCpu.size() == policyCount * deviceCount);

            for (ui32 policyId = 0; policyId < policyCount; ++policyId) {
                for (ui32 dev = 0; dev < deviceCount; ++dev) {
                    const auto& current = bestSplitCpu[policyCount * dev + policyId];
                    if (current.Score < bestForPolicies[policyId].Score) {
                        bestForPolicies[policyId] = current;
                    }
                }
            }

            TBestSplitPropertiesWithIndex bestSplit;
            EFeaturesGroupingPolicy bestPolicy;

            for (ui32 policyId = 0; policyId < policyCount; ++policyId) {
                EFeaturesGroupingPolicy policy = policies[policyId];

                const auto& current = bestForPolicies[policyId];
                if (current.Score < bestSplit.Score) {
                    bestSplit = current;
                    bestPolicy = policy;
                }
            }

            TBestSplitResult bestSplitResult;
            bestSplitResult.BestSplit = static_cast<TBestSplitProperties&>(bestSplit);

            if (needBestSolution) {
                bestSplitResult.Solution = MakeHolder<TVector<float>>();
                Solutions[bestPolicy]->ReadBestSolution(bestSplit.Index,
                                                        bestSplitResult.Solution.Get());
            }
            return bestSplitResult;
        }

    private:
        const TCompressedDataSet<TLayoutPolicy>& Features;
        const TPairwiseOptimizationSubsets& Subsets;
        const NCatboostOptions::TObliviousTreeLearnerOptions& TreeConfig;
        bool StoreTempResults;

        using TScoreHelperPtr = THolder<TComputePairwiseScoresHelper>;
        using TResultPtr = THolder<TBinaryFeatureSplitResults>;

        TMap<EFeaturesGroupingPolicy, TScoreHelperPtr> Helpers;
        TMap<EFeaturesGroupingPolicy, TResultPtr> Solutions;
    };

}
