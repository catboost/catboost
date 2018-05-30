#include "pairwise_scores_calcer.h"

NCatboostCuda::TBestSplitResult NCatboostCuda::TPairwiseScoreCalcer::FindOptimalSplit(bool needBestSolution) {
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

void NCatboostCuda::TPairwiseScoreCalcer::Compute() {
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
