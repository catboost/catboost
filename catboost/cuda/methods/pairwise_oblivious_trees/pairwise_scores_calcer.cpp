#include "pairwise_scores_calcer.h"

#include <util/generic/array_ref.h>
#include <util/generic/ymath.h>
#include <util/stream/labeled.h>

#include <limits>

static void ValidateSplits(
    const TConstArrayRef<TBestSplitPropertiesWithIndex> splits,
    const TConstArrayRef<NCatboostCuda::EFeaturesGroupingPolicy> policies,
    const ui32 deviceCount) {
    Y_ASSERT(splits.size() == policies.size() * deviceCount);

    for (size_t policyIdx = 0; policyIdx < policies.size(); ++policyIdx) {
        const auto policy = policies[policyIdx];
        for (ui32 deviceIdx = 0; deviceIdx < deviceCount; ++deviceIdx) {
            const auto split = splits[policyIdx * deviceCount + deviceIdx];
            const auto message = TStringBuilder()
                                 << "got invalid split ("
                                 << LabeledOutput(policy, policyIdx, deviceIdx, split.Index, split.FeatureId, split.BinId, split.Score)
                                 << "), this may be caused by anomalies in your data (e.g. your target absolute value is too big)"
                                 << " that cause numeric errors during training";
            CB_ENSURE(split.Index != std::numeric_limits<ui32>::max(), message);
            CB_ENSURE(split.FeatureId != std::numeric_limits<ui32>::max(), message);
            CB_ENSURE(split.BinId != std::numeric_limits<ui32>::max(), message);
            CB_ENSURE(IsValidFloat(split.Score), message);
        }
    }
}

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

    ValidateSplits(bestSplitCpu, policies, deviceCount);

    for (ui32 policyId = 0; policyId < policyCount; ++policyId) {
        bool found = false;
        for (ui32 dev = 0; dev < deviceCount; ++dev) {
            const auto& current = bestSplitCpu[policyCount * dev + policyId];
            if (current.Score < bestForPolicies[policyId].Score) {
                bestForPolicies[policyId] = current;
                found = true;
            }
        }

        const auto policy = policies[policyId];
        CB_ENSURE(found, "failed to find best score for " << LabeledOutput(policy, policyId));
    }

    TBestSplitPropertiesWithIndex bestSplit;
    EFeaturesGroupingPolicy bestPolicy;

    bool foundBestScoreAmongPolicies = false;
    for (ui32 policyId = 0; policyId < policyCount; ++policyId) {
        EFeaturesGroupingPolicy policy = policies[policyId];

        const auto& current = bestForPolicies[policyId];
        if (current.Score < bestSplit.Score) {
            bestSplit = current;
            bestPolicy = policy;
            foundBestScoreAmongPolicies = true;
        }
    }

    CB_ENSURE(foundBestScoreAmongPolicies);

    TBestSplitResult bestSplitResult;
    bestSplitResult.BestSplit = static_cast<TBestSplitProperties&>(bestSplit);

    if (needBestSolution) {
        bestSplitResult.Solution = MakeHolder<TVector<float>>();
        bestSplitResult.MatrixDiag = MakeHolder<TVector<float>>();
        Solutions[bestPolicy]->ReadBestSolution(bestSplit.Index,
                                                bestSplitResult.Solution.Get(),
                                                bestSplitResult.MatrixDiag.Get());
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
