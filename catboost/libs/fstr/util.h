#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/private/libs/options/enums.h>

#include <util/generic/vector.h>


namespace NPar {
    class ILocalExecutor;
}

class TFullModel;
struct TModelEstimatedFeature;
struct TModelTrees;

namespace NCatboostOptions {
    class TLossDescription;
}


TVector<double> CollectLeavesStatistics(
    const NCB::TDataProvider& dataset,
    const TFullModel& model,
    NPar::ILocalExecutor* localExecutor);

bool TryGetLossDescription(const TFullModel& model, NCatboostOptions::TLossDescription* lossDescription);

bool TryGetObjectiveMetric(const TFullModel& model, NCatboostOptions::TLossDescription* lossDescription);

bool HasNonZeroApproxForZeroWeightLeaf(const TFullModel& model);

bool NeedDatasetForLeavesWeights(const TFullModel& model, bool fstrOnTrainPool);

TVector<int> GetBinFeatureCombinationClassByDepth(
    const TModelTrees& forest,
    const TVector<int>& binFeatureCombinationClass,
    size_t treeIdx
);

EFeatureCalcerType GetEstimatedFeatureCalcerType(
    const TFullModel& model,
    const TModelEstimatedFeature& estimatedFeature
);
