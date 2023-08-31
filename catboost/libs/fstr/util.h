#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/model/model.h>
#include <catboost/private/libs/options/enums.h>
#include <catboost/private/libs/options/loss_description.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/vector.h>


struct TModelEstimatedFeature;


TVector<double> CollectLeavesStatistics(
    const NCB::TDataProvider& dataset,
    const TFullModel& model,
    NPar::ILocalExecutor* localExecutor);

bool TryGetLossDescription(const TFullModel& model, NCatboostOptions::TLossDescription* lossDescription);

bool TryGetObjectiveMetric(const TFullModel& model, NCatboostOptions::TLossDescription* lossDescription);

bool HasNonZeroApproxForZeroWeightLeaf(const TFullModel& model);

TVector<int> GetBinFeatureCombinationClassByDepth(
    const TModelTrees& forest,
    const TVector<int>& binFeatureCombinationClass,
    size_t treeIdx
);

EFeatureCalcerType GetEstimatedFeatureCalcerType(
    const TFullModel& model,
    const TModelEstimatedFeature& estimatedFeature
);
