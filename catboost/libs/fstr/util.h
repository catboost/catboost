#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/model/model.h>
#include <catboost/private/libs/options/loss_description.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/vector.h>

TVector<double> CollectLeavesStatistics(
    const NCB::TDataProvider& dataset,
    const TFullModel& model,
    NPar::TLocalExecutor* localExecutor);

bool TryGetLossDescription(const TFullModel& model, NCatboostOptions::TLossDescription& lossDescription);

void CheckNonZeroApproxForZeroWeightLeaf(const TFullModel& model);
