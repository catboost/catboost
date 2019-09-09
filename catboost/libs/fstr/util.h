#pragma once

#include <catboost/libs/data_new/data_provider.h>
#include <catboost/libs/model/model.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/vector.h>

TVector<double> CollectLeavesStatistics(
    const NCB::TDataProvider& dataset,
    const TFullModel& model,
    NPar::TLocalExecutor* localExecutor);
