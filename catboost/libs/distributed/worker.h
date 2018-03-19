#pragma once

#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/options/catboost_options.h>

#include <util/generic/maybe.h>

void RunWorker(ui32 numThreads, ui32 nodePort);
void SetWorkerCustomObjective(const TMaybe<TCustomObjectiveDescriptor>& objective);
void SetWorkerParams(const NCatboostOptions::TCatBoostOptions& params);

