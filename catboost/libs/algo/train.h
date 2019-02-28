#pragma once

#include "learn_context.h"

#include <catboost/libs/data_new/data_provider.h>
#include <catboost/libs/overfitting_detector/error_tracker.h>

void TrainOneIteration(const NCB::TTrainingForCPUDataProviders& data, TLearnContext* ctx);

TErrorTracker BuildErrorTracker(EMetricBestValue bestValueType, double bestPossibleValue, bool hasTest, const TLearnContext& ctx);
