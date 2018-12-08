#pragma once

#include "learn_context.h"

#include <catboost/libs/data_new/data_provider.h>
#include <catboost/libs/overfitting_detector/error_tracker.h>

using TTrainOneIterationFunc = std::function<void(const NCB::TTrainingForCPUDataProviders& data,
                                                  TLearnContext* ctx)>;

TTrainOneIterationFunc GetOneIterationFunc(ELossFunction lossFunction);

TErrorTracker BuildErrorTracker(EMetricBestValue bestValueType, double bestPossibleValue, bool hasTest, TLearnContext* ctx);
