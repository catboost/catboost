#pragma once

#include "learn_context.h"
#include <catboost/libs/overfitting_detector/error_tracker.h>

using TTrainOneIterationFunc = std::function<void(const TDataset& learnData,
                                                  const TDataset* testData,
                                                  TLearnContext* ctx)>;

TTrainOneIterationFunc GetOneIterationFunc(ELossFunction lossFunction);

TErrorTracker BuildErrorTracker(EMetricBestValue bestValueType, float bestPossibleValue, bool hasTest, TLearnContext* ctx);
