#pragma once

#include "learn_context.h"
#include <catboost/libs/overfitting_detector/error_tracker.h>

using TTrainOneIterationFunc = std::function<void(const TTrainData& data,
                                                  TLearnContext* ctx)>;


TTrainOneIterationFunc GetOneIterationFunc(ELossFunction lossFunction);

TErrorTracker BuildErrorTracker(bool isMaxOptimal, bool hasTest, TLearnContext* ctx);
