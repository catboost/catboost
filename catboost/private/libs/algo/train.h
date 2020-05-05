#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/overfitting_detector/error_tracker.h>

class TLearnContext;

void TrainOneIteration(const NCB::TTrainingDataProviders& data, TLearnContext* ctx);

TErrorTracker BuildErrorTracker(
    EMetricBestValue bestValueType,
    double bestPossibleValue,
    bool hasTest,
    const TLearnContext& ctx);
