#pragma once

#include "learn_context.h"

#include <catboost/libs/metrics/metric.h>
#include <catboost/private/libs/options/enums.h>

#include <util/generic/ptr.h>

void CreateBacktrackingObjective(
    const TLearnContext& ctx,
    bool* haveBacktrackingObjective,
    double* minimizationSign,
    TVector<THolder<IMetric>>* lossFunction);
