#pragma once

#include <util/generic/fwd.h>

TMetricHolder CalcBalancedAccuracyMetric(TConstArrayRef<TConstArrayRef<double>> approx,
                                         TConstArrayRef<float> target,
                                         TConstArrayRef<float> weight,
                                         int begin,
                                         int end,
                                         int positiveClass,
                                         double targetBorder,
                                         double predictionBorder);

double CalcBalancedAccuracyMetric(const TMetricHolder& error);
