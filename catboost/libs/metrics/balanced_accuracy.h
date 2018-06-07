#pragma once

#include <util/generic/fwd.h>

TMetricHolder CalcBalancedAccuracyMetric(TConstArrayRef<TVector<double>> approx,
                                         TConstArrayRef<float> target,
                                         TConstArrayRef<float> weight,
                                         int begin,
                                         int end,
                                         int positiveClass,
                                         double border);

double CalcBalancedAccuracyMetric(const TMetricHolder& error);
