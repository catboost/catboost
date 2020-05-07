#pragma once

#include <util/generic/fwd.h>

struct TMetricHolder;

TMetricHolder ComputeHingeLossMetric(TConstArrayRef<TConstArrayRef<double>> approx,
                                     TConstArrayRef<float> target,
                                     TConstArrayRef<float> weight,
                                     int begin,
                                     int end,
                                     double targetBorder);
