#pragma once

#include <util/generic/fwd.h>

struct TMetricHolder;

TMetricHolder ComputeHingeLossMetric(TConstArrayRef<TVector<double>> approx,
                                     TConstArrayRef<float> target,
                                     TConstArrayRef<float> weight,
                                     int begin,
                                     int end,
                                     double border);
