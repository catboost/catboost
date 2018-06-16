#pragma once

#include <util/generic/fwd.h>

TMetricHolder ComputeBrierScoreMetric(TConstArrayRef<double> approx,
                                      TConstArrayRef<float> target,
                                      TConstArrayRef<float> weight,
                                      int begin,
                                      int end);

