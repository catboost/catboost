#pragma once

#include <util/generic/fwd.h>

struct TMetricHolder;

TMetricHolder CalcLlp(TConstArrayRef<double> approx,
                      TConstArrayRef<float> target,
                      TConstArrayRef<float> weight,
                      int begin,
                      int end);

double CalcLlp(const TMetricHolder& error);
