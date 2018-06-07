#pragma once

#include <util/generic/fwd.h>

enum class EKappaMetricType;
struct TMetricHolder;

double CalcKappa(TMetricHolder metric, int classCount, EKappaMetricType type);

TMetricHolder CalcKappaMatrix(TConstArrayRef<TVector<double>> approx,
                              TConstArrayRef<float> target,
                              int begin,
                              int end,
                              double border);
