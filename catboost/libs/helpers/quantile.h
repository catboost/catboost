#pragma once

#include <util/generic/fwd.h>

double CalcSampleQuantile(
    TConstArrayRef<float> sample,
    TConstArrayRef<float> weights,
    double alpha,
    double delta
);

double CalcSampleQuantileSorted(
    TConstArrayRef<float> sample,
    TConstArrayRef<float> weights,
    double alpha,
    double delta
);
