#pragma once

#include <util/generic/fwd.h>

/*
 * Returns the minimal value Q such that sum of weights for values less or equal than Q is at least (alpha * totalWeight)
 */
double CalcSampleQuantile(
    TConstArrayRef<float> sample,
    TConstArrayRef<float> weights,
    double alpha
);
