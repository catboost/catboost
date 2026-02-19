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

// Finds the optimum of the target dependent loss function through binary search.
// boundaries: N-1 sorted boundary values; quantiles: N quantile levels.
double CalcTargetDependentMinimum(
    TConstArrayRef<float> sampleRef,
    TConstArrayRef<float> weightsRef,
    TConstArrayRef<float> origTarget,
    TConstArrayRef<double> boundaries,
    TConstArrayRef<double> quantiles
);
