#pragma once

#include <util/generic/array_ref.h>

/*
Helper function for the target dependent quantile loss, corresponding metric and exact leaf estimate.
Selects one of N quantile levels based on where "value" falls relative to the sorted boundaries.

Given N-1 boundaries and N quantiles:
  - value <= boundaries[0]           -> quantiles[0]
  - boundaries[i-1] < value <= boundaries[i] -> quantiles[i]
  - value > boundaries[N-2]          -> quantiles[N-1]
*/
static inline double select_quantile(
    TConstArrayRef<double> boundaries,
    TConstArrayRef<double> quantiles,
    double value)
{
    for (size_t i = 0; i < boundaries.size(); ++i) {
        if (value <= boundaries[i]) {
            return quantiles[i];
        }
    }
    return quantiles[boundaries.size()];
}
