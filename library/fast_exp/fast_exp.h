#pragma once

#include <stdlib.h>

double fast_exp(double x);

inline double Logistic(double x) {
    return 1 / (1 + fast_exp(-x));
}

void FastExpInplace(double* x, size_t count);
