#include "gamma_function.h"

double Digamma(double x) {
    if (x > 0.0 && x <= 1.0E-5) {
        return -0.5772156649015329 - 1.0 / x;
    } else if (x >= 49.0) {
        double inv = 1.0 / (x * x);
        return log(x) - 0.5 / x - inv * (0.08333333333333333 + inv * (0.008333333333333333 - inv / 252.0));
    } else {
        return Digamma(x + 1.0) - 1.0 / x;
    }
}

double Trigamma(double x) {
    if (x > 0.0 && x <= 1.0E-5) {
        return 1.0 / (x * x);
    } else if (x >= 49.0) {
        double inv = 1.0 / (x * x);
        //  1    1      1       1       1
        //  - + ---- + ---- - ----- + -----
        //  x      2      3       5       7
        //      2 x    6 x    30 x    42 x
        return 1.0 / x + inv / 2.0 + inv / x * (0.16666666666666666 - inv * (0.03333333333333333 + inv / 42.0));
    } else {
        return Trigamma(x + 1.0) + 1.0 / (x * x);
    }
}

