#pragma once

#include <util/generic/vector.h>
#include <util/generic/ymath.h>
#include <util/generic/algorithm.h>

/*
 * Inspired by vpdelta@ FML implementation
 */
struct TWxTestResult {
    double WPlus = 0;
    double WMinus = 0;
    double PValue = 0;
};

TWxTestResult WxTest(const TVector<double>& baseline, const TVector<double>& test);
