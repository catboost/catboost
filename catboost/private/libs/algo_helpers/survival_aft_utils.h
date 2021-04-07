#pragma once

#include <catboost/libs/helpers/distribution_helpers.h>

#include <library/cpp/fast_log/fast_log.h>

#include <util/generic/ymath.h>
#include <util/generic/string.h>

#include <cmath>
#include <tuple>

static constexpr double MIN_FIRST_DER = -15.0;
static constexpr double MAX_FIRST_DER = 15.0;
static constexpr double MIN_SECOND_DER = 1e-16;
static constexpr double MAX_SECOND_DER = 15.0;
static constexpr double EPS = 1e-12; 

enum class ECensoredType {
    Uncensored,
    IntervalCensored,
    RightCensored,
    LeftCensored
};

enum class EDerivativeOrder {
    First,
    Second
};

double InverseMonotoneTransform(double approx, double target, double scale);

double ClipDerivatives(double der, double minDerivative, double maxDerivative);

template<EDistributionType Distribution>
std::tuple<double, double> GetDerivativeLimits(EDerivativeOrder order, ECensoredType censoredType, double scale);

std::tuple<double, double> DispatchDerivativeLimits(EDistributionType type, EDerivativeOrder derivativeOrder, ECensoredType censoredType, double scale);
