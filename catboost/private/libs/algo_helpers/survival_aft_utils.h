#pragma once

#include <catboost/libs/helpers/distribution_helpers.h>

#include <tuple>


namespace NCB {
    struct TDerivativeConstants {
        static constexpr double MinFirstDer = -15.0;
        static constexpr double MaxFirstDer = 15.0;
        static constexpr double MinSecondDer = 1e-16;
        static constexpr double MaxSecondDer = 15.0;
        static constexpr double Epsilon = 1e-12;
    };

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
}
