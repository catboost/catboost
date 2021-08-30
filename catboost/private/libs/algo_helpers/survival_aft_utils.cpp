#include "survival_aft_utils.h"


namespace NCB {
    double InverseMonotoneTransform(double approx, double target, double scale) {
        return (FastLogf(target) - approx) / scale;
    }

    double ClipDerivatives(double der, double minDerivative, double maxDerivative) {
        return Max(Min(der, maxDerivative), minDerivative);
    }

    template<>
    std::tuple<double, double> GetDerivativeLimits<EDistributionType::Normal>(EDerivativeOrder order, ECensoredType censoredType, double scale) {
        switch (order) {
            case EDerivativeOrder::First:
                switch (censoredType) {
                    case ECensoredType::IntervalCensored:
                    case ECensoredType::Uncensored:
                        return std::make_tuple(TDerivativeConstants::MinFirstDer, TDerivativeConstants::MaxFirstDer);
                    case ECensoredType::RightCensored:
                        return std::make_tuple(TDerivativeConstants::MinFirstDer, 0);
                    case ECensoredType::LeftCensored:
                        return std::make_tuple(0, TDerivativeConstants::MaxFirstDer);
                }
            case EDerivativeOrder::Second:
                switch (censoredType) {
                    case ECensoredType::IntervalCensored:
                    case ECensoredType::Uncensored:
                        return std::make_tuple(1 / std::pow(scale, 2), 1 / std::pow(scale, 2));
                    case ECensoredType::RightCensored:
                        return std::make_tuple(1 / std::pow(scale, 2), TDerivativeConstants::MinSecondDer);
                    case ECensoredType::LeftCensored:
                        return std::make_tuple(TDerivativeConstants::MinSecondDer, 1 / std::pow(scale, 2));
                }
        }
    }


    template<>
    std::tuple<double, double> GetDerivativeLimits<EDistributionType::Extreme>(EDerivativeOrder order, ECensoredType censoredType, double scale) {
        switch (order) {
            case EDerivativeOrder::First:
                switch (censoredType) {
                    case ECensoredType::IntervalCensored:
                    case ECensoredType::Uncensored:
                        return std::make_tuple(-15, 1 / scale);
                    case ECensoredType::RightCensored:
                        return std::make_tuple(-15, 0);
                    case ECensoredType::LeftCensored:
                        return std::make_tuple(0, 1 / scale);
                }
            case EDerivativeOrder::Second:
                switch (censoredType) {
                    case ECensoredType::IntervalCensored:
                    case ECensoredType::Uncensored:
                    case ECensoredType::RightCensored:
                        return std::make_tuple(15, TDerivativeConstants::MinSecondDer);
                    case ECensoredType::LeftCensored:
                        return std::make_tuple(TDerivativeConstants::MinSecondDer, TDerivativeConstants::MinSecondDer);
                }
        }
    }

    template<>
    std::tuple<double, double> GetDerivativeLimits<EDistributionType::Logistic>(EDerivativeOrder order, ECensoredType censoredType, double scale) {
        switch (order) {
            case EDerivativeOrder::First:
                switch (censoredType) {
                    case ECensoredType::IntervalCensored:
                    case ECensoredType::Uncensored:
                        return std::make_tuple(-1 / scale, 1 / scale);
                    case ECensoredType::RightCensored:
                        return std::make_tuple(-1 / scale, 0);
                    case ECensoredType::LeftCensored:
                        return std::make_tuple(0, 1 / scale);
                }
            case EDerivativeOrder::Second:
                switch (censoredType) {
                    case ECensoredType::IntervalCensored:
                    case ECensoredType::Uncensored:
                    case ECensoredType::RightCensored:
                    case ECensoredType::LeftCensored:
                        return std::make_tuple(TDerivativeConstants::MinSecondDer, TDerivativeConstants::MinSecondDer);
                }
        }
    }

    std::tuple<double, double> DispatchDerivativeLimits(EDistributionType type, EDerivativeOrder derivativeOrder, ECensoredType censoredType, double scale) {
        switch (type) {
            case EDistributionType::Normal:
                return GetDerivativeLimits<EDistributionType::Normal>(derivativeOrder, censoredType, scale);
            case EDistributionType::Extreme:
                return GetDerivativeLimits<EDistributionType::Extreme>(derivativeOrder, censoredType, scale);
            case EDistributionType::Logistic:
                return GetDerivativeLimits<EDistributionType::Logistic>(derivativeOrder, censoredType, scale);
            }
    }
}
