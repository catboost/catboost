#include "unimodal.h"

#include "linear_regression.h"

#include <util/generic/map.h>
#include <util/generic/ymath.h>

namespace {
    double SimpleUnimodal(const double value) {
        if (value > 5) {
            return 0.;
        }
        return 1. / (value * value + 1.);
    }

    struct TOptimizationState {
        double Mode = 0.;
        double Normalizer = 1.;

        double RegressionFactor = 0.;
        double RegressionIntercept = 0.;

        double SSE = 0.;

        TOptimizationState(const TVector<double>& values) {
            SSE = InnerProduct(values, values);
        }

        double NoRegressionTransform(const double value) const {
            const double arg = (value - Mode) / Normalizer;
            return SimpleUnimodal(arg);
        }

        double RegressionTransform(const double value) const {
            return NoRegressionTransform(value) * RegressionFactor + RegressionIntercept;
        }
    };
}

double TGreedyParams::Point(const size_t step) const {
    Y_ASSERT(step <= StepsCount);

    const double alpha = (double)step / StepsCount;
    return LowerBound * (1 - alpha) + UpperBound * alpha;
}

double MakeUnimodal(TVector<double>& values, const TOptimizationParams& optimizationParams) {
    TOptimizationState state(values);
    TOptimizationState bestState = state;

    for (size_t modeStep = 0; modeStep <= optimizationParams.ModeParams.StepsCount; ++modeStep) {
        state.Mode = optimizationParams.ModeParams.Point(modeStep);
        for (size_t normalizerStep = 0; normalizerStep <= optimizationParams.NormalizerParams.StepsCount; ++normalizerStep) {
            state.Normalizer = optimizationParams.NormalizerParams.Point(normalizerStep);

            TSLRSolver solver;
            for (size_t i = 0; i < values.size(); ++i) {
                solver.Add(state.NoRegressionTransform(i), values[i]);
            }

            state.SSE = solver.SumSquaredErrors(optimizationParams.RegressionShrinkage);
            if (state.SSE >= bestState.SSE) {
                continue;
            }

            bestState = state;
            solver.Solve(bestState.RegressionFactor, bestState.RegressionIntercept, optimizationParams.RegressionShrinkage);
        }
    }

    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = bestState.RegressionTransform(i);
    }

    const double residualSSE = bestState.SSE;
    const double totalSSE = InnerProduct(values, values);

    const double determination = 1. - residualSSE / totalSSE;

    return determination;
}

double MakeUnimodal(TVector<double>& values) {
    return MakeUnimodal(values, TOptimizationParams::Default(values));
}

double MakeUnimodal(TVector<double>& values, const TVector<double>& arguments, const TOptimizationParams& optimizationParams) {
    Y_ASSERT(values.size() == arguments.size());

    TMap<double, double> mapping;
    for (size_t i = 0; i < values.size(); ++i) {
        mapping[arguments[i]] = values[i];
    }

    TVector<double> preparedValues;
    preparedValues.reserve(mapping.size());

    for (auto&& argWithValue : mapping) {
        preparedValues.push_back(argWithValue.second);
    }

    const double result = MakeUnimodal(preparedValues, optimizationParams);

    size_t pos = 0;
    for (auto&& argWithValue : mapping) {
        argWithValue.second = preparedValues[pos++];
    }

    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = mapping[arguments[i]];
    }

    return result;
}

double MakeUnimodal(TVector<double>& values, const TVector<double>& arguments) {
    return MakeUnimodal(values, arguments, TOptimizationParams::Default(values, arguments));
}
