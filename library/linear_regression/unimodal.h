#pragma once

#include "linear_regression.h"

struct TGreedyParams {
    double LowerBound = 0;
    double UpperBound = 0;
    size_t StepsCount = 0;

    double Point(const size_t step) const;
};

struct TOptimizationParams {
    TGreedyParams ModeParams;
    TGreedyParams NormalizerParams;

    double OptimizationShrinkage = 1e-2;
    double RegressionShrinkage = 1e-5;

    size_t IterationsCount = 1000;

    TOptimizationParams() = default;

    static TOptimizationParams Default(const TVector<double>& values) {
        TOptimizationParams optimizationParams;

        optimizationParams.ModeParams.LowerBound = 0;
        optimizationParams.ModeParams.UpperBound = values.size();
        optimizationParams.ModeParams.StepsCount = values.size() + 1;

        optimizationParams.NormalizerParams.LowerBound = 0.5;
        optimizationParams.NormalizerParams.UpperBound = values.size() * 2;
        optimizationParams.NormalizerParams.StepsCount = values.size() * 2 + 1;

        return optimizationParams;
    }

    static TOptimizationParams Default(const TVector<double>& values, const TVector<double>& arguments) {
        Y_ASSERT(values.size() == arguments.size());

        TOptimizationParams optimizationParams;

        optimizationParams.ModeParams.LowerBound = *MinElement(arguments.begin(), arguments.end());
        optimizationParams.ModeParams.UpperBound = *MaxElement(arguments.begin(), arguments.end());
        optimizationParams.ModeParams.StepsCount = arguments.size() + 1;

        optimizationParams.NormalizerParams.UpperBound = optimizationParams.ModeParams.UpperBound - optimizationParams.ModeParams.LowerBound;
        optimizationParams.NormalizerParams.StepsCount = arguments.size() * 2 + 1;
        optimizationParams.NormalizerParams.LowerBound = optimizationParams.NormalizerParams.UpperBound / optimizationParams.NormalizerParams.StepsCount;

        return optimizationParams;
    }
};

double MakeUnimodal(TVector<double>& values, const TOptimizationParams& optimizationParams);
double MakeUnimodal(TVector<double>& values);

double MakeUnimodal(TVector<double>& values, const TVector<double>& arguments, const TOptimizationParams& optimizationParams);
double MakeUnimodal(TVector<double>& values, const TVector<double>& arguments);
