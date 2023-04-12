#pragma once

#include <catboost/private/libs/algo_helpers/approx_updater_helpers.h>

#include <util/generic/vector.h>

template <typename TLeafUpdater, typename TApproxUpdater, typename TLossCalcer, typename TApproxCopier>
void GradientWalker(
    bool isTrivial,
    int iterationCount,
    int leafCount,
    int dimensionCount,
    const TLeafUpdater& leafUpdaterFunc,
    const TApproxUpdater& approxUpdaterFunc,
    const TLossCalcer& lossCalcerFunc,
    const TApproxCopier& approxCopyFunc,
    TVector<TVector<double>>* point,
    TVector<TVector<double>>* stepSum
) {
    TVector<TVector<double>> step(dimensionCount, TVector<double>(leafCount));

    if (isTrivial) {
        for (int iterationIdx = 0; iterationIdx < iterationCount; ++iterationIdx) {
            leafUpdaterFunc(iterationIdx == 0, *point, &step);
            approxUpdaterFunc(step, point);
            if (stepSum != nullptr) {
                AddElementwise(step, stepSum);
            }
        }
        return;
    }
    TVector<TVector<double>> startPoint; // iteration scratch space
    double lossValue = lossCalcerFunc(*point);
    for (int iterationIdx = 0; iterationIdx < iterationCount; ++iterationIdx)
    {
        leafUpdaterFunc(iterationIdx == 0, *point, &step);
        approxCopyFunc(*point, &startPoint);
        double scale = 1.0;
        // if monotone constraints are nontrivial the scale should be less or equal to 1.0.
        // Otherwise monotonicity may be violated.
        do {
            const auto scaledStep = ScaleElementwise(scale, step);
            approxUpdaterFunc(scaledStep, point);
            const double valueAfterStep = lossCalcerFunc(*point);
            if (valueAfterStep < lossValue) {
                lossValue = valueAfterStep;
                if (stepSum != nullptr) {
                    AddElementwise(scaledStep, stepSum);
                }
                break;
            }
            approxCopyFunc(startPoint, point);
            scale /= 2;
            ++iterationIdx;
        } while (iterationIdx < iterationCount);
    }
}

template <typename TLeafUpdater, typename TApproxUpdater, typename TLossCalcer>
void FastGradientWalker(
    bool isTrivial,
    int iterationCount,
    int leafCount,
    int dimensionCount,
    const TLeafUpdater& leafUpdaterFunc,
    const TApproxUpdater& approxUpdaterFunc,
    const TLossCalcer& lossCalcerFunc,
    TVector<TVector<double>>* point,
    TVector<TVector<double>>* stepSum
) {
    TVector<TVector<double>> step(dimensionCount, TVector<double>(leafCount));

    if (isTrivial) {
        for (int iterationIdx = 0; iterationIdx < iterationCount; ++iterationIdx) {
            leafUpdaterFunc(iterationIdx == 0, *point, &step);
            approxUpdaterFunc(step, point);
            if (stepSum != nullptr) {
                AddElementwise(step, stepSum);
            }
        }
        return;
    }
    double lossValue = lossCalcerFunc(*point, step); // step is zero so far
    for (int iterationIdx = 0; iterationIdx < iterationCount; ++iterationIdx) {
        leafUpdaterFunc(iterationIdx == 0, *point, &step);
        double scale = 1.0;
        // if monotone constraints are nontrivial the scale should be less or equal to 1.0.
        // Otherwise monotonicity may be violated.
        do {
            const auto scaledStep = ScaleElementwise(scale, step);
            const double valueAfterStep = lossCalcerFunc(*point, scaledStep);
            if (valueAfterStep < lossValue) {
                lossValue = valueAfterStep;
                approxUpdaterFunc(scaledStep, point);
                if (stepSum != nullptr) {
                    AddElementwise(scaledStep, stepSum);
                }
                break;
            }
            scale /= 2;
            ++iterationIdx;
        } while (iterationIdx < iterationCount);
    }
}
