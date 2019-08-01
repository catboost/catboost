#pragma once

#include "approx_updater_helpers.h"

#include <util/generic/vector.h>

template <typename TLeafUpdater, typename TApproxUpdater, typename TLossCalcer, typename TApproxCopier>
void GradientWalker(
    bool isTrivial,
    int iterationCount,
    int leafCount,
    int dimensionCount,
    const TLeafUpdater& calculateStep,
    const TApproxUpdater& updatePoint,
    const TLossCalcer& calculateLoss,
    const TApproxCopier& copyPoint,
    TVector<TVector<double>>* point,
    TVector<TVector<double>>* stepSum
) {
    TVector<TVector<double>> step(dimensionCount, TVector<double>(leafCount)); // iteration scratch space
    if (isTrivial) {
        for (int iterationIdx = 0; iterationIdx < iterationCount; ++iterationIdx) {
            calculateStep(iterationIdx == 0, *point, &step);
            updatePoint(step, point);
            if (stepSum != nullptr) {
                AddElementwise(step, stepSum);
            }
        }
        return;
    }
    TVector<TVector<double>> startPoint; // iteration scratch space
    double lossValue = calculateLoss(*point);
    for (int iterationIdx = 0; iterationIdx < iterationCount; ++iterationIdx)
    {
        calculateStep(iterationIdx == 0, *point, &step);
        copyPoint(*point, &startPoint);
        double scale = 1.0;
        // if monotone constraints are nontrivial the scale should be less or equal to 1.0.
        // Otherwise monotonicity may be violated.
        do {
            const auto scaledStep = ScaleElementwise(scale, step);
            updatePoint(scaledStep, point);
            const double valueAfterStep = calculateLoss(*point);
            if (valueAfterStep < lossValue) {
                lossValue = valueAfterStep;
                if (stepSum != nullptr) {
                    AddElementwise(scaledStep, stepSum);
                }
                break;
            }
            copyPoint(startPoint, point);
            scale /= 2;
            ++iterationIdx;
        } while (iterationIdx < iterationCount);
    }
}

