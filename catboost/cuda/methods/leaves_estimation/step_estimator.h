#pragma once

#include <util/generic/vector.h>
#include <catboost/libs/options/enums.h>

namespace NCatboostCuda {
    class IStepEstimator {
    public:
        virtual ~IStepEstimator() {
        }

        virtual bool IsSatisfied(double step,
                                 double nextFuncValue,
                                 const TVector<double>& nextFuncGradient) const = 0;
    };

    THolder<IStepEstimator> CreateStepEstimator(ELeavesEstimationStepBacktracking type,
                                                double currentPoint,
                                                const TVector<double>& gradientAtPoint,
                                                const TVector<float>& moveDirection);
}
