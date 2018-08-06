#pragma once

#include "step_estimator.h"
#include "oracle_interface.h"

namespace NCatboostCuda {


    class TNewtonLikeWalker {
    public:
        ILeavesEstimationOracle& Oracle;
        const ui32 Iterations;
        ELeavesEstimationStepBacktracking StepEstimationType;

    public:
        TNewtonLikeWalker(ILeavesEstimationOracle& oracle,
                          const ui32 iterations,
                          ELeavesEstimationStepBacktracking backtrackingType)
            : Oracle(oracle)
            , Iterations(iterations)
            , StepEstimationType(backtrackingType)
        {
        }

        TVector<float> Estimate(TVector<float> startPoint);
    };

}
