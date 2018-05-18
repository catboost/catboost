#pragma once

namespace NCatboostCuda {


    struct TLeavesEstimationConfig {
        bool UseNewton = true;
        double Lambda = 1.0; //l2 reg
        ui32 Iterations = 10;
        double MinLeafWeight = 1e-20;
        bool IsNormalize = false;
        bool AddRidgeToTargetFunction = false;
        bool MakeZeroAverage = false;
        ELeavesEstimationStepBacktracking BacktrackingType;

        TLeavesEstimationConfig(bool useNewton,
                                double lambda,
                                ui32 iterations,
                                double minLeafWeight,
                                bool normalize,
                                bool addRidgeToTargetFunction,
                                bool zeroAverage,
                                ELeavesEstimationStepBacktracking backtracking = ELeavesEstimationStepBacktracking::AnyImprovment)
                : UseNewton(useNewton)
                  , Lambda(lambda)
                  , Iterations(iterations)
                  , MinLeafWeight(minLeafWeight)
                  , IsNormalize(normalize)
                  , AddRidgeToTargetFunction(addRidgeToTargetFunction)
                  , MakeZeroAverage(zeroAverage)
                  , BacktrackingType(backtracking) {
        }
    };

}
