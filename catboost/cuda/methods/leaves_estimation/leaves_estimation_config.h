#pragma once

#include <catboost/private/libs/options/oblivious_tree_options.h>

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
        double NonDiagLambda = 0;
        bool ZeroLastDimHack = false;

        TLeavesEstimationConfig(bool useNewton,
                                double lambda,
                                ui32 iterations,
                                double minLeafWeight,
                                bool normalize,
                                bool addRidgeToTargetFunction,
                                bool zeroAverage,
                                ELeavesEstimationStepBacktracking backtracking,
                                double bayesianLambda)
            : UseNewton(useNewton)
            , Lambda(lambda)
            , Iterations(iterations)
            , MinLeafWeight(minLeafWeight)
            , IsNormalize(normalize)
            , AddRidgeToTargetFunction(addRidgeToTargetFunction)
            , MakeZeroAverage(zeroAverage)
            , BacktrackingType(backtracking)
            , NonDiagLambda(bayesianLambda)
        {
        }
    };

    inline TLeavesEstimationConfig CreateLeavesEstimationConfig(const NCatboostOptions::TObliviousTreeLearnerOptions& treeConfig,
                                                                bool makeZeroAverage) {
        return TLeavesEstimationConfig(treeConfig.LeavesEstimationMethod == ELeavesEstimation::Newton,
                                       treeConfig.L2Reg,
                                       treeConfig.LeavesEstimationIterations,
                                       1e-20,
                                       treeConfig.FoldSizeLossNormalization,
                                       treeConfig.AddRidgeToTargetFunctionFlag,
                                       makeZeroAverage,
                                       treeConfig.LeavesEstimationBacktrackingType,
                                       treeConfig.PairwiseNonDiagReg);
    }

}
