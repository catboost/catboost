#pragma once

#include <catboost/private/libs/options/oblivious_tree_options.h>
#include <catboost/private/libs/options/boosting_options.h>

namespace NCatboostCuda {
    struct TLeavesEstimationConfig {
        ELeavesEstimation LeavesEstimationMethod;
        double Lambda = 1.0; //l2 reg
        ui32 Iterations = 10;
        double MinLeafWeight = 1e-20;
        bool IsNormalize = false;
        bool AddRidgeToTargetFunction = false;
        bool MakeZeroAverage = false;
        ELeavesEstimationStepBacktracking BacktrackingType;
        double NonDiagLambda = 0;
        bool ZeroLastDimHack = false;
        NCatboostOptions::TLossDescription LossDescription;
        bool Langevin = false;
        float DiffusionTemperature = 0;
        float LearningRate = 0.03;

        TLeavesEstimationConfig(const ELeavesEstimation& leavesEstimationMethod,
                                double lambda,
                                ui32 iterations,
                                double minLeafWeight,
                                bool normalize,
                                bool addRidgeToTargetFunction,
                                bool zeroAverage,
                                ELeavesEstimationStepBacktracking backtracking,
                                double bayesianLambda,
                                const NCatboostOptions::TLossDescription& lossDescription,
                                bool shouldApplyLangevin,
                                float diffusionTemperature,
                                float learningRate)
            : LeavesEstimationMethod(leavesEstimationMethod)
            , Lambda(lambda)
            , Iterations(iterations)
            , MinLeafWeight(minLeafWeight)
            , IsNormalize(normalize)
            , AddRidgeToTargetFunction(addRidgeToTargetFunction)
            , MakeZeroAverage(zeroAverage)
            , BacktrackingType(backtracking)
            , NonDiagLambda(bayesianLambda)
            , LossDescription(lossDescription)
            , Langevin(shouldApplyLangevin)
            , DiffusionTemperature(diffusionTemperature)
            , LearningRate(learningRate)
        {
        }
    };

    inline TLeavesEstimationConfig CreateLeavesEstimationConfig(const NCatboostOptions::TObliviousTreeLearnerOptions& treeConfig,
                                                                bool makeZeroAverage,
                                                                const NCatboostOptions::TLossDescription& lossDescription,
                                                                const NCatboostOptions::TBoostingOptions& boostingOptions) {
        return TLeavesEstimationConfig(treeConfig.LeavesEstimationMethod,
                                       treeConfig.L2Reg,
                                       treeConfig.LeavesEstimationIterations,
                                       1e-20,
                                       treeConfig.FoldSizeLossNormalization,
                                       treeConfig.AddRidgeToTargetFunctionFlag,
                                       makeZeroAverage,
                                       treeConfig.LeavesEstimationBacktrackingType,
                                       treeConfig.PairwiseNonDiagReg,
                                       lossDescription,
                                       boostingOptions.Langevin,
                                       boostingOptions.DiffusionTemperature,
                                       boostingOptions.LearningRate);
    }

}
