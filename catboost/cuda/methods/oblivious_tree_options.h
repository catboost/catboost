#pragma once

#include "bootstrap.h"
#include "score_function.h"

struct TObliviousTreeLearnerOptions {
public:
    ui32 GetMaxDepth() const {
        return MaxDepth;
    }

    ui32 GetLeavesEstimationIters() const {
        return LeavesEstimationIters;
    }

    double GetL2Reg() const {
        return L2Reg;
    }

    bool IsNormalize() const {
        return NormalizeLossInEstimation;
    }

    bool AddRidgeToTargetFunction() const {
        return AddRidgeToTargetFunctionFlag;
    }

    bool IsBootstrapTestOnly() const {
        return !BootstrapLearn;
    }

    TObliviousTreeLearnerOptions& SetLeavesEstimationIterations(ui32 iters) {
        LeavesEstimationIters = iters;
        return *this;
    }

    bool IsUseNewton() const {
        return UseNewton;
    }

    EScoreFunction GetScoreFunction() const {
        return ScoreFunction;
    }

    const TBootstrapConfig& GetBootstrapConfig() const {
        return BootstrapConfig;
    }

    ui32 GetMaxCtrComplexityForBordersCaching() const {
        return MaxCtrComplexityForBordersCaching;
    }

    bool IsDumpFreeMemory() const {
        return DumpFreeMemoryFlag;
    }

    template <class TConfig>
    friend class TOptionsBinder;

private:
    ui32 MaxDepth = 6;
    ui32 LeavesEstimationIters = 10;
    ui32 MaxCtrComplexityForBordersCaching = 1;
    double L2Reg = 0.00001;
    bool DumpFreeMemoryFlag = false;
    bool UseNewton = true;
    EScoreFunction ScoreFunction = EScoreFunction::Correlation;
    TBootstrapConfig BootstrapConfig;
    bool NormalizeLossInEstimation = false;
    bool AddRidgeToTargetFunctionFlag = false;
    bool BootstrapLearn = false;
};
