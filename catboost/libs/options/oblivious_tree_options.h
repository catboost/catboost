#pragma once

#include "option.h"
#include "bootstrap_options.h"

#include <util/system/types.h>

namespace NJson {
    class TJsonValue;
}

namespace NCatboostOptions {
    class TObliviousTreeLearnerOptions {
    public:
        explicit TObliviousTreeLearnerOptions(ETaskType taskType);

        void Save(NJson::TJsonValue* options) const;
        void Load(const NJson::TJsonValue& options) ;

        bool operator==(const TObliviousTreeLearnerOptions& rhs) const;
        bool operator!=(const TObliviousTreeLearnerOptions& rhs) const;

        void Validate() const;

        TOption<ui32> MaxDepth;
        TOption<ui32> LeavesEstimationIterations;
        TOption<ELeavesEstimation> LeavesEstimationMethod;
        TOption<float> L2Reg;
        TOption<float> PairwiseNonDiagReg;
        TOption<float> RandomStrength;
        TOption<TBootstrapConfig> BootstrapConfig;
        TOption<float> Rsm;
        TOption<ELeavesEstimationStepBacktracking> LeavesEstimationBacktrackingType;

        TCpuOnlyOption<ESamplingFrequency> SamplingFrequency;
        TCpuOnlyOption<float> ModelSizeReg;

        // changing this parameter can affect results due to numerical accuracy differences
        TCpuOnlyOption<ui32> DevScoreCalcObjBlockSize;

        TGpuOnlyOption<EObservationsToBootstrap> ObservationsToBootstrap;
        TGpuOnlyOption<bool> FoldSizeLossNormalization;
        TGpuOnlyOption<bool> AddRidgeToTargetFunctionFlag;
        TGpuOnlyOption<EScoreFunction> ScoreFunction;
        TGpuOnlyOption<ui32> MaxCtrComplexityForBordersCaching;
        TGpuOnlyOption<EGrowingPolicy> GrowingPolicy;
        TGpuOnlyOption<ui32> MaxLeavesCount;
        TGpuOnlyOption<double> MinSamplesInLeaf;
    };
}
