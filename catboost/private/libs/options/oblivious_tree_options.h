#pragma once

#include "option.h"
#include "bootstrap_options.h"
#include "feature_penalties_options.h"

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
        TOption<ERandomScoreType> RandomScoreType;
        TOption<TBootstrapConfig> BootstrapConfig;
        TOption<float> Rsm;
        TOption<ELeavesEstimationStepBacktracking> LeavesEstimationBacktrackingType;
        TOption<EScoreFunction> ScoreFunction;
        TOption<EGrowPolicy> GrowPolicy;
        TOption<ui32> MaxLeaves;
        TOption<double> MinDataInLeaf;
        TOption<ui32> DevExclusiveFeaturesBundleMaxBuckets;

        TCpuOnlyOption<ESamplingFrequency> SamplingFrequency;
        TOption<float> ModelSizeReg;

        // changing this parameter can affect results due to numerical accuracy differences
        TCpuOnlyOption<ui32> DevScoreCalcObjBlockSize;

        TCpuOnlyOption<float> SparseFeaturesConflictFraction;

        TGpuOnlyOption<EObservationsToBootstrap> ObservationsToBootstrap;
        TGpuOnlyOption<bool> FoldSizeLossNormalization;
        TGpuOnlyOption<bool> AddRidgeToTargetFunctionFlag;
        TGpuOnlyOption<ui32> MaxCtrComplexityForBordersCaching;
        TGpuOnlyOption<float> MetaL2Exponent;
        TGpuOnlyOption<float> MetaL2Frequency;

        TGpuOnlyOption<TVector<ui32>> FixedBinarySplits;

        TCpuOnlyOption<TMap<ui32, int>> MonotoneConstraints;
        TCpuOnlyOption <bool> DevLeafwiseApproxes;
        TOption<TFeaturePenaltiesOptions> FeaturePenalties;

    private:
        TOption<ETaskType> TaskType;
    };
}
