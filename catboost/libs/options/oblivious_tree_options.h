#pragma once

#include "option.h"
#include "json_helper.h"
#include "bootstrap_options.h"

#include <catboost/libs/logging/logging_level.h>
#include <catboost/libs/logging/logging.h>
#include <library/json/json_value.h>
#include <util/system/types.h>
#include <util/system/types.h>

namespace NCatboostOptions {
    class TObliviousTreeLearnerOptions {
    public:
        explicit TObliviousTreeLearnerOptions(ETaskType taskType)
            : MaxDepth("depth", 6)
            , LeavesEstimationIterations("leaf_estimation_iterations", 1)
            , LeavesEstimationMethod("leaf_estimation_method", ELeavesEstimation::Gradient)
            , L2Reg("l2_leaf_reg", 3.0)
            , RandomStrength("random_strength", 1.0)
            , BootstrapConfig("bootstrap", TBootstrapConfig(taskType))
            , Rsm("rsm", 1.0, taskType)
            , SamplingFrequency("sampling_frequency", ESamplingFrequency::PerTreeLevel, taskType)
            , ModelSizeReg("model_size_reg", 0.5, taskType)
            , ObservationsToBootstrap("observations_to_bootstrap", EObservationsToBootstrap::TestOnly, taskType) //it's specific for fold-based scheme, so here and not in bootstrap options
            , FoldSizeLossNormalization("fold_size_loss_normalization", false, taskType)
            , AddRidgeToTargetFunctionFlag("add_ridge_penalty_to_loss_function", false, taskType)
            , ScoreFunction("score_function", EScoreFunction::Correlation, taskType)
            , MaxCtrComplexityForBordersCaching("max_ctr_complexity_for_borders_cache", 1, taskType)
        {
            Rsm.ChangeLoadUnimplementedPolicy(ELoadUnimplementedPolicy::ExceptionOnChange);
            SamplingFrequency.ChangeLoadUnimplementedPolicy(ELoadUnimplementedPolicy::ExceptionOnChange);

            FoldSizeLossNormalization.ChangeLoadUnimplementedPolicy(ELoadUnimplementedPolicy::ExceptionOnChange);
            AddRidgeToTargetFunctionFlag.ChangeLoadUnimplementedPolicy(ELoadUnimplementedPolicy::ExceptionOnChange);
            ScoreFunction.ChangeLoadUnimplementedPolicy(ELoadUnimplementedPolicy::ExceptionOnChange);

            MaxCtrComplexityForBordersCaching.ChangeLoadUnimplementedPolicy(ELoadUnimplementedPolicy::SkipWithWarning);
        }

        void Load(const NJson::TJsonValue& options) {
            CheckedLoad(options,
                        &MaxDepth, &LeavesEstimationIterations, &LeavesEstimationMethod, &L2Reg, &ModelSizeReg,
                        &RandomStrength,
                        &BootstrapConfig, &FoldSizeLossNormalization, &AddRidgeToTargetFunctionFlag,
                        &ScoreFunction,
                        &MaxCtrComplexityForBordersCaching,
                        &Rsm,
                        &ObservationsToBootstrap,
                        &SamplingFrequency);

            Validate();
        }

        void Save(NJson::TJsonValue* options) const {
            SaveFields(options, MaxDepth, LeavesEstimationIterations, LeavesEstimationMethod, L2Reg, ModelSizeReg,
                       RandomStrength,
                       BootstrapConfig, FoldSizeLossNormalization, AddRidgeToTargetFunctionFlag,
                       ScoreFunction,
                       MaxCtrComplexityForBordersCaching, Rsm, ObservationsToBootstrap, SamplingFrequency);
        }

        bool operator==(const TObliviousTreeLearnerOptions& rhs) const {
            return std::tie(MaxDepth, LeavesEstimationIterations, LeavesEstimationMethod, L2Reg, ModelSizeReg, RandomStrength,
                            BootstrapConfig, Rsm, SamplingFrequency, ObservationsToBootstrap, FoldSizeLossNormalization,
                            AddRidgeToTargetFunctionFlag, ScoreFunction, MaxCtrComplexityForBordersCaching) ==
                   std::tie(rhs.MaxDepth, rhs.LeavesEstimationIterations, rhs.LeavesEstimationMethod, rhs.L2Reg, rhs.ModelSizeReg,
                            rhs.RandomStrength, rhs.BootstrapConfig, rhs.Rsm, rhs.SamplingFrequency,
                            rhs.ObservationsToBootstrap, rhs.FoldSizeLossNormalization, rhs.AddRidgeToTargetFunctionFlag,
                            rhs.ScoreFunction, rhs.MaxCtrComplexityForBordersCaching);
        }

        bool operator!=(const TObliviousTreeLearnerOptions& rhs) const {
            return !(rhs == *this);
        }

        void Validate() const {
            BootstrapConfig.Get().Validate();
            const float rsm = Rsm.GetUnchecked();
            CB_ENSURE(rsm > 0 && rsm <= 1, "Rsm should be in (0, 1]");
            const ui32 maxModelDepth = 16;
            CB_ENSURE(MaxDepth.Get() <= maxModelDepth, "Maximum depth is " << maxModelDepth);
            CB_ENSURE(LeavesEstimationIterations.Get() > 0, "Leaves estimation iterations should be positive");
            CB_ENSURE(L2Reg.Get() >= 0, "L2LeafRegularizer should be >= 0, current value: " << L2Reg.Get());
        }

        TOption<ui32> MaxDepth;
        TOption<ui32> LeavesEstimationIterations;
        TOption<ELeavesEstimation> LeavesEstimationMethod;
        TOption<float> L2Reg;
        TOption<float> RandomStrength;
        TOption<TBootstrapConfig> BootstrapConfig;

        TCpuOnlyOption<float> Rsm;
        TCpuOnlyOption<ESamplingFrequency> SamplingFrequency;
        TCpuOnlyOption<float> ModelSizeReg;

        TGpuOnlyOption<EObservationsToBootstrap> ObservationsToBootstrap;
        TGpuOnlyOption<bool> FoldSizeLossNormalization;
        TGpuOnlyOption<bool> AddRidgeToTargetFunctionFlag;
        TGpuOnlyOption<EScoreFunction> ScoreFunction;
        TGpuOnlyOption<ui32> MaxCtrComplexityForBordersCaching;
    };
}
