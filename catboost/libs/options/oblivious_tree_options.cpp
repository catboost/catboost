#include "oblivious_tree_options.h"
#include "json_helper.h"

#include <catboost/libs/logging/logging_level.h>
#include <catboost/libs/logging/logging.h>

#include <library/json/json_value.h>

NCatboostOptions::TObliviousTreeLearnerOptions::TObliviousTreeLearnerOptions(ETaskType taskType)
    : MaxDepth("depth", 6)
      , LeavesEstimationIterations("leaf_estimation_iterations", 1)
      , LeavesEstimationMethod("leaf_estimation_method", ELeavesEstimation::Gradient)
      , L2Reg("l2_leaf_reg", 3.0)
      , PairwiseNonDiagReg("bayesian_matrix_reg", 0.1)
      , RandomStrength("random_strength", 1.0)
      , BootstrapConfig("bootstrap", TBootstrapConfig(taskType))
      , Rsm("rsm", 1.0)
      , LeavesEstimationBacktrackingType("leaf_estimation_backtracking", ELeavesEstimationStepBacktracking::AnyImprovement)
      , SamplingFrequency("sampling_frequency", ESamplingFrequency::PerTree, taskType)
      , ModelSizeReg("model_size_reg", 0.5, taskType)
      , DevScoreCalcObjBlockSize("dev_score_calc_obj_block_size", 5000000, taskType)
      , ObservationsToBootstrap("observations_to_bootstrap", EObservationsToBootstrap::TestOnly, taskType) //it's specific for fold-based scheme, so here and not in bootstrap options
      , FoldSizeLossNormalization("fold_size_loss_normalization", false, taskType)
      , AddRidgeToTargetFunctionFlag("add_ridge_penalty_to_loss_function", false, taskType)
      , ScoreFunction("score_function", EScoreFunction::Correlation, taskType)
      , MaxCtrComplexityForBordersCaching("max_ctr_complexity_for_borders_cache", 1, taskType)
      , GrowingPolicy("growing_policy", EGrowingPolicy::ObliviousTree, taskType)
      , MaxLeavesCount("max_leaves_count", 31, taskType)
      , MinSamplesInLeaf("min_samples_in_leaf", 1, taskType)

{
    SamplingFrequency.ChangeLoadUnimplementedPolicy(ELoadUnimplementedPolicy::ExceptionOnChange);

    FoldSizeLossNormalization.ChangeLoadUnimplementedPolicy(ELoadUnimplementedPolicy::ExceptionOnChange);
    AddRidgeToTargetFunctionFlag.ChangeLoadUnimplementedPolicy(ELoadUnimplementedPolicy::ExceptionOnChange);
    ScoreFunction.ChangeLoadUnimplementedPolicy(ELoadUnimplementedPolicy::ExceptionOnChange);

    MaxCtrComplexityForBordersCaching.ChangeLoadUnimplementedPolicy(ELoadUnimplementedPolicy::SkipWithWarning);
}

void NCatboostOptions::TObliviousTreeLearnerOptions::Load(const NJson::TJsonValue& options) {
    CheckedLoad(options,
            &MaxDepth, &LeavesEstimationIterations, &LeavesEstimationMethod, &L2Reg, &ModelSizeReg,
            &RandomStrength,
            &BootstrapConfig, &FoldSizeLossNormalization, &AddRidgeToTargetFunctionFlag,
            &ScoreFunction,
            &MaxCtrComplexityForBordersCaching,
            &Rsm,
            &ObservationsToBootstrap,
            &PairwiseNonDiagReg,
            &LeavesEstimationBacktrackingType,
            &SamplingFrequency,
            &DevScoreCalcObjBlockSize,
            &GrowingPolicy,
            &MaxLeavesCount,
            &MinSamplesInLeaf
            );

    Validate();
}

void NCatboostOptions::TObliviousTreeLearnerOptions::Save(NJson::TJsonValue* options) const {
    SaveFields(options, MaxDepth, LeavesEstimationIterations, LeavesEstimationMethod, L2Reg, ModelSizeReg,
            RandomStrength,
            BootstrapConfig, FoldSizeLossNormalization, AddRidgeToTargetFunctionFlag,
            ScoreFunction,
            PairwiseNonDiagReg,
            LeavesEstimationBacktrackingType,
            MaxCtrComplexityForBordersCaching, Rsm, ObservationsToBootstrap, SamplingFrequency,
            DevScoreCalcObjBlockSize,
            GrowingPolicy,
            MaxLeavesCount,
            MinSamplesInLeaf
            );
}

bool NCatboostOptions::TObliviousTreeLearnerOptions::operator==(const TObliviousTreeLearnerOptions& rhs) const {
    return std::tie(MaxDepth, LeavesEstimationIterations, LeavesEstimationMethod, L2Reg, ModelSizeReg, RandomStrength,
            BootstrapConfig, Rsm, SamplingFrequency, ObservationsToBootstrap, FoldSizeLossNormalization,
            AddRidgeToTargetFunctionFlag, ScoreFunction, MaxCtrComplexityForBordersCaching,
            PairwiseNonDiagReg, LeavesEstimationBacktrackingType, DevScoreCalcObjBlockSize,
            GrowingPolicy, MaxLeavesCount, MinSamplesInLeaf
            ) ==
        std::tie(rhs.MaxDepth, rhs.LeavesEstimationIterations, rhs.LeavesEstimationMethod, rhs.L2Reg, rhs.ModelSizeReg,
                rhs.RandomStrength, rhs.BootstrapConfig, rhs.Rsm, rhs.SamplingFrequency,
                rhs.ObservationsToBootstrap, rhs.FoldSizeLossNormalization, rhs.AddRidgeToTargetFunctionFlag,
                rhs.ScoreFunction, rhs.MaxCtrComplexityForBordersCaching, rhs.PairwiseNonDiagReg, rhs.LeavesEstimationBacktrackingType,
                rhs.DevScoreCalcObjBlockSize, rhs.GrowingPolicy, rhs.MaxLeavesCount, rhs.MinSamplesInLeaf);
}

bool NCatboostOptions::TObliviousTreeLearnerOptions::operator!=(const TObliviousTreeLearnerOptions& rhs) const {
    return !(rhs == *this);
}

void NCatboostOptions::TObliviousTreeLearnerOptions::Validate() const {
    BootstrapConfig.Get().Validate();
    const float rsm = Rsm.Get();
    CB_ENSURE(rsm > 0 && rsm <= 1, "Rsm should be in (0, 1]");
    const ui32 maxModelDepth = 16;
    CB_ENSURE(MaxDepth.Get() <= maxModelDepth, "Maximum depth is " << maxModelDepth);
    CB_ENSURE(DevScoreCalcObjBlockSize.GetUnchecked() > 0, "DevScoreCalcObjBlockSize must be > 0");
    CB_ENSURE(LeavesEstimationIterations.Get() > 0, "Leaves estimation iterations should be positive");
    CB_ENSURE(L2Reg.Get() >= 0, "L2LeafRegularizer should be >= 0, current value: " << L2Reg.Get());
    CB_ENSURE(PairwiseNonDiagReg.Get() >= 0, "PairwiseNonDiagReg should be >= 0, current value: " << PairwiseNonDiagReg.Get());
}
