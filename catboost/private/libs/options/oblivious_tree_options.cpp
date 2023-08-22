#include "enum_helpers.h"
#include "json_helper.h"
#include "oblivious_tree_options.h"

#include <catboost/libs/logging/logging_level.h>
#include <catboost/libs/logging/logging.h>

#include <library/cpp/json/json_value.h>
#include <util/string/cast.h>

NCatboostOptions::TObliviousTreeLearnerOptions::TObliviousTreeLearnerOptions(ETaskType taskType)
    : MaxDepth("depth", 6)
      , LeavesEstimationIterations("leaf_estimation_iterations", 1)
      , LeavesEstimationMethod("leaf_estimation_method", ELeavesEstimation::Gradient)
      , L2Reg("l2_leaf_reg", 3.0)
      , PairwiseNonDiagReg("bayesian_matrix_reg", 0.1)
      , RandomStrength("random_strength", 1.0)
      , RandomScoreType("random_score_type", ERandomScoreType::NormalWithModelSizeDecrease)
      , BootstrapConfig("bootstrap", TBootstrapConfig(taskType))
      , Rsm("rsm", 1.0)
      , LeavesEstimationBacktrackingType("leaf_estimation_backtracking", ELeavesEstimationStepBacktracking::AnyImprovement)
      , ScoreFunction("score_function", EScoreFunction::Cosine)
      , GrowPolicy("grow_policy", EGrowPolicy::SymmetricTree)
      , MaxLeaves("max_leaves", 31)
      , MinDataInLeaf("min_data_in_leaf", 1)
      , DevExclusiveFeaturesBundleMaxBuckets("dev_efb_max_buckets", taskType == ETaskType::CPU ? 1 << 10 : 254)
      , SamplingFrequency("sampling_frequency", ESamplingFrequency::PerTree, taskType)
      , ModelSizeReg("model_size_reg", 0.5f)
      , DevScoreCalcObjBlockSize("dev_score_calc_obj_block_size", 5000000, taskType)
      , SparseFeaturesConflictFraction("sparse_features_conflict_fraction", 0.0f, taskType)
      , ObservationsToBootstrap("observations_to_bootstrap", EObservationsToBootstrap::TestOnly, taskType) //it's specific for fold-based scheme, so here and not in bootstrap options
      , FoldSizeLossNormalization("fold_size_loss_normalization", false, taskType)
      , AddRidgeToTargetFunctionFlag("add_ridge_penalty_to_loss_function", false, taskType)
      , MaxCtrComplexityForBordersCaching("dev_max_ctr_complexity_for_borders_cache", 1, taskType)
      , MetaL2Exponent("meta_l2_exponent", 1.0, taskType)
      , MetaL2Frequency("meta_l2_frequency", 0.0, taskType)
      , FixedBinarySplits("fixed_binary_splits", {}, taskType)
      , MonotoneConstraints("monotone_constraints", {}, taskType)
      , DevLeafwiseApproxes("dev_leafwise_approxes", false, taskType)
      , FeaturePenalties("penalties", TFeaturePenaltiesOptions())
      , TaskType("task_type", taskType)
{
    SamplingFrequency.ChangeLoadUnimplementedPolicy(ELoadUnimplementedPolicy::ExceptionOnChange);

    FoldSizeLossNormalization.ChangeLoadUnimplementedPolicy(ELoadUnimplementedPolicy::ExceptionOnChange);
    AddRidgeToTargetFunctionFlag.ChangeLoadUnimplementedPolicy(ELoadUnimplementedPolicy::ExceptionOnChange);

    MaxCtrComplexityForBordersCaching.ChangeLoadUnimplementedPolicy(ELoadUnimplementedPolicy::SkipWithWarning);
}

void NCatboostOptions::TObliviousTreeLearnerOptions::Load(const NJson::TJsonValue& options) {
    CheckedLoad(options,
            &MaxDepth, &LeavesEstimationIterations, &LeavesEstimationMethod, &L2Reg, &MetaL2Exponent, &MetaL2Frequency, &ModelSizeReg,
            &RandomStrength, &RandomScoreType,
            &BootstrapConfig, &FoldSizeLossNormalization, &AddRidgeToTargetFunctionFlag,
            &ScoreFunction,
            &GrowPolicy,
            &MaxLeaves,
            &MinDataInLeaf,
            &MaxCtrComplexityForBordersCaching,
            &Rsm,
            &ObservationsToBootstrap,
            &PairwiseNonDiagReg,
            &LeavesEstimationBacktrackingType,
            &SamplingFrequency,
            &DevScoreCalcObjBlockSize,
            &DevExclusiveFeaturesBundleMaxBuckets,
            &SparseFeaturesConflictFraction,
            &FixedBinarySplits,
            &MonotoneConstraints,
            &DevLeafwiseApproxes,
            &FeaturePenalties
            );

    Validate();
}

void NCatboostOptions::TObliviousTreeLearnerOptions::Save(NJson::TJsonValue* options) const {
    SaveFields(options, MaxDepth, LeavesEstimationIterations, LeavesEstimationMethod, L2Reg, MetaL2Exponent, MetaL2Frequency, ModelSizeReg,
            RandomStrength, RandomScoreType,
            BootstrapConfig, FoldSizeLossNormalization, AddRidgeToTargetFunctionFlag,
            ScoreFunction,
            GrowPolicy,
            MaxLeaves,
            MinDataInLeaf,
            PairwiseNonDiagReg,
            LeavesEstimationBacktrackingType,
            MaxCtrComplexityForBordersCaching, Rsm, ObservationsToBootstrap, SamplingFrequency,
            DevScoreCalcObjBlockSize,
            DevExclusiveFeaturesBundleMaxBuckets,
            SparseFeaturesConflictFraction,
            FixedBinarySplits,
            MonotoneConstraints,
            DevLeafwiseApproxes,
            FeaturePenalties
            );
}

bool NCatboostOptions::TObliviousTreeLearnerOptions::operator==(const TObliviousTreeLearnerOptions& rhs) const {
    return std::tie(MaxDepth, LeavesEstimationIterations, LeavesEstimationMethod, L2Reg, MetaL2Exponent, MetaL2Frequency, ModelSizeReg,
            RandomStrength, RandomScoreType,
            BootstrapConfig, Rsm, SamplingFrequency, ObservationsToBootstrap, FoldSizeLossNormalization,
            AddRidgeToTargetFunctionFlag, ScoreFunction, GrowPolicy, MaxLeaves, MinDataInLeaf, MaxCtrComplexityForBordersCaching,
            PairwiseNonDiagReg, LeavesEstimationBacktrackingType, DevScoreCalcObjBlockSize,
            DevExclusiveFeaturesBundleMaxBuckets, SparseFeaturesConflictFraction, FixedBinarySplits,
            MonotoneConstraints, DevLeafwiseApproxes, FeaturePenalties
            ) ==
        std::tie(rhs.MaxDepth, rhs.LeavesEstimationIterations, rhs.LeavesEstimationMethod, rhs.L2Reg, rhs.MetaL2Exponent, rhs.MetaL2Frequency, rhs.ModelSizeReg,
                rhs.RandomStrength, rhs.RandomScoreType,
                rhs.BootstrapConfig, rhs.Rsm, rhs.SamplingFrequency,
                rhs.ObservationsToBootstrap, rhs.FoldSizeLossNormalization, rhs.AddRidgeToTargetFunctionFlag,
                rhs.ScoreFunction, rhs.GrowPolicy, rhs.MaxLeaves, rhs.MinDataInLeaf, rhs.MaxCtrComplexityForBordersCaching,
                rhs.PairwiseNonDiagReg, rhs.LeavesEstimationBacktrackingType, rhs.DevScoreCalcObjBlockSize,
                rhs.DevExclusiveFeaturesBundleMaxBuckets, rhs.SparseFeaturesConflictFraction,
                rhs.FixedBinarySplits, rhs.MonotoneConstraints, rhs.DevLeafwiseApproxes, rhs.FeaturePenalties);
}

bool NCatboostOptions::TObliviousTreeLearnerOptions::operator!=(const TObliviousTreeLearnerOptions& rhs) const {
    return !(rhs == *this);
}

void NCatboostOptions::TObliviousTreeLearnerOptions::Validate() const {
    BootstrapConfig.Get().Validate();
    const float rsm = Rsm.Get();
    CB_ENSURE(rsm > 0 && rsm <= 1, "Rsm should be in (0, 1]");
    const ui32 maxFullBinaryTreeDepth = 16;
    if (IsBuildingFullBinaryTree(GrowPolicy)) {
        CB_ENSURE(MaxDepth.Get() <= maxFullBinaryTreeDepth, "Maximum tree depth is " << maxFullBinaryTreeDepth);
    }
    if (GrowPolicy == EGrowPolicy::Lossguide) {
        const ui32 maxLeavesCount = 1 << 16;
        CB_ENSURE(MaxLeaves.Get() <= maxLeavesCount, "Maximum leaves count for Lossguide grow policy is " << maxLeavesCount);
    }
    CB_ENSURE(DevScoreCalcObjBlockSize.GetUnchecked() > 0, "DevScoreCalcObjBlockSize must be > 0");
    CB_ENSURE(DevExclusiveFeaturesBundleMaxBuckets.Get() < (1U << 16), "DevExclusiveFeaturesBundleMaxBuckets must be less than 65536");
    CB_ENSURE(
        (SparseFeaturesConflictFraction.GetUnchecked() >= 0.f) && (SparseFeaturesConflictFraction.GetUnchecked() < 1.f),
        "SparseFeaturesConflictFraction should be in [0, 1)"
    );
    CB_ENSURE(LeavesEstimationIterations.Get() > 0, "Leaves estimation iterations should be positive");
    CB_ENSURE(L2Reg.Get() >= 0, "L2LeafRegularizer should be >= 0, current value: " << L2Reg.Get());
    CB_ENSURE(PairwiseNonDiagReg.Get() >= 0, "PairwiseNonDiagReg should be >= 0, current value: " << PairwiseNonDiagReg.Get());
    CB_ENSURE(
        TaskType.Get() == ETaskType::GPU || EqualToOneOf(ScoreFunction, EScoreFunction::Cosine, EScoreFunction::L2),
        "Only Cosine and L2 score functions are supported for CPU."
    );

    // TODO(akhropov): Implement ERandomScoreType::Gumbel for GPU
    CB_ENSURE(
        TaskType.Get() == ETaskType::CPU || (RandomScoreType == ERandomScoreType::NormalWithModelSizeDecrease),
        "random_score_type must be NormalWithModelSizeDecrease for GPU"
    );
}
