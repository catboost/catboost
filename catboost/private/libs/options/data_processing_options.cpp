#include "data_processing_options.h"
#include "json_helper.h"
#include "restrictions.h"
#include "metric_options.h"

#include <catboost/libs/helpers/exception.h>
#include <util/string/cast.h>

NCatboostOptions::TDataProcessingOptions::TDataProcessingOptions(ETaskType type)
    : IgnoredFeatures("ignored_features", TVector<ui32>())
      , HasTimeFlag("has_time", false)
      , AllowConstLabel("allow_const_label", false)
      , TargetBorder("target_border", Nothing())
      , FloatFeaturesBinarization("float_features_binarization", TBinarizationOptions(
          EBorderSelectionType::GreedyLogSum,
          type == ETaskType::GPU ? 128 : 254,
          ENanMode::Min,
          200000
      ))
      , PerFloatFeatureQuantization("per_float_feature_quantization", TMap<ui32, TBinarizationOptions>())
      , TextProcessingOptions("text_processing_options", TTextProcessingOptions())
      , EmbeddingProcessingOptions("embedding_processing_options", TEmbeddingProcessingOptions())
      , ClassesCount("classes_count", 0)
      , ClassWeights("class_weights", TVector<float>())
      , AutoClassWeights("auto_class_weights", EAutoClassWeightsType::None)
      , ClassLabels("class_names", TVector<NJson::TJsonValue>()) // "class_names" is used for compatibility
      , DevDefaultValueFractionToEnableSparseStorage("dev_default_value_fraction_for_sparse", 0.83f)
      , DevSparseArrayIndexingType("dev_sparse_array_indexing", NCB::ESparseArrayIndexingType::Indices)
      , ForceUnitAutoPairWeights("force_unit_auto_pair_weights", false)
      , EvalFraction("eval_fraction", 0.0f)
      , GpuCatFeaturesStorage("gpu_cat_features_storage", EGpuCatFeaturesStorage::GpuRam, type)
      , DevLeafwiseScoring("dev_leafwise_scoring", false, type)
      , DevGroupFeatures("dev_group_features", false, type)
{
    GpuCatFeaturesStorage.ChangeLoadUnimplementedPolicy(ELoadUnimplementedPolicy::SkipWithWarning);
    DevGroupFeatures.ChangeLoadUnimplementedPolicy(ELoadUnimplementedPolicy::SkipWithWarning);
}

void NCatboostOptions::TDataProcessingOptions::Load(const NJson::TJsonValue& options) {
    CheckedLoad(
        options, &IgnoredFeatures, &HasTimeFlag, &AllowConstLabel, &TargetBorder,
        &FloatFeaturesBinarization, &PerFloatFeatureQuantization,
        &TextProcessingOptions, &EmbeddingProcessingOptions,
        &ClassesCount, &ClassWeights, &AutoClassWeights, &ClassLabels,
        &DevDefaultValueFractionToEnableSparseStorage,
        &DevSparseArrayIndexingType, &ForceUnitAutoPairWeights, &EvalFraction,
        &GpuCatFeaturesStorage, &DevLeafwiseScoring, &DevGroupFeatures
    );
    Validate();
    SetPerFeatureMissingSettingToCommonValues();
}

void NCatboostOptions::TDataProcessingOptions::Save(NJson::TJsonValue* options) const {
    SaveFields(
        options, IgnoredFeatures, HasTimeFlag, AllowConstLabel, TargetBorder,
        FloatFeaturesBinarization, PerFloatFeatureQuantization,
        TextProcessingOptions, EmbeddingProcessingOptions,
        ClassesCount, ClassWeights, AutoClassWeights, ClassLabels,
        DevDefaultValueFractionToEnableSparseStorage,
        DevSparseArrayIndexingType, ForceUnitAutoPairWeights, EvalFraction,
        GpuCatFeaturesStorage, DevLeafwiseScoring, DevGroupFeatures
    );
}

bool NCatboostOptions::TDataProcessingOptions::operator==(const TDataProcessingOptions& rhs) const {
    return std::tie(IgnoredFeatures, HasTimeFlag, AllowConstLabel, TargetBorder,
                    FloatFeaturesBinarization, PerFloatFeatureQuantization,
                    TextProcessingOptions, EmbeddingProcessingOptions,
                    ClassesCount, ClassWeights, ClassLabels,
                    DevDefaultValueFractionToEnableSparseStorage,
                    DevSparseArrayIndexingType, ForceUnitAutoPairWeights, EvalFraction,
                    GpuCatFeaturesStorage, DevLeafwiseScoring,
                    DevGroupFeatures, AutoClassWeights) ==
           std::tie(rhs.IgnoredFeatures, rhs.HasTimeFlag, rhs.AllowConstLabel, rhs.TargetBorder,
                    rhs.FloatFeaturesBinarization, rhs.PerFloatFeatureQuantization,
                    rhs.TextProcessingOptions, rhs.EmbeddingProcessingOptions,
                    rhs.ClassesCount, rhs.ClassWeights, rhs.ClassLabels,
                    rhs.DevDefaultValueFractionToEnableSparseStorage,
                    rhs.DevSparseArrayIndexingType, rhs.ForceUnitAutoPairWeights, rhs.EvalFraction,
                    rhs.GpuCatFeaturesStorage, rhs.DevLeafwiseScoring,
                    rhs.DevGroupFeatures, rhs.AutoClassWeights);
}

bool NCatboostOptions::TDataProcessingOptions::operator!=(const TDataProcessingOptions& rhs) const {
    return !(rhs == *this);
}

void NCatboostOptions::TDataProcessingOptions::Validate() const {
    CB_ENSURE(
        (DevDefaultValueFractionToEnableSparseStorage.Get() >= 0.f) &&
        (DevDefaultValueFractionToEnableSparseStorage.Get() < 1.f),
        "DevDefaultValueFractionToEnableSparseStorage must be in [0, 1)"
    );
    CB_ENSURE(
        DevGroupFeatures.NotSet() || DevLeafwiseScoring.IsSet(),
        "DevGroupFeatures is supported only with DevLeafwiseScoring"
    );
    CB_ENSURE(AutoClassWeights.Get() == EAutoClassWeightsType::None || ClassWeights.IsDefault(),
        "ClassWeights should be default if AutoClassWeights is not None");
    CB_ENSURE((EvalFraction >= 0.0) && (EvalFraction < 1.0), "EvalFraction must be in [0, 1) range");
}


void NCatboostOptions::TDataProcessingOptions::SetPerFeatureMissingSettingToCommonValues() {
    if (!PerFloatFeatureQuantization.IsSet()) {
        return;
    }
    const auto& commonSettings = FloatFeaturesBinarization.Get();
    for (auto& [id, binarizationOption] : PerFloatFeatureQuantization.Get()) {
        Y_UNUSED(id);
        binarizationOption.BorderSelectionType.SetDefault(commonSettings.BorderSelectionType.GetDefaultValue());
        binarizationOption.BorderCount.SetDefault(commonSettings.BorderCount.GetDefaultValue());
        binarizationOption.NanMode.SetDefault(commonSettings.NanMode.GetDefaultValue());
        binarizationOption.MaxSubsetSizeForBuildBorders.SetDefault(commonSettings.MaxSubsetSizeForBuildBorders.GetDefaultValue());

        if (!binarizationOption.BorderCount.IsSet() && commonSettings.BorderCount.IsSet()) {
            binarizationOption.BorderCount = commonSettings.BorderCount;
        }
        if (!binarizationOption.BorderSelectionType.IsSet() && commonSettings.BorderSelectionType.IsSet()) {
            binarizationOption.BorderSelectionType = commonSettings.BorderSelectionType;
        }
        if (!binarizationOption.NanMode.IsSet() && commonSettings.NanMode.IsSet()) {
            binarizationOption.NanMode = commonSettings.NanMode;
        }
        if (!binarizationOption.MaxSubsetSizeForBuildBorders.IsSet() && commonSettings.MaxSubsetSizeForBuildBorders.IsSet()) {
            binarizationOption.MaxSubsetSizeForBuildBorders = commonSettings.MaxSubsetSizeForBuildBorders;
        }
    }
}
