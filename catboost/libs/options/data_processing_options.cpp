#include "data_processing_options.h"
#include "json_helper.h"
#include "restrictions.h"


NCatboostOptions::TDataProcessingOptions::TDataProcessingOptions(ETaskType type)
    : IgnoredFeatures("ignored_features", TVector<ui32>())
      , HasTimeFlag("has_time", false)
      , AllowConstLabel("allow_const_label", false)
      , TargetBorder("target_border", Nothing())
      , FloatFeaturesBinarization("float_features_binarization", TBinarizationOptions(
          EBorderSelectionType::GreedyLogSum,
          type == ETaskType::GPU ? 128 : 254,
          ENanMode::Min,
          type
      ))
      , PerFloatFeatureBinarization("per_float_feature_binarization", TMap<ui32, TBinarizationOptions>())
      , TextProcessing("text_processing", TTextProcessingOptionCollection())
      , ClassesCount("classes_count", 0)
      , ClassWeights("class_weights", TVector<float>())
      , ClassNames("class_names", TVector<TString>())
      , GpuCatFeaturesStorage("gpu_cat_features_storage", EGpuCatFeaturesStorage::GpuRam, type)
{
    GpuCatFeaturesStorage.ChangeLoadUnimplementedPolicy(ELoadUnimplementedPolicy::SkipWithWarning);
}

void NCatboostOptions::TDataProcessingOptions::Load(const NJson::TJsonValue& options) {
    CheckedLoad(
        options, &IgnoredFeatures, &HasTimeFlag, &AllowConstLabel, &TargetBorder,
        &FloatFeaturesBinarization, &PerFloatFeatureBinarization, &TextProcessing,
        &ClassesCount, &ClassWeights, &ClassNames, &GpuCatFeaturesStorage
    );
    SetPerFeatureMissingSettingToCommonValues();
}

void NCatboostOptions::TDataProcessingOptions::Save(NJson::TJsonValue* options) const {
    SaveFields(
        options, IgnoredFeatures, HasTimeFlag, AllowConstLabel, TargetBorder,
        FloatFeaturesBinarization, PerFloatFeatureBinarization, TextProcessing,
        ClassesCount, ClassWeights, ClassNames, GpuCatFeaturesStorage
    );
}

bool NCatboostOptions::TDataProcessingOptions::operator==(const TDataProcessingOptions& rhs) const {
    return std::tie(IgnoredFeatures, HasTimeFlag, AllowConstLabel, TargetBorder,
            FloatFeaturesBinarization, PerFloatFeatureBinarization, TextProcessing,
            ClassesCount, ClassWeights, ClassNames, GpuCatFeaturesStorage) ==
        std::tie(rhs.IgnoredFeatures, rhs.HasTimeFlag, rhs.AllowConstLabel, rhs.TargetBorder,
                rhs.FloatFeaturesBinarization, rhs.PerFloatFeatureBinarization, rhs.TextProcessing,
                rhs.ClassesCount, rhs.ClassWeights, rhs.ClassNames, rhs.GpuCatFeaturesStorage);
}

bool NCatboostOptions::TDataProcessingOptions::operator!=(const TDataProcessingOptions& rhs) const {
    return !(rhs == *this);
}

void NCatboostOptions::TDataProcessingOptions::SetPerFeatureMissingSettingToCommonValues() {
    if (!PerFloatFeatureBinarization.IsSet()) {
        return;
    }
    const auto& commonSettings = FloatFeaturesBinarization.Get();
    for (auto& [id, binarizationOption] : PerFloatFeatureBinarization.Get()) {
        Y_UNUSED(id);
        if (!binarizationOption.BorderCount.IsSet() && commonSettings.BorderCount.IsSet()) {
            binarizationOption.BorderCount = commonSettings.BorderCount;
        }
        if (!binarizationOption.BorderSelectionType.IsSet() && commonSettings.BorderSelectionType.IsSet()) {
            binarizationOption.BorderSelectionType = commonSettings.BorderSelectionType;
        }
        if (!binarizationOption.NanMode.IsSet() && commonSettings.NanMode.IsSet()) {
            binarizationOption.NanMode = commonSettings.NanMode;
        }
    }
}
