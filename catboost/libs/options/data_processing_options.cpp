#include "data_processing_options.h"
#include "json_helper.h"
#include "restrictions.h"

NCatboostOptions::TDataProcessingOptions::TDataProcessingOptions(ETaskType type)
    : IgnoredFeatures("ignored_features", TVector<ui32>())
      , HasTimeFlag("has_time", false)
      , AllowConstLabel("allow_const_label", false)
      , FloatFeaturesBinarization("float_features_binarization", TBinarizationOptions(
            EBorderSelectionType::GreedyLogSum, type == ETaskType::GPU ? 128 : 254, ENanMode::Min))
      , ClassesCount("classes_count", 0)
      , ClassWeights("class_weights", TVector<float>())
      , ClassNames("class_names", TVector<TString>())
      , GpuCatFeaturesStorage("gpu_cat_features_storage", EGpuCatFeaturesStorage::GpuRam, type)
{
    GpuCatFeaturesStorage.ChangeLoadUnimplementedPolicy(ELoadUnimplementedPolicy::SkipWithWarning);
}

void NCatboostOptions::TDataProcessingOptions::Load(const NJson::TJsonValue& options) {
    CheckedLoad(options, &IgnoredFeatures, &HasTimeFlag, &AllowConstLabel, &FloatFeaturesBinarization, &ClassesCount, &ClassWeights, &ClassNames, &GpuCatFeaturesStorage);
    CB_ENSURE(FloatFeaturesBinarization->BorderCount <= GetMaxBinCount(), "Error: catboost doesn't support binarization with >= 256 levels");
}

void NCatboostOptions::TDataProcessingOptions::Save(NJson::TJsonValue* options) const {
    SaveFields(options, IgnoredFeatures, HasTimeFlag, AllowConstLabel, FloatFeaturesBinarization, ClassesCount, ClassWeights, ClassNames, GpuCatFeaturesStorage);
}

bool NCatboostOptions::TDataProcessingOptions::operator==(const TDataProcessingOptions& rhs) const {
    return std::tie(IgnoredFeatures, HasTimeFlag, AllowConstLabel, FloatFeaturesBinarization, ClassesCount, ClassWeights,
            ClassNames, GpuCatFeaturesStorage) ==
        std::tie(rhs.IgnoredFeatures, rhs.HasTimeFlag, rhs.AllowConstLabel, rhs.FloatFeaturesBinarization, rhs.ClassesCount,
                rhs.ClassWeights, rhs.ClassNames, rhs.GpuCatFeaturesStorage);
}

bool NCatboostOptions::TDataProcessingOptions::operator!=(const TDataProcessingOptions& rhs) const {
    return !(rhs == *this);
}
