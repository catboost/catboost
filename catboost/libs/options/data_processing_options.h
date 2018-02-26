#pragma once

#include "option.h"
#include "json_helper.h"
#include "binarization_options.h"
#include "restrictions.h"
#include <library/grid_creator/binarization.h>

namespace NCatboostOptions {

    struct TDataProcessingOptions {
        explicit TDataProcessingOptions(ETaskType type)
            : IgnoredFeatures("ignored_features", TVector<int>())
            , HasTimeFlag("has_time", false)
            , FloatFeaturesBinarization("float_features_binarization", TBinarizationOptions(EBorderSelectionType::GreedyLogSum, 128, ENanMode::Min))
            , ClassesCount("classes_count", 0)
            , ClassWeights("class_weights", TVector<float>())
            , ClassNames("class_names", TVector<TString>())
            , GpuCatFeaturesStorage("gpu_cat_features_storage", EGpuCatFeaturesStorage::GpuRam, type)
        {
            GpuCatFeaturesStorage.ChangeLoadUnimplementedPolicy(ELoadUnimplementedPolicy::SkipWithWarning);
        }

        void Load(const NJson::TJsonValue& options) {
            CheckedLoad(options, &IgnoredFeatures, &HasTimeFlag, &FloatFeaturesBinarization, &ClassesCount, &ClassWeights, &ClassNames, &GpuCatFeaturesStorage);
            CB_ENSURE(FloatFeaturesBinarization->BorderCount <= GetMaxBinCount(), "Error: catboost doesn't support binarization with >= 256 levels");
        }

        void Save(NJson::TJsonValue* options) const {
            SaveFields(options, IgnoredFeatures, HasTimeFlag, FloatFeaturesBinarization, ClassesCount, ClassWeights, ClassNames, GpuCatFeaturesStorage);
        }

        bool operator==(const TDataProcessingOptions& rhs) const {
            return std::tie(IgnoredFeatures, HasTimeFlag, FloatFeaturesBinarization, ClassesCount, ClassWeights,
                            ClassNames, GpuCatFeaturesStorage) ==
                   std::tie(rhs.IgnoredFeatures, rhs.HasTimeFlag, rhs.FloatFeaturesBinarization, rhs.ClassesCount,
                            rhs.ClassWeights, rhs.ClassNames, rhs.GpuCatFeaturesStorage);
        }

        bool operator!=(const TDataProcessingOptions& rhs) const {
            return !(rhs == *this);
        }

        TOption<TVector<int>> IgnoredFeatures;
        TOption<bool> HasTimeFlag;
        TOption<TBinarizationOptions> FloatFeaturesBinarization;
        TOption<ui32> ClassesCount;
        TOption<TVector<float>> ClassWeights;
        TOption<TVector<TString>> ClassNames;
        TGpuOnlyOption<EGpuCatFeaturesStorage> GpuCatFeaturesStorage;
    };

}
