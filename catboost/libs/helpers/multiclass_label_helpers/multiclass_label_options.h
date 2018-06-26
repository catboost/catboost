#pragma once
#include <catboost/libs/options/option.h>
#include <catboost/libs/options/json_helper.h>

struct TMulticlassLabelOptions {
public:
    explicit TMulticlassLabelOptions()
            : ClassToLabel("class_to_label", TVector<float>())
              , ClassNames("class_names", TVector<TString>())
              , ClassesCount("classes_count", 0) {
    }

    void Load(const NJson::TJsonValue& options) {
        NCatboostOptions::CheckedLoad(options, &ClassToLabel, &ClassNames, &ClassesCount);
        Validate();
    }

    void Save(NJson::TJsonValue* options) const {
        NCatboostOptions::SaveFields(options, ClassToLabel, ClassNames, ClassesCount);
    }

    void Validate() {
        CB_ENSURE(!ClassToLabel.Get().empty(), "ClassToLabel mapping must be not empty.");
    }

    bool operator==(const TMulticlassLabelOptions& rhs) const {
        return std::tie(ClassToLabel, ClassNames, ClassesCount) ==
               std::tie(rhs.ClassToLabel, rhs.ClassNames, rhs.ClassesCount);
    }

    bool operator!=(const TMulticlassLabelOptions& rhs) const {
        return !(rhs == *this);
    }

    NCatboostOptions::TOption<TVector<float>> ClassToLabel;
    NCatboostOptions::TOption<TVector<TString>> ClassNames;
    NCatboostOptions::TOption<int> ClassesCount;
};
