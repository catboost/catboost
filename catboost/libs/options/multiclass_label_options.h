#pragma once

#include "option.h"
#include "json_helper.h"


namespace NCatboostOptions {
    struct TMulticlassLabelOptions {
    public:
        explicit TMulticlassLabelOptions()
            : ClassToLabel("class_to_label", TVector<float>())
            , ClassNames("class_names", TVector<TString>())
            , ClassesCount("classes_count", 0) {
        }

        void Load(const NJson::TJsonValue& options) {
            CheckedLoad(options, &ClassToLabel, &ClassNames, &ClassesCount);
            Validate();
        }

        void Save(NJson::TJsonValue* options) const {
            SaveFields(options, ClassToLabel, ClassNames, ClassesCount);
        }

        void Validate() {
            CB_ENSURE(ClassesCount.Get() > 0 || ClassNames.Get().ysize() > 0 || ClassToLabel.Get().ysize() > 0,
                      "At least one parameter must be non-default");
        }

        bool operator==(const TMulticlassLabelOptions& rhs) const {
            return std::tie(ClassToLabel, ClassNames, ClassesCount) ==
                   std::tie(rhs.ClassToLabel, rhs.ClassNames, rhs.ClassesCount);
        }

        bool operator!=(const TMulticlassLabelOptions& rhs) const {
            return !(rhs == *this);
        }

        TOption<TVector<float>> ClassToLabel;
        TOption<TVector<TString>> ClassNames;
        TOption<int> ClassesCount;
    };
}
