#include "multiclass_label_options.h"
#include "json_helper.h"

TMulticlassLabelOptions::TMulticlassLabelOptions()
    : ClassToLabel("class_to_label", TVector<float>())
    , ClassNames("class_names", TVector<TString>())
    , ClassesCount("classes_count", 0) {
}

void TMulticlassLabelOptions::Load(const NJson::TJsonValue& options) {
    NCatboostOptions::CheckedLoad(options, &ClassToLabel, &ClassNames, &ClassesCount);
    Validate();
}

void TMulticlassLabelOptions::Save(NJson::TJsonValue* options) const {
    NCatboostOptions::SaveFields(options, ClassToLabel, ClassNames, ClassesCount);
}

void TMulticlassLabelOptions::Validate() {
    CB_ENSURE(!ClassToLabel.Get().empty(), "ClassToLabel mapping must be not empty.");
}

bool TMulticlassLabelOptions::operator==(const TMulticlassLabelOptions& rhs) const {
    return std::tie(ClassToLabel, ClassNames, ClassesCount) ==
        std::tie(rhs.ClassToLabel, rhs.ClassNames, rhs.ClassesCount);
}

bool TMulticlassLabelOptions::operator!=(const TMulticlassLabelOptions& rhs) const {
    return !(rhs == *this);
}
