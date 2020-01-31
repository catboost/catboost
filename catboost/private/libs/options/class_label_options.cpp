#include "class_label_options.h"
#include "json_helper.h"

TClassLabelOptions::TClassLabelOptions()
    : ClassLabelType("class_label_type", NCB::ERawTargetType::None)
    , ClassToLabel("class_to_label", TVector<float>())
    , ClassLabels("class_names", TVector<NJson::TJsonValue>()) // "class_names" historically because they were always strings before
    , ClassesCount("classes_count", 0) {
}

void TClassLabelOptions::Load(const NJson::TJsonValue& options) {
    NCatboostOptions::CheckedLoad(options, &ClassLabelType, &ClassToLabel, &ClassLabels, &ClassesCount);

    if (!ClassLabels.Get().empty()) {
        // compatibility with old "multiclass_params" without "class_label_type"
        if (ClassLabelType == NCB::ERawTargetType::None) {
            ClassLabelType = NCB::ERawTargetType::String;
        } else if (ClassLabelType == NCB::ERawTargetType::Float) {
            /* float values can be deserialized as JSON_INTEGER if they have 0 fractional part
             * because JSON format does not distinguish between integer and float numeric types
             * , fix this
             */
            for (NJson::TJsonValue& classLabel : ClassLabels.Get()) {
                if (classLabel.GetType() == NJson::JSON_INTEGER) {
                    classLabel = NJson::TJsonValue(double(classLabel.GetInteger()));
                }
            }
        }
    }

    Validate();
}

void TClassLabelOptions::Save(NJson::TJsonValue* options) const {
    NCatboostOptions::SaveFields(options, ClassLabelType, ClassToLabel, ClassLabels, ClassesCount);
}

void TClassLabelOptions::Validate() {
    CB_ENSURE(!ClassToLabel.Get().empty(), "ClassToLabel mapping must be not empty.");
}

bool TClassLabelOptions::operator==(const TClassLabelOptions& rhs) const {
    return std::tie(ClassLabelType, ClassToLabel, ClassLabels, ClassesCount) ==
        std::tie(rhs.ClassLabelType, rhs.ClassToLabel, rhs.ClassLabels, rhs.ClassesCount);
}

bool TClassLabelOptions::operator!=(const TClassLabelOptions& rhs) const {
    return !(rhs == *this);
}
