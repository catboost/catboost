#include "helpers.h"

#include <catboost/libs/helpers/exception.h>

#include <library/json/json_value.h>

#include <util/string/cast.h>
#include <util/system/compiler.h>


namespace NCB {
    ERawTargetType GetRawTargetType(const NJson::TJsonValue& classLabel) {
        switch (classLabel.GetType()) {
            case NJson::JSON_INTEGER:
                return ERawTargetType::Integer;
            case NJson::JSON_DOUBLE:
                return ERawTargetType::Float;
            case NJson::JSON_STRING:
                return ERawTargetType::String;
            default:
                CB_ENSURE_INTERNAL(false, "bad class label type: " << classLabel.GetType());
        }
        Y_UNREACHABLE();
    }

    TString ClassLabelToString(const NJson::TJsonValue& classLabel) {
        switch (classLabel.GetType()) {
            case NJson::JSON_INTEGER:
                return ToString(classLabel.GetInteger());
            case NJson::JSON_DOUBLE:
                return ToString(classLabel.GetDouble());
            case NJson::JSON_STRING:
                return classLabel.GetString();
            default:
                CB_ENSURE_INTERNAL(false, "bad class label type: " << classLabel.GetType());
        }
        Y_UNREACHABLE();
    }

    TVector<TString> ClassLabelsToStrings(TConstArrayRef<NJson::TJsonValue> classLabels) {
        TVector<TString> result;
        result.reserve(classLabels.size());
        for (const auto& classLabel : classLabels) {
            result.push_back(ClassLabelToString(classLabel));
        }
        return result;
    }

}
