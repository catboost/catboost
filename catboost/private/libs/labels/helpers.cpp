#include "helpers.h"

#include <catboost/libs/helpers/exception.h>

#include <util/string/cast.h>
#include <util/system/compiler.h>


namespace NCB {
    ERawTargetType GetRawTargetType(const NJson::TJsonValue& classLabel) {
        switch (classLabel.GetType()) {
            case NJson::JSON_BOOLEAN:
                return ERawTargetType::Boolean;
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
        static TString trueStr("true");
        static TString falseStr("false");

        switch (classLabel.GetType()) {
            case NJson::JSON_BOOLEAN:
                return classLabel.GetBoolean() ? trueStr : falseStr;
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

    void MaybeAddPhantomSecondClass(TVector<NJson::TJsonValue>* classLabels) {
        if (classLabels->empty()) {
            return;
        }
        CB_ENSURE_INTERNAL(classLabels->size() == 1, "MaybeAddPhantomSecondClass: classLabels's size != 1");

        const auto classLabel0 = classLabels->front();

        switch (classLabel0.GetType()) {
            case NJson::JSON_BOOLEAN:
                *classLabels = {NJson::TJsonValue(false), NJson::TJsonValue(true)};
                break;
            case NJson::JSON_INTEGER:
                classLabels->push_back(NJson::TJsonValue(classLabel0.GetInteger() + 1));
                break;
            case NJson::JSON_DOUBLE:
                classLabels->push_back(NJson::TJsonValue(classLabel0.GetDouble() + 1.0));
                break;
            case NJson::JSON_STRING:
                classLabels->push_back(NJson::TJsonValue(classLabel0.GetString() + "_2"));
                break;
            default:
                CB_ENSURE_INTERNAL(false, "bad class label type: " << classLabel0.GetType());
        }
    }

    void CheckBooleanClassLabels(TConstArrayRef<NJson::TJsonValue> booleanClassLabels) {
        CB_ENSURE_INTERNAL(
            booleanClassLabels.size() == 2,
            "Boolean target can have only exactly two classes"
        );
        CB_ENSURE_INTERNAL(
            booleanClassLabels[0].GetBoolean() == false,
            "Expected class label 0 to be 'false'"
        );
        CB_ENSURE_INTERNAL(
            booleanClassLabels[1].GetBoolean() == true,
            "Expected lass label 1 to be 'true'"
        );
    }

}
