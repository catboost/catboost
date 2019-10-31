#include "monotone_constraints.h"

#include <util/generic/strbuf.h>
#include <util/string/cast.h>
#include <util/string/split.h>

#include <regex>

using namespace NCB;
using namespace NJson;

TMap<TString, int> ParseMonotonicConstraintsFromString(const TString& monotoneConstraints) {
    TMap<TString, int> constraintsAsMap;
    std::regex denseFormat("^\\((0|1|-1)(,(0|1|-1))*\\)$");           // like: (1,0,0,-1,0)
    std::regex sparseFormat("^[^:,]+:(0|1|-1)(,[^:,]+:(0|1|-1))*$");  // like: 0:1,3:-1 or FeatureName1:-1,FeatureName2:-1
    if (std::regex_match(monotoneConstraints.data(), denseFormat)) {
        ui32 featureIdx = 0;
        auto constraints = StringSplitter(monotoneConstraints.substr(1, monotoneConstraints.size() - 2))
            .Split(',')
            .SkipEmpty();
        for (const auto& constraint : constraints) {
            const auto value = FromString<int>(constraint.Token());
            if (value != 0) {
                constraintsAsMap[ToString(featureIdx)] = value;
            }
            featureIdx++;
        }
    } else if (std::regex_match(monotoneConstraints.data(), sparseFormat)) {
        for (const auto& oneFeatureMonotonic : StringSplitter(monotoneConstraints).Split(',').SkipEmpty()) {
            auto parts = StringSplitter(oneFeatureMonotonic.Token()).Split(':');
            const TString feature(parts.Next()->Token());
            const auto value = FromString<int>(parts.Next()->Token());
            if (value != 0) {
                constraintsAsMap[feature] = value;
            }
        }
    } else {
        CB_ENSURE(false,
            "Incorrect format of monotone constraints. Possible formats: \"(1,0,0,-1)\", \"0:1,3:-1\", \"FeatureName1:-1,FeatureName2:-1\".");
    }
    return constraintsAsMap;
}


static int GetConstraintValue(const TJsonValue& constraint) {
    if (constraint.IsInteger()) {
        return constraint.GetInteger();
    }
    if (constraint.IsString()) {
        return FromString<int>(constraint.GetString());
    }
    CB_ENSURE(false, "Incorrect format of monotone constraints");
}


void ConvertMonotoneConstraintsToCanonicalFormat(TJsonValue* catBoostJsonOptions) {
    auto& treeOptions = (*catBoostJsonOptions)["tree_learner_options"];
    if (!treeOptions.Has("monotone_constraints")) {
        return;
    }
    TJsonValue& constraintsRef = treeOptions["monotone_constraints"];
    TJsonValue canonicalConstraints(EJsonValueType::JSON_MAP);
    switch (constraintsRef.GetType()) {
        case NJson::EJsonValueType::JSON_STRING: {
            TMap<TString, int> constraintsAsMap = ParseMonotonicConstraintsFromString(constraintsRef.GetString());
            for (const auto& [key, value] : constraintsAsMap) {
                canonicalConstraints.InsertValue(key, value);
            }
        }
            break;
        case NJson::EJsonValueType::JSON_ARRAY: {
            ui32 featureIdx = 0;
            for (const auto& constraint : constraintsRef.GetArray()) {
                int value = GetConstraintValue(constraint);
                if (value != 0) {
                    canonicalConstraints.InsertValue(ToString(featureIdx), value);
                }
                featureIdx++;
            }
        }
            break;
        case NJson::EJsonValueType::JSON_MAP: {
            TMap<TString, int> constraintsAsMap;
            for (const auto& [feature, constraint] : constraintsRef.GetMap()) {
                int value = GetConstraintValue(constraint);
                if (value != 0) {
                    canonicalConstraints.InsertValue(feature, value);
                }
            }
        }
            break;
        default:
            CB_ENSURE(false, "Incorrect format of monotone constraints");
    }

    constraintsRef = canonicalConstraints;
}
