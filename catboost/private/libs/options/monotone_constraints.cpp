#include "monotone_constraints.h"

#include <catboost/libs/column_description/cd_parser.h>

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

static void ConvertFeatureNamesToIndicesInMonotoneConstraints(const TMap<TString, ui32>& indicesFromNames, TJsonValue* treeOptions) {
    if (indicesFromNames.empty()) {
        return;
    }
    auto& constraintsRef = (*treeOptions)["monotone_constraints"];
    Y_ASSERT(constraintsRef.GetType() == EJsonValueType::JSON_MAP);
    bool needConvert = AnyOf(constraintsRef.GetMap(), [](auto element) {
        int index = 0;
        return !TryFromString(element.first, index);
    });
    if (needConvert) {
        TJsonValue constraintsWithIndices(EJsonValueType::JSON_MAP);
        for (const auto& [featureName, value] : constraintsRef.GetMap()) {
            CB_ENSURE(indicesFromNames.contains(featureName), "Unknown feature name in monotone constraints: " << featureName);
            constraintsWithIndices.InsertValue(ToString(indicesFromNames.at(featureName)), value);
        }
        constraintsRef.Swap(constraintsWithIndices);
    }
}


void ConvertFeatureNamesToIndicesInMonotoneConstraints(const TDataMetaInfo& metaInfo, TJsonValue* catBoostJsonOptions) {
    auto& treeOptions = (*catBoostJsonOptions)["tree_learner_options"];
    if (!treeOptions.Has("monotone_constraints")) {
        return;
    }

    TMap<TString, ui32> indicesFromNames;
    ui32 columnIdx = 0;
    for (const auto& columnInfo : metaInfo.FeaturesLayout->GetExternalFeaturesMetaInfo()) {
        if (!columnInfo.Name.empty()) {
            indicesFromNames[columnInfo.Name] = columnIdx;
        }
        columnIdx++;
    }
    ConvertFeatureNamesToIndicesInMonotoneConstraints(indicesFromNames, &treeOptions);
}

void ConvertFeatureNamesToIndicesInMonotoneConstraints(const NCatboostOptions::TPoolLoadParams& poolLoadParams, TJsonValue* catBoostJsonOptions) {
    auto& treeOptions = (*catBoostJsonOptions)["tree_learner_options"];
    if (!treeOptions.Has("monotone_constraints")) {
        return;
    }

    TMap<TString, ui32> indicesFromNames;
    if (poolLoadParams.ColumnarPoolFormatParams.CdFilePath.Inited()) {
        const TVector<TColumn> columns = ReadCD(poolLoadParams.ColumnarPoolFormatParams.CdFilePath,
                                                TCdParserDefaults(EColumn::Num));
        ui32 featureIdx = 0;
        for (const auto& column : columns) {
            if (IsFactorColumn(column.Type)) {
                if (!column.Id.empty()) {
                    indicesFromNames[column.Id] = featureIdx;
                }
                featureIdx++;
            }
        }
    }
    ConvertFeatureNamesToIndicesInMonotoneConstraints(indicesFromNames, &treeOptions);
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
