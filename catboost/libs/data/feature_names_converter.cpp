#include "feature_names_converter.h"

#include <catboost/libs/column_description/cd_parser.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/private/libs/options/json_helper.h>

#include <util/string/split.h>
#include <util/string/type.h>

static bool TryParseRange(const TString& ignoredFeatureDescription, ui32& left, ui32& right) {
    return StringSplitter(ignoredFeatureDescription).Split('-').TryCollectInto(&left, &right);
}

static bool IsNumberOrRange(const TString& ignoredFeatureDescription) {
    ui32 left, right;
    return IsNumber(ignoredFeatureDescription) || TryParseRange(ignoredFeatureDescription, left, right);
}

static void ConvertStringIndicesIntoIntegerIndices(NJson::TJsonValue* ignoredFeaturesJson) {
    NJson::TJsonValue ignoredFeaturesIndicesJson(NJson::JSON_ARRAY);
    for (NJson::TJsonValue ignoredFeature : ignoredFeaturesJson->GetArray()) {
        if (IsNumber(ignoredFeature.GetString())) {
            ignoredFeaturesIndicesJson.AppendValue(FromString<ui32>(ignoredFeature.GetString()));
        } else {
            ui32 left, right;
            CB_ENSURE_INTERNAL(TryParseRange(ignoredFeature.GetString(), left, right), "Bad feature range");
            for (ui32 index = left; index <= right; index++) {
                ignoredFeaturesIndicesJson.AppendValue(index);
            }
        }
    }
    ignoredFeaturesJson->Swap(ignoredFeaturesIndicesJson);
}

static void ConvertNamesIntoIndices(const TMap<TString, ui32>& indicesFromNames, NJson::TJsonValue* ignoredFeaturesJson) {
    NJson::TJsonValue ignoredFeaturesIndicesJson(NJson::JSON_ARRAY);
    for (NJson::TJsonValue ignoredFeature : ignoredFeaturesJson->GetArray()) {
        CB_ENSURE(indicesFromNames.contains(ignoredFeature.GetString()), "There is no feature with name '" + ignoredFeature.GetString() + "' in dataset");
        ignoredFeaturesIndicesJson.AppendValue(indicesFromNames.at(ignoredFeature.GetString()));
    }
    ignoredFeaturesJson->Swap(ignoredFeaturesIndicesJson);
}

static bool IsNumbersOrRangesConvert(const NJson::TJsonValue& ignoredFeaturesJson) {
    return AllOf(ignoredFeaturesJson.GetArray(), [](const NJson::TJsonValue& ignoredFeature) {
        return IsNumberOrRange(ignoredFeature.GetString());
    });
}

static bool IsNumbersConvert(const NJson::TJsonValue& ignoredFeaturesJson) {
    return AllOf(ignoredFeaturesJson.GetArray(), [](const NJson::TJsonValue& ignoredFeature) {
        return IsNumber(ignoredFeature.GetString());
    });
}

static bool IsArrayOfIntegers(const NJson::TJsonValue& ignoredFeaturesJson) {
    return AllOf(ignoredFeaturesJson.GetArray(), [](const NJson::TJsonValue& ignoredFeature) {
        return ignoredFeature.IsInteger();
    });
}

TMap<TString, ui32> MakeIndicesFromNames(const NCatboostOptions::TPoolLoadParams& poolLoadParams) {
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
    return indicesFromNames;
}

TMap<TString, ui32> MakeIndicesFromNames(const NCB::TDataMetaInfo& metaInfo) {
    TMap<TString, ui32> indicesFromNames;
    ui32 columnIdx = 0;
    for (const auto& columnInfo : metaInfo.FeaturesLayout->GetExternalFeaturesMetaInfo()) {
        if (!columnInfo.Name.empty()) {
            indicesFromNames[columnInfo.Name] = columnIdx;
        }
        columnIdx++;
    }
    return indicesFromNames;
}

static void ConvertStringsArrayIntoIndicesArray(const NCatboostOptions::TPoolLoadParams& poolLoadParams, NJson::TJsonValue* ignoredFeaturesJson) {
    if (IsNumbersOrRangesConvert(*ignoredFeaturesJson)) {
        ConvertStringIndicesIntoIntegerIndices(ignoredFeaturesJson);
    } else {
        CB_ENSURE(!poolLoadParams.LearnSetPath.Scheme.Contains("quantized") || poolLoadParams.ColumnarPoolFormatParams.CdFilePath.Inited(),
                "quatized pool without CD file doesn't support ignoring features by names");
        const auto& indicesFromNames = MakeIndicesFromNames(poolLoadParams);
        ConvertNamesIntoIndices(indicesFromNames, ignoredFeaturesJson);
    }
}

static void ConvertStringsArrayIntoIndicesArray(const NCB::TDataMetaInfo& metaInfo, NJson::TJsonValue* ignoredFeaturesJson) {
    if (IsArrayOfIntegers(*ignoredFeaturesJson)) {
        return;
    }
    if (IsNumbersConvert(*ignoredFeaturesJson)) {
        ConvertStringIndicesIntoIntegerIndices(ignoredFeaturesJson);
    } else {
        TMap<TString, ui32> indicesFromNames;
        ui32 currentId = 0;
        auto featuresArray = metaInfo.FeaturesLayout.Get()->GetExternalFeaturesMetaInfo();
        for (const auto& feature : featuresArray) {
            if (!feature.Name.empty()) {
                indicesFromNames[feature.Name] = currentId;
            }
            currentId++;
        }
        ConvertNamesIntoIndices(indicesFromNames, ignoredFeaturesJson);
    }
}

void ConvertIgnoredFeaturesFromStringToIndices(const NCatboostOptions::TPoolLoadParams& poolLoadParams, NJson::TJsonValue* catBoostJsonOptions) {
    if (catBoostJsonOptions->Has("ignored_features")) {
        ConvertStringsArrayIntoIndicesArray(poolLoadParams, &(*catBoostJsonOptions)["ignored_features"]);
    }
}

void ConvertIgnoredFeaturesFromStringToIndices(const NCB::TDataMetaInfo& metaInfo, NJson::TJsonValue* catBoostJsonOptions) {
    if (catBoostJsonOptions->Has("ignored_features")) {
        ConvertStringsArrayIntoIndicesArray(metaInfo, &(*catBoostJsonOptions)["ignored_features"]);
    }
}

void ConvertPerFeatureOptionsFromStringToIndices(const TMap<TString, ui32>& indicesFromNames, NJson::TJsonValue* options) {
    if (indicesFromNames.empty()) {
        return;
    }
    auto& optionsRef = *options;
    Y_ASSERT(optionsRef.GetType() == NJson::EJsonValueType::JSON_MAP);
    bool needConvert = AnyOf(optionsRef.GetMap(), [](auto& element) {
        int index = 0;
        return !TryFromString(element.first, index);
    });
    if (needConvert) {
        NJson::TJsonValue optionsWithIndices(NJson::EJsonValueType::JSON_MAP);
        for (const auto& [featureName, value] : optionsRef.GetMap()) {
            auto it = indicesFromNames.find(featureName);
            CB_ENSURE(it != indicesFromNames.end(), "Unknown feature name: " << featureName);
            optionsWithIndices.InsertValue(ToString(it->second), value);
        }
        optionsRef.Swap(optionsWithIndices);
    }
}

static inline void ConvertPerFeatureOptionsFromStringToIndices(const NCB::TDataMetaInfo& metaInfo, NJson::TJsonValue* options) {
    ConvertPerFeatureOptionsFromStringToIndices(MakeIndicesFromNames(metaInfo), options);
}

static inline void ConvertPerFeatureOptionsFromStringToIndices(const NCatboostOptions::TPoolLoadParams& poolLoadParams, NJson::TJsonValue* options) {
    ConvertPerFeatureOptionsFromStringToIndices(MakeIndicesFromNames(poolLoadParams), options);
}

void ConvertMonotoneConstraintsFromStringToIndices(const NCB::TDataMetaInfo& metaInfo, NJson::TJsonValue* catBoostJsonOptions) {
    auto& treeOptions = (*catBoostJsonOptions)["tree_learner_options"];
    if (!treeOptions.Has("monotone_constraints")) {
        return;
    }

    ConvertPerFeatureOptionsFromStringToIndices(metaInfo, &treeOptions["monotone_constraints"]);
}

void ConvertMonotoneConstraintsFromStringToIndices(const NCatboostOptions::TPoolLoadParams& poolLoadParams, NJson::TJsonValue* catBoostJsonOptions) {
    auto& treeOptions = (*catBoostJsonOptions)["tree_learner_options"];
    if (!treeOptions.Has("monotone_constraints")) {
        return;
    }

    ConvertPerFeatureOptionsFromStringToIndices(poolLoadParams, &treeOptions["monotone_constraints"]);
}

ui32 ConvertToIndex(const TString& nameOrIndex, const TMap<TString, ui32>& indicesFromNames) {
    if (IsNumber(nameOrIndex)) {
        return FromString<ui32>(nameOrIndex);
    } else {
        CB_ENSURE(
            indicesFromNames.contains(nameOrIndex),
            "String " + nameOrIndex + " is not a feature name");
        return indicesFromNames.at(nameOrIndex);
    }
}

void ConvertFeaturesToEvaluateFromStringToIndices(const NCatboostOptions::TPoolLoadParams& poolLoadParams, NJson::TJsonValue* catBoostJsonOptions) {
    if (catBoostJsonOptions->Has("features_to_evaluate")) {
        const auto& indicesFromNames = MakeIndicesFromNames(poolLoadParams);
        const auto& featureNamesToEvaluate = (*catBoostJsonOptions)["features_to_evaluate"].GetString();
        TVector<TVector<int>> featuresToEvaluate;
        for (const auto& nameSet : StringSplitter(featureNamesToEvaluate).Split(';')) {
            const TString nameSetAsString(nameSet);
            featuresToEvaluate.emplace_back(TVector<int>{});
            for (const auto& nameOrRange : StringSplitter(nameSetAsString).Split(',')) {
                const TString nameOrRangeAsString(nameOrRange);
                auto left = nameOrRangeAsString;
                auto right = nameOrRangeAsString;
                StringSplitter(nameOrRangeAsString).Split('-').TryCollectInto(&left, &right);
                for (ui32 idx : xrange(ConvertToIndex(left, indicesFromNames), ConvertToIndex(right, indicesFromNames) + 1)) {
                    featuresToEvaluate.back().emplace_back(idx);
                }
            }
        }
        NCatboostOptions::TJsonFieldHelper<TVector<TVector<int>>>::Write(
            featuresToEvaluate,
            &(*catBoostJsonOptions)["features_to_evaluate"]);
    }
}
