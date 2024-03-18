#include "feature_names_converter.h"

#include "loader.h"

#include <catboost/libs/column_description/cd_parser.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/private/libs/options/json_helper.h>
#include <catboost/private/libs/options/pool_metainfo_options.h>

#include <util/generic/cast.h>
#include <util/generic/xrange.h>
#include <util/string/split.h>
#include <util/string/type.h>

static bool IsArrayOfIntegers(const NJson::TJsonValue& jsonArray) {
    return AllOf(jsonArray.GetArray(), [](const NJson::TJsonValue& item) {
        return item.IsInteger();
    });
}

static void ConvertNamesIntoIndices(const TIndicesMapper& mapper,
                                    NJson::TJsonValue* featuresArrayJson) {
    if (IsArrayOfIntegers(*featuresArrayJson)) {
        return;
    }
    NJson::TJsonValue featuresIndicesArrayJson(NJson::JSON_ARRAY);
    for (NJson::TJsonValue featureOrRange : featuresArrayJson->GetArray()) {
        for (ui32 feature : mapper.Map(featureOrRange.GetString())) {
            featuresIndicesArrayJson.AppendValue(feature);
        }
    }
    featuresArrayJson->Swap(featuresIndicesArrayJson);
}


static void AddIndicesFromNames(
    TConstArrayRef<TColumn> columns,
    TMap<TString, ui32>* indicesFromNames,
    ui32* featureIdx
) {
    for (const auto& column : columns) {
        if (IsFactorColumn(column.Type)) {
            if (!column.Id.empty()) {
                (*indicesFromNames)[column.Id] = *featureIdx;
            }
            ++(*featureIdx);
        } else if (column.Type == EColumn::Features) {
            AddIndicesFromNames(column.SubColumns, indicesFromNames, featureIdx);
        }
    }
}

TMap<TString, ui32> MakeIndicesFromNamesByCdFile(const NCB::TPathWithScheme& cdFilePath) {
    TMap<TString, ui32> indicesFromNames;
    if (cdFilePath.Inited()) {
        const TVector<TColumn> columns = ReadCD(cdFilePath, TCdParserDefaults(EColumn::Num));
        ui32 featureIdx = 0;
        AddIndicesFromNames(columns, &indicesFromNames, &featureIdx);
    }
    return indicesFromNames;
}

TMap<TString, ui32> MakeIndicesFromNamesFromFeatureNamesFile(const NCB::TPathWithScheme& featureNamesPath) {
    TMap<TString, ui32> indicesFromNames;
    if (featureNamesPath.Inited()) {
        const TVector<TString> featureNames = LoadFeatureNames(featureNamesPath);
        for (auto featureIdx : xrange(SafeIntegerCast<ui32>(featureNames.size()))) {
            indicesFromNames.emplace(featureNames[featureIdx], featureIdx);
        }
    }
    return indicesFromNames;
}

void AddIndicesMap(
    TStringBuf addName,
    const TMap<TString, ui32>& add,
    TStringBuf resultName,
    TMap<TString, ui32>* /*inout*/ result
) {
    for (const auto& [name, idx] : add) {
        auto resIt = result->find(name);
        if (resIt == result->end()) {
            result->emplace(name, idx);
        } else {
            CB_ENSURE(
                resIt->second == idx,
                "feature \"" << name << "\": indices conflict: index " << idx << " in " << addName
                << " and index " << resIt->second << " in " << resultName
            );
        }
    }
}


TMap<TString, ui32> MakeIndicesFromNames(const NCatboostOptions::TPoolLoadParams& poolLoadParams) {
    // TODO: support header in tsv file
    auto result = MakeIndicesFromNamesFromFeatureNamesFile(poolLoadParams.FeatureNamesPath);
    auto mapFromCdFile = MakeIndicesFromNamesByCdFile(poolLoadParams.ColumnarPoolFormatParams.CdFilePath);
    AddIndicesMap("Column Description file", mapFromCdFile, "Feature Names file", &result);
    return result;
}

TMap<TString, ui32> MakeIndicesFromNames(const NCB::TFeaturesLayout& featuresLayout) {
    TMap<TString, ui32> indicesFromNames;
    ui32 featureIdx = 0;
    for (const auto& featureInfo : featuresLayout.GetExternalFeaturesMetaInfo()) {
        if (!featureInfo.Name.empty()) {
            indicesFromNames[featureInfo.Name] = featureIdx;
        }
        featureIdx++;
    }
    return indicesFromNames;
}

TMap<TString, ui32> MakeIndicesFromNames(const NCB::TDataMetaInfo& metaInfo) {
    return MakeIndicesFromNames(*metaInfo.FeaturesLayout);
}

static THashMap<TString, TVector<ui32>> MakeIndicesFromTagsFromPoolMetaInfo(const NCB::TPathWithScheme& poolMetaInfoPath) {
    const auto featureTags = NCatboostOptions::LoadPoolMetaInfoOptions(poolMetaInfoPath).Tags.Get();
    THashMap<TString, TVector<ui32>> indicesFromTags;
    for (const auto& [tag, description] : featureTags) {
        indicesFromTags[tag] = description.Features;
    }
    return indicesFromTags;
}

THashMap<TString, TVector<ui32>> MakeIndicesFromTags(const NCatboostOptions::TPoolLoadParams& poolLoadParams) {
    return MakeIndicesFromTagsFromPoolMetaInfo(poolLoadParams.PoolMetaInfoPath);
}

THashMap<TString, TVector<ui32>> MakeIndicesFromTags(const NCB::TFeaturesLayout& featuresLayout) {
    return featuresLayout.GetTagToExternalIndices();
}

THashMap<TString, TVector<ui32>> MakeIndicesFromTags(const NCB::TDataMetaInfo& metaInfo) {
    return MakeIndicesFromTags(*metaInfo.FeaturesLayout);
}

void ConvertFeaturesFromStringToIndices(
    const NCB::TPathWithScheme& cdFilePath,
    const NCB::TPathWithScheme& featureNamesPath,
    const NCB::TPathWithScheme& poolMetaInfoPath,
    NJson::TJsonValue* featuresArrayJson
) {
    // TODO: support header in tsv file
    auto indicesFromNames = MakeIndicesFromNamesFromFeatureNamesFile(featureNamesPath);
    auto indicesFromCdFile = MakeIndicesFromNamesByCdFile(cdFilePath);
    AddIndicesMap("Column Description file", indicesFromCdFile, "Feature Names file", &indicesFromNames);
    auto indicesFromTags = MakeIndicesFromTagsFromPoolMetaInfo(poolMetaInfoPath);
    ConvertNamesIntoIndices(TIndicesMapper(std::move(indicesFromNames), std::move(indicesFromTags)), featuresArrayJson);
}

void ConvertIgnoredFeaturesFromStringToIndices(const NCatboostOptions::TPoolLoadParams& poolLoadParams, NJson::TJsonValue* catBoostJsonOptions) {
    if (catBoostJsonOptions->Has("ignored_features")) {
        ConvertNamesIntoIndices(TIndicesMapper(poolLoadParams), &(*catBoostJsonOptions)["ignored_features"]);
    }
}

void ConvertIgnoredFeaturesFromStringToIndices(const NCB::TDataMetaInfo& metaInfo, NJson::TJsonValue* catBoostJsonOptions) {
    if (catBoostJsonOptions->Has("ignored_features")) {
        ConvertNamesIntoIndices(TIndicesMapper(metaInfo), &(*catBoostJsonOptions)["ignored_features"]);
    }
}

void ConvertFixedBinarySplitsFromStringToIndices(const NCatboostOptions::TPoolLoadParams& poolLoadParams, NJson::TJsonValue* catBoostJsonOptions) {
    if (catBoostJsonOptions->Has("fixed_binary_splits")) {
        ConvertNamesIntoIndices(TIndicesMapper(poolLoadParams), &(*catBoostJsonOptions)["fixed_binary_splits"]);
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

void ConvertFeaturesToEvaluateFromStringToIndices(const NCatboostOptions::TPoolLoadParams& poolLoadParams, NJson::TJsonValue* catBoostJsonOptions) {
    if (catBoostJsonOptions->Has("features_to_evaluate")) {
        TIndicesMapper mapper(poolLoadParams);
        const auto& featureNamesToEvaluate = (*catBoostJsonOptions)["features_to_evaluate"].GetString();
        TVector<TVector<int>> featuresToEvaluate;
        for (const auto& nameSet : StringSplitter(featureNamesToEvaluate).Split(';')) {
            const TString nameSetAsString(nameSet);
            featuresToEvaluate.emplace_back(TVector<int>{});
            for (const auto& nameOrRange : StringSplitter(nameSetAsString).Split(',')) {
                const TString nameOrRangeAsString(nameOrRange);
                for (ui32 idx : mapper.Map(nameOrRangeAsString)) {
                    featuresToEvaluate.back().emplace_back(idx);
                }
            }
        }
        TJsonFieldHelper<TVector<TVector<int>>>::Write(
            featuresToEvaluate,
            &(*catBoostJsonOptions)["features_to_evaluate"]);
    }
}


static const TStringBuf FEATURE_NAMES_DEPENDENT_KEYS[] = {
    TStringBuf("features_to_evaluate"),
    TStringBuf("ignored_features"),
    TStringBuf("monotone_constraints"),
    TStringBuf("penalties")
};


NJson::TJsonValue ExtractFeatureNamesDependentParams(NJson::TJsonValue* catBoostJsonOptions) {
    NJson::TJsonValue result(NJson::EJsonValueType::JSON_MAP);
    auto& treeOptionsMap = (*catBoostJsonOptions)["tree_learner_options"].GetMapSafe();

    auto& resultTreeOptions = result["tree_learner_options"];
    resultTreeOptions.SetType(NJson::EJsonValueType::JSON_MAP);

    for (const auto& key : FEATURE_NAMES_DEPENDENT_KEYS) {
        auto it = treeOptionsMap.find(key);
        if (it != treeOptionsMap.end()) {
            resultTreeOptions.InsertValue(key, it->second);
            treeOptionsMap.erase(it);
        }
    }

    return result;
}

// feature names - dependent params are added to catBoostJsonOptions
void AddFeatureNamesDependentParams(const NJson::TJsonValue& featureNamesDependentOptions, NJson::TJsonValue* catBoostJsonOptions) {
    const auto& treeOptionsMap = featureNamesDependentOptions["tree_learner_options"].GetMapSafe();
    auto& resultTreeOptionsMap = (*catBoostJsonOptions)["tree_learner_options"].GetMapSafe();

    for (const auto& key : FEATURE_NAMES_DEPENDENT_KEYS) {
        auto it = treeOptionsMap.find(key);
        if (it != treeOptionsMap.end()) {
            resultTreeOptionsMap.insert(*it);
        }
    }
}
