#pragma once

#include "features_layout.h"
#include "meta_info.h"

#include <catboost/private/libs/options/feature_penalties_options.h>
#include <catboost/private/libs/options/json_helper.h>
#include <catboost/private/libs/options/load_options.h>
#include <catboost/private/libs/options/monotone_constraints.h>

#include <catboost/libs/helpers/json_helpers.h>

#include <library/cpp/json/json_reader.h>

#include <util/generic/algorithm.h>
#include <util/string/split.h>
#include <util/string/type.h>


TMap<TString, ui32> MakeIndicesFromNames(const NCatboostOptions::TPoolLoadParams& poolLoadParams);
TMap<TString, ui32> MakeIndicesFromNames(const NCB::TFeaturesLayout& featuresLayout);
TMap<TString, ui32> MakeIndicesFromNames(const NCB::TDataMetaInfo& metaInfo);
THashMap<TString, TVector<ui32>> MakeIndicesFromTags(const NCatboostOptions::TPoolLoadParams& poolLoadParams);
THashMap<TString, TVector<ui32>> MakeIndicesFromTags(const NCB::TFeaturesLayout& featuresLayout);
THashMap<TString, TVector<ui32>> MakeIndicesFromTags(const NCB::TDataMetaInfo& metaInfo);

inline ui32 ConvertToIndex(const TString& nameOrIndex, const TMap<TString, ui32>& indicesFromNames) {
    if (IsNumber(nameOrIndex)) {
        return FromString<ui32>(nameOrIndex);
    } else {
        CB_ENSURE(
            indicesFromNames.contains(nameOrIndex),
            "String '" + nameOrIndex + "' is not a feature name");
        return indicesFromNames.at(nameOrIndex);
    }
}

namespace {
    class TIndicesMapper {
    public:
        TIndicesMapper(
            TMap<TString, ui32> indicesFromNames,
            THashMap<TString, TVector<ui32>> indicesFromTags
        )
            : IndicesFromNames(std::move(indicesFromNames))
            , IndicesFromTags(std::move(indicesFromTags))
        { }

        TIndicesMapper(const NCatboostOptions::TPoolLoadParams& poolLoadParams)
            : TIndicesMapper(MakeIndicesFromNames(poolLoadParams), MakeIndicesFromTags(poolLoadParams))
        { }

        TIndicesMapper(const NCB::TFeaturesLayout& featuresLayout)
            : TIndicesMapper(MakeIndicesFromNames(featuresLayout), MakeIndicesFromTags(featuresLayout))
        { }

        TIndicesMapper(const NCB::TDataMetaInfo& metaInfo)
            : TIndicesMapper(MakeIndicesFromNames(metaInfo), MakeIndicesFromTags(metaInfo))
        { }

        void Map(const TString& str, TVector<ui32>* indices) const {
            if (IsNumber(str)) {
                indices->push_back(FromString<ui32>(str));
            } else if (str.StartsWith("#")) {
                const TStringBuf tag(str.data() + 1, str.size() - 1);
                const auto tagIndices = IndicesFromTags.FindPtr(tag);
                CB_ENSURE(tagIndices, TStringBuf() << "There is no tag '#" << tag << "' in pool metainfo");
                indices->insert(indices->end(), tagIndices->begin(), tagIndices->end());
            } else {
                auto left = str;
                auto right = str;
                StringSplitter(str).Split('-').TryCollectInto(&left, &right);
                for (ui32 idx : xrange(ConvertToIndex(left, IndicesFromNames), ConvertToIndex(right, IndicesFromNames) + 1)) {
                    indices->push_back(idx);
                }
            }
        }

        TVector<ui32> Map(const TString& str) const {
            TVector<ui32> indices;
            Map(str, &indices);
            return indices;
        }

    private:
        TMap<TString, ui32> IndicesFromNames;
        THashMap<TString, TVector<ui32>> IndicesFromTags;
    };
}

void ConvertPerFeatureOptionsFromStringToIndices(const TMap<TString, ui32>& indicesFromNames, NJson::TJsonValue* options);

template <typename TSource>
void ConvertAllFeaturePenaltiesFromStringToIndices(const TSource& matchingSource, NJson::TJsonValue* catBoostJsonOptions) {
    auto& treeOptions = (*catBoostJsonOptions)["tree_learner_options"];
    if (!treeOptions.Has("penalties")) {
        return;
    }

    auto& penaltiesRef = treeOptions["penalties"];
    const auto namesToIndicesMap = MakeIndicesFromNames(matchingSource);

    if (penaltiesRef.Has("feature_weights")) {
        ConvertPerFeatureOptionsFromStringToIndices(namesToIndicesMap, &penaltiesRef["feature_weights"]);
    }
    if (penaltiesRef.Has("first_feature_use_penalties")) {
        ConvertPerFeatureOptionsFromStringToIndices(namesToIndicesMap, &penaltiesRef["first_feature_use_penalties"]);
    }
    if (penaltiesRef.Has("per_object_feature_penalties")) {
        ConvertPerFeatureOptionsFromStringToIndices(namesToIndicesMap, &penaltiesRef["per_object_feature_penalties"]);
    }
}

void ConvertFeaturesFromStringToIndices(
    const NCB::TPathWithScheme& cdFilePath,
    const NCB::TPathWithScheme& featureNamesPath,
    const NCB::TPathWithScheme& poolMetaInfoPath,
    NJson::TJsonValue* featuresArrayJson
);
void ConvertIgnoredFeaturesFromStringToIndices(const NCatboostOptions::TPoolLoadParams& poolLoadParams, NJson::TJsonValue* catBoostJsonOptions);
void ConvertIgnoredFeaturesFromStringToIndices(const NCB::TDataMetaInfo& metaInfo, NJson::TJsonValue* catBoostJsonOptions);
void ConvertMonotoneConstraintsFromStringToIndices(const NCB::TDataMetaInfo& metaInfo, NJson::TJsonValue* catBoostJsonOptions);
void ConvertMonotoneConstraintsFromStringToIndices(const NCatboostOptions::TPoolLoadParams& poolLoadParams, NJson::TJsonValue* catBoostJsonOptions);
void ConvertFeaturesToEvaluateFromStringToIndices(const NCatboostOptions::TPoolLoadParams& poolLoadParams, NJson::TJsonValue* catBoostJsonOptions);
void ConvertFixedBinarySplitsFromStringToIndices(const NCatboostOptions::TPoolLoadParams& poolLoadParams, NJson::TJsonValue* catBoostJsonOptions);

template <typename TSource>
void ConvertFeaturesForSelectFromStringToIndices(const TSource& stringsToIndicesMatchingSource, NJson::TJsonValue* featuresSelectJsonOptions) {
    const TIndicesMapper mapper(stringsToIndicesMatchingSource);
    const auto& featureNamesForSelect = (*featuresSelectJsonOptions)["features_for_select"].GetString();
    TVector<int> featuresForSelect;
    for (const auto& nameOrRange : StringSplitter(featureNamesForSelect).Split(',').SkipEmpty()) {
        const TString nameOrRangeAsString(nameOrRange);
        for (ui32 idx : mapper.Map(nameOrRangeAsString)) {
            featuresForSelect.push_back(idx);
        }
    }
    Sort(featuresForSelect);
    TJsonFieldHelper<TVector<int>>::Write(
        TVector<int>(featuresForSelect.begin(), Unique(featuresForSelect.begin(), featuresForSelect.end())),
        &(*featuresSelectJsonOptions)["features_for_select"]);
}

template <typename TSource>
void ConvertParamsToCanonicalFormat(const TSource& stringsToIndicesMatchingSource, NJson::TJsonValue* catBoostJsonOptions) {
    if (!catBoostJsonOptions->Has("tree_learner_options")) {
        return;
    }
    auto& treeOptions = (*catBoostJsonOptions)["tree_learner_options"];
    ConvertMonotoneConstraintsToCanonicalFormat(&treeOptions);
    ConvertMonotoneConstraintsFromStringToIndices(stringsToIndicesMatchingSource, catBoostJsonOptions);
    if (treeOptions.Has("penalties")) {
        NCatboostOptions::ConvertAllFeaturePenaltiesToCanonicalFormat(&treeOptions["penalties"]);
    }
    ConvertAllFeaturePenaltiesFromStringToIndices(stringsToIndicesMatchingSource, catBoostJsonOptions);
    if (catBoostJsonOptions->Has("flat_params")) {
        auto& flatParams = (*catBoostJsonOptions)["flat_params"];
        ConvertMonotoneConstraintsToCanonicalFormat(&flatParams);
        NCatboostOptions::ConvertAllFeaturePenaltiesToCanonicalFormat(&flatParams);
    }
}

// feature names - dependent params are returned in result and removed from catBoostJsonOptions
NJson::TJsonValue ExtractFeatureNamesDependentParams(NJson::TJsonValue* catBoostJsonOptions);

// feature names - dependent params are added to catBoostJsonOptions
void AddFeatureNamesDependentParams(const NJson::TJsonValue& featureNamesDependentOptions, NJson::TJsonValue* catBoostJsonOptions);
