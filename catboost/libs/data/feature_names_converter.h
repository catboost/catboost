#pragma once

#include "meta_info.h"

#include <catboost/private/libs/options/load_options.h>

#include <library/json/json_reader.h>

TMap<TString, ui32> MakeIndicesFromNames(const NCatboostOptions::TPoolLoadParams& poolLoadParams);
TMap<TString, ui32> MakeIndicesFromNames(const NCB::TDataMetaInfo& metaInfo);

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
}

void ConvertIgnoredFeaturesFromStringToIndices(const NCatboostOptions::TPoolLoadParams& poolLoadParams, NJson::TJsonValue* catBoostJsonOptions);
void ConvertIgnoredFeaturesFromStringToIndices(const NCB::TDataMetaInfo& metaInfo, NJson::TJsonValue* catBoostJsonOptions);
void ConvertMonotoneConstraintsFromStringToIndices(const NCB::TDataMetaInfo& metaInfo, NJson::TJsonValue* catBoostJsonOptions);
void ConvertMonotoneConstraintsFromStringToIndices(const NCatboostOptions::TPoolLoadParams& poolLoadParams, NJson::TJsonValue* catBoostJsonOptions);
void ConvertFeaturesToEvaluateFromStringToIndices(const NCatboostOptions::TPoolLoadParams& poolLoadParams, NJson::TJsonValue* catBoostJsonOptions);
