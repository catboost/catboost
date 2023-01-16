#include "options_helper.h"

#include <catboost/libs/data/feature_names_converter.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/train_lib/options_helper.h>

#include <catboost/private/libs/options/catboost_options.h>
#include <catboost/private/libs/options/plain_options_helper.h>

#include <library/cpp/json/json_reader.h>

#include <util/generic/cast.h>


NJson::TJsonValue ParseCatBoostPlainParamsToJson(const TString& plainJsonParamsAsString) {
    NJson::TJsonValue plainJsonParams;
    try {
        NJson::ReadJsonTree(plainJsonParamsAsString, &plainJsonParams, /*throwOnError*/ true);
    } catch (const std::exception& e) {
        throw TCatBoostException() << "Error while parsing params JSON: " << e.what();
    }
    return plainJsonParams;
}

NCatboostOptions::TCatBoostOptions LoadCatBoostOptions(const NJson::TJsonValue& plainJsonParams) {
    NJson::TJsonValue jsonParams;
    NJson::TJsonValue outputJsonParams;
    NCatboostOptions::PlainJsonToOptions(plainJsonParams, &jsonParams, &outputJsonParams);

    /* we don't need feature names dependent params but LoadOptions can fail is they are present,
     * so delete them
     */
    ExtractFeatureNamesDependentParams(&jsonParams);

    return NCatboostOptions::LoadOptions(jsonParams);
}


void InitCatBoostOptions(
    const TString& plainJsonParamsAsString,
    NCatboostOptions::TCatBoostOptions* result
) {
    *result = LoadCatBoostOptions(ParseCatBoostPlainParamsToJson(plainJsonParamsAsString));
}


i32 GetOneHotMaxSize(
    i32 maxCategoricalFeaturesUniqValuesOnLearn,
    bool hasLearnTarget,
    const TString& plainJsonParamsAsString
) {
    NCatboostOptions::TCatBoostOptions catBoostOptions
        = LoadCatBoostOptions(ParseCatBoostPlainParamsToJson(plainJsonParamsAsString));

    UpdateOneHotMaxSize(
        SafeIntegerCast<ui32>(maxCategoricalFeaturesUniqValuesOnLearn),
        hasLearnTarget,
        &catBoostOptions
    );

    return catBoostOptions.CatFeatureParams->OneHotMaxSize.Get();
}
