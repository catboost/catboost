#pragma once

#include <catboost/private/libs/options/catboost_options.h>

#include <util/generic/fwd.h>
#include <util/generic/yexception.h>
#include <util/system/types.h>


namespace NJson {
    class TJsonValue;
}


NJson::TJsonValue ParseCatBoostPlainParamsToJson(const TString& plainJsonParamsAsString);

NCatboostOptions::TCatBoostOptions LoadCatBoostOptions(const NJson::TJsonValue& plainJsonParams);

// 'result' is an output param because SWIG cannot return classes without default copy constructor
void InitCatBoostOptions(
    const TString& plainJsonParamsAsString,
    NCatboostOptions::TCatBoostOptions* result
);

i32 GetOneHotMaxSize(
    i32 maxCategoricalFeaturesUniqValuesOnLearn,
    bool hasLearnTarget,
    const TString& plainJsonParamsAsString
);
