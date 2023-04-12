%{
#include <catboost/spark/catboost4j-spark/core/src/native_impl/options_helper.h>
#include <catboost/private/libs/options/catboost_options.h>
%}

%include "catboost_enums.i"


%catches(yexception) NCatboostOptions::TCatBoostOptions::TCatBoostOptions(ETaskType taskType);

namespace NCatboostOptions {
    class TCatBoostOptions {
    public:
        explicit TCatBoostOptions(ETaskType taskType);
    };
}


%catches(yexception) ParseCatBoostPlainParamsToJson(const TString& plainJsonParamsAsString);

%catches(yexception) LoadCatBoostOptions(const NJson::TJsonValue& plainJsonParams);

%catches(yexception) InitCatBoostOptions(
    const TString& plainJsonParamsAsString,
    NCatboostOptions::TCatBoostOptions* result
);

%catches(yexception) GetOneHotMaxSize(
    i32 maxCategoricalFeaturesUniqValuesOnLearn,
    bool hasLearnTarget,
    const TString& plainJsonParamsAsString
);

%include "options_helper.h"
