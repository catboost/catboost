#pragma once

#include <catboost/libs/data/meta_info.h>
#include <catboost/private/libs/options/catboost_options.h>

void UpdateYetiRankEvalMetric(
    const TMaybe<NCB::TTargetStats>& learnTargetStats,
    const TMaybe<NCB::TTargetStats>& testTargetStats,
    NCatboostOptions::TCatBoostOptions* catBoostOptions
);

void SetDataDependentDefaults(
    const NCB::TDataMetaInfo& trainDataMetaInfo,
    const TMaybe<NCB::TDataMetaInfo>& testDataMetaInfo,
    bool continueFromModel,
    bool continueFromProgress,
    NCatboostOptions::TOption<bool>* useBestModel,
    NCatboostOptions::TCatBoostOptions* catBoostOptions
);
