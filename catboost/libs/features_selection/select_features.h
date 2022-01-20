#pragma once

#include "selection_results.h"

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/eval_result/eval_result.h>
#include <catboost/libs/loggers/catboost_logger_helpers.h>
#include <catboost/private/libs/options/catboost_options.h>
#include <catboost/private/libs/options/features_select_options.h>
#include <catboost/private/libs/options/load_options.h>
#include <catboost/private/libs/options/output_file_options.h>


namespace NCB {
    TFeaturesSelectionSummary SelectFeatures(
        NCatboostOptions::TCatBoostOptions catBoostOptions,
        NCatboostOptions::TOutputFilesOptions outputFileOptions,
        const NCatboostOptions::TPoolLoadParams* poolLoadParams, // can be nullptr
        const NCatboostOptions::TFeaturesSelectOptions& featuresSelectOptions,
        const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
        const TDataProviders& pools,
        TFullModel* dstModel, // can be nullptr
        const TVector<TEvalResult*>& evalResultPtrs, // can be nullptr
        TMetricsAndTimeLeftHistory* metricsAndTimeHistory, // can be nullptr
        NPar::ILocalExecutor* executor
    );

    NJson::TJsonValue SelectFeatures(
        const NJson::TJsonValue& plainJsonParams,
        const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
        const TDataProviders& pools,
        TFullModel* dstModel,
        const TVector<TEvalResult*>& testApproxes,
        TMetricsAndTimeLeftHistory* metricsAndTimeHistory
    );
}
