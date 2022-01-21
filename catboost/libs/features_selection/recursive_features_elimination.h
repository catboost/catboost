#pragma once

#include "selection_results.h"

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/eval_result/eval_result.h>
#include <catboost/libs/helpers/restorable_rng.h>
#include <catboost/libs/loggers/catboost_logger_helpers.h>
#include <catboost/private/libs/labels/label_converter.h>
#include <catboost/private/libs/options/catboost_options.h>
#include <catboost/private/libs/options/features_select_options.h>
#include <catboost/private/libs/options/output_file_options.h>

namespace NCB {
    TFeaturesSelectionSummary DoRecursiveFeaturesElimination(
        const NCatboostOptions::TCatBoostOptions& catBoostOptions,
        const NCatboostOptions::TOutputFilesOptions& outputFileOptions,
        const NCatboostOptions::TFeaturesSelectOptions& featuresSelectOptions,
        const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
        const TDataProviderPtr fstrPool,
        const TLabelConverter& labelConverter,
        TTrainingDataProviders trainingData,
        TFullModel* dstModel,
        const TVector<TEvalResult*>& evalResultPtrs,
        TMetricsAndTimeLeftHistory* metricsAndTimeHistory,
        NPar::ILocalExecutor* executor
    );
}
