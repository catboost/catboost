#pragma once

#include <catboost/libs/algo/learn_context.h>
#include <catboost/libs/data/pool.h>
#include <catboost/libs/eval_result/eval_result.h>
#include <catboost/libs/loggers/catboost_logger_helpers.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/options/load_options.h>
#include <catboost/libs/options/output_file_options.h>

#include <library/json/json_value.h>
#include <library/object_factory/object_factory.h>

#include <util/generic/maybe.h>
#include <util/generic/string.h>

using NCB::TEvalResult;

class IModelTrainer {
public:
    virtual void TrainModel(
        const NJson::TJsonValue& jsonParams,
        const NCatboostOptions::TOutputFilesOptions& outputOptions,
        const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
        const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
        const TClearablePoolPtrs& pools,
        TFullModel* model,
        const TVector<TEvalResult*>& evalResultPtrs,
        TMetricsAndTimeLeftHistory* metricsAndTimeHistory) const = 0;

    virtual void TrainModel(const NCatboostOptions::TPoolLoadParams& poolLoadParams,
                            const NCatboostOptions::TOutputFilesOptions& outputOptions,
                            const NJson::TJsonValue& jsonParams) const = 0;

    virtual ~IModelTrainer() = default;
};

void TrainModel(
    const NJson::TJsonValue& plainJsonParams,
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    const TClearablePoolPtrs& pools,
    const TString& outputModelPath,
    TFullModel* model,
    const TVector<TEvalResult*>& evalResultPtrs,
    TMetricsAndTimeLeftHistory* metricsAndTimeHistory = nullptr);

/// Used by cross validation, hence one test dataset.
void TrainOneIteration(
    const TDataset& trainData,
    const TDataset* testData,
    TLearnContext* ctx);

using TTrainerFactory = NObjectFactory::TParametrizedObjectFactory<IModelTrainer, ETaskType>;
