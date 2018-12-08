#pragma once

#include <catboost/libs/algo/learn_context.h>
#include <catboost/libs/data_new/data_provider.h>
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
        NCB::TTrainingDataProviders trainingData,
        const TLabelConverter& labelConverter,
        NPar::TLocalExecutor* localExecutor,
        TRestorableFastRng64* rand,
        TFullModel* model,
        const TVector<TEvalResult*>& evalResultPtrs,
        TMetricsAndTimeLeftHistory* metricsAndTimeHistory) const = 0;

    virtual ~IModelTrainer() = default;
};

void TrainModel(
    const NCatboostOptions::TPoolLoadParams& poolLoadParams,
    const NCatboostOptions::TOutputFilesOptions& outputOptions,
    const NJson::TJsonValue& jsonParams);


void TrainModel(
    const NJson::TJsonValue& plainJsonParams,
    NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo, // can be nullptr
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    NCB::TDataProviders pools, // not rvalue reference because Cython does not support them
    const TString& outputModelPath,
    TFullModel* model,
    const TVector<TEvalResult*>& evalResultPtrs,
    TMetricsAndTimeLeftHistory* metricsAndTimeHistory = nullptr);

/// Used by cross validation, hence one test dataset.
void TrainOneIteration(const NCB::TTrainingForCPUDataProviders& data, TLearnContext* ctx);

using TTrainerFactory = NObjectFactory::TParametrizedObjectFactory<IModelTrainer, ETaskType>;
