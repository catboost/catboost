#pragma once

#include <catboost/libs/algo/learn_context.h>
#include <catboost/libs/algo/full_features.h>

#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/data/pool.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/helpers/eval_helpers.h>
#include <catboost/libs/options/load_options.h>

#include <util/generic/maybe.h>

#include <library/json/json_reader.h>
#include <library/object_factory/object_factory.h>

class IModelTrainer {
public:
    virtual void TrainModel(
        const NJson::TJsonValue& jsonParams,
        const NCatboostOptions::TOutputFilesOptions& outputOptions,
        const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
        const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
        TPool& learnPool,
        bool allowClearPool,
        const TPool& testPool,
        TFullModel* model,
        TEvalResult* evalResult) const = 0;

    virtual void TrainModel(const NCatboostOptions::TPoolLoadParams& poolLoadParams,
                            const NCatboostOptions::TOutputFilesOptions& outputOptions,
                            const NJson::TJsonValue& jsonParams) const = 0;

    virtual ~IModelTrainer() = default;
};

void TrainModel(
    const NJson::TJsonValue& plainJsonParams,
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    TPool& learnPool,
    bool allowClearPool,
    const TPool& testPool,
    const TString& outputModelPath,
    TFullModel* model,
    TEvalResult* testResult);

void TrainOneIteration(
    const TDataset& trainData,
    const TDataset* testData,
    TLearnContext* ctx);

using TTrainerFactory = NObjectFactory::TParametrizedObjectFactory<IModelTrainer, ETaskType>;
