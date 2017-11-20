#pragma once

#include "learn_context.h"
#include "full_features.h"

#include <catboost/libs/params/params.h>
#include <catboost/libs/data/pool.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/helpers/eval_helpers.h>

#include <util/generic/maybe.h>

#include <library/json/json_reader.h>
#include <library/object_factory/object_factory.h>

class IModelTrainer {
public:
    virtual void TrainModel(
        const NJson::TJsonValue& params,
        const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
        const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
        TPool& learnPool,
        bool allowClearPool,
        const TPool& testPool,
        const TString& outputModelPath,
        TFullModel* model,
        TEvalResult* evalResult) const = 0;

    virtual ~IModelTrainer() = default;
};

void TrainModel(
    const NJson::TJsonValue& params,
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    TPool& learnPool,
    bool allowClearPool,
    const TPool& testPool,
    const TString& outputModelPath,
    TFullModel* model,
    TEvalResult* testResult);

void TrainOneIteration(
    const TTrainData& trainData,
    TLearnContext* ctx);

using TTrainerFactory = NObjectFactory::TParametrizedObjectFactory<IModelTrainer, ETaskType>;
