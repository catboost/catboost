#pragma once

#include "params.h"
#include "learn_context.h"
#include "full_features.h"

#include <catboost/libs/data/pool.h>
#include <catboost/libs/model/model.h>

#include <util/generic/maybe.h>

#include <library/json/json_reader.h>

void TrainModel(
    const NJson::TJsonValue& params,
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    TPool& learnPool,
    bool allowClearPool,
    const TPool& testPool,
    const TString& outputModelPath,
    TFullModel* model,
    yvector<yvector<double>>* testApprox);

void TrainOneIteration(
    const TTrainData& trainData,
    TLearnContext* ctx);
