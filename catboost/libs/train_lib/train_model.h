#pragma once

#include <catboost/libs/algo/learn_context.h>
#include <catboost/libs/algo/custom_objective_descriptor.h>
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

#include <functional>


using NCB::TEvalResult;

// returns whether training should be continued
using TOnEndIterationCallback = std::function<bool(const TMetricsAndTimeLeftHistory&)>;


// options used internally in CatBoost libraries, not to be specified by end users
struct TTrainModelInternalOptions {
    // Don't save the model, only calculate metrics during training.
    bool CalcMetricsOnly = false;

    // force it even if overfitting detector is disabled, used in Cross-Validation
    bool ForceCalcEvalMetricOnEveryIteration = false;
};


class IModelTrainer {
public:
    virtual void TrainModel(
        const TTrainModelInternalOptions& internalOptions,
        const NJson::TJsonValue& jsonParams,
        const NCatboostOptions::TOutputFilesOptions& outputOptions,
        const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
        const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
        const TMaybe<TOnEndIterationCallback>& onEndIterationCallback,
        NCB::TTrainingDataProviders trainingData,
        const TLabelConverter& labelConverter,
        NPar::TLocalExecutor* localExecutor,
        const TMaybe<TRestorableFastRng64*> rand,
        TFullModel* model,
        const TVector<TEvalResult*>& evalResultPtrs,
        TMetricsAndTimeLeftHistory* metricsAndTimeHistory
    ) const = 0;

    virtual void ModelBasedEval(
        const NJson::TJsonValue& params,
        const NCatboostOptions::TOutputFilesOptions& outputOptions,
        NCB::TTrainingDataProviders trainingData,
        const TLabelConverter& labelConverter,
        NPar::TLocalExecutor* localExecutor
    ) const = 0;

    virtual ~IModelTrainer() = default;
};

void TrainModel(
    const NCatboostOptions::TPoolLoadParams& poolLoadParams,
    const NCatboostOptions::TOutputFilesOptions& outputOptions,
    const NJson::TJsonValue& jsonParams);

void ModelBasedEval(
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
