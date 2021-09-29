#pragma once

#include <catboost/private/libs/algo/learn_context.h>
#include <catboost/private/libs/algo_helpers/custom_objective_descriptor.h>
#include <catboost/libs/data/ctrs.h>
#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/eval_result/eval_result.h>
#include <catboost/libs/loggers/catboost_logger_helpers.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/model/model.h>
#include <catboost/private/libs/options/load_options.h>
#include <catboost/private/libs/options/output_file_options.h>
#include <catboost/private/libs/options/catboost_options.h>

#include <library/cpp/json/json_value.h>
#include <library/cpp/object_factory/object_factory.h>

#include <util/generic/maybe.h>
#include <util/generic/ptr.h>
#include <util/generic/string.h>

#include <functional>


using NCB::TEvalResult;

struct TLearnProgress;


// returns whether training should be continued
using TOnEndIterationCallback = std::function<bool(const TMetricsAndTimeLeftHistory&)>;


// options used internally in CatBoost libraries, not to be specified by end users
struct TTrainModelInternalOptions {
    // Don't save the model, only calculate metrics during training.
    bool CalcMetricsOnly = false;

    // force it even if overfitting detector is disabled, used in Cross-Validation
    bool ForceCalcEvalMetricOnEveryIteration = false;

    /* Save final Ctr statistics in the model.
     * Disable to save resources if the resulting model is intended to be used to continue training only,
     * not for immediate application
     */
    bool SaveFinalCtrsToModel = true;

    /*
     * When choosing iterations to evaluate metrics on, consider initModel size (if it is specified)
     *   to ensure equal metrics steps in learning continuation.
     * Used in Cross-Validation
     */
    bool OffsetMetricPeriodByInitModelSize = false;

    /*
     * True means that all of training dataset is loaded into RAM on host where training is started
     * Maybe false only in multi-host training with quantized dataset
     */
    bool HaveLearnFeatureInMemory = true;
};

struct TCustomCallbackDescriptor {
    using TAfterIteration = bool (*)(const TMetricsAndTimeLeftHistory& history, void* customData);

    void* CustomData = nullptr;
    TAfterIteration AfterIterationFunc = nullptr;
};

class ITrainingCallbacks {
public:
    virtual bool IsContinueTraining(const TMetricsAndTimeLeftHistory& /*history*/) {
        return true;
    }
    virtual void OnSaveSnapshot(const NJson::TJsonValue& /*processors*/, IOutputStream* /*snapshot*/) {
    }
    virtual bool OnLoadSnapshot(IInputStream* /*snapshot*/) {
        return true;
    }

    virtual ~ITrainingCallbacks() = default;
};

class ICustomCallbacks {
public:
    virtual bool AfterIteration(const TMetricsAndTimeLeftHistory& /*history*/) = 0;
    virtual ~ICustomCallbacks() = default;
};

class TCustomCallbacks : public ICustomCallbacks {
public:
    explicit TCustomCallbacks(TMaybe<TCustomCallbackDescriptor> callbackDescriptor);
    bool AfterIteration(const TMetricsAndTimeLeftHistory& history) override;
private:
    const TMaybe<TCustomCallbackDescriptor> CallbackDescriptor;
};

class IModelTrainer {
public:
    virtual void TrainModel(
        const TTrainModelInternalOptions& internalOptions,
        const NCatboostOptions::TCatBoostOptions& catboostOptions,
        const NCatboostOptions::TOutputFilesOptions& outputOptions,
        const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
        const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
        NCB::TTrainingDataProviders trainingData,

        // can be non-empty only if there is single fold
        TMaybe<NCB::TPrecomputedOnlineCtrData> precomputedSingleOnlineCtrDataForSingleFold,
        const TLabelConverter& labelConverter,
        ITrainingCallbacks* trainingCallbacks,
        ICustomCallbacks* customCallbacks,
        TMaybe<TFullModel*> initModel,
        THolder<TLearnProgress> initLearnProgress, // can be nullptr, can be modified if non-nullptr

        /* It might be needed to apply initModel to trainingData to get initial approxes
         * but quantization for initModel could be different from quantization for training continuation
         * used in trainingData
         * that's why this additional parameter is necessary
         */
        NCB::TDataProviders initModelApplyCompatiblePools,
        NPar::ILocalExecutor* localExecutor,
        const TMaybe<TRestorableFastRng64*> rand,
        TFullModel* dstModel,

        // can be empty if eval results are not needed
        const TVector<TEvalResult*>& evalResultPtrs, // [testIdx]
        TMetricsAndTimeLeftHistory* metricsAndTimeHistory,
        THolder<TLearnProgress>* dstLearnProgress
    ) const = 0;

    virtual void ModelBasedEval(
        const NCatboostOptions::TCatBoostOptions& catboostOptions,
        const NCatboostOptions::TOutputFilesOptions& outputOptions,
        NCB::TTrainingDataProviders trainingData,
        const TLabelConverter& labelConverter,
        NPar::ILocalExecutor* localExecutor
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
    NJson::TJsonValue plainJsonParams,
    NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo, // can be nullptr
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
    const TMaybe<TCustomCallbackDescriptor>& callbackDescriptor,
    NCB::TDataProviders pools, // not rvalue reference because Cython does not support them
    TMaybe<TFullModel*> initModel,

    // can be nullptr, if not nullptr, THolder will be freed. Not r-value or value because of Cython
    THolder<TLearnProgress>* initLearnProgress,
    const TString& outputModelPath,
    TFullModel* dstModel,
    const TVector<TEvalResult*>& evalResultPtrs,
    TMetricsAndTimeLeftHistory* metricsAndTimeHistory = nullptr,
    THolder<TLearnProgress>* dstLearnProgress = nullptr);


NCatboostOptions::TCatBoostOptions GetCatBoostOptions(NCB::TDataProviderPtr pool, NJson::TJsonValue plainJsonParams, int* threadCount);


using TTrainerFactory = NObjectFactory::TParametrizedObjectFactory<IModelTrainer, ETaskType>;
