#pragma once

#include "mappers.h"

#include <catboost/private/libs/algo/approx_delta_calcer_multi.h>
#include <catboost/private/libs/algo/learn_context.h>
#include <catboost/private/libs/algo/split.h>
#include <catboost/private/libs/algo/tensor_search_helpers.h>
#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/data/loader.h>
#include <catboost/private/libs/options/load_options.h>

#include <util/generic/maybe.h>


class TLabelConverter;


class TMasterContext {
public:
    TMasterContext(const NCatboostOptions::TSystemOptions& systemOptions);
    ~TMasterContext();
};

void SetTrainDataFromQuantizedPools(
    const NCatboostOptions::TPoolLoadParams& poolLoadOptions,
    const NCatboostOptions::TCatBoostOptions& catBoostOptions,
    NCB::TObjectsGrouping&& learnObjectsGrouping,
    TVector<NCB::TObjectsGrouping>&& testObjectsGroupings,
    const NCB::TFeaturesLayout& featuresLayout,
    const TLabelConverter& labelConverter,
    TRestorableFastRng64* rand
);
void SetTrainDataFromMaster(
    const NCB::TTrainingDataProviders& trainData,
    ui64 cpuUsedRamLimit,
    NPar::ILocalExecutor* localExecutor);
void MapBuildPlainFold(TLearnContext* ctx);
void MapRestoreApproxFromTreeStruct(TMaybe<int> bestIteration, TLearnContext* ctx);
void MapTensorSearchStart(TLearnContext* ctx);
void MapBootstrap(TLearnContext* ctx);
double MapCalcDerivativesStDevFromZero(ui32 learnSampleCount, TLearnContext* ctx);
void MapCalcScore(
    double scoreStDev,
    int depth,
    TCandidatesContext* candidatesContext,
    TLearnContext* ctx);
void MapRemoteCalcScore(
    double scoreStDev,
    TVector<TCandidatesContext>* candidatesContext,
    TLearnContext* ctx);
void MapRemotePairwiseCalcScore(
    double scoreStDev,
    TVector<TCandidatesContext>* candidatesContext,
    TLearnContext* ctx);
void MapSetIndices(const TSplit& bestSplit, TLearnContext* ctx);
int MapGetRedundantSplitIdx(TLearnContext* ctx);
void CalcErrorsDistributed(
    const NCB::TTrainingDataProviders& trainData,
    const TVector<THolder<IMetric>>& metrics,
    bool calcAllMetrics,
    bool calcErrorTrackerMetric,
    TLearnContext* ctx);

template <typename TMapper>
TVector<typename TMapper::TOutput> ApplyMapper(
    int workerCount,
    TObj<NPar::IEnvironment> environment,
    const typename TMapper::TInput& value = typename TMapper::TInput()) {

    NPar::TJobDescription job;
    TVector<typename TMapper::TInput> mapperInput(1);
    mapperInput[0] = value;
    NPar::Map(&job, new TMapper(), &mapperInput);
    job.SeparateResults(workerCount);
    NPar::TJobExecutor exec(&job, environment);
    TVector<typename TMapper::TOutput> mapperOutput;
    exec.GetResultVec(&mapperOutput);
    return mapperOutput;
}

void MapSetApproxesSimple(
    const IDerCalcer& error,
    const std::variant<TSplitTree, TNonSymmetricTreeStructure>& splitTree,
    TVector<TVector<double>>* averageLeafValues,
    TVector<double>* sumLeafWeights,
    TLearnContext* ctx);

void MapSetApproxesMulti(
    const IDerCalcer& error,
    const std::variant<TSplitTree, TNonSymmetricTreeStructure>& splitTree,
    TVector<TVector<double>>* averageLeafValues,
    TVector<double>* sumLeafWeights,
    TLearnContext* ctx);

void MapSetDerivatives(TLearnContext* ctx);

void MapSetBestTestApprox(TLearnContext* ctx);
void MapGetApprox(
    const NCB::TTrainingDataProviders& data,
    const TLearnContext& ctx,
    bool useBestModel,
    TVector<TVector<double>>* learnApprox,  // [dim][docIdx], can be nullptr
    TVector<TVector<TVector<double>>>* testApprox); // [test][dim][docIdx], can be nullptr
