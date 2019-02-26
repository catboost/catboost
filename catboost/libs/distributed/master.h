#pragma once

#include "mappers.h"

#include <catboost/libs/algo/approx_calcer_multi.h>
#include <catboost/libs/algo/learn_context.h>
#include <catboost/libs/algo/split.h>
#include <catboost/libs/algo/tensor_search_helpers.h>
#include <catboost/libs/data_new/data_provider.h>

void InitializeMaster(TLearnContext* ctx);
void FinalizeMaster(TLearnContext* ctx);
void MapBuildPlainFold(NCB::TTrainingForCPUDataProviderPtr trainData, TLearnContext* ctx);
void MapRestoreApproxFromTreeStruct(TLearnContext* ctx);
void MapTensorSearchStart(TLearnContext* ctx);
void MapBootstrap(TLearnContext* ctx);
void MapCalcScore(
    double scoreStDev,
    int depth,
    TConstArrayRef<NCB::TBinaryFeaturesPack> perPackMasks,
    TCandidateList* candidateList,
    TLearnContext* ctx);
void MapRemoteCalcScore(
    double scoreStDev,
    TConstArrayRef<NCB::TBinaryFeaturesPack> perPackMasks,
    TCandidateList* candidateList,
    TLearnContext* ctx);
void MapRemotePairwiseCalcScore(
    double scoreStDev,
    TConstArrayRef<NCB::TBinaryFeaturesPack> perPackMasks,
    TCandidateList* candidateList,
    TLearnContext* ctx);
void MapSetIndices(const TSplit& bestSplit, TLearnContext* ctx);
int MapGetRedundantSplitIdx(TLearnContext* ctx);
void MapCalcErrors(TLearnContext* ctx);

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
    const TSplitTree& splitTree,
    TConstArrayRef<NCB::TTrainingForCPUDataProviderPtr> testData,
    TVector<TVector<double>>* averageLeafValues,
    TVector<double>* sumLeafWeights,
    TLearnContext* ctx);

void MapSetApproxesMulti(
    const IDerCalcer& error,
    const TSplitTree& splitTree,
    TConstArrayRef<NCB::TTrainingForCPUDataProviderPtr> testData,
    TVector<TVector<double>>* averageLeafValues,
    TVector<double>* sumLeafWeights,
    TLearnContext* ctx);

void MapSetDerivatives(TLearnContext* ctx);
