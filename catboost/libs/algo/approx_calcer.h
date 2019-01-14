#pragma once

#include "fold.h"
#include "online_predictor.h"
#include "learn_context.h"
#include "error_functions.h"

#include <catboost/libs/options/catboost_options.h>
#include <catboost/libs/options/enum_helpers.h>


void UpdateApproxDeltas(
    bool storeExpApprox,
    const TVector<TIndexType>& indices,
    int docCount,
    NPar::TLocalExecutor* localExecutor,
    TVector<double>* leafValues,
    TVector<double>* resArr
);

static constexpr int APPROX_BLOCK_SIZE = 500;

void UpdateBucketsSimple(
    const TVector<TIndexType>& indices,
    const TFold& ff,
    const TFold::TBodyTail& bt,
    const TVector<double>& approxes,
    const TVector<double>& approxDeltas,
    const IDerCalcer& error,
    int sampleCount,
    int queryCount,
    int iteration,
    ELeavesEstimation estimationMethod,
    const NCatboostOptions::TCatBoostOptions& params,
    ui64 randomSeed,
    NPar::TLocalExecutor* localExecutor,
    TVector<TSum>* buckets,
    TArray2D<double>* pairwiseBuckets,
    TVector<TDers>* scratchDers
);

void CalcMixedModelSimple(
    const TVector<TSum>& buckets,
    const TArray2D<double>& pairwiseWeightSums,
    const NCatboostOptions::TCatBoostOptions& params,
    double sumAllWeights,
    int allDocCount,
    TVector<double>* leafValues
);

void CalcLeafValues(
    const NCB::TTrainingForCPUDataProviders& data,
    const IDerCalcer& error,
    const TFold& fold,
    const TSplitTree& tree,
    TLearnContext* ctx,
    TVector<TVector<double>>* leafValues,
    TVector<TIndexType>* indices
);

// output is permuted (learnSampleCount samples are permuted by LearnPermutation, test is indexed directly)
void CalcApproxForLeafStruct(
    const NCB::TTrainingForCPUDataProviders& data,
    const IDerCalcer& error,
    const TFold& fold,
    const TSplitTree& tree,
    ui64 randomSeed,
    TLearnContext* ctx,
    TVector<TVector<TVector<double>>>* approxesDelta // [bodyTailId][approxDim][docIdxInPermuted]
);
