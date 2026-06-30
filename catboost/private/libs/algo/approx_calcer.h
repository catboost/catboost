#pragma once

#include "fold.h"

#include <catboost/private/libs/algo_helpers/online_predictor.h>
#include <catboost/private/libs/options/enum_helpers.h>
#include <catboost/private/libs/options/restrictions.h>



class IDerCalcer;
class TLearnContext;
struct TSplitTree;
struct TNonSymmetricTreeStructure;

template <class T>
class TArray2D;

namespace NCatboostOptions {
    class TCatBoostOptions;
}

namespace NPar {
    class ITLocalExecutor;
}


void UpdateApproxDeltas(
    bool storeExpApprox,
    const TVector<TIndexType>& indices,
    int docCount,
    NPar::ILocalExecutor* localExecutor,
    TVector<double>* leafDeltas,
    TVector<double>* deltasDimension
);

void CalcLeafDersSimple(
    const TVector<TIndexType>& indices,
    const TFold& fold,
    const TFold::TBodyTail& bt,
    const TVector<double>& approxes,
    const TVector<double>& approxDeltas,
    const IDerCalcer& error,
    int sampleCount,
    int queryCount,
    bool recalcLeafWeights,
    ELeavesEstimation estimationMethod,
    const NCatboostOptions::TCatBoostOptions& params,
    ui64 randomSeed,
    NPar::ILocalExecutor* localExecutor,
    TVector<TSum>* leafDers,
    TArray2D<double>* pairwiseBuckets,
    TVector<TDers>* scratchDers
);

void CalcLeafDeltasSimple(
    const TVector<TSum>& leafDers,
    const TArray2D<double>& pairwiseWeightSums,
    const NCatboostOptions::TCatBoostOptions& params,
    double sumAllWeights,
    int allDocCount,
    TVector<double>* leafDeltas
);

void CalcLeafValues(
    const NCB::TTrainingDataProviders& data,
    const IDerCalcer& error,
    const std::variant<TSplitTree, TNonSymmetricTreeStructure>& tree,
    TLearnContext* ctx,
    TVector<TVector<double>>* leafDeltas,
    TVector<TIndexType>* indices
);

// output is permuted (learnSampleCount samples are permuted by LearnPermutation, test is indexed directly)
void CalcApproxForLeafStruct(
    const NCB::TTrainingDataProviders& data,
    const IDerCalcer& error,
    const TFold& fold,
    const std::variant<TSplitTree, TNonSymmetricTreeStructure>& tree,
    ui64 randomSeed,
    TLearnContext* ctx,
    TVector<TVector<TVector<double>>>* approxesDelta // [bodyTailId][approxDim][docIdxInPermuted]
);
