#pragma once

#include "full_features.h"
#include "projection.h"
#include "target_classifier.h"
#include "train_data.h"


struct TFold;

const int SIMPLE_CLASSES_COUNT = 2;


struct TOnlineCTR {
    TVector<TArray2D<TVector<ui8>>> Feature; // Feature[ctrIdx][classIdx][priorIdx][docIdx]
    size_t FeatureValueCount = 0;
};

using TOnlineCTRHash = THashMap<TProjection, TOnlineCTR>;

inline ui8 CalcCTR(float countInClass, int totalCount, float prior, float shift, float norm, int borderCount) {
    float ctr = (countInClass + prior) / (totalCount + 1);
    return (ctr + shift) / norm * borderCount;
}

void CalcNormalization(const TVector<float>& priors, TVector<float>* shift, TVector<float>* norm);

class TLearnContext;
class TTrainData;


void ComputeOnlineCTRs(const TTrainData& data,
                       const TProjection& proj,
                       TLearnContext* ctx,
                       TFold* fold);

void ComputeOnlineCTRs(const TTrainData& data,
                       const TFold& fold,
                       const TProjection& proj,
                       TLearnContext* ctx,
                       TOnlineCTR* dst);

struct TCalcOnlineCTRsBatchTask {
    TProjection Projection;
    TFold* Fold;
    TOnlineCTR* Ctr;
};

void CalcOnlineCTRsBatch(const TVector<TCalcOnlineCTRsBatchTask>& tasks, const TTrainData& data, TLearnContext* ctx);

class TCtrValueTable;

void CalcFinalCtrs(
    const ECtrType ctrType,
    const TProjection& projection,
    const TTrainData& data,
    const TVector<size_t>& learnPermutation,
    const TVector<int>& permutedTargetClass,
    int targetClassesCount,
    ui64 ctrLeafCountLimit,
    bool storeAllSimpleCtr,
    ECounterCalc counterCalcMethod,
    TCtrValueTable* result);

void CalcFinalCtrs(
    const ECtrType ctrType,
    const TFeatureCombination& projection,
    const TPool& pool,
    ui64 sampleCount,
    const TVector<int>& permutedTargetClass,
    const TVector<float>& permutedTargets,
    int targetClassesCount,
    ui64 ctrLeafCountLimit,
    bool storeAllSimpleCtr,
    TCtrValueTable* result);
