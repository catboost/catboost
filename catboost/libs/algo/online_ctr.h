#pragma once

#include "full_features.h"
#include "train_data.h"
#include "target_classifier.h"
#include "bin_tracker.h"

#include <catboost/libs/model/online_ctr.h>
#include <catboost/libs/model/model.h>


struct TFold;

const int SIMPLE_CLASSES_COUNT = 2;


struct TOnlineCTR {
    yvector<TArray2D<yvector<ui8>>> Feature; // Feature[ctrIdx][classIdx][priorIdx][docIdx]
};

using TOnlineCTRHash = yhash<TProjection, TOnlineCTR>;

inline ui8 CalcCTR(float countInClass, int totalCount, float prior, float shift, float norm, int borderCount) {
    float ctr = (countInClass + prior) / (totalCount + 1);
    return (ctr + shift) / norm * borderCount;
}

void CalcNormalization(const yvector<float>& priors, yvector<float>* shift, yvector<float>* norm);
int GetCtrBorderCount(int targetClassesCount, ECtrType ctrType);

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

void CalcOnlineCTRsBatch(const yvector<TCalcOnlineCTRsBatchTask>& tasks, const TTrainData& data, TLearnContext* ctx);

void CalcFinalCtrs(const TModelCtr& ctr,
                   const TAllFeatures& features,
                   const ui64 learnSampleCount,
                   const yvector<int>& learnPermutation,
                   const yvector<int>& permutedTargetClass,
                   int targetClassesCount,
                   ui64 ctrLeafCountLimit,
                   bool storeAllSimpleCtr,
                   TCtrValueTable* result);

inline void CalcFinalCtrs(const TModelCtr& ctr,
                          const TTrainData& data,
                          const yvector<int>& learnPermutation,
                          const yvector<int>& permutedTargetClass,
                          int targetClassesCount,
                          ui64 ctrLeafCountLimit,
                          bool storeAllSimpleCtr,
                          TCtrValueTable* result) {
    CalcFinalCtrs(ctr, data.AllFeatures, static_cast<ui64>(data.LearnSampleCount), learnPermutation, permutedTargetClass, targetClassesCount, ctrLeafCountLimit, storeAllSimpleCtr, result);
}
