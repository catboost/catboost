#pragma once

#include "full_features.h"
#include "target_classifier.h"
#include "bin_tracker.h"

#include <catboost/libs/model/online_ctr.h>


struct TFold;

const int SIMPLE_CLASSES_COUNT = 2;


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
