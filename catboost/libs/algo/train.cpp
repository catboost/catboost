#include "train.h"

#include "error_functions.h"
#include "online_predictor.h"
#include "bin_tracker.h"
#include "rand_score.h"
#include "fold.h"
#include "online_ctr.h"
#include "score_calcer.h"
#include "approx_calcer.h"
#include "index_hash_calcer.h"
#include "greedy_tensor_search.h"
#include "metric.h"
#include <catboost/libs/model/hash.h>
#include <catboost/libs/model/projection.h>
#include <catboost/libs/model/tensor_struct.h>

void CalcFinalCtrs(const TCtr& ctr,
                   const TTrainData& data,
                   const yvector<int>& learnPermutation,
                   const yvector<int>& permutedTargetClass,
                   int targetClassesCount,
                   TLearnContext* ctx,
                   TCtrValueTable* result) {
    const ECtrType ctrType = ctx->Params.CtrParams.Ctrs[ctr.CtrTypeIdx].CtrType;
    yvector<ui64> hashArr;
    CalcHashes(ctr.Projection, data.AllFeatures, data.LearnSampleCount, learnPermutation, &hashArr);

    ui64 topSize = ctx->Params.CtrLeafCountLimit;
    if (ctr.Projection.IsSingleCatFeature() && ctx->Params.StoreAllSimpleCtr) {
        topSize = Max<ui64>();
    }

    auto leafCount = ReindexHash(
        data.LearnSampleCount,
        topSize,
        &hashArr,
        &result->Hash).first;

    if (ctrType == ECtrType::MeanValue) {
        result->CtrMean.resize(leafCount);
    } else if (IsCounter(ctrType)) {
        result->CtrTotal.resize(leafCount);
        result->CounterDenominator = 0;
    } else {
        result->Ctr.resize(leafCount, yvector<int>(targetClassesCount));
    }

    // TODO(annaveronika): remove code dup
    Y_ASSERT(hashArr.ysize() == data.LearnSampleCount);
    int targetBorderCount = targetClassesCount - 1;
    for (int z = 0; z < data.LearnSampleCount; ++z) {
        const ui64 elemId = hashArr[z];

        if (ctrType == ECtrType::MeanValue) {
            TCtrMeanHistory& elem = result->CtrMean[elemId];
            elem.Add(static_cast<float>(permutedTargetClass[z]) / targetBorderCount);
        } else if (IsCounter(ctrType)) {
            ++result->CtrTotal[elemId];
        } else {
            yvector<int>& elem = result->Ctr[elemId];
            ++elem[permutedTargetClass[z]];
        }
    }

    if (IsCounter(ctrType)) {
        if (ctrType == ECtrType::CounterMax) {
            result->CounterDenominator = *MaxElement(result->CtrTotal.begin(), result->CtrTotal.end());
        } else {
            Y_ASSERT(ctrType == ECtrType::CounterTotal);
            result->CounterDenominator = data.LearnSampleCount;
        }
    }
}

void ShrinkModel(int itCount, TCoreModel* model) {
    model->LeafValues.resize(itCount);
    model->TreeStruct.resize(itCount);
}

yvector<int> CountSplits(
    const yhash_set<int>& categFeatures,
    const yvector<yvector<float>>& borders) {
    yvector<int> result;
    for (int i = 0; i < borders.ysize(); ++i) {
        if (categFeatures.has(i)) {
            continue;
        }
        result.push_back(borders[i].ysize());
    }
    return result;
}

TErrorTracker BuildErrorTracker(bool isMaxOptimal, bool hasTest, TLearnContext* ctx) {
    return TErrorTracker(ctx->Params.OverfittingDetectorType,
                         isMaxOptimal,
                         ctx->Params.AutoStopPval,
                         ctx->Params.OverfittingDetectorIterationsWait,
                         true,
                         hasTest);
}
