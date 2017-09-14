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


void ShrinkModel(int itCount, TCoreModel* model) {
    model->LeafValues.resize(itCount);
    model->TreeStruct.resize(itCount);
}

yvector<int> CountSplits(
    const yvector<yvector<float>>& borders) {
    yvector<int> result;
    for (int i = 0; i < borders.ysize(); ++i) {
        result.push_back(borders[i].ysize());
    }
    return result;
}

TErrorTracker BuildErrorTracker(bool isMaxOptimal, bool hasTest, TLearnContext* ctx) {
    return TErrorTracker(ctx->Params.OdParams.OverfittingDetectorType,
                         isMaxOptimal,
                         ctx->Params.OdParams.AutoStopPval,
                         ctx->Params.OdParams.OverfittingDetectorIterationsWait,
                         true,
                         hasTest);
}
