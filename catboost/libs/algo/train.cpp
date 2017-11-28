#include "train.h"
#include "online_predictor.h"
#include "bin_tracker.h"
#include "rand_score.h"
#include "fold.h"
#include "online_ctr.h"
#include "score_calcer.h"
#include "approx_calcer.h"
#include "index_hash_calcer.h"
#include "greedy_tensor_search.h"

#include <catboost/libs/model/hash.h>


void ShrinkModel(int itCount, TLearnProgress* progress) {
    progress->LeafValues.resize(itCount);
    progress->TreeStruct.resize(itCount);
}

TVector<int> CountSplits(const TVector<TFloatFeature>& floatFeatures) {
    TVector<int> result;
    for (int i = 0; i < floatFeatures.ysize(); ++i) {
        result.push_back(floatFeatures[i].Borders.ysize());
    }
    return result;
}

TErrorTracker BuildErrorTracker(bool isMaxOptimal, bool hasTest, TLearnContext* ctx) {
    const auto& odOptions = ctx->Params.BoostingOptions->OverfittingDetector;
    return TErrorTracker(odOptions->OverfittingDetectorType,
                         isMaxOptimal,
                         odOptions->AutoStopPValue,
                         odOptions->IterationsWait,
                         true,
                         hasTest);
}

void NormalizeLeafValues(const TVector<TIndexType>& indices, int learnSampleCount, TVector<TVector<double>>* treeValues) {
    TVector<int> weights((*treeValues)[0].ysize());
    for (int docIdx = 0; docIdx < learnSampleCount; ++docIdx) {
        ++weights[indices[docIdx]];
    }

    double avrg = 0;
    for (int i = 0; i < weights.ysize(); ++i) {
        avrg += weights[i] * (*treeValues)[0][i];
    }

    int sumWeight = 0;
    for (int w : weights) {
        sumWeight += w;
    }
    avrg /= sumWeight;

    for (auto& value : (*treeValues)[0]) {
        value -= avrg;
    }
}

