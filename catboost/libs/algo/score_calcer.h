#pragma once

#include "online_predictor.h"
#include "params.h"
#include "fold.h"
#include "online_ctr.h"
#include "bin_tracker.h"
#include "rand_score.h"
#include "error_functions.h"
#include "index_hash_calcer.h"
#include "split.h"
#include <catboost/libs/model/tensor_struct.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/vector.h>

struct TFeatureScore {
    TSplit Split;
    int ScoreGroup;
    TRandomScore Score;

    size_t GetHash() const {
        size_t hashValue = Split.GetHash();
        hashValue = MultiHash(hashValue,
                              Score.StDev,
                              Score.Val,
                              ScoreGroup);
        return hashValue;
    }
};

yvector<double> CalcScore(
    const TAllFeatures& af,
    const yvector<int>& splitsCount,
    const TFold& fold,
    const yvector<TIndexType>& indices,
    const TSplitCandidate& split,
    int depth,
    int ctrBorderCount,
    float l2Regularizer);
