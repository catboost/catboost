#pragma once

#include "error_functions.h"
#include "fold.h"
#include "score_calcer.h"
#include "target_classifier.h"
#include "learn_context.h"
#include <catboost/libs/model/tensor_struct.h>

#include <catboost/libs/logging/profile_info.h>

#include <util/generic/vector.h>
#include <util/random/fast.h>

void TrimOnlineCTRcache(const yvector<TFold*>& folds);

void GreedyTensorSearch(const TTrainData& data,
                        const yvector<int>& splitCounts,
                        double modelLength,
                        float l2Regularizer,
                        float randomStrength,
                        TProfileInfo& profile,
                        TFold* fold,
                        TLearnContext* ctx,
                        TTensorStructure3* resTree,
                        yvector<TSplit>* resSplitTree);
