#pragma once

#include "fold.h"
#include "score_calcer.h"
#include "target_classifier.h"
#include "learn_context.h"
#include "error_functions.h"

#include <catboost/libs/logging/profile_info.h>

#include <util/generic/vector.h>
#include <util/random/fast.h>

void TrimOnlineCTRcache(const TVector<TFold*>& folds);

void GreedyTensorSearch(const TTrainData& data,
                        const TVector<int>& splitCounts,
                        double modelLength,
                        TProfileInfo& profile,
                        TFold* fold,
                        TLearnContext* ctx,
                        TSplitTree* resSplitTree);
