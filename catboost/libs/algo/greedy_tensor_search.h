#pragma once

#include "fold.h"
#include "learn_context.h"
#include "train_data.h"

#include <util/generic/vector.h>

void TrimOnlineCTRcache(const TVector<TFold*>& folds);

void GreedyTensorSearch(const TTrainData& learnData,
                        const TTrainData* testData,
                        const TVector<int>& splitCounts,
                        double modelLength,
                        TProfileInfo& profile,
                        TFold* fold,
                        TLearnContext* ctx,
                        TSplitTree* resSplitTree);
