#pragma once

#include "fold.h"
#include "learn_context.h"
#include "dataset.h"

#include <util/generic/vector.h>

void TrimOnlineCTRcache(const TVector<TFold*>& folds);

void GreedyTensorSearch(const TDataset& learnData,
                        const TDataset* testData,
                        const TVector<int>& splitCounts,
                        double modelLength,
                        TProfileInfo& profile,
                        TFold* fold,
                        TLearnContext* ctx,
                        TSplitTree* resSplitTree);
