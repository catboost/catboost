#pragma once

#include "fold.h"
#include "learn_context.h"

#include <catboost/libs/data_new/data_provider.h>

#include <util/generic/vector.h>

void TrimOnlineCTRcache(const TVector<TFold*>& folds);

void GreedyTensorSearch(const NCB::TTrainingForCPUDataProviders& data,
                        double modelLength,
                        TProfileInfo& profile,
                        TFold* fold,
                        TLearnContext* ctx,
                        TSplitTree* resSplitTree);
