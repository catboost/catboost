#pragma once

#include <catboost/libs/data/data_provider.h>

#include <util/generic/vector.h>


class TFold;
class TLearnContext;
class TProfileInfo;
struct TSplitTree;


void TrimOnlineCTRcache(const TVector<TFold*>& folds);

void GreedyTensorSearch(
    const NCB::TTrainingForCPUDataProviders& data,
    double modelLength,
    TProfileInfo& profile,
    TFold* fold,
    TLearnContext* ctx,
    TSplitTree* resSplitTree);
