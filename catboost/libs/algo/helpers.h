#pragma once

#include "learn_context.h"

#include <catboost/libs/data/pool.h>

#include <library/grid_creator/binarization.h>

#include <util/generic/vector.h>
#include <util/generic/hash_set.h>

void GenerateBorders(const TPool& pool, TLearnContext* ctx, TVector<TFloatFeature>* floatFeatures);

void ConfigureMalloc();

void CalcErrors(
    const TDataset& learnData,
    const TDatasetPtrs& testDataPtrs,
    const TVector<THolder<IMetric>>& errors,
    bool calcMetrics, // bool value for each error
    TLearnContext* ctx
);
