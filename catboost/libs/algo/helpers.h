#pragma once

#include "learn_context.h"

#include <catboost/libs/data/pool.h>

#include <library/grid_creator/binarization.h>

#include <util/generic/vector.h>
#include <util/generic/hash_set.h>

void GenerateBorders(const TPool& pool, TLearnContext* ctx, TVector<TFloatFeature>* floatFeatures);

int GetClassesCount(const TVector<float>& target, int classesCount);

void ConfigureMalloc();

void CalcErrors(
    const TDataset& learnData,
    const TDataset& testData,
    const TVector<THolder<IMetric>>& errors,
    TLearnContext* ctx
);
