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
    const TTrainData& data,
    const TVector<THolder<IMetric>>& errors,
    bool hasTrain,
    bool hasTest,
    TLearnContext* ctx
);

namespace {
    template <typename T>
    struct TMinMax {
        T Min;
        T Max;
        TMinMax(const TVector<T>& v) {
            Min = *MinElement(v.begin(), v.end());
            Max = *MaxElement(v.begin(), v.end());
        }
    };
}
