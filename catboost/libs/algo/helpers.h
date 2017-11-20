#pragma once

#include "learn_context.h"

#include <catboost/libs/data/pool.h>

#include <library/grid_creator/binarization.h>

#include <util/generic/vector.h>
#include <util/generic/hash_set.h>

void GenerateBorders(const TPool& pool, TLearnContext* ctx, TVector<TFloatFeature>* floatFeatures);

void ApplyPermutation(const TVector<size_t>& permutation, TPool* pool);
TVector<size_t> InvertPermutation(const TVector<size_t>& permutation);
int GetClassesCount(const TVector<float>& target, int classesCount);
