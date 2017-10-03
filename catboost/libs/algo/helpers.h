#pragma once

#include "learn_context.h"

#include <library/grid_creator/binarization.h>

#include <catboost/libs/data/pool.h>

#include <util/generic/vector.h>
#include <util/generic/hash_set.h>

void GenerateBorders(const yvector<TDocInfo>& docInfos, TLearnContext* ctx, yvector<yvector<float>>* borders, yvector<bool>* hasNans);

void ApplyPermutation(const yvector<size_t>& permutation, TPool* pool);
yvector<size_t> InvertPermutation(const yvector<size_t>& permutation);
int GetClassesCount(const yvector<float>& target, int classesCount);
