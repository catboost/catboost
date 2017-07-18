#pragma once

#include <library/grid_creator/binarization.h>
#include <library/threading/local_executor/local_executor.h>

#include <catboost/libs/data/pool.h>

#include <util/generic/vector.h>
#include <util/generic/hash_set.h>

yvector<yvector<float>> GenerateBorders(
    const yvector<TDocInfo>& docInfos,
    const yhash_set<int>& categFeatures,
    NPar::TLocalExecutor& localExecutor,
    int borderCount,
    const EBorderSelectionType borderType);

void ApplyPermutation(const yvector<size_t>& permutation, TPool* pool);
yvector<size_t> InvertPermutation(const yvector<size_t>& permutation);

int GetClassesCount(const yvector<float>& target, int classesCount);
