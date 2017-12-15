#pragma once

#include <catboost/libs/data/pool.h>

#include <library/threading/local_executor/local_executor.h>

#include <numeric>

void ApplyPermutation(const TVector<ui64>& permutation, TPool* pool, NPar::TLocalExecutor* localExecutor);
TVector<ui64> InvertPermutation(const TVector<ui64>& permutation);
TVector<ui64> CreateOrderByKey(const TVector<ui64>& key);
