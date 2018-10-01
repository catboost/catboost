#pragma once

#include "pool.h"

#include <catboost/libs/pool_builder/pool_builder.h>

#include <util/generic/hash.h>
#include <util/generic/vector.h>

// Returns hash map M s.t.
// M.has(C) iff column C contains float, categorical, or sparse feature
// if M.has(C), F.at(C) is flat index of feature in column C
THashMap<size_t, size_t> GetColumnIndexToFlatIndexMap(const NCB::TQuantizedPool& pool);

TVector<TString> GetFlatFeatureNames(const NCB::TQuantizedPool& pool);

THashMap<size_t, size_t> GetColumnIndexToNumericFeatureIndexMap(const NCB::TQuantizedPool& pool);

TPoolMetaInfo GetPoolMetaInfo(const NCB::TQuantizedPool& pool, bool hasAdditionalGroupWeight);

// Returns flat indices of all categorical features
// Sorted from min to max
TVector<int> GetCategoricalFeatureIndices(const NCB::TQuantizedPool& pool);

// Returns flat indices of all ignored features
// Sorted from min to max
TVector<int> GetIgnoredFlatIndices(const NCB::TQuantizedPool& pool);
