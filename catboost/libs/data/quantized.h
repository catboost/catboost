#pragma once

#include "pool.h"

#include <catboost/libs/data/load_data.h>
#include <catboost/libs/quantized_pool/pool.h>

#include <util/generic/hash.h>
#include <util/generic/vector.h>

THashMap<size_t, size_t> GetColumnIndexToFeatureIndexMap(const NCB::TQuantizedPool& pool);

TPoolMetaInfo GetPoolMetaInfo(const NCB::TQuantizedPool& pool);

TVector<int> GetCategoricalFeatureIndices(const NCB::TQuantizedPool& pool);

TVector<int> GetIgnoredFeatureIndices(const NCB::TQuantizedPool& pool);
