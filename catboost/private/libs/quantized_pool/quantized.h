#pragma once

#include "pool.h"

#include <catboost/private/libs/data_util/path_with_scheme.h>

#include <catboost/libs/data/meta_info.h>

#include <util/generic/hash.h>
#include <util/generic/vector.h>

// Returns hash map M s.t.
// M.has(C) iff column C contains float, categorical, or sparse feature
// if M.has(C), F.at(C) is flat index of feature in column C
THashMap<size_t, size_t> GetColumnIndexToTargetIndexMap(const NCB::TQuantizedPool& pool);

THashMap<size_t, size_t> GetColumnIndexToFlatIndexMap(const NCB::TQuantizedPool& pool);

THashMap<size_t, size_t> GetColumnIndexToBaselineIndexMap(const NCB::TQuantizedPool& pool);

NCB::TDataMetaInfo GetDataMetaInfo(
    const NCB::TQuantizedPool& pool,
    bool hasAdditionalGroupWeight,
    bool hasTimestamps,
    bool hasPairs,
    bool forceUnitAutoPairWeights,
    TMaybe<ui32> baselineCount,
    const NCB::TPathWithScheme& featureNamesPath, // featureNamesPath can be uninitialized
    const NCB::TPathWithScheme& poolMetaInfoPath); // poolMetaInfoPath can be uninitialized

// Returns flat indices of all ignored features
// Sorted from min to max
TVector<ui32> GetIgnoredFlatIndices(const NCB::TQuantizedPool& pool);
