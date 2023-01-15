#pragma once

#include <catboost/libs/data/exclusive_feature_bundling.h>
#include <catboost/libs/data/objects.h>

#include <library/cpp/containers/dense_hash/dense_hash.h>

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>
#include <util/system/yassert.h>

#include <array>
#include <functional>


struct TProjection;


/// Calculate document hashes into range [begin,end) for CTR bucket identification.
/// @param proj - Projection delivering the feature ids to hash
/// @param objectsDataProvider - Values of features to hash
/// @param featuresSubsetIndexing - Use these indices when accessing raw arrays data
/// @param perfectHashedToHashedCatValuesMap - if not nullptr use it to Hash original hashed cat values
//                                             if nullptr - used perfectHashed values
/// @param processAggregatedFeatures - process bundled, grouped and binary features in packs.
///                                       Faster, but not compatible with current model format.
///                                       So, enabled only during training, disabled for FinalCtr.
/// @param begin, @param end - Result range
void CalcHashes(
    const TProjection& proj,
    const NCB::TQuantizedObjectsDataProvider& objectsDataProvider,
    const NCB::TFeaturesArraySubsetIndexing& featuresSubsetIndexing,
    const NCB::TPerfectHashedToHashedCatValuesMap* perfectHashedToHashedCatValuesMap,
    ui64* begin,
    ui64* end,
    NPar::ILocalExecutor* localExecutor);


/// Compute reindexHash and reindex hash values in range [begin,end).
/// After reindex, hash values belong to [0, reindexHash.Size()].
/// If reindexHash would become larger than topSize, keep only topSize most
/// frequent mappings and map other hash values to value reindexHash.Size().
/// @return the size of reindexHash.
size_t ComputeReindexHash(ui64 topSize, TDenseHash<ui64, ui32>* reindexHashPtr, ui64* begin, ui64* end);

/// Update reindexHash and reindex hash values in range [begin,end).
/// If a hash value is not present in reindexHash, then update reindexHash for that value.
/// @return the size of updated reindexHash.
size_t UpdateReindexHash(TDenseHash<ui64, ui32>* reindexHashPtr, ui64* begin, ui64* end);
