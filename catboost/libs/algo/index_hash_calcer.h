#pragma once

#include "projection.h"

#include <catboost/libs/data_new/objects.h>
#include <catboost/libs/helpers/clear_array.h>

#include <library/containers/dense_hash/dense_hash.h>

#include <util/generic/utility.h>


/// Calculate document hashes into range [begin,end) for CTR bucket identification.
/// @param proj - Projection delivering the feature ids to hash
/// @param objectsDataProvider - Values of features to hash
/// @param featuresSubsetIndexing - Use these indices when accessing raw arrays data
/// @param perfectHashedToHashedCatValuesMap - if not nullptr use it to Hash original hashed cat values
//                                             if nullptr - used perfectHashed values
/// @param begin, @param end - Result range
inline void CalcHashes(const TProjection& proj,
                       const NCB::TQuantizedForCPUObjectsDataProvider& objectsDataProvider,
                       const NCB::TFeaturesArraySubsetIndexing& featuresSubsetIndexing,
                       const NCB::TPerfectHashedToHashedCatValuesMap* perfectHashedToHashedCatValuesMap,
                       ui64* begin,
                       ui64* end) {
    const size_t sampleCount = end - begin;
    Y_VERIFY((size_t)featuresSubsetIndexing.Size() == sampleCount);
    if (sampleCount == 0) {
        return;
    }

    ui64* hashArr = begin;
    if (perfectHashedToHashedCatValuesMap) {
        for (const int featureIdx : proj.CatFeatures) {
            const auto& ohv = (*perfectHashedToHashedCatValuesMap)[featureIdx];

            NCB::SubsetWithAlternativeIndexing(
                objectsDataProvider.GetCatFeature((ui32)featureIdx),
                &featuresSubsetIndexing
            ).ForEach(
                [hashArr, &ohv] (ui32 i, ui32 featureValue) {
                    hashArr[i] = CalcHash(hashArr[i], (ui64)(int)ohv[featureValue]);
                }
            );
        }
    } else {
        for (const int featureIdx : proj.CatFeatures) {
            NCB::SubsetWithAlternativeIndexing(
                objectsDataProvider.GetCatFeature((ui32)featureIdx),
                &featuresSubsetIndexing
            ).ForEach(
                [hashArr] (ui32 i, ui32 featureValue) {
                    hashArr[i] = CalcHash(hashArr[i], (ui64)featureValue + 1);
                }
            );
        }
    }

    for (const TBinFeature& feature : proj.BinFeatures) {
        NCB::SubsetWithAlternativeIndexing(
            objectsDataProvider.GetFloatFeature((ui32)feature.FloatFeature),
            &featuresSubsetIndexing
        ).ForEach(
            [feature, hashArr] (ui32 i, ui8 featureValue) {
                const bool isTrueFeature = IsTrueHistogram(featureValue, (ui8)feature.SplitIdx);
                hashArr[i] = CalcHash(hashArr[i], (ui64)isTrueFeature);
            }
        );
    }

    const auto& quantizedFeaturesInfo = *objectsDataProvider.GetQuantizedFeaturesInfo();

    for (const TOneHotSplit& feature : proj.OneHotFeatures) {
        auto catFeatureIdx = NCB::TCatFeatureIdx((ui32)feature.CatFeatureIdx);
        const auto uniqueValuesCounts = quantizedFeaturesInfo.GetUniqueValuesCounts(
            catFeatureIdx
        );
        const ui32 maxBin = uniqueValuesCounts.OnLearnOnly;

        NCB::SubsetWithAlternativeIndexing(
            objectsDataProvider.GetCatFeature(*catFeatureIdx),
            &featuresSubsetIndexing
        ).ForEach(
            [feature, hashArr, maxBin] (ui32 i, ui32 featureValue) {
                const bool isTrueFeature = IsTrueOneHotFeature(Min(featureValue, maxBin), (ui32)feature.Value);
                hashArr[i] = CalcHash(hashArr[i], (ui64)isTrueFeature);
            }
        );
    }
}

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
