#pragma once

#include "projection.h"
#include "train_data.h"

#include <catboost/libs/helpers/clear_array.h>

#include <library/containers/dense_hash/dense_hash.h>

/**
 * Calculate hashes for CTR bucket identification
 * @param proj
 * @param af
 * @param sampleCount
 * @param learnPermutation
 * @param calculateExactCatHashes enable calculation of hashes for model (use original cat feature hash values)
 * @param [out]res result
 */
inline void CalcHashes(const TProjection& proj,
                       const TAllFeatures& af,
                       size_t sampleCount,
                       const TVector<size_t>& learnPermutation,
                       bool calculateExactCatHashes,
                       TVector<ui64>* res) {
    TVector<ui64>& hashArr = *res;
    Clear(&hashArr, sampleCount);
    TVector<int> exactValues;
    const size_t learnSize = learnPermutation.size();
    for (const int featureIdx : proj.CatFeatures) {
        const int* featureValues = af.CatFeaturesRemapped[featureIdx].data();
        if (calculateExactCatHashes) {
            // Calculate hashes for model CTR table
            exactValues.resize(af.CatFeaturesRemapped[featureIdx].size());
            auto& ohv = af.OneHotValues[featureIdx];
            for (size_t i = 0; i < sampleCount; ++i) {
                exactValues[i] = ohv[featureValues[i]];
            }
            featureValues = exactValues.data();
            for (size_t i = 0; i < learnSize; ++i) {
                hashArr[i] = CalcHash(hashArr[i], (ui64)featureValues[learnPermutation[i]]);
            }
            for (size_t i = learnSize; i < sampleCount; ++i) {
                hashArr[i] = CalcHash(hashArr[i], (ui64)featureValues[i]);
            }
        } else {
            for (size_t i = 0; i < learnSize; ++i) {
                hashArr[i] = CalcHash(hashArr[i], (ui64)featureValues[learnPermutation[i]] + 1);
            }
            for (size_t i = learnSize; i < sampleCount; ++i) {
                hashArr[i] = CalcHash(hashArr[i], (ui64)featureValues[i] + 1);
            }
        }
    }

    for (const TBinFeature& feature : proj.BinFeatures) {
        const ui8* featureValues = af.FloatHistograms[feature.FloatFeature].data();
        for (size_t i = 0; i < learnSize; ++i) {
            const bool isTrueFeature = IsTrueHistogram(featureValues[learnPermutation[i]], feature.SplitIdx);
            hashArr[i] = CalcHash(hashArr[i], (ui64)isTrueFeature);
        }
        for (size_t i = learnSize; i < sampleCount; ++i) {
            const bool isTrueFeature = IsTrueHistogram(featureValues[i], feature.SplitIdx);
            hashArr[i] = CalcHash(hashArr[i], (ui64)isTrueFeature);
        }
    }

    for (const TOneHotSplit& feature : proj.OneHotFeatures) {
        const int* featureValues = af.CatFeaturesRemapped[feature.CatFeatureIdx].data();
        for (size_t i = 0; i < learnSize; ++i) {
            const bool isTrueFeature = IsTrueOneHotFeature(featureValues[learnPermutation[i]], feature.Value);
            hashArr[i] = CalcHash(hashArr[i], (ui64)isTrueFeature);
        }
        for (size_t i = learnSize; i < sampleCount; ++i) {
            const bool isTrueFeature = IsTrueOneHotFeature(featureValues[i], feature.Value);
            hashArr[i] = CalcHash(hashArr[i], (ui64)isTrueFeature);
        }
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

/// Use reindexHash to reindex hash values in range [begin,end).
/// If a hash value is not present in reindexHash, map it to reindexHash.Size().
void UseReindexHash(const TDenseHash<ui64, ui32>& reindexHash, ui64* begin, ui64* end);
