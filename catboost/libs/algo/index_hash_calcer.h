#pragma once

#include "projection.h"
#include "dataset.h"

#include <catboost/libs/helpers/clear_array.h>

#include <library/containers/dense_hash/dense_hash.h>

/// Calculate document hashes into range [begin,end) for CTR bucket identification.
/// @param proj - Projection delivering the feature ids to hash
/// @param allFeatures - Values of features to hash
/// @param offset - Begin from this offset when accessing `allFeatures`
/// @param learnPermutation - Use this permutation when accessing `allFeatures`
/// @param calculateExactCatHashes - Hash original cat features (true) or one-hot-encoded (false)
/// @param begin, @param end - Result range
inline void CalcHashes(const TProjection& proj,
                       const TAllFeatures& allFeatures,
                       size_t offset,
                       const TVector<size_t>* learnPermutation,
                       bool calculateExactCatHashes,
                       ui64* begin,
                       ui64* end) {
    const size_t sampleCount = end - begin;
    if (sampleCount == 0) {
        return;
    }
    if (learnPermutation != nullptr) {
        Y_VERIFY(offset == 0);
        Y_VERIFY(sampleCount == learnPermutation->size());
    }

    ui64* hashArr = begin;
    if (calculateExactCatHashes) {
        TVector<int> exactValues;
        for (const int featureIdx : proj.CatFeatures) {
            const int* featureValues = offset + allFeatures.CatFeaturesRemapped[featureIdx].data();
            // Calculate hashes for model CTR table
            exactValues.resize(sampleCount);
            auto& ohv = allFeatures.OneHotValues[featureIdx];
            for (size_t i = 0; i < sampleCount; ++i) {
                exactValues[i] = ohv[featureValues[i]];
            }
            if (learnPermutation != nullptr) {
                const auto& perm = *learnPermutation;
                for (size_t i = 0; i < sampleCount; ++i) {
                    hashArr[i] = CalcHash(hashArr[i], (ui64)exactValues[perm[i]]);
                }
            } else {
                for (size_t i = 0; i < sampleCount; ++i) {
                    hashArr[i] = CalcHash(hashArr[i], (ui64)exactValues[i]);
                }
            }
        }
    } else {
        for (const int featureIdx : proj.CatFeatures) {
            const int* featureValues = offset + allFeatures.CatFeaturesRemapped[featureIdx].data();
            if (learnPermutation != nullptr) {
                const auto& perm = *learnPermutation;
                for (size_t i = 0; i < sampleCount; ++i) {
                    hashArr[i] = CalcHash(hashArr[i], (ui64)featureValues[perm[i]] + 1);
                }
            } else {
                for (size_t i = 0; i < sampleCount; ++i) {
                    hashArr[i] = CalcHash(hashArr[i], (ui64)featureValues[i] + 1);
                }
            }
        }
    }

    for (const TBinFeature& feature : proj.BinFeatures) {
        const ui8* featureValues = offset + allFeatures.FloatHistograms[feature.FloatFeature].data();
        if (learnPermutation != nullptr) {
            const auto& perm = *learnPermutation;
            for (size_t i = 0; i < sampleCount; ++i) {
                const bool isTrueFeature = IsTrueHistogram(featureValues[perm[i]], feature.SplitIdx);
                hashArr[i] = CalcHash(hashArr[i], (ui64)isTrueFeature);
            }
        } else {
            for (size_t i = 0; i < sampleCount; ++i) {
                const bool isTrueFeature = IsTrueHistogram(featureValues[i], feature.SplitIdx);
                hashArr[i] = CalcHash(hashArr[i], (ui64)isTrueFeature);
            }
        }
    }

    for (const TOneHotSplit& feature : proj.OneHotFeatures) {
        const int* featureValues = offset + allFeatures.CatFeaturesRemapped[feature.CatFeatureIdx].data();
        if (learnPermutation != nullptr) {
            const auto& perm = *learnPermutation;
            for (size_t i = 0; i < sampleCount; ++i) {
                const bool isTrueFeature = IsTrueOneHotFeature(featureValues[perm[i]], feature.Value);
                hashArr[i] = CalcHash(hashArr[i], (ui64)isTrueFeature);
            }
        } else {
            for (size_t i = 0; i < sampleCount; ++i) {
                const bool isTrueFeature = IsTrueOneHotFeature(featureValues[i], feature.Value);
                hashArr[i] = CalcHash(hashArr[i], (ui64)isTrueFeature);
            }
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
