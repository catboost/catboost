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


/**
 *  Function for calculation of zero based bucket numbers for given hash values array.
 *  if number of unique values in hashVecPtr is greater than topSize and topSize + trashMask + 1 > learnSize
 *  only topSize hash values would be reindexed directly, rest will be remapped into single bin
 *  Function returns pair of (number of leaves for learn, number of leaves for test)
*/
std::pair<size_t, size_t> ReindexHash(size_t learnSize, ui64 topSize, TVector<ui64>* hashVecPtr, TDenseHash<ui64, ui32>* reindexHashPtr);
