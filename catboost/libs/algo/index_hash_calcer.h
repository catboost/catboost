#pragma once

#include "fold.h"
#include "online_ctr.h"
#include "train_data.h"
#include <catboost/libs/model/tensor_struct.h>
#include <catboost/libs/helpers/clear_array.h>

#include <library/containers/dense_hash/dense_hash.h>

inline void CalcHashes(const TProjection& proj,
                       const TAllFeatures& af,
                       size_t sampleCount,
                       const yvector<int>& learnPermutation,
                       yvector<ui64>* res) {
    yvector<ui64>& hashArr = *res;
    Clear(&hashArr, sampleCount);
    const size_t learnSize = learnPermutation.size();
    for (const int featureIdx : proj.CatFeatures) {
        const int* featureValues = af.CatFeatures[featureIdx].data();
        for (size_t i = 0; i < learnSize; ++i) {
            hashArr[i] = CalcHash(hashArr[i], (ui64)featureValues[learnPermutation[i]]);
        }
        for (size_t i = learnSize; i < sampleCount; ++i) {
            hashArr[i] = CalcHash(hashArr[i], (ui64)featureValues[i]);
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

    for (const TOneHotFeature& feature : proj.OneHotFeatures) {
        const int* featureValues = af.CatFeatures[feature.CatFeatureIdx].data();
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

inline void CalcHashes(const TProjection& proj,
                       const TTrainData& data,
                       const TFold& fold,
                       yvector<ui64>* res) {
    CalcHashes(proj, data.AllFeatures, fold.EffectiveDocCount, fold.LearnPermutation, res);
}



/* Function for calculation of zero based bucket numbers for given hash values array.
   if number of unique values in hashVecPtr is greater than topSize and topSize + trashMask + 1 > learnSize
   only topSize hash values would be reindexed directly, rest will be remapped into trash bins: trash_hash = hash & trashMask
   trashMask is bitmask

   Function returns pair of (number of leaves for learn, number of leaves for test)
*/
std::pair<size_t, size_t> ReindexHash(size_t learnSize, ui64 topSize, yvector<ui64>* hashVecPtr, TDenseHash<ui64, ui32>* reindexHashPtr);
