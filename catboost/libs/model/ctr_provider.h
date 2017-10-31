#pragma once

#include "hash.h"
#include "online_ctr.h"

class IFeatureIndexProvider {
public:
    virtual ~IFeatureIndexProvider() {
    }
    virtual int GetBinFeatureIdx(const TBinFeature& feature) const = 0;
    virtual int GetBinFeatureIdx(const TOneHotFeature& feature) const = 0;
};

class ICtrProvider : public TThrRefBase {
public:
    virtual ~ICtrProvider() {
    }

    virtual bool HasNeededCtrs(const yvector<TModelCtr>& neededCtrs) const = 0;

    virtual void CalcCtrs(
        const yvector<TModelCtr>& neededCtrs,
        const TConstArrayRef<ui8>& binarizedFeatures, // vector of binarized float & one hot features
        const TConstArrayRef<int>& hashedCatFeatures,
        const IFeatureIndexProvider& binFeatureIndexProvider,
        size_t docCount,
        TArrayRef<float> result) = 0;
};

// slow reference realization
inline ui64 CalcHash(
    const TProjection& proj,
    const TConstArrayRef<ui8>& binarizedFeatures,
    const TConstArrayRef<int>& hashedCatFeatures,
    const IFeatureIndexProvider& binFeatureIndexProvider) {
    ui64 result = 0;
    for (const int featureIdx : proj.CatFeatures) {
        result = CalcHash(result, (ui64)hashedCatFeatures[featureIdx]);
    }
    for (const TBinFeature& feature : proj.BinFeatures) {
        result = CalcHash(result, (ui64)binarizedFeatures[binFeatureIndexProvider.GetBinFeatureIdx(feature)]);
    }
    for (const TOneHotFeature& feature : proj.OneHotFeatures) {
        result = CalcHash(result, (ui64)binarizedFeatures[binFeatureIndexProvider.GetBinFeatureIdx(feature)]);
    }
    return result;
}

inline void CalcHashes(
    const TProjection& proj,
    const TConstArrayRef<ui8>& binarizedFeatures,
    const TConstArrayRef<int>& hashedCatFeatures,
    const IFeatureIndexProvider& binFeatureIndexProvider,
    size_t docCount,
    yvector<ui64>* result) {
    result->resize(docCount);
    std::fill(result->begin(), result->end(), 0);
    ui64* ptr = result->data();
    for (const int featureIdx : proj.CatFeatures) {
        auto valPtr = &hashedCatFeatures[featureIdx * docCount];
        for (size_t i = 0; i < docCount; ++i) {
            ptr[i] = CalcHash(ptr[i], (ui64)valPtr[i]);
        }
    }
    for (const TBinFeature& feature : proj.BinFeatures) {
        const auto idx = binFeatureIndexProvider.GetBinFeatureIdx(feature) * docCount;
        const ui8* binFPtr = &binarizedFeatures[idx];
        for (size_t i = 0; i < docCount; ++i) {
            ptr[i] = CalcHash(ptr[i], (ui64)binFPtr[i]);
        }
    }
    for (const TOneHotFeature& feature : proj.OneHotFeatures) {
        const auto idx = binFeatureIndexProvider.GetBinFeatureIdx(feature) * docCount;
        auto valPtr = &binarizedFeatures[idx];
        for (size_t i = 0; i < docCount; ++i) {
            ptr[i] = CalcHash(ptr[i], (ui64)valPtr[i]);
        }
    }
}
