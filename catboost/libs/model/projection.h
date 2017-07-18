#pragma once

#include "hash.h"

#include <util/generic/vector.h>
#include <util/generic/algorithm.h>
#include <util/digest/multi.h>

#include <library/binsaver/bin_saver.h>

template <class T>
static bool HasDuplicates(const yvector<T>& x) {
    int count = x.ysize();
    for (int i = 0; i < count; ++i) {
        for (int j = i + 1; j < count; ++j) {
            if (x[j] == x[i]) {
                return true;
            }
        }
    }
    return false;
}

struct TBinFeature {
    int FloatFeature = 0;
    int SplitIdx = 0;

    bool operator==(const TBinFeature& other) const {
        return FloatFeature == other.FloatFeature && SplitIdx == other.SplitIdx;
    }

    bool operator<(const TBinFeature& other) const {
        return std::tie(FloatFeature, SplitIdx) < std::tie(other.FloatFeature, other.SplitIdx);
    }

    TBinFeature() = default;

    TBinFeature(int floatFeature, int splitIdx)
        : FloatFeature(floatFeature)
        , SplitIdx(splitIdx)
    {
    }

    ui64 GetHash() const {
        return MultiHash(FloatFeature, SplitIdx);
    }

    Y_SAVELOAD_DEFINE(FloatFeature, SplitIdx);
};


struct TOneHotFeature {
    int CatFeatureIdx = 0;
    int Value = 0;

    bool operator==(const TOneHotFeature& other) const {
        return CatFeatureIdx == other.CatFeatureIdx && Value == other.Value;
    }

    bool operator<(const TOneHotFeature& other) const {
        return std::tie(CatFeatureIdx, Value) < std::tie(other.CatFeatureIdx, other.Value);
    }

    TOneHotFeature() = default;

    TOneHotFeature(int catFeatureIdx, int value)
        : CatFeatureIdx(catFeatureIdx)
        , Value(value)
    {
    }

    ui64 GetHash() const {
        return MultiHash(CatFeatureIdx, Value);
    }

    Y_SAVELOAD_DEFINE(CatFeatureIdx, Value);
};

struct TProjection {
    yvector<int> CatFeatures;
    yvector<TBinFeature> BinFeatures;
    yvector<TOneHotFeature> OneHotFeatures;

    Y_SAVELOAD_DEFINE(CatFeatures, BinFeatures, OneHotFeatures)

    void Add(const TProjection& proj) {
        CatFeatures.insert(CatFeatures.end(), proj.CatFeatures.begin(), proj.CatFeatures.end());
        BinFeatures.insert(BinFeatures.end(), proj.BinFeatures.begin(), proj.BinFeatures.end());
        OneHotFeatures.insert(OneHotFeatures.end(), proj.OneHotFeatures.begin(), proj.OneHotFeatures.end());
        Sort(CatFeatures.begin(), CatFeatures.end());
        Sort(BinFeatures.begin(), BinFeatures.end());
        Sort(OneHotFeatures.begin(), OneHotFeatures.end());
    }

    bool IsRedundant() const {
        return HasDuplicates(CatFeatures) || HasDuplicates(BinFeatures) || HasDuplicates(OneHotFeatures);
    }

    bool IsEmpty() const {
        return CatFeatures.empty() && BinFeatures.empty() && OneHotFeatures.empty();
    }

    bool IsSingleCatFeature() const {
        return BinFeatures.empty() && OneHotFeatures.empty() && CatFeatures.ysize() == 1;
    }

    void AddCatFeature(int f) {
        CatFeatures.push_back(f);
        Sort(CatFeatures.begin(), CatFeatures.end());
    }

    void AddBinFeature(const TBinFeature& f) {
        BinFeatures.push_back(f);
        Sort(BinFeatures.begin(), BinFeatures.end());
    }

    void AddOneHotFeature(const TOneHotFeature& f) {
        OneHotFeatures.push_back(f);
        Sort(OneHotFeatures.begin(), OneHotFeatures.end());
    }

    size_t GetHash() const {
        TVecHash<int> intVectorHash;
        TVecHash<TBinFeature> binFeatureHash;
        if (OneHotFeatures.empty()) {
            return MultiHash(intVectorHash(CatFeatures), binFeatureHash(BinFeatures));
        }
        TVecHash<TOneHotFeature> oneHotFeatureHash;
        return MultiHash(intVectorHash(CatFeatures), binFeatureHash(BinFeatures), oneHotFeatureHash(OneHotFeatures));
    }

    bool operator==(const TProjection& other) const {
        return CatFeatures == other.CatFeatures &&
               BinFeatures == other.BinFeatures &&
               OneHotFeatures == other.OneHotFeatures;
    }

    bool operator<(const TProjection& other) const {
        return std::tie(CatFeatures, BinFeatures, OneHotFeatures) <
               std::tie(other.CatFeatures, other.BinFeatures, other.OneHotFeatures);
    }
};

struct TProjHash {
    inline size_t operator()(const TProjection& projection) const {
        return projection.GetHash();
    }
};
