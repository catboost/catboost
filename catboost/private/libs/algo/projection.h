#pragma once

#include <catboost/libs/model/split.h>

#include <library/cpp/binsaver/bin_saver.h>

#include <util/digest/multi.h>
#include <util/generic/algorithm.h>
#include <util/generic/vector.h>


template <class T>
static bool HasDuplicates(const TVector<T>& x) {
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

public:
    TBinFeature() = default;

    TBinFeature(int floatFeature, int splitIdx)
        : FloatFeature(floatFeature)
        , SplitIdx(splitIdx)
    {
    }

    bool operator==(const TBinFeature& other) const {
        return FloatFeature == other.FloatFeature && SplitIdx == other.SplitIdx;
    }

    bool operator<(const TBinFeature& other) const {
        return std::tie(FloatFeature, SplitIdx) < std::tie(other.FloatFeature, other.SplitIdx);
    }

    SAVELOAD(FloatFeature, SplitIdx);
    Y_SAVELOAD_DEFINE(FloatFeature, SplitIdx);

    ui64 GetHash() const {
        return MultiHash(FloatFeature, SplitIdx);
    }
};

template <>
struct THash<TBinFeature> {
    inline size_t operator()(const TBinFeature& projection) const {
        return projection.GetHash();
    }
};

struct TProjection {
    TVector<int> CatFeatures;
    TVector<TBinFeature> BinFeatures;
    TVector<TOneHotSplit> OneHotFeatures;

public:
    bool operator==(const TProjection& other) const {
        return CatFeatures == other.CatFeatures &&
            BinFeatures == other.BinFeatures &&
            OneHotFeatures == other.OneHotFeatures;
    }

    bool operator!=(const TProjection& other) const {
        return !(*this == other);
    }

    bool operator<(const TProjection& other) const {
        return std::tie(CatFeatures, BinFeatures, OneHotFeatures) <
            std::tie(other.CatFeatures, other.BinFeatures, other.OneHotFeatures);
    }

    SAVELOAD(CatFeatures, BinFeatures, OneHotFeatures);
    Y_SAVELOAD_DEFINE(CatFeatures, BinFeatures, OneHotFeatures);

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

    bool HasSingleFeature() const {
        return BinFeatures.ysize() + CatFeatures.ysize() == 1;
    }

    void AddCatFeature(int f) {
        CatFeatures.push_back(f);
        Sort(CatFeatures.begin(), CatFeatures.end());
    }

    void AddBinFeature(const TBinFeature& f) {
        BinFeatures.push_back(f);
        Sort(BinFeatures.begin(), BinFeatures.end());
    }

    void AddOneHotFeature(const TOneHotSplit& f) {
        OneHotFeatures.push_back(f);
        Sort(OneHotFeatures.begin(), OneHotFeatures.end());
    }

    size_t GetHash() const {
        TVecHash<int> intVectorHash;
        TVecHash<TBinFeature> binFeatureHash;
        if (OneHotFeatures.empty()) {
            return MultiHash(intVectorHash(CatFeatures), binFeatureHash(BinFeatures));
        }
        TVecHash<TOneHotSplit> oneHotFeatureHash;
        return MultiHash(
            intVectorHash(CatFeatures),
            binFeatureHash(BinFeatures),
            oneHotFeatureHash(OneHotFeatures));
    }

    size_t GetFullProjectionLength() const {
        size_t addition = 0;
        if (BinFeatures.size() + OneHotFeatures.size() > 0) {
            addition = 1;
        }
        return CatFeatures.size() + addition;
    }
};

template <>
struct THash<TProjection> {
    inline size_t operator()(const TProjection& projection) const {
        return projection.GetHash();
    }
};

template <>
inline void Out<TProjection>(IOutputStream& out, const TProjection& proj) {
    for (auto cf : proj.CatFeatures) {
        out << "c" << cf;
    }
    for (auto bf : proj.BinFeatures) {
        out << "(f" << bf.FloatFeature << ";b" << bf.SplitIdx << ")";
    }
    for (auto ohef : proj.OneHotFeatures) {
        out << "(c" << ohef.CatFeatureIdx << ";v" << ohef.Value << ")";
    }
}
