#pragma once

#include "helpers.h"
#include <catboost/cuda/data/columns.h>
#include <catboost/cuda/ctrs/ctr.h>

#include <util/system/types.h>
#include <util/generic/map.h>
#include <util/generic/vector.h>
#include <util/generic/set.h>
#include <catboost/libs/model/hash.h>
#include <util/digest/multi.h>
#include <util/generic/algorithm.h>

enum class EBinSplitType {
    TakeBin,
    TakeGreater
};

struct TBinarySplit {
    ui32 FeatureId = 0; //from feature manager
    ui32 BinIdx = 0;
    EBinSplitType SplitType;

    bool operator<(const TBinarySplit& other) const {
        return std::tie(FeatureId, BinIdx, SplitType) < std::tie(other.FeatureId, other.BinIdx, other.SplitType);
    }

    bool operator==(const TBinarySplit& other) const {
        return std::tie(FeatureId, BinIdx, SplitType) == std::tie(other.FeatureId, other.BinIdx, other.SplitType);
    }

    bool operator!=(const TBinarySplit& other) const {
        return !(*this == other);
    }

    ui64 GetHash() const {
        return MultiHash(FeatureId, BinIdx, SplitType);
    }
};

template <>
struct THash<TBinarySplit> {
    inline size_t operator()(const TBinarySplit& value) const {
        return value.GetHash();
    }
};

template <class TVector>
inline void Unique(TVector& vector) {
    ui64 size = Unique(vector.begin(), vector.end()) - vector.begin();
    vector.resize(size);
}

struct TFeatureTensor {
public:
    bool IsSimple() const {
        return (Splits.size() + CatFeatures.size()) == 1;
    }

    TFeatureTensor& AddBinarySplit(const TBinarySplit& bin) {
        Splits.push_back(bin);
        SortUniqueSplits();
        return *this;
    }

    TFeatureTensor& AddBinarySplit(const yvector<TBinarySplit>& splits) {
        for (auto& bin : splits) {
            Splits.push_back(bin);
        }
        SortUniqueSplits();
        return *this;
    }

    void SortUniqueSplits() {
        Sort(Splits.begin(), Splits.end());
        Unique(Splits);
    }

    TFeatureTensor& AddCatFeature(ui32 featureId) {
        CatFeatures.push_back(featureId);
        SortUniqueCatFeatures();
        return *this;
    }

    void SortUniqueCatFeatures() {
        Sort(CatFeatures.begin(), CatFeatures.end());
        Unique(CatFeatures);
    }

    TFeatureTensor& AddTensor(const TFeatureTensor& tensor) {
        for (auto& split : tensor.Splits) {
            Splits.push_back(split);
        }
        for (auto& catFeature : tensor.CatFeatures) {
            CatFeatures.push_back(catFeature);
        }
        SortUniqueSplits();
        SortUniqueCatFeatures();
        return *this;
    }

    bool operator==(const TFeatureTensor& other) const {
        return (Splits == other.GetSplits()) && (CatFeatures == other.GetCatFeatures());
    }

    bool operator!=(const TFeatureTensor& other) const {
        return !(*this == other);
    }

    bool IsEmpty() const {
        return CatFeatures.size() == 0 && Splits.size() == 0;
    }

    ui64 Size() const {
        return CatFeatures.size() + Splits.size();
    }

    ui64 GetHash() const {
        return MultiHash(TVecHash<TBinarySplit>()(Splits), VecCityHash(CatFeatures));
    }

    bool operator<(const TFeatureTensor& other) const {
        return std::tie(Splits, CatFeatures) < std::tie(other.Splits, other.CatFeatures);
    }

    bool IsSubset(const TFeatureTensor other) const {
        return ::IsSubset(Splits, other.Splits) && ::IsSubset(CatFeatures, other.CatFeatures);
    }

    const yvector<TBinarySplit>& GetSplits() const {
        return Splits;
    }

    const yvector<ui32>& GetCatFeatures() const {
        return CatFeatures;
    }

    SAVELOAD(Splits, CatFeatures);

private:
    yvector<TBinarySplit> Splits;
    yvector<ui32> CatFeatures;
};

struct TCtr {
    TFeatureTensor FeatureTensor;
    TCtrConfig Configuration;

    TCtr(const TCtr& other) = default;
    TCtr() = default;

    TCtr(const TFeatureTensor& tensor,
         const TCtrConfig& config)
        : FeatureTensor(tensor)
        , Configuration(config)
    {
    }

    bool operator==(const TCtr& other) const {
        return std::tie(FeatureTensor, Configuration) == std::tie(other.FeatureTensor, other.Configuration);
    }

    bool operator!=(const TCtr& other) const {
        return !(*this == other);
    }

    ui64 GetHash() const {
        return MultiHash(FeatureTensor, Configuration);
    }

    bool IsSimple() const {
        return FeatureTensor.IsSimple();
    }

    bool operator<(const TCtr& other) const {
        return std::tie(FeatureTensor, Configuration) < std::tie(other.FeatureTensor, other.Configuration);
    }

    SAVELOAD(FeatureTensor, Configuration);
};

template <>
struct THash<TFeatureTensor> {
    inline size_t operator()(const TFeatureTensor& tensor) const {
        return tensor.GetHash();
    }
};

template <>
struct THash<TCtrConfig> {
    inline size_t operator()(const TCtrConfig& config) const {
        return config.GetHash();
    }
};

template <>
struct THash<TCtr> {
    inline size_t operator()(const TCtr& value) const {
        return value.GetHash();
    }
};
