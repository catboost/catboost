#pragma once

#include "hash.h"

#include "fwd.h"
#include <catboost/private/libs/ctr_description/ctr_type.h>
#include <catboost/libs/helpers/guid.h>

#include <contrib/libs/flatbuffers/include/flatbuffers/flatbuffers.h>

#include <util/digest/multi.h>
#include <util/generic/vector.h>
#include <util/stream/fwd.h>
#include <util/str_stl.h>
#include <util/ysaveload.h>

#include <tuple>


class TModelPartsCachingSerializer;


/* make sure floating-point values do not contain negative zeros -
 * flatbuffers serializer will deserialize them as positive zeros
 */
inline void CanonizeFloatForFbs(float* value) {
    if (*value == -0.0f) {
        *value = 0.0f;
    }
}


struct TFloatSplit {
    int FloatFeature = 0;
    float Split = 0.f;

public:
    TFloatSplit() = default;
    TFloatSplit(int feature, float split)
        : FloatFeature(feature)
        , Split(split)
    {}

    bool operator==(const TFloatSplit& other) const {
        return std::tie(FloatFeature, Split) == std::tie(other.FloatFeature, other.Split);
    }

    bool operator<(const TFloatSplit& other) const {
        return std::tie(FloatFeature, Split) < std::tie(other.FloatFeature, other.Split);
    }

    ui64 GetHash() const {
        return MultiHash(FloatFeature, Split);
    }

    Y_SAVELOAD_DEFINE(FloatFeature, Split);

    /* make sure floating-point values do not contain negative zeros -
     * flatbuffers serializer will deserialize them as positive zeros
     */
    void Canonize() {
        CanonizeFloatForFbs(&Split);
    }
};

template <>
struct THash<TFloatSplit> {
    inline size_t operator()(const TFloatSplit& split) const {
        return split.GetHash();
    }
};

struct TOneHotSplit {
    int CatFeatureIdx = 0;
    int Value = 0;

public:
    TOneHotSplit() = default;

    TOneHotSplit(int featureIdx, int value)
        : CatFeatureIdx(featureIdx)
        , Value(value)
    {}

    bool operator==(const TOneHotSplit& other) const {
        return CatFeatureIdx == other.CatFeatureIdx && Value == other.Value;
    }

    bool operator<(const TOneHotSplit& other) const {
        return std::tie(CatFeatureIdx, Value) < std::tie(other.CatFeatureIdx, other.Value);
    }


    ui64 GetHash() const {
        return MultiHash(CatFeatureIdx, Value);
    }
    Y_SAVELOAD_DEFINE(CatFeatureIdx, Value);
};

template <>
struct THash<TOneHotSplit> {
    inline size_t operator()(const TOneHotSplit& projection) const {
        return projection.GetHash();
    }
};


struct TFeatureCombination {
    TVector<int> CatFeatures;
    TVector<TFloatSplit> BinFeatures;
    TVector<TOneHotSplit> OneHotFeatures;

public:
    bool operator==(const TFeatureCombination& other) const {
        return std::tie(CatFeatures, BinFeatures, OneHotFeatures) ==
            std::tie(other.CatFeatures, other.BinFeatures, other.OneHotFeatures);
    }

    bool operator!=(const TFeatureCombination& other) const {
        return !(*this == other);
    }

    bool operator<(const TFeatureCombination& other) const {
        return std::tie(CatFeatures, BinFeatures, OneHotFeatures) <
               std::tie(other.CatFeatures, other.BinFeatures, other.OneHotFeatures);
    }

    void Clear() {
        CatFeatures.clear();
        BinFeatures.clear();
        OneHotFeatures.clear();
    }

    Y_SAVELOAD_DEFINE(CatFeatures, BinFeatures, OneHotFeatures);

    bool IsSingleCatFeature() const {
        return BinFeatures.empty() && OneHotFeatures.empty() && CatFeatures.ysize() == 1;
    }

    size_t GetHash() const {
        TVecHash<int> intVectorHash;
        TVecHash<TFloatSplit> binFeatureHash;
        if (OneHotFeatures.empty()) {
            return MultiHash(intVectorHash(CatFeatures), binFeatureHash(BinFeatures));
        }
        TVecHash<TOneHotSplit> oneHotFeatureHash;
        return MultiHash(intVectorHash(CatFeatures), binFeatureHash(BinFeatures), oneHotFeatureHash(OneHotFeatures));
    }

    flatbuffers::Offset<NCatBoostFbs::TFeatureCombination> FBSerialize(TModelPartsCachingSerializer& serializer) const;

    void FBDeserialize(const NCatBoostFbs::TFeatureCombination* fbObj);

    /* make sure floating-point values do not contain negative zeros -
     * flatbuffers serializer will deserialize them as positive zeros
     */
    void Canonize() {
        for (auto& binFeature : BinFeatures) {
            binFeature.Canonize();
        }
    }
};

template <>
struct THash<TFeatureCombination> {
    inline size_t operator()(const TFeatureCombination& projection) const {
        return projection.GetHash();
    }
};

struct TModelCtrBaseMergeKey {
    TFeatureCombination Projection;
    ECtrType CtrType = ECtrType::Borders;

public:
    bool operator==(const TModelCtrBaseMergeKey& other) const {
        return std::tie(Projection, CtrType) ==
               std::tie(other.Projection, other.CtrType);
    }

    bool operator!=(const TModelCtrBaseMergeKey& other) const {
        return !(*this == other);
    }

    bool operator<(const TModelCtrBaseMergeKey& other) const {
        return std::tie(Projection, CtrType) <
               std::tie(other.Projection, other.CtrType);
    }

    Y_SAVELOAD_DEFINE(Projection, CtrType);

    size_t GetHash() const {
        return MultiHash(Projection.GetHash(), CtrType);
    }

    /* make sure floating-point values do not contain negative zeros -
     * flatbuffers serializer will deserialize them as positive zeros
     */
    void Canonize() {
        Projection.Canonize();
    }
};

template <>
struct THash<TModelCtrBaseMergeKey> {
    size_t operator()(const TModelCtrBaseMergeKey& ctr) const noexcept {
        return ctr.GetHash();
    }
};


struct TModelCtrBase : public TModelCtrBaseMergeKey {
    int TargetBorderClassifierIdx = 0;

public:
    bool operator==(const TModelCtrBase& other) const {
        return std::tie((const TModelCtrBaseMergeKey&)(*this), TargetBorderClassifierIdx) ==
               std::tie((const TModelCtrBaseMergeKey&)other, other.TargetBorderClassifierIdx);
    }

    bool operator!=(const TModelCtrBase& other) const {
        return !(*this == other);
    }

    bool operator<(const TModelCtrBase& other) const {
        return std::tie((const TModelCtrBaseMergeKey&)(*this), TargetBorderClassifierIdx) <
               std::tie((const TModelCtrBaseMergeKey&)other, other.TargetBorderClassifierIdx);
    }

    inline void Save(IOutputStream* s) const {
        ::SaveMany(s, (const TModelCtrBaseMergeKey&)(*this), TargetBorderClassifierIdx);
    }

    inline void Load(IInputStream* s) {
        ::LoadMany(s, (TModelCtrBaseMergeKey&)(*this), TargetBorderClassifierIdx);
    }

    size_t GetHash() const {
        return MultiHash(TModelCtrBaseMergeKey::GetHash(), TargetBorderClassifierIdx);
    }

    flatbuffers::Offset<NCatBoostFbs::TModelCtrBase> FBSerialize(TModelPartsCachingSerializer& serializer) const;
    void FBDeserialize(const NCatBoostFbs::TModelCtrBase* fbObj);

    /* make sure floating-point values do not contain negative zeros -
     * flatbuffers serializer will deserialize them as positive zeros
     */
    void Canonize() {
        TModelCtrBaseMergeKey::Canonize();
    }
};

template <>
struct THash<TModelCtrBase> {
    size_t operator()(const TModelCtrBase& ctr) const noexcept {
        return ctr.GetHash();
    }
};

struct TModelCtr{
    TModelCtrBase Base;
    int TargetBorderIdx = 0;
    float PriorNum = 0.0f;
    float PriorDenom = 1.0f;
    float Shift = 0.0f;
    float Scale = 1.0f;

public:
    TModelCtr() = default;

    bool operator==(const TModelCtr& other) const {
        return std::tie(Base, TargetBorderIdx, PriorNum, PriorDenom, Shift, Scale) ==
               std::tie(other.Base, other.TargetBorderIdx, other.PriorNum, other.PriorDenom, other.Shift, other.Scale);
    }

    bool operator!=(const TModelCtr& other) const {
        return !(*this == other);
    }

    bool operator<(const TModelCtr& other) const {
        return std::tie(Base, TargetBorderIdx, PriorNum, PriorDenom, Shift, Scale) <
               std::tie(other.Base, other.TargetBorderIdx, other.PriorNum, other.PriorDenom, other.Shift, other.Scale);
    }

    size_t GetHash() const {
        return MultiHash(Base.GetHash(), TargetBorderIdx,  PriorNum, PriorDenom, Shift, Scale);
    }

    inline float Calc(float countInClass, float totalCount) const {
        float ctr = (countInClass + PriorNum) / (totalCount + PriorDenom);
        return (ctr + Shift) * Scale;
    }

    inline void Save(IOutputStream* s) const {
        ::SaveMany(s, Base, TargetBorderIdx, PriorNum, PriorDenom, Shift, Scale);
    }

    inline void Load(IInputStream* s) {
        ::LoadMany(s, Base, TargetBorderIdx, PriorNum, PriorDenom, Shift, Scale);
    }

    flatbuffers::Offset<NCatBoostFbs::TModelCtr> FBSerialize(TModelPartsCachingSerializer& serializer) const;
    void FBDeserialize(const NCatBoostFbs::TModelCtr* fbObj);

    /* make sure floating-point values do not contain negative zeros -
     * flatbuffers serializer will deserialize them as positive zeros
     */
    void Canonize() {
        Base.Canonize();
        CanonizeFloatForFbs(&PriorNum);
        CanonizeFloatForFbs(&PriorDenom);
        CanonizeFloatForFbs(&Shift);
        CanonizeFloatForFbs(&Scale);
    }
};

template <>
struct THash<TModelCtr> {
    size_t operator()(const TModelCtr& ctr) const noexcept {
        return ctr.GetHash();
    }
};

struct TModelCtrSplit {
    TModelCtr Ctr;
    float Border = 0.0f;

public:
    TModelCtrSplit() = default;
    TModelCtrSplit(const TModelCtr& ctr, float border)
        : Ctr(ctr)
        , Border(border)
    {}

    bool operator==(const TModelCtrSplit& other) const {
        return std::tie(Ctr, Border) == std::tie(other.Ctr, other.Border);
    }

    bool operator!=(const TModelCtrSplit& other) const {
        return !(*this == other);
    }

    bool operator<(const TModelCtrSplit& other) const {
        return std::tie(Ctr, Border) < std::tie(other.Ctr, other.Border);
    }

    size_t GetHash() const {
        return MultiHash(Ctr, Border);
    }

    Y_SAVELOAD_DEFINE(Ctr, Border);

    /* make sure floating-point values do not contain negative zeros -
     * flatbuffers serializer will deserialize them as positive zeros
     */
    void Canonize() {
        Ctr.Canonize();
        CanonizeFloatForFbs(&Border);
    }
};

template <>
struct THash<TModelCtrSplit> {
    size_t operator()(const TModelCtrSplit& ctr) const noexcept {
        return ctr.GetHash();
    }
};

struct TCtrHistory {
    int N[2];

public:
    void Clear() {
        N[0] = 0;
        N[1] = 0;
    }
    Y_SAVELOAD_DEFINE(N[0], N[1]);
};

struct TCtrMeanHistory {
    float Sum;
    int Count;

public:
    bool operator==(const TCtrMeanHistory& other) const {
        return std::tie(Sum, Count) == std::tie(other.Sum, other.Count);
    }
    void Clear() {
        Sum = 0;
        Count = 0;
    }
    void Add(float target) {
        Sum += target;
        ++Count;
    }
    void Add(const TCtrMeanHistory& target) {
        Sum += target.Sum;
        Count += target.Count;
    }
    Y_SAVELOAD_DEFINE(Sum, Count);
};
