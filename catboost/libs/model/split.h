#pragma once

#include <catboost/libs/model/projection.h>

#include <util/digest/multi.h>
#include <library/binsaver/bin_saver.h>

template <typename T>
inline bool IsTrueFeature(T value, T border) {
    return value > border;
}

inline bool IsTrueHistogram(ui8 bucket, ui8 splitIdx) {
    return bucket > splitIdx;
}

inline bool IsTrueOneHotFeature(int featureValue, int splitValue) {
    return featureValue == splitValue;
}

struct TCtr {
    Y_SAVELOAD_DEFINE(Projection, CtrTypeIdx, TargetBorderIdx, PriorIdx);

    TProjection Projection;
    ui8 CtrTypeIdx = 0;
    ui8 TargetBorderIdx = 0;
    ui8 PriorIdx = 0;

    TCtr() = default;

    bool operator==(const TCtr& other) const {
        return std::tie(Projection, CtrTypeIdx, TargetBorderIdx, PriorIdx) ==
            std::tie(other.Projection, other.CtrTypeIdx, other.TargetBorderIdx, other.PriorIdx);
    }

    bool operator!=(const TCtr& other) const {
        return !(*this == other);
    }

    TCtr(const TProjection& proj, ui8 ctrTypeIdx, ui8 targetBorderIdx, ui8 priorIdx)
        : Projection(proj)
        , CtrTypeIdx(ctrTypeIdx)
        , TargetBorderIdx(targetBorderIdx)
        , PriorIdx(priorIdx)
    {
    }

    size_t GetHash() const {
        return MultiHash(Projection.GetHash(), CtrTypeIdx, TargetBorderIdx, PriorIdx);
    }
    bool operator<(const TCtr& other) const {
        return std::tie(Projection, CtrTypeIdx, TargetBorderIdx, PriorIdx) < std::tie(other.Projection, other.CtrTypeIdx, other.TargetBorderIdx, other.PriorIdx);
    }
};

// TODO(annaveronika): Merge split type + feature type
enum class ESplitType {
    FloatFeature,
    OnlineCtr,
    OneHotFeature
};

struct TCtrSplit {
    TCtr Ctr;
    ui8 Border;

    Y_SAVELOAD_DEFINE(Ctr, Border)

    TCtrSplit()
        : Border(0)
    {
    }

    TCtrSplit(const TCtr& ctr, ui8 border)
        : Ctr(ctr)
        , Border(border)
    {
    }

    bool operator==(const TCtrSplit& other) const {
        return Ctr == other.Ctr && Border == other.Border;
    }

    bool operator<(const TCtrSplit& other) const {
        return std::tie(Ctr, Border) < std::tie(other.Ctr, other.Border);
    }
};

struct TSplitCandidate {
    TCtr Ctr;
    int FeatureIdx;
    ESplitType Type;

    const size_t FloatFeatureBaseHash = 12321;
    const size_t CtrBaseHash = 89321;
    const size_t OneHotFeatureBaseHash = 517931;

    size_t GetHash() const {
        if (Type == ESplitType::FloatFeature) {
            return MultiHash(FloatFeatureBaseHash, FeatureIdx);
        } else if (Type == ESplitType::OnlineCtr) {
            return MultiHash(CtrBaseHash, Ctr.GetHash());
        } else {
            Y_ASSERT(Type == ESplitType::OneHotFeature);
            return MultiHash(OneHotFeatureBaseHash, FeatureIdx);
        }
    }

    bool operator==(const TSplitCandidate& other) const {
        return Type == other.Type && ((Type == ESplitType::FloatFeature || Type == ESplitType::OneHotFeature) &&
                                      FeatureIdx == other.FeatureIdx ||
                                      Type == ESplitType::OnlineCtr && Ctr == other.Ctr);
    }
};

struct TSplit {
    ESplitType Type;
    TBinFeature BinFeature;
    TCtrSplit OnlineCtr;
    TOneHotFeature OneHotFeature;

    // TODO(annaveronika): change TSplit class to storing
    // TSplitCandidate+splitIdx
    void BuildTFeatureFormat(TSplitCandidate* feature, int* splitIdxOrValue) const {
        feature->Type = Type;
        if (Type == ESplitType::FloatFeature) {
            (*splitIdxOrValue) = BinFeature.SplitIdx;
            feature->FeatureIdx = BinFeature.FloatFeature;
        } else if (Type == ESplitType::OnlineCtr) {
            (*splitIdxOrValue) = OnlineCtr.Border;
            feature->Ctr = OnlineCtr.Ctr;
        } else {
            Y_ASSERT(Type == ESplitType::OneHotFeature);
            (*splitIdxOrValue) = OneHotFeature.Value;
            feature->FeatureIdx = OneHotFeature.CatFeatureIdx;
        }
    }

    Y_SAVELOAD_DEFINE(Type, BinFeature, OnlineCtr, OneHotFeature);

    TSplit() {
    }

    TSplit(ESplitType splitType, int featureIdx, int splitIdxOrValue)
        : Type(splitType)
    {
        if (splitType == ESplitType::FloatFeature) {
            BinFeature = TBinFeature(featureIdx, splitIdxOrValue);
        } else {
            Y_ASSERT(splitType == ESplitType::OneHotFeature);
            OneHotFeature = TOneHotFeature(featureIdx, splitIdxOrValue);
        }
    }

    explicit TSplit(const TCtrSplit& onlineCtr)
        : Type(ESplitType::OnlineCtr)
        , OnlineCtr(onlineCtr)
    {
    }

    size_t GetHash() const {
        if (Type == ESplitType::FloatFeature) {
            return MultiHash(BinFeature.FloatFeature, BinFeature.SplitIdx);
        } else if (Type == ESplitType::OnlineCtr)  {
            TProjHash projHash;
            return MultiHash(projHash(OnlineCtr.Ctr.Projection), OnlineCtr.Border, OnlineCtr.Ctr.PriorIdx);
        } else {
            Y_ASSERT(Type == ESplitType::OneHotFeature);
            return MultiHash(OneHotFeature.CatFeatureIdx, OneHotFeature.Value);
        }
    }

    bool operator==(const TSplit& other) const {
        return Type == other.Type && (Type == ESplitType::FloatFeature && BinFeature == other.BinFeature ||
                                      Type == ESplitType::OnlineCtr && OnlineCtr == other.OnlineCtr ||
                                      Type == ESplitType::OneHotFeature && OneHotFeature == other.OneHotFeature);
    }

    bool operator<(const TSplit& other) const {
        if (Type < other.Type) {
            return true;
        } if (Type > other.Type) {
            return false;
        }
        if (Type == ESplitType::FloatFeature) {
            return BinFeature < other.BinFeature;
        } else if (Type == ESplitType::OnlineCtr)  {
            return OnlineCtr < other.OnlineCtr;
        } else {
            Y_ASSERT(Type == ESplitType::OneHotFeature);
            return OneHotFeature < other.OneHotFeature;
        }
    }
};

struct TSplitHash {
    inline size_t operator()(const TSplit& split) const {
        return split.GetHash();
    }
};
