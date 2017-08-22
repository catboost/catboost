#pragma once

#include "online_ctr.h"
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

enum class ESplitType {
    FloatFeature,
    OnlineCtr,
    OneHotFeature
};

struct TModelSplit {
    ESplitType Type;
    TBinFeature BinFeature;
    TModelCtrSplit OnlineCtr;
    TOneHotFeature OneHotFeature;

    Y_SAVELOAD_DEFINE(Type, BinFeature, OnlineCtr, OneHotFeature);

    TModelSplit() {
    }

    explicit TModelSplit(const TBinFeature& binFeature) {
        Type = ESplitType::FloatFeature;
        BinFeature = binFeature;
    }

    explicit TModelSplit(const TOneHotFeature& oheFeature) {
        Type = ESplitType::OneHotFeature;
        OneHotFeature = oheFeature;
    }

    explicit TModelSplit(const TModelCtrSplit& onlineCtr)
        : Type(ESplitType::OnlineCtr)
        , OnlineCtr(onlineCtr)
    {
    }

    TModelSplit(ESplitType splitType, int featureIdx, int splitIdxOrValue)
        : Type(splitType)
    {
        if (splitType == ESplitType::FloatFeature) {
            BinFeature = TBinFeature(featureIdx, splitIdxOrValue);
        } else {
            Y_ASSERT(splitType == ESplitType::OneHotFeature);
            OneHotFeature = TOneHotFeature(featureIdx, splitIdxOrValue);
        }
    }

    size_t GetHash() const {
        if (Type == ESplitType::FloatFeature) {
            return MultiHash(BinFeature.FloatFeature, BinFeature.SplitIdx);
        } else if (Type == ESplitType::OnlineCtr) {
            return OnlineCtr.GetHash();
        } else {
            Y_ASSERT(Type == ESplitType::OneHotFeature);
            return MultiHash(OneHotFeature.CatFeatureIdx, OneHotFeature.Value);
        }
    }

    bool operator==(const TModelSplit& other) const {
        return Type == other.Type && ((Type == ESplitType::FloatFeature && BinFeature == other.BinFeature) ||
                                      (Type == ESplitType::OnlineCtr && OnlineCtr == other.OnlineCtr) ||
                                      (Type == ESplitType::OneHotFeature && OneHotFeature == other.OneHotFeature));
    }

    bool operator<(const TModelSplit& other) const {
        if (Type < other.Type) {
            return true;
        }
        if (Type > other.Type) {
            return false;
        }
        if (Type == ESplitType::FloatFeature) {
            return BinFeature < other.BinFeature;
        } else if (Type == ESplitType::OnlineCtr) {
            return OnlineCtr < other.OnlineCtr;
        } else {
            Y_ASSERT(Type == ESplitType::OneHotFeature);
            return OneHotFeature < other.OneHotFeature;
        }
    }
};

template <>
struct THash<TModelSplit> {
    inline size_t operator()(const TModelSplit& split) const {
        return split.GetHash();
    }
};
