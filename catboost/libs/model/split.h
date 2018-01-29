#pragma once

#include "online_ctr.h"

#include <util/digest/multi.h>

#include <library/binsaver/bin_saver.h>

inline bool IsTrueHistogram(ui8 bucket, ui8 splitIdx) {
    return bucket > splitIdx;
}

inline bool IsTrueOneHotFeature(int featureValue, int splitValue) {
    return featureValue == splitValue;
}

enum class ESplitType {
    FloatFeature,
    OneHotFeature,
    OnlineCtr
};

struct TModelSplit {
    ESplitType Type;
    TFloatSplit FloatFeature;
    TModelCtrSplit OnlineCtr;
    TOneHotSplit OneHotFeature;

    Y_SAVELOAD_DEFINE(Type, FloatFeature, OnlineCtr, OneHotFeature);

    TModelSplit() = default;

    explicit TModelSplit(const TFloatSplit& floatFeature)
        : Type(ESplitType::FloatFeature)
        , FloatFeature(floatFeature)
    {
    }

    explicit TModelSplit(const TOneHotSplit& oheFeature)
        : Type(ESplitType::OneHotFeature)
        , OneHotFeature(oheFeature)
    {
    }

    explicit TModelSplit(const TModelCtrSplit& onlineCtr)
        : Type(ESplitType::OnlineCtr)
        , OnlineCtr(onlineCtr)
    {
    }

    size_t GetHash() const {
        if (Type == ESplitType::FloatFeature) {
            return FloatFeature.GetHash();
        } else if (Type == ESplitType::OnlineCtr) {
            return OnlineCtr.GetHash();
        } else {
            Y_ASSERT(Type == ESplitType::OneHotFeature);
            return OneHotFeature.GetHash();
        }
    }

    bool operator==(const TModelSplit& other) const {
        return Type == other.Type && ((Type == ESplitType::FloatFeature && FloatFeature == other.FloatFeature) ||
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
            return FloatFeature < other.FloatFeature;
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
