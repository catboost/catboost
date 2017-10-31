#pragma once

#include <catboost/libs/model/split.h>

struct TCtr {
    Y_SAVELOAD_DEFINE(Projection, CtrIdx, TargetBorderIdx, PriorIdx);

    TProjection Projection;
    ui8 CtrIdx = 0;
    ui8 TargetBorderIdx = 0;
    ui8 PriorIdx = 0;

    TCtr() = default;

    bool operator==(const TCtr& other) const {
        return std::tie(Projection, CtrIdx, TargetBorderIdx, PriorIdx) ==
               std::tie(other.Projection, other.CtrIdx, other.TargetBorderIdx, other.PriorIdx);
    }

    bool operator!=(const TCtr& other) const {
        return !(*this == other);
    }

    TCtr(const TProjection& proj, ui8 ctrTypeIdx, ui8 targetBorderIdx, ui8 priorIdx)
        : Projection(proj)
          , CtrIdx(ctrTypeIdx)
          , TargetBorderIdx(targetBorderIdx)
          , PriorIdx(priorIdx)
    {
    }

    size_t GetHash() const {
        return MultiHash(Projection.GetHash(), CtrIdx, TargetBorderIdx, PriorIdx);
    }
};

template<>
struct THash<TCtr> {
    size_t operator()(const TCtr& ctr) const noexcept {
        return ctr.GetHash();
    }
};

struct TSplitCandidate {
    TCtr Ctr;
    int FeatureIdx;
    ESplitType Type;

    static const size_t FloatFeatureBaseHash = 12321;
    static const size_t CtrBaseHash = 89321;
    static const size_t OneHotFeatureBaseHash = 517931;

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
        return Type == other.Type &&
               (((Type == ESplitType::FloatFeature || Type == ESplitType::OneHotFeature) && FeatureIdx == other.FeatureIdx)
                || (Type == ESplitType::OnlineCtr && Ctr == other.Ctr));
    }
};

struct TSplit : public TSplitCandidate {
    TSplit(const TSplitCandidate& split, int border)
        : TSplitCandidate(split)
        , BinBorder(border)
    {}
    int BinBorder = 0;
};
