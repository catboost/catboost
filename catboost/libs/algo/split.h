#pragma once

#include "projection.h"

#include <library/binsaver/bin_saver.h>

struct TCtr {
    Y_SAVELOAD_DEFINE(Projection, CtrIdx, TargetBorderIdx, PriorIdx, BorderCount);
    SAVELOAD(Projection, CtrIdx, TargetBorderIdx, PriorIdx, BorderCount);

    TProjection Projection;
    ui8 CtrIdx = 0;
    ui8 TargetBorderIdx = 0;
    ui8 PriorIdx = 0;
    ui8 BorderCount = 0;

    TCtr() = default;

    bool operator==(const TCtr& other) const {
        return std::tie(Projection, CtrIdx, TargetBorderIdx, PriorIdx, BorderCount) ==
               std::tie(other.Projection, other.CtrIdx, other.TargetBorderIdx, other.PriorIdx, other.BorderCount);
    }

    bool operator!=(const TCtr& other) const {
        return !(*this == other);
    }

    TCtr(const TProjection& proj, ui8 ctrTypeIdx, ui8 targetBorderIdx, ui8 priorIdx, ui8 borderCount)
        : Projection(proj)
          , CtrIdx(ctrTypeIdx)
          , TargetBorderIdx(targetBorderIdx)
          , PriorIdx(priorIdx)
          , BorderCount(borderCount)
    {
    }

    size_t GetHash() const {
        return MultiHash(Projection.GetHash(), CtrIdx, TargetBorderIdx, PriorIdx, BorderCount);
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
    int FeatureIdx = -1;
    ESplitType Type = ESplitType::FloatFeature;

    Y_SAVELOAD_DEFINE(Ctr, FeatureIdx, Type);
    SAVELOAD(Ctr, FeatureIdx, Type);

    static const size_t FloatFeatureBaseHash;
    static const size_t CtrBaseHash;
    static const size_t OneHotFeatureBaseHash;

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

template <>
struct THash<TSplitCandidate> {
    inline size_t operator()(const TSplitCandidate& split) const {
        return split.GetHash();
    }
};

class TLearnContext;
class TDataset;

// TODO(kirillovs): this structure has doppelganger (TBinarySplit) in cuda code, merge them later
struct TSplit : public TSplitCandidate {
    TSplit(const TSplitCandidate& split, int border)
        : TSplitCandidate(split)
        , BinBorder(border)
    {}
    TSplit() = default;
    int BinBorder = 0;

    inline void Save(IOutputStream* s) const {
        ::SaveMany(s, static_cast<const TSplitCandidate&>(*this), BinBorder);
    }

    inline void Load(IInputStream* s) {
        ::LoadMany(s, static_cast<TSplitCandidate&>(*this), BinBorder);
    }
    TModelSplit GetModelSplit(const TLearnContext& ctx, const TDataset& learnData) const;

    static inline float EmulateUi8Rounding(int value) {
        return value + 0.999999f;
    }
    using TBase = TSplitCandidate;
    SAVELOAD_BASE(BinBorder);
};

struct TSplitTree {
    TVector<TSplit> Splits;

    void AddSplit(const TSplit& split) {
        Splits.push_back(split);
    }

    void DeleteSplit(int splitIdx) {
        Splits.erase(Splits.begin() + splitIdx);
    }

    inline int GetLeafCount() const {
        return 1 << Splits.ysize();
    }

    inline int GetDepth() const {
        return Splits.ysize();
    }
    TVector<TBinFeature> GetBinFeatures() const {
        TVector<TBinFeature> result;
        for (const auto& split : Splits) {
            if (split.Type == ESplitType::FloatFeature) {
                result.push_back(TBinFeature{split.FeatureIdx, split.BinBorder});
            }
        }
        return result;
    }

    TVector<TOneHotSplit> GetOneHotFeatures() const {
        TVector<TOneHotSplit> result;
        for (const auto& split : Splits) {
            if (split.Type == ESplitType::OneHotFeature) {
                result.push_back(TOneHotSplit{split.FeatureIdx, split.BinBorder});
            }
        }
        return result;
    }

    TVector<TCtr> GetCtrSplits() const {
        TVector<TCtr> result;
        for (const auto& split : Splits) {
            if (split.Type == ESplitType::OnlineCtr) {
                result.push_back(split.Ctr);
            }
        }
        return result;
    }

    Y_SAVELOAD_DEFINE(Splits)
    SAVELOAD(Splits);
};

struct TTreeStats {
    TVector<double> LeafWeightsSum;

    Y_SAVELOAD_DEFINE(LeafWeightsSum);
};
