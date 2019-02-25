#pragma once

#include "projection.h"

#include <catboost/libs/data_new/packed_binary_features.h>
#include <catboost/libs/data_new/quantized_features_info.h>
#include <catboost/libs/model/split.h>

#include <library/binsaver/bin_saver.h>

#include <util/digest/multi.h>
#include <util/digest/numeric.h>
#include <util/generic/array_ref.h>
#include <util/generic/vector.h>
#include <util/system/types.h>
#include <util/str_stl.h>
#include <util/ysaveload.h>

#include <limits>
#include <tuple>


struct TCtr {
    TProjection Projection;
    ui8 CtrIdx = 0;
    ui8 TargetBorderIdx = 0;
    ui8 PriorIdx = 0;
    ui8 BorderCount = 0;

public:
    TCtr() = default;

    TCtr(const TProjection& proj, ui8 ctrTypeIdx, ui8 targetBorderIdx, ui8 priorIdx, ui8 borderCount)
        : Projection(proj)
          , CtrIdx(ctrTypeIdx)
          , TargetBorderIdx(targetBorderIdx)
          , PriorIdx(priorIdx)
          , BorderCount(borderCount)
    {
    }

    bool operator==(const TCtr& other) const {
        return std::tie(Projection, CtrIdx, TargetBorderIdx, PriorIdx, BorderCount) ==
               std::tie(other.Projection, other.CtrIdx, other.TargetBorderIdx, other.PriorIdx, other.BorderCount);
    }

    bool operator!=(const TCtr& other) const {
        return !(*this == other);
    }

    SAVELOAD(Projection, CtrIdx, TargetBorderIdx, PriorIdx, BorderCount);
    Y_SAVELOAD_DEFINE(Projection, CtrIdx, TargetBorderIdx, PriorIdx, BorderCount);

    size_t GetHash() const {
        return MultiHash(Projection.GetHash(), CtrIdx, TargetBorderIdx, PriorIdx, BorderCount);
    }
};

template <>
struct THash<TCtr> {
    size_t operator()(const TCtr& ctr) const noexcept {
        return ctr.GetHash();
    }
};

struct TSplitCandidate {
    TCtr Ctr;
    int FeatureIdx = -1;
    ESplitType Type = ESplitType::FloatFeature;

    static const size_t FloatFeatureBaseHash;
    static const size_t CtrBaseHash;
    static const size_t OneHotFeatureBaseHash;

public:
    bool operator==(const TSplitCandidate& other) const {
        return Type == other.Type &&
           (((Type == ESplitType::FloatFeature || Type == ESplitType::OneHotFeature) &&
               FeatureIdx == other.FeatureIdx)
            || (Type == ESplitType::OnlineCtr && Ctr == other.Ctr));
    }

    SAVELOAD(Ctr, FeatureIdx, Type);
    Y_SAVELOAD_DEFINE(Ctr, FeatureIdx, Type);

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
};


template <>
struct THash<TSplitCandidate> {
    inline size_t operator()(const TSplitCandidate& split) const {
        return split.GetHash();
    }
};


struct TBinarySplitsPack {
    ui32 PackIdx = std::numeric_limits<ui32>::max();

public:
    bool operator==(const TBinarySplitsPack& other) const {
        return PackIdx == other.PackIdx;
    }
};


// could have been a TVariant but SAVELOAD is easier this way
struct TSplitEnsemble {
    // variant switch
    bool IsBinarySplitsPack;

    // variant members
    TSplitCandidate SplitCandidate;
    TBinarySplitsPack BinarySplitsPack;

    static constexpr size_t BinarySplitsPackHash = 118223;

public:
    TSplitEnsemble()
        : IsBinarySplitsPack(false)
    {}

    explicit TSplitEnsemble(TSplitCandidate&& splitCandidate)
        : IsBinarySplitsPack(false)
        , SplitCandidate(std::move(splitCandidate))
    {}

    /* move is not really needed for such a simple structure but do it in the same way as splitCandidate for
     * consistency
     */
    explicit TSplitEnsemble(TBinarySplitsPack&& binarySplitsPack)
        : IsBinarySplitsPack(true)
        , BinarySplitsPack(std::move(binarySplitsPack))
    {}

    bool operator==(const TSplitEnsemble& other) const {
        if (IsBinarySplitsPack) {
            return other.IsBinarySplitsPack && (BinarySplitsPack == other.BinarySplitsPack);
        }
        return !other.IsBinarySplitsPack && (SplitCandidate == other.SplitCandidate);
    }

    SAVELOAD(IsBinarySplitsPack, SplitCandidate, BinarySplitsPack);

    size_t GetHash() const {
        if (IsBinarySplitsPack) {
            return MultiHash(BinarySplitsPackHash, BinarySplitsPack.PackIdx);
        }
        return SplitCandidate.GetHash();
    }

    bool IsSplitOfType(ESplitType type) const {
        return !IsBinarySplitsPack && (SplitCandidate.Type == type);
    }
};

template <>
struct THash<TSplitEnsemble> {
    inline size_t operator()(const TSplitEnsemble& splitEnsemble) const {
        return splitEnsemble.GetHash();
    }
};


struct TSplitEnsembleSpec {
    bool IsBinarySplitsPack;
    ESplitType SplitType; // not used if IsBinarySplitsPack == true

public:
    explicit TSplitEnsembleSpec(
        bool isBinarySplitsPack = false,
        ESplitType splitType = ESplitType::FloatFeature
    )
        : IsBinarySplitsPack(isBinarySplitsPack)
        , SplitType(splitType)
    {}

    explicit TSplitEnsembleSpec(const TSplitEnsemble& splitEnsemble)
        : IsBinarySplitsPack(splitEnsemble.IsBinarySplitsPack)
        , SplitType(splitEnsemble.SplitCandidate.Type)
    {}

    SAVELOAD(IsBinarySplitsPack, SplitType);

    bool operator==(const TSplitEnsembleSpec& other) const {
        if (IsBinarySplitsPack) {
            return other.IsBinarySplitsPack;
        }
        return !other.IsBinarySplitsPack && (SplitType == other.SplitType);
    }

    static TSplitEnsembleSpec OneSplit(ESplitType splitType) {
        return TSplitEnsembleSpec(false, splitType);
    }

    static TSplitEnsembleSpec BinarySplitsPack() {
        return TSplitEnsembleSpec(true, ESplitType::FloatFeature);
    }
};


int GetBucketCount(
    const TSplitEnsemble& splitEnsemble,
    const NCB::TQuantizedFeaturesInfo& quantizedFeaturesInfo,
    size_t packedBinaryFeaturesCount
);


class TLearnContext;

// TODO(kirillovs): this structure has doppelganger (TBinarySplit) in cuda code, merge them later
struct TSplit : public TSplitCandidate {
public:
    using TBase = TSplitCandidate;

public:
    int BinBorder = 0;

public:
    TSplit() = default;

    TSplit(const TSplitCandidate& split, int border)
        : TSplitCandidate(split)
        , BinBorder(border)
    {}

    SAVELOAD_BASE(BinBorder);

    inline void Save(IOutputStream* s) const {
        ::SaveMany(s, static_cast<const TSplitCandidate&>(*this), BinBorder);
    }

    inline void Load(IInputStream* s) {
        ::LoadMany(s, static_cast<TSplitCandidate&>(*this), BinBorder);
    }

    TModelSplit GetModelSplit(
        const TLearnContext& ctx,
        const NCB::TPerfectHashedToHashedCatValuesMap& perfectHashedToHashedCatValuesMap) const;

    static inline float EmulateUi8Rounding(int value) {
        return value + 0.999999f;
    }
};

struct TSplitTree {
    TVector<TSplit> Splits;

public:
    SAVELOAD(Splits);
    Y_SAVELOAD_DEFINE(Splits)

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
};

struct TTreeStats {
    TVector<double> LeafWeightsSum;

public:
    Y_SAVELOAD_DEFINE(LeafWeightsSum);
};
