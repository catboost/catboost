#pragma once

#include "estimated_features.h"
#include "projection.h"

#include <catboost/libs/data/exclusive_feature_bundling.h>
#include <catboost/libs/data/feature_estimators.h>
#include <catboost/libs/data/feature_grouping.h>
#include <catboost/libs/data/packed_binary_features.h>
#include <catboost/libs/data/quantized_features_info.h>
#include <catboost/libs/model/split.h>

#include <library/cpp/binsaver/bin_saver.h>

#include <util/digest/multi.h>
#include <util/digest/numeric.h>
#include <util/generic/array_ref.h>
#include <util/generic/cast.h>
#include <util/generic/vector.h>
#include <util/system/compiler.h>
#include <util/system/types.h>
#include <util/str_stl.h>
#include <util/ysaveload.h>

#include <limits>
#include <tuple>


class TLearnContext;


namespace NCB {
    struct TQuantizedEstimatedFeaturesInfo;
}


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
    int FeatureIdx = -1;  // not valid if Type == ESplitType::OnlineCtr
    bool IsOnlineEstimatedFeature = false; // valid only if Type == ESplitType::EstimatedFeature
    ESplitType Type = ESplitType::FloatFeature;

    static const size_t FloatFeatureBaseHash;
    static const size_t CtrBaseHash;
    static const size_t OneHotFeatureBaseHash;
    static const size_t EstimatedFeatureBaseHash;

public:
    bool operator==(const TSplitCandidate& other) const {
        if (Type != other.Type) {
            return false;
        }
        switch (Type) {
            case ESplitType::FloatFeature:
            case ESplitType::OneHotFeature:
                return FeatureIdx == other.FeatureIdx;
            case ESplitType::EstimatedFeature:
                return (IsOnlineEstimatedFeature == other.IsOnlineEstimatedFeature) &&
                    (FeatureIdx == other.FeatureIdx);
            case ESplitType::OnlineCtr:
                return Ctr == other.Ctr;
        }
        Y_UNREACHABLE();
    }

    SAVELOAD(Ctr, FeatureIdx, IsOnlineEstimatedFeature, Type);
    Y_SAVELOAD_DEFINE(Ctr, FeatureIdx, IsOnlineEstimatedFeature, Type);

    size_t GetHash() const {
        if (Type == ESplitType::FloatFeature) {
            return MultiHash(FloatFeatureBaseHash, FeatureIdx);
        } else if (Type == ESplitType::OnlineCtr) {
            return MultiHash(CtrBaseHash, Ctr.GetHash());
        } else if (Type == ESplitType::OneHotFeature) {
            return MultiHash(OneHotFeatureBaseHash, FeatureIdx);
        } else {
            Y_ASSERT(Type == ESplitType::EstimatedFeature);
            return MultiHash(EstimatedFeatureBaseHash, IsOnlineEstimatedFeature, FeatureIdx);
        }
    }

    template <class Function>
    // Function must get two params - internal feature idx (int) and EFeatureType
    void IterateOverUsedFeatures(
        const NCB::TCombinedEstimatedFeaturesContext& estimatedFeaturesContext,
        Function&& f
    ) const {
        if (Type == ESplitType::FloatFeature) {
            Y_ASSERT(FeatureIdx != -1);
            f(FeatureIdx, EFeatureType::Float);
        } else if (Type == ESplitType::OneHotFeature) {
            Y_ASSERT(FeatureIdx != -1);
            f(FeatureIdx, EFeatureType::Categorical);
        } else if (Type == ESplitType::EstimatedFeature) {
            Y_ASSERT(FeatureIdx != -1);

            const NCB::TEstimatorId estimatorId = (
                IsOnlineEstimatedFeature ?
                    estimatedFeaturesContext.OnlineEstimatedFeaturesLayout
                    : estimatedFeaturesContext.OfflineEstimatedFeaturesLayout
            )[FeatureIdx].EstimatorId;

            f(
                SafeIntegerCast<int>(
                    estimatedFeaturesContext.FeatureEstimators->GetEstimatorSourceFeatureIdx(
                        estimatorId
                    ).TextFeatureId
                ),
                estimatedFeaturesContext.FeatureEstimators->GetFeatureEstimator(estimatorId)->GetSourceType()
            );
        } else if (Type == ESplitType::OnlineCtr) {
            for (const int catFeatureIdx : Ctr.Projection.CatFeatures) {
                f(catFeatureIdx, EFeatureType::Categorical);
            }
            for (const auto& binFeature : Ctr.Projection.BinFeatures) {
                f(binFeature.FloatFeature, EFeatureType::Float);
            }
            for (const auto& oneHotFeature : Ctr.Projection.OneHotFeatures) {
                f(oneHotFeature.CatFeatureIdx, EFeatureType::Categorical);
            }
        } else {
            ythrow TCatBoostException() << "Unknown feature type" << Type;
        }
    }

    bool IsOnline() const {
        return (Type == ESplitType::OnlineCtr) || IsOnlineEstimatedFeature;
    }
};


template <>
struct THash<TSplitCandidate> {
    inline size_t operator()(const TSplitCandidate& split) const {
        return split.GetHash();
    }
};


struct TBinarySplitsPackRef {
    ui32 PackIdx = std::numeric_limits<ui32>::max();

public:
    bool operator==(const TBinarySplitsPackRef& other) const {
        return PackIdx == other.PackIdx;
    }
};


struct TExclusiveFeaturesBundleRef {
    ui32 BundleIdx = std::numeric_limits<ui32>::max();

public:
    bool operator==(const TExclusiveFeaturesBundleRef& other) const {
        return BundleIdx == other.BundleIdx;
    }
};

struct TFeaturesGroupRef {
    ui32 GroupIdx = std::numeric_limits<ui32>::max();

public:
    bool operator==(const TFeaturesGroupRef& other) const {
        return GroupIdx == other.GroupIdx;
    }
};


enum class ESplitEnsembleType {
    OneFeature,
    BinarySplits,
    ExclusiveBundle,
    FeaturesGroup
};


// could have been a std::variant but SAVELOAD is easier this way
struct TSplitEnsemble {
    bool IsEstimated;  // either online or offline
    bool IsOnlineEstimated;

    // variant switch
    ESplitEnsembleType Type;

    // variant members
    TSplitCandidate SplitCandidate;
    TBinarySplitsPackRef BinarySplitsPackRef;
    TExclusiveFeaturesBundleRef ExclusiveFeaturesBundleRef;
    TFeaturesGroupRef FeaturesGroupRef;

    static constexpr size_t BinarySplitsPackHash = 118223;
    static constexpr size_t ExclusiveBundleHash = 981490;
    static constexpr size_t FeaturesGroupHash = 735019;

public:
    TSplitEnsemble()
        : IsEstimated(false)
        , IsOnlineEstimated(false)
        , Type(ESplitEnsembleType::OneFeature)
    {}

    explicit TSplitEnsemble(
        TSplitCandidate&& splitCandidate,
        bool isEstimated = false,
        bool isOnlineEstimated = false
    )
        : IsEstimated(isEstimated)
        , IsOnlineEstimated(isOnlineEstimated)
        , Type(ESplitEnsembleType::OneFeature)
        , SplitCandidate(std::move(splitCandidate))
    {}

    /* move is not really needed for such a simple structure but do it in the same way as splitCandidate for
     * consistency
     */
    explicit TSplitEnsemble(
        TBinarySplitsPackRef&& binarySplitsPackRef,
        bool isEstimated = false,
        bool isOnlineEstimated = false
    )
        : IsEstimated(isEstimated)
        , IsOnlineEstimated(isOnlineEstimated)
        , Type(ESplitEnsembleType::BinarySplits)
        , BinarySplitsPackRef(std::move(binarySplitsPackRef))
    {}

    /* move is not really needed for such a simple structure but do it in the same way as splitCandidate for
     * consistency
     */
    explicit TSplitEnsemble(TExclusiveFeaturesBundleRef&& exclusiveFeaturesBundleRef)
        : IsEstimated(false)
        , IsOnlineEstimated(false)
        , Type(ESplitEnsembleType::ExclusiveBundle)
        , ExclusiveFeaturesBundleRef(std::move(exclusiveFeaturesBundleRef))
    {}

    /* move is not really needed for such a simple structure but do it in the same way as splitCandidate for
     * consistency
     */
    explicit TSplitEnsemble(TFeaturesGroupRef&& featuresGroupRef)
        : IsEstimated(false)
        , IsOnlineEstimated(false)
        , Type(ESplitEnsembleType::FeaturesGroup)
        , FeaturesGroupRef(std::move(featuresGroupRef))
    {}

    bool operator==(const TSplitEnsemble& other) const {
        if (IsEstimated != other.IsEstimated) {
            return false;
        }
        if (IsEstimated && (IsOnlineEstimated != other.IsOnlineEstimated)) {
            return false;
        }

        switch (Type) {
            case ESplitEnsembleType::OneFeature:
                return (other.Type == ESplitEnsembleType::OneFeature)
                    && (SplitCandidate == other.SplitCandidate);
            case ESplitEnsembleType::BinarySplits:
                return (other.Type == ESplitEnsembleType::BinarySplits) &&
                    (BinarySplitsPackRef == other.BinarySplitsPackRef);
            case ESplitEnsembleType::ExclusiveBundle:
                return (other.Type == ESplitEnsembleType::ExclusiveBundle) &&
                    (ExclusiveFeaturesBundleRef == other.ExclusiveFeaturesBundleRef);
            case ESplitEnsembleType::FeaturesGroup:
                return (other.Type == ESplitEnsembleType::FeaturesGroup) &&
                       (FeaturesGroupRef == other.FeaturesGroupRef);
        }
        Y_UNREACHABLE();
    }

    SAVELOAD(
        IsEstimated,
        IsOnlineEstimated,
        Type,
        SplitCandidate,
        BinarySplitsPackRef,
        ExclusiveFeaturesBundleRef,
        FeaturesGroupRef);

    size_t GetHash() const {
        switch (Type) {
            case ESplitEnsembleType::OneFeature:
                return MultiHash(IsEstimated, IsOnlineEstimated, SplitCandidate.GetHash());
            case ESplitEnsembleType::BinarySplits:
                return MultiHash(
                    IsEstimated,
                    IsOnlineEstimated,
                    BinarySplitsPackHash,
                    BinarySplitsPackRef.PackIdx);
            case ESplitEnsembleType::ExclusiveBundle:
                return MultiHash(
                    IsEstimated,
                    IsOnlineEstimated,
                    ExclusiveBundleHash,
                    ExclusiveFeaturesBundleRef.BundleIdx);
            case ESplitEnsembleType::FeaturesGroup:
                return MultiHash(IsEstimated, IsOnlineEstimated, FeaturesGroupHash, FeaturesGroupRef.GroupIdx);
        }
        Y_UNREACHABLE();
    }

    bool IsSplitOfType(ESplitType type) const {
        return (Type == ESplitEnsembleType::OneFeature) && (SplitCandidate.Type == type);
    }
};

template <>
struct THash<TSplitEnsemble> {
    inline size_t operator()(const TSplitEnsemble& splitEnsemble) const {
        return splitEnsemble.GetHash();
    }
};


struct TSplitEnsembleSpec {
    bool IsEstimated; // either online or offline
    bool IsOnlineEstimated;

    ESplitEnsembleType Type;

    ESplitType OneSplitType; // used only if Type == OneFeature
    NCB::TExclusiveFeaturesBundle ExclusiveFeaturesBundle; // used only if Type == ExclusiveBundle
    NCB::TFeaturesGroup FeaturesGroup; // used only if Type == FeaturesGroup

public:
    explicit TSplitEnsembleSpec(
        ESplitEnsembleType type = ESplitEnsembleType::OneFeature,
        ESplitType oneSplitType = ESplitType::FloatFeature,
        NCB::TExclusiveFeaturesBundle exclusiveFeaturesBundle = NCB::TExclusiveFeaturesBundle(),
        NCB::TFeaturesGroup featuresGroup = NCB::TFeaturesGroup(),
        bool isEstimated = false,
        bool isOnlineEstimated = false
    )
        : IsEstimated(isEstimated)
        , IsOnlineEstimated(isOnlineEstimated)
        , Type(type)
        , OneSplitType(oneSplitType)
        , ExclusiveFeaturesBundle(std::move(exclusiveFeaturesBundle))
        , FeaturesGroup(std::move(featuresGroup))
    {}

    TSplitEnsembleSpec(
        const TSplitEnsemble& splitEnsemble,
        TConstArrayRef<NCB::TExclusiveFeaturesBundle> exclusiveFeaturesBundles,
        TConstArrayRef<NCB::TFeaturesGroup> featuresGroups
    )
        : IsEstimated(splitEnsemble.IsEstimated)
        , IsOnlineEstimated(splitEnsemble.IsOnlineEstimated)
        , Type(splitEnsemble.Type)
        , OneSplitType(splitEnsemble.SplitCandidate.Type)
    {
        if (Type == ESplitEnsembleType::ExclusiveBundle) {
            ExclusiveFeaturesBundle
                = exclusiveFeaturesBundles[splitEnsemble.ExclusiveFeaturesBundleRef.BundleIdx];
        }
        if (Type == ESplitEnsembleType::FeaturesGroup) {
            FeaturesGroup = featuresGroups[splitEnsemble.FeaturesGroupRef.GroupIdx];
        }
    }

    SAVELOAD(IsEstimated, IsOnlineEstimated, Type, OneSplitType, ExclusiveFeaturesBundle, FeaturesGroup);

    bool operator==(const TSplitEnsembleSpec& other) const {
        if (IsEstimated != other.IsEstimated) {
            return false;
        }
        if (IsEstimated && (IsOnlineEstimated != other.IsOnlineEstimated)) {
            return false;
        }

        switch (Type) {
            case ESplitEnsembleType::OneFeature:
                return (other.Type == ESplitEnsembleType::OneFeature) && (OneSplitType == other.OneSplitType);
            case ESplitEnsembleType::BinarySplits:
                return other.Type == ESplitEnsembleType::BinarySplits;
            case ESplitEnsembleType::ExclusiveBundle:
                return (other.Type == ESplitEnsembleType::ExclusiveBundle) &&
                    (ExclusiveFeaturesBundle == other.ExclusiveFeaturesBundle);
            case ESplitEnsembleType::FeaturesGroup:
                return (other.Type == ESplitEnsembleType::FeaturesGroup) &&
                    (FeaturesGroup == other.FeaturesGroup);
        }
        Y_UNREACHABLE();
    }

    static TSplitEnsembleSpec OneSplit(ESplitType splitType) {
        return TSplitEnsembleSpec(ESplitEnsembleType::OneFeature, splitType);
    }

    static TSplitEnsembleSpec BinarySplitsPack() {
        return TSplitEnsembleSpec(ESplitEnsembleType::BinarySplits);
    }

    static TSplitEnsembleSpec ExclusiveFeatureBundle(
        const NCB::TExclusiveFeaturesBundle& exclusiveFeaturesBundle
    ) {
        return TSplitEnsembleSpec(
            ESplitEnsembleType::ExclusiveBundle,
            ESplitType::FloatFeature, // dummy value
            exclusiveFeaturesBundle
        );
    }

    static TSplitEnsembleSpec FeatureGroup(
        const NCB::TFeaturesGroup& featuresGroup
    ) {
        return TSplitEnsembleSpec(
            ESplitEnsembleType::FeaturesGroup,
            ESplitType::FloatFeature, // dummy value
            {}, // dummy value
            featuresGroup
        );
    }
};


int GetBucketCount(
    const TSplitEnsemble& splitEnsemble,
    const NCB::TQuantizedFeaturesInfo& quantizedFeaturesInfo,
    size_t packedBinaryFeaturesCount,
    TConstArrayRef<NCB::TExclusiveFeaturesBundle> exclusiveFeaturesBundles,
    TConstArrayRef<NCB::TFeaturesGroup> featuresGroups
);


inline bool UseForCalcScores(const NCB::TExclusiveBundlePart& exclusiveBundlePart, ui32 oneHotMaxSize) {
    if (exclusiveBundlePart.FeatureType == EFeatureType::Categorical) {
        return (exclusiveBundlePart.Bounds.GetSize() + 1) <= oneHotMaxSize;
    }
    return true;
}


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
        const NCB::TPerfectHashedToHashedCatValuesMap& perfectHashedToHashedCatValuesMap,
        const NCB::TFeatureEstimators& featureEstimators,
        const NCB::TQuantizedEstimatedFeaturesInfo& offlineEstimatedFeaturesInfo,
        const NCB::TQuantizedEstimatedFeaturesInfo& onlineEstimatedFeaturesInfo
    ) const;

    static inline float EmulateUi8Rounding(int value) {
        return value + 0.999999f;
    }
};

struct TSplitTree {
    TVector<TSplit> Splits;

public:
    SAVELOAD(Splits);
    Y_SAVELOAD_DEFINE(Splits);

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

    TVector<TCtr> GetUsedCtrs() const {
        TVector<TCtr> result;
        for (const auto& split : Splits) {
            if (split.Type == ESplitType::OnlineCtr) {
                result.push_back(split.Ctr);
            }
        }
        return result;
    }
};

/*
 * TSplitNode is node in non-symmetric tree
 * It is used in model training (in other places non-symmetric tree can have another layout).
 * Here negative values of Left and Right means leaves, non-negative values - split nodes.
 * To get leaf index from negative value of Left or Right, we use ~ operator.
 */
struct TSplitNode {
    TSplit Split;
    int Left = -1;
    int Right = -1;

    TSplitNode() = default;
    TSplitNode(const TSplit& split, int left, int right)
        : Split(split), Left(left), Right(right)
    {
    }

    SAVELOAD(Split, Left, Right);
    Y_SAVELOAD_DEFINE(Split, Left, Right);
};

struct TNonSymmetricTreeStructure {
    TVector<TSplitNode> Nodes;
    TVector<int> LeafParent;

public:
    SAVELOAD(Nodes, LeafParent);
    Y_SAVELOAD_DEFINE(Nodes, LeafParent);

    TNonSymmetricTreeStructure()
        : LeafParent((size_t)1, -1)
    {
    }

    int GetRoot() const {
        return Nodes.empty() ? -1 : 0;
    }

    TConstArrayRef<TSplitNode> GetNodes() const {
        return Nodes;
    }

    const TSplitNode& AddSplit(const TSplit& split, int leafIdx) {
        Y_ASSERT(0 <= leafIdx && leafIdx < SafeIntegerCast<int>(GetLeafCount()));
        int newLeafIdx = GetLeafCount();
        int newNodeIdx = Nodes.size();
        int parent = LeafParent[leafIdx];
        if (parent >= 0) {
            if (Nodes[parent].Left == ~leafIdx) {
                Nodes[parent].Left = newNodeIdx;
            } else {
                Nodes[parent].Right = newNodeIdx;
            }
        }
        Nodes.emplace_back(split, ~leafIdx, ~newLeafIdx);
        LeafParent[leafIdx] = newNodeIdx;
        LeafParent.emplace_back(newNodeIdx);
        return Nodes.back();
    }

    inline ui32 GetNodesCount() const {
        return Nodes.ysize();
    }

    inline ui32 GetLeafCount() const {
        return Nodes.ysize() + 1;
    }

    TVector<TSplit> GetSplits() const {
        TVector<TSplit> splits;
        splits.reserve(Nodes.size());
        for (const auto& node : Nodes) {
            splits.push_back(node.Split);
        }
        return splits;
    }

    TVector<TCtr> GetUsedCtrs() const {
        TVector<TCtr> result;
        for (const auto& node : Nodes) {
            if (node.Split.Type == ESplitType::OnlineCtr) {
                result.push_back(node.Split.Ctr);
            }
        }
        return result;
    }
};

inline TVector<TCtr> GetUsedCtrs(const std::variant<TSplitTree, TNonSymmetricTreeStructure>& tree) {
    if (std::holds_alternative<TSplitTree>(tree)) {
        return std::get<TSplitTree>(tree).GetUsedCtrs();
    } else {
        return std::get<TNonSymmetricTreeStructure>(tree).GetUsedCtrs();
    }
}

inline int GetLeafCount(const std::variant<TSplitTree, TNonSymmetricTreeStructure>& tree) {
    if (std::holds_alternative<TSplitTree>(tree)) {
        return std::get<TSplitTree>(tree).GetLeafCount();
    } else {
        return std::get<TNonSymmetricTreeStructure>(tree).GetLeafCount();
    }
}

struct TTreeStats {
    TVector<double> LeafWeightsSum;

public:
    Y_SAVELOAD_DEFINE(LeafWeightsSum);
};
