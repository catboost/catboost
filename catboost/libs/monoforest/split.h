#pragma once

#include "enums.h"

#include <util/system/types.h>
#include <tuple>
#include <util/digest/multi.h>
#include <util/ysaveload.h>
#include <util/string/builder.h>

namespace NMonoForest {
    struct TBinarySplit {
        ui32 FeatureId = 0;
        ui32 BinIdx = 0;
        EBinSplitType SplitType;

        TBinarySplit(const ui32 featureId,
                     const ui32 binIdx,
                     EBinSplitType splitType)
            : FeatureId(featureId), BinIdx(binIdx), SplitType(splitType) {
        }

        TBinarySplit() = default;

        bool operator<(const TBinarySplit& other) const {
            return std::tie(FeatureId, BinIdx, SplitType) <
                   std::tie(other.FeatureId, other.BinIdx, other.SplitType);
        }

        bool operator<=(const TBinarySplit& other) const {
            return std::tie(FeatureId, BinIdx, SplitType) <=
                   std::tie(other.FeatureId, other.BinIdx, other.SplitType);
        }

        bool operator==(const TBinarySplit& other) const {
            return std::tie(FeatureId, BinIdx, SplitType) ==
                   std::tie(other.FeatureId, other.BinIdx, other.SplitType);
        }

        bool operator!=(const TBinarySplit& other) const {
            return !(*this == other);
        }

        ui64 GetHash() const {
            return MultiHash(FeatureId, BinIdx, SplitType);
        }

        Y_SAVELOAD_DEFINE(FeatureId, BinIdx, SplitType);
    };
}

template <>
struct THash<NMonoForest::TBinarySplit> {
    inline size_t operator()(const NMonoForest::TBinarySplit& value) const {
        return value.GetHash();
    }
};
