#pragma once

#include <catboost/private/libs/data_types/pair.h>

#include <util/digest/multi.h>
#include <util/generic/array_ref.h>
#include <util/generic/fwd.h>
#include <util/system/types.h>
#include <util/str_stl.h>
#include <util/ysaveload.h>

#include <tuple>


namespace NCB {

    struct TPairInGroup {
    public:
        ui32 GroupIdx;
        ui32 WinnerIdxInGroup;
        ui32 LoserIdxInGroup;
        float Weight = 1.0f;

    public:
        bool operator==(const TPairInGroup& other) const {
            return std::tie(GroupIdx, WinnerIdxInGroup, LoserIdxInGroup, Weight)
                == std::tie(other.GroupIdx, other.WinnerIdxInGroup, other.LoserIdxInGroup, other.Weight);
        }
        Y_SAVELOAD_DEFINE(GroupIdx, WinnerIdxInGroup, LoserIdxInGroup, Weight);
    };

}

template <>
struct THash<NCB::TPairInGroup> {
    inline size_t operator()(const NCB::TPairInGroup& pair) const {
        return MultiHash(pair.GroupIdx, pair.WinnerIdxInGroup, pair.LoserIdxInGroup, pair.Weight);
    }
};


namespace NCB {
    using TGroupedPairsInfo = TVector<TPairInGroup>;
    using TRawPairsData = std::variant<TFlatPairsInfo, TGroupedPairsInfo>;
    using TRawPairsDataRef = std::variant<TConstArrayRef<TPair>, TConstArrayRef<TPairInGroup>>;

    bool EqualWithoutOrder(const TRawPairsData& lhs, const TRawPairsData& rhs);
}
