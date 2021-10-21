#include "pairs.h"

#include <catboost/libs/helpers/equal.h>

#include <util/generic/variant.h>
#include <util/stream/output.h>


template <>
void Out<NCB::TPairInGroup>(IOutputStream& out, const NCB::TPairInGroup& pairInGroup) {
    out << "(GroupIdx=" << pairInGroup.GroupIdx
        << ",WinnerIdxInGroup=" << pairInGroup.WinnerIdxInGroup
        << ",LoserIdxInGroup=" << pairInGroup.LoserIdxInGroup
        << ",Weight=" << pairInGroup.Weight << ')';
}


namespace NCB {

    bool EqualWithoutOrder(const TRawPairsData& lhs, const TRawPairsData& rhs) {
        if (lhs.index() != rhs.index()) {
            return false;
        }
        if (const TFlatPairsInfo* lhsFlatPairs = std::get_if<TFlatPairsInfo>(&lhs)) {
            return EqualAsMultiSets<TPair>(*lhsFlatPairs, std::get<TFlatPairsInfo>(rhs));
        }
        return EqualAsMultiSets<TPairInGroup>(std::get<TGroupedPairsInfo>(lhs), std::get<TGroupedPairsInfo>(rhs));
    }

}

