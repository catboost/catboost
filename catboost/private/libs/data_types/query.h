#pragma once

#include <catboost/private/libs/index_range/index_range.h>

#include "groupid.h"
#include "pair.h"

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>
#include <util/system/yassert.h>

#include <library/cpp/binsaver/bin_saver.h>


using TGroupBounds = NCB::TIndexRange<ui32>;


struct TQueryInfo : public TGroupBounds {
    TQueryInfo() = default;

    TQueryInfo(const TGroupBounds& bounds)
        : TGroupBounds(bounds)
        , Weight(1.0f)
    {
    }

    TQueryInfo(ui32 begin, ui32 end)
        : TGroupBounds(begin, end)
        , Weight(1.0f)
    {
    }

    bool operator==(const TQueryInfo& rhs) const {
        if (!((TGroupBounds)(*this) == (TGroupBounds)rhs)) {
            return false;
        }
        return (Weight == rhs.Weight) && (SubgroupId == rhs.SubgroupId) && (Competitors == rhs.Competitors);
    }

    float Weight = 0.;
    TVector<ui32> SubgroupId; // can be empty if there's no subgroup data
    TVector<TVector<TCompetitor>> Competitors;
    SAVELOAD(Begin, End, Weight, SubgroupId, Competitors);
};

TVector<TGroupBounds> GroupSamples(TConstArrayRef<TGroupId> qid);
