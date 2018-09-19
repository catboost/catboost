#pragma once

#include "pair.h"

#include <util/generic/vector.h>
#include <util/system/yassert.h>

#include <library/binsaver/bin_saver.h>

struct TQueryInfo {
    TQueryInfo() = default;
    TQueryInfo(ui32 begin, ui32 end)
        : Begin(begin)
        , End(end)
        , Weight(1.0f)
    {
        Y_ASSERT(End >= Begin);
    }

    ui32 GetSize() const noexcept {
        Y_ASSERT(End >= Begin);
        return End - Begin;
    }

    ui32 Begin;
    ui32 End;
    float Weight;
    TVector<ui32> SubgroupId; // can be empty if there's no subgroup data
    TVector<TVector<TCompetitor>> Competitors;
    SAVELOAD(Begin, End, Weight, SubgroupId, Competitors);
};
