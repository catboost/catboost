#pragma once

#include "pair.h"

#include <util/generic/vector.h>

#include <library/binsaver/bin_saver.h>

struct TQueryInfo {
    TQueryInfo() = default;
    TQueryInfo(int begin, int end)
        : Begin(begin)
        , End(end)
        , Weight(1.0f)
    {
    }

    int GetSize() const noexcept {
        return End - Begin;
    }

    int Begin;
    int End;
    float Weight;
    TVector<ui32> SubgroupId;
    TVector<TVector<TCompetitor>> Competitors;
    SAVELOAD(Begin, End, Weight, SubgroupId, Competitors);
};
