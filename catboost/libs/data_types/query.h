#pragma once

#include "pair.h"

#include <util/generic/vector.h>

#include <library/binsaver/bin_saver.h>

struct TQueryInfo {
    TQueryInfo() = default;
    TQueryInfo(int begin, int end)
        : Begin(begin)
        , End(end)
    {
    }

    int Begin;
    int End;
    TVector<ui32> SubgroupId;
    TVector<TVector<TCompetitor>> Competitors;
    SAVELOAD(Begin, End, SubgroupId, Competitors);
};
