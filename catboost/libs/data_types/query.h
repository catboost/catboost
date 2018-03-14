#pragma once

#include "pair.h"

#include <util/generic/vector.h>

struct TQueryInfo {
    TQueryInfo(int begin, int end)
        : Begin(begin)
        , End(end)
    {
    }

    int Begin;
    int End;
    TVector<ui32> SubgroupId;
    TVector<TVector<TCompetitor>> Competitors;
};
