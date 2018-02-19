#pragma once

#include "pair.h"

#include <util/generic/vector.h>

struct TQueryInfo {
    int Begin;
    int End;
    TVector<TVector<TCompetitor>> Competitors;
};
