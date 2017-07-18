#pragma once

#include "projection.h"
#include <library/containers/2d_array/2d_array.h>

struct TOnlineCTR {
    yvector<TArray2D<yvector<ui8>>> Feature; // Feature[ctrIdx][classIdx][priorIdx][docIdx]
};

struct TCtrHistory {
    int N[2];
    void Clear() {
        N[0] = 0;
        N[1] = 0;
    }
    Y_SAVELOAD_DEFINE(N[0], N[1]);
};

struct TCtrMeanHistory {
    float Sum;
    int Count;
    bool operator==(const TCtrMeanHistory& other) const {
        return std::tie(Sum, Count) == std::tie(other.Sum, other.Count);
    }
    void Clear() {
        Sum = 0;
        Count = 0;
    }
    void Add(float target) {
        Sum += target;
        ++Count;
    }
    Y_SAVELOAD_DEFINE(Sum, Count);
};

using TOnlineCTRHash = yhash<TProjection, TOnlineCTR, TProjHash>;
