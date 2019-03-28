#pragma once

#include <library/binsaver/bin_saver.h>

#include <util/random/normal.h>
#include <util/random/fast.h>

constexpr double MINIMAL_SCORE = -1e38;

struct TRandomScore {
    double Val, StDev;

    TRandomScore()
        : Val(MINIMAL_SCORE)
        , StDev(0)
    {
    }
    TRandomScore(double v, double s)
        : Val(v)
        , StDev(s)
    {
    }
    template <typename TRng>
    double GetInstance(TRng& rand) const {
        return Val + NormalDistribution<double>(rand, 0, StDev);
    }
    SAVELOAD(Val, StDev);
};
