#pragma once

#include <library/cpp/binsaver/bin_saver.h>

#include <util/random/fast.h>
#include <util/random/normal.h>

#include <limits>

constexpr double MINIMAL_SCORE = std::numeric_limits<double>::lowest();

struct TRandomScore {
    double Val;
    double StDev;

public:
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

    SAVELOAD(Val, StDev);

    template <typename TRng>
    double GetInstance(TRng& rand) const {
        return Val + NormalDistribution<double>(rand, 0, StDev);
    }
};
