#pragma once

#include <library/cpp/binsaver/bin_saver.h>

#include <util/random/fast.h>
#include <util/random/normal.h>
#include <util/system/yassert.h>

#include <cmath>
#include <limits>

constexpr double MINIMAL_SCORE = std::numeric_limits<double>::lowest();

enum class ERandomScoreDistribution {
    Normal,
    Gumbel
};

struct TRandomScore {
    ERandomScoreDistribution Distribution;
    double Val;
    double StDev;

public:
    TRandomScore()
        : Distribution(ERandomScoreDistribution::Normal)
        , Val(MINIMAL_SCORE)
        , StDev(0)
    {
    }

    TRandomScore(ERandomScoreDistribution distribution, double v, double s)
        : Distribution(distribution)
        , Val(v)
        , StDev(s)
    {
    }

    SAVELOAD(Distribution, Val, StDev);

    template <typename TRng>
    double GetInstance(TRng& rand) const {
        if (Distribution == ERandomScoreDistribution::Normal) {
            return Val + NormalDistribution<double>(rand, 0, StDev);
        } else {
            Y_ASSERT(Distribution == ERandomScoreDistribution::Gumbel);
            return Val + StDev * std::log(std::log(1.0 / rand.GenRandReal1()));
        }
    }
};
