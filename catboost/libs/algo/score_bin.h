#pragma once

#include <util/generic/vector.h>
#include <util/generic/xrange.h>

#include <cmath>


struct TSplitEnsembleSpec;


// The class that stores final stats for a split and provides interface to calculate the deterministic score.
struct TScoreBin {
    double DP = 0;
    double D2 = 1e-100;

public:
    inline double GetScore() const {
        return DP / sqrt(D2);
    }
};

// Helper function that calculates deterministic scores given bins with statistics for each split.
inline TVector<double> GetScores(const TVector<TScoreBin>& scoreBin) {
    TVector<double> scores(scoreBin.size());
    for (auto i : xrange(scoreBin.size())) {
        scores[i] = scoreBin[i].GetScore();
    }
    return scores;
}


int CalcScoreBinCount(
    const TSplitEnsembleSpec& splitEnsembleSpec,
    int bucketCount,
    ui32 oneHotMaxSize
);
