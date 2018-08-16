#pragma once

#include <util/generic/vector.h>

#include <cmath>

// The class that stores final stats for a split and provides interface to calculate the deterministic score.
struct TScoreBin {
    double DP = 0, D2 = 1e-100;

    inline double GetScore() const {
        return DP / sqrt(D2);
    }
};

// Helper function that calculates deterministic scores given bins with statistics for each split.
inline TVector<double> GetScores(const TVector<TScoreBin>& scoreBin) {
    const int splitCount = scoreBin.ysize() - 1;
    TVector<double> scores(splitCount);
    for (int splitIdx = 0; splitIdx < splitCount; ++splitIdx) {
        scores[splitIdx] = scoreBin[splitIdx].GetScore();
    }
    return scores;
}
