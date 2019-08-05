#pragma once

#include "calc_score_cache.h"

#include <util/generic/vector.h>
#include <util/generic/xrange.h>

#include <cmath>


struct TSplitEnsembleSpec;

class IScoreCalcer {
public:
    virtual ~IScoreCalcer() = default;

    virtual void SetSplitsCount(int splitsCount) {
        SplitsCount = splitsCount;
    }

    int GetSplitsCount() {
        return SplitsCount;
    }

    virtual TVector<double> GetScores() const = 0;

protected:
    int SplitsCount;
};

class IPointwiseScoreCalcer : public IScoreCalcer {
public:
    virtual void AddLeaf(int splitIdx, double leafApprox, const TBucketStats& leafStats) = 0;
};

class TCosineScoreCalcer final : public IPointwiseScoreCalcer {
public:
    void SetSplitsCount(int splitsCount) override {
        IPointwiseScoreCalcer::SetSplitsCount(splitsCount);
        Numerators.resize(splitsCount);
        Denominators.resize(splitsCount, 1e-100);
    }

    TVector<double> GetScores() const override {
        TVector<double> scores(SplitsCount);
        for (int i : xrange(SplitsCount)) {
            scores[i] = Numerators[i] / sqrt(Denominators[i]);
        }
        return scores;
    }

    void AddLeaf(int splitIdx, double leafApprox, const TBucketStats& leafStats) override {
        Numerators[splitIdx] += leafApprox * leafStats.SumWeightedDelta;
        Denominators[splitIdx] += leafApprox * leafApprox * leafStats.SumWeight;
    }

private:
    TVector<double> Numerators;
    TVector<double> Denominators;
};

class TL2ScoreCalcer final : public IPointwiseScoreCalcer {
public:
    void SetSplitsCount(int splitsCount) override {
        IPointwiseScoreCalcer::SetSplitsCount(splitsCount);
        Scores.resize(splitsCount);
    }

    TVector<double> GetScores() const override {
        return Scores;
    }

    void AddLeaf(int splitIdx, double leafApprox, const TBucketStats& leafStats) override {
        Scores[splitIdx] += 2 * leafApprox * leafStats.SumWeightedDelta - leafApprox * leafApprox * leafStats.SumWeight;
    }

private:
    TVector<double> Scores;
};


int CalcSplitsCount(
    const TSplitEnsembleSpec& splitEnsembleSpec,
    int bucketCount,
    ui32 oneHotMaxSize
);
