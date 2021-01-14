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

    int GetSplitsCount() const {
        return SplitsCount;
    }

    virtual TVector<double> GetScores() const = 0;

protected:
    int SplitsCount;
};

class IPointwiseScoreCalcer : public IScoreCalcer {
public:
    void SetL2Regularizer(double l2Regularizer) {
        L2Regularizer = l2Regularizer;
    }

    virtual void AddLeaf(int splitIdx, double leafApprox, const TBucketStats& leafStats) = 0;

    virtual void AddLeafPlain(int splitIdx, const TBucketStats& leftStats, const TBucketStats& rightStats) = 0;

    virtual void AddLeafOrdered(int splitIdx, const TBucketStats& leftStats, const TBucketStats& rightStats) = 0;

protected:
    double L2Regularizer = 1e-20;
};

class TCosineScoreCalcer final : public IPointwiseScoreCalcer {
    using TFraction = std::array<double, 2>;
public:
    void SetSplitsCount(int splitsCount) override {
        IPointwiseScoreCalcer::SetSplitsCount(splitsCount);
        Scores.resize(splitsCount, {0, 1e-100});
    }

    TVector<double> GetScores() const override {
        TVector<double> scores(SplitsCount);
        for (int i : xrange(SplitsCount)) {
            scores[i] = Scores[i][0] / sqrt(Scores[i][1]);
        }
        return scores;
    }

    void AddLeaf(int splitIdx, double leafApprox, const TBucketStats& leafStats) override {
        Scores[splitIdx][0] += leafApprox * leafStats.SumWeightedDelta;
        Scores[splitIdx][1] += leafApprox * leafApprox * leafStats.SumWeight;
    }

    void AddLeafPlain(int splitIdx, const TBucketStats& leftStats, const TBucketStats& rightStats) override;

    void AddLeafOrdered(int splitIdx, const TBucketStats& leftStats, const TBucketStats& rightStats) override;

private:
    TVector<TFraction> Scores;
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
        Scores[splitIdx] += leafApprox * leafStats.SumWeightedDelta;
    }

    void AddLeafPlain(int splitIdx, const TBucketStats& leftStats, const TBucketStats& rightStats) override;

    void AddLeafOrdered(int splitIdx, const TBucketStats& leftStats, const TBucketStats& rightStats) override;

private:
    TVector<double> Scores;
};


int CalcSplitsCount(
    const TSplitEnsembleSpec& splitEnsembleSpec,
    int bucketCount,
    ui32 oneHotMaxSize
);

inline THolder<IPointwiseScoreCalcer> MakePointwiseScoreCalcer(EScoreFunction scoreFunction) {
    switch(scoreFunction) {
        case EScoreFunction::Cosine:
            return MakeHolder<TCosineScoreCalcer>();
        case EScoreFunction::L2:
            return MakeHolder<TL2ScoreCalcer>();
        default:
            CB_ENSURE(false, "Only Cosine and L2 score functions are supported for CPU.");
    }
}
