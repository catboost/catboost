#pragma once

#include <catboost/libs/model/model.h>
#include <catboost/libs/data/pool.h>

#include <util/generic/vector.h>

struct TRocPoint {
    double Boundary = 0.0;
    double FalseNegativeRate = 0.0;
    double FalsePositiveRate = 0.0;

    TRocPoint() = default;
    TRocPoint(
        double boundary,
        double falseNegativeRate,
        double falsePositiveRate
    )
        : Boundary(boundary)
        , FalseNegativeRate(falseNegativeRate)
        , FalsePositiveRate(falsePositiveRate)
    {
    }
};

struct TRocCurve {
    TRocCurve(const TFullModel& model, const TVector<TPool>& pool, int threadCount = 1);

    TRocCurve(
        const TVector<TVector<double>>& approxes,
        const TVector<TPool>& pool,
        NPar::TLocalExecutor* localExecutor
    );

    TRocCurve(
        const TVector<TVector<double>>& approxes,
        const TVector<TVector<float>>& labels,
        int threadCount
    );

    TRocCurve(const TVector<TRocPoint>& points);

    TRocCurve() = default;

    double SelectDecisionBoundaryByFalsePositiveRate(double falsePositiveRate);

    double SelectDecisionBoundaryByFalseNegativeRate(double falseNegativeRate);

    double SelectDecisionBoundaryByIntersection();

    TVector<TRocPoint> GetCurvePoints();

    void OutputRocCurve(const TString& outputPath);
private:
    TVector<TRocPoint> Points; // Points are sorted by Boundary from higher to lower
    size_t RateCurvesIntersection;

    void BuildCurve(
        const TVector<TVector<double>>& approxes, // [poolId][docId]
        const TVector<TVector<float>>& labels, // [poolId][docId]
        NPar::TLocalExecutor* localExecutor
    );

    void AddPoint(double newBoundary, double newFnr, double newFpr);
};
