#pragma once

#include <catboost/libs/data/data_provider.h>

#include <util/generic/fwd.h>
#include <util/generic/vector.h>


class TFullModel;

namespace NPar {
    class ILocalExecutor;
}


struct TRocPoint {
    double Boundary = 0.0;
    double FalseNegativeRate = 0.0;
    double FalsePositiveRate = 0.0;

public:
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
    constexpr static double EPS = 1e-13; // for comparisons of probabilities and coordinates

public:
    TRocCurve(const TFullModel& model, const TVector<NCB::TDataProviderPtr>& datasets, int threadCount = 1);

    TRocCurve(
        const TVector<TVector<double>>& approxes,
        const TVector<TConstArrayRef<float>>& labels,
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

private:
    void BuildCurve(
        const TVector<TVector<double>>& approxes, // [poolId][docId]
        const TVector<TConstArrayRef<float>>& labels, // [poolId][docId]
        NPar::ILocalExecutor* localExecutor
    );

    static TRocPoint IntersectSegments(const TRocPoint& leftEnds, const TRocPoint& rightEnds);

    void AddPoint(double newBoundary, double newFnr, double newFpr);
};
