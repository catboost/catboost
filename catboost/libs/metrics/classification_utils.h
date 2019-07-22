#pragma once

#include <util/generic/fwd.h>

struct TMetricHolder;

int GetApproxClass(TConstArrayRef<TVector<double>> approx, int docIdx);

void GetPositiveStats(
        TConstArrayRef<TVector<double>> approx,
        TConstArrayRef<float> target,
        TConstArrayRef<float> weight,
        int begin,
        int end,
        int positiveClass,
        double border,
        double* truePositive,
        double* targetPositive,
        double* approxPositive
);

void GetSpecificity(
        TConstArrayRef<TVector<double>> approx,
        TConstArrayRef<float> target,
        TConstArrayRef<float> weight,
        int begin,
        int end,
        int positiveClass,
        double border,
        double* trueNegative,
        double* targetNegative
);

void GetTotalPositiveStats(
        TConstArrayRef<TVector<double>> approx,
        TConstArrayRef<float> target,
        TConstArrayRef<float> weight,
        int begin,
        int end,
        TVector<double>* truePositive,
        TVector<double>* targetPositive,
        TVector<double>* approxPositive,
        double border
);

TMetricHolder GetAccuracy(
        const TVector<TVector<double>>& approx,
        TConstArrayRef<float> target,
        TConstArrayRef<float> weight,
        int begin,
        int end,
        double border
);
