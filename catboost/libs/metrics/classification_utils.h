#pragma once

#include <util/generic/fwd.h>

struct TMetricHolder;

int GetApproxClass(TConstArrayRef<TVector<double>> approx, int docIdx, double predictionLogitBorder);
int GetApproxClass(TConstArrayRef<TConstArrayRef<double>> approx, int docIdx, double predictionLogitBorder);

void GetPositiveStats(
        TConstArrayRef<TConstArrayRef<double>> approx,
        TConstArrayRef<float> target,
        TConstArrayRef<float> weight,
        int begin,
        int end,
        int positiveClass,
        double targetBorder,
        double predictionLogitBorder,
        double* truePositive,
        double* targetPositive,
        double* approxPositive
);

void GetSpecificity(
        TConstArrayRef<TConstArrayRef<double>> approx,
        TConstArrayRef<float> target,
        TConstArrayRef<float> weight,
        int begin,
        int end,
        int positiveClass,
        double targetBorder,
        double predictionLogitBorder,
        double* trueNegative,
        double* targetNegative
);
