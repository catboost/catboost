#include "online_predictor.h"

#include <catboost/libs/helpers/matrix.h>

void CalcModelNewtonMulti(const TSumMulti& ss,
                          int gradientIteration,
                          float l2Regularizer,
                          double sumAllWeights,
                          int allDocCount,
                          TVector<double>* res) {
    TVector<double> total1st = ss.SumDerHistory[gradientIteration];
    for (auto& elem : total1st) {
        elem = -elem;
    }
    TArray2D<double> total2nd = ss.SumDer2History[gradientIteration];
    const int approxDimension = ss.SumDerHistory[gradientIteration].ysize();

    l2Regularizer *= sumAllWeights / allDocCount;
    for (int dim = 0; dim < approxDimension; ++dim) {
        total2nd[dim][dim] -= l2Regularizer;
    }
    SolveLinearSystem(total2nd, total1st, res);
}
