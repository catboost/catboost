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

    l2Regularizer *= sumAllWeights / allDocCount;
    SolveNewtonEquation(ss.SumDer2History[gradientIteration],
                        total1st,
                        l2Regularizer,
                        res);
}
