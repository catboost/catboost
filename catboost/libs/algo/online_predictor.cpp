#include "online_predictor.h"

#include <catboost/libs/helpers/matrix.h>


void CalcDeltaNewtonMulti(
    const TSumMulti& ss,
    float l2Regularizer,
    double sumAllWeights,
    int allDocCount,
    TVector<double>* res) {

    TVector<double> total1st = ss.SumDer;
    for (auto& elem : total1st) {
        elem = -elem;
    }

    l2Regularizer *= sumAllWeights / allDocCount;
    SolveNewtonEquation(ss.SumDer2, total1st, l2Regularizer, res);
}
