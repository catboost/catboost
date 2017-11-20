#include "online_predictor.h"

#include <catboost/libs/helpers/matrix.h>

void CalcModelNewtonMulti(const TSumMulti& ss, int gradientIteration, float l2Regularizer, TVector<double>* res) {
    const int approxDimension = ss.SumDerHistory.ysize();
    res->resize(approxDimension);
    TVector<double> total1st(approxDimension);
    TArray2D<double> total2nd(approxDimension, approxDimension);
    for (int dimY = 0; dimY < approxDimension; ++dimY) {
        total1st[dimY] = ss.SumDerHistory[dimY].ysize() <= gradientIteration ?
                         0 : -ss.SumDerHistory[dimY][gradientIteration];
        for (int dimX = 0; dimX < approxDimension; ++dimX) {
            total2nd[dimY][dimX] = ss.SumDer2History[dimY][dimX].ysize() <= gradientIteration ?
                                   0 : ss.SumDer2History[dimY][dimX][gradientIteration];
        }
        total2nd[dimY][dimY] -= l2Regularizer;
    }
    FindSomeLinearSolution(total2nd, total1st, res);
}
