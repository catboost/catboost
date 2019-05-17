#include "hessian.h"

#include <catboost/libs/helpers/matrix.h>
#include <catboost/libs/lapack/linear_system.h>


void SolveNewtonEquation(
    const THessianInfo& hessian,
    const TVector<double>& negativeDer,
    const float l2Regularizer,
    TVector<double>* res)
{
    if (hessian.HessianType == EHessianType::Symmetric) {
        TSymmetricHessian::SolveNewtonEquation(hessian, negativeDer, l2Regularizer, res);
    } else {
        Y_ASSERT(hessian.HessianType == EHessianType::Diagonal);
        TDiagonalHessian::SolveNewtonEquation(hessian, negativeDer, l2Regularizer, res);
    }
}

void TSymmetricHessian::SolveNewtonEquation(
    const THessianInfo& hessian,
    const TVector<double>& negativeDer,
    const float l2Regularizer,
    TVector<double>* res)
{
    Y_ASSERT(hessian.ApproxDimension == negativeDer.ysize());
    const int approxDimension = hessian.ApproxDimension;
    TArray2D<double> der2(approxDimension, approxDimension);
    int idx = 0;
    for (int dimY = 0; dimY < approxDimension; ++dimY) {
        for (int dimX = dimY; dimX < approxDimension; ++dimX) {
            der2[dimY][dimX] = hessian.Data[idx];
            der2[dimX][dimY] = hessian.Data[idx++];
        }
        der2[dimY][dimY] -= l2Regularizer;
    }

    SolveLinearSystem(der2, negativeDer, res);
}


void TDiagonalHessian::SolveNewtonEquation(
    const THessianInfo& hessian,
    const TVector<double>& negativeDer,
    const float l2Regularizer,
    TVector<double>* res)
{
    Y_ASSERT(res);
    Y_ASSERT(hessian.ApproxDimension == negativeDer.ysize());
    const int approxDimension = hessian.ApproxDimension;
    res->resize(approxDimension);
    for (int dim = 0; dim < approxDimension; ++dim) {
        (*res)[dim] = negativeDer[dim] / (hessian.Data[dim] - l2Regularizer);
    }
}


int TSymmetricHessian::CalcInternalDer2DataSize(int approxDimenstion) {
    return (approxDimenstion * (approxDimenstion + 1)) / 2;
}

int TDiagonalHessian::CalcInternalDer2DataSize(int approxDimension) {
    return approxDimension;
}


void THessianInfo::AddDer2(const THessianInfo& hessian) {
    Y_ASSERT(HessianType == hessian.HessianType && Data.ysize() == hessian.Data.ysize());
    for (int dim = 0; dim < Data.ysize(); ++dim) {
        Data[dim] += hessian.Data[dim];
    }
}
