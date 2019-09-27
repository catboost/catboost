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

    *res = negativeDer;
    auto localHessian = hessian.Data;
    int idx = 0;
    for (int rowSize = approxDimension; rowSize >= 1; --rowSize) {
        localHessian[idx] -= l2Regularizer;
        idx += rowSize;
    }
    for (double& value : localHessian) {
        value = - value;
    }
    SolveLinearSystem(localHessian, *res);
    for (double& value : *res) {
        value = - value;
    }
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
