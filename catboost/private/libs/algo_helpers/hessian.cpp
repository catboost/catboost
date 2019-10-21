#include "hessian.h"

#include <catboost/libs/helpers/matrix.h>
#include <catboost/private/libs/lapack/linear_system.h>

#include <util/generic/algorithm.h>

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
    const auto hessianSize = (approxDimension + 1) * approxDimension / 2;

    float maxTraceElement = l2Regularizer;
    for (int idx = 0, rowSize = approxDimension; idx < hessianSize; idx += rowSize, --rowSize) {
        maxTraceElement = Max<float>(maxTraceElement, -localHessian[idx]);
    }

    const float adjustedL2Regularizer = Max(l2Regularizer, maxTraceElement * std::numeric_limits<float>::epsilon());

    for (int idx = 0, rowSize = approxDimension; idx < hessianSize; idx += rowSize, --rowSize) {
        localHessian[idx] -= adjustedL2Regularizer;
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
