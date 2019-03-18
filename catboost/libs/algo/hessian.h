#pragma once

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/options/enums.h>

#include <util/generic/vector.h>

#include <library/binsaver/bin_saver.h>


class THessianInfo;

class THessian {
public:
    THessian() = delete;

    /// (Hessian - L2) x optimalDirection = -Gradient
    static void SolveNewtonEquation(
        const THessianInfo& /*hessian*/,
        const TVector<double>& /*negativeDer*/,
        const float /*l2Regularizer*/,
        TVector<double>* /*res*/)
    {
        CB_ENSURE(false, "Not implemented");
    }

    static int CalcInternalDer2DataSize(int /*approxDimension*/) {
        CB_ENSURE(false, "Not implemented");
    }
};

class TSymmetricHessian : public THessian {
public:
    static void SolveNewtonEquation(
        const THessianInfo& /*hessian*/,
        const TVector<double>& /*negativeDer*/,
        const float /*l2Regularizer*/,
        TVector<double>* /*res*/);

    static int CalcInternalDer2DataSize(int /*approxDimension*/);
};

class TDiagonalHessian : public THessian {
public:
    static void SolveNewtonEquation(
        const THessianInfo& /*hessian*/,
        const TVector<double>& /*negativeDer*/,
        const float /*l2Regularizer*/,
        TVector<double>* /*res*/);

    static int CalcInternalDer2DataSize(int /*approxDimension*/);
};



static int CalcInternalDer2DataSize(EHessianType hessianType, int approxDimension) {
    if (hessianType == EHessianType::Symmetric) {
        return TSymmetricHessian::CalcInternalDer2DataSize(approxDimension);
    } else {
        Y_ASSERT(hessianType == EHessianType::Diagonal);
        return TDiagonalHessian::CalcInternalDer2DataSize(approxDimension);
    }
}


void SolveNewtonEquation(
    const THessianInfo& /*hessian*/,
    const TVector<double>& /*negativeDer*/,
    const float /*l2Regularizer*/,
    TVector<double>* /*res*/);


class THessianInfo {
public:
    explicit THessianInfo(int approxDimension, EHessianType hessianType)
        : ApproxDimension(approxDimension)
        , HessianType(hessianType)
        , Data(CalcInternalDer2DataSize(hessianType, approxDimension))
    {}

    THessianInfo() = default;

    bool operator==(const THessianInfo& other) const;

    SAVELOAD(HessianType, ApproxDimension, Data);

    void AddDer2(const THessianInfo& hessian);

public:
    int ApproxDimension;
    EHessianType HessianType;
    TVector<double> Data;
};
