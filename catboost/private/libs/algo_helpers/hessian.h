#pragma once

#include <catboost/libs/helpers/exception.h>
#include <catboost/private/libs/options/enums.h>

#include <library/cpp/binsaver/bin_saver.h>

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>


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

    inline void AddDer2(const THessianInfo& hessian) {
        Y_ASSERT(HessianType == hessian.HessianType && Data.ysize() == hessian.Data.ysize());
        const auto hessianRef = MakeArrayRef(hessian.Data);
        for (int dim = 0; dim < Data.ysize(); ++dim) {
            Data[dim] += hessianRef[dim];
        }
    }

public:
    int ApproxDimension;
    EHessianType HessianType;
    TVector<double> Data;
};
