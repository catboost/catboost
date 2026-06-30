#pragma once

#include "hessian.h"

#include <catboost/private/libs/options/enums.h>

#include <library/cpp/binsaver/bin_saver.h>

#include <util/generic/vector.h>
#include <util/system/yassert.h>


struct TSum {
    double SumDer = 0.0;
    double SumDer2 = 0.0;
    double SumWeights = 0.0;

public:
    explicit TSum(int approxDimension = 1, EHessianType hessianType = EHessianType::Symmetric) {
        Y_ASSERT(approxDimension == 1);
        Y_ASSERT(hessianType == EHessianType::Symmetric);
    }

    SAVELOAD(SumDer, SumDer2, SumWeights);

    bool operator==(const TSum& other) const {
        return SumDer == other.SumDer &&
            SumWeights == other.SumWeights &&
            SumDer2 == other.SumDer2;
    }

    inline void SetZeroDers() {
        SumDer = 0.0;
        SumDer2 = 0.0;
    }

    inline void AddDerWeight(double delta, double weight, bool updateWeight) {
        SumDer += delta;
        if (updateWeight) {
            SumWeights += weight;
        }
    }

    inline void AddDerDer2(double delta, double der2) {
        SumDer += delta;
        SumDer2 += der2;
    }
};

struct TSumMulti {
    TVector<double> SumDer; // [approxIdx]
    THessianInfo SumDer2; // [approxIdx1][approxIdx2]
    double SumWeights = 0.0;

public:
    TSumMulti() = default;

    explicit TSumMulti(int approxDimension, EHessianType hessianType)
        : SumDer(approxDimension)
        , SumDer2(approxDimension, hessianType)
    {}

    explicit TSumMulti(int approxDimension)
        : SumDer(approxDimension)
    {}

    SAVELOAD(SumDer, SumDer2, SumWeights);

    bool operator==(const TSumMulti& other) const {
        return SumDer == other.SumDer &&
            SumWeights == other.SumWeights &&
            SumDer2 == other.SumDer2;
    }

    inline void SetZeroDers() {
        Fill(SumDer.begin(), SumDer.end(),  0.0);
        Fill(SumDer2.Data.begin(), SumDer2.Data.end(), 0.0);
    }

    inline void AddDerWeight(const TVector<double>& delta, double weight, bool updateWeight) {
        Y_ASSERT(delta.ysize() == SumDer.ysize());
        for (int dim = 0; dim < SumDer.ysize(); ++dim) {
            SumDer[dim] += delta[dim];
        }
        if (updateWeight) {
            SumWeights += weight;
        }
    }

    inline void AddDerDer2(const TVector<double>& delta, const THessianInfo& der2) {
        Y_ASSERT(delta.ysize() == SumDer.ysize());
        for (int dim = 0; dim < SumDer.ysize(); ++dim) {
            SumDer[dim] += delta[dim];
        }
        SumDer2.AddDer2(der2);
    }

};

inline static TSumMulti MakeZeroDers(
    int approxDimension,
    ELeavesEstimation estimationMethod,
    EHessianType hessianType
) {
    if (estimationMethod == ELeavesEstimation::Gradient) {
        return TSumMulti(approxDimension);
    } else {
        return TSumMulti(approxDimension, hessianType);
    }
}

inline double CalcAverage(
    double sumDelta,
    double count,
    double scaledL2Regularizer) {

    double inv = count > 0 ? 1. / (count + scaledL2Regularizer) : 0;
    return sumDelta * inv;
}

inline double ScaleL2Reg(
    float l2Regularizer,
    double sumAllWeights,
    int allDocCount) {

    return l2Regularizer * (sumAllWeights / allDocCount);
}

inline double CalcAverage(
    double sumDelta,
    double count,
    float l2Regularizer,
    double sumAllWeights,
    int allDocCount) {

    return CalcAverage(sumDelta, count, ScaleL2Reg(l2Regularizer, sumAllWeights, allDocCount));
}

inline double CalcDeltaGradient(
    const TSum& ss,
    float l2Regularizer,
    double sumAllWeights,
    int allDocCount) {

    return CalcAverage(ss.SumDer, ss.SumWeights, l2Regularizer, sumAllWeights, allDocCount);
}

inline void CalcDeltaGradientMulti(
    const TSumMulti& ss,
    float l2Regularizer,
    double sumAllWeights,
    int allDocCount,
    TVector<double>* res) {

    const int approxDimension = ss.SumDer.ysize();
    res->resize(approxDimension);
    for (int dim = 0; dim < approxDimension; ++dim) {
        (*res)[dim] = CalcAverage(ss.SumDer[dim], ss.SumWeights, l2Regularizer, sumAllWeights, allDocCount);
    }
}

inline double CalcDeltaNewtonBody(
    double sumDer,
    double sumDer2,
    float l2Regularizer,
    double sumAllWeights,
    int allDocCount) {

    return sumDer / (-sumDer2 + l2Regularizer * (sumAllWeights / allDocCount));
}

inline double CalcDeltaNewton(
    const TSum& ss,
    float l2Regularizer,
    double sumAllWeights,
    int allDocCount) {

    return CalcDeltaNewtonBody(ss.SumDer, ss.SumDer2, l2Regularizer, sumAllWeights, allDocCount);
}

void CalcDeltaNewtonMulti(
    const TSumMulti& ss,
    float l2Regularizer,
    double sumAllWeights,
    int allDocCount,
    TVector<double>* res);
