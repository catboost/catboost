#pragma once

#include "hessian.h"

#include <catboost/libs/options/enums.h>

#include <library/binsaver/bin_saver.h>
#include <library/containers/2d_array/2d_array.h>

#include <util/generic/vector.h>
#include <util/system/yassert.h>


struct TSum {
    TVector<double> SumDerHistory;
    TVector<double> SumDer2History;
    double SumWeights = 0.0;

    TSum() = default;
    explicit TSum(int iterationCount, int approxDimension = 1,
            EHessianType hessianType = EHessianType::Symmetric)
        : SumDerHistory(iterationCount)
        , SumDer2History(iterationCount) {
        Y_ASSERT(approxDimension == 1);
        Y_ASSERT(hessianType == EHessianType::Symmetric);
    }

    bool operator==(const TSum& other) const {
        return SumDerHistory == other.SumDerHistory &&
               SumWeights == other.SumWeights &&
               SumDer2History == other.SumDer2History;
    }

    inline void AddDerWeight(double delta, double weight, int gradientIteration) {
        SumDerHistory[gradientIteration] += delta;
        if (gradientIteration == 0) {
            SumWeights += weight;
        }
    }

    inline void AddDerDer2(double delta, double der2, int gradientIteration) {
        SumDerHistory[gradientIteration] += delta;
        SumDer2History[gradientIteration] += der2;
    }
    SAVELOAD(SumDerHistory, SumDer2History, SumWeights);
};

struct TSumMulti {
    TVector<TVector<double>> SumDerHistory; // [gradIter][approxIdx]
    TVector<THessianInfo> SumDer2History; // [gradIter][approxIdx1][approxIdx2]
    double SumWeights = 0.0;

    TSumMulti() = default;

    explicit TSumMulti(int iterationCount, int approxDimension, EHessianType hessianType)
    : SumDerHistory(iterationCount, TVector<double>(approxDimension))
    , SumDer2History(iterationCount, THessianInfo(approxDimension, hessianType))
    {}

    bool operator==(const TSumMulti& other) const {
        return SumDerHistory == other.SumDerHistory &&
               SumWeights == other.SumWeights &&
               SumDer2History == other.SumDer2History;
    }

    void AddDerWeight(const TVector<double>& delta, double weight, int gradientIteration) {
        Y_ASSERT(delta.ysize() == SumDerHistory[gradientIteration].ysize());
        for (int dim = 0; dim < SumDerHistory[gradientIteration].ysize(); ++dim) {
            SumDerHistory[gradientIteration][dim] += delta[dim];
        }
        if (gradientIteration == 0) {
            SumWeights += weight;
        }
    }

    void AddDerDer2(const TVector<double>& delta, const THessianInfo& der2, int gradientIteration) {
        Y_ASSERT(delta.ysize() == SumDerHistory[gradientIteration].ysize());
        for (int dim = 0; dim < SumDerHistory[gradientIteration].ysize(); ++dim) {
            SumDerHistory[gradientIteration][dim] += delta[dim];
        }
        SumDer2History[gradientIteration].AddDer2(der2);
    }
    SAVELOAD(SumDerHistory, SumDer2History, SumWeights);
};

namespace {

inline double CalcAverage(double sumDelta,
                          double count,
                          float l2Regularizer,
                          double sumAllWeights,
                          int allDocCount) {
    double inv = count > 0 ? 1. / (count + l2Regularizer * (sumAllWeights / allDocCount)) : 0;
    return sumDelta * inv;
}

inline double CalcModelGradient(const TSum& ss,
                                int gradientIteration,
                                float l2Regularizer,
                                double sumAllWeights,
                                int allDocCount) {
    return CalcAverage(ss.SumDerHistory[gradientIteration],
                       ss.SumWeights,
                       l2Regularizer,
                       sumAllWeights,
                       allDocCount);
}

inline void CalcModelGradientMulti(const TSumMulti& ss,
                                   int gradientIteration,
                                   float l2Regularizer,
                                   double sumAllWeights,
                                   int allDocCount,
                                   TVector<double>* res) {
    const int approxDimension = ss.SumDerHistory[gradientIteration].ysize();
    res->resize(approxDimension);
    for (int dim = 0; dim < approxDimension; ++dim) {
        (*res)[dim] = CalcAverage(ss.SumDerHistory[gradientIteration][dim],
                                  ss.SumWeights,
                                  l2Regularizer,
                                  sumAllWeights,
                                  allDocCount);
    }
}

inline double CalcModelNewtonBody(double sumDer,
                                  double sumDer2,
                                  float l2Regularizer,
                                  double sumAllWeights,
                                  int allDocCount) {
    return sumDer / (-sumDer2 + l2Regularizer * (sumAllWeights / allDocCount));
}

inline double CalcModelNewton(const TSum& ss,
                              int gradientIteration,
                              float l2Regularizer,
                              double sumAllWeights,
                              int allDocCount) {
    return CalcModelNewtonBody(ss.SumDerHistory[gradientIteration],
                               ss.SumDer2History[gradientIteration],
                               l2Regularizer,
                               sumAllWeights,
                               allDocCount);
}
}

void CalcModelNewtonMulti(const TSumMulti& ss,
                          int gradientIteration,
                          float l2Regularizer,
                          double sumAllWeights,
                          int allDocCount,
                          TVector<double>* res);
