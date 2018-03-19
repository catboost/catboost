#pragma once

#include <library/containers/2d_array/2d_array.h>

#include <util/generic/vector.h>
#include <util/system/yassert.h>

#include <library/binsaver/bin_saver.h>

struct TSum {
    TVector<double> SumDerHistory;
    TVector<double> SumDer2History;
    double SumWeights;

    TSum() = default;
    TSum(int iterationCount)
        : SumDerHistory(iterationCount)
        , SumDer2History(iterationCount)
        , SumWeights(0) {
    }

    bool operator==(const TSum& other) const {
        return SumDerHistory == other.SumDerHistory &&
               SumWeights == other.SumWeights &&
               SumDer2History == other.SumDer2History;
    }

    void Clear() {
        SumWeights = 0;
        SumDerHistory.clear();
        SumDer2History.clear();
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
    TVector<TVector<double>> SumDerHistory;
    TArray2D<TVector<double>> SumDer2History;
    double SumWeights;

    TSumMulti()
        : SumWeights(0)
    {
    }

    explicit TSumMulti(int approxDimension)
        : SumWeights(0)
    {
        SetApproxDimension(approxDimension);
    }

    void SetApproxDimension(int approxDimension) {
        SumDerHistory.resize(approxDimension);
        SumDer2History.SetSizes(approxDimension, approxDimension);
    }

    bool operator==(const TSumMulti& other) const {
        return SumDerHistory == other.SumDerHistory &&
               SumWeights == other.SumWeights &&
               SumDer2History == other.SumDer2History;
    }

    void Clear() {
        SumWeights = 0;
        SumDerHistory.clear();
        SumDer2History.Clear();
    }

    void AddDerWeight(const TVector<double>& delta, double weight, int gradientIteration) {
        for (int dim = 0; dim < SumDerHistory.ysize(); ++dim) {
            if (SumDerHistory[dim].ysize() < gradientIteration + 1) {
                SumDerHistory[dim].resize(gradientIteration + 1);
            }
            SumDerHistory[dim][gradientIteration] += delta[dim];
        }
        if (gradientIteration == 0) {
            SumWeights += weight;
        }
    }

    void AddDerDer2(const TVector<double>& delta, const TArray2D<double>& der2, int gradientIteration) {
        for (size_t dimY = 0; dimY < SumDer2History.GetYSize(); ++dimY) {
            if (SumDerHistory[dimY].ysize() < gradientIteration + 1) {
                SumDerHistory[dimY].resize(gradientIteration + 1);
            }
            SumDerHistory[dimY][gradientIteration] += delta[dimY];
            for (size_t dimX = 0; dimX < SumDer2History.GetXSize(); ++dimX) {
                if (SumDer2History[dimY][dimX].ysize() < gradientIteration + 1) {
                    SumDer2History[dimY][dimX].resize(gradientIteration + 1);
                }
                SumDer2History[dimY][dimX][gradientIteration] += der2[dimY][dimX];
            }
        }
    }
};

namespace {

inline double CalcAverage(double sumDelta, double count, float l2Regularizer) {
    double inv = count > 0 ? 1. / (count + l2Regularizer) : 0;
    return sumDelta * inv;
}

inline double CalcModelGradient(const TSum& ss, int gradientIteration, float l2Regularizer) {
    if (ss.SumDerHistory.ysize() <= gradientIteration) {
        return 0;
    }
    return CalcAverage(ss.SumDerHistory[gradientIteration], ss.SumWeights, l2Regularizer);
}

inline void CalcModelGradientMulti(const TSumMulti& ss, int gradientIteration, float l2Regularizer, TVector<double>* res) {
    const int approxDimension = ss.SumDerHistory.ysize();
    res->resize(approxDimension);
    for (int dim = 0; dim < approxDimension; ++dim) {
        if (ss.SumDerHistory[dim].ysize() <= gradientIteration) {
            (*res)[dim] = 0;
        } else {
            (*res)[dim] = CalcAverage(ss.SumDerHistory[dim][gradientIteration], ss.SumWeights, l2Regularizer);
        }
    }
}

inline double CalcModelNewtonBody(double sumDer, double sumDer2, float l2Regularizer) {
    return sumDer / (-sumDer2 + l2Regularizer);
}

inline double CalcModelNewton(const TSum& ss, int gradientIteration, float l2Regularizer) {
    if (ss.SumDerHistory.ysize() <= gradientIteration) {
        return 0;
    }
    return CalcModelNewtonBody(ss.SumDerHistory[gradientIteration], ss.SumDer2History[gradientIteration], l2Regularizer);
}
}

void CalcModelNewtonMulti(const TSumMulti& ss, int gradientIteration, float l2Regularizer, TVector<double>* res);
