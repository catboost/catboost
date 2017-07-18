#pragma once

#include <algorithm>
#include <tuple>

#include <util/generic/yexception.h>
#include <util/generic/vector.h>

#include "math_utils.h"

template <class TEl, class TRes = TEl>
class TRandVarParamsEstimator {
    TRes Sum;
    TRes SumOfSq;

    ui64 NEls;

public:
    TRandVarParamsEstimator()
        : Sum(TRes(0))
        , SumOfSq(TRes(0))
        , NEls(0)
    {
    }

    void Add(TEl el) {
        TRes el1 = TRes(el);
        Sum += el1;
        SumOfSq += el1 * el1;
        ++NEls;
    }

    void Merge(const TRandVarParamsEstimator& rhs) {
        Sum += rhs.Sum;
        SumOfSq += rhs.SumOfSq;
        NEls += rhs.NEls;
    }

    ui64 GetNEls() const {
        return NEls;
    }

    TRes E() const {
        if (NEls == 0)
            ythrow yexception() << "Can't estimate E for 0 elements";
        return Sum / TRes(NEls);
    }

    TRes D() const {
        if (NEls < 2)
            ythrow yexception() << "Can't estimate D for < 2 elements";
        return (SumOfSq - (Sum * Sum) / TRes(NEls)) / TRes(NEls - 1);
    }
};

template <class T>
class TEqSpacedXLinTrendEstimator // x = 0,1...
{
    T SumY;
    T SumXY;
    T N;
    bool NonTrend;

public:
    TEqSpacedXLinTrendEstimator(bool nonTrend = false)
        : SumY(0)
        , SumXY(0)
        , N(0)
        , NonTrend(nonTrend)
    {
    }

    void Add(T y) {
        SumY += y;
        SumXY += N * y;
        ++N;
    }

    // return  A,B of Ex_i = a * x_i + b
    std::pair<T, T> Estimate() const {
        if (NonTrend)
            return std::make_pair(T(0), SumY / N);

        T b = (2 * (2 * N - 1) * SumY - 6 * SumXY) / (N * (N + 1));
        T a = 2 * (SumY - N * b) / (N * (N - 1));

        return std::make_pair(a, b);
    }
};

template <class T>
struct TLinTrendEstimator {
    T SumX;
    T SumY;
    T SumXY;
    T SumSqX;
    T N;
    bool NonTrend;

public:
    TLinTrendEstimator(bool nonTrend = false)
        : SumX(0)
        , SumY(0)
        , SumXY(0)
        , SumSqX(0)
        , N(0)
        , NonTrend(nonTrend)
    {
    }

    void Add(T x, T y) {
        SumX += x;
        SumY += y;
        SumXY += x * y;
        SumSqX += x * x;
        ++N;
    }

    void Sub(T x, T y) {
        Y_ASSERT(N > 0);
        SumX -= x;
        SumY -= y;
        SumXY -= x * y;
        SumSqX -= x * x;
        --N;
    }

    // return  A,B of Ex_i = a * x_i + b
    std::pair<T, T> Estimate() const {
        if (NonTrend)
            return std::make_pair(T(0), SumY / N);

        T a = (N * SumXY - SumX * SumY) / (N * SumSqX - SumX * SumX);
        T b = (SumY - SumX * a) / N;
        return std::make_pair(a, b);
    }
};

template <class TEl, class TRes>
class TRobustTrendyRandVarParamsEstimator {
    yvector<TRes> Values;

    TRes Frac;
    TRes Thresh;

    bool NonTrend; // disable flag

    bool Estimated;
    TRes Mean;
    TRes Var;

    size_t NEst;

private:
    class TCompByGreaterValues {
        const yvector<TRes>& Values;

    public:
        TCompByGreaterValues(const yvector<TRes>& values)
            : Values(values)
        {
        }

        bool operator()(size_t i1, size_t i2) const {
            Y_ASSERT((i1 < Values.size()) && (i2 < Values.size()));
            return Values[i1] > Values[i2];
        }
    };

    TRes ComputeVar(size_t nonZeroI, TRes a, TRes b,
                    const yvector<bool>* isOutlier = nullptr) // starts from nonZeroI
    {
        TRes sumDiff = 0;
        size_t i = nonZeroI;
        TRes N = 0;
        for (; i < Values.size(); ++i) {
            if (isOutlier && (*isOutlier)[i - nonZeroI])
                continue;

            TRes diff = Values[i] - a * TRes(i - nonZeroI) - b;
            sumDiff += diff * diff;
            ++N;
        }
        Y_ASSERT(N > 1);
        return sumDiff / (N - 1);
    }

    void EstimateNonRobust(size_t nonZeroI) {
        TEqSpacedXLinTrendEstimator<TRes> est(NonTrend);
        for (size_t i = nonZeroI; i < Values.size(); ++i) {
            est.Add(Values[i]);
        }
        TRes a, b;
        std::tie(a, b) = est.Estimate();
        NEst = Values.size() - nonZeroI;
        Var = ComputeVar(nonZeroI, a, b);
        Mean = a * TRes(NEst) + b;
    }

    void EstimateRobust(size_t nonZeroI, size_t maxOutliers) {
        size_t nonZeroEls = Values.size() - nonZeroI;

        yvector<bool> isOutlier(nonZeroEls, false); // for nonZeroEls only

        TLinTrendEstimator<TRes> est(NonTrend);

        yvector<size_t> outIndVec(nonZeroEls - 1);

        size_t i = nonZeroI;
        for (; i < Values.size() - 1; ++i) { // can't check last one for outliers
            outIndVec[i - nonZeroI] = i;
            est.Add(TRes(i - nonZeroI), Values[i]);
        }
        est.Add(TRes(i - nonZeroI), Values[i]);

        std::partial_sort(outIndVec.begin(), outIndVec.begin() + maxOutliers, outIndVec.end(),
                          TCompByGreaterValues(Values));

        std::reverse(outIndVec.begin(), outIndVec.begin() + maxOutliers);

        size_t nOutliers = maxOutliers;
        // set all to true
        for (size_t oi = 0; oi < maxOutliers; ++oi) {
            size_t outInd = outIndVec[oi];
            TRes outX = TRes(outInd - nonZeroI);
            TRes outY = Values[outInd];
            isOutlier[outInd - nonZeroI] = true;
            est.Sub(outX, outY);
        }

        TRes a, b;
        std::tie(a, b) = est.Estimate();
        Var = ComputeVar(nonZeroI, a, b, &isOutlier);

        while (true) {
            size_t foundNonOutliers = 0;
            for (size_t oi = 0; oi < maxOutliers; ++oi) {
                size_t outInd = outIndVec[oi];
                if (!isOutlier[outInd - nonZeroI])
                    continue;

                TRes outX = TRes(outInd - nonZeroI);
                TRes outY = Values[outInd];

                if (outY < a * outX + b + Thresh * sqrt(Var)) {
                    isOutlier[outInd - nonZeroI] = false;
                    est.Add(outX, outY);
                    ++foundNonOutliers;
                }
            }
            if (foundNonOutliers != 0) {
                std::tie(a, b) = est.Estimate();
                Var = ComputeVar(nonZeroI, a, b, &isOutlier);
                nOutliers -= foundNonOutliers;
                if (nOutliers == 0)
                    break;
            } else {
                break;
            }
        }
        Mean = a * TRes(nonZeroEls) + b;
        NEst = nonZeroEls - nOutliers;
    }

    void Estimate() // Ex_i = a * x_i + b (for last non-zero part)
    {
        size_t nonZeroI = 0;
        for (; nonZeroI < Values.size(); ++nonZeroI) {
            if (!FuzzyEquals(TRes(1) + Values[nonZeroI], TRes(1)))
                break;
        }
        size_t nonZeroEls = Values.size() - nonZeroI;
        switch (nonZeroEls) {
            case 0:
                Mean = 0;
                NEst = 0;
            case 1:
                Mean = TRes(Values.back());
                Var = std::numeric_limits<TRes>::infinity();
                NEst = 1;
                break;
            case 2: // perfect trend? better be cautious and assume random
            {
                TRes v1 = TRes(Values[Values.size() - 2]);
                TRes v2 = TRes(Values.back());
                Mean = TRes(0.5) * (v1 + v2);
                TRes diff = v1 - v2;
                Var = TRes(0.5) * diff * diff;
                NEst = 2;
            } break;
            default: {
                size_t maxOutliers = size_t(Frac * TRes(nonZeroEls));

                if ((nonZeroEls < 4) || (maxOutliers == 0)) {
                    EstimateNonRobust(nonZeroI);
                } else {
                    EstimateRobust(nonZeroI, maxOutliers);
                }
            }
        }
        Estimated = true;
    }

public:
    TRobustTrendyRandVarParamsEstimator(TRes frac = 0.3, TRes thresh = 3.0, bool nonTrend = false, size_t reserve = 0)
        : Frac(frac)
        , Thresh(thresh)
        , NonTrend(nonTrend)
        , Estimated(false)
    {
        Values.reserve(reserve);
    }

    void Add(TEl el) {
        Y_ASSERT(!Estimated);
        Values.push_back(TRes(el));
    }

    void Add(const yvector<TEl>& els) {
        Y_ASSERT(!Estimated);
        Values.insert(Values.end(), els.begin(), els.end());
    }

    ui64 GetNEls() const {
        return Values.size();
    }

    TRes E() {
        if (!Estimated)
            Estimate();
        return Mean;
    }

    TRes D() {
        if (!Estimated)
            Estimate();
        return Var;
    }

    size_t N() {
        if (!Estimated)
            Estimate();
        return NEst;
    }
};
