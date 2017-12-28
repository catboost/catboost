#pragma once

#include <util/generic/ymath.h>
#include <util/generic/algorithm.h>
#include <util/generic/vector.h>

namespace NSelectionRankAucMetric
{

struct TDocWithSize {
    int Idx;
    int SortedIdx;
    double Target;
    double EffTarget;
    double Approx;
    double Size;
    double EffSize;
    double SumSizesAfter;
    double SumTargetsAfter;

    TDocWithSize(int idx, double target, double effTarget, double approx, double size, double effSize)
        : Idx(idx)
        , SortedIdx(idx)
        , Target(target)
        , EffTarget(effTarget)
        , Approx(approx)
        , Size(size)
        , EffSize(effSize)
        , SumSizesAfter(0)
        , SumTargetsAfter(0)
    {
    }

    static bool ApproxGreater(const TDocWithSize& a, const TDocWithSize& b) {
        if (a.Approx != b.Approx) {
            return a.Approx > b.Approx;
        } else if (a.EffTarget != b.EffTarget) {
            return a.EffTarget < b.EffTarget;
        }
        return a.Size > b.Size;
    }

    static bool QualityGreater(const TDocWithSize& a, const TDocWithSize& b) {
        return (a.EffTarget / Max<double>(a.EffSize, 1e-8)) > (b.EffTarget / Max<double>(b.EffSize, 1e-8));
    }

    static bool IdealTargetGreater(const TDocWithSize& a, const TDocWithSize& b) {
        if (a.EffTarget != b.EffTarget) {
            return a.EffTarget > b.EffTarget;
        }
        return a.Size < b.Size;
    }
};

template <typename TSortPred>
TVector<TDocWithSize> FillDocsWithSizes(
        int docsInQuery,
        const double* approx,
        const double* target,
        const double* docWeights,
        TSortPred sortPred)
{
    TVector<TDocWithSize> docs;
    docs.reserve(docsInQuery);

    for (int i = 0; i < docsInQuery; ++i) {
        docs.emplace_back(i, target[i], Abs(target[i]), approx[i], docWeights[i], docWeights[i]);
    }

    Sort(docs.begin(), docs.end(), sortPred);

    double sumSizes = 0;
    double sumTargets = 0;
    for (int i = 0; i < docs.ysize(); ++i) {
        int revIdx = docs.ysize() - i - 1;
        docs[revIdx].SortedIdx = revIdx;
        docs[revIdx].SumSizesAfter = sumSizes;
        docs[revIdx].SumTargetsAfter = sumTargets;
        sumTargets += docs[revIdx].EffTarget;
        sumSizes += docs[revIdx].EffSize;
    }

    for (auto& doc : docs) {
        doc.Size /= sumSizes;
        doc.EffSize /= sumSizes;
        doc.SumSizesAfter /= sumSizes;
    }

    return docs;
}

double CalcAuc(const TVector<TDocWithSize>& docs) {
    double result = 0;
    for (const auto& doc : docs) {
        result += doc.EffTarget * doc.SumSizesAfter;
    }
    return result;
}

double CalcQueryError(
    int docsInQuery,
    const double* approx,
    const double* target,
    const double* docWeights)
{
    auto idealDocs = FillDocsWithSizes(docsInQuery, approx, target, docWeights, TDocWithSize::QualityGreater);
    auto docs = FillDocsWithSizes(docsInQuery, approx, target, docWeights, TDocWithSize::ApproxGreater);
    return CalcAuc(docs) / CalcAuc(idealDocs);
}

}
