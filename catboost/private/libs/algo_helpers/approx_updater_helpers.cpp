#include "approx_updater_helpers.h"

#include <catboost/libs/helpers/map_merge.h>


using namespace NCB;

static void NormalizeLeafValues(const TVector<double>& leafWeightsSum, TVector<double>* leafValues) {
    double averageLeafValue = 0;
    for (size_t leafIdx : xrange(leafWeightsSum.size())) {
        averageLeafValue += (*leafValues)[leafIdx] * leafWeightsSum[leafIdx];
    }
    averageLeafValue /= Accumulate(leafWeightsSum, /*val*/0.0);
    for (size_t leafIdx : xrange(leafWeightsSum.size())) {
        if (abs(leafWeightsSum[leafIdx]) > 1e-9) {
            (*leafValues)[leafIdx] -= averageLeafValue;
        } else {
           (*leafValues)[leafIdx] = 0;
        }
    }
}

void NormalizeLeafValues(
    bool isPairwise,
    double learningRate,
    const TVector<double>& leafWeightsSum,
    TVector<TVector<double>>* treeValues
) {
    if (isPairwise) {
        NormalizeLeafValues(leafWeightsSum, &(*treeValues)[0]);
    }

    for (auto& treeDimension : *treeValues) {
        for (auto& leafValue : treeDimension) {
            leafValue *= learningRate;
        }
    }
}

void InitApproxes(
    int size,
    const TMaybe<TVector<double>>& startingApprox,
    double approxDimension,
    bool storeExpApproxes,
    TVector<TVector<double>>* approx
) {
    approx->resize(approxDimension);
    Y_ASSERT(!startingApprox.Defined() || startingApprox->ysize() == approxDimension);
    for (auto dim : xrange(approxDimension)) {
        (*approx)[dim].resize(
            size,
            startingApprox ? ExpApproxIf(storeExpApproxes, (*startingApprox)[dim]) : GetNeutralApprox(storeExpApproxes)
        );
    }
}

TVector<double> SumLeafWeights(
    size_t leafCount,
    TConstArrayRef<TIndexType> leafIndices,
    TConstArrayRef<ui32> learnPermutation,
    TConstArrayRef<float> learnWeights, // can be empty
    NPar::ILocalExecutor* localExecutor
) {
    TVector<double> weightSum;
    NCB::MapMerge(
        localExecutor,
        TSimpleIndexRangesGenerator(TIndexRange(learnPermutation.ysize()), /*blockSize*/10000),
        /*map*/[=] (const auto& range, TVector<double>* output) {
            output->resize(leafCount);
            for (auto docIdx : range.Iter()) {
                (*output)[leafIndices[learnPermutation[docIdx]]] += learnWeights.empty() ? 1.0 : learnWeights[docIdx];
            }
        },
        /*merge*/[] (TVector<double>* weightSum, TVector<TVector<double>>&& outputs) {
            for (const auto& output : outputs) {
                AddElementwise(output, weightSum);
            }
        },
        &weightSum);
    return weightSum;
}
