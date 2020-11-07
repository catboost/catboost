#include "build_subset_in_leaf.h"

TVector<TLeafStatistics> BuildSubset(
    TConstArrayRef<TIndexType> leafIndices,
    TConstArrayRef<TVector<double>> approx,
    TConstArrayRef<TVector<float>> labels,
    TConstArrayRef<float> weights,
    TConstArrayRef<float> sampleWeights,
    int leafCount,
    int sampleCount,
    double sumWeight,
    bool needSampleWeights,
    NPar::ILocalExecutor* localExecutor
) {
    const int approxDimension = approx.ysize();
    TVector<TLeafStatistics> leafStatistics(
        leafCount,
        TLeafStatistics(labels.size(), approxDimension, sampleCount, sumWeight));

    TVector<int> docIndices;
    docIndices.yresize(sampleCount);
    TVector<int> objectsCount(leafCount, 0);

    for (int idx = 0; idx < sampleCount; ++idx) {
        docIndices[idx] = objectsCount[leafIndices[idx]]++;
    }

    TVector<TArrayRef<TArrayRef<float>>> leafLabels(leafCount);
    TVector<TArrayRef<float>> leafWeights(leafCount);
    TVector<TArrayRef<float>> leafSampleWeights(leafCount);

    TVector<TConstArrayRef<double>> approxRef(approxDimension);
    TVector<TVector<TArrayRef<double>>> approxInLeafRef;
    approxInLeafRef.yresize(leafCount);
    for (int dimIdx = 0; dimIdx < approxDimension; ++dimIdx) {
        approxRef[dimIdx] = approx[dimIdx];
    }

    for (int leafIdx = 0; leafIdx < leafCount; ++leafIdx) {
        auto& statistics = leafStatistics[leafIdx];

        statistics.SetLeafIdx(leafIdx);
        statistics.Resize(
            objectsCount[leafIdx],
            needSampleWeights,
            !weights.empty());

        leafLabels[leafIdx] = statistics.GetLabels();
        leafWeights[leafIdx] = statistics.GetWeights();
        leafSampleWeights[leafIdx] = statistics.GetSampleWeights();

        approxInLeafRef[leafIdx].yresize(approxDimension);
        for (int dimIdx = 0; dimIdx < approxDimension; ++dimIdx) {
            approxInLeafRef[leafIdx][dimIdx] = statistics.GetApprox(dimIdx);
        }
    }

    NPar::ParallelFor(
        *localExecutor,
        0,
        sampleCount,
        [&](int docIdx) {
            const int leafIdx = leafIndices[docIdx];
            const int insideIdx = docIndices[docIdx];

            for (auto labelDim : xrange(leafLabels[leafIdx].size())) {
                leafLabels[leafIdx][labelDim][insideIdx] = labels[labelDim][docIdx];
            }
            if (!needSampleWeights && !weights.empty()) {
                leafWeights[leafIdx][insideIdx] = weights[docIdx];
            }
            if (needSampleWeights) {
                leafSampleWeights[leafIdx][insideIdx] = sampleWeights[docIdx];
            }

            for (int dimIdx = 0; dimIdx < approxDimension; ++dimIdx) {
                approxInLeafRef[leafIdx][dimIdx][insideIdx] = approxRef[dimIdx][docIdx];
            }
        });

    return leafStatistics;
}

TVector<TLeafStatistics> BuildSubset(
    TConstArrayRef<TIndexType> leafIndices,
    int leafCount,
    TLearnContext* ctx
) {
    return BuildSubset(
        leafIndices,
        ctx->LearnProgress->AveragingFold.BodyTailArr[0].Approx,
        ctx->LearnProgress->AveragingFold.LearnTarget,
        ctx->LearnProgress->AveragingFold.GetLearnWeights(),
        ctx->LearnProgress->AveragingFold.SampleWeights,
        leafCount,
        ctx->LearnProgress->AveragingFold.GetLearnSampleCount(),
        ctx->LearnProgress->AveragingFold.GetSumWeight(),
        ctx->Params.ObliviousTreeOptions->LeavesEstimationMethod == ELeavesEstimation::Exact,
        ctx->LocalExecutor);
}
