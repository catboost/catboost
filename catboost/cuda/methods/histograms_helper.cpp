#include "histograms_helper.h"
#include <util/system/env.h>

bool IsReduceCompressed() {
    static const bool reduceCompressed = GetEnv("CB_COMPRESSED_REDUCE", "false") == "true";
    return reduceCompressed;
}

namespace NCatboostCuda {
    template class TComputeHistogramsHelper<TFeatureParallelLayout>;
    template class TComputeHistogramsHelper<TDocParallelLayout>;
    template class TComputeHistogramsHelper<TSingleDevLayout>;

    template class TFindBestSplitsHelper<TFeatureParallelLayout>;
    template class TFindBestSplitsHelper<TDocParallelLayout>;
    template class TFindBestSplitsHelper<TSingleDevLayout>;

    template class TScoreHelper<TFeatureParallelLayout>;
    template class TScoreHelper<TDocParallelLayout>;
    template class TScoreHelper<TSingleDevLayout>;

    TFindBestSplitsHelper<TDocParallelLayout>& TFindBestSplitsHelper<TDocParallelLayout>::ComputeOptimalSplit(
        const TMirrorBuffer<const TPartitionStatistics>& reducedStats,
        const TMirrorBuffer<const float>& catFeatureWeights,
        const TMirrorBuffer<const float>& featureWeights,
        double scoreBeforeSplit,
        TComputeHistogramsHelper<TDocParallelLayout>& histHelper, double scoreStdDev, ui64 seed) {

        CB_ENSURE(histHelper.GetGroupingPolicy() == Policy);
        auto& profiler = NCudaLib::GetProfiler();
        if (DataSet->GetGridSize(Policy)) {
            const ui32 leavesCount = reducedStats.GetObjectsSlice().Size();
            const auto& binFeatures = DataSet->GetBinFeaturesForBestSplits(Policy);
            const auto streamId = Stream;

            if (NCudaLib::GetCudaManager().GetDeviceCount() == 1) {
                //shortcut to fast search
                const auto& histogram = histHelper.GetHistograms(streamId);

                auto guard = profiler.Profile(TStringBuilder() << "Find optimal split for #" << DataSet->GetBinFeatures(Policy).size());
                FindOptimalSplit(binFeatures,
                                 catFeatureWeights,
                                 featureWeights,
                                 histogram,
                                 reducedStats,
                                 FoldCount,
                                 scoreBeforeSplit,
                                 BestScores,
                                 ScoreFunction,
                                 L2,
                                 MetaL2Exponent,
                                 MetaL2Frequency,
                                 Normalize,
                                 scoreStdDev,
                                 seed,
                                 false /* gathered by leaves */,
                                 streamId);
            } else {
                //otherwise reduce-scatter histograms
                auto reducedMapping = binFeatures.GetMapping().Transform([&](const TSlice& binFeatures) {
                    return leavesCount * FoldCount * binFeatures.Size() * 2;
                });
                histHelper.GatherHistogramsByLeaves(ReducedHistograms, streamId);
                {
                    auto guard = profiler.Profile(TStringBuilder() << "Reduce " << ReducedHistograms.GetObjectsSlice().Size() << " histograms");
                    ReduceScatter(ReducedHistograms,
                                  reducedMapping,
                                  IsReduceCompressed(),
                                  streamId);
                }

                auto guard = profiler.Profile(
                    TStringBuilder() << "Find optimal split for #" << DataSet->GetBinFeatures(Policy).size());
                FindOptimalSplit(binFeatures,
                                 catFeatureWeights,
                                 featureWeights,
                                 ReducedHistograms,
                                 reducedStats,
                                 FoldCount,
                                 scoreBeforeSplit,
                                 BestScores,
                                 ScoreFunction,
                                 L2,
                                 MetaL2Exponent,
                                 MetaL2Frequency,
                                 Normalize,
                                 scoreStdDev,
                                 seed,
                                 true /*gathered by leaves */,
                                 streamId);
            }
        }
        return *this;
    }
}
