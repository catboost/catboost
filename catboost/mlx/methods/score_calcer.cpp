#include "score_calcer.h"

#include <catboost/libs/logging/logging.h>

namespace NCatboostMlx {

    TBestSplitProperties FindBestSplit(
        const THistogramResult& histograms,
        const TVector<TPartitionStatistics>& partitionStats,
        const TVector<TCFeature>& features,
        float l2RegLambda,
        ui32 numPartitions
    ) {
        TBestSplitProperties bestSplit;

        // Read histogram data to CPU for scoring
        // Layout: [numPartitions, numStats, totalBinFeatures]
        TMLXDevice::EvalNow(histograms.Histograms);
        const float* histData = histograms.Histograms.data<float>();
        const ui32 numStats = histograms.NumStats;
        const ui32 totalBinFeatures = histograms.TotalBinFeatures;

        if (totalBinFeatures == 0 || numPartitions == 0) {
            return bestSplit;
        }

        // For oblivious trees, we pick the single best split across all partitions.
        // The score is the sum over all partition pairs of the L2 split gain.
        //
        // For each bin-feature candidate:
        //   For each partition p:
        //     sumLeft = histogram[p][0][binFeature]
        //     weightLeft = histogram[p][1][binFeature]  (or count if numStats==1)
        //     sumRight = partStats[p].Sum - sumLeft
        //     weightRight = partStats[p].Weight - weightLeft
        //     gain += sumLeft^2 / (weightLeft + lambda) + sumRight^2 / (weightRight + lambda)
        //           - partStats[p].Sum^2 / (partStats[p].Weight + lambda)

        float bestGain = -std::numeric_limits<float>::infinity();

        for (ui32 featIdx = 0; featIdx < features.size(); ++featIdx) {
            const auto& feat = features[featIdx];
            for (ui32 bin = 0; bin < feat.Folds; ++bin) {
                const ui32 binFeatureIdx = feat.FirstFoldIndex + bin;

                float totalGain = 0.0f;
                bool validSplit = true;

                for (ui32 p = 0; p < numPartitions; ++p) {
                    const float* partHist = histData + p * numStats * totalBinFeatures;

                    float sumLeft = partHist[binFeatureIdx];  // gradient histogram (stat=0)
                    float weightLeft;
                    if (numStats > 1) {
                        weightLeft = partHist[totalBinFeatures + binFeatureIdx];  // weight histogram (stat=1)
                    } else {
                        // Without weight histogram, we can't properly compute per-bin weights.
                        // This path is used for RMSE where hessian=1, so weight = doc count.
                        // TODO: Compute per-bin doc counts from the histogram
                        weightLeft = static_cast<float>(partitionStats[p].Weight) * 0.5f;
                    }

                    float totalSum = static_cast<float>(partitionStats[p].Sum);
                    float totalWeight = static_cast<float>(partitionStats[p].Weight);

                    float sumRight = totalSum - sumLeft;
                    float weightRight = totalWeight - weightLeft;

                    // Skip degenerate splits
                    if (weightLeft < 1e-15f || weightRight < 1e-15f) {
                        validSplit = false;
                        break;
                    }

                    // L2 gain: sum^2/(weight+lambda) for each side, minus parent
                    float gainLeft = (sumLeft * sumLeft) / (weightLeft + l2RegLambda);
                    float gainRight = (sumRight * sumRight) / (weightRight + l2RegLambda);
                    float gainParent = (totalSum * totalSum) / (totalWeight + l2RegLambda);

                    totalGain += gainLeft + gainRight - gainParent;
                }

                if (validSplit && totalGain > bestGain) {
                    bestGain = totalGain;
                    bestSplit.FeatureId = featIdx;
                    bestSplit.BinId = bin;
                    bestSplit.Gain = totalGain;
                    bestSplit.Score = -totalGain;  // Score is negative gain (lower = better)
                }
            }
        }

        if (bestSplit.Defined()) {
            CATBOOST_DEBUG_LOG << "CatBoost-MLX: FindBestSplit: feature=" << bestSplit.FeatureId
                << " bin=" << bestSplit.BinId << " gain=" << bestSplit.Gain << Endl;
        }

        return bestSplit;
    }

    TBestSplitProperties FindBestSplitMultiDim(
        const TVector<THistogramResult>& perDimHistograms,
        const TVector<TVector<TPartitionStatistics>>& perDimPartStats,
        const TVector<TCFeature>& features,
        float l2RegLambda,
        ui32 numPartitions
    ) {
        const ui32 approxDim = perDimHistograms.size();

        // Single-dim: delegate
        if (approxDim == 1) {
            return FindBestSplit(
                perDimHistograms[0], perDimPartStats[0],
                features, l2RegLambda, numPartitions
            );
        }

        TBestSplitProperties bestSplit;

        // Evaluate histograms for each dimension
        TVector<const float*> histDataPtrs(approxDim);
        TVector<ui32> totalBinFeaturesVec(approxDim);
        for (ui32 k = 0; k < approxDim; ++k) {
            TMLXDevice::EvalNow(perDimHistograms[k].Histograms);
            histDataPtrs[k] = perDimHistograms[k].Histograms.data<float>();
            totalBinFeaturesVec[k] = perDimHistograms[k].TotalBinFeatures;
        }

        const ui32 totalBinFeatures = totalBinFeaturesVec[0];
        if (totalBinFeatures == 0 || numPartitions == 0) {
            return bestSplit;
        }

        float bestGain = -std::numeric_limits<float>::infinity();

        for (ui32 featIdx = 0; featIdx < features.size(); ++featIdx) {
            const auto& feat = features[featIdx];
            for (ui32 bin = 0; bin < feat.Folds; ++bin) {
                const ui32 binFeatureIdx = feat.FirstFoldIndex + bin;

                float totalGain = 0.0f;
                bool validSplit = true;

                for (ui32 p = 0; p < numPartitions && validSplit; ++p) {
                    for (ui32 k = 0; k < approxDim && validSplit; ++k) {
                        const ui32 numStats = perDimHistograms[k].NumStats;
                        const float* partHist = histDataPtrs[k] + p * numStats * totalBinFeatures;

                        float sumLeft = partHist[binFeatureIdx];
                        float weightLeft = (numStats > 1) ?
                            partHist[totalBinFeatures + binFeatureIdx] :
                            static_cast<float>(perDimPartStats[k][p].Weight) * 0.5f;

                        float totalSum = static_cast<float>(perDimPartStats[k][p].Sum);
                        float totalWeight = static_cast<float>(perDimPartStats[k][p].Weight);

                        float sumRight = totalSum - sumLeft;
                        float weightRight = totalWeight - weightLeft;

                        if (weightLeft < 1e-15f || weightRight < 1e-15f) {
                            validSplit = false;
                            break;
                        }

                        float gainLeft = (sumLeft * sumLeft) / (weightLeft + l2RegLambda);
                        float gainRight = (sumRight * sumRight) / (weightRight + l2RegLambda);
                        float gainParent = (totalSum * totalSum) / (totalWeight + l2RegLambda);

                        totalGain += gainLeft + gainRight - gainParent;
                    }
                }

                if (validSplit && totalGain > bestGain) {
                    bestGain = totalGain;
                    bestSplit.FeatureId = featIdx;
                    bestSplit.BinId = bin;
                    bestSplit.Gain = totalGain;
                    bestSplit.Score = -totalGain;
                }
            }
        }

        if (bestSplit.Defined()) {
            CATBOOST_DEBUG_LOG << "CatBoost-MLX: FindBestSplitMultiDim (K="
                << approxDim << "): feature=" << bestSplit.FeatureId
                << " bin=" << bestSplit.BinId << " gain=" << bestSplit.Gain << Endl;
        }

        return bestSplit;
    }

}  // namespace NCatboostMlx
