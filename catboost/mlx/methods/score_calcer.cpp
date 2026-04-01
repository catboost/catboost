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

        // For oblivious trees, pick the single best split across all partitions.
        // Score = sum of L2 split gain over all partitions.

        float bestGain = -std::numeric_limits<float>::infinity();

        // The Metal histogram kernel produces per-bin sums with a +1 offset:
        //   hist[firstFold + b] = sum of docs where featureValue == b+1
        //
        // For OneHot features: each bin represents one category.
        //   Split "go right if value == bin": sumRight = hist[bin], sumLeft = total - sumRight
        //
        // For ordinal features: we need suffix-sum to compute left/right.
        //   Split threshold b (value > b → right):
        //     sumRight = sum(hist[b..folds-1]) = docs with value ∈ {b+1,...,folds}
        //     sumLeft = total - sumRight

        for (ui32 featIdx = 0; featIdx < features.size(); ++featIdx) {
            const auto& feat = features[featIdx];
            for (ui32 bin = 0; bin < feat.Folds; ++bin) {
                float totalGain = 0.0f;

                for (ui32 p = 0; p < numPartitions; ++p) {
                    const float* partHist = histData + p * numStats * totalBinFeatures;

                    float totalSum = static_cast<float>(partitionStats[p].Sum);
                    float totalWeight = static_cast<float>(partitionStats[p].Weight);

                    float sumRight, weightRight;
                    if (feat.OneHotFeature) {
                        // OneHot: hist[bin] = docs where featureValue == bin+1
                        sumRight = partHist[feat.FirstFoldIndex + bin];
                        weightRight = (numStats > 1)
                            ? partHist[totalBinFeatures + feat.FirstFoldIndex + bin]
                            : 0.0f;
                    } else {
                        // Ordinal: suffix sum over hist[bin..folds-1]
                        sumRight = 0.0f;
                        weightRight = 0.0f;
                        for (ui32 b = bin; b < feat.Folds; ++b) {
                            sumRight += partHist[feat.FirstFoldIndex + b];
                            if (numStats > 1) {
                                weightRight += partHist[totalBinFeatures + feat.FirstFoldIndex + b];
                            }
                        }
                    }

                    float sumLeft = totalSum - sumRight;
                    float weightLeft = totalWeight - weightRight;

                    // Skip partitions where one side is empty — zero gain contribution
                    if (weightLeft < 1e-15f || weightRight < 1e-15f) continue;

                    totalGain += (sumLeft * sumLeft) / (weightLeft + l2RegLambda)
                               + (sumRight * sumRight) / (weightRight + l2RegLambda)
                               - (totalSum * totalSum) / (totalWeight + l2RegLambda);
                }

                if (totalGain > bestGain) {
                    bestGain = totalGain;
                    bestSplit.FeatureId = featIdx;
                    bestSplit.BinId = bin;
                    bestSplit.Gain = totalGain;
                    bestSplit.Score = -totalGain;
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
                float totalGain = 0.0f;

                for (ui32 p = 0; p < numPartitions; ++p) {
                    for (ui32 k = 0; k < approxDim; ++k) {
                        const ui32 numStats = perDimHistograms[k].NumStats;
                        const float* partHist = histDataPtrs[k] + p * numStats * totalBinFeatures;

                        float totalSum = static_cast<float>(perDimPartStats[k][p].Sum);
                        float totalWeight = static_cast<float>(perDimPartStats[k][p].Weight);

                        float sumRight, weightRight;
                        if (feat.OneHotFeature) {
                            sumRight = partHist[feat.FirstFoldIndex + bin];
                            weightRight = (numStats > 1)
                                ? partHist[totalBinFeatures + feat.FirstFoldIndex + bin]
                                : 0.0f;
                        } else {
                            sumRight = 0.0f;
                            weightRight = 0.0f;
                            for (ui32 b = bin; b < feat.Folds; ++b) {
                                sumRight += partHist[feat.FirstFoldIndex + b];
                                if (numStats > 1) {
                                    weightRight += partHist[totalBinFeatures + feat.FirstFoldIndex + b];
                                }
                            }
                        }

                        float sumLeft = totalSum - sumRight;
                        float weightLeft = totalWeight - weightRight;

                        if (weightLeft < 1e-15f || weightRight < 1e-15f) continue;

                        totalGain += (sumLeft * sumLeft) / (weightLeft + l2RegLambda)
                                   + (sumRight * sumRight) / (weightRight + l2RegLambda)
                                   - (totalSum * totalSum) / (totalWeight + l2RegLambda);
                    }
                }

                if (totalGain > bestGain) {
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
