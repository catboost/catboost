#include "score_calcer.h"

#include <catboost/mlx/kernels/kernel_sources.h>
#include <catboost/libs/logging/logging.h>
#include <mlx/mlx.h>
#include <mlx/fast.h>

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

    TBestSplitProperties FindBestSplitGPU(
        const TVector<THistogramResult>& perDimHistograms,
        const TVector<TVector<TPartitionStatistics>>& perDimPartStats,
        const TVector<TCFeature>& features,
        float l2RegLambda,
        ui32 numPartitions
    ) {
        const ui32 approxDim = perDimHistograms.size();
        const ui32 totalBinFeatures = perDimHistograms[0].TotalBinFeatures;
        const ui32 numStats = perDimHistograms[0].NumStats;
        const ui32 numFeatures = features.size();

        if (totalBinFeatures == 0 || numPartitions == 0 || numFeatures == 0) {
            return TBestSplitProperties();
        }

        // Build feature metadata arrays
        TVector<uint32_t> firstFoldVec(numFeatures);
        TVector<uint32_t> foldsVec(numFeatures);
        TVector<uint32_t> isOneHotVec(numFeatures);
        for (ui32 f = 0; f < numFeatures; ++f) {
            firstFoldVec[f] = features[f].FirstFoldIndex;
            foldsVec[f] = features[f].Folds;
            isOneHotVec[f] = features[f].OneHotFeature ? 1u : 0u;
        }
        auto firstFoldArr = mx::array(
            reinterpret_cast<const int32_t*>(firstFoldVec.data()),
            {static_cast<int>(numFeatures)}, mx::uint32);
        auto foldsArr = mx::array(
            reinterpret_cast<const int32_t*>(foldsVec.data()),
            {static_cast<int>(numFeatures)}, mx::uint32);
        auto isOneHotArr = mx::array(
            reinterpret_cast<const int32_t*>(isOneHotVec.data()),
            {static_cast<int>(numFeatures)}, mx::uint32);

        // Stack per-dimension histograms: [approxDim * numPartitions * numStats * totalBinFeatures]
        mx::array stackedHist;
        if (approxDim == 1) {
            stackedHist = perDimHistograms[0].Histograms;
        } else {
            TVector<mx::array> histParts;
            for (ui32 k = 0; k < approxDim; ++k) {
                histParts.push_back(perDimHistograms[k].Histograms);
            }
            stackedHist = mx::concatenate(histParts, 0);
        }

        // Build partition stats arrays: [approxDim * numPartitions]
        TVector<float> partSumVec(approxDim * numPartitions);
        TVector<float> partWeightVec(approxDim * numPartitions);
        for (ui32 k = 0; k < approxDim; ++k) {
            for (ui32 p = 0; p < numPartitions; ++p) {
                partSumVec[k * numPartitions + p] = static_cast<float>(perDimPartStats[k][p].Sum);
                partWeightVec[k * numPartitions + p] = static_cast<float>(perDimPartStats[k][p].Weight);
            }
        }
        auto partSumArr = mx::array(partSumVec.data(),
            {static_cast<int>(approxDim * numPartitions)}, mx::float32);
        auto partWeightArr = mx::array(partWeightVec.data(),
            {static_cast<int>(approxDim * numPartitions)}, mx::float32);

        auto numFeatArr = mx::array(static_cast<uint32_t>(numFeatures), mx::uint32);
        auto totalBinsArr = mx::array(static_cast<uint32_t>(totalBinFeatures), mx::uint32);
        auto numStatsArr = mx::array(static_cast<uint32_t>(numStats), mx::uint32);
        auto numPartsArr = mx::array(static_cast<uint32_t>(numPartitions), mx::uint32);
        auto approxDimArr = mx::array(static_cast<uint32_t>(approxDim), mx::uint32);
        auto l2Arr = mx::array(l2RegLambda, mx::float32);

        // ===== Phase A: Suffix-sum transform =====
        // Each thread handles one (feature, partition, stat) triple
        auto suffixKernel = mx::fast::metal_kernel(
            "suffix_sum_histogram",
            /*input_names=*/{"histogram", "featureFirstFold", "featureFolds", "featureIsOneHot",
                "numFeatures", "totalBinFeatures", "numStats"},
            /*output_names=*/{"histogram_out"},
            /*source=*/KernelSources::kSuffixSumSource,
            /*header=*/KernelSources::kScoreHeader,
            /*ensure_row_contiguous=*/true,
            /*atomic_outputs=*/false
        );

        auto suffixGrid = std::make_tuple(
            static_cast<int>(numFeatures),
            static_cast<int>(approxDim * numPartitions),
            static_cast<int>(numStats)
        );
        auto suffixTG = std::make_tuple(1, 1, 1);

        auto suffixResult = suffixKernel(
            /*inputs=*/{stackedHist, firstFoldArr, foldsArr, isOneHotArr,
                numFeatArr, totalBinsArr, numStatsArr},
            /*output_shapes=*/{stackedHist.shape()},
            /*output_dtypes=*/{mx::float32},
            suffixGrid, suffixTG,
            /*template_args=*/{},
            /*init_value=*/std::nullopt,
            /*verbose=*/false,
            /*stream=*/mx::Device::gpu
        );
        auto transformedHist = suffixResult[0];

        // ===== Phase B: Score + reduce =====
        const ui32 numBlocks = (totalBinFeatures + 255) / 256;

        auto scoreKernel = mx::fast::metal_kernel(
            "score_splits",
            /*input_names=*/{"histogram", "partTotalSum", "partTotalWeight",
                "featureFirstFold", "featureFolds", "featureIsOneHot",
                "numFeatures", "totalBinFeatures", "numStats", "l2RegLambda",
                "numPartitions", "approxDim"},
            /*output_names=*/{"bestScores", "bestFeatureIds", "bestBinIds"},
            /*source=*/KernelSources::kScoreSplitsSource,
            /*header=*/KernelSources::kScoreHeader,
            /*ensure_row_contiguous=*/true,
            /*atomic_outputs=*/false
        );

        auto scoreGrid = std::make_tuple(
            static_cast<int>(256 * numBlocks), 1, 1
        );
        auto scoreTG = std::make_tuple(256, 1, 1);

        auto scoreResult = scoreKernel(
            /*inputs=*/{transformedHist, partSumArr, partWeightArr,
                firstFoldArr, foldsArr, isOneHotArr,
                numFeatArr, totalBinsArr, numStatsArr, l2Arr,
                numPartsArr, approxDimArr},
            /*output_shapes=*/{{static_cast<int>(numBlocks)},
                               {static_cast<int>(numBlocks)},
                               {static_cast<int>(numBlocks)}},
            /*output_dtypes=*/{mx::float32, mx::uint32, mx::uint32},
            scoreGrid, scoreTG,
            /*template_args=*/{},
            /*init_value=*/std::nullopt,
            /*verbose=*/false,
            /*stream=*/mx::Device::gpu
        );

        // ===== CPU final reduction over blocks =====
        auto bestScoresArr = scoreResult[0];
        auto bestFeatArr = scoreResult[1];
        auto bestBinArr = scoreResult[2];
        TMLXDevice::EvalNow({bestScoresArr, bestFeatArr, bestBinArr});

        const float* scores = bestScoresArr.data<float>();
        const uint32_t* featIds = bestFeatArr.data<uint32_t>();
        const uint32_t* binIds = bestBinArr.data<uint32_t>();

        TBestSplitProperties bestSplit;
        float bestGain = -std::numeric_limits<float>::infinity();
        for (ui32 i = 0; i < numBlocks; ++i) {
            if (scores[i] > bestGain) {
                bestGain = scores[i];
                bestSplit.FeatureId = featIds[i];
                bestSplit.BinId = binIds[i];
                bestSplit.Gain = bestGain;
                bestSplit.Score = -bestGain;
            }
        }

        if (bestSplit.Defined()) {
            CATBOOST_DEBUG_LOG << "CatBoost-MLX: FindBestSplitGPU (K="
                << approxDim << "): feature=" << bestSplit.FeatureId
                << " bin=" << bestSplit.BinId << " gain=" << bestSplit.Gain << Endl;
        }

        return bestSplit;
    }

    TBestSplitProperties FindBestSplitGPU(
        const TVector<THistogramResult>& perDimHistograms,
        const mx::array& partGradSums,
        const mx::array& partHessSums,
        const TVector<TCFeature>& features,
        float l2RegLambda,
        ui32 numPartitions,
        ui32 approxDim
    ) {
        const ui32 totalBinFeatures = perDimHistograms[0].TotalBinFeatures;
        const ui32 numStats = perDimHistograms[0].NumStats;
        const ui32 numFeatures = features.size();

        if (totalBinFeatures == 0 || numPartitions == 0 || numFeatures == 0) {
            return TBestSplitProperties();
        }

        // Build feature metadata arrays
        TVector<uint32_t> firstFoldVec(numFeatures);
        TVector<uint32_t> foldsVec(numFeatures);
        TVector<uint32_t> isOneHotVec(numFeatures);
        for (ui32 f = 0; f < numFeatures; ++f) {
            firstFoldVec[f] = features[f].FirstFoldIndex;
            foldsVec[f] = features[f].Folds;
            isOneHotVec[f] = features[f].OneHotFeature ? 1u : 0u;
        }
        auto firstFoldArr = mx::array(
            reinterpret_cast<const int32_t*>(firstFoldVec.data()),
            {static_cast<int>(numFeatures)}, mx::uint32);
        auto foldsArr = mx::array(
            reinterpret_cast<const int32_t*>(foldsVec.data()),
            {static_cast<int>(numFeatures)}, mx::uint32);
        auto isOneHotArr = mx::array(
            reinterpret_cast<const int32_t*>(isOneHotVec.data()),
            {static_cast<int>(numFeatures)}, mx::uint32);

        // Build precomputed bin-to-feature lookup table (OPT-2)
        TVector<uint32_t> binToFeatureVec(totalBinFeatures);
        for (ui32 f = 0; f < numFeatures; ++f) {
            const ui32 start = features[f].FirstFoldIndex;
            const ui32 end = start + features[f].Folds;
            for (ui32 b = start; b < end; ++b) {
                binToFeatureVec[b] = f;
            }
        }
        auto binToFeatureArr = mx::array(
            reinterpret_cast<const int32_t*>(binToFeatureVec.data()),
            {static_cast<int>(totalBinFeatures)}, mx::uint32);

        // Stack per-dimension histograms: [approxDim * numPartitions * numStats * totalBinFeatures]
        mx::array stackedHist;
        if (approxDim == 1) {
            stackedHist = perDimHistograms[0].Histograms;
        } else {
            TVector<mx::array> histParts;
            for (ui32 k = 0; k < approxDim; ++k) {
                histParts.push_back(perDimHistograms[k].Histograms);
            }
            stackedHist = mx::concatenate(histParts, 0);
        }

        auto numFeatArr    = mx::array(static_cast<uint32_t>(numFeatures), mx::uint32);
        auto totalBinsArr  = mx::array(static_cast<uint32_t>(totalBinFeatures), mx::uint32);
        auto numStatsArr   = mx::array(static_cast<uint32_t>(numStats), mx::uint32);
        auto numPartsArr   = mx::array(static_cast<uint32_t>(numPartitions), mx::uint32);
        auto approxDimArr  = mx::array(static_cast<uint32_t>(approxDim), mx::uint32);
        auto l2Arr         = mx::array(l2RegLambda, mx::float32);

        // ===== Phase A: Suffix-sum transform =====
        auto suffixKernel = mx::fast::metal_kernel(
            "suffix_sum_histogram",
            /*input_names=*/{"histogram", "featureFirstFold", "featureFolds", "featureIsOneHot",
                "numFeatures", "totalBinFeatures", "numStats"},
            /*output_names=*/{"histogram_out"},
            /*source=*/KernelSources::kSuffixSumSource,
            /*header=*/KernelSources::kScoreHeader,
            /*ensure_row_contiguous=*/true,
            /*atomic_outputs=*/false
        );

        auto suffixGrid = std::make_tuple(
            static_cast<int>(numFeatures),
            static_cast<int>(approxDim * numPartitions),
            static_cast<int>(numStats)
        );
        auto suffixTG = std::make_tuple(1, 1, 1);

        auto suffixResult = suffixKernel(
            /*inputs=*/{stackedHist, firstFoldArr, foldsArr, isOneHotArr,
                numFeatArr, totalBinsArr, numStatsArr},
            /*output_shapes=*/{stackedHist.shape()},
            /*output_dtypes=*/{mx::float32},
            suffixGrid, suffixTG,
            /*template_args=*/{},
            /*init_value=*/std::nullopt,
            /*verbose=*/false,
            /*stream=*/mx::Device::gpu
        );
        auto transformedHist = suffixResult[0];

        // ===== Phase B: Score + reduce (with binToFeature lookup) =====
        const ui32 numBlocks = (totalBinFeatures + 255) / 256;

        auto scoreKernel = mx::fast::metal_kernel(
            "score_splits_lookup",
            /*input_names=*/{"histogram", "partTotalSum", "partTotalWeight",
                "featureFirstFold", "featureFolds", "featureIsOneHot", "binToFeature",
                "numFeatures", "totalBinFeatures", "numStats", "l2RegLambda",
                "numPartitions", "approxDim"},
            /*output_names=*/{"bestScores", "bestFeatureIds", "bestBinIds"},
            /*source=*/KernelSources::kScoreSplitsLookupSource,
            /*header=*/KernelSources::kScoreHeader,
            /*ensure_row_contiguous=*/true,
            /*atomic_outputs=*/false
        );

        auto scoreGrid = std::make_tuple(static_cast<int>(256 * numBlocks), 1, 1);
        auto scoreTG   = std::make_tuple(256, 1, 1);

        auto scoreResult = scoreKernel(
            /*inputs=*/{transformedHist, partGradSums, partHessSums,
                firstFoldArr, foldsArr, isOneHotArr, binToFeatureArr,
                numFeatArr, totalBinsArr, numStatsArr, l2Arr,
                numPartsArr, approxDimArr},
            /*output_shapes=*/{{static_cast<int>(numBlocks)},
                               {static_cast<int>(numBlocks)},
                               {static_cast<int>(numBlocks)}},
            /*output_dtypes=*/{mx::float32, mx::uint32, mx::uint32},
            scoreGrid, scoreTG,
            /*template_args=*/{},
            /*init_value=*/std::nullopt,
            /*verbose=*/false,
            /*stream=*/mx::Device::gpu
        );

        // ===== CPU final reduction over blocks =====
        auto bestScoresArr = scoreResult[0];
        auto bestFeatArr   = scoreResult[1];
        auto bestBinArr    = scoreResult[2];
        TMLXDevice::EvalNow({bestScoresArr, bestFeatArr, bestBinArr});

        const float*    scores  = bestScoresArr.data<float>();
        const uint32_t* featIds = bestFeatArr.data<uint32_t>();
        const uint32_t* binIds  = bestBinArr.data<uint32_t>();

        TBestSplitProperties bestSplit;
        float bestGain = -std::numeric_limits<float>::infinity();
        for (ui32 i = 0; i < numBlocks; ++i) {
            if (scores[i] > bestGain) {
                bestGain = scores[i];
                bestSplit.FeatureId = featIds[i];
                bestSplit.BinId = binIds[i];
                bestSplit.Gain = bestGain;
                bestSplit.Score = -bestGain;
            }
        }

        if (bestSplit.Defined()) {
            CATBOOST_DEBUG_LOG << "CatBoost-MLX: FindBestSplitGPU/lookup (K="
                << approxDim << "): feature=" << bestSplit.FeatureId
                << " bin=" << bestSplit.BinId << " gain=" << bestSplit.Gain << Endl;
        }

        return bestSplit;
    }

}  // namespace NCatboostMlx
