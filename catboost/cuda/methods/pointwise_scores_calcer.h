#pragma once

#include "histograms_helper.h"
#include "helpers.h"
#include "pointiwise_optimization_subsets.h"
#include <catboost/cuda/gpu_data/compressed_index.h>
#include <catboost/libs/options/oblivious_tree_options.h>

namespace NCatboostCuda {
    template <EFeaturesGroupingPolicy Policy,
              class TLayoutPolicy>
    inline THolder<TScoreHelper<Policy, TLayoutPolicy>> CreateScoreHelper(const TCompressedDataSet<TLayoutPolicy>& dataSet,
                                                                          ui32 foldCount,
                                                                          const NCatboostOptions::TObliviousTreeLearnerOptions& treeConfig,
                                                                          bool requestStream = false) {
        using TFeatureScoresHelper = TScoreHelper<Policy, TLayoutPolicy>;
        return MakeHolder<TFeatureScoresHelper>(dataSet,
                                                foldCount,
                                                treeConfig.MaxDepth,
                                                treeConfig.ScoreFunction,
                                                treeConfig.L2Reg,
                                                treeConfig.FoldSizeLossNormalization,
                                                requestStream);
    };

    template <class TLayoutPolicy = TFeatureParallelLayout>
    class TScoresCalcerOnCompressedDataSet {
    public:
        using TSamplesMapping = typename TLayoutPolicy::TSamplesMapping;

        TScoresCalcerOnCompressedDataSet(const TCompressedDataSet<TLayoutPolicy>& features,
                                         const NCatboostOptions::TObliviousTreeLearnerOptions& treeConfig,
                                         ui32 foldCount,
                                         bool requestStream = false)
            : Features(features)
            , TreeConfig(treeConfig)
            , FoldCount(foldCount)
        {
            if (Features.GetGridSize(EFeaturesGroupingPolicy::BinaryFeatures)) {
                BinaryFeatureHelper = CreateScoreHelper<EFeaturesGroupingPolicy::BinaryFeatures, TLayoutPolicy>(Features,
                                                                                                                foldCount,
                                                                                                                TreeConfig,
                                                                                                                requestStream);
            }
            if (Features.GetGridSize(EFeaturesGroupingPolicy::HalfByteFeatures)) {
                HalfByteFeatureHelper = CreateScoreHelper<EFeaturesGroupingPolicy::HalfByteFeatures, TLayoutPolicy>(Features,
                                                                                                                    foldCount,
                                                                                                                    TreeConfig,
                                                                                                                    requestStream);
            }
            if (Features.GetGridSize(EFeaturesGroupingPolicy::OneByteFeatures)) {
                ByteFeatureHelper = CreateScoreHelper<EFeaturesGroupingPolicy::OneByteFeatures, TLayoutPolicy>(Features,
                                                                                                               foldCount,
                                                                                                               TreeConfig,
                                                                                                               requestStream);
            }
        }

        bool HasByteFeatureHelper() const {
            return ByteFeatureHelper != nullptr;
        }

        bool HasBinaryFeatureHelper() const {
            return BinaryFeatureHelper != nullptr;
        }

        bool HasHalfByteFeatureHelper() const {
            return HalfByteFeatureHelper != nullptr;
        }

        const TScoreHelper<EFeaturesGroupingPolicy::OneByteFeatures, TLayoutPolicy>& GetByteFeatureHelper() const {
            return *ByteFeatureHelper;
        }

        const TScoreHelper<EFeaturesGroupingPolicy::BinaryFeatures, TLayoutPolicy>& GetBinaryFeatureHelper() const {
            return *BinaryFeatureHelper;
        }

        const TScoreHelper<EFeaturesGroupingPolicy::HalfByteFeatures, TLayoutPolicy>& GetHalfByteFeatureHelper() const {
            return *HalfByteFeatureHelper;
        }

        template <bool IsConst, class TUi32>
        TScoresCalcerOnCompressedDataSet& SubmitCompute(const TOptimizationSubsets<TSamplesMapping, IsConst>& newSubsets,
                                                        const TCudaBuffer<TUi32, TSamplesMapping>& docs) {
            if (BinaryFeatureHelper) {
                BinaryFeatureHelper->SubmitCompute(newSubsets,
                                                   docs);
            }
            if (HalfByteFeatureHelper) {
                HalfByteFeatureHelper->SubmitCompute(newSubsets,
                                                     docs);
            }
            if (ByteFeatureHelper) {
                ByteFeatureHelper->SubmitCompute(newSubsets,
                                                 docs);
            }
            return *this;
        }

        TScoresCalcerOnCompressedDataSet& ComputeOptimalSplit(const TCudaBuffer<const TPartitionStatistics, NCudaLib::TMirrorMapping>& partStats,
                                                              double scoreStdDev = 0,
                                                              ui64 seed = 0) {
            TRandom rand(seed);
            if (BinaryFeatureHelper) {
                BinaryFeatureHelper->ComputeOptimalSplit(partStats, scoreStdDev, rand.NextUniformL());
            }
            if (HalfByteFeatureHelper) {
                HalfByteFeatureHelper->ComputeOptimalSplit(partStats, scoreStdDev, rand.NextUniformL());
            }
            if (ByteFeatureHelper) {
                ByteFeatureHelper->ComputeOptimalSplit(partStats, scoreStdDev, rand.NextUniformL());
            }
            return *this;
        }

        TBestSplitProperties ReadOptimalSplit() {
            TBestSplitProperties best = {static_cast<ui32>(-1),
                                         0,
                                         std::numeric_limits<float>::infinity()};
            if (BinaryFeatureHelper) {
                best = TakeBest(BinaryFeatureHelper->ReadOptimalSplit(), best);
            }
            if (HalfByteFeatureHelper) {
                best = TakeBest(HalfByteFeatureHelper->ReadOptimalSplit(), best);
            }
            if (ByteFeatureHelper) {
                best = TakeBest(ByteFeatureHelper->ReadOptimalSplit(), best);
            }
            return best;
        }

    private:
        const TCompressedDataSet<TLayoutPolicy>& Features;
        const NCatboostOptions::TObliviousTreeLearnerOptions& TreeConfig;
        ui32 FoldCount;

        THolder<TScoreHelper<EFeaturesGroupingPolicy::BinaryFeatures, TLayoutPolicy>> BinaryFeatureHelper;
        THolder<TScoreHelper<EFeaturesGroupingPolicy::HalfByteFeatures, TLayoutPolicy>> HalfByteFeatureHelper;
        THolder<TScoreHelper<EFeaturesGroupingPolicy::OneByteFeatures, TLayoutPolicy>> ByteFeatureHelper;
    };

}
