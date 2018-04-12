#pragma once

#include "pointwise_kernels.h"
#include "pointiwise_optimization_subsets.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/reduce_scatter.h>
#include <catboost/cuda/gpu_data/feature_parallel_dataset.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/gpu_data/gpu_structures.h>

bool IsReduceCompressed();

namespace NCatboostCuda {
    template <EFeaturesGroupingPolicy Policy,
              class TLayoutPolicy = TFeatureParallelLayout>
    class TComputeHistogramsHelper: public TMoveOnly {
    public:
        using TGpuDataSet = typename TSharedCompressedIndex<TLayoutPolicy>::TCompressedDataSet;
        using TFeaturesMapping = typename TLayoutPolicy::TFeaturesMapping;
        using TSamplesMapping = typename TLayoutPolicy::TSamplesMapping;

    public:
        TComputeHistogramsHelper(const TGpuDataSet& dataSet,
                                 ui32 foldCount,
                                 ui32 maxDepth,
                                 TComputationStream& stream)
            : DataSet(&dataSet)
            , Stream(stream)
            , FoldCount(foldCount)
            , MaxDepth(maxDepth)
        {
        }

        template <bool IsConst, class TUi32>
        TComputeHistogramsHelper& Compute(const TOptimizationSubsets<TSamplesMapping, IsConst>& newSubsets,
                                          const TCudaBuffer<TUi32, TSamplesMapping>& docs) {
            Y_ASSERT(DataSet);
            ++CurrentBit;
            if (static_cast<ui32>(CurrentBit) != newSubsets.CurrentDepth || CurrentBit == 0) {
                BuildFromScratch = true;
                CurrentBit = newSubsets.CurrentDepth;
            }

            if (BuildFromScratch) {
                ResetHistograms();
            }

            {
                auto currentStripe = DataSet->GetHistogramsMapping(Policy).Transform([&](const TSlice& features) -> ui64 {
                    return (1 << CurrentBit) * FoldCount * features.Size() * 2;
                });
                Histograms.Reset(currentStripe);
            }

            if (DataSet->GetGridSize(Policy)) {
                auto& profiler = NCudaLib::GetProfiler();
                auto guard = profiler.Profile(TStringBuilder() << "Compute histograms (" << Policy << ") for  #" << DataSet->GetGridSize(Policy)
                                                               << " features, depth " << CurrentBit);

                ComputeHistogram2<Policy>(*DataSet,
                                          newSubsets.WeightedTarget,
                                          newSubsets.Weights,
                                          docs,
                                          newSubsets.Partitions,
                                          static_cast<ui32>(1 << CurrentBit),
                                          FoldCount,
                                          Histograms,
                                          BuildFromScratch,
                                          Stream.GetId());

                BuildFromScratch = false;
                Computing = true;
            }
            return *this;
        }

        const TCudaBuffer<float, TFeaturesMapping>& GetHistograms(const ui32 streamId) const {
            if (Stream.GetId() != streamId) {
                EnsureHistCompute();
            }
            return Histograms;
        };

        void GatherHistogramsByLeaves(TCudaBuffer<float, TFeaturesMapping>& gatheredHistogramsByLeaves, ui32 streamId) const {
            if (streamId != Stream.GetId()) {
                EnsureHistCompute();
            }
            auto currentStripe = DataSet->GetHistogramsMapping(Policy).Transform([&](const TSlice& features) -> ui64 {
                return (1 << CurrentBit) * FoldCount * features.Size() * 2;
            });
            gatheredHistogramsByLeaves.Reset(currentStripe);

            if (DataSet->GetGridSize(Policy)) {
                GatherHistogramByLeaves(Histograms,
                                        DataSet->GetBinFeatureCount(Policy),
                                        2,
                                        static_cast<ui32>(1 << (CurrentBit)),
                                        FoldCount,
                                        gatheredHistogramsByLeaves,
                                        streamId);
            }
        }

        TVector<float> ReadHistograms() const {
            TVector<float> dst;
            TCudaBuffer<float, TFeaturesMapping> gatheredHistogramsByLeaves;

            auto currentStripe = DataSet->GetHistogramsMapping(Policy).Transform([&](const TSlice& features) -> ui64 {
                return (1 << CurrentBit) * FoldCount * features.Size() * 2;
            });
            gatheredHistogramsByLeaves.Reset(currentStripe);

            if (DataSet->GetGridSize(Policy)) {
                GatherHistogramByLeaves(Histograms,
                                        DataSet->GetBinFeatureCount(Policy),
                                        2,
                                        static_cast<ui32>(1 << (CurrentBit)),
                                        FoldCount,
                                        gatheredHistogramsByLeaves,
                                        Stream.GetId());
            }
            gatheredHistogramsByLeaves.Read(dst);
            return dst;
        }

        const TCudaBuffer<float, TFeaturesMapping>& GetHistograms() const {
            EnsureHistCompute();
            return Histograms;
        }

    private:
        void EnsureHistCompute() const {
            if (Computing) {
                Stream.Synchronize();
                Computing = false;
            }
        }

        void ResetHistograms() {
            auto histMapping = DataSet->GetHistogramsMapping(Policy).Transform([&](const TSlice& histograms) -> ui64 {
                return (1 << MaxDepth) * FoldCount * histograms.Size() * 2;
            });

            Histograms.Reset(histMapping);
            FillBuffer(Histograms, 0.0f, Stream);
        }

    private:
        const TGpuDataSet* DataSet = nullptr;
        TComputationStream& Stream;

        ui32 FoldCount;
        ui32 MaxDepth;
        int CurrentBit = -1;
        bool BuildFromScratch = true;
        mutable bool Computing = false;
        TCudaBuffer<float, TFeaturesMapping> Histograms;
    };

    template <EFeaturesGroupingPolicy Policy,
              class TLayoutPolicy = TFeatureParallelLayout>
    class TFindBestSplitsHelper: public TMoveOnly {
    public:
        using TGpuDataSet = typename TSharedCompressedIndex<TLayoutPolicy>::TCompressedDataSet;
        using TFeaturesMapping = typename TLayoutPolicy::TFeaturesMapping;
        using TSamplesMapping = typename TLayoutPolicy::TSamplesMapping;

    public:
        TFindBestSplitsHelper(const TGpuDataSet& dataSet,
                              ui32 foldCount,
                              ui32 maxDepth,
                              EScoreFunction score = EScoreFunction::Correlation,
                              double l2 = 1.0,
                              bool normalize = false,
                              ui32 stream = 0)
            : DataSet(&dataSet)
            , Stream(stream)
            , FoldCount(foldCount)
            , MaxDepth(maxDepth)
            , ScoreFunction(score)
            , L2(l2)
            , Normalize(normalize)
        {
            if (DataSet->GetGridSize(Policy)) {
                const ui64 blockCount = 32;
                auto bestScoresMapping = dataSet.GetBestSplitStatsMapping(Policy).Transform([&](const TSlice& histograms) -> ui64 {
                    return std::min(NHelpers::CeilDivide(histograms.Size(), 128), blockCount);
                });

                BestScores.Reset(bestScoresMapping);
            }
        }

        TFindBestSplitsHelper& ComputeOptimalSplit(const TCudaBuffer<const TPartitionStatistics, NCudaLib::TMirrorMapping>& partStats,
                                                   const TComputeHistogramsHelper<Policy, TLayoutPolicy>& histCalcer,
                                                   double scoreStdDev = 0,
                                                   ui64 seed = 0) {
            auto& profiler = NCudaLib::GetProfiler();
            const TCudaBuffer<float, TFeaturesMapping>& histograms = histCalcer.GetHistograms(Stream);
            if (DataSet->GetGridSize(Policy)) {
                auto guard = profiler.Profile(TStringBuilder() << "Find optimal split for #" << DataSet->GetBinFeatures(Policy).size());
                FindOptimalSplit(DataSet->GetBinFeaturesForBestSplits(Policy),
                                 histograms,
                                 partStats,
                                 FoldCount,
                                 BestScores,
                                 ScoreFunction,
                                 L2,
                                 Normalize,
                                 scoreStdDev,
                                 seed,
                                 false,
                                 Stream);
            }
            return *this;
        }

        TBestSplitProperties ReadOptimalSplit() {
            if (DataSet->GetGridSize(Policy)) {
                auto split = BestSplit(BestScores, Stream);
                return {split.FeatureId, split.BinId, split.Score};
            } else {
                return {static_cast<ui32>(-1), 0, std::numeric_limits<float>::infinity()};
            }
        }

    private:
        const TGpuDataSet* DataSet = nullptr;
        ui32 Stream;
        ui32 FoldCount;
        ui32 MaxDepth;
        EScoreFunction ScoreFunction;
        double L2 = 1.0;
        bool Normalize = false;
        TCudaBuffer<TBestSplitProperties, TFeaturesMapping> BestScores;
    };

    template <EFeaturesGroupingPolicy Policy>
    class TFindBestSplitsHelper<Policy, TDocParallelLayout>: public TMoveOnly {
    public:
        using TGpuDataSet = typename TSharedCompressedIndex<TDocParallelLayout>::TCompressedDataSet;
        using TFeaturesMapping = typename TFeatureParallelLayout::TFeaturesMapping;
        using TSamplesMapping = typename TFeatureParallelLayout::TSamplesMapping;

    public:
        TFindBestSplitsHelper(const TGpuDataSet& dataSet,
                              ui32 foldCount,
                              ui32 maxDepth,
                              EScoreFunction score = EScoreFunction::Correlation,
                              double l2 = 1.0,
                              bool normalize = false,
                              ui32 stream = 0)
            : DataSet(&dataSet)
            , Stream(stream)
            , FoldCount(foldCount)
            , MaxDepth(maxDepth)
            , ScoreFunction(score)
            , L2(l2)
            , Normalize(normalize)
        {
            const ui64 blockCount = 32;
            if (DataSet->GetGridSize(Policy)) {
                auto bestScoresMapping = DataSet->GetBinFeaturesForBestSplits(Policy).GetMapping().Transform([&](const TSlice& histograms) -> ui64 {
                    return std::min(NHelpers::CeilDivide(histograms.Size(), 128),
                                    blockCount);
                });

                BestScores.Reset(bestScoresMapping);

                ReducedHistograms.Reset(DataSet->GetBinFeaturesForBestSplits(Policy).GetMapping().Transform([&](const TSlice binFeatures) {
                    return (1 << maxDepth) * foldCount * binFeatures.Size() * 2;
                }));
            }
        }

        TFindBestSplitsHelper& ComputeOptimalSplit(const TMirrorBuffer<const TPartitionStatistics>& reducedStats,
                                                   TComputeHistogramsHelper<Policy, TDocParallelLayout>& histHelper,
                                                   double scoreStdDev = 0,
                                                   ui64 seed = 0) {
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
                                     histogram,
                                     reducedStats,
                                     FoldCount,
                                     BestScores,
                                     ScoreFunction,
                                     L2,
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
                                     ReducedHistograms,
                                     reducedStats,
                                     FoldCount,
                                     BestScores,
                                     ScoreFunction,
                                     L2,
                                     Normalize,
                                     scoreStdDev,
                                     seed,
                                     true /*gathered by leaves */,
                                     streamId);
                }
            }
            return *this;
        }

        TBestSplitProperties ReadOptimalSplit() {
            if (DataSet->GetGridSize(Policy)) {
                auto split = BestSplit(BestScores, Stream);
                return {split.FeatureId,
                        split.BinId,
                        split.Score};
            } else {
                return {static_cast<ui32>(-1), 0, std::numeric_limits<float>::infinity()};
            }
        }

    private:
        const TGpuDataSet* DataSet = nullptr;
        ui32 Stream = 0;
        ui32 FoldCount = 0;
        ui32 MaxDepth = 0;
        EScoreFunction ScoreFunction;
        double L2 = 1.0;
        bool Normalize = false;
        TCudaBuffer<TBestSplitProperties, TFeaturesMapping> BestScores;
        TCudaBuffer<float, TFeaturesMapping> ReducedHistograms;
    };

    template <EFeaturesGroupingPolicy Policy,
              class TLayoutPolicy = TFeatureParallelLayout>
    class TScoreHelper: public TMoveOnly {
    public:
        using TGpuDataSet = typename TSharedCompressedIndex<TLayoutPolicy>::TCompressedDataSet;
        using TFeaturesMapping = typename TLayoutPolicy::TFeaturesMapping;
        using TSamplesMapping = typename TLayoutPolicy::TSamplesMapping;

    public:
        TScoreHelper(const TGpuDataSet& dataSet,
                     ui32 foldCount,
                     ui32 maxDepth,
                     EScoreFunction score = EScoreFunction::Correlation,
                     double l2 = 1.0,
                     bool normalize = false,
                     bool requestStream = true)
            : Stream(requestStream ? NCudaLib::GetCudaManager().RequestStream()
                                   : NCudaLib::GetCudaManager().DefaultStream())
            , ComputeHistogramsHelper(dataSet, foldCount, maxDepth, Stream)
            , FindBestSplitsHelper(dataSet, foldCount, maxDepth, score, l2, normalize, Stream.GetId())
        {
        }

        template <bool IsConst, class TUi32>
        TScoreHelper& SubmitCompute(const TOptimizationSubsets<TSamplesMapping, IsConst>& subsets,
                                    const TCudaBuffer<TUi32, TSamplesMapping>& docs) {
            ComputeHistogramsHelper.Compute(subsets, docs);
            return *this;
        }

        TVector<float> ReadHistograms() const {
            return ComputeHistogramsHelper.ReadHistograms();
        }

        TScoreHelper& ComputeOptimalSplit(const TCudaBuffer<const TPartitionStatistics, NCudaLib::TMirrorMapping>& partStats,
                                          double scoreStdDev = 0,
                                          ui64 seed = 0) {
            FindBestSplitsHelper.ComputeOptimalSplit(partStats,
                                                     ComputeHistogramsHelper,
                                                     scoreStdDev,
                                                     seed);
            return *this;
        }

        TBestSplitProperties ReadOptimalSplit() {
            return FindBestSplitsHelper.ReadOptimalSplit();
        }

        const TCudaBuffer<float, TFeaturesMapping>& GetHistograms() const {
            return ComputeHistogramsHelper.GetHistograms();
        }

    private:
        NCudaLib::TCudaManager::TComputationStream Stream;
        TComputeHistogramsHelper<Policy, TLayoutPolicy> ComputeHistogramsHelper;
        TFindBestSplitsHelper<Policy, TLayoutPolicy> FindBestSplitsHelper;
    };
}
