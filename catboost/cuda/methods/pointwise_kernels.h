#pragma once

#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/gpu_data/gpu_structures.h>
#include <catboost/cuda/gpu_data/grid_policy.h>
#include <catboost/cuda/cuda_util/gpu_data/partitions.h>
#include <catboost/cuda/methods/kernel/pointwise_hist2.cuh>
#include <catboost/cuda/methods/kernel/pointwise_hist1.cuh>
#include <catboost/cuda/methods/kernel/pointwise_scores.cuh>
#include <catboost/private/libs/options/enums.h>
#include <catboost/cuda/gpu_data/folds_histogram.h>
#include <catboost/libs/helpers/math_utils.h>

namespace NKernelHost {
    class TComputeHist2Kernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const TCFeature> Features;
        TSlice BinFeaturesSlice;

        TCudaBufferPtr<const ui32> Cindex;
        TCudaBufferPtr<const float> Target;
        TCudaBufferPtr<const float> Weight;
        TCudaBufferPtr<const ui32> Indices;

        TCudaBufferPtr<const TDataPartition> Partition;
        ui32 PartCount;
        ui32 FoldCount;

        TCudaBufferPtr<float> BinSums;
        ui32 HistLineSize;
        bool FullPass;
        NCatboostCuda::EFeaturesGroupingPolicy Policy;

        NCatboostCuda::TFoldsHistogram FoldsHist;

    public:
        TComputeHist2Kernel() = default;

        TComputeHist2Kernel(TCudaBufferPtr<const TCFeature> nbFeatures,
                            TCudaBufferPtr<const ui32> cindex,
                            TCudaBufferPtr<const float> target,
                            TCudaBufferPtr<const float> weight,
                            TCudaBufferPtr<const ui32> indices,
                            TCudaBufferPtr<const TDataPartition> partition,
                            ui32 partCount,
                            ui32 foldCount,
                            TCudaBufferPtr<float> binSums,
                            ui32 binFeatureCount,
                            bool fullPass,
                            NCatboostCuda::EFeaturesGroupingPolicy policy,
                            NCatboostCuda::TFoldsHistogram histogram)
            : Features(nbFeatures)
            , BinFeaturesSlice(TSlice(0, binFeatureCount))
            , Cindex(cindex)
            , Target(target)
            , Weight(weight)
            , Indices(indices)
            , Partition(partition)
            , PartCount(partCount)
            , FoldCount(foldCount)
            , BinSums(binSums)
            , HistLineSize(binFeatureCount)
            , FullPass(fullPass)
            , Policy(policy)
            , FoldsHist(histogram)
        {
        }

        TComputeHist2Kernel(TCudaBufferPtr<const TCFeature> nbFeatures,
                            TSlice binFeaturesSlice,
                            TCudaBufferPtr<const ui32> cindex,
                            TCudaBufferPtr<const float> target,
                            TCudaBufferPtr<const float> weight,
                            TCudaBufferPtr<const ui32> indices,
                            TCudaBufferPtr<const TDataPartition> partition,
                            ui32 partCount,
                            ui32 foldCount,
                            TCudaBufferPtr<float> binSums,
                            const ui32 binFeatureCount,
                            bool fullPass,
                            NCatboostCuda::EFeaturesGroupingPolicy policy,
                            NCatboostCuda::TFoldsHistogram histogram)
            : Features(nbFeatures)
            , BinFeaturesSlice(binFeaturesSlice)
            , Cindex(cindex)
            , Target(target)
            , Weight(weight)
            , Indices(indices)
            , Partition(partition)
            , PartCount(partCount)
            , FoldCount(foldCount)
            , BinSums(binSums)
            , HistLineSize(binFeatureCount)
            , FullPass(fullPass)
            , Policy(policy)
            , FoldsHist(histogram)
        {
        }

        Y_SAVELOAD_DEFINE(Features,
                          BinFeaturesSlice,
                          Cindex,
                          Target,
                          Weight,
                          Indices,
                          Partition,
                          PartCount,
                          FoldCount,
                          BinSums,
                          HistLineSize,
                          FullPass,
                          Policy,
                          FoldsHist);

        void Run(const TCudaStream& stream) const;
    };

    class TComputeHist1Kernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const TCFeature> Features;
        TSlice BinFeaturesSlice;

        TCudaBufferPtr<const ui32> Cindex;
        TCudaBufferPtr<const float> Target;
        TCudaBufferPtr<const ui32> Indices;

        TCudaBufferPtr<const TDataPartition> Partition;
        ui32 PartCount;
        ui32 FoldCount;

        TCudaBufferPtr<float> BinSums;
        int HistLineSize;
        bool FullPass;
        NCatboostCuda::EFeaturesGroupingPolicy Policy;

    public:
        TComputeHist1Kernel() = default;

        TComputeHist1Kernel(TCudaBufferPtr<const TCFeature> features,
                            TSlice binFeaturesSlice,
                            TCudaBufferPtr<const ui32> cindex,
                            TCudaBufferPtr<const float> target,
                            TCudaBufferPtr<const ui32> indices,
                            TCudaBufferPtr<const TDataPartition> partition,
                            ui32 partCount,
                            ui32 foldCount,
                            TCudaBufferPtr<float> binSums,
                            const ui32 binFeatureCount,
                            bool fullPass,
                            NCatboostCuda::EFeaturesGroupingPolicy policy)
            : Features(features)
            , BinFeaturesSlice(binFeaturesSlice)
            , Cindex(cindex)
            , Target(target)
            , Indices(indices)
            , Partition(partition)
            , PartCount(partCount)
            , FoldCount(foldCount)
            , BinSums(binSums)
            , HistLineSize(binFeatureCount)
            , FullPass(fullPass)
            , Policy(policy)
        {
        }

        Y_SAVELOAD_DEFINE(Features,
                          BinFeaturesSlice,
                          Cindex,
                          Target,
                          Indices,
                          Partition,
                          PartCount,
                          FoldCount,
                          BinSums,
                          HistLineSize,
                          FullPass,
                          Policy);

        void Run(const TCudaStream& stream) const;
    };

    class TUpdateFoldBinsKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<ui32> DstBins;
        TCudaBufferPtr<const ui32> Bins;
        TCudaBufferPtr<const ui32> DocIndices;
        ui32 LoadBit;
        ui32 FoldBits;

    public:
        TUpdateFoldBinsKernel() = default;

        TUpdateFoldBinsKernel(TCudaBufferPtr<ui32> dstBins,
                              TCudaBufferPtr<const ui32> bins, TCudaBufferPtr<const ui32> docIndices,
                              ui32 loadBit, ui32 foldBits)
            : DstBins(dstBins)
            , Bins(bins)
            , DocIndices(docIndices)
            , LoadBit(loadBit)
            , FoldBits(foldBits)
        {
        }

        Y_SAVELOAD_DEFINE(DstBins, Bins, DocIndices, LoadBit, FoldBits);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE(DstBins.Size() == DocIndices.Size());
            NKernel::UpdateFoldBins(DstBins.Get(), Bins.Get(), DocIndices.Get(), DocIndices.Size(), LoadBit, FoldBits, stream.GetStream());
        }
    };

    class TUpdatePartitionPropsKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> Target;
        TCudaBufferPtr<const float> Weights;
        TCudaBufferPtr<const TDataPartition> Parts;
        TCudaBufferPtr<TPartitionStatistics> PartStats;

    public:
        TUpdatePartitionPropsKernel() = default;

        TUpdatePartitionPropsKernel(TCudaBufferPtr<const float> target,
                                    TCudaBufferPtr<const float> weights,
                                    TCudaBufferPtr<const TDataPartition> parts,
                                    TCudaBufferPtr<TPartitionStatistics> partStats)
            : Target(target)
            , Weights(weights)
            , Parts(parts)
            , PartStats(partStats)
        {
        }

        Y_SAVELOAD_DEFINE(Target, Weights, Parts, PartStats);

        void Run(const TCudaStream& stream) const {
            NKernel::UpdatePartitionProps(Target.Get(),
                                          Weights.Get(),
                                          nullptr,
                                          Parts.Get(),
                                          PartStats.Get(),
                                          PartStats.Size(),
                                          stream.GetStream());
        }
    };

    class TGatherHistogramByLeavesKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> Histogram;
        ui32 BinFeatureCount;
        ui32 HistCount;
        ui32 LeafCount;
        ui32 FoldCount;
        TCudaBufferPtr<float> Result;

    public:
        TGatherHistogramByLeavesKernel() = default;

        TGatherHistogramByLeavesKernel(TCudaBufferPtr<const float> histogram, const ui32 binFeatureCount,
                                       const ui32 histCount, const ui32 leafCount, const ui32 foldCount, TCudaBufferPtr<float> result)
            : Histogram(histogram)
            , BinFeatureCount(binFeatureCount)
            , HistCount(histCount)
            , LeafCount(leafCount)
            , FoldCount(foldCount)
            , Result(result)
        {
        }

        Y_SAVELOAD_DEFINE(Histogram, BinFeatureCount, HistCount, LeafCount, FoldCount, Result);

        void Run(const TCudaStream& stream) const {
            NKernel::GatherHistogramByLeaves(Histogram.Get(), BinFeatureCount, HistCount,
                                             LeafCount, FoldCount, Result.Get(), stream.GetStream());
        }
    };

    class TFindOptimalSplitKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const TCBinFeature> BinaryFeatures;
        TCudaBufferPtr<const float> CatFeatureWeights;
        TCudaBufferPtr<const float> FeatureWeights;
        TCudaBufferPtr<const float> Splits;
        TCudaBufferPtr<const TPartitionStatistics> Parts;
        ui32 FoldCount;
        double ScoreBeforeSplit;
        TCudaBufferPtr<TBestSplitProperties> Result;
        EScoreFunction ScoreFunction;
        double L2;
        double MetaL2Exponent;
        double MetaL2Frequency;
        bool Normalize;
        double ScoreStdDev;
        ui64 Seed;
        bool GatheredByLeaves;

    public:
        TFindOptimalSplitKernel() = default;

        TFindOptimalSplitKernel(TCudaBufferPtr<const TCBinFeature> binaryFeatures,
                                TCudaBufferPtr<const float> catFeatureWeights,
                                TCudaBufferPtr<const float> featureWeights,
                                TCudaBufferPtr<const float> splits,
                                TCudaBufferPtr<const TPartitionStatistics> parts,
                                ui32 foldCount,
                                double scoreBeforSplit,
                                TCudaBufferPtr<TBestSplitProperties> result,
                                EScoreFunction scoreFunction,
                                double l2,
                                double metaL2Exponent,
                                double metaL2Frequency,
                                bool normalize,
                                double scoreStdDev,
                                ui64 seed,
                                bool gatheredByLeaves)
            : BinaryFeatures(binaryFeatures)
            , CatFeatureWeights(catFeatureWeights)
            , FeatureWeights(featureWeights)
            , Splits(splits)
            , Parts(parts)
            , FoldCount(foldCount)
            , ScoreBeforeSplit(scoreBeforSplit)
            , Result(result)
            , ScoreFunction(scoreFunction)
            , L2(l2)
            , MetaL2Exponent(metaL2Exponent)
            , MetaL2Frequency(metaL2Frequency)
            , Normalize(normalize)
            , ScoreStdDev(scoreStdDev)
            , Seed(seed)
            , GatheredByLeaves(gatheredByLeaves)
        {
        }

        Y_SAVELOAD_DEFINE(BinaryFeatures, CatFeatureWeights, FeatureWeights, Splits, Parts, FoldCount, ScoreBeforeSplit, Result, ScoreFunction, L2, MetaL2Exponent, MetaL2Frequency, Normalize, ScoreStdDev, Seed, GatheredByLeaves);

        void Run(const TCudaStream& stream) const {
            const ui32 foldBits = NCB::IntLog2(FoldCount);
            const ui32 leavesCount = static_cast<ui32>(Parts.Size() >> foldBits);
            CB_ENSURE(Result.Size());


            for (auto feature : BinaryFeatures.Read(stream)) {
                Y_ASSERT(feature.FeatureId < FeatureWeights.Size());
            }
            Y_ASSERT(CatFeatureWeights.Size() <= FeatureWeights.Size());

            NKernel::FindOptimalSplit(BinaryFeatures.Get(),
                                      static_cast<ui32>(BinaryFeatures.Size()),
                                      CatFeatureWeights.Get(),
                                      FeatureWeights.Get(),
                                      FeatureWeights.Size(),
                                      Splits.Get(),
                                      Parts.Get(),
                                      leavesCount,
                                      FoldCount,
                                      ScoreBeforeSplit,
                                      Result.Get(),
                                      static_cast<ui32>(Result.Size()),
                                      ScoreFunction,
                                      L2, MetaL2Exponent, MetaL2Frequency, Normalize,
                                      ScoreStdDev,
                                      Seed,
                                      GatheredByLeaves,
                                      stream.GetStream());
        }
    };
}

template <class TGpuDataSet,
          class TTargetsMapping,
          class TFloat,
          class TUi32,
          class TMaybeConstPartition>
inline void ComputeHistogram2(NCatboostCuda::EFeaturesGroupingPolicy policy,
                              const TGpuDataSet& dataSet,
                              const TCudaBuffer<TFloat, TTargetsMapping>& targets,
                              const TCudaBuffer<TFloat, TTargetsMapping>& weights,
                              const TCudaBuffer<TUi32, TTargetsMapping>& indices,
                              const TCudaBuffer<TMaybeConstPartition, TTargetsMapping>& dataPartitions,
                              ui32 partCount,
                              ui32 foldCount,
                              TCudaBuffer<float, typename TGpuDataSet::THistogramsMapping>& histograms,
                              bool fullPass,
                              ui32 stream) {
    using TKernel = NKernelHost::TComputeHist2Kernel;

    const auto& grid = dataSet.GetGrid(policy);
    LaunchKernels<TKernel>(targets.NonEmptyDevices(),
                           stream,
                           grid,
                           dataSet.GetCompressedIndex(),
                           targets,
                           weights,
                           indices,
                           dataPartitions,
                           partCount,
                           foldCount,
                           histograms,
                           dataSet.GetBinFeatureCount(policy),
                           fullPass,
                           policy,
                           dataSet.GetFoldsHistogram(policy));
}

template <class TFloat,
          class TUi32,
          class TMaybeConstPartition>
inline void ComputeBlockHistogram2(NCatboostCuda::EFeaturesGroupingPolicy policy,
                                   const TCudaBuffer<const TCFeature, NCudaLib::TStripeMapping>& gridBlock,
                                   const NCatboostCuda::TFoldsHistogram& foldsHistogram,
                                   const TSlice& binFeaturesSlice,
                                   const TCudaBuffer<ui32, NCudaLib::TStripeMapping>& compressedIndex,
                                   const TCudaBuffer<TFloat, NCudaLib::TStripeMapping>& targets,
                                   const TCudaBuffer<TFloat, NCudaLib::TStripeMapping>& weights,
                                   const TCudaBuffer<TUi32, NCudaLib::TStripeMapping>& indices,
                                   const TCudaBuffer<TMaybeConstPartition, NCudaLib::TStripeMapping>& dataPartitions,
                                   ui32 partCount,
                                   TCudaBuffer<float, NCudaLib::TStripeMapping>& histograms,
                                   ui32 histogramLineSize,
                                   bool fullPass,
                                   ui32 stream) {
    using TKernel = NKernelHost::TComputeHist2Kernel;

    LaunchKernels<TKernel>(targets.NonEmptyDevices(),
                           stream,
                           gridBlock,
                           binFeaturesSlice,
                           compressedIndex,
                           targets,
                           weights,
                           indices,
                           dataPartitions,
                           partCount,
                           1u,
                           histograms,
                           histogramLineSize,
                           fullPass,
                           policy,
                           foldsHistogram);
}

template <class TFloat,
          class TUi32,
          class TMaybeConstPartition>
inline void ComputeBlockHistogram1(NCatboostCuda::EFeaturesGroupingPolicy policy,
                                   const TCudaBuffer<const TCFeature, NCudaLib::TStripeMapping>& gridBlock,
                                   const TSlice& binFeaturesSlice,
                                   const TCudaBuffer<ui32, NCudaLib::TStripeMapping>& compressedIndex,
                                   const TCudaBuffer<TFloat, NCudaLib::TStripeMapping>& targets,
                                   const TCudaBuffer<TUi32, NCudaLib::TStripeMapping>& indices,
                                   const TCudaBuffer<TMaybeConstPartition, NCudaLib::TStripeMapping>& dataPartitions,
                                   ui32 partCount,
                                   TCudaBuffer<float, NCudaLib::TStripeMapping>& histograms,
                                   ui32 histogramLineSize,
                                   bool fullPass,
                                   ui32 stream) {
    using TKernel = NKernelHost::TComputeHist1Kernel;

    LaunchKernels<TKernel>(targets.NonEmptyDevices(),
                           stream,
                           gridBlock,
                           binFeaturesSlice,
                           compressedIndex,
                           targets,
                           indices,
                           dataPartitions,
                           partCount,
                           1u,
                           histograms,
                           histogramLineSize,
                           fullPass,
                           policy);
}

template <class THistMapping>
inline void GatherHistogramByLeaves(const TCudaBuffer<float, THistMapping>& histograms,
                                    NCudaLib::TDistributedObject<ui32> binFeatureCount,
                                    ui32 histCount, ui32 leavesCount, ui32 foldCount,
                                    TCudaBuffer<float, THistMapping>& byLeavesHist,
                                    ui32 stream = 0) {
    using TKernel = NKernelHost::TGatherHistogramByLeavesKernel;
    LaunchKernels<TKernel>(histograms.NonEmptyDevices(), stream, histograms, binFeatureCount, histCount, leavesCount, foldCount, byLeavesHist);
}

template <class TMapping>
inline void UpdatePartitionStats(TCudaBuffer<TPartitionStatistics, TMapping>& partStats,
                                 const TCudaBuffer<TDataPartition, TMapping>& parts,
                                 const TCudaBuffer<float, TMapping>& target,
                                 const TCudaBuffer<float, TMapping>& weights,
                                 ui32 stream = 0) {
    using TKernel = NKernelHost::TUpdatePartitionPropsKernel;
    LaunchKernels<TKernel>(partStats.NonEmptyDevices(), stream, target, weights, parts, partStats);
}

template <class TMapping>
inline void UpdatePartitionStatsTargetOnly(TCudaBuffer<TPartitionStatistics, TMapping>& partStats,
                                           const TCudaBuffer<TDataPartition, TMapping>& parts,
                                           const TCudaBuffer<float, TMapping>& target,
                                           ui32 stream = 0) {
    using TKernel = NKernelHost::TUpdatePartitionPropsKernel;
    LaunchKernels<TKernel>(partStats.NonEmptyDevices(), stream, target, nullptr, parts, partStats);
}

template <class TMapping>
inline void UpdatePartitionStatsWeightsOnly(TCudaBuffer<TPartitionStatistics, TMapping>& partStats,
                                            const TCudaBuffer<TDataPartition, TMapping>& parts,
                                            const TCudaBuffer<float, TMapping>& weights,
                                            ui32 stream = 0) {
    using TKernel = NKernelHost::TUpdatePartitionPropsKernel;
    LaunchKernels<TKernel>(partStats.NonEmptyDevices(), stream, (const TCudaBuffer<float, TMapping>*)nullptr, weights, parts, partStats);
}

template <class TFeaturesMapping, class TFeatureWeightsMapping>
inline void FindOptimalSplit(const TCudaBuffer<TCBinFeature, TFeaturesMapping>& features,
                             const TCudaBuffer<const float, TFeatureWeightsMapping>& catFeatureWeights,
                             const TMirrorBuffer<const float>& featureWeights,
                             const TCudaBuffer<float, TFeaturesMapping>& histograms,
                             const TMirrorBuffer<const TPartitionStatistics>& partStats,
                             ui32 foldCount,
                             double scoreBeforeSplit,
                             TCudaBuffer<TBestSplitProperties, TFeaturesMapping>& scores,
                             EScoreFunction scoreFunction,
                             double l2,
                             double metaL2Exponent,
                             double metaL2Frequency,
                             bool normalize,
                             double scoreStdDev,
                             ui64 seed,
                             bool gatheredByLeaves,
                             ui32 stream = 0) {
    if (foldCount > 1) {
        CB_ENSURE(!gatheredByLeaves, "Best split search for gathered by leaves splits is not implemented yet");
    }
    using TKernel = NKernelHost::TFindOptimalSplitKernel;
    LaunchKernels<TKernel>(scores.NonEmptyDevices(), stream, features, catFeatureWeights, featureWeights, histograms, partStats, foldCount, scoreBeforeSplit, scores, scoreFunction, l2, metaL2Exponent, metaL2Frequency, normalize, scoreStdDev, seed, gatheredByLeaves);
}

template <class TFeaturesMapping, class TUi32>
inline void UpdateBins(TCudaBuffer<ui32, TFeaturesMapping>& bins,
                       const TCudaBuffer<TUi32, TFeaturesMapping>& docBins,
                       const TCudaBuffer<TUi32, TFeaturesMapping>& docIndices,
                       ui32 currentBit, ui32 foldBits, ui32 stream = 0) {
    using TKernel = NKernelHost::TUpdateFoldBinsKernel;
    LaunchKernels<TKernel>(bins.NonEmptyDevices(), stream, bins, docBins, docIndices, currentBit, foldBits);
}
