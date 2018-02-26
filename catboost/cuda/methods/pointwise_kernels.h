#pragma once

#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/gpu_data/gpu_structures.h>
#include <catboost/cuda/gpu_data/grid_policy.h>
#include <catboost/cuda/cuda_util/gpu_data/partitions.h>
#include <catboost/cuda/methods/kernel/pointwise_hist2.cuh>
#include <catboost/cuda/methods/kernel/pointwise_scores.cuh>
#include <catboost/libs/options/enums.h>
#include <catboost/cuda/utils/compression_helpers.h>

namespace NKernelHost {
    template <NCatboostCuda::EFeaturesGroupingPolicy>
    class TComputeHistKernel;

    template <>
    class TComputeHistKernel<NCatboostCuda::EFeaturesGroupingPolicy::BinaryFeatures>: public TStatelessKernel {
    private:
        TCudaBufferPtr<const TCFeature> BFeatures;
        TCudaBufferPtr<const ui32> Cindex;
        TCudaBufferPtr<const float> Target;
        TCudaBufferPtr<const float> Weight;
        TCudaBufferPtr<const ui32> Indices;
        TCudaBufferPtr<const TDataPartition> Partition;
        ui32 PartsCount;
        ui32 FoldCount;
        TCudaBufferPtr<float> BinSums;
        int BinFeatureCount;
        bool FullPass;

    public:
        TComputeHistKernel() = default;

        TComputeHistKernel(TCudaBufferPtr<const TCFeature> bFeatures,
                           TCudaBufferPtr<const ui32> cindex,
                           TCudaBufferPtr<const float> target,
                           TCudaBufferPtr<const float> weight,
                           TCudaBufferPtr<const ui32> indices,
                           TCudaBufferPtr<const TDataPartition> partition,
                           ui32 partsCount,
                           ui32 foldCount,
                           TCudaBufferPtr<float> binSums,
                           const ui32 binFeatureCount,
                           bool fullPass)
            : BFeatures(bFeatures)
            , Cindex(cindex)
            , Target(target)
            , Weight(weight)
            , Indices(indices)
            , Partition(partition)
            , PartsCount(partsCount)
            , FoldCount(foldCount)
            , BinSums(binSums)
            , BinFeatureCount(binFeatureCount)
            , FullPass(fullPass)
        {
            CB_ENSURE(binFeatureCount == bFeatures.Size(), TStringBuilder() << binFeatureCount << " ≠ " << bFeatures.Size());
        }

        Y_SAVELOAD_DEFINE(BFeatures, Cindex,
                          Target, Weight, Indices,
                          Partition, PartsCount, FoldCount,
                          BinSums, BinFeatureCount, FullPass);

        void Run(const TCudaStream& stream) const {
            NKernel::ComputeHist2Binary(BFeatures.Get(), static_cast<int>(BFeatures.Size()),
                                        Cindex.Get(),
                                        Target.Get(), Weight.Get(), Indices.Get(), Indices.Size(),
                                        Partition.Get(), PartsCount, FoldCount, BinSums.Get(), FullPass,
                                        stream.GetStream());
        }
    };

    template <>
    class TComputeHistKernel<NCatboostCuda::EFeaturesGroupingPolicy::OneByteFeatures>: public TStatelessKernel {
    private:
        TCudaBufferPtr<const TCFeature> NbFeatures;
        TCudaBufferPtr<const ui32> Cindex;
        TCudaBufferPtr<const float> Target;
        TCudaBufferPtr<const float> Weight;
        TCudaBufferPtr<const ui32> Indices;
        TCudaBufferPtr<const TDataPartition> Partition;
        ui32 PartCount;
        ui32 FoldCount;

        TCudaBufferPtr<float> BinSums;
        int BinFeatureCount;
        bool FullPass;

    public:
        TComputeHistKernel() = default;

        TComputeHistKernel(TCudaBufferPtr<const TCFeature> nbFeatures,
                           TCudaBufferPtr<const ui32> cindex,
                           TCudaBufferPtr<const float> target, TCudaBufferPtr<const float> weight, TCudaBufferPtr<const ui32> indices,
                           TCudaBufferPtr<const TDataPartition> partition, ui32 partCount, ui32 foldCount,
                           TCudaBufferPtr<float> binSums, const ui32 binFeatureCount, bool fullPass)
            : NbFeatures(nbFeatures)
            , Cindex(cindex)
            , Target(target)
            , Weight(weight)
            , Indices(indices)
            , Partition(partition)
            , PartCount(partCount)
            , FoldCount(foldCount)
            , BinSums(binSums)
            , BinFeatureCount(binFeatureCount)
            , FullPass(fullPass)
        {
        }

        Y_SAVELOAD_DEFINE(NbFeatures,
                          Cindex,
                          Target,
                          Weight,
                          Indices,
                          Partition, PartCount, FoldCount,
                          BinSums, BinFeatureCount, FullPass);

        void Run(const TCudaStream& stream) const {
            NKernel::ComputeHist2NonBinary(NbFeatures.Get(), static_cast<int>(NbFeatures.Size()),
                                           Cindex.Get(),
                                           Target.Get(),
                                           Weight.Get(),
                                           Indices.Get(), Indices.Size(),
                                           Partition.Get(),
                                           PartCount, FoldCount,
                                           BinSums.Get(), BinFeatureCount,
                                           FullPass,
                                           stream.GetStream());
        }
    };

    template <>
    class TComputeHistKernel<NCatboostCuda::EFeaturesGroupingPolicy::HalfByteFeatures>: public TStatelessKernel {
    private:
        TCudaBufferPtr<const TCFeature> NbFeatures;
        TCudaBufferPtr<const ui32> Cindex;
        TCudaBufferPtr<const float> Target;
        TCudaBufferPtr<const float> Weight;
        TCudaBufferPtr<const ui32> Indices;
        TCudaBufferPtr<const TDataPartition> Partition;
        ui32 PartCount;
        ui32 FoldCount;

        TCudaBufferPtr<float> BinSums;
        int BinFeatureCount;
        bool FullPass;

    public:
        TComputeHistKernel() = default;

        TComputeHistKernel(TCudaBufferPtr<const TCFeature> nbFeatures,
                           TCudaBufferPtr<const ui32> cindex,
                           TCudaBufferPtr<const float> target, TCudaBufferPtr<const float> weight, TCudaBufferPtr<const ui32> indices,
                           TCudaBufferPtr<const TDataPartition> partition, ui32 partCount, ui32 foldCount,
                           TCudaBufferPtr<float> binSums, const ui32 binFeatureCount, bool fullPass)
            : NbFeatures(nbFeatures)
            , Cindex(cindex)
            , Target(target)
            , Weight(weight)
            , Indices(indices)
            , Partition(partition)
            , PartCount(partCount)
            , FoldCount(foldCount)
            , BinSums(binSums)
            , BinFeatureCount(binFeatureCount)
            , FullPass(fullPass)
        {
        }

        Y_SAVELOAD_DEFINE(NbFeatures, Cindex, Target, Weight, Indices, Partition, PartCount, FoldCount, BinSums, BinFeatureCount, FullPass);

        void Run(const TCudaStream& stream) const {
            NKernel::ComputeHist2HalfByte(NbFeatures.Get(), static_cast<int>(NbFeatures.Size()),
                                          Cindex.Get(),
                                          Target.Get(), Weight.Get(), Indices.Get(), Indices.Size(), Partition.Get(),
                                          PartCount, FoldCount,
                                          BinSums.Get(), BinFeatureCount,
                                          FullPass,
                                          stream.GetStream());
        }
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
            NKernel::UpdatePartitionProps(Target.Get(), Weights.Get(), nullptr, Parts.Get(),
                                          PartStats.Get(), PartStats.Size(), stream.GetStream());
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
        TCudaBufferPtr<const float> Splits;
        TCudaBufferPtr<const TPartitionStatistics> Parts;
        ui32 FoldCount;
        TCudaBufferPtr<TBestSplitProperties> Result;
        EScoreFunction ScoreFunction;
        double L2;
        bool Normalize;
        double ScoreStdDev;
        ui64 Seed;
        bool GatheredByLeaves;

    public:
        TFindOptimalSplitKernel() = default;

        TFindOptimalSplitKernel(TCudaBufferPtr<const TCBinFeature> binaryFeatures,
                                TCudaBufferPtr<const float> splits,
                                TCudaBufferPtr<const TPartitionStatistics> parts,
                                ui32 foldCount,
                                TCudaBufferPtr<TBestSplitProperties> result,
                                EScoreFunction scoreFunction,
                                double l2,
                                bool normalize,
                                double scoreStdDev,
                                ui64 seed,
                                bool gatheredByLeaves)
            : BinaryFeatures(binaryFeatures)
            , Splits(splits)
            , Parts(parts)
            , FoldCount(foldCount)
            , Result(result)
            , ScoreFunction(scoreFunction)
            , L2(l2)
            , Normalize(normalize)
            , ScoreStdDev(scoreStdDev)
            , Seed(seed)
            , GatheredByLeaves(gatheredByLeaves)
        {
        }

        Y_SAVELOAD_DEFINE(BinaryFeatures, Splits, Parts, FoldCount, Result, ScoreFunction, L2, Normalize, ScoreStdDev, Seed, GatheredByLeaves);

        void Run(const TCudaStream& stream) const {
            const ui32 foldBits = IntLog2(FoldCount);
            const ui32 leavesCount = static_cast<ui32>(Parts.Size() >> foldBits);
            CB_ENSURE(Result.Size());
            NKernel::FindOptimalSplit(BinaryFeatures.Get(),
                                      static_cast<ui32>(BinaryFeatures.Size()),
                                      Splits.Get(),
                                      Parts.Get(),
                                      leavesCount,
                                      FoldCount,
                                      Result.Get(),
                                      static_cast<ui32>(Result.Size()),
                                      ScoreFunction,
                                      L2, Normalize,
                                      ScoreStdDev,
                                      Seed,
                                      GatheredByLeaves,
                                      stream.GetStream());
        }
    };
}

template <NCatboostCuda::EFeaturesGroupingPolicy Policy,
          class TGpuDataSet,
          class TTargetsMapping,
          class TFloat,
          class TUi32,
          class TMaybeConstPartition>
inline void ComputeHistogram2(const TGpuDataSet& dataSet,
                              const TCudaBuffer<TFloat, TTargetsMapping>& targets,
                              const TCudaBuffer<TFloat, TTargetsMapping>& weights,
                              const TCudaBuffer<TUi32, TTargetsMapping>& indices,
                              const TCudaBuffer<TMaybeConstPartition, TTargetsMapping>& dataPartitions,
                              ui32 partCount,
                              ui32 foldCount,
                              TCudaBuffer<float, typename TGpuDataSet::THistogramsMapping>& histograms,
                              bool fullPass,
                              ui32 stream) {
    using TKernel = NKernelHost::TComputeHistKernel<Policy>;

    const auto& grid = dataSet.GetGrid(Policy);
    LaunchKernels<TKernel>(grid.NonEmptyDevices(),
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
                           dataSet.GetBinFeatureCount(Policy),
                           fullPass);
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

template <class TFeaturesMapping>
inline void FindOptimalSplit(const TCudaBuffer<TCBinFeature, TFeaturesMapping>& features,
                             const TCudaBuffer<float, TFeaturesMapping>& histograms,
                             const TMirrorBuffer<const TPartitionStatistics>& partStats,
                             ui32 foldCount,
                             TCudaBuffer<TBestSplitProperties, TFeaturesMapping>& scores,
                             EScoreFunction scoreFunction,
                             double l2,
                             bool normalize,
                             double scoreStdDev,
                             ui64 seed,
                             bool gatheredByLeaves,
                             ui32 stream = 0) {
    if (foldCount > 1) {
        CB_ENSURE(!gatheredByLeaves, "Best split search for gathered by leaves splits is not implemented yet");
    }
    using TKernel = NKernelHost::TFindOptimalSplitKernel;
    LaunchKernels<TKernel>(features.NonEmptyDevices(), stream, features, histograms, partStats, foldCount, scores, scoreFunction, l2, normalize, scoreStdDev, seed, gatheredByLeaves);
}

template <class TFeaturesMapping, class TUi32>
inline void UpdateBins(TCudaBuffer<ui32, TFeaturesMapping>& bins,
                       const TCudaBuffer<TUi32, TFeaturesMapping>& docBins,
                       const TCudaBuffer<TUi32, TFeaturesMapping>& docIndices,
                       ui32 currentBit, ui32 foldBits, ui32 stream = 0) {
    using TKernel = NKernelHost::TUpdateFoldBinsKernel;
    LaunchKernels<TKernel>(bins.NonEmptyDevices(), stream, bins, docBins, docIndices, currentBit, foldBits);
}
