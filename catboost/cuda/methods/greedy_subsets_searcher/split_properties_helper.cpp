#include "split_properties_helper.h"
#include "compute_by_blocks_helper.h"
#include "split_points.h"
#include <catboost/cuda/gpu_data/splitter.h>
#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/reduce_scatter.h>
#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/all_reduce.h>
#include <catboost/cuda/methods/helpers.h>
#include <catboost/cuda/methods/pointwise_kernels.h>
#include <catboost/cuda/cuda_util/partitions_reduce.h>

#include <catboost/cuda/methods/greedy_subsets_searcher/kernel/histogram_utils.cuh>
#include <catboost/cuda/methods/greedy_subsets_searcher/kernel/hist.cuh>
#include <catboost/cuda/methods/greedy_subsets_searcher/kernel/gather_bins.cuh>

namespace NKernelHost {
    class TWriteInitPartitions: public TStatelessKernel {
    private:
        TCudaHostBufferPtr<TDataPartition> PartCpu;
        TCudaBufferPtr<TDataPartition> PartGpu;
        TCudaBufferPtr<const ui32> Indices;

    public:
        TWriteInitPartitions() = default;

        TWriteInitPartitions(const TCudaHostBufferPtr<TDataPartition>& partCpu,
                             const TCudaBufferPtr<TDataPartition>& partGpu,
                             const TCudaBufferPtr<const ui32>& indices)
            : PartCpu(partCpu)
            , PartGpu(partGpu)
            , Indices(indices)
        {
        }

        Y_SAVELOAD_DEFINE(PartCpu, PartGpu, Indices);

        void Run(const TCudaStream& stream) const {
            PartCpu.Get()->Size = static_cast<ui32>(Indices.Size());
            PartCpu.Get()->Offset = 0;
            NCudaLib::CopyMemoryAsync(PartCpu.Get(), PartGpu.Get(), 1, stream);
        }
    };

    class TCopyHistogramsKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const ui32> LeftLeaves;
        TCudaBufferPtr<const ui32> RightLeaves;
        ui32 NumStats;
        ui32 BinFeatures;
        TCudaBufferPtr<float> Histograms;

    public:
        TCopyHistogramsKernel() = default;

        TCopyHistogramsKernel(TCudaBufferPtr<const ui32> leftLeaves,
                              TCudaBufferPtr<const ui32> rightLeaves,
                              ui32 numStats,
                              ui32 binFeatures,
                              TCudaBufferPtr<float> histograms)
            : LeftLeaves(leftLeaves)
            , RightLeaves(rightLeaves)
            , NumStats(numStats)
            , BinFeatures(binFeatures)
            , Histograms(histograms)
        {
        }

        Y_SAVELOAD_DEFINE(LeftLeaves, RightLeaves, NumStats, BinFeatures, Histograms);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE(LeftLeaves.Size() == RightLeaves.Size());

            NKernel::CopyHistograms(LeftLeaves.Get(),
                                    RightLeaves.Get(),
                                    LeftLeaves.Size(),
                                    NumStats,
                                    BinFeatures,
                                    Histograms.Get(),
                                    stream.GetStream());
        }
    };

    class TWriteReducesHistogramsKernel: public TStatelessKernel {
    private:
        ui32 BlockOffset;
        ui32 HistBlockSize;
        TCudaBufferPtr<const ui32> HistogramIds;
        ui32 StatCount;
        TCudaBufferPtr<const float> BlockHistogram;
        ui32 BinFeatureCount;
        TCudaBufferPtr<float> DstHistogram;

    public:
        TWriteReducesHistogramsKernel() = default;

        TWriteReducesHistogramsKernel(ui32 blockOffset,
                                      ui32 histBlockSize,
                                      TCudaBufferPtr<const ui32> histogramIds,
                                      ui32 statCount,
                                      TCudaBufferPtr<const float> blockHistogram,
                                      ui32 binFeatureCount,
                                      TCudaBufferPtr<float> dstHistogram)
            : BlockOffset(blockOffset)
            , HistBlockSize(histBlockSize)
            , HistogramIds(histogramIds)
            , StatCount(statCount)
            , BlockHistogram(blockHistogram)
            , BinFeatureCount(binFeatureCount)
            , DstHistogram(dstHistogram)
        {
        }

        Y_SAVELOAD_DEFINE(BlockOffset, HistBlockSize, HistogramIds, StatCount, BlockHistogram, BinFeatureCount,
                          DstHistogram);

        void Run(const TCudaStream& stream) const {
            NKernel::WriteReducesHistograms(BlockOffset, HistBlockSize, HistogramIds.Get(), HistogramIds.Size(),
                                            StatCount, BlockHistogram.Get(), BinFeatureCount, DstHistogram.Get(),
                                            stream.GetStream());
        }
    };

    class TZeroHistogramsKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const ui32> HistIds;
        ui32 StatCount;
        ui32 BinFeatureCount;
        TCudaBufferPtr<float> DstHistogram;

    public:
        TZeroHistogramsKernel() = default;

        TZeroHistogramsKernel(TCudaBufferPtr<const ui32> histIds,
                              ui32 statCount,
                              ui32 binFeatureCount,
                              TCudaBufferPtr<float> dstHistogram)
            : HistIds(histIds)
            , StatCount(statCount)
            , BinFeatureCount(binFeatureCount)
            , DstHistogram(dstHistogram)
        {
        }

        Y_SAVELOAD_DEFINE(HistIds, StatCount, BinFeatureCount, DstHistogram);

        void Run(const TCudaStream& stream) const {
            NKernel::ZeroHistograms(HistIds.Get(), HistIds.Size(), StatCount, BinFeatureCount, DstHistogram.Get(),
                                    stream.GetStream());
        }
    };

    class TSubstractHistgoramsKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const ui32> FromIds;
        TCudaBufferPtr<const ui32> WhatIds;
        ui32 StatCount;
        ui32 BinFeatureCount;
        TCudaBufferPtr<float> DstHistogram;

    public:
        TSubstractHistgoramsKernel() = default;

        TSubstractHistgoramsKernel(TCudaBufferPtr<const ui32> fromIds, TCudaBufferPtr<const ui32> whatIds,
                                   ui32 statCount, ui32 binFeatureCount, TCudaBufferPtr<float> dstHistogram)
            : FromIds(fromIds)
            , WhatIds(whatIds)
            , StatCount(statCount)
            , BinFeatureCount(binFeatureCount)
            , DstHistogram(dstHistogram)
        {
        }

        Y_SAVELOAD_DEFINE(FromIds, WhatIds, StatCount, BinFeatureCount, DstHistogram);

        void Run(const TCudaStream& stream) const {
            const int idsCount = static_cast<const int>(FromIds.Size());
            CB_ENSURE(idsCount == (int)WhatIds.Size());
            NKernel::SubstractHistgorams(FromIds.Get(), WhatIds.Get(), idsCount, StatCount, BinFeatureCount,
                                         DstHistogram.Get(), stream.GetStream());
        }
    };

    class TScanHistgoramsKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const TBinarizedFeature> Features;
        TCudaBufferPtr<const ui32> Ids;
        ui32 StatCount;
        ui32 BinFeatureCount;
        TCudaBufferPtr<float> Histograms;

    public:
        TScanHistgoramsKernel() = default;

        TScanHistgoramsKernel(TCudaBufferPtr<const TBinarizedFeature> features, TCudaBufferPtr<const ui32> ids,
                              ui32 statCount, ui32 binFeatureCount, TCudaBufferPtr<float> histograms)
            : Features(features)
            , Ids(ids)
            , StatCount(statCount)
            , BinFeatureCount(binFeatureCount)
            , Histograms(histograms)
        {
        }

        Y_SAVELOAD_DEFINE(Features, Ids, StatCount, BinFeatureCount, Histograms);

        void Run(const TCudaStream& stream) const {
            NKernel::ScanHistgorams(Features.Get(), Features.Size(),
                                    Ids.Get(), Ids.Size(),
                                    StatCount, BinFeatureCount, Histograms.Get(),
                                    stream.GetStream());
        }
    };

    class TComputeHistKernelLoadByIndex: public TStatelessKernel {
    private:
        NCatboostCuda::EFeaturesGroupingPolicy Policy;
        int MaxBins;
        TCudaBufferPtr<const TFeatureInBlock> Groups;
        TCudaBufferPtr<const TDataPartition> Parts;
        TCudaBufferPtr<const ui32> PartIds;
        TCudaBufferPtr<const ui32> Cindex;
        TCudaBufferPtr<const ui32> Indices;
        TCudaBufferPtr<const float> Stats;
        TCudaBufferPtr<float> Histograms;

    public:
        TComputeHistKernelLoadByIndex() = default;

        TComputeHistKernelLoadByIndex(NCatboostCuda::EFeaturesGroupingPolicy policy,
                                      int maxBins,
                                      TCudaBufferPtr<const TFeatureInBlock> groups,
                                      TCudaBufferPtr<const TDataPartition> parts,
                                      TCudaBufferPtr<const ui32> partIds,
                                      TCudaBufferPtr<const ui32> cindex,
                                      TCudaBufferPtr<const ui32> indices,
                                      TCudaBufferPtr<const float> stats,
                                      TCudaBufferPtr<float> histograms)
            : Policy(policy)
            , MaxBins(maxBins)
            , Groups(groups)
            , Parts(parts)
            , PartIds(partIds)
            , Cindex(cindex)
            , Indices(indices)
            , Stats(stats)
            , Histograms(histograms)
        {
        }

        Y_SAVELOAD_DEFINE(Policy, MaxBins, Groups, Parts, PartIds, Cindex, Indices, Stats, Histograms);

        void Run(const TCudaStream& stream) const {
            if (Policy == NCatboostCuda::EFeaturesGroupingPolicy::BinaryFeatures) {
                NKernel::ComputeHistBinary(Groups.Get(),
                                           Groups.Size(),
                                           Parts.Get(),
                                           PartIds.Get(),
                                           PartIds.Size(),
                                           Cindex.Get(),
                                           reinterpret_cast<const int*>(Indices.Get()),
                                           Stats.Get(),
                                           Stats.GetColumnCount(),
                                           Stats.AlignedColumnSize(),
                                           Histograms.Get(),
                                           stream.GetStream());

            } else if (Policy == NCatboostCuda::EFeaturesGroupingPolicy::HalfByteFeatures) {
                NKernel::ComputeHistHalfByte(Groups.Get(),
                                             Groups.Size(),
                                             Parts.Get(),
                                             PartIds.Get(),
                                             PartIds.Size(),
                                             Cindex.Get(),
                                             reinterpret_cast<const int*>(Indices.Get()),
                                             Stats.Get(),
                                             Stats.GetColumnCount(),
                                             Stats.AlignedColumnSize(),
                                             Histograms.Get(),
                                             stream.GetStream());

            } else {
                CB_ENSURE(Policy == NCatboostCuda::EFeaturesGroupingPolicy::OneByteFeatures);
                NKernel::ComputeHistOneByte(MaxBins,
                                            Groups.Get(),
                                            Groups.Size(),
                                            Parts.Get(),
                                            PartIds.Get(),
                                            PartIds.Size(),
                                            Cindex.Get(),
                                            reinterpret_cast<const int*>(Indices.Get()),
                                            Stats.Get(),
                                            Stats.GetColumnCount(),
                                            Stats.AlignedColumnSize(),
                                            Histograms.Get(),
                                            stream.GetStream());
            }
        }
    };

    class TComputeHistKernelGatherBins: public TStatelessKernel {
    private:
        NCatboostCuda::EFeaturesGroupingPolicy Policy;
        int MaxBins;
        TCudaBufferPtr<const TFeatureInBlock> FeaturesInBlock;
        TCudaBufferPtr<const TDataPartition> Parts;
        TCudaBufferPtr<const ui32> PartIds;

        TCudaBufferPtr<const ui32> Cindex;
        TCudaBufferPtr<const ui32> Indices;
        TCudaBufferPtr<const float> Stats;

        TCudaBufferPtr<ui32> TempIndex;
        TCudaBufferPtr<float> Histograms;

    public:
        TComputeHistKernelGatherBins() = default;

        TComputeHistKernelGatherBins(NCatboostCuda::EFeaturesGroupingPolicy policy,
                                     int maxBins,
                                     TCudaBufferPtr<const TFeatureInBlock> groups,
                                     TCudaBufferPtr<const TDataPartition> parts,
                                     TCudaBufferPtr<const ui32> partIds,
                                     TCudaBufferPtr<const ui32> cindex,
                                     TCudaBufferPtr<const ui32> indices,
                                     TCudaBufferPtr<const float> stats,
                                     TCudaBufferPtr<ui32> tempBins,
                                     TCudaBufferPtr<float> histograms)
            : Policy(policy)
            , MaxBins(maxBins)
            , FeaturesInBlock(groups)
            , Parts(parts)
            , PartIds(partIds)
            , Cindex(cindex)
            , Indices(indices)
            , Stats(stats)
            , TempIndex(tempBins)
            , Histograms(histograms)
        {
        }

         Y_SAVELOAD_DEFINE(Policy, MaxBins, FeaturesInBlock, Parts, PartIds, Cindex, Indices, Stats, Histograms, TempIndex);

        void Run(const TCudaStream& stream) const {
            const int featuresPerInt = NCatboostCuda::GetFeaturesPerInt(Policy);
            NKernel::GatherCompressedIndex(FeaturesInBlock.Get(),
                                           FeaturesInBlock.Size(),
                                           featuresPerInt,
                                           Parts.Get(),
                                           PartIds.Get(),
                                           PartIds.Size(),
                                           Indices.Get(),
                                           Cindex.Get(),
                                           TempIndex.AlignedColumnSize(),
                                           TempIndex.Get(),
                                           stream.GetStream());

            if (Policy == NCatboostCuda::EFeaturesGroupingPolicy::BinaryFeatures) {
                NKernel::ComputeHistBinary(FeaturesInBlock.Get(),
                                           FeaturesInBlock.Size(),
                                           Parts.Get(),
                                           PartIds.Get(),
                                           PartIds.Size(),
                                           TempIndex.Get(),
                                           TempIndex.AlignedColumnSize(),
                                           Stats.Get(),
                                           Stats.GetColumnCount(),
                                           Stats.AlignedColumnSize(),
                                           Histograms.Get(),
                                           stream.GetStream());

            } else if (Policy == NCatboostCuda::EFeaturesGroupingPolicy::HalfByteFeatures) {
                NKernel::ComputeHistHalfByte(FeaturesInBlock.Get(),
                                             FeaturesInBlock.Size(),
                                             Parts.Get(),
                                             PartIds.Get(),
                                             PartIds.Size(),
                                             TempIndex.Get(),
                                             TempIndex.AlignedColumnSize(),
                                             Stats.Get(),
                                             Stats.GetColumnCount(),
                                             Stats.AlignedColumnSize(),
                                             Histograms.Get(),
                                             stream.GetStream());

            } else {
                CB_ENSURE(Policy == NCatboostCuda::EFeaturesGroupingPolicy::OneByteFeatures);
                NKernel::ComputeHistOneByte(MaxBins,
                                            FeaturesInBlock.Get(),
                                            FeaturesInBlock.Size(),
                                            Parts.Get(),
                                            PartIds.Get(),
                                            PartIds.Size(),
                                            TempIndex.Get(),
                                            TempIndex.AlignedColumnSize(),
                                            Stats.Get(),
                                            Stats.GetColumnCount(),
                                            Stats.AlignedColumnSize(),
                                            Histograms.Get(),
                                            stream.GetStream());
            }
        }
    };
}

inline void WriteInitPartitions(const TCudaBuffer<ui32, NCudaLib::TStripeMapping>& indices,
                                TCudaBuffer<TDataPartition, NCudaLib::TStripeMapping, NCudaLib::EPtrType::CudaHost>* partsCpu,
                                TCudaBuffer<TDataPartition, NCudaLib::TStripeMapping>* partsGpu,
                                ui32 stream = 0) {
    using TKernel = NKernelHost::TWriteInitPartitions;
    LaunchKernels<TKernel>(partsGpu->NonEmptyDevices(), stream, partsCpu, partsGpu, indices);
}

namespace NCatboostCuda {
    static TVector<ui32> NonZeroLeaves(const TPointsSubsets& subsets,
                                       const TVector<ui32>& leaves) {
        TVector<ui32> result;
        for (const auto leafId : leaves) {
            auto& leaf = subsets.Leaves.at(leafId);
            if (leaf.Size != 0) {
                result.push_back(leafId);
            }
        }
        return result;
    }

    static TVector<ui32> ZeroLeaves(const TPointsSubsets& subsets,
                                    const TVector<ui32>& leaves) {
        TVector<ui32> result;
        for (const auto& leafId : leaves) {
            auto& leaf = subsets.Leaves.at(leafId);
            if (leaf.Size == 0) {
                result.push_back(leafId);
            }
        }
        return result;
    }

    static inline TLeaf SplitLeaf(const TLeaf& leaf,
                                  const TBinarySplit& split,
                                  ESplitValue direction) {
        TLeaf newLeaf;
        newLeaf.Size = 0;
        newLeaf.Path = leaf.Path;
        newLeaf.Path.AddSplit(split, direction);
        if (leaf.HistogramsType == EHistogramsType::CurrentPath) {
            newLeaf.HistogramsType = EHistogramsType::PreviousPath;
        }
        newLeaf.BestSplit.Reset();
        return newLeaf;
    }

    inline static void RebuildLeavesSizes(TPointsSubsets* subsets) {
        TVector<TDataPartition> partsCpu;
        auto currentParts = NCudaLib::ParallelStripeView(subsets->PartitionsCpu, TSlice(0, subsets->Leaves.size()));
        currentParts.Read(partsCpu);
        const ui32 devCount = static_cast<const ui32>(NCudaLib::GetCudaManager().GetDeviceCount());

        for (size_t i = 0; i < subsets->Leaves.size(); ++i) {
            ui32 partSize = 0;
            for (ui32 dev = 0; dev < devCount; ++dev) {
                partSize += partsCpu[i + dev * subsets->Leaves.size()].Size;
            }
            subsets->Leaves[i].Size = partSize;
        }
    }

    /* this one split points. Everything, except histograms will be good after */
    void TSplitPropertiesHelper::MakeSplit(const TVector<ui32>& leavesToSplit,
                                           TPointsSubsets* subsets) {
        const ui32 leavesCount = static_cast<const ui32>(subsets->Leaves.size());
        const ui32 newLeaves = static_cast<const ui32>(leavesToSplit.size());
        auto& profiler = NCudaLib::GetProfiler();
        auto guard = profiler.Profile(TStringBuilder() << "Leaves split #" << newLeaves << " leaves");
        if (newLeaves == 0) {
            return;
        }

        subsets->Leaves.resize(leavesCount + newLeaves);

        NCudaLib::TParallelStripeVectorBuilder<TCFeature> splitsFeaturesBuilder;

        TVector<ui32> leftIds;
        TVector<ui32> rightIds;
        TVector<ui32> splitBins;

        for (size_t i = 0; i < newLeaves; ++i) {
            const ui32 leftId = leavesToSplit[i];
            const ui32 rightId = static_cast<const ui32>(leavesCount + i);

            TLeaf leaf = subsets->Leaves[leftId];

            Y_VERIFY(subsets->Leaves[leftId].BestSplit.Defined());

            TBinarySplit splitFeature = ToSplit(FeaturesManager,
                                                subsets->Leaves[leftId].BestSplit);

            TLeaf left = SplitLeaf(leaf, splitFeature, ESplitValue::Zero);
            TLeaf right = SplitLeaf(leaf, splitFeature, ESplitValue::One);

            subsets->Leaves[leftId] = left;
            subsets->Leaves[rightId] = right;

            splitsFeaturesBuilder.Add(DataSet.GetTCFeature(splitFeature.FeatureId));
            splitBins.push_back(splitFeature.BinIdx);

            leftIds.push_back(leftId);
            rightIds.push_back(rightId);
        }


        auto splitBinsGpu = TMirrorBuffer<ui32>::Create(NCudaLib::TMirrorMapping(splitBins.size()));
        splitBinsGpu.Write(splitBins);

        TStripeBuffer<TCFeature> splitFeaturesGpu;
        splitsFeaturesBuilder.Build(splitFeaturesGpu);

        auto leftIdsGpu = TMirrorBuffer<ui32>::Create(NCudaLib::TMirrorMapping(leftIds.size()));
        auto leftIdsCpu = NCudaLib::TCudaBuffer<ui32, NCudaLib::TMirrorMapping, NCudaLib::EPtrType::CudaHost>::Create(NCudaLib::TMirrorMapping(leftIds.size()));

        leftIdsCpu.Write(leftIds);
        leftIdsGpu.Copy(leftIdsCpu);

        auto rightIdsGpu = TMirrorBuffer<ui32>::Create(NCudaLib::TMirrorMapping(rightIds.size()));
        rightIdsGpu.Write(rightIds);

        /*
         * Allocates temp memory
         * Makes segmented sequence
         * Write parts flags
         * Makes segmented sort
         * Segmented gather for each stats + indices
         * Update part offsets and sizes
         * Update part stats
         * copy histograms for new leaves so we could skip complex "choose leaf to compute" logic
         */
        {
            using TKernel = NKernelHost::TSplitPointsKernel;

            const ui32 statsPerKernel = 8;

            LaunchKernels<TKernel>(subsets->Partitions.NonEmptyDevices(),
                                   0,
                                   statsPerKernel,
                                   DataSet.GetCompressedIndex().GetStorage(),
                                   splitFeaturesGpu,
                                   splitBinsGpu,
                                   leftIdsGpu,
                                   rightIdsGpu,
                                   leftIdsCpu,
                                   subsets->Partitions,
                                   subsets->PartitionsCpu,
                                   subsets->PartitionStats,
                                   subsets->Target.Indices,
                                   subsets->Target.StatsToAggregate,
                                   ComputeByBlocksHelper.BinFeatureCount(),
                                   subsets->Histograms);
        }

        //here faster to read everything
        RebuildLeavesSizes(subsets);
    }

    TPointsSubsets TSplitPropertiesHelper::CreateInitialSubsets(TOptimizationTarget&& target, ui32 maxLeaves) {
        TPointsSubsets subsets;
        subsets.Target = std::move(target);

        const ui32 statsDim = static_cast<const ui32>(subsets.Target.StatsToAggregate.GetColumnCount());
        auto partsMapping = NCudaLib::TStripeMapping::RepeatOnAllDevices(maxLeaves);

        subsets.Partitions.Reset(partsMapping);
        subsets.PartitionsCpu.Reset(partsMapping);

        WriteInitPartitions(subsets.Target.Indices,
                            &subsets.PartitionsCpu,
                            &subsets.Partitions);

        auto partStatsMapping = NCudaLib::TStripeMapping::RepeatOnAllDevices(maxLeaves, statsDim);
        subsets.PartitionStats = TStripeBuffer<double>::Create(partStatsMapping);
        FillBuffer(subsets.PartitionStats, 0.0);

        ComputeByBlocksHelper.ResetHistograms(statsDim, maxLeaves, &subsets.Histograms);
        FillBuffer(subsets.Histograms, 0.0f);

        subsets.BinFeatures = ComputeByBlocksHelper.GetBinFeatures().ConstCopyView();

        auto partIds = TMirrorBuffer<ui32>::Create(NCudaLib::TMirrorMapping(1));
        FillBuffer(partIds, 0u);

        ComputePartitionStats(subsets.Target.StatsToAggregate,
                              subsets.Partitions,
                              partIds,
                              &subsets.PartitionStats);

        subsets.Leaves.push_back(TLeaf());
        RebuildLeavesSizes(&subsets);
        return subsets;
    }

    /*
     * This one don't need to know anything about update/from scratch
     * just compute histograms in blocks for leaves with load policy
     */
    void TSplitPropertiesHelper::ComputeSplitProperties(const ELoadFromCompressedIndexPolicy loadPolicy,
                                                        const TVector<ui32>& leavesToCompute,
                                                        TPointsSubsets* subsetsPtr) {

        auto& subsets = *subsetsPtr;
        ui32 activeStreamsCount = Min<ui32>(MaxStreamCount, ComputeByBlocksHelper.GetBlockCount());

        const ui32 statsCount = static_cast<const ui32>(subsets.Target.StatsToAggregate.GetColumnCount());
        const ui32 leavesCount = static_cast<const ui32>(leavesToCompute.size());
        auto leavesGpu = TMirrorBuffer<ui32>::Create(NCudaLib::TMirrorMapping(leavesCount));
        leavesGpu.Write(leavesToCompute);


        TVector<TStripeBuffer<float>> tempHistograms(activeStreamsCount);
        TVector<TStripeBuffer<ui32>> tempGatheredCompressedIndex(activeStreamsCount);

        //compute max memory so we don't need deallocation
        {
            using TMappingBuilder = NCudaLib::TMappingBuilder<NCudaLib::TStripeMapping>;
            TVector<TMappingBuilder> tempHistogramsMappingBuilder(activeStreamsCount);

            const auto devCount = NCudaLib::GetCudaManager().GetDeviceCount();

            for (ui32 blockId = 0; blockId < ComputeByBlocksHelper.GetBlockCount(); ++blockId) {
                auto histMapping = ComputeByBlocksHelper.BlockHistogramsMapping(blockId, leavesCount, statsCount);
                for (ui32 dev = 0; dev < devCount; ++dev) {
                    tempHistogramsMappingBuilder[blockId % activeStreamsCount].UpdateMaxSizeAt(dev,
                                                                                               histMapping.DeviceSlice(dev).Size());
                }
            }

            for (ui64 tempHistId = 0; tempHistId < tempHistograms.size(); ++tempHistId) {
                tempHistograms[tempHistId].Reset(tempHistogramsMappingBuilder[tempHistId].Build());
            }
        }

        if (loadPolicy == ELoadFromCompressedIndexPolicy::GatherBins) {
            TVector<ui32> maxBlockSizes(activeStreamsCount);

            for (ui32 blockId = 0; blockId < ComputeByBlocksHelper.GetBlockCount(); ++blockId) {
                maxBlockSizes[blockId % activeStreamsCount] = Max<ui32>(ComputeByBlocksHelper.GetIntsPerSample(blockId),
                                                                        maxBlockSizes[blockId % activeStreamsCount]);
            }

            for (ui32 i = 0; i < activeStreamsCount; ++i) {
                tempGatheredCompressedIndex[i].Reset(subsets.Target.Indices.GetMapping(), maxBlockSizes[i]);
            }
        }

        NCudaLib::GetCudaManager().Barrier();

        for (ui32 blockId = 0; blockId < ComputeByBlocksHelper.GetBlockCount(); ++blockId) {
            ui32 streamId = GetStream(blockId);
            auto policy = ComputeByBlocksHelper.GetBlockPolicy(blockId);
            auto blockFeatures = ComputeByBlocksHelper.GetBlockFeatures(blockId);
            const int maxBins = ComputeByBlocksHelper.GetBlockHistogramMaxBins(blockId);
            auto& blockHistograms = tempHistograms[blockId % activeStreamsCount];

            blockHistograms.Reset(ComputeByBlocksHelper.BlockHistogramsMapping(blockId, leavesCount, statsCount));
            FillBuffer(blockHistograms, 0.0f, streamId);

            if (loadPolicy == ELoadFromCompressedIndexPolicy::LoadByIndexBins) {
                using TKernel = NKernelHost::TComputeHistKernelLoadByIndex;
                LaunchKernels<TKernel>(blockHistograms.NonEmptyDevices(),
                                       streamId,
                                       policy,
                                       maxBins,
                                       blockFeatures,
                                       subsets.Partitions,
                                       leavesGpu,
                                       DataSet.GetCompressedIndex().GetStorage(),
                                       subsets.Target.Indices,
                                       subsets.Target.StatsToAggregate,
                                       blockHistograms);

            } else {
                CB_ENSURE(loadPolicy == ELoadFromCompressedIndexPolicy::GatherBins);
                using TKernel = NKernelHost::TComputeHistKernelGatherBins;

                LaunchKernels<TKernel>(blockHistograms.NonEmptyDevices(),
                                       streamId,
                                       policy,
                                       maxBins,
                                       blockFeatures,
                                       subsets.Partitions,
                                       leavesGpu,
                                       DataSet.GetCompressedIndex().GetStorage(),
                                       subsets.Target.Indices,
                                       subsets.Target.StatsToAggregate,
                                       tempGatheredCompressedIndex[blockId % activeStreamsCount],
                                       blockHistograms);
            }

            auto reducedMapping = ComputeByBlocksHelper.ReducedBlockHistogramsMapping(blockId, leavesCount, statsCount);
            ReduceScatter(blockHistograms,
                          reducedMapping,
                          false,
                          streamId);
            {
                using TKernel = NKernelHost::TWriteReducesHistogramsKernel;
                LaunchKernels<TKernel>(subsets.Histograms.NonEmptyDevices(),
                                       streamId,
                                       ComputeByBlocksHelper.GetWriteOffset(blockId),
                                       ComputeByBlocksHelper.GetWriteSizes(blockId),
                                       leavesGpu,
                                       statsCount,
                                       blockHistograms,
                                       ComputeByBlocksHelper.BinFeatureCount(),
                                       subsets.Histograms);
            }
        }

        NCudaLib::GetCudaManager().Barrier();

        {
            using TKernel = NKernelHost::TScanHistgoramsKernel;
            LaunchKernels<TKernel>(subsets.Histograms.NonEmptyDevices(),
                                   0,
                                   ComputeByBlocksHelper.GetFeatures(),
                                   leavesGpu,
                                   subsets.GetStatCount(),
                                   ComputeByBlocksHelper.BinFeatureCount(),
                                   subsets.Histograms);
        }
    }

    void TSplitPropertiesHelper::BuildNecessaryHistograms(TPointsSubsets* subsets) {
        auto& profiler = NCudaLib::GetProfiler();
        auto& leaves = subsets->Leaves;

        //to compute hists
        TVector<ui32> computeLeaves;
        //with substraction
        TVector<ui32> smallLeaves;
        TVector<ui32> bigLeaves;

        THashMap<TLeafPath, TVector<ui32>> rebuildLeaves;

        for (size_t i = 0; i < leaves.size(); ++i) {
            const auto& leaf = leaves[i];

            if (leaf.HistogramsType == EHistogramsType::PreviousPath) {
                auto prevPath = PreviousSplit(leaf.Path);
                rebuildLeaves[prevPath].push_back(i);
            } else if (leaf.HistogramsType == EHistogramsType::Zeroes) {
                computeLeaves.push_back(i);
            }
        }

        for (auto& rebuildLeavesPair : rebuildLeaves) {
            auto& ids = rebuildLeavesPair.second;
            CB_ENSURE(ids.size() == 2 || ids.size() == 1);
            if (ids.size() == 1) {
                const ui32 leafId = ids[0];
                CB_ENSURE(subsets->Leaves[leafId].IsTerminal, "Error: this leaf should be terminal");
            } else {
                const auto& firstLeaf = leaves[ids[0]];
                const auto& secondLeaf = leaves[ids[1]];
                ui32 smallLeafId = 0;
                ui32 bigLeafId = 0;

                if (firstLeaf.Size < secondLeaf.Size) {
                    smallLeafId = ids[0];
                    bigLeafId = ids[1];
                } else {
                    smallLeafId = ids[1];
                    bigLeafId = ids[0];
                }

                if (subsets->Leaves[smallLeafId].IsTerminal && subsets->Leaves[bigLeafId].IsTerminal) {
                    continue;
                }

                smallLeaves.push_back(smallLeafId);
                computeLeaves.push_back(smallLeafId);
                bigLeaves.push_back(bigLeafId);
            }
        }
        auto guard = profiler.Profile(TStringBuilder() << "Compute histograms for #" << subsets->Leaves.size() << " leaves");

        //TODO(noxoomo): load by index for 2 stats
        const ELoadFromCompressedIndexPolicy loadPolicy = subsets->Leaves.size() == 1 || subsets->GetStatCount() <= 2
                                                              ? ELoadFromCompressedIndexPolicy::LoadByIndexBins
                                                              : ELoadFromCompressedIndexPolicy::GatherBins;

        auto nonZeroComputeLeaves = NonZeroLeaves(*subsets, computeLeaves);

        auto zeroLeaves = ZeroLeaves(*subsets,
                                     computeLeaves);

        ComputeSplitProperties(loadPolicy,
                               nonZeroComputeLeaves,
                               subsets);

        ZeroLeavesHistograms(zeroLeaves,
                             subsets);

        SubstractHistograms(bigLeaves,
                            smallLeaves,
                            subsets);

        {
            TVector<ui32> allUpdatedLeaves;

            allUpdatedLeaves.insert(allUpdatedLeaves.end(),
                                    computeLeaves.begin(),
                                    computeLeaves.end());

            allUpdatedLeaves.insert(allUpdatedLeaves.end(),
                                    bigLeaves.begin(),
                                    bigLeaves.end());
            for (auto leafId : allUpdatedLeaves) {
                subsets->Leaves[leafId].HistogramsType = EHistogramsType::CurrentPath;
                subsets->Leaves[leafId].BestSplit.Reset();
            }
        }
    }

    void TSplitPropertiesHelper::ZeroLeavesHistograms(const TVector<ui32>& leaves,
                                                      TPointsSubsets* subsets) {
        auto ids = TMirrorBuffer<ui32>::Create(NCudaLib::TMirrorMapping(leaves.size()));
        ids.Write(leaves);
        using TKernel = NKernelHost::TZeroHistogramsKernel;

        LaunchKernels<TKernel>(subsets->Histograms.NonEmptyDevices(),
                               0,
                               ids,
                               subsets->GetStatCount(),
                               ComputeByBlocksHelper.BinFeatureCount(),
                               subsets->Histograms);
    }

    void TSplitPropertiesHelper::SubstractHistograms(const TVector<ui32>& from,
                                                     const TVector<ui32>& what,
                                                     TPointsSubsets* subsets) {
        Y_VERIFY(from.size() == what.size());

        auto fromIds = TMirrorBuffer<ui32>::Create(NCudaLib::TMirrorMapping(from.size()));
        auto whatIds = TMirrorBuffer<ui32>::Create(NCudaLib::TMirrorMapping(what.size()));

        fromIds.Write(from);
        whatIds.Write(what);

        using TKernel = NKernelHost::TSubstractHistgoramsKernel;

        LaunchKernels<TKernel>(subsets->Histograms.NonEmptyDevices(),
                               0,
                               fromIds,
                               whatIds,
                               subsets->GetStatCount(),
                               ComputeByBlocksHelper.BinFeatureCount(),
                               subsets->Histograms);
    }

}

namespace NCudaLib {
    REGISTER_KERNEL(0xD2DAA0, NKernelHost::TWriteInitPartitions);
    REGISTER_KERNEL(0xD2DAA1, NKernelHost::TCopyHistogramsKernel);
    REGISTER_KERNEL(0xD2DAA2, NKernelHost::TWriteReducesHistogramsKernel);
    REGISTER_KERNEL(0xD2DAA3, NKernelHost::TZeroHistogramsKernel);
    REGISTER_KERNEL(0xD2DAA4, NKernelHost::TScanHistgoramsKernel);
    REGISTER_KERNEL(0xD2DAA5, NKernelHost::TComputeHistKernelLoadByIndex);
    REGISTER_KERNEL(0xD2DAA6, NKernelHost::TComputeHistKernelGatherBins);
    REGISTER_KERNEL(0xD2DAA7, NKernelHost::TSubstractHistgoramsKernel);
}
