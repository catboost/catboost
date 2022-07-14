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
#include <catboost/cuda/methods/greedy_subsets_searcher/kernel/hist_single_leaf.cuh>
#include <catboost/cuda/methods/greedy_subsets_searcher/kernel/gather_bins.cuh>

namespace NKernelHost {
    template <class T>
    class TZeroBuffer: public TStatelessKernel {
    private:
        TCudaBufferPtr<T> Buffer;

    public:
        TZeroBuffer(TCudaBufferPtr<T> ptr)
            : Buffer(ptr)
        {
        }

        TZeroBuffer() = default;

        Y_SAVELOAD_DEFINE(Buffer);

        void Run(const TCudaStream& stream) const {
            ui64 size = Buffer.Size() * sizeof(T);
            void* ptr = Buffer.Get();

            CUDA_SAFE_CALL(cudaMemsetAsync(ptr, 0, size, stream.GetStream()));
        }
    };

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

    class TWriteReducesHistogramKernel: public TStatelessKernel {
    private:
        ui32 BlockOffset;
        ui32 HistBlockSize;
        ui32 HistogramId;
        ui32 StatCount;
        TCudaBufferPtr<const float> BlockHistogram;
        ui32 BinFeatureCount;
        TCudaBufferPtr<float> DstHistogram;

    public:
        TWriteReducesHistogramKernel() = default;

        TWriteReducesHistogramKernel(ui32 blockOffset,
                                     ui32 histBlockSize,
                                     ui32 histogramId,
                                     ui32 statCount,
                                     TCudaBufferPtr<const float> blockHistogram,
                                     ui32 binFeatureCount,
                                     TCudaBufferPtr<float> dstHistogram)
            : BlockOffset(blockOffset)
            , HistBlockSize(histBlockSize)
            , HistogramId(histogramId)
            , StatCount(statCount)
            , BlockHistogram(blockHistogram)
            , BinFeatureCount(binFeatureCount)
            , DstHistogram(dstHistogram)
        {
        }

        Y_SAVELOAD_DEFINE(BlockOffset, HistBlockSize, HistogramId, StatCount, BlockHistogram, BinFeatureCount,
                          DstHistogram);

        void Run(const TCudaStream& stream) const {
            NKernel::WriteReducesHistogram(BlockOffset, HistBlockSize, HistogramId,
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

    class TZeroHistogramKernel: public TStatelessKernel {
    private:
        ui32 HistId;
        ui32 StatCount;
        ui32 BinFeatureCount;
        TCudaBufferPtr<float> DstHistogram;

    public:
        TZeroHistogramKernel() = default;

        TZeroHistogramKernel(ui32 histId,
                             ui32 statCount,
                             ui32 binFeatureCount,
                             TCudaBufferPtr<float> dstHistogram)
            : HistId(histId)
            , StatCount(statCount)
            , BinFeatureCount(binFeatureCount)
            , DstHistogram(dstHistogram)
        {
        }

        Y_SAVELOAD_DEFINE(HistId, StatCount, BinFeatureCount, DstHistogram);

        void Run(const TCudaStream& stream) const {
            NKernel::ZeroHistogram(HistId, StatCount, BinFeatureCount, DstHistogram.Get(),
                                   stream.GetStream());
        }
    };

    class TSubstractHistogramsKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const ui32> FromIds;
        TCudaBufferPtr<const ui32> WhatIds;
        ui32 StatCount;
        ui32 BinFeatureCount;
        TCudaBufferPtr<float> DstHistogram;

    public:
        TSubstractHistogramsKernel() = default;

        TSubstractHistogramsKernel(TCudaBufferPtr<const ui32> fromIds, TCudaBufferPtr<const ui32> whatIds,
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

    class TSubstractHistogramKernel: public TStatelessKernel {
    private:
        ui32 FromId;
        ui32 WhatId;
        ui32 StatCount;
        ui32 BinFeatureCount;
        TCudaBufferPtr<float> DstHistogram;

    public:
        TSubstractHistogramKernel() = default;

        TSubstractHistogramKernel(ui32 fromIds, ui32 whatIds,
                                  ui32 statCount, ui32 binFeatureCount, TCudaBufferPtr<float> dstHistogram)
            : FromId(fromIds)
            , WhatId(whatIds)
            , StatCount(statCount)
            , BinFeatureCount(binFeatureCount)
            , DstHistogram(dstHistogram)
        {
        }

        Y_SAVELOAD_DEFINE(FromId, WhatId, StatCount, BinFeatureCount, DstHistogram);

        void Run(const TCudaStream& stream) const {
            NKernel::SubstractHistgoram(FromId, WhatId, StatCount, BinFeatureCount,
                                        DstHistogram.Get(), stream.GetStream());
        }
    };

    class TScanHistogramsKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const TBinarizedFeature> Features;
        TCudaBufferPtr<const ui32> Ids;
        ui32 StatCount;
        ui32 BinFeatureCount;
        TCudaBufferPtr<float> Histograms;

    public:
        TScanHistogramsKernel() = default;

        TScanHistogramsKernel(TCudaBufferPtr<const TBinarizedFeature> features, TCudaBufferPtr<const ui32> ids,
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
            NKernel::ScanHistograms(Features.Get(), Features.Size(),
                                    Ids.Get(), Ids.Size(),
                                    StatCount, BinFeatureCount, Histograms.Get(),
                                    stream.GetStream());
        }
    };

    class TScanHistogramKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const TBinarizedFeature> Features;
        ui32 Id;
        ui32 StatCount;
        ui32 BinFeatureCount;
        TCudaBufferPtr<float> Histograms;

    public:
        TScanHistogramKernel() = default;

        TScanHistogramKernel(TCudaBufferPtr<const TBinarizedFeature> features,
                             ui32 id,
                             ui32 statCount, ui32 binFeatureCount, TCudaBufferPtr<float> histograms)
            : Features(features)
            , Id(id)
            , StatCount(statCount)
            , BinFeatureCount(binFeatureCount)
            , Histograms(histograms)
        {
        }

        Y_SAVELOAD_DEFINE(Features, Id, StatCount, BinFeatureCount, Histograms);

        void Run(const TCudaStream& stream) const {
            NKernel::ScanHistogram(Features.Get(), Features.Size(),
                                   Id,
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

    /* Single histograms */

    class TComputeSingleHistKernelLoadByIndex: public TStatelessKernel {
    private:
        NCatboostCuda::EFeaturesGroupingPolicy Policy;
        int MaxBins;
        TCudaBufferPtr<const TFeatureInBlock> Groups;
        TCudaBufferPtr<const TDataPartition> Parts;
        ui32 PartId;
        TCudaBufferPtr<const ui32> Cindex;
        TCudaBufferPtr<const ui32> Indices;
        TCudaBufferPtr<const float> Stats;
        TCudaBufferPtr<float> Histograms;

    public:
        TComputeSingleHistKernelLoadByIndex() = default;

        TComputeSingleHistKernelLoadByIndex(NCatboostCuda::EFeaturesGroupingPolicy policy,
                                            int maxBins,
                                            TCudaBufferPtr<const TFeatureInBlock> groups,
                                            TCudaBufferPtr<const TDataPartition> parts,
                                            ui32 partId,
                                            TCudaBufferPtr<const ui32> cindex,
                                            TCudaBufferPtr<const ui32> indices,
                                            TCudaBufferPtr<const float> stats,
                                            TCudaBufferPtr<float> histograms)
            : Policy(policy)
            , MaxBins(maxBins)
            , Groups(groups)
            , Parts(parts)
            , PartId(partId)
            , Cindex(cindex)
            , Indices(indices)
            , Stats(stats)
            , Histograms(histograms)
        {
        }

        Y_SAVELOAD_DEFINE(Policy, MaxBins, Groups, Parts, PartId, Cindex, Indices, Stats, Histograms);

        void Run(const TCudaStream& stream) const {
            if (Policy == NCatboostCuda::EFeaturesGroupingPolicy::BinaryFeatures) {
                NKernel::ComputeHistBinary(Groups.Get(),
                                           Groups.Size(),
                                           Parts.Get(),
                                           PartId,
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
                                             PartId,
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
                                            PartId,
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

    class TComputeSingleHistKernelGatherBins: public TStatelessKernel {
    private:
        NCatboostCuda::EFeaturesGroupingPolicy Policy;
        int MaxBins;
        TCudaBufferPtr<const TFeatureInBlock> FeaturesInBlock;
        TCudaBufferPtr<const TDataPartition> Parts;
        ui32 PartId;

        TCudaBufferPtr<const ui32> Cindex;
        TCudaBufferPtr<const ui32> Indices;
        TCudaBufferPtr<const float> Stats;

        TCudaBufferPtr<ui32> TempIndex;
        TCudaBufferPtr<float> Histograms;

    public:
        TComputeSingleHistKernelGatherBins() = default;

        TComputeSingleHistKernelGatherBins(NCatboostCuda::EFeaturesGroupingPolicy policy,
                                           int maxBins,
                                           TCudaBufferPtr<const TFeatureInBlock> groups,
                                           TCudaBufferPtr<const TDataPartition> parts,
                                           ui32 partId,
                                           TCudaBufferPtr<const ui32> cindex,
                                           TCudaBufferPtr<const ui32> indices,
                                           TCudaBufferPtr<const float> stats,
                                           TCudaBufferPtr<ui32> tempBins,
                                           TCudaBufferPtr<float> histograms)
            : Policy(policy)
            , MaxBins(maxBins)
            , FeaturesInBlock(groups)
            , Parts(parts)
            , PartId(partId)
            , Cindex(cindex)
            , Indices(indices)
            , Stats(stats)
            , TempIndex(tempBins)
            , Histograms(histograms)
        {
        }

        Y_SAVELOAD_DEFINE(Policy, MaxBins, FeaturesInBlock, Parts, PartId, Cindex, Indices, Stats, Histograms, TempIndex);

        void Run(const TCudaStream& stream) const {
            const int featuresPerInt = NCatboostCuda::GetFeaturesPerInt(Policy);
            NKernel::GatherCompressedIndex(FeaturesInBlock.Get(),
                                           FeaturesInBlock.Size(),
                                           featuresPerInt,
                                           Parts.Get(),
                                           PartId,
                                           Indices.Get(),
                                           Cindex.Get(),
                                           TempIndex.AlignedColumnSize(),
                                           TempIndex.Get(),
                                           stream.GetStream());

            if (Policy == NCatboostCuda::EFeaturesGroupingPolicy::BinaryFeatures) {
                NKernel::ComputeHistBinary(FeaturesInBlock.Get(),
                                           FeaturesInBlock.Size(),
                                           Parts.Get(),
                                           PartId,
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
                                             PartId,
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
                                            PartId,
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

template <typename T, class TMapping>
void ZeroBuffer(
    NCudaLib::TCudaBuffer<T, TMapping>& buffer,
    ui32 streamId = 0) {
    using TKernel = NKernelHost::TZeroBuffer<T>;
    LaunchKernels<TKernel>(buffer.NonEmptyDevices(), streamId, buffer);
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

    inline static void FastUpdateLeavesSizes(const TVector<ui32>& ids,
                                             TPointsSubsets* subsets) {
        const ui32 devCount = static_cast<const ui32>(NCudaLib::GetCudaManager().GetDeviceCount());

        for (const auto newId : ids) {
            TVector<TDataPartition> partsCpu;
            auto currentParts = NCudaLib::ParallelStripeView(subsets->PartitionsCpu, TSlice(newId, newId + 1));
            currentParts.Read(partsCpu);

            ui32 partSize = 0;
            for (ui32 dev = 0; dev < devCount; ++dev) {
                partSize += partsCpu[dev].Size;
            }
            subsets->Leaves[newId].Size = partSize;
        }
    }

    /* this one split points. Everything, except histograms will be good after */
    void TSplitPropertiesHelper::MakeSplit(const TVector<ui32>& leavesToSplit,
                                           TPointsSubsets* subsets,
                                           TVector<ui32>* leftIdsPtr,
                                           TVector<ui32>* rightIdsPtr) {
        const ui32 leavesCount = static_cast<const ui32>(subsets->Leaves.size());
        const ui32 newLeaves = static_cast<const ui32>(leavesToSplit.size());
        if (newLeaves == 1) {
            MakeSplit(leavesToSplit[0], subsets, leftIdsPtr, rightIdsPtr);
        } else {
            auto& profiler = NCudaLib::GetProfiler();
            auto guard = profiler.Profile(TStringBuilder() << "Leaves split #" << newLeaves << " leaves");
            if (newLeaves == 0) {
                return;
            }
            subsets->Leaves.resize(leavesCount + newLeaves);

            TVector<ui32>& leftIds = *leftIdsPtr;
            TVector<ui32>& rightIds = *rightIdsPtr;
            leftIds.clear();
            rightIds.clear();

            TVector<ui32> splitBins;

            TStripeBuffer<TCFeature> splitFeaturesGpu;
            {
                NCudaLib::TParallelStripeVectorBuilder<TCFeature> splitsFeaturesBuilder;

                for (size_t i = 0; i < newLeaves; ++i) {
                    const ui32 leftId = leavesToSplit[i];
                    const ui32 rightId = static_cast<const ui32>(leavesCount + i);

                    TLeaf leaf = subsets->Leaves[leftId];

                    CB_ENSURE(subsets->Leaves[leftId].BestSplit.Defined(), "Best split is undefined for leaf " << leftId);

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
                splitsFeaturesBuilder.Build(splitFeaturesGpu);
            }

            const TSlice binsSlice = TSlice(0, splitBins.size());
            const TSlice leftIdsSlice = TSlice(binsSlice.Right, binsSlice.Right + leftIds.size());
            const TSlice rightIdsSlice = TSlice(leftIdsSlice.Right, leftIdsSlice.Right + rightIds.size());

            auto allUi32Data = TMirrorBuffer<ui32>::Create(NCudaLib::TMirrorMapping(rightIdsSlice.Right));
            auto allUi32DataCpu = TMirrorHostBuffer<ui32>::Create(NCudaLib::TMirrorMapping(rightIdsSlice.Right));

            auto splitBinsGpu = allUi32Data.SliceView(binsSlice);
            auto leftIdsGpu = allUi32Data.SliceView(leftIdsSlice);
            auto rightIdsGpu = allUi32Data.SliceView(rightIdsSlice);
            auto leftIdsCpu = allUi32DataCpu.SliceView(leftIdsSlice);

            {
                TVector<ui32> tmp;
                tmp.insert(tmp.end(), splitBins.begin(), splitBins.end());
                tmp.insert(tmp.end(), leftIds.begin(), leftIds.end());
                tmp.insert(tmp.end(), rightIds.begin(), rightIds.end());
                allUi32DataCpu.Write(tmp);
                allUi32Data.Copy(allUi32DataCpu);
            }

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
                auto guard = profiler.Profile(TStringBuilder() << "Split points kernel #" << newLeaves << " leaves");

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
            {
                //here faster to read everything
                if (leavesToSplit.size() == 1) {
                    //fast path for lossguide learning
                    CB_ENSURE(leftIds.size() == 1, "Unexpected number of left children " << leftIds.size() << " (should be 1)");
                    CB_ENSURE(rightIds.size() == 1, "Unexpected number of left children " << rightIds.size() << " (should be 1)");
                    TVector<ui32> ids = {leftIds[0], rightIds[0]};
                    FastUpdateLeavesSizes(ids, subsets);
                } else {
                    RebuildLeavesSizes(subsets);
                }
            }
        }
    }

    void TSplitPropertiesHelper::MakeSplit(
        const ui32 leafId,
        TPointsSubsets* subsets,
        TVector<ui32>* leftIdsPtr,
        TVector<ui32>* rightIdsPtr) {
        NCudaLib::GetCudaManager().DefaultStream().Synchronize();

        auto& profiler = NCudaLib::GetProfiler();
        auto guard = profiler.Profile(TStringBuilder() << "Leaves split #" << 1 << " leaves");

        const ui32 leavesCount = static_cast<const ui32>(subsets->Leaves.size());
        const ui32 newLeaves = 1;

        subsets->Leaves.resize(leavesCount + newLeaves);

        TVector<ui32>& leftIds = *leftIdsPtr;
        TVector<ui32>& rightIds = *rightIdsPtr;
        leftIds.clear();
        rightIds.clear();

        const ui32 leftId = leafId;
        const ui32 rightId = static_cast<const ui32>(leavesCount);

        TLeaf leaf = subsets->Leaves[leftId];

        CB_ENSURE(subsets->Leaves[leftId].BestSplit.Defined(), "Best split is undefined for leaf " << leftId);

        TBinarySplit binarySplit = ToSplit(FeaturesManager,
                                           subsets->Leaves[leftId].BestSplit);

        TLeaf left = SplitLeaf(leaf, binarySplit, ESplitValue::Zero);
        TLeaf right = SplitLeaf(leaf, binarySplit, ESplitValue::One);

        subsets->Leaves[leftId] = left;
        subsets->Leaves[rightId] = right;

        auto splitFeature = DataSet.GetTCFeature(binarySplit.FeatureId);
        const ui32 splitBin = binarySplit.BinIdx;

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
            auto guard = profiler.Profile(TStringBuilder() << "Split points kernel #" << newLeaves << " leaves");

            using TKernel = NKernelHost::TSplitPointsSingleLeafKernel;

            const ui32 statsPerKernel = 8;

            LaunchKernels<TKernel>(subsets->Partitions.NonEmptyDevices(),
                                   0,
                                   statsPerKernel,
                                   DataSet.GetCompressedIndex().GetStorage(),
                                   splitFeature,
                                   splitBin,
                                   leftId,
                                   rightId,
                                   subsets->Partitions,
                                   subsets->PartitionsCpu,
                                   subsets->PartitionStats,
                                   subsets->Target.Indices,
                                   subsets->Target.StatsToAggregate,
                                   ComputeByBlocksHelper.BinFeatureCount(),
                                   subsets->Histograms);
        }
        TVector<ui32> ids = {leftId, rightId};
        leftIds = {leftId};
        rightIds = {rightId};
        FastUpdateLeavesSizes(ids, subsets);
    }

    TPointsSubsets TSplitPropertiesHelper::CreateInitialSubsets(
        TOptimizationTarget&& target,
        ui32 maxLeaves,
        TConstArrayRef<float> featureWeights
    ) {
        TPointsSubsets subsets;
        subsets.Leaves.reserve(maxLeaves + 1);

        subsets.Target = std::move(target);

        const ui32 statsDim = static_cast<const ui32>(subsets.Target.StatsToAggregate.GetColumnCount());
        auto partsMapping = NCudaLib::TStripeMapping::RepeatOnAllDevices(maxLeaves);

        subsets.Partitions.Reset(partsMapping);
        subsets.PartitionsCpu.Reset(partsMapping);
        ZeroBuffer(subsets.Partitions);


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

        subsets.FeatureWeights = TMirrorBuffer<float>::Create(NCudaLib::TMirrorMapping(featureWeights.size()));
        subsets.FeatureWeights.Write(featureWeights);

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
        if (leavesToCompute.size() == 0) {
            return;
        }
        auto& subsets = *subsetsPtr;
        ui32 activeStreamsCount = Min<ui32>(MaxStreamCount, ComputeByBlocksHelper.GetBlockCount());

        const ui32 statsCount = static_cast<const ui32>(subsets.Target.StatsToAggregate.GetColumnCount());
        const ui32 leavesCount = static_cast<const ui32>(leavesToCompute.size());

        TMaybe<ui32> singleLeaf;
        TMaybe<TMirrorBuffer<ui32>> leavesGpu;
        if (leavesToCompute.size() > 1) {
            leavesGpu = TMirrorBuffer<ui32>::Create(NCudaLib::TMirrorMapping(leavesCount));
            (*leavesGpu).Write(leavesToCompute);
        } else {
            singleLeaf = leavesToCompute[0];
        }

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

        if (!IsOnlyDefaultStream()) {
            NCudaLib::GetCudaManager().Barrier();
        }

        for (ui32 blockId = 0; blockId < ComputeByBlocksHelper.GetBlockCount(); ++blockId) {
            ui32 streamId = GetStream(blockId);
            auto policy = ComputeByBlocksHelper.GetBlockPolicy(blockId);
            auto blockFeatures = ComputeByBlocksHelper.GetBlockFeatures(blockId);
            const int maxBins = ComputeByBlocksHelper.GetBlockHistogramMaxBins(blockId);
            auto& blockHistograms = tempHistograms[blockId % activeStreamsCount];

            blockHistograms.Reset(ComputeByBlocksHelper.BlockHistogramsMapping(blockId, leavesCount, statsCount));
            //            FillBuffer(blockHistograms, 0.0f, streamId);
            ZeroBuffer(blockHistograms, streamId);

            if (leavesGpu) {
                if (loadPolicy == ELoadFromCompressedIndexPolicy::LoadByIndexBins) {
                    using TKernel = NKernelHost::TComputeHistKernelLoadByIndex;
                    LaunchKernels<TKernel>(blockHistograms.NonEmptyDevices(),
                                           streamId,
                                           policy,
                                           maxBins,
                                           blockFeatures,
                                           subsets.Partitions,
                                           *leavesGpu,
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
                                           *leavesGpu,
                                           DataSet.GetCompressedIndex().GetStorage(),
                                           subsets.Target.Indices,
                                           subsets.Target.StatsToAggregate,
                                           tempGatheredCompressedIndex[blockId % activeStreamsCount],
                                           blockHistograms);
                }
            } else {
                CB_ENSURE(singleLeaf);

                if (loadPolicy == ELoadFromCompressedIndexPolicy::LoadByIndexBins) {
                    using TKernel = NKernelHost::TComputeSingleHistKernelLoadByIndex;
                    LaunchKernels<TKernel>(blockHistograms.NonEmptyDevices(),
                                           streamId,
                                           policy,
                                           maxBins,
                                           blockFeatures,
                                           subsets.Partitions,
                                           *singleLeaf,
                                           DataSet.GetCompressedIndex().GetStorage(),
                                           subsets.Target.Indices,
                                           subsets.Target.StatsToAggregate,
                                           blockHistograms);

                } else {
                    CB_ENSURE(loadPolicy == ELoadFromCompressedIndexPolicy::GatherBins);
                    using TKernel = NKernelHost::TComputeSingleHistKernelGatherBins;

                    LaunchKernels<TKernel>(blockHistograms.NonEmptyDevices(),
                                           streamId,
                                           policy,
                                           maxBins,
                                           blockFeatures,
                                           subsets.Partitions,
                                           *singleLeaf,
                                           DataSet.GetCompressedIndex().GetStorage(),
                                           subsets.Target.Indices,
                                           subsets.Target.StatsToAggregate,
                                           tempGatheredCompressedIndex[blockId % activeStreamsCount],
                                           blockHistograms);
                }
            }

            auto reducedMapping = ComputeByBlocksHelper.ReducedBlockHistogramsMapping(blockId, leavesCount, statsCount);
            ReduceScatter(blockHistograms,
                          reducedMapping,
                          false,
                          streamId);

            if (leavesGpu) {
                using TKernel = NKernelHost::TWriteReducesHistogramsKernel;
                LaunchKernels<TKernel>(subsets.Histograms.NonEmptyDevices(),
                                       streamId,
                                       ComputeByBlocksHelper.GetWriteOffset(blockId),
                                       ComputeByBlocksHelper.GetWriteSizes(blockId),
                                       *leavesGpu,
                                       statsCount,
                                       blockHistograms,
                                       ComputeByBlocksHelper.BinFeatureCount(),
                                       subsets.Histograms);
            } else {
                using TKernel = NKernelHost::TWriteReducesHistogramKernel;
                LaunchKernels<TKernel>(subsets.Histograms.NonEmptyDevices(),
                                       streamId,
                                       ComputeByBlocksHelper.GetWriteOffset(blockId),
                                       ComputeByBlocksHelper.GetWriteSizes(blockId),
                                       *singleLeaf,
                                       statsCount,
                                       blockHistograms,
                                       ComputeByBlocksHelper.BinFeatureCount(),
                                       subsets.Histograms);
            }
        }

        if (!IsOnlyDefaultStream()) {
            NCudaLib::GetCudaManager().Barrier();
        }

        if (leavesGpu) {
            using TKernel = NKernelHost::TScanHistogramsKernel;
            LaunchKernels<TKernel>(subsets.Histograms.NonEmptyDevices(),
                                   0,
                                   ComputeByBlocksHelper.GetFeatures(),
                                   *leavesGpu,
                                   subsets.GetStatCount(),
                                   ComputeByBlocksHelper.BinFeatureCount(),
                                   subsets.Histograms);
        } else {
            CB_ENSURE(singleLeaf);
            using TKernel = NKernelHost::TScanHistogramKernel;
            LaunchKernels<TKernel>(subsets.Histograms.NonEmptyDevices(),
                                   0,
                                   ComputeByBlocksHelper.GetFeatures(),
                                   *singleLeaf,
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
        auto guard = profiler.Profile(TStringBuilder() << "Compute histograms for #" << computeLeaves.size() << " leaves");

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
        if (leaves.size() <= 2) {
            for (auto leaf : leaves) {
                using TKernel = NKernelHost::TZeroHistogramKernel;

                LaunchKernels<TKernel>(subsets->Histograms.NonEmptyDevices(),
                                       0,
                                       leaf,
                                       subsets->GetStatCount(),
                                       ComputeByBlocksHelper.BinFeatureCount(),
                                       subsets->Histograms);
            }
        } else {
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
    }

    void TSplitPropertiesHelper::SubstractHistograms(const TVector<ui32>& from,
                                                     const TVector<ui32>& what,
                                                     TPointsSubsets* subsets) {
        CB_ENSURE(
            from.size() == what.size(),
            "Sizes of subtracted histograms do not match, " << from.size() << " != " << what.size());

        if (from.size() <= 2) {
            using TKernel = NKernelHost::TSubstractHistogramKernel;
            for (ui32 i = 0; i < from.size(); ++i) {
                LaunchKernels<TKernel>(subsets->Histograms.NonEmptyDevices(),
                                       0,
                                       from[i],
                                       what[i],
                                       subsets->GetStatCount(),
                                       ComputeByBlocksHelper.BinFeatureCount(),
                                       subsets->Histograms);
            }
        } else {
            auto ids = TMirrorBuffer<ui32>::Create(NCudaLib::TMirrorMapping(2 * from.size()));
            {
                TVector<ui32> tmp;
                tmp.insert(tmp.end(), from.begin(), from.end());
                tmp.insert(tmp.end(), what.begin(), what.end());
                ids.Write(tmp);
            }
            auto fromIds = ids.SliceView(TSlice(0, from.size()));
            auto whatIds = ids.SliceView(TSlice(from.size(), 2 * from.size()));

            using TKernel = NKernelHost::TSubstractHistogramsKernel;

            LaunchKernels<TKernel>(subsets->Histograms.NonEmptyDevices(),
                                   0,
                                   fromIds,
                                   whatIds,
                                   subsets->GetStatCount(),
                                   ComputeByBlocksHelper.BinFeatureCount(),
                                   subsets->Histograms);
        }
    }

    bool TSplitCandidate::operator<(const TSplitCandidate& rhs) const {
        return Score < rhs.Score;
    }

    bool TSplitCandidate::operator>(const TSplitCandidate& rhs) const {
        return rhs < *this;
    }
    bool TSplitCandidate::operator<=(const TSplitCandidate& rhs) const {
        return !(rhs < *this);
    }
    bool TSplitCandidate::operator>=(const TSplitCandidate& rhs) const {
        return !(*this < rhs);
    }
}

namespace NCudaLib {
    REGISTER_KERNEL(0xD2DAA0, NKernelHost::TWriteInitPartitions);
    REGISTER_KERNEL(0xD2DAA1, NKernelHost::TCopyHistogramsKernel);
    REGISTER_KERNEL(0xD2DAA2, NKernelHost::TWriteReducesHistogramsKernel);
    REGISTER_KERNEL(0xD2DAA3, NKernelHost::TZeroHistogramsKernel);
    REGISTER_KERNEL(0xD2DAA4, NKernelHost::TScanHistogramsKernel);
    REGISTER_KERNEL(0xD2DAA5, NKernelHost::TComputeHistKernelLoadByIndex);
    REGISTER_KERNEL(0xD2DAA6, NKernelHost::TComputeHistKernelGatherBins);
    REGISTER_KERNEL(0xD2DAA7, NKernelHost::TSubstractHistogramsKernel);
    REGISTER_KERNEL(0xD2DAA8, NKernelHost::TZeroHistogramKernel);
    REGISTER_KERNEL(0xD2DAA9, NKernelHost::TSubstractHistogramKernel);
    REGISTER_KERNEL(0xD2DAB0, NKernelHost::TScanHistogramKernel);
    REGISTER_KERNEL(0xD2DAB1, NKernelHost::TComputeSingleHistKernelLoadByIndex);
    REGISTER_KERNEL(0xD2DAB2, NKernelHost::TComputeSingleHistKernelGatherBins);
    REGISTER_KERNEL(0xD2DAB3, NKernelHost::TWriteReducesHistogramKernel);
    REGISTER_KERNEL_TEMPLATE(0xD2DAB4, NKernelHost::TZeroBuffer, float);
    REGISTER_KERNEL_TEMPLATE(0xD2DAB5, NKernelHost::TZeroBuffer, TDataPartition);

}
