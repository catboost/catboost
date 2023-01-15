#include "split_points.h"
#include <catboost/cuda/methods/greedy_subsets_searcher/kernel/histogram_utils.cuh>
#include <catboost/cuda/cuda_lib/cuda_base.h>

namespace NKernelHost {
    THolder<TSplitPointsKernel::TKernelContext> TSplitPointsKernel::PrepareContext(IMemoryManager& manager) const {
        THolder<TSplitPointsKernel::TKernelContext> context = MakeHolder<TSplitPointsKernel::TKernelContext>();
        context->TempStorageSizes.resize(LeafIdsToSplitCpu.Size());

        ui64 tempStorageMemory = Min<ui64>(StatsPerKernel, Statistics.GetColumnCount()) * Statistics.AlignedColumnSize() * sizeof(float);

        NKernel::TCudaStream zeroStream = 0;
        SortByFlagsInLeaves(LeafIdsToSplitCpu.Get(), LeafIdsToSplitCpu.Size(),
                            PartitionsCpu.Get(),
                            *context,
                            zeroStream);

        for (auto memoryForRadixSort : context->TempStorageSizes) {
            tempStorageMemory = Max<ui64>(memoryForRadixSort, tempStorageMemory);
        }

        ui32 docCount = Indices.Size();

        context->TempFlags = manager.Allocate<bool>(docCount);
        context->Flags = manager.Allocate<bool>(docCount);

        context->TempIndices = manager.Allocate<ui32>(docCount);
        context->Indices = manager.Allocate<ui32>(docCount);

        context->TempStorage = manager.Allocate<char>(tempStorageMemory);

        context->UpdatePropsTempBufferSize = NKernel::GetTempVarsCount(Statistics.GetColumnCount(), 2 * LeafIdsToSplitCpu.Size());
        context->UpdatePropsTempBuffer = manager.Allocate<double>(context->UpdatePropsTempBufferSize);
        return context;
    }

    void TSplitPointsKernel::Run(const TCudaStream& stream, TKernelContext& context) const {
        const ui32 numLeaves = LeafIdsToSplitGpu.Size();
        const ui32 numStats = Statistics.GetColumnCount();

        NKernel::SplitAndMakeSequenceInLeaves(CompressedIndex.Get(),
                                              Indices.Get(),
                                              PartitionsGpu.Get(),
                                              LeafIdsToSplitGpu.Get(),
                                              LeafIdsToSplitGpu.Size(),
                                              FeaturesToSplit.Get(),
                                              FeatureBins.Get(),
                                              context.TempFlags,
                                              context.TempIndices,
                                              stream);

        CB_ENSURE(LeafIdsToSplitCpu.Size() == LeafIdsToSplitGpu.Size());
        //TODO(noxoomo): for oblivious trees we have overhead  for launching kernel per leaf
        //this is a constant overhead that matters on small datasets
        const auto cpuLeafIdsPtr = LeafIdsToSplitCpu.Get();
        const auto partitionsCpuPtr = PartitionsCpu.Get();
        SortByFlagsInLeaves(cpuLeafIdsPtr, LeafIdsToSplitCpu.Size(), partitionsCpuPtr, context, stream.GetStream());
        const ui64 lineSize = Statistics.AlignedColumnSize();

        ui32 maxLeafSize = 0;
        for (ui32 leaf = 0; leaf < numLeaves; ++leaf) {
            maxLeafSize = Max(partitionsCpuPtr[cpuLeafIdsPtr[leaf]].Size, maxLeafSize);
        }

        if (maxLeafSize > 1024) {
            for (ui32 firstStat = 0; firstStat < numStats; firstStat += StatsPerKernel) {
                const ui32 lastStat = Min<ui32>(firstStat + StatsPerKernel, numStats);

                float* tempData = (float*)(context.TempStorage.Get());

                NKernel::CopyInLeaves<float>(LeafIdsToSplitGpu.Get(),
                                             numLeaves,
                                             PartitionsGpu.Get(),
                                             Statistics.GetColumn(firstStat),
                                             tempData,
                                             lastStat - firstStat,
                                             lineSize,
                                             stream);

                NKernel::GatherInLeaves<float>(LeafIdsToSplitGpu.Get(),
                                               numLeaves,
                                               PartitionsGpu.Get(),
                                               tempData,
                                               context.Indices,
                                               Statistics.GetColumn(firstStat),
                                               lastStat - firstStat,
                                               lineSize,
                                               stream);
            }

            //now copy indices
            {
                ui32* tempData = (ui32*)(context.TempStorage.Get());
                NKernel::CopyInLeaves<ui32>(LeafIdsToSplitGpu.Get(),
                                            numLeaves,
                                            PartitionsGpu.Get(),
                                            Indices.Get(),
                                            tempData,
                                            1,
                                            Indices.Size(),
                                            stream);

                NKernel::GatherInLeaves(LeafIdsToSplitGpu.Get(),
                                        numLeaves,
                                        PartitionsGpu.Get(),
                                        tempData,
                                        context.Indices,
                                        Indices.Get(),
                                        1,
                                        Indices.Size(),
                                        stream);
            }
        } else {
//fast path for small datasets
// gather everything in shared memory, one kernel launch only
#define FAST_PATH(SIZE)                                           \
    NKernel::GatherInplaceLeqSize<SIZE>(LeafIdsToSplitGpu.Get(),  \
                                        LeafIdsToSplitGpu.Size(), \
                                        PartitionsGpu.Get(),      \
                                        context.Indices,          \
                                        Statistics.GetColumn(0),  \
                                        numStats,                 \
                                        lineSize,                 \
                                        Indices.Get(),            \
                                        stream);

            if (maxLeafSize > 6144) {
                FAST_PATH(12288)
            } else if (maxLeafSize > 3072) {
                FAST_PATH(6144)
            } else if (maxLeafSize > 1024) {
                FAST_PATH(3072)
            } else {
                FAST_PATH(1024)
            }
#undef FAST_PATH
        }

        NKernel::CopyHistograms(LeafIdsToSplitGpu.Get(),
                                RightLeafIdsAfterSplit.Get(),
                                numLeaves,
                                Statistics.GetColumnCount(),
                                BinFeatureCount,
                                Histograms.Get(),
                                stream);

        NKernel::UpdatePartitionsAfterSplit(LeafIdsToSplitGpu.Get(),
                                            RightLeafIdsAfterSplit.Get(),
                                            numLeaves,
                                            context.Flags.Get(),
                                            PartitionsGpu.Get(),
                                            partitionsCpuPtr,
                                            stream.GetStream());

        CB_ENSURE(Statistics.GetColumnCount() == PartitionStats.ObjectSize());
        NKernel::UpdatePartitionsPropsForSplit(PartitionsGpu.Get(),
                                               LeafIdsToSplitGpu.Get(),
                                               RightLeafIdsAfterSplit.Get(),
                                               LeafIdsToSplitGpu.Size(),
                                               Statistics.Get(),
                                               Statistics.GetColumnCount(),
                                               Statistics.AlignedColumnSize(),
                                               context.UpdatePropsTempBufferSize,
                                               context.UpdatePropsTempBuffer.Get(),
                                               PartitionStats.Get(),
                                               stream.GetStream());
    }

    TSplitPointsKernel::TSplitPointsKernel(ui32 statsPerKernel,
                                           TCudaBufferPtr<const unsigned int> compressedIndex,
                                           TCudaBufferPtr<const TCFeature> featuresToSplit,
                                           TCudaBufferPtr<const unsigned int> featureBins,
                                           TCudaBufferPtr<const unsigned int> leafIdsToSplitGpu,
                                           TCudaBufferPtr<const unsigned int> rightLeafIdsAfterSplit,
                                           TCudaHostBufferPtr<const ui32> leafIdsToSplitCpu,
                                           TCudaBufferPtr<TDataPartition> partitionsGpu,
                                           TCudaHostBufferPtr<TDataPartition> partitionsCpu,
                                           TCudaBufferPtr<double> partitionStats,
                                           TCudaBufferPtr<unsigned int> indices,
                                           TCudaBufferPtr<float> stats,
                                           ui32 binFeatureCount,
                                           TCudaBufferPtr<float> histograms)
        : StatsPerKernel(statsPerKernel)
        , CompressedIndex(compressedIndex)
        , FeaturesToSplit(featuresToSplit)
        , FeatureBins(featureBins)
        , LeafIdsToSplitGpu(leafIdsToSplitGpu)
        , RightLeafIdsAfterSplit(rightLeafIdsAfterSplit)
        , LeafIdsToSplitCpu(leafIdsToSplitCpu)
        , PartitionsGpu(partitionsGpu)
        , PartitionsCpu(partitionsCpu)
        , PartitionStats(partitionStats)
        , Indices(indices)
        , Statistics(stats)
        , BinFeatureCount(binFeatureCount)
        , Histograms(histograms)
    {
    }

    THolder<TSplitPointsSingleLeafKernel::TKernelContext> TSplitPointsSingleLeafKernel::PrepareContext(IMemoryManager& manager) const {
        THolder<TSplitPointsSingleLeafKernel::TKernelContext> context = MakeHolder<TSplitPointsSingleLeafKernel::TKernelContext>();
        context->TempStorageSizes.resize(1);

        ui64 tempStorageMemory = Min<ui64>(StatsPerKernel, Statistics.GetColumnCount()) * Statistics.AlignedColumnSize() * sizeof(float);

        NKernel::TCudaStream zeroStream = 0;

        SortByFlagsInLeaf(LeafIdToSplit,
                          PartitionsCpu.Get(),
                          *context,
                          zeroStream);

        for (auto memoryForRadixSort : context->TempStorageSizes) {
            tempStorageMemory = Max<ui64>(memoryForRadixSort, tempStorageMemory);
        }

        ui32 leafSize = PartitionsCpu.Get()[LeafIdToSplit].Size;

        context->TempFlags = manager.Allocate<bool>(leafSize);
        context->Flags = manager.Allocate<bool>(leafSize);

        context->TempIndices = manager.Allocate<ui32>(leafSize);
        context->Indices = manager.Allocate<ui32>(leafSize);

        context->TempStorage = manager.Allocate<char>(tempStorageMemory);

        context->UpdatePropsTempBufferSize = NKernel::GetTempVarsCount(Statistics.GetColumnCount(), 2);
        context->UpdatePropsTempBuffer = manager.Allocate<double>(context->UpdatePropsTempBufferSize);
        return context;
    }

    void TSplitPointsSingleLeafKernel::Run(const TCudaStream& stream, TKernelContext& context) const {
        const ui32 numStats = Statistics.GetColumnCount();
        const auto partitionsCpuPtr = PartitionsCpu.Get();

        const auto leafSize = partitionsCpuPtr[LeafIdToSplit].Size;

        NKernel::SplitAndMakeSequenceInLeaf(CompressedIndex.Get(),
                                            Indices.Get(),
                                            PartitionsGpu.Get(),
                                            LeafIdToSplit,
                                            leafSize,
                                            Feature,
                                            FeatureBin,
                                            context.TempFlags,
                                            context.TempIndices,
                                            stream);

        SortByFlagsInLeaf(LeafIdToSplit,
                          partitionsCpuPtr,
                          context,
                          stream.GetStream());

        const ui64 lineSize = Statistics.AlignedColumnSize();

        if (leafSize > 6144) {
            for (ui32 firstStat = 0; firstStat < numStats; firstStat += StatsPerKernel) {
                const ui32 lastStat = Min<ui32>(firstStat + StatsPerKernel, numStats);

                float* tempData = (float*)(context.TempStorage.Get());

                NKernel::CopyLeaf<float>(LeafIdToSplit,
                                         leafSize,
                                         PartitionsGpu.Get(),
                                         Statistics.GetColumn(firstStat),
                                         tempData,
                                         lastStat - firstStat,
                                         lineSize,
                                         stream);

                NKernel::GatherLeaf<float>(LeafIdToSplit,
                                           leafSize,
                                           PartitionsGpu.Get(),
                                           tempData,
                                           context.Indices,
                                           Statistics.GetColumn(firstStat),
                                           lastStat - firstStat,
                                           lineSize,
                                           stream);
            }

            //now copy indices
            {
                ui32* tempData = (ui32*)(context.TempStorage.Get());
                NKernel::CopyLeaf<ui32>(LeafIdToSplit,
                                        leafSize,
                                        PartitionsGpu.Get(),
                                        Indices.Get(),
                                        tempData,
                                        1,
                                        Indices.Size(),
                                        stream);

                NKernel::GatherLeaf<ui32>(LeafIdToSplit,
                                          leafSize,
                                          PartitionsGpu.Get(),
                                          tempData,
                                          context.Indices,
                                          Indices.Get(),
                                          1,
                                          Indices.Size(),
                                          stream);
            }
        } else {
#define FAST_PATH(SIZE)                                             \
    NKernel::GatherInplaceSingleLeaf<SIZE>(LeafIdToSplit,           \
                                           PartitionsGpu.Get(),     \
                                           context.Indices,         \
                                           Statistics.GetColumn(0), \
                                           numStats,                \
                                           lineSize,                \
                                           Indices.Get(),           \
                                           stream);

            if (leafSize > 6144) {
                FAST_PATH(12288)
            } else if (leafSize > 3072) {
                FAST_PATH(6144)
            } else if (leafSize > 1024) {
                FAST_PATH(3072)
            } else {
                FAST_PATH(1024)
            }
#undef FAST_PATH
        }

        NKernel::CopyHistogram(LeafIdToSplit,
                               RightLeafIdAfterSplit,
                               Statistics.GetColumnCount(),
                               BinFeatureCount,
                               Histograms.Get(),
                               stream);

        NKernel::UpdatePartitionAfterSplit(LeafIdToSplit,
                                           RightLeafIdAfterSplit,
                                           leafSize,
                                           context.Flags.Get(),
                                           PartitionsGpu.Get(),
                                           partitionsCpuPtr,
                                           stream.GetStream());

        CB_ENSURE(Statistics.GetColumnCount() == PartitionStats.ObjectSize());

        NKernel::UpdatePartitionsPropsForSingleSplit(PartitionsGpu.Get(),
                                                     LeafIdToSplit,
                                                     RightLeafIdAfterSplit,
                                                     Statistics.Get(),
                                                     Statistics.GetColumnCount(),
                                                     Statistics.AlignedColumnSize(),
                                                     context.UpdatePropsTempBufferSize,
                                                     context.UpdatePropsTempBuffer.Get(),
                                                     PartitionStats.Get(),
                                                     stream.GetStream());
    }

    TSplitPointsSingleLeafKernel::TSplitPointsSingleLeafKernel(ui32 statsPerKernel,
                                                               TCudaBufferPtr<const unsigned int> compressedIndex,
                                                               TCFeature featuresToSplit,
                                                               ui32 featureBin,
                                                               ui32 leftLeaf,
                                                               ui32 rightLeaf,
                                                               TCudaBufferPtr<TDataPartition> partitionsGpu,
                                                               TCudaHostBufferPtr<TDataPartition> partitionsCpu,
                                                               TCudaBufferPtr<double> partitionStats,
                                                               TCudaBufferPtr<unsigned int> indices,
                                                               TCudaBufferPtr<float> stats,
                                                               ui32 binFeatureCount,
                                                               TCudaBufferPtr<float> histograms)
        : StatsPerKernel(statsPerKernel)
        , CompressedIndex(compressedIndex)
        , Feature(featuresToSplit)
        , FeatureBin(featureBin)
        , LeafIdToSplit(leftLeaf)
        , RightLeafIdAfterSplit(rightLeaf)
        , PartitionsGpu(partitionsGpu)
        , PartitionsCpu(partitionsCpu)
        , PartitionStats(partitionStats)
        , Indices(indices)
        , Statistics(stats)
        , BinFeatureCount(binFeatureCount)
        , Histograms(histograms)
    {
    }
}

namespace NCudaLib {
    REGISTER_KERNEL(0xAD2AA0, NKernelHost::TSplitPointsKernel);
    REGISTER_KERNEL(0xAD2AA1, NKernelHost::TSplitPointsSingleLeafKernel);
}
