#include "split_points.h"
#include <catboost/cuda/methods/greedy_subsets_searcher/kernel/histogram_utils.cuh>
#include <catboost/cuda/cuda_lib/cuda_base.h>

namespace NKernelHost {
    THolder<TSplitPointsKernel::TKernelContext> TSplitPointsKernel::PrepareContext(IMemoryManager& manager) const {
        THolder<TSplitPointsKernel::TKernelContext> context = new TSplitPointsKernel::TKernelContext;
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

        context->UpdatePropsTempBufferSize = NKernel::GetTempVarsCount(Statistics.GetColumnCount(), LeafIdsToSplitCpu.Size());
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
        SortByFlagsInLeaves(LeafIdsToSplitCpu.Get(), LeafIdsToSplitCpu.Size(), PartitionsCpu.Get(), context, stream.GetStream());
        const ui64 lineSize = Statistics.AlignedColumnSize();

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
                                            stream.GetStream());

        NCudaLib::CopyMemoryAsync(PartitionsGpu.Get(), PartitionsCpu.Get(), PartitionsGpu.Size(), stream);

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
}

namespace NCudaLib {
    REGISTER_KERNEL(0xAD2AA0, NKernelHost::TSplitPointsKernel);
}
