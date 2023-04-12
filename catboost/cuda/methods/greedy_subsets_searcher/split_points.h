#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/cuda/cuda_util/kernel/transform.cuh>
#include <catboost/cuda/cuda_util/gpu_data/partitions.h>
#include <catboost/cuda/gpu_data/gpu_structures.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/cuda/methods/greedy_subsets_searcher/kernel/split_points.cuh>

namespace NKernelHost {
    class TSplitPointsKernel: public TKernelBase<NKernel::TSplitPointsContext> {
    private:
        ui32 StatsPerKernel;
        TCudaBufferPtr<const ui32> CompressedIndex;

        TCudaBufferPtr<const TCFeature> FeaturesToSplit;
        TCudaBufferPtr<const ui32> FeatureBins;
        /* they'll be modified */
        TCudaBufferPtr<const ui32> LeafIdsToSplitGpu;
        TCudaBufferPtr<const ui32> RightLeafIdsAfterSplit;
        TCudaHostBufferPtr<const ui32> LeafIdsToSplitCpu;

        TCudaBufferPtr<TDataPartition> PartitionsGpu;
        TCudaHostBufferPtr<TDataPartition> PartitionsCpu;
        TCudaBufferPtr<double> PartitionStats;

        TCudaBufferPtr<ui32> Indices;
        TCudaBufferPtr<float> Statistics;

        ui32 BinFeatureCount;
        TCudaBufferPtr<float> Histograms;

    public:
        TSplitPointsKernel(ui32 statsPerKernel,
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
                           TCudaBufferPtr<float> histograms);

        using TKernelContext = NKernel::TSplitPointsContext;

        Y_SAVELOAD_DEFINE(StatsPerKernel,
                          CompressedIndex,
                          FeaturesToSplit,
                          FeatureBins,
                          LeafIdsToSplitGpu,
                          RightLeafIdsAfterSplit,
                          LeafIdsToSplitCpu,
                          PartitionsGpu,
                          PartitionsCpu,
                          PartitionStats,
                          Indices,
                          Statistics,
                          BinFeatureCount,
                          Histograms);

        TSplitPointsKernel() = default;

        THolder<TKernelContext> PrepareContext(IMemoryManager& manager) const;

        void Run(const TCudaStream& stream, TKernelContext& context) const;
    };

    class TSplitPointsSingleLeafKernel: public TKernelBase<NKernel::TSplitPointsContext> {
    private:
        ui32 StatsPerKernel;
        TCudaBufferPtr<const ui32> CompressedIndex;

        TCFeature Feature;
        ui32 FeatureBin;
        ui32 LeafIdToSplit;
        ui32 RightLeafIdAfterSplit;

        TCudaBufferPtr<TDataPartition> PartitionsGpu;
        TCudaHostBufferPtr<TDataPartition> PartitionsCpu;
        TCudaBufferPtr<double> PartitionStats;

        TCudaBufferPtr<ui32> Indices;
        TCudaBufferPtr<float> Statistics;

        ui32 BinFeatureCount;
        TCudaBufferPtr<float> Histograms;

    public:
        TSplitPointsSingleLeafKernel(ui32 statsPerKernel,
                                     TCudaBufferPtr<const unsigned int> compressedIndex,
                                     TCFeature featuresToSplit,
                                     ui32 featureBin,
                                     ui32 leafIdToSplit,
                                     ui32 rightLeafIdAfterSplit,
                                     TCudaBufferPtr<TDataPartition> partitionsGpu,
                                     TCudaHostBufferPtr<TDataPartition> partitionsCpu,
                                     TCudaBufferPtr<double> partitionStats,
                                     TCudaBufferPtr<unsigned int> indices,
                                     TCudaBufferPtr<float> stats,
                                     ui32 binFeatureCount,
                                     TCudaBufferPtr<float> histograms);

        using TKernelContext = NKernel::TSplitPointsContext;

        Y_SAVELOAD_DEFINE(StatsPerKernel,
                          CompressedIndex,
                          Feature,
                          FeatureBin,
                          LeafIdToSplit,
                          RightLeafIdAfterSplit,
                          PartitionsGpu,
                          PartitionsCpu,
                          PartitionStats,
                          Indices,
                          Statistics,
                          BinFeatureCount,
                          Histograms);

        TSplitPointsSingleLeafKernel() = default;

        THolder<TKernelContext> PrepareContext(IMemoryManager& manager) const;

        void Run(const TCudaStream& stream, TKernelContext& context) const;
    };

}
