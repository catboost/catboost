#pragma once

#include "configs.h"
#include "structure_searcher_options.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/gpu_data/gpu_structures.h>
#include <catboost/cuda/gpu_data/gpu_structures.h>
#include <catboost/cuda/gpu_data/grid_policy.h>
#include <catboost/cuda/gpu_data/compressed_index.h>
#include <catboost/cuda/gpu_data/doc_parallel_dataset.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/utils/helpers.h>

namespace NCatboostCuda {
    //will be cached
    class TComputeSplitPropertiesByBlocksHelper {
    public:
        using TGpuDataSet = typename TSharedCompressedIndex<TDocParallelLayout>::TCompressedDataSet;

        TComputeSplitPropertiesByBlocksHelper(const TDocParallelDataSet& dataSet,
                                              const TComputeByBlocksConfig& splitPropsConfig)
            : DataSet(dataSet)
            , CompressedIndexStorage(dataSet.GetCompressedIndex().GetStorage())
            , StreamsCount(splitPropsConfig.StreamCount)
        {
            Rebuild(splitPropsConfig);
        }

        void ResetHistograms(ui32 numStats, ui32 maxHistCount,
                             TStripeBuffer<float>* histograms) {
            auto histogramsMapping = BinFeatures.GetMapping().Transform([&](const TSlice& deviceBinFeatures) -> ui64 {
                return deviceBinFeatures.Size() * maxHistCount * numStats;
            });
            histograms->Reset(histogramsMapping);
        }

        ui32 GetBlockCount() const {
            return BlockSlices.size();
        }

        TStripeBuffer<const TFeatureInBlock> GetBlockFeatures(ui32 blockId) const {
            CB_ENSURE(blockId < BlockSlices.size(), "Block id is too large");
            TSlice blockSlice = BlockSlices[blockId];
            return NCudaLib::ParallelStripeView(Features, blockSlice).AsConstBuf();
        }

        ui32 GetIntsPerSample(ui32 blockId) const {
            CB_ENSURE(blockId < BlockSlices.size(), "Block id is too large");
            TSlice blockSlice = BlockSlices[blockId];
            EFeaturesGroupingPolicy policy = GetBlockPolicy(blockId);
            ui32 featuresPerInt = GetFeaturesPerInt(policy);
            return CeilDivide(blockSlice.Size(), featuresPerInt);
        }

        const NCudaLib::TDistributedObject<ui32>& GetWriteOffset(ui32 blockId) const {
            return WriteOffsets[blockId];
        }

        const NCudaLib::TDistributedObject<ui32>& GetWriteSizes(ui32 blockId) const {
            return WriteSizes[blockId];
        }

        EFeaturesGroupingPolicy GetBlockPolicy(ui32 blockId) const {
            CB_ENSURE(blockId < BlockPolicies.size(), "Block id is too large");
            return BlockPolicies[blockId];
        }

        int GetBlockHistogramMaxBins(ui32 blockId) const {
            return MaxFolds[blockId];
        }

        NCudaLib::TStripeMapping BlockHistogramsMapping(ui32 blockId, ui32 histCount, ui32 statCount) const {
            return BeforeReduceBinFeaturesMappings[blockId].Transform([&](const TSlice blockSize) -> ui64 {
                return blockSize.Size() * histCount * statCount;
            });
        }

        //should be transformed to obtain real memory (e.g. multiply by systemSize to get correct siz)
        NCudaLib::TStripeMapping ReducedBlockHistogramsMapping(ui32 blockId, ui32 histCount, ui32 statCount) const {
            return AfterReduceBinFeaturesMappings[blockId].Transform([&](const TSlice blockSize) -> ui64 {
                return blockSize.Size() * histCount * statCount;
            });
        }

        /* we need to make just one scan and histograms remove kernels,
         * because number of kernel launches should not depend on
         * dataset
         *
         * This histograms/binFeatures after reduceScatter
         * */
        const TStripeBuffer<TCBinFeature>& GetBinFeatures() const {
            return BinFeatures;
        }

        NCudaLib::TDistributedObject<ui32> BinFeatureCount() const;

        const TStripeBuffer<TBinarizedFeature>& GetFeatures() const {
            return BinarizedFeatures;
        }

        ui32 GetStreamCount() const {
            return StreamsCount;
        }

    private:
        void Rebuild(const TComputeByBlocksConfig& splitPropsConfig);

    private:
        const TDocParallelDataSet& DataSet;
        const TStripeBuffer<ui32>& CompressedIndexStorage;

        /* this block for histograms */
        //compute
        //ComputeSplitProps(blockId)
        TStripeBuffer<TFeatureInBlock> Features;
        TVector<TSlice> BlockSlices;
        TVector<EFeaturesGroupingPolicy> BlockPolicies;
        TVector<ui32> MaxFolds;
        //ReduceScatter(blockId)
        TVector<NCudaLib::TStripeMapping> BeforeReduceBinFeaturesMappings;
        //reduce-scatter
        TVector<NCudaLib::TStripeMapping> AfterReduceBinFeaturesMappings;

        //Just copy to histograms
        TVector<NCudaLib::TDistributedObject<ui32>> WriteOffsets;
        TVector<NCudaLib::TDistributedObject<ui32>> WriteSizes;

        //scan + substract
        TStripeBuffer<TBinarizedFeature> BinarizedFeatures;
        //Compute best splits
        TStripeBuffer<TCBinFeature> BinFeatures;

        //TODO(noxoomo): tune it
        ui32 StreamsCount = 3;
    };

    TComputeSplitPropertiesByBlocksHelper& GetComputeByBlocksHelper(const TDocParallelDataSet& dataSet,
                                                                    const TTreeStructureSearcherOptions& options, ui32 statCount);

}
