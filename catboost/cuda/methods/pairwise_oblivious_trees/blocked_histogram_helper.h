#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/gpu_data/gpu_structures.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/gpu_data/gpu_structures.h>
#include <catboost/cuda/gpu_data/grid_policy.h>
#include <catboost/cuda/gpu_data/feature_layout_doc_parallel.h>
#include <catboost/cuda/gpu_data/compressed_index.h>

namespace NCatboostCuda {
    class TBlockedHistogramsHelper {
    public:
        using TGpuDataSet = typename TSharedCompressedIndex<TDocParallelLayout>::TCompressedDataSet;

        TBlockedHistogramsHelper(EFeaturesGroupingPolicy policy,
                                 ui32 depth,
                                 const TCpuGrid& grid,
                                 int maxStreamCount)
            : Policy(policy)
            , Depth(depth)
            , Grid(grid)
            , MaxStreamCount(maxStreamCount)
        {
            Rebuild();
        }

        //-------stripe, before reduce-------
        template <class T, NCudaLib::EPtrType Type>
        NCudaLib::TCudaBuffer<const TCFeature, NCudaLib::TStripeMapping, Type> GetFeatures(const NCudaLib::TCudaBuffer<T, NCudaLib::TStripeMapping, Type>& features, ui32 block) const {
            return NCudaLib::ParallelStripeView(features, FeatureSlices[block]).AsConstBuf();
        }

        TFoldsHistogram ComputeFoldsHistogram(ui32 block) const {
            return Grid.ComputeFoldsHistogram(FeatureSlices[block]);
        }

        TSlice GetBinFeatureSlice(ui32 blockId) const {
            return BinFeatureSlices[blockId];
        }

        ui32 GetBinFeatureCount(ui32 blockId) const {
            return BinFeatureSlices[blockId].Size();
        }

        ui32 GetBlockCount() const {
            return static_cast<ui32>(FeatureSlices.size());
        }

        //---- After reduce ---
        //binFeatures load for block (indices in mirror array for reduce matrices)
        NCudaLib::TDistributedObject<TSlice> GetBinFeatureAccessorSlice(ui32 blockId) const {
            return BlockedBinFeatures[blockId];
        }

        //should be transformed to obtain real memory (e.g. multiply by systemSize to get correct siz)
        NCudaLib::TStripeMapping ReduceMapping(ui32 blockId) {
            return AfterReduceMappings[blockId];
        }

        NCudaLib::TStripeMapping GetFlatResultsMapping() {
            return FlatResultsMapping;
        }

        //where to write solutions. Should be transformed in a proper way (e.g. rowSize * idx to write solution)
        NCudaLib::TDistributedObject<TSlice> GetFlatResultsSlice(ui32 blockId) {
            return FlatResultsSlice[blockId];
        }

    private:
        void Rebuild();

        ui32 MeanFoldCount() const {
            ui32 total = 0;
            for (const auto& foldSize : Grid.Folds) {
                total += foldSize;
            }
            return total / Grid.Folds.size();
        }

        ui32 TotalFoldCount() const {
            ui32 total = 0;
            for (const auto& foldSize : Grid.Folds) {
                total += foldSize;
            }
            return total;
        }

    private:
        EFeaturesGroupingPolicy Policy;
        const ui32 Depth;
        const TCpuGrid& Grid;
        int MaxStreamCount;

        TVector<TSlice> FeatureSlices;
        TVector<TSlice> BinFeatureSlices;

        TVector<NCudaLib::TStripeMapping> BeforeReduceMappings;
        TVector<NCudaLib::TStripeMapping> AfterReduceMappings;

        NCudaLib::TStripeMapping FlatResultsMapping;
        TVector<NCudaLib::TDistributedObject<TSlice>> FlatResultsSlice;

        TVector<NCudaLib::TDistributedObject<TSlice>> BlockedBinFeatures;
    };

}
