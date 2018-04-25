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
                                 const TCpuGrid& grid)
            : Policy(policy)
            , Depth(depth)
            , Grid(grid)
        {
            Rebuild();
        }

        //-------stripe, before reduce-------
        TStripeBuffer<const TCFeature> GetFeatures(const TStripeBuffer<TCFeature>& features, ui32 block) const {
            return NCudaLib::ParallelStripeView(features, FeatureSlices[block]);
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
        void Rebuild() {
            FeatureSlices.clear();
            BinFeatureSlices.clear();

            const ui32 MB = 1024 * 1024;
            //TODO(noxoomo): tune it +  specializations for 1Gbs, 10Gbs networks and infiniband
            const ui32 reduceBlockSize = NCudaLib::GetCudaManager().HasRemoteDevices() ? 4 * MB : 32 * MB;

            //TODO(noxoomo): there can be done more sophisticated balancing based on fold counts for feature
            //do it, if reduce'll be bottleneck in distributed setting
            const ui32 oneIntFeatureGroup = GetFeaturesPerInt(Policy);

            //depth is current depth, linear system is systems obtained after split
            const ui32 leavesCount = 1 << (Depth + 1);
            const ui32 singleLinearSystem = leavesCount + leavesCount * (leavesCount + 1) / 2;
            const ui32 meanFoldsPerFeature = MeanFoldCount();

            const ui32 groupSizeBytes = meanFoldsPerFeature * oneIntFeatureGroup * singleLinearSystem * sizeof(float);

            ui32 featuresPerGroup = Grid.FeatureIds.size();
            if (NCudaLib::GetCudaManager().GetDeviceCount() != 1) {
                featuresPerGroup = Max<ui32>(reduceBlockSize / groupSizeBytes, 1) * oneIntFeatureGroup;
            }

            ui32 binFeatureOffset = 0;
            NCudaLib::TDistributedObject<ui32> solutionOffsets = CreateDistributedObject<ui32>(0);

            for (ui32 firstFeatureInGroup = 0; firstFeatureInGroup < Grid.FeatureIds.size(); firstFeatureInGroup += featuresPerGroup) {
                const ui32 end = Min<ui32>(firstFeatureInGroup + featuresPerGroup, Grid.FeatureIds.size());
                FeatureSlices.push_back(TSlice(firstFeatureInGroup, end));
                ui32 binFeatureInSlice = 0;
                for (ui32 f = firstFeatureInGroup; f < end; ++f) {
                    binFeatureInSlice += Grid.Folds[f];
                }

                BinFeatureSlices.push_back(TSlice(binFeatureOffset,
                                                  binFeatureOffset + binFeatureInSlice));

                BeforeReduceMappings.push_back(NCudaLib::TStripeMapping::RepeatOnAllDevices(binFeatureInSlice));
                AfterReduceMappings.push_back(NCudaLib::TStripeMapping::SplitBetweenDevices(binFeatureInSlice));

                //how to get binFeatureIdx
                auto afterReduceLoadBinFeaturesSlice = CreateDistributedObject<TSlice>(TSlice(0, 0));
                auto blockSolutions = CreateDistributedObject<TSlice>(TSlice(0, 0));

                for (auto dev : AfterReduceMappings.back().NonEmptyDevices()) {
                    auto afterReduceDeviceBinFeatures = AfterReduceMappings.back().DeviceSlice(dev);
                    {
                        TSlice loadSlice;
                        loadSlice.Left = binFeatureOffset + afterReduceDeviceBinFeatures.Left;
                        loadSlice.Right = binFeatureOffset + afterReduceDeviceBinFeatures.Right;
                        afterReduceLoadBinFeaturesSlice.Set(dev, loadSlice);
                    }

                    {
                        TSlice blockSolutionsOnDevice;

                        blockSolutionsOnDevice.Left = solutionOffsets.At(dev);
                        blockSolutionsOnDevice.Right = solutionOffsets.At(dev) + afterReduceDeviceBinFeatures.Size();
                        blockSolutions.Set(dev, blockSolutionsOnDevice);
                        solutionOffsets.Set(dev, solutionOffsets.At(dev) + afterReduceDeviceBinFeatures.Size());
                    }
                }
                BlockedBinFeatures.push_back(afterReduceLoadBinFeaturesSlice);
                FlatResultsSlice.push_back(blockSolutions);

                binFeatureOffset += binFeatureInSlice;
            }

            NCudaLib::TMappingBuilder<NCudaLib::TStripeMapping> solutionsMappingBuilder;
            for (ui32 dev = 0; dev < NCudaLib::GetCudaManager().GetDeviceCount(); ++dev) {
                Y_VERIFY(solutionOffsets.At(dev) == FlatResultsSlice.back().At(dev).Right);
                solutionsMappingBuilder.SetSizeAt(dev, solutionOffsets.At(dev));
            }

            FlatResultsMapping = solutionsMappingBuilder.Build();
        }

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

        TVector<TSlice> FeatureSlices;
        TVector<TSlice> BinFeatureSlices;

        TVector<NCudaLib::TStripeMapping> BeforeReduceMappings;
        TVector<NCudaLib::TStripeMapping> AfterReduceMappings;

        NCudaLib::TStripeMapping FlatResultsMapping;
        TVector<NCudaLib::TDistributedObject<TSlice>> FlatResultsSlice;

        TVector<NCudaLib::TDistributedObject<TSlice>> BlockedBinFeatures;
    };

}
