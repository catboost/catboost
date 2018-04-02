#pragma once

#include "gpu_structures.h"
#include "grid_policy.h"
#include "feature_layout_common.h"
#include "feature_layout_feature_parallel.h"
#include "feature_layout_doc_parallel.h"
#include "feature_layout_single.h"

#include <catboost/cuda/cuda_lib/mapping.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/data/feature.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/data/data_provider.h>

namespace NCatboostCuda {
    //just meta for printing/debugging
    struct TDataSetDescription {
        TDataSetDescription() = default;
        TDataSetDescription(const TString& name)
            : Name(name)
        {}
        TString Name = "";
    };

    /*
     * Shared compressed index allows for fast GPU-tree apply in doc-parallel mode
     */
    template <class TLayoutPolicy = TFeatureParallelLayout>
    class TSharedCompressedIndex: public TMoveOnly {
    public:
        using TFeaturesMapping = typename TLayoutPolicy::TFeaturesMapping;
        using TSamplesMapping = typename TLayoutPolicy::TSamplesMapping;
        using TCompressedIndexMapping = typename TLayoutPolicy::TCompressedIndexMapping;
        using TPolicyBlock = TGpuFeaturesBlockDescription<TFeaturesMapping, TSamplesMapping>;

        class TCompressedDataSet: public TGuidHolder {
        public:
            using THistogramsMapping = typename TLayoutPolicy::TFeaturesMapping;

            TCompressedDataSet(const TDataSetDescription& description,
                               const TSamplesMapping& samplesMapping,
                               TCudaBuffer<ui32, TCompressedIndexMapping>& storage,
                               TVector<ui32> featureIds)
                : Description(description)
                , Storage(&storage)
                , SamplesMapping(samplesMapping)
                , FeatureIds(std::move(featureIds))
            {
            }

            bool HasFeature(ui32 featureId) const {
                return FeaturePolicy.has(featureId);
            }

            const NCudaLib::TDistributedObject<TCFeature>& GetTCFeature(ui32 featureId) const {
                Y_ASSERT(FeaturePolicy.has(featureId));
                const auto& policy = FeaturePolicy.at(featureId);
                return PolicyBlocks.at(policy)->GetTCFeature(featureId);
            }

            const TCudaBuffer<ui32, TCompressedIndexMapping>& GetCompressedIndex() const {
                return *Storage;
            };

            bool HasFeaturesForPolicy(EFeaturesGroupingPolicy policy) const {
                return PolicyBlocks.has(policy);
            }

            const THistogramsMapping& GetHistogramsMapping(EFeaturesGroupingPolicy policy) const {
                return PolicyBlocks.at(policy)->HistogramsMapping;
            }

            const THistogramsMapping& GetBestSplitStatsMapping(EFeaturesGroupingPolicy policy) const {
                return PolicyBlocks.at(policy)->BinFeaturesForBestSplits.GetMapping();
            }

            const TCudaBuffer<TCBinFeature, THistogramsMapping>& GetBinFeaturesForBestSplits(EFeaturesGroupingPolicy policy) const {
                return PolicyBlocks.at(policy)->BinFeaturesForBestSplits;
            }

            const NCudaLib::TDistributedObject<ui32>& GetBinFeatureCount(EFeaturesGroupingPolicy policy) const {
                return PolicyBlocks.at(policy)->BinFeatureCount;
            }

            const TVector<TCBinFeature>& GetBinFeatures(EFeaturesGroupingPolicy policy) const {
                return PolicyBlocks.at(policy)->BinFeatures;
            }

            ui32 GetGridSize(EFeaturesGroupingPolicy policy) const {
                if (!PolicyBlocks.has(policy)) {
                    return 0;
                }
                return PolicyBlocks.at(policy)->CudaFeaturesHost.size();
            }

            const TCudaBuffer<TCFeature, THistogramsMapping>& GetGrid(EFeaturesGroupingPolicy policy) const {
                return PolicyBlocks.at(policy)->CudaFeaturesDevice;
            }

            const TSamplesMapping& GetSamplesMapping() const {
                return SamplesMapping;
            }

            const NCudaLib::TDistributedObject<ui32> GetSampleCount() const {
                NCudaLib::TDistributedObject<ui32> size = CreateDistributedObject<ui32>(0);
                for (auto dev : SamplesMapping.NonEmptyDevices()) {
                    size.Set(dev, SamplesMapping.DeviceSlice(dev).Size());
                }
                return size;
            }

            ui64 SizeAt(ui32 dev) const {
                ui64 size = 0;
                for (auto& featuresBlock : PolicyBlocks) {
                    const auto& block = featuresBlock.second;
                    size += block->CIndexSizes.At(dev);
                }
                return size;
            }

            const TVector<ui32>& GetFeatures() const {
                return FeatureIds;
            }

            const TCpuGrid& GetCpuGrid(EFeaturesGroupingPolicy policy) const {
                return PolicyBlocks.at(policy)->Grid;
            }

            bool IsOneHot(ui32 featureId) const {
                ui32 localId = GetCpuGrid(GetFeaturePolicy(featureId)).InverseFeatures.at(featureId);
                return GetCpuGrid(GetFeaturePolicy(featureId)).IsOneHot[localId];
            }

            NCudaLib::TDistributedObject<ui64> Size() const {
                auto size = NCudaLib::GetCudaManager().CreateDistributedObject<ui64>(0u);
                for (auto& featuresBlock : PolicyBlocks) {
                    const auto& block = featuresBlock.second;
                    size += block->CIndexSizes;
                }
                return size;
            }

            EFeaturesGroupingPolicy GetFeaturePolicy(ui32 featureId) const {
                return FeaturePolicy.at(featureId);
            }

            ui32 GetFoldCount(ui32 featureId) const {
                return PolicyBlocks.at(GetFeaturePolicy(featureId))->CpuGrid.FoldCount(featureId);
            }

            ui32 GetFeatureCount() const {
                return FeatureIds.size();
            }

            ui32 GetDocCount() const {
                return SamplesMapping.GetObjectsSlice().Size();
            }

            void PrintInfo() const {
                MATRIXNET_INFO_LOG << "Compressed DataSet " << Description.Name << " with features #" << FeatureIds.size() << " features" << Endl;

                for (const auto& entry : PolicyBlocks) {
                    EFeaturesGroupingPolicy policy = entry.first;
                    const TPolicyBlock& block = *entry.second;
                    auto featuresMapping = block.CudaFeaturesDevice.GetMapping();
                    for (auto dev : featuresMapping.NonEmptyDevices()) {
                        const ui32 featuresAtDevice = featuresMapping.DeviceSlice(dev).Size();
                        const ui32 docsAtDevice = block.Samples.DeviceSlice(dev).Size();
                        MATRIXNET_INFO_LOG << "Grid policy " << policy << Endl << "Memory usage for " << featuresAtDevice << " features, " << docsAtDevice << " documents at #"
                                            << dev << ": " << block.CIndexSizes.At(dev) * sizeof(ui32) * 1.0 / 1024 / 1024 << " MB" << Endl;
                    }
                }
            }

        private:
            TDataSetDescription Description;
            TCudaBuffer<ui32, TCompressedIndexMapping>* Storage;
            TSamplesMapping SamplesMapping;
            TVector<ui32> FeatureIds;

            //featureId -> policy
            TMap<ui32, EFeaturesGroupingPolicy> FeaturePolicy;
            TMap<EFeaturesGroupingPolicy, THolder<TPolicyBlock>> PolicyBlocks;

            template <class>
            friend class TSharedCompressedIndexBuilder;
            friend class TTreeCtrDataSetBuilder;
        };

        ui32 DataSetCount() const {
            return DataSets.size();
        }

        const TCompressedDataSet& GetDataSet(ui32 dataSetId) const {
            CB_ENSURE(dataSetId < DataSets.size());
            Y_ASSERT(DataSets[dataSetId] != nullptr);
            return *DataSets[dataSetId];
        }

        NCudaLib::TDistributedObject<ui64> ComputeCompressedIndexSizes() const {
            NCudaLib::TDistributedObject<ui64> sizes = NCudaLib::GetCudaManager().CreateDistributedObject<ui64>(0);
            for (auto& dataSet : DataSets) {
                sizes += dataSet->Size();
            }
            return sizes;
        }

        const TCudaBuffer<ui32, TCompressedIndexMapping>& GetStorage() const {
            return FlatStorage;
        };

    private:
        /*flat storage allows to build dataSets with features overlap:
         * f1 f2 f3 from block1 and  f4 from block 2
         * f1 f2 f3 from block 1 and f4 from block 3
         * This layout allows in doc-parallel mode to apply model in one kernel without
        */
        TCudaBuffer<ui32, TCompressedIndexMapping> FlatStorage;
        TVector<THolder<TCompressedDataSet>> DataSets;

        template <class>
        friend class TSharedCompressedIndexBuilder;

        friend class TTreeCtrDataSetBuilder;
    };

    template <class TLayoutPolicy = TFeatureParallelLayout>
    using TCompressedDataSet = typename TSharedCompressedIndex<TLayoutPolicy>::TCompressedDataSet;

}
