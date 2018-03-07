#pragma once

#include "compressed_index.h"
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_util/transform.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/cuda_util/helpers.h>
#include <catboost/cuda/utils/cpu_random.h>
#include <util/random/shuffle.h>

namespace NCatboostCuda {
    template <class TLayoutPolicy = TFeatureParallelLayout>
    class TSharedCompressedIndexBuilder: public TNonCopyable {
    public:
        using TFeaturesMapping = typename TLayoutPolicy::TFeaturesMapping;
        using TSamplesMapping = typename TLayoutPolicy::TSamplesMapping;
        using TIndex = TSharedCompressedIndex<TLayoutPolicy>;

        TSharedCompressedIndexBuilder(TIndex& compressedIndex)
            : CompressedIndex(compressedIndex)
        {
        }

        template <EFeaturesGroupingPolicy Policy,
                  class TFeaturesBinarizationDescription>
        static void SplitByPolicy(const TFeaturesBinarizationDescription& featuresInfo,
                                  const TVector<ui32>& features,
                                  TVector<ui32>* policyFeatures,
                                  TVector<ui32>* restFeatures) {
            policyFeatures->clear();
            restFeatures->clear();

            for (auto feature : features) {
                const ui32 foldCount = featuresInfo.GetFoldsCount(feature);
                if (foldCount <= TCompressedIndexHelper<Policy>::MaxFolds()) {
                    policyFeatures->push_back(feature);
                } else {
                    restFeatures->push_back(feature);
                }
            }
        }

        using TDataSet = typename TIndex::TCompressedDataSet;

        template <class TFeaturesBinarizationDescription,
                  EFeaturesGroupingPolicy Policy>
        static TVector<ui32> ProceedPolicy(const TFeaturesBinarizationDescription& featuresInfo,
                                           const TSamplesMapping& samplesMapping,
                                           const TVector<ui32>& features,
                                           TDataSet* dataSet,
                                           NCudaLib::TDistributedObject<ui64>* compressedIndexOffsets) {
            auto& ds = *dataSet;
            TVector<ui32> policyFeatures;
            TVector<ui32> restFeatures;
            SplitByPolicy<Policy>(featuresInfo,
                                  features,
                                  &policyFeatures,
                                  &restFeatures);

            if (policyFeatures.size()) {
                ds.PolicyBlocks[Policy] = TCudaFeaturesLayoutHelper<TLayoutPolicy>::template CreateFeaturesBlock<Policy>(policyFeatures,
                                                                                                                         featuresInfo,
                                                                                                                         samplesMapping,
                                                                                                                         *compressedIndexOffsets);
                for (const ui32 f : policyFeatures) {
                    ds.FeaturePolicy[f] = Policy;
                }
                (*compressedIndexOffsets) += ds.PolicyBlocks[Policy]->CIndexSizes;
            }
            return restFeatures;
        }

        template <class TFeaturesBinarizationDescription>
        static inline TVector<ui32> FilterZeroFeatures(const TFeaturesBinarizationDescription& featuresInfo,
                                                       const TVector<ui32>& featureIds) {
            TVector<ui32> result;
            for (auto f : featureIds) {
                if (featuresInfo.GetFoldsCount(f) != 0) {
                    result.push_back(f);
                }
            }
            return result;
        }

        template <class TFeaturesBinarizationDescription>
        static ui32 AddDataSetToCompressedIndex(const TFeaturesBinarizationDescription& featuresInfo,
                                                const TDataSetDescription& description,
                                                const TSamplesMapping& samplesMapping,
                                                const TVector<ui32>& featureIds,
                                                TIndex* dst) {
            const ui32 blockId = dst->DataSets.size();
            TVector<ui32> restFeatures = FilterZeroFeatures(featuresInfo,
                                                            featureIds);

            dst->DataSets.push_back(new TDataSet(description,
                                                 samplesMapping,
                                                 dst->FlatStorage,
                                                 featureIds));
            auto& ds = *dst->DataSets.back();

            auto compressedIndexOffsets = dst->ComputeCompressedIndexSizes();

#define POLICY_BLOCK(Policy) \
    restFeatures = ProceedPolicy<TFeaturesBinarizationDescription, Policy>(featuresInfo, samplesMapping, restFeatures, &ds, &compressedIndexOffsets);

            POLICY_BLOCK(EFeaturesGroupingPolicy::BinaryFeatures)
            POLICY_BLOCK(EFeaturesGroupingPolicy::HalfByteFeatures)
            POLICY_BLOCK(EFeaturesGroupingPolicy::OneByteFeatures)

            CB_ENSURE(restFeatures.size() == 0, "Error: can't proceed some features");
            return blockId;
        }

        static void ResetStorage(TIndex* index) {
            auto compressedIndexSizes = index->ComputeCompressedIndexSizes();
            using TMapping = typename TIndex::TCompressedIndexMapping;
            index->FlatStorage.Reset(CreateMapping<TMapping>(compressedIndexSizes));
            FillBuffer(index->FlatStorage, static_cast<ui32>(0));
        }

        ui32 AddDataSet(const TBinarizationInfoProvider& featuresInfo,
                        const TDataSetDescription& description,
                        const TSamplesMapping& samplesMapping,
                        const TVector<ui32>& featureIds,
                        TAtomicSharedPtr<TVector<ui32>> gatherIndices = nullptr) {
            CB_ENSURE(!IsWritingStage, "Can't add block after writing stage");

            const ui32 blockId = AddDataSetToCompressedIndex(featuresInfo,
                                                             description,
                                                             samplesMapping,
                                                             featureIds,
                                                             &CompressedIndex);
            GatherIndex.push_back(gatherIndices);
            SeenFeatures.push_back(TSet<ui32>());
            return blockId;
        }

        TSharedCompressedIndexBuilder& PrepareToWrite() {
            StartWrite = Now();
            ResetStorage(&CompressedIndex);
            IsWritingStage = true;
            return *this;
        }

        template <class TBinType>
        TSharedCompressedIndexBuilder& Write(const ui32 dataSetId,
                                             const ui32 featureId,
                                             const ui32 binCount,
                                             const TVector<TBinType>& bins) {
            CB_ENSURE(IsWritingStage, "Error: prepare to write first");
            CB_ENSURE(dataSetId < GatherIndex.size(), "DataSet id is out of bounds: " << dataSetId << " "
                                                                                      << " total dataSets " << GatherIndex.size());
            auto& dataSet = *CompressedIndex.DataSets[dataSetId];

            const auto& docsMapping = dataSet.SamplesMapping;
            const NCudaLib::TDistributedObject<TCFeature>& feature = dataSet.GetTCFeature(featureId);
            CB_ENSURE(bins.size() == docsMapping.GetObjectsSlice().Size());
            CB_ENSURE(binCount > 1, "Feature is empty");
            for (ui32 dev = 0; dev < feature.DeviceCount(); ++dev) {
                if (!feature.IsEmpty(dev)) {
                    const ui32 folds = feature.At(dev).Folds;
                    CB_ENSURE(binCount <= (folds + 1), "There are #" << folds + 1 << " but need at least " << binCount << " to store feature");
                }
            }
            CB_ENSURE(!SeenFeatures[dataSetId].has(featureId), "Error: can't write feature twice");

            TVector<ui8> writeBins(bins.size());

            if (GatherIndex[dataSetId]) {
                NPar::ParallelFor(0, bins.size(), [&](ui32 i) {
                    writeBins[i] = bins[(*GatherIndex[dataSetId])[i]];
                    Y_ASSERT(writeBins[i] <= binCount);
                });
            } else {
                for (ui32 i = 0; i < bins.size(); ++i) {
                    writeBins[i] = bins[i];
                    Y_ASSERT(writeBins[i] <= binCount);
                }
            }
            //TODO(noxoomo): we could optimize this (for feature-parallel datasets)
            // by async write (common machines have 2 pci root complex, so it could be almost 2 times faster)
            // + some speedup on multi-host mode
            TCudaFeaturesLayoutHelper<TLayoutPolicy>::WriteToCompressedIndex(feature,
                                                                             writeBins,
                                                                             dataSet.GetSamplesMapping(),
                                                                             &CompressedIndex.FlatStorage);

            SeenFeatures[dataSetId].insert(featureId);
            return *this;
        }

        void Finish() {
            CB_ENSURE(!BuildIsDone, "Build could be finished only once");
            MATRIXNET_INFO_LOG << "Compressed index written in " << (Now() - StartWrite).SecondsFloat() << " seconds" << Endl;

            const ui32 blockCount = SeenFeatures.size();

            for (ui32 dataSetId = 0; dataSetId < blockCount; ++dataSetId) {
                auto& ds = *CompressedIndex.DataSets[dataSetId];
                const TDataSetDescription& description = ds.Description;
                ds.PrintInfo();
                for (ui32 f : ds.GetFeatures()) {
                    CB_ENSURE(SeenFeatures[dataSetId].count(f), "Unseen feature #" << f << " in dataset " << description.Name);
                }
            }

            BuildIsDone = true;
        };

    private:
        bool IsWritingStage = false;
        TInstant StartWrite = Now();
        bool BuildIsDone = false;

        TIndex& CompressedIndex;
        TVector<TSet<ui32>> SeenFeatures;
        TVector<TAtomicSharedPtr<TVector<ui32>>> GatherIndex;
    };
}
