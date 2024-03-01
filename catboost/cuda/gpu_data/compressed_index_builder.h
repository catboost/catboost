#pragma once

#include "compressed_index.h"
#include "kernels.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_util/transform.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/cuda_util/helpers.h>
#include <catboost/libs/data/lazy_columns.h>
#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/private/libs/data_util/path_with_scheme.h>
#include <catboost/private/libs/quantized_pool/loader.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/fwd.h>
#include <util/random/shuffle.h>

namespace NCatboostCuda {
    struct TDatasetPermutationOrderAndSubsetIndexing {
        TVector<ui32> IndicesVec;
        NCB::TFeaturesArraySubsetIndexing SubsetIndexing;
        TMaybe<NCB::TFeaturesArraySubsetInvertedIndexing> InvertedSubsetIndexing;

        static TAtomicSharedPtr<TDatasetPermutationOrderAndSubsetIndexing> ConstructShared(
            const NCB::TFeaturesArraySubsetIndexing& featuresArraySubsetIndexing,
            TVector<ui32>&& indicesVec
        ) {
            TVector<ui32> indicesCopy(indicesVec);
            return MakeAtomicShared<TDatasetPermutationOrderAndSubsetIndexing>(
                std::move(indicesCopy),
                NCB::Compose(
                    featuresArraySubsetIndexing,
                    NCB::TFeaturesArraySubsetIndexing(std::move(indicesVec))
                ),
                Nothing()
            );
        }
    };

    template <class TLayoutPolicy = TFeatureParallelLayout>
    class TSharedCompressedIndexBuilder: public TNonCopyable {
    public:
        using TFeaturesMapping = typename TLayoutPolicy::TFeaturesMapping;
        using TSamplesMapping = typename TLayoutPolicy::TSamplesMapping;
        using TIndex = TSharedCompressedIndex<TLayoutPolicy>;

        TSharedCompressedIndexBuilder(TIndex& compressedIndex,
                                      NPar::ILocalExecutor* localExecutor)
            : CompressedIndex(compressedIndex)
            , LocalExecutor(localExecutor)
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

            dst->DataSets.push_back(MakeHolder<TDataSet>(description,
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
                        TAtomicSharedPtr<TDatasetPermutationOrderAndSubsetIndexing> gatherIndices = nullptr) {
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

        void CheckBinCount(
            const NCudaLib::TDistributedObject<TCFeature>& feature,
            const ui32 binCount
        ) const {
            for (ui32 dev = 0; dev < feature.DeviceCount(); ++dev) {
                if (!feature.IsEmpty(dev)) {
                    const ui32 folds = feature.At(dev).Folds;
                    CB_ENSURE_INTERNAL(
                        folds == 0 || binCount <= (folds + 1),
                        "There are #" << folds + 1 << " but need at least " << binCount << " to store feature");
                }
            }
        }

        template <typename IQuantizedFeatureColumn>
        TSharedCompressedIndexBuilder& Write(
            const ui32 dataSetId,
            const ui32 featureId,
            const ui32 binCount,
            IQuantizedFeatureColumn* quantizedFeatureColumn,
            TMaybe<ui16> baseValue = Nothing()
        ) {
            CB_ENSURE(IsWritingStage, "Error: prepare to write first");
            CB_ENSURE(dataSetId < GatherIndex.size(), "DataSet id is out of bounds: " << dataSetId << " "
                                                                                      << " total dataSets " << GatherIndex.size());
            CB_ENSURE_INTERNAL(binCount > 1, "Feature #" << featureId << " is empty");

            auto& dataSet = *CompressedIndex.DataSets[dataSetId];
            const auto& docsMapping = dataSet.SamplesMapping;
            CB_ENSURE(quantizedFeatureColumn->GetSize() == docsMapping.GetObjectsSlice().Size());
            CB_ENSURE(!SeenFeatures[dataSetId].contains(featureId), "Error: can't write feature twice");
            THolder<IQuantizedFeatureColumn> reorderedColumn;
            if (GatherIndex[dataSetId]) {
                NCB::TCloningParams cloningParams;
                cloningParams.SubsetIndexing = &GatherIndex[dataSetId]->SubsetIndexing;
                if (quantizedFeatureColumn->IsSparse()) {
                    if (!GatherIndex[dataSetId]->InvertedSubsetIndexing) {
                        GatherIndex[dataSetId]->InvertedSubsetIndexing = NCB::GetInvertedIndexing(
                            GatherIndex[dataSetId]->SubsetIndexing,
                            GatherIndex[dataSetId]->SubsetIndexing.Size(),
                            LocalExecutor
                        );
                    }
                    cloningParams.InvertedSubsetIndexing = GatherIndex[dataSetId]->InvertedSubsetIndexing.Get();
                }
                reorderedColumn = NCB::DynamicHolderCast<IQuantizedFeatureColumn>(
                    quantizedFeatureColumn->CloneWithNewSubsetIndexing(
                        cloningParams,
                        LocalExecutor
                    ),
                    "Column feature type changed after cloning"
                );
                quantizedFeatureColumn = reorderedColumn.Get();
            }

            if (NCB::CastToLazyQuantizedFloatValuesHolder(quantizedFeatureColumn)) {
                CATBOOST_DEBUG_LOG << "Loading featureId " << featureId << " on device side" << Endl;
                NeedToDropLoaders = true;
                WriteLazyBinsVector(
                    dataSetId,
                    featureId,
                    binCount,
                    /*permute*/false,
                    NCB::CastToLazyQuantizedFloatValuesHolder(quantizedFeatureColumn),
                    baseValue);
            } else {
                TVector<ui8> writeBins;
                writeBins.yresize(quantizedFeatureColumn->GetSize());
                quantizedFeatureColumn->ParallelForEachBlock(
                    LocalExecutor,
                    [writeBinsPtr = writeBins.data(), baseValue] (size_t blockStartIdx, auto block) {
                        auto writePtr = writeBinsPtr + blockStartIdx;
                        if (baseValue.Defined()) {
                            for (auto i : xrange(block.size())) {
                                writePtr[i] = ClipWideHistValue(block[i], *baseValue);
                            }
                        } else {
                            for (auto i : xrange(block.size())) {
                                writePtr[i] = block[i];
                            }
                        }
                    },
                    4096 /*blockSize*/
                );
                WriteBinsVector(
                    dataSetId,
                    featureId,
                    binCount,
                    /*permute=*/ false,
                    writeBins
                );
            }
            return *this;
        }

        void WriteBinsVector(
            const ui32 dataSetId,
            const ui32 featureId,
            const ui32 binCount,
            bool permute,
            TConstArrayRef<ui8> binsVector
        ) {
            auto& dataSet = *CompressedIndex.DataSets[dataSetId];
            const NCudaLib::TDistributedObject<TCFeature>& feature = dataSet.GetTCFeature(featureId);

            CheckBinCount(feature, binCount);

            //TODO(noxoomo): we could optimize this (for feature-parallel datasets)
            // by async write (common machines have 2 pci root complex, so it could be almost 2 times faster)
            // + some speedup on multi-host mode
            if (!permute || !GatherIndex[dataSetId]) {
                TCudaFeaturesLayoutHelper<TLayoutPolicy>::WriteToCompressedIndex(
                    feature,
                    binsVector,
                    dataSet.GetSamplesMapping(),
                    &CompressedIndex.FlatStorage
                );
            } else {
                TVector<ui8> permutedBins;
                permutedBins.yresize(binsVector.size());
                auto& permutation = GatherIndex[dataSetId]->IndicesVec;
                Y_ASSERT(permutedBins.size() == permutation.size());
                for (ui32 i : xrange(permutation.size())) {
                    permutedBins[i] = binsVector[permutation[i]];
                }
                TCudaFeaturesLayoutHelper<TLayoutPolicy>::WriteToCompressedIndex(
                    feature,
                    permutedBins,
                    dataSet.GetSamplesMapping(),
                    &CompressedIndex.FlatStorage
                );
            }
            SeenFeatures[dataSetId].insert(featureId);
        }

        void WriteLazyBinsVector(
            const ui32 dataSetId,
            const ui32 featureId,
            const ui32 binCount,
            bool permute,
            const NCB::TLazyQuantizedFloatValuesHolder* lazyQuantizedColumn,
            TMaybe<ui16> baseValue
        ) {
            auto& dataSet = *CompressedIndex.DataSets[dataSetId];
            const NCudaLib::TDistributedObject<TCFeature>& feature = dataSet.GetTCFeature(featureId);

            CheckBinCount(feature, binCount);

            CB_ENSURE_INTERNAL(!permute, "Lazy columns should be shuffled by quantizer");

            TCudaFeaturesLayoutHelper<TLayoutPolicy>::WriteToLazyCompressedIndex(
                feature,
                lazyQuantizedColumn,
                featureId,
                baseValue,
                dataSet.GetSamplesMapping(),
                &CompressedIndex.FlatStorage
            );
            SeenFeatures[dataSetId].insert(featureId);
        }

        void Finish() {
            CB_ENSURE(!BuildIsDone, "Build could be finished only once");
            CATBOOST_DEBUG_LOG << "Compressed index was written in " << (Now() - StartWrite).SecondsFloat() << " seconds" << Endl;
            const ui32 blockCount = SeenFeatures.size();

            for (ui32 dataSetId = 0; dataSetId < blockCount; ++dataSetId) {
                auto& ds = *CompressedIndex.DataSets[dataSetId];
                ds.PrintInfo();
            }

            DropLoaders();

            BuildIsDone = true;
        };

        ~TSharedCompressedIndexBuilder() {
            DropLoaders();
        }
    private:
        bool IsWritingStage = false;
        TInstant StartWrite = Now();
        bool BuildIsDone = false;
        bool NeedToDropLoaders = false;

        TIndex& CompressedIndex;
        TVector<TSet<ui32>> SeenFeatures;
        TVector<TAtomicSharedPtr<TDatasetPermutationOrderAndSubsetIndexing>> GatherIndex;
        NPar::ILocalExecutor* LocalExecutor;

        void DropLoaders() {
            if (NeedToDropLoaders) {
                CATBOOST_DEBUG_LOG << __PRETTY_FUNCTION__ << ": waiting for cached loaders to go away..." << Endl;
                NCudaLib::GetCudaManager().WaitComplete();
                DropAllLoaders(CompressedIndex.DataSets[0]->GetSamplesMapping().NonEmptyDevices());
                NCudaLib::GetCudaManager().WaitComplete();
                NeedToDropLoaders = false;
                CATBOOST_DEBUG_LOG << __PRETTY_FUNCTION__ << ": done" << Endl;
            }
        }
    };

    extern template class TSharedCompressedIndexBuilder<TFeatureParallelLayout>;

    extern template class TSharedCompressedIndexBuilder<TDocParallelLayout>;

}
