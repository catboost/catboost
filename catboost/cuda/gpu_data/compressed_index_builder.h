#pragma once

#include "compressed_index.h"
#include "kernels.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/buffer_resharding.h>
#include <catboost/cuda/cuda_util/transform.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/data/gpu_input_columns.h>
#include <catboost/cuda/cuda_util/helpers.h>
#include <catboost/cuda/gpu_data/kernel/gpu_input_factorize.cuh>
#include <catboost/cuda/gpu_data/kernel/gpu_input_utils.cuh>
#include <catboost/libs/data/lazy_columns.h>
#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/private/libs/data_util/path_with_scheme.h>
#include <catboost/private/libs/quantized_pool/loader.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/fwd.h>
#include <util/random/shuffle.h>

#include <type_traits>

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
            GpuGatherIndex.push_back(nullptr);
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

            if (auto* gpuFloatHolder = dynamic_cast<const NCB::TGpuExternalFloatValuesHolder*>(quantizedFeatureColumn)) {
                CB_ENSURE_INTERNAL(!baseValue.Defined(), "Wide float features are not supported for native GPU inputs yet");

                const auto& column = gpuFloatHolder->GetColumnDesc();
                CB_ENSURE(column.DeviceId >= 0, "Invalid device id for native GPU float column");
                const ui32 srcDev = SafeIntegerCast<ui32>(column.DeviceId);

                auto values = TSingleBuffer<float>::Create(NCudaLib::TSingleMapping(srcDev, docsMapping.GetObjectsSlice().Size()));
                // TCudaBuffer allocations are asynchronous (via device task queue). We need a raw device pointer
                // for cudaMemcpy2D below, so ensure allocation is materialized before dereferencing handle-based
                // pointers on host.
                NCudaLib::GetCudaManager().WaitComplete(NCudaLib::TDevicesListBuilder::SingleDevice(srcDev));
                {
                    NCudaLib::SetDevice(column.DeviceId);
                    const void* src = reinterpret_cast<const void*>(static_cast<uintptr_t>(column.Data));
                    if (column.DType == NCB::EGpuInputDType::Float32) {
                        CUDA_SAFE_CALL(cudaMemcpy2D(
                            values.At(srcDev).Get(),
                            sizeof(float),
                            src,
                            column.StrideBytes,
                            sizeof(float),
                            docsMapping.GetObjectsSlice().Size(),
                            cudaMemcpyDeviceToDevice
                        ));
                    } else {
                        NKernel::CopyStridedGpuInputToFloat(
                            src,
                            column.StrideBytes,
                            SafeIntegerCast<ui32>(docsMapping.GetObjectsSlice().Size()),
                            static_cast<NKernel::EGpuInputDType>(static_cast<ui8>(column.DType)),
                            values.At(srcDev).Get(),
                            /*stream*/ 0
                        );
                        CUDA_SAFE_CALL(cudaStreamSynchronize(0));
                    }
                }

                const auto& qfi = gpuFloatHolder->GetQuantizedFeaturesInfo();
                CB_ENSURE(qfi, "QuantizedFeaturesInfo is not set for GPU float holder");
                CB_ENSURE(
                    qfi->GetFeaturesLayout()->IsCorrectExternalFeatureIdxAndType(gpuFloatHolder->GetId(), EFeatureType::Float),
                    "Invalid float feature id " << gpuFloatHolder->GetId()
                );
                const auto floatFeatureIdx = qfi->GetFeaturesLayout()->GetInternalFeatureIdx<EFeatureType::Float>(
                    gpuFloatHolder->GetId()
                );

                const auto& borders = qfi->GetBorders(floatFeatureIdx);
                const auto valuesConst = values.ConstCopyView();
                WriteFloatFeatureFromDeviceValues(
                    dataSetId,
                    featureId,
                    binCount,
                    /*permute*/ true,
                    valuesConst,
                    borders,
                    /*stream*/ 0
                );
                return *this;
            }

            if (auto* gpuCatHolder = dynamic_cast<const NCB::TGpuExternalCatValuesHolder*>(quantizedFeatureColumn)) {
                CB_ENSURE_INTERNAL(!baseValue.Defined(), "Wide categorical features are not supported for native GPU inputs yet");
                CB_ENSURE_INTERNAL(binCount <= 256, "Categorical binCount is too large for compressed index (expected one-hot feature)");

                const ui32 docCount = SafeIntegerCast<ui32>(docsMapping.GetObjectsSlice().Size());

                CB_ENSURE(gpuCatHolder->GetDeviceId() >= 0, "Invalid device id for native GPU categorical bins");
                const ui32 srcDev = SafeIntegerCast<ui32>(gpuCatHolder->GetDeviceId());

                const NCudaLib::TDistributedObject<TCFeature>& feature = dataSet.GetTCFeature(featureId);

                // Optionally apply dataset gather/permutation on the source device (full vector) before sharding.
                const auto& binsSrc = gpuCatHolder->GetBins();
                const TSingleBuffer<ui32>* binsForWrite = &binsSrc;

                TSingleBuffer<ui32> permutedBins;
                if (GatherIndex[dataSetId] && !GatherIndex[dataSetId]->IndicesVec.empty()) {
                    const auto& indices = GatherIndex[dataSetId]->IndicesVec;
                    CB_ENSURE(indices.size() == docCount, "Gather index size mismatch");

                    TSingleBuffer<ui32> gatherMap = TSingleBuffer<ui32>::Create(NCudaLib::TSingleMapping(srcDev, docCount));
                    gatherMap.Write(indices);

                    permutedBins = TSingleBuffer<ui32>::Create(NCudaLib::TSingleMapping(srcDev, docCount));
                    ::Gather(permutedBins, binsSrc, gatherMap, /*stream*/ 0);
                    binsForWrite = &permutedBins;
                }

                if constexpr (std::is_same_v<TSamplesMapping, NCudaLib::TStripeMapping>) {
                    // Doc-parallel: shard bins across devices, then cast to ui8 locally.
                    TStripeBuffer<ui32> binsStripe = TStripeBuffer<ui32>::Create(docsMapping);
                    NCudaLib::Reshard(*binsForWrite, binsStripe, /*stream*/ 0);
                    NCudaLib::GetCudaManager().WaitComplete(binsStripe.NonEmptyDevices());

                    TStripeBuffer<ui8> binsUi8 = TStripeBuffer<ui8>::Create(docsMapping);
                    NCudaLib::GetCudaManager().WaitComplete(binsUi8.NonEmptyDevices());
                    for (ui32 dev : binsUi8.NonEmptyDevices()) {
                        NCudaLib::SetDevice(SafeIntegerCast<i32>(dev));
                        const ui32 sliceSize = SafeIntegerCast<ui32>(docsMapping.DeviceSlice(dev).Size());
                        NKernel::GatherUi32BinsToUi8(
                            binsStripe.At(dev).Get(),
                            sliceSize,
                            /*gatherIndices*/ nullptr,
                            binsUi8.At(dev).Get(),
                            /*stream*/ 0
                        );
                        CUDA_SAFE_CALL(cudaStreamSynchronize(0));
                    }
                    WriteCompressedFeature(feature, binsUi8, CompressedIndex.FlatStorage, /*stream*/ 0);
                } else {
                    // Feature-parallel: feature is stored on a single device.
                    ui32 writeDev = static_cast<ui32>(-1);
                    for (ui32 dev = 0; dev < feature.DeviceCount(); ++dev) {
                        if (!feature.IsEmpty(dev)) {
                            CB_ENSURE(writeDev == static_cast<ui32>(-1));
                            writeDev = dev;
                        }
                    }
                    CB_ENSURE(writeDev != static_cast<ui32>(-1));
                    CB_ENSURE(
                        srcDev == writeDev,
                        "GPU categorical bins are on device " << srcDev << " but feature is written on device " << writeDev
                    );

                    auto binsGpu = TSingleBuffer<ui8>::Create(NCudaLib::TSingleMapping(writeDev, docCount));
                    NCudaLib::GetCudaManager().WaitComplete(NCudaLib::TDevicesListBuilder::SingleDevice(writeDev));

                    NKernel::GatherUi32BinsToUi8(
                        binsForWrite->At(writeDev).Get(),
                        docCount,
                        /*gatherIndices*/ nullptr,
                        binsGpu.At(writeDev).Get(),
                        /*stream*/ 0
                    );
                    CUDA_SAFE_CALL(cudaStreamSynchronize(0));
                    WriteCompressedFeature(feature, binsGpu, CompressedIndex.FlatStorage, /*stream*/ 0);
                }

                SeenFeatures[dataSetId].insert(featureId);
                return *this;
            }

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

        // Binarize a GPU-resident float vector (native GPU input / CTR) and write directly to the compressed index.
        // This avoids host staging of dataset-sized buffers.
        void WriteFloatFeatureFromDeviceValues(
            const ui32 dataSetId,
            const ui32 featureId,
            const ui32 binCount,
            bool permute,
            const TSingleBuffer<const float>& values,
            TConstArrayRef<float> borders,
            ui32 stream
        ) {
            auto& dataSet = *CompressedIndex.DataSets[dataSetId];
            const NCudaLib::TDistributedObject<TCFeature>& feature = dataSet.GetTCFeature(featureId);

            CheckBinCount(feature, binCount);

            const auto srcMapping = values.GetMapping();
            const ui32 srcDev = srcMapping.GetDeviceId();
            const ui32 docCount = SafeIntegerCast<ui32>(srcMapping.GetObjectsSlice().Size());
            CB_ENSURE(docCount == SafeIntegerCast<ui32>(values.GetObjectsSlice().Size()), "Unexpected GPU float values size");

            // Determine where this feature is stored.
            TVector<ui32> nonEmptyDevs;
            nonEmptyDevs.reserve(feature.DeviceCount());
            for (ui32 dev = 0; dev < feature.DeviceCount(); ++dev) {
                if (!feature.IsEmpty(dev)) {
                    nonEmptyDevs.push_back(dev);
                }
            }
            CB_ENSURE(!nonEmptyDevs.empty(), "Attempt to write an empty feature " << featureId);

            // Prepare borders in the format expected by the GPU binarization kernel: [count, borders...].
            TVector<float> bordersWithHeader;
            bordersWithHeader.yresize(borders.size() + 1);
            bordersWithHeader[0] = static_cast<float>(borders.size());
            Copy(borders.begin(), borders.end(), bordersWithHeader.begin() + 1);

            // If dataset permutation is requested, materialize permuted values on the source device.
            const bool needPermute = permute && GatherIndex[dataSetId] && !GatherIndex[dataSetId]->IndicesVec.empty();
            const TSingleBuffer<const float>* valuesForWrite = &values;

            TSingleBuffer<float> permutedValues;
            TSingleBuffer<const float> permutedValuesConst;
            if (needPermute) {
                const auto& indices = GatherIndex[dataSetId]->IndicesVec;
                CB_ENSURE(indices.size() == docCount, "Gather index size mismatch");

                TSingleBuffer<ui32> gatherMap = TSingleBuffer<ui32>::Create(NCudaLib::TSingleMapping(srcDev, docCount));
                gatherMap.Write(indices);

                permutedValues = TSingleBuffer<float>::Create(NCudaLib::TSingleMapping(srcDev, docCount));
                ::Gather(permutedValues, values, gatherMap, stream);
                permutedValuesConst = permutedValues.ConstCopyView();
                valuesForWrite = &permutedValuesConst;
            }

            using TKernel = NKernelHost::TBinarizeFloatFeatureKernel;

            if constexpr (std::is_same_v<TSamplesMapping, NCudaLib::TStripeMapping>) {
                if (nonEmptyDevs.size() == 1) {
                    const ui32 writeDev = nonEmptyDevs[0];
                    CB_ENSURE(
                        writeDev == srcDev,
                        "Native GPU float values are on device " << srcDev
                            << " but feature is written on device " << writeDev
                    );

                    auto bordersGpu = TSingleBuffer<float>::Create(NCudaLib::TSingleMapping(writeDev, bordersWithHeader.size()));
                    bordersGpu.Write(bordersWithHeader);

                    LaunchKernels<TKernel>(
                        valuesForWrite->NonEmptyDevices(),
                        stream,
                        *valuesForWrite,
                        bordersGpu,
                        feature,
                        /*gatherIndex*/ NKernelHost::TCudaBufferPtr<const ui32>::Nullptr(),
                        CompressedIndex.FlatStorage,
                        /*atomicUpdate*/ false
                    );
                } else {
                    // Doc-parallel: distribute (possibly permuted) values across devices and binarize locally on each device.
                    auto bordersGpu = TMirrorBuffer<float>::Create(NCudaLib::TMirrorMapping(bordersWithHeader.size()));
                    bordersGpu.Write(bordersWithHeader);

                    TStripeBuffer<float> valuesStripe = TStripeBuffer<float>::Create(dataSet.GetSamplesMapping());
                    NCudaLib::Reshard(*valuesForWrite, valuesStripe, stream);

                    LaunchKernels<TKernel>(
                        valuesStripe.NonEmptyDevices(),
                        stream,
                        valuesStripe,
                        bordersGpu,
                        feature,
                        /*gatherIndex*/ NKernelHost::TCudaBufferPtr<const ui32>::Nullptr(),
                        CompressedIndex.FlatStorage,
                        /*atomicUpdate*/ false
                    );
                }
            } else {
                CB_ENSURE_INTERNAL(
                    nonEmptyDevs.size() == 1,
                    "Mirror samples mapping expects feature to be stored on a single device"
                );

                const ui32 writeDev = nonEmptyDevs[0];
                CB_ENSURE(
                    writeDev == srcDev,
                    "Native GPU float values are on device " << srcDev
                        << " but feature is written on device " << writeDev
                );

                auto bordersGpu = TSingleBuffer<float>::Create(NCudaLib::TSingleMapping(writeDev, bordersWithHeader.size()));
                bordersGpu.Write(bordersWithHeader);

                LaunchKernels<TKernel>(
                    valuesForWrite->NonEmptyDevices(),
                    stream,
                    *valuesForWrite,
                    bordersGpu,
                    feature,
                    /*gatherIndex*/ NKernelHost::TCudaBufferPtr<const ui32>::Nullptr(),
                    CompressedIndex.FlatStorage,
                    /*atomicUpdate*/ false
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
        TVector<THolder<TSingleBuffer<ui32>>> GpuGatherIndex;
        NPar::ILocalExecutor* LocalExecutor;

        const TSingleBuffer<ui32>* GetOrCreateGpuGatherIndex(ui32 dataSetId, ui32 devId, ui32 docCount) {
            if (!GatherIndex[dataSetId]) {
                return nullptr;
            }
            const auto& indices = GatherIndex[dataSetId]->IndicesVec;
            if (indices.empty()) {
                return nullptr;
            }
            CB_ENSURE(indices.size() == docCount, "Gather index size mismatch");

            if (!GpuGatherIndex[dataSetId]) {
                auto bufHolder = MakeHolder<TSingleBuffer<ui32>>();
                *bufHolder = TSingleBuffer<ui32>::Create(NCudaLib::TSingleMapping(devId, docCount));
                bufHolder->Write(indices);
                GpuGatherIndex[dataSetId] = std::move(bufHolder);
            }
            return GpuGatherIndex[dataSetId].Get();
        }

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
