#include "gpu_input_quantization.h"

#include "gpu_input_columns.h"
#include "gpu_input_provider.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>

#include <catboost/cuda/gpu_data/kernel/binarize.cuh>
#include <catboost/cuda/gpu_data/kernel/gpu_input_factorize.cuh>
#include <catboost/cuda/gpu_data/kernel/gpu_input_utils.cuh>

#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/helpers/exception.h>

#include <util/generic/algorithm.h>
#include <util/generic/xrange.h>
#include <util/string/cast.h>

#include <cuda_runtime_api.h>

#include <type_traits>

#include <cmath>

namespace NCB {
    namespace {

        static inline cudaStream_t GetCudaStreamFromCudaArrayInterface(ui64 stream) {
            if (stream == 0) {
                return 0;
            }
            if (stream == 1) {
                return cudaStreamPerThread;
            }
            return reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream));
        }

        static inline EBorderSelectionType NormalizeBorderSelectionTypeForGpuInput(
            EBorderSelectionType borderSelectionType
        ) {
            switch (borderSelectionType) {
                case EBorderSelectionType::GreedyLogSum:
                case EBorderSelectionType::GreedyMinEntropy:
                case EBorderSelectionType::Median:
                    return EBorderSelectionType::Median;
                case EBorderSelectionType::Uniform:
                    return EBorderSelectionType::Uniform;
                default:
                    // GPU-only quantization for native GPU inputs currently supports only border selection
                    // modes that can be computed fully on device without transferring dataset-sized buffers to CPU.
                    // Fall back to Median which is deterministic given fixed seed and matches GPU kernel support.
                    return EBorderSelectionType::Median;
            }
        }

        class TCudaDeviceBuffer final {
        public:
            TCudaDeviceBuffer() = default;

            explicit TCudaDeviceBuffer(ui64 bytes) {
                if (bytes) {
                    CUDA_SAFE_CALL(cudaMalloc(&Ptr, bytes));
                }
            }

            TCudaDeviceBuffer(const TCudaDeviceBuffer&) = delete;
            TCudaDeviceBuffer& operator=(const TCudaDeviceBuffer&) = delete;

            TCudaDeviceBuffer(TCudaDeviceBuffer&& other) noexcept
                : Ptr(other.Ptr)
            {
                other.Ptr = nullptr;
            }

            TCudaDeviceBuffer& operator=(TCudaDeviceBuffer&& other) noexcept {
                if (this != &other) {
                    Reset();
                    Ptr = other.Ptr;
                    other.Ptr = nullptr;
                }
                return *this;
            }

            ~TCudaDeviceBuffer() {
                Reset();
            }

            void* Get() const noexcept {
                return Ptr;
            }

        private:
            void Reset() noexcept {
                if (Ptr) {
                    cudaFree(Ptr);
                    Ptr = nullptr;
                }
            }

        private:
            void* Ptr = nullptr;
        };

        static TVector<float> ComputeFloatBordersOnGpu(
            const TGpuInputColumnDesc& column,
            ui32 objectCount,
            const NCatboostOptions::TBinarizationOptions& binarizationOptions
        ) {
            CB_ENSURE(column.Data != 0, "GPU input column pointer is null");
            CB_ENSURE(column.DeviceId >= 0, "Invalid device id for GPU input column");
            CB_ENSURE(column.StrideBytes > 0, "Invalid stride for GPU input column");

            const ui32 borderCount = binarizationOptions.BorderCount.Get();
            CB_ENSURE(borderCount > 0, "BorderCount must be > 0 for GPU input quantization");
            // BinarizeFloatFeature kernel stores borders in shared memory (256 entries).
            CB_ENSURE(borderCount <= 255, "BorderCount is too large for GPU float binarization kernel");

            const EBorderSelectionType borderSelectionType = NormalizeBorderSelectionTypeForGpuInput(
                binarizationOptions.BorderSelectionType.Get()
            );

            CUDA_SAFE_CALL(cudaSetDevice(column.DeviceId));
            const cudaStream_t stream = GetCudaStreamFromCudaArrayInterface(column.Stream);

            TCudaDeviceBuffer valuesBuf(static_cast<ui64>(objectCount) * sizeof(float));
            {
                const void* src = reinterpret_cast<const void*>(static_cast<uintptr_t>(column.Data));
                NKernel::CopyStridedGpuInputToFloat(
                    src,
                    column.StrideBytes,
                    objectCount,
                    static_cast<NKernel::EGpuInputDType>(static_cast<ui8>(column.DType)),
                    static_cast<float*>(valuesBuf.Get()),
                    stream
                );
            }

            if (borderSelectionType == EBorderSelectionType::Uniform) {
                float minValue = 0.0f;
                float maxValue = 0.0f;
                NKernel::ComputeMinMaxToHost(
                    static_cast<const float*>(valuesBuf.Get()),
                    objectCount,
                    &minValue,
                    &maxValue,
                    stream
                );

                if (!std::isfinite(minValue) || !std::isfinite(maxValue) || (minValue == maxValue)) {
                    return {};
                }

                TVector<float> borders;
                borders.yresize(borderCount);
                for (ui32 i = 0; i < borderCount; ++i) {
                    const double v = static_cast<double>(minValue)
                        + (static_cast<double>(i) + 1.0)
                            * (static_cast<double>(maxValue) - static_cast<double>(minValue))
                            / (static_cast<double>(borderCount) + 1.0);
                    borders[i] = static_cast<float>(v);
                }
                SortUnique(borders);
                return borders;
            } else {
                TCudaDeviceBuffer bordersBuf(static_cast<ui64>(borderCount + 1) * sizeof(float));
                float* bordersDev = static_cast<float*>(bordersBuf.Get());

                // Use FastGpuBorders (approximate quantiles). This is GPU-only and keeps D2H traffic bounded
                // to the resulting border values.
                NKernel::FastGpuBorders(
                    static_cast<const float*>(valuesBuf.Get()),
                    objectCount,
                    bordersDev,
                    borderCount,
                    stream
                );

                TVector<float> borders;
                borders.yresize(borderCount);
                NCudaLib::TMemcpyTracker::Instance().RecordMemcpyAsync(
                    /*dst*/ borders.data(),
                    /*src*/ bordersDev + 1,
                    static_cast<ui64>(borderCount) * sizeof(float),
                    cudaMemcpyDeviceToHost
                );
                CUDA_SAFE_CALL(cudaMemcpyAsync(
                    borders.data(),
                    bordersDev + 1,
                    static_cast<size_t>(borderCount) * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    stream
                ));
                CUDA_SAFE_CALL(cudaStreamSynchronize(stream));

                // Remove NaNs/Infs and duplicates; keep borders sorted and strict.
                EraseIf(borders, [] (float v) {
                    return !std::isfinite(v);
                });
                SortUnique(borders);
                return borders;
            }
        }

        static inline bool IsGpuInputIntegerDType(EGpuInputDType dtype) {
            switch (dtype) {
                case EGpuInputDType::Int8:
                case EGpuInputDType::Int16:
                case EGpuInputDType::Int32:
                case EGpuInputDType::Int64:
                case EGpuInputDType::UInt8:
                case EGpuInputDType::UInt16:
                case EGpuInputDType::UInt32:
                case EGpuInputDType::UInt64:
                    return true;
                default:
                    return false;
            }
        }

        static inline ui32 GetGpuInputItemSize(EGpuInputDType dtype) {
            switch (dtype) {
                case EGpuInputDType::Int8:
                case EGpuInputDType::UInt8:
                case EGpuInputDType::Bool:
                    return 1;
                case EGpuInputDType::Int16:
                case EGpuInputDType::UInt16:
                    return 2;
                case EGpuInputDType::Float32:
                case EGpuInputDType::Int32:
                case EGpuInputDType::UInt32:
                    return 4;
                case EGpuInputDType::Float64:
                case EGpuInputDType::Int64:
                case EGpuInputDType::UInt64:
                    return 8;
            }
            return 0;
        }

        static TCatFeaturePerfectHash UpdateCatPerfectHashFromHashes(
            const TCatFeaturePerfectHash& base,
            const TVector<ui32>& uniqueHashes,
            const TVector<ui32>& uniqueCounts,
            TVector<ui32>* binsForRank
        ) {
            CB_ENSURE(uniqueHashes.size() == uniqueCounts.size(), "uniqueHashes/uniqueCounts size mismatch");

            TCatFeaturePerfectHash result = base;
            binsForRank->yresize(uniqueHashes.size());

            for (ui32 i = 0; i < uniqueHashes.size(); ++i) {
                const ui32 hash = uniqueHashes[i];
                const ui32 count = uniqueCounts[i];

                if (result.DefaultMap && (result.DefaultMap->SrcValue == hash)) {
                    (*binsForRank)[i] = result.DefaultMap->DstValueWithCount.Value;
                    result.DefaultMap->DstValueWithCount.Count += count;
                    continue;
                }

                auto it = result.Map.find(hash);
                if (it == result.Map.end()) {
                    const ui32 bin = static_cast<ui32>(result.GetSize());
                    auto inserted = result.Map.emplace(hash, TValueWithCount{bin, count});
                    (*binsForRank)[i] = inserted.first->second.Value;
                } else {
                    (*binsForRank)[i] = it->second.Value;
                    it->second.Count += count;
                }
            }

            return result;
        }

    } // namespace

    TQuantizedObjectsDataProviderPtr QuantizeGpuInputData(
        const NCatboostOptions::TCatBoostOptions& params,
        TDataProviderPtr srcData,
        bool isLearnData,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        NPar::ILocalExecutor* localExecutor,
        TRestorableFastRng64* rand
    ) {
        Y_UNUSED(localExecutor);
        Y_UNUSED(rand);

        CB_ENSURE(srcData, "srcData is nullptr");
        CB_ENSURE(srcData->MetaInfo.FeaturesLayout, "FeaturesLayout must be set");

        const auto* gpuRawObjects
            = dynamic_cast<const TGpuRawObjectsDataProvider*>(srcData->ObjectsData.Get());
        CB_ENSURE(gpuRawObjects, "Expected TGpuRawObjectsDataProvider");

        const ui32 objectCount = gpuRawObjects->GetObjectCount();
        const auto& gpuInput = gpuRawObjects->GetData();

        const auto& dataProcessingOptions = params.DataProcessingOptions.Get();
        const auto& binarizationOptions = dataProcessingOptions.FloatFeaturesBinarization.Get();
        const auto& perFloatFeatureQuantization = dataProcessingOptions.PerFloatFeatureQuantization.Get();

        TQuantizedObjectsData quantizedObjectsData;
        if (quantizedFeaturesInfo) {
            quantizedObjectsData.QuantizedFeaturesInfo = quantizedFeaturesInfo;
        }
        quantizedObjectsData.PrepareForInitialization(
            srcData->MetaInfo,
            binarizationOptions,
            perFloatFeatureQuantization
        );

        quantizedFeaturesInfo = quantizedObjectsData.QuantizedFeaturesInfo;
        CB_ENSURE(quantizedFeaturesInfo, "QuantizedFeaturesInfo is not initialized");

        const auto& featuresLayout = *srcData->MetaInfo.FeaturesLayout;

        featuresLayout.IterateOverAvailableFeatures<EFeatureType::Float>(
            [&] (TFloatFeatureIdx floatFeatureIdx) {
                const ui32 flatFeatureIdx = featuresLayout.GetExternalFeatureIdx(*floatFeatureIdx, EFeatureType::Float);
                CB_ENSURE(flatFeatureIdx < gpuInput.Columns.size(), "GPU input column index is out of bounds");

                if (isLearnData && !quantizedFeaturesInfo->HasBorders(floatFeatureIdx)) {
                    const auto& binarization = quantizedFeaturesInfo->GetFloatFeatureBinarization(flatFeatureIdx);
                    quantizedFeaturesInfo->SetBorders(
                        floatFeatureIdx,
                        ComputeFloatBordersOnGpu(gpuInput.Columns[flatFeatureIdx], objectCount, binarization)
                    );
                }
                if (!isLearnData) {
                    CB_ENSURE(
                        quantizedFeaturesInfo->HasBorders(floatFeatureIdx),
                        "Borders are not initialized for float feature " << flatFeatureIdx
                    );
                }

                if (!quantizedObjectsData.FloatFeatures[*floatFeatureIdx]) {
                    quantizedObjectsData.FloatFeatures[*floatFeatureIdx] = MakeHolder<TGpuExternalFloatValuesHolder>(
                        flatFeatureIdx,
                        objectCount,
                        gpuInput.Columns[flatFeatureIdx],
                        quantizedFeaturesInfo
                    );
                }
            }
        );

        featuresLayout.IterateOverAvailableFeatures<EFeatureType::Categorical>(
            [&] (TCatFeatureIdx catFeatureIdx) {
                const ui32 flatFeatureIdx = featuresLayout.GetExternalFeatureIdx(*catFeatureIdx, EFeatureType::Categorical);
                CB_ENSURE(flatFeatureIdx < gpuInput.Columns.size(), "GPU input column index is out of bounds");

                const auto& column = gpuInput.Columns[flatFeatureIdx];
                CB_ENSURE(column.Data != 0, "GPU input categorical column pointer is null");
                CB_ENSURE(column.DeviceId >= 0, "Invalid device id for GPU input categorical column");
                CB_ENSURE(column.StrideBytes > 0, "Invalid stride for GPU input categorical column");
                CB_ENSURE(
                    IsGpuInputIntegerDType(column.DType),
                    "Native GPU categorical features currently support only integer dtypes; got dtype="
                        << static_cast<ui32>(static_cast<ui8>(column.DType))
                );

                NCudaLib::SetDevice(column.DeviceId);
                const cudaStream_t stream = GetCudaStreamFromCudaArrayInterface(column.Stream);

                const ui32 itemSize = GetGpuInputItemSize(column.DType);
                CB_ENSURE(itemSize > 0, "Unsupported GPU input dtype for categorical feature");

                // Factorize values on GPU: ranks per row + unique values/counts.
                TCudaDeviceBuffer ranksBuf(static_cast<ui64>(objectCount) * sizeof(ui32));
                TCudaDeviceBuffer countsBuf(static_cast<ui64>(objectCount) * sizeof(ui32));
                TCudaDeviceBuffer uniqueCountBuf(sizeof(ui32));
                TCudaDeviceBuffer uniqueValuesBuf(static_cast<ui64>(objectCount) * itemSize);

                const void* src = reinterpret_cast<const void*>(static_cast<uintptr_t>(column.Data));
                NKernel::FactorizeStridedGpuInputToUnique(
                    src,
                    column.StrideBytes,
                    objectCount,
                    static_cast<NKernel::EGpuInputDType>(static_cast<ui8>(column.DType)),
                    static_cast<ui32*>(ranksBuf.Get()),
                    uniqueValuesBuf.Get(),
                    static_cast<ui32*>(countsBuf.Get()),
                    static_cast<ui32*>(uniqueCountBuf.Get()),
                    stream
                );

                ui32 uniqueCount = 0;
                NCudaLib::TMemcpyTracker::Instance().RecordMemcpyAsync(
                    /*dst*/ &uniqueCount,
                    /*src*/ uniqueCountBuf.Get(),
                    sizeof(ui32),
                    cudaMemcpyDeviceToHost
                );
                CUDA_SAFE_CALL(cudaMemcpyAsync(
                    &uniqueCount,
                    uniqueCountBuf.Get(),
                    sizeof(ui32),
                    cudaMemcpyDeviceToHost,
                    stream
                ));
                CUDA_SAFE_CALL(cudaStreamSynchronize(stream));

                if (uniqueCount <= 1 && isLearnData) {
                    quantizedFeaturesInfo->GetFeaturesLayout()->IgnoreExternalFeature(flatFeatureIdx);
                    return;
                }

                TVector<ui32> uniqueCounts;
                uniqueCounts.yresize(uniqueCount);
                NCudaLib::TMemcpyTracker::Instance().RecordMemcpyAsync(
                    /*dst*/ uniqueCounts.data(),
                    /*src*/ countsBuf.Get(),
                    static_cast<ui64>(uniqueCount) * sizeof(ui32),
                    cudaMemcpyDeviceToHost
                );
                CUDA_SAFE_CALL(cudaMemcpyAsync(
                    uniqueCounts.data(),
                    countsBuf.Get(),
                    static_cast<size_t>(uniqueCount) * sizeof(ui32),
                    cudaMemcpyDeviceToHost,
                    stream
                ));

                TVector<ui32> binsForRank;
                TCatFeaturePerfectHash basePerfectHash;
                if (!isLearnData) {
                    basePerfectHash = quantizedFeaturesInfo->GetCategoricalFeaturesPerfectHash(catFeatureIdx);
                } else if (quantizedFeaturesInfo->GetUniqueValuesCounts(catFeatureIdx).OnAll > 0) {
                    basePerfectHash = quantizedFeaturesInfo->GetCategoricalFeaturesPerfectHash(catFeatureIdx);
                }

                // Compute CatBoost categorical hashes on GPU for unique values, then update perfect hash on CPU.
                TCudaDeviceBuffer uniqueHashesBuf(static_cast<ui64>(uniqueCount) * sizeof(ui32));
                NKernel::HashUniqueNumericToCatHash(
                    uniqueValuesBuf.Get(),
                    uniqueCount,
                    static_cast<NKernel::EGpuInputDType>(static_cast<ui8>(column.DType)),
                    static_cast<ui32*>(uniqueHashesBuf.Get()),
                    stream
                );

                TVector<ui32> uniqueHashes;
                uniqueHashes.yresize(uniqueCount);
                NCudaLib::TMemcpyTracker::Instance().RecordMemcpyAsync(
                    /*dst*/ uniqueHashes.data(),
                    /*src*/ uniqueHashesBuf.Get(),
                    static_cast<ui64>(uniqueCount) * sizeof(ui32),
                    cudaMemcpyDeviceToHost
                );
                CUDA_SAFE_CALL(cudaMemcpyAsync(
                    uniqueHashes.data(),
                    uniqueHashesBuf.Get(),
                    static_cast<size_t>(uniqueCount) * sizeof(ui32),
                    cudaMemcpyDeviceToHost,
                    stream
                ));
                CUDA_SAFE_CALL(cudaStreamSynchronize(stream));

                TCatFeaturePerfectHash updatedPerfectHash = UpdateCatPerfectHashFromHashes(
                    basePerfectHash,
                    uniqueHashes,
                    uniqueCounts,
                    &binsForRank
                );

                quantizedFeaturesInfo->UpdateCategoricalFeaturesPerfectHash(catFeatureIdx, std::move(updatedPerfectHash));

                // Build per-row bins on GPU by mapping ranks to perfect-hash bins.
                TSingleBuffer<ui32> bins = TSingleBuffer<ui32>::Create(NCudaLib::TSingleMapping(column.DeviceId, objectCount));
                // Ensure TCudaBuffer allocation is materialized before using raw pointers with CUDA runtime.
                NCudaLib::GetCudaManager().WaitComplete(NCudaLib::TDevicesListBuilder::SingleDevice(column.DeviceId));

                TCudaDeviceBuffer binsForRankBuf(static_cast<ui64>(uniqueCount) * sizeof(ui32));
                CUDA_SAFE_CALL(cudaMemcpyAsync(
                    binsForRankBuf.Get(),
                    binsForRank.data(),
                    static_cast<size_t>(uniqueCount) * sizeof(ui32),
                    cudaMemcpyHostToDevice,
                    stream
                ));

                NKernel::MapRanksToBins(
                    static_cast<const ui32*>(ranksBuf.Get()),
                    objectCount,
                    static_cast<const ui32*>(binsForRankBuf.Get()),
                    bins.At(column.DeviceId).Get(),
                    stream
                );
                CUDA_SAFE_CALL(cudaStreamSynchronize(stream));

                if (!quantizedObjectsData.CatFeatures[*catFeatureIdx]) {
                    const ui32* binsPtr = bins.At(column.DeviceId).Get();
                    auto binsShared = MakeAtomicShared<TSingleBuffer<ui32>>(std::move(bins));
                    quantizedObjectsData.CatFeatures[*catFeatureIdx] = MakeHolder<TGpuExternalCatValuesHolder>(
                        flatFeatureIdx,
                        objectCount,
                        column.DeviceId,
                        binsPtr,
                        std::move(binsShared),
                        quantizedFeaturesInfo
                    );
                }
            }
        );

        TCommonObjectsData commonData;
        commonData.PrepareForInitialization(srcData->MetaInfo, objectCount, /*prevTailCount*/ 0);
        commonData.SubsetIndexing = MakeAtomicShared<TArraySubsetIndexing<ui32>>(TFullSubset<ui32>(objectCount));
        commonData.Order = srcData->ObjectsData->GetOrder();

        TGpuInputTargets gpuTargets;
        gpuTargets.TargetCount = gpuInput.TargetCount;
        gpuTargets.Targets = gpuInput.Targets;
        gpuTargets.HasWeights = gpuInput.HasWeights;
        gpuTargets.Weights = gpuInput.Weights;

        return MakeIntrusive<TGpuInputQuantizedObjectsDataProvider>(
            srcData->ObjectsGrouping,
            std::move(commonData),
            std::move(quantizedObjectsData),
            std::move(gpuTargets),
            /*skipCheck*/ true,
            /*localExecutor*/ Nothing()
        );
    }

}
