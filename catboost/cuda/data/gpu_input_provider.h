#pragma once

#include <catboost/libs/data/objects.h>
#include <catboost/libs/helpers/resource_holder.h>

#include <util/generic/vector.h>
#include <util/system/types.h>

namespace NCB {

    enum class EGpuInputDType : ui8 {
        Float32,
        Float64,
        Int8,
        Int16,
        Int32,
        Int64,
        UInt8,
        UInt16,
        UInt32,
        UInt64,
        Bool,
    };

    struct TGpuInputColumnDesc {
        ui64 Data = 0; // device pointer
        ui64 StrideBytes = 0; // distance between consecutive rows in bytes
        ui32 FullObjectCount = 0;
        EGpuInputDType DType = EGpuInputDType::Float32;
        i32 DeviceId = -1;
        ui64 Stream = 0; // cudaStream_t stored as integer (can be 0/1 as in __cuda_array_interface__)

        // Optional categorical dictionary support (used for cuDF categorical columns):
        // if non-zero, `Data` points to integer category codes and `CatHashDictDevicePtr` maps code -> CalcCatFeatureHash(str(category)).
        ui64 CatHashDictDevicePtr = 0; // device pointer to ui32[CatHashDictSize]
        ui32 CatHashDictSize = 0;
        ui32 CatHashNullValue = 0; // hash used for null / out-of-range codes

        bool operator==(const TGpuInputColumnDesc& rhs) const {
            return std::tie(Data, StrideBytes, FullObjectCount, DType, DeviceId, Stream, CatHashDictDevicePtr, CatHashDictSize, CatHashNullValue)
                == std::tie(rhs.Data, rhs.StrideBytes, rhs.FullObjectCount, rhs.DType, rhs.DeviceId, rhs.Stream, rhs.CatHashDictDevicePtr, rhs.CatHashDictSize, rhs.CatHashNullValue);
        }
    };

    struct TGpuInputData {
        ui32 FeatureCount = 0;
        TVector<TGpuInputColumnDesc> Columns; // [featureIdx]

        // Optional GPU-resident targets/weights for native GPU input path.
        // Targets can be 1D (TargetCount=1) or 2D (TargetCount>1) represented as per-column descriptors.
        ui32 TargetCount = 0;
        TVector<TGpuInputColumnDesc> Targets; // [targetIdx]
        bool HasWeights = false;
        TGpuInputColumnDesc Weights;

        // Keeps backing Python/DLPack objects alive for the duration of training.
        // Not included in equality checks intentionally.
        TVector<TIntrusivePtr<IResourceHolder>> ResourceHolders;

        bool operator==(const TGpuInputData& rhs) const {
            return std::tie(FeatureCount, Columns, TargetCount, Targets, HasWeights, Weights)
                == std::tie(rhs.FeatureCount, rhs.Columns, rhs.TargetCount, rhs.Targets, rhs.HasWeights, rhs.Weights);
        }
    };

    struct TGpuInputTargets {
        ui32 TargetCount = 0;
        TVector<TGpuInputColumnDesc> Targets; // [targetIdx]
        bool HasWeights = false;
        TGpuInputColumnDesc Weights;
    };

    class TGpuRawObjectsDataProvider final : public TObjectsDataProvider {
    public:
        using TData = TGpuInputData;

    public:
        TGpuRawObjectsDataProvider(
            TMaybe<TObjectsGroupingPtr> objectsGrouping,
            TCommonObjectsData&& commonData,
            TGpuInputData&& data,
            bool skipCheck
        )
            : TObjectsDataProvider(std::move(objectsGrouping), std::move(commonData), skipCheck)
            , Data(std::move(data))
        {}

        bool EqualTo(const TObjectsDataProvider& rhs, bool ignoreSparsity = false) const override {
            const auto* rhsGpuData = dynamic_cast<const TGpuRawObjectsDataProvider*>(&rhs);
            if (!rhsGpuData) {
                return false;
            }
            return TObjectsDataProvider::EqualTo(rhs, ignoreSparsity) && (Data == rhsGpuData->Data);
        }

        bool HasDenseData() const override {
            return true;
        }

        bool HasSparseData() const override {
            return false;
        }

        TIntrusivePtr<TObjectsDataProvider> GetSubsetImpl(
            const TObjectsGroupingSubset& objectsGroupingSubset,
            TMaybe<TConstArrayRef<ui32>> ignoredFeatures,
            ui64 cpuRamLimit,
            NPar::ILocalExecutor* localExecutor
        ) const override;

        const TGpuInputData& GetData() const noexcept {
            return Data;
        }

    private:
        TGpuInputData Data;
    };

    class TGpuInputQuantizedObjectsDataProvider final : public TQuantizedObjectsDataProvider {
    public:
        TGpuInputQuantizedObjectsDataProvider(
            TMaybe<TObjectsGroupingPtr> objectsGrouping,
            TCommonObjectsData&& commonData,
            TQuantizedObjectsData&& data,
            TGpuInputTargets&& gpuTargets,
            bool skipCheck,
            TMaybe<NPar::ILocalExecutor*> localExecutor
        )
            : TQuantizedObjectsDataProvider(
                std::move(objectsGrouping),
                std::move(commonData),
                std::move(data),
                skipCheck,
                localExecutor
              )
            , GpuTargets(std::move(gpuTargets))
        {}

        const TGpuInputTargets& GetGpuTargets() const noexcept {
            return GpuTargets;
        }

    private:
        TGpuInputTargets GpuTargets;
    };

} // namespace NCB
