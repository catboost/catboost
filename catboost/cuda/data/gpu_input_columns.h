#pragma once

#include "gpu_input_provider.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>

#include <catboost/libs/data/columns.h>
#include <catboost/libs/data/quantized_features_info.h>

#include <util/generic/ptr.h>

namespace NCB {

    class TGpuExternalFloatValuesHolder final : public IQuantizedFloatValuesHolder {
    public:
        TGpuExternalFloatValuesHolder(
            ui32 featureId,
            ui32 size,
            TGpuInputColumnDesc column,
            TQuantizedFeaturesInfoPtr quantizedFeaturesInfo
        )
            : IQuantizedFloatValuesHolder(featureId, size)
            , Column(std::move(column))
            , QuantizedFeaturesInfo(std::move(quantizedFeaturesInfo))
        {}

        bool IsSparse() const override {
            return false;
        }

        ui64 EstimateMemoryForCloning(const TCloningParams& cloningParams) const override {
            Y_UNUSED(cloningParams);
            return 0;
        }

        ui32 CalcChecksum(NPar::ILocalExecutor* localExecutor) const override;

        THolder<IFeatureValuesHolder> CloneWithNewSubsetIndexing(
            const TCloningParams& cloningParams,
            NPar::ILocalExecutor* localExecutor
        ) const override;

        IDynamicBlockIteratorBasePtr GetBlockIterator(ui32 offset = 0) const override;

        const TGpuInputColumnDesc& GetColumnDesc() const noexcept {
            return Column;
        }

        const TQuantizedFeaturesInfoPtr& GetQuantizedFeaturesInfo() const noexcept {
            return QuantizedFeaturesInfo;
        }

    private:
        TGpuInputColumnDesc Column;
        TQuantizedFeaturesInfoPtr QuantizedFeaturesInfo;
    };

    class TGpuExternalCatValuesHolder final : public IQuantizedCatValuesHolder {
    public:
        TGpuExternalCatValuesHolder(
            ui32 featureId,
            ui32 size,
            i32 deviceId,
            const ui32* deviceBinsPtr,
            TAtomicSharedPtr<TSingleBuffer<ui32>> bins,
            TQuantizedFeaturesInfoPtr quantizedFeaturesInfo
        )
            : IQuantizedCatValuesHolder(featureId, size)
            , DeviceId(deviceId)
            , DeviceBinsPtr(deviceBinsPtr)
            , Bins(std::move(bins))
            , QuantizedFeaturesInfo(std::move(quantizedFeaturesInfo))
        {}

        bool IsSparse() const override {
            return false;
        }

        ui64 EstimateMemoryForCloning(const TCloningParams& cloningParams) const override {
            Y_UNUSED(cloningParams);
            return 0;
        }

        ui32 CalcChecksum(NPar::ILocalExecutor* localExecutor) const override;

        THolder<IFeatureValuesHolder> CloneWithNewSubsetIndexing(
            const TCloningParams& cloningParams,
            NPar::ILocalExecutor* localExecutor
        ) const override;

        IDynamicBlockIteratorBasePtr GetBlockIterator(ui32 offset = 0) const override;

        const TSingleBuffer<ui32>& GetBins() const {
            CB_ENSURE_INTERNAL(Bins, "GPU bins buffer is null");
            return *Bins;
        }

        const ui32* GetDeviceBinsPtr() const noexcept {
            return DeviceBinsPtr;
        }

        i32 GetDeviceId() const noexcept {
            return DeviceId;
        }

        const TQuantizedFeaturesInfoPtr& GetQuantizedFeaturesInfo() const noexcept {
            return QuantizedFeaturesInfo;
        }

    private:
        i32 DeviceId = -1;
        const ui32* DeviceBinsPtr = nullptr;
        TAtomicSharedPtr<TSingleBuffer<ui32>> Bins;
        TQuantizedFeaturesInfoPtr QuantizedFeaturesInfo;
    };

}
