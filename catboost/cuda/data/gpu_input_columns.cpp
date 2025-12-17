#include "gpu_input_columns.h"

#include <catboost/cuda/cuda_lib/cuda_base.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/libs/helpers/dynamic_iterator.h>
#include <catboost/libs/helpers/checksum.h>
#include <catboost/libs/helpers/exception.h>

#include <algorithm>

namespace NCB {

    ui32 TGpuExternalFloatValuesHolder::CalcChecksum(NPar::ILocalExecutor* localExecutor) const {
        Y_UNUSED(localExecutor);

        ui32 checkSum = 0;
        checkSum = UpdateCheckSum(checkSum, GetId());
        checkSum = UpdateCheckSum(checkSum, GetSize());
        checkSum = UpdateCheckSum(checkSum, Column.StrideBytes);
        checkSum = UpdateCheckSum(checkSum, Column.DeviceId);
        checkSum = UpdateCheckSum(checkSum, static_cast<ui8>(Column.DType));

        // Note: avoid using CatBoost CUDA manager here (it may be uninitialized at checksum time).
        // Instead, sample a few values directly via CUDA runtime. This keeps D2H traffic bounded.
        if ((GetSize() == 0) || (Column.Data == 0)) {
            return checkSum;
        }

        CB_ENSURE(Column.DeviceId >= 0, "Invalid device id for GPU input pointer");

        NCudaLib::SetDevice(Column.DeviceId);

        auto getItemSize = [] (EGpuInputDType dtype) -> ui32 {
            switch (dtype) {
                case EGpuInputDType::Float32:
                case EGpuInputDType::Int32:
                case EGpuInputDType::UInt32:
                    return 4;
                case EGpuInputDType::Float64:
                case EGpuInputDType::Int64:
                case EGpuInputDType::UInt64:
                    return 8;
                case EGpuInputDType::Int16:
                case EGpuInputDType::UInt16:
                    return 2;
                case EGpuInputDType::Int8:
                case EGpuInputDType::UInt8:
                case EGpuInputDType::Bool:
                    return 1;
            }
            return 0;
        };

        const ui32 itemSize = getItemSize(Column.DType);
        CB_ENSURE(itemSize > 0, "Unsupported GPU input dtype for checksum");

        const ui32 last = GetSize() - 1;
        const ui32 mid = GetSize() / 2;
        const ui32 q1 = GetSize() / 4;
        const ui32 q3 = (GetSize() * 3) / 4;
        const ui32 sampleIdxs[5] = {0, q1, mid, q3, last};

        for (const ui32 i : sampleIdxs) {
            ui64 bits = 0;
            const auto offset = static_cast<ui64>(i) * Column.StrideBytes;
            const auto* src = reinterpret_cast<const void*>(static_cast<uintptr_t>(Column.Data + offset));

            NCudaLib::TMemcpyTracker::Instance().RecordMemcpyAsync(
                /*dst*/ &bits,
                /*src*/ src,
                itemSize,
                cudaMemcpyDeviceToHost
            );
            CUDA_SAFE_CALL(cudaMemcpy(&bits, src, itemSize, cudaMemcpyDeviceToHost));
            checkSum = UpdateCheckSum(checkSum, bits);
        }

        return checkSum;
    }

    THolder<IFeatureValuesHolder> TGpuExternalFloatValuesHolder::CloneWithNewSubsetIndexing(
        const TCloningParams& cloningParams,
        NPar::ILocalExecutor* localExecutor
    ) const {
        Y_UNUSED(localExecutor);
        CB_ENSURE_INTERNAL(
            !cloningParams.MakeConsecutive,
            "Consecutive cloning of TGpuExternalFloatValuesHolder is not supported"
        );
        if (cloningParams.SubsetIndexing && !cloningParams.SubsetIndexing->IsFullSubset()) {
            CB_ENSURE_INTERNAL(
                false,
                "Cloning TGpuExternalFloatValuesHolder with non-full subset indexing is not supported yet"
            );
        }
        return MakeHolder<TGpuExternalFloatValuesHolder>(
            GetId(),
            GetSize(),
            Column,
            QuantizedFeaturesInfo
        );
    }

    IDynamicBlockIteratorBasePtr TGpuExternalFloatValuesHolder::GetBlockIterator(ui32 offset) const {
        Y_UNUSED(offset);
        CB_ENSURE_INTERNAL(
            false,
            "TGpuExternalFloatValuesHolder does not support CPU block iteration"
        );
    }

    ui32 TGpuExternalCatValuesHolder::CalcChecksum(NPar::ILocalExecutor* localExecutor) const {
        Y_UNUSED(localExecutor);

        ui32 checkSum = 0;
        checkSum = UpdateCheckSum(checkSum, GetId());
        checkSum = UpdateCheckSum(checkSum, GetSize());
        checkSum = UpdateCheckSum(checkSum, DeviceId);

        if (GetSize() == 0) {
            return checkSum;
        }
        CB_ENSURE(DeviceId >= 0, "Invalid device id for GPU bins buffer");

        NCudaLib::SetDevice(DeviceId);

        const ui32* src = DeviceBinsPtr;
        CB_ENSURE(src != nullptr, "GPU bins pointer is null");

        const ui32 last = GetSize() - 1;
        const ui32 mid = GetSize() / 2;
        const ui32 q1 = GetSize() / 4;
        const ui32 q3 = (GetSize() * 3) / 4;
        const ui32 sampleIdxs[5] = {0, q1, mid, q3, last};

        for (const ui32 i : sampleIdxs) {
            ui32 v = 0;
            const void* srcPtr = src + i;
            NCudaLib::TMemcpyTracker::Instance().RecordMemcpyAsync(
                /*dst*/ &v,
                /*src*/ srcPtr,
                sizeof(ui32),
                cudaMemcpyDeviceToHost
            );
            CUDA_SAFE_CALL(cudaMemcpy(&v, srcPtr, sizeof(ui32), cudaMemcpyDeviceToHost));
            checkSum = UpdateCheckSum(checkSum, v);
        }
        return checkSum;
    }

    THolder<IFeatureValuesHolder> TGpuExternalCatValuesHolder::CloneWithNewSubsetIndexing(
        const TCloningParams& cloningParams,
        NPar::ILocalExecutor* localExecutor
    ) const {
        Y_UNUSED(localExecutor);
        CB_ENSURE_INTERNAL(
            !cloningParams.MakeConsecutive,
            "Consecutive cloning of TGpuExternalCatValuesHolder is not supported"
        );
        if (cloningParams.SubsetIndexing && !cloningParams.SubsetIndexing->IsFullSubset()) {
            CB_ENSURE_INTERNAL(
                false,
                "Cloning TGpuExternalCatValuesHolder with non-full subset indexing is not supported yet"
            );
        }
        return MakeHolder<TGpuExternalCatValuesHolder>(
            GetId(),
            GetSize(),
            DeviceId,
            DeviceBinsPtr,
            Bins,
            QuantizedFeaturesInfo
        );
    }

    IDynamicBlockIteratorBasePtr TGpuExternalCatValuesHolder::GetBlockIterator(ui32 offset) const {
        class TGpuBinsBlockIterator final : public IDynamicBlockIterator<ui32> {
        public:
            TGpuBinsBlockIterator(const ui32* src, i32 deviceId, size_t size, ui32 offset)
                : DeviceId(deviceId)
                , Src(src)
                , Size(size)
                , Pos(offset)
            {
                CB_ENSURE(DeviceId >= 0, "Invalid device id for GPU bins iterator");
                NCudaLib::SetDevice(DeviceId);
                CB_ENSURE(Src != nullptr, "GPU bins pointer is null");
            }

            TConstArrayRef<ui32> Next(size_t maxBlockSize) override {
                if (Pos >= Size) {
                    Buffer.clear();
                    return {};
                }
                const size_t remaining = Size - Pos;
                const size_t blockSize = std::min(maxBlockSize, remaining);
                Buffer.yresize(blockSize);

                const void* srcPtr = Src + Pos;
                NCudaLib::TMemcpyTracker::Instance().RecordMemcpyAsync(
                    /*dst*/ Buffer.data(),
                    /*src*/ srcPtr,
                    blockSize * sizeof(ui32),
                    cudaMemcpyDeviceToHost
                );
                CUDA_SAFE_CALL(cudaMemcpy(Buffer.data(), srcPtr, blockSize * sizeof(ui32), cudaMemcpyDeviceToHost));

                Pos += blockSize;
                return Buffer;
            }

        private:
            i32 DeviceId = -1;
            const ui32* Src = nullptr;
            size_t Size = 0;
            size_t Pos = 0;
            TVector<ui32> Buffer;
        };

        CB_ENSURE(offset <= GetSize(), "Block iterator offset is out of bounds");
        CB_ENSURE(DeviceBinsPtr != nullptr, "GPU bins pointer is null");
        return MakeHolder<TGpuBinsBlockIterator>(DeviceBinsPtr, DeviceId, static_cast<size_t>(GetSize()), offset);
    }

}
