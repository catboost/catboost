#pragma once
#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>

namespace NKernel {

    struct TRadixSortContext : public IKernelContext {

        TRadixSortContext(ui32 firstBit, ui32 lastBit, ui32 valueSize, bool descending) {
            FirstBit = firstBit;
            LastBit = lastBit;
            ValueSize = valueSize;
            Descending = descending;
        }

        ui32 FirstBit = 0;
        ui32 LastBit = 0;
        ui32 ValueSize = 0;

        ui64 TempStorageSize = 0;

        TDevicePointer<char> TempStorage;
        TDevicePointer<char> TempKeys;
        TDevicePointer<char> TempValues;

        bool Descending = false;
        bool UseExternalBufferForTempKeysAndValues = false;

        template <class T>
        inline T* GetTempKeys() {
            return reinterpret_cast<T*>(TempKeys.Get());
        }


        template <class T>
        inline T* GetTempValues() {
            return reinterpret_cast<T*>(TempValues.Get());
        }


        TRadixSortContext() = default;
    };

    template <typename K, typename V>
    cudaError_t RadixSort(K *keys, V *values, ui32 size, TRadixSortContext& context, TCudaStream stream);
}
