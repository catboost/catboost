#pragma once
#include "scan.cuh"

namespace NKernel {

    template <class K, class T>
    struct TReorderOneBitContext : public IKernelContext {
        TDevicePointer<i32> Offsets;

        TDevicePointer<K> TempKeys;
        TDevicePointer<T> TempValues;

        TDevicePointer<char> ScanTempBuffer;
        ui64 ScanTempBufferSize =0;
    };

    ui64 ReorderBitTempSize(ui32 size);

    template <class T>
    void ReorderOneBit(ui32 size, TReorderOneBitContext<ui32, T> context, ui32* keys, T* values, int bit, TCudaStream stream);

}
