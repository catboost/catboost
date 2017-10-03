#pragma once

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>


namespace NKernel {

    constexpr ui32 CompressCudaBlockSize() {
        return 128;
    }

    template <class TStorageType>
    inline ui32 KeysPerBlock(ui32 bitsPerKey) {
        ui32 keysPerStorage = 8 * sizeof(TStorageType) / bitsPerKey;
        return keysPerStorage * CompressCudaBlockSize();
    }

    template <class TStorageType>
    void Decompress(const TStorageType* src, ui32* dst,  ui32 size, ui32 bitsPerKey, TCudaStream stream);

    template <class TStorageType>
    void Compress(const ui32* src, TStorageType* dst, ui32 size, ui32 bitsPerKey, TCudaStream stream);

    template <class TStorageType>
    void GatherFromCompressed(const TStorageType* src, const ui32* map, ui32 mapMask,
                              ui32* dst, ui32 size, ui32 bitsPerKey, TCudaStream stream);

}
