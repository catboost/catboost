#pragma once

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>

namespace NKernel {

    enum class EGpuInputDType : ui8 {
        Float32 = 0,
        Float64 = 1,
        Int8 = 2,
        Int16 = 3,
        Int32 = 4,
        Int64 = 5,
        UInt8 = 6,
        UInt16 = 7,
        UInt32 = 8,
        UInt64 = 9,
        Bool = 10,
    };

    // Copy a strided device column (one value per row) into a contiguous float buffer on device,
    // casting element type as needed. No synchronization is performed.
    void CopyStridedGpuInputToFloat(
        const void* src,
        ui64 srcStrideBytes,
        ui32 size,
        EGpuInputDType dtype,
        float* dst,
        TCudaStream stream
    );

    // Compute CatBoost categorical hash for a strided integer GPU column:
    //   dst[i] = CityHash64(ToString(value[i])) & 0xffffffff
    // (matching CalcCatFeatureHash(ToString(value)) for integer categorical values).
    //
    // Supported dtypes: Int8/Int16/Int32/Int64/UInt8/UInt16/UInt32/UInt64.
    // No synchronization is performed.
    void HashStridedGpuInputToCatHash(
        const void* src,
        ui64 srcStrideBytes,
        ui32 size,
        EGpuInputDType dtype,
        ui32* dst,
        TCudaStream stream
    );

    // Map a strided integer categorical codes column to CatBoost hashed categorical values using a device dictionary:
    //   if code in [0, dictSize) => dst[i] = dict[code]
    //   else => dst[i] = nullValue
    //
    // Supported dtypes: Int8/Int16/Int32/Int64/UInt8/UInt16/UInt32/UInt64.
    // No synchronization is performed.
    void MapStridedCatCodesToCatHash(
        const void* src,
        ui64 srcStrideBytes,
        ui32 size,
        EGpuInputDType dtype,
        const ui32* dict,
        ui32 dictSize,
        ui32 nullValue,
        ui32* dst,
        TCudaStream stream
    );

    // Compute min/max for a float device array and copy results to host. Synchronizes the stream.
    void ComputeMinMaxToHost(
        const float* values,
        ui32 size,
        float* minValue,
        float* maxValue,
        TCudaStream stream
    );

}
