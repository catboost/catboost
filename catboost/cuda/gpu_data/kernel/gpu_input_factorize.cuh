#pragma once

#include "gpu_input_utils.cuh"

namespace NKernel {

    // Factorize a strided integer GPU column into per-row ranks and unique values/counts.
    //
    // Supported dtypes: Int8/Int16/Int32/Int64/UInt8/UInt16/UInt32/UInt64.
    //
    // Outputs:
    //  - ranksOut: device array [size], with values in [0, uniqueCount)
    //  - uniqueValuesOut: device array [size] (typed by dtype), first uniqueCount entries are valid unique values
    //  - countsOut: device array [size], first uniqueCount entries are valid counts for corresponding unique values
    //  - uniqueCountOut: device pointer to single ui32 with number of unique values
    //
    // No synchronization is performed.
    void FactorizeStridedGpuInputToUnique(
        const void* src,
        ui64 srcStrideBytes,
        ui32 size,
        EGpuInputDType dtype,
        ui32* ranksOut,
        void* uniqueValuesOut,
        ui32* countsOut,
        ui32* uniqueCountOut,
        TCudaStream stream
    );

    // Compute dstBins[i] = binsForRank[ranks[i]] for i in [0, size).
    // No synchronization is performed.
    void MapRanksToBins(
        const ui32* ranks,
        ui32 size,
        const ui32* binsForRank,
        ui32* dstBins,
        TCudaStream stream
    );

    // Gather and cast perfect-hashed categorical bins to ui8.
    // If gatherIndices is nullptr, performs a simple cast: dst[i] = (ui8)src[i].
    // Otherwise: dst[i] = (ui8)src[gatherIndices[i]].
    // No synchronization is performed.
    void GatherUi32BinsToUi8(
        const ui32* srcBins,
        ui32 size,
        const ui32* gatherIndices,
        ui8* dstBins,
        TCudaStream stream
    );

    // Compute CatBoost categorical hash (CityHash64(string) & 0xffffffff) for unique integer values.
    //
    // The input values are interpreted as integers and converted to their decimal string representation
    // (matching ToString(i64/ui64) semantics), then hashed with CityHash64 (v1) like CalcCatFeatureHash.
    // Supported dtypes: Int8/Int16/Int32/Int64/UInt8/UInt16/UInt32/UInt64.
    //
    // No synchronization is performed.
    void HashUniqueNumericToCatHash(
        const void* uniqueValues,
        ui32 uniqueCount,
        EGpuInputDType dtype,
        ui32* hashesOut,
        TCudaStream stream
    );

}
