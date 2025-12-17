#pragma once

#include "gpu_input_utils.cuh"

#include <util/system/types.h>

namespace NKernel {

    // hashes[i] = CalcHash(hashes[i], (ui64)(i32)binToHash[bins[i]])
    // No synchronization is performed.
    void UpdateHashesFromCatFeature(
        const ui32* bins,
        ui32 size,
        const ui32* binToHash,
        ui64* hashes,
        TCudaStream stream
    );

    // hashes[i] = CalcHash(hashes[i], (ui64)(values[i] > borderValue))
    // No synchronization is performed.
    void UpdateHashesFromFloatSplit(
        const float* values,
        ui32 size,
        float borderValue,
        ui64* hashes,
        TCudaStream stream
    );

    // hashes[i] = CalcHash(hashes[i], (ui64)(bins[i] == value))
    // No synchronization is performed.
    void UpdateHashesFromOneHotFeature(
        const ui32* bins,
        ui32 size,
        ui32 value,
        ui64* hashes,
        TCudaStream stream
    );

    // hashes[i] = CalcHash(hashes[i], (ui64)(binarizedBins[i] >= threshold))
    // No synchronization is performed.
    void UpdateHashesFromBinarizedSplit(
        const ui8* binarizedBins,
        ui32 size,
        ui8 threshold,
        ui64* hashes,
        TCudaStream stream
    );

    // dst[i] = remap[src[i]]
    // No synchronization is performed.
    void RemapIndices(
        const ui32* src,
        ui32 size,
        const ui32* remap,
        ui32* dst,
        TCudaStream stream
    );

    // sums[indices[i]] += values[i]
    // No synchronization is performed.
    void AccumulateFloatByIndex(
        const ui32* indices,
        const float* values,
        ui32 size,
        float* sums,
        TCudaStream stream
    );

    // sums[indices[i]] += targetClass[i] * invTargetBorderCount
    // No synchronization is performed.
    void AccumulateBinarizedTargetSumByIndex(
        const ui32* indices,
        const ui32* targetClass,
        ui32 size,
        float invTargetBorderCount,
        float* sums,
        TCudaStream stream
    );

    // counts[indices[i] * classCount + targetClass[i]] += 1
    // No synchronization is performed.
    void AccumulateClassCountsByIndex(
        const ui32* indices,
        const ui32* targetClass,
        ui32 size,
        ui32 classCount,
        ui32* counts,
        TCudaStream stream
    );

}
