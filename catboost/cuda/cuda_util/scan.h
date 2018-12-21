#pragma once

#include <catboost/cuda/cuda_lib/fwd.h>

template <typename T, class TMapping>
void ScanVector(
    const NCudaLib::TCudaBuffer<T, TMapping>& input,
    NCudaLib::TCudaBuffer<std::remove_const_t<T>, TMapping>& output,
    bool inclusive = false,
    ui32 streamId = 0);

//TODO(noxoomo): we should be able to run exclusive also
template <typename T, class TMapping>
void InclusiveSegmentedScanNonNegativeVector(
    const NCudaLib::TCudaBuffer<T, TMapping>& input,
    NCudaLib::TCudaBuffer<std::remove_const_t<T>, TMapping>& output,
    ui32 streamId = 0);

//Not the safest way…
template <typename T, class TMapping, class TUi32 = ui32>
void SegmentedScanAndScatterNonNegativeVector(
    const NCudaLib::TCudaBuffer<T, TMapping>& inputWithSignMasks,
    const NCudaLib::TCudaBuffer<TUi32, TMapping>& indices,
    NCudaLib::TCudaBuffer<std::remove_const_t<T>, TMapping>& output,
    bool inclusive = false,
    ui32 streamId = 0);
