#pragma once

#include <catboost/cuda/cuda_lib/fwd.h>

#include <type_traits>

template <typename T, class TMapping, class TUi32>
void SegmentedScanVector(
    const NCudaLib::TCudaBuffer<T, TMapping>& input,
    const NCudaLib::TCudaBuffer<TUi32, TMapping>& flags,
    NCudaLib::TCudaBuffer<std::remove_const_t<T>, TMapping>& output,
    bool inclusive = false,
    ui32 flagMask = 1,
    ui32 streamId = 0);
