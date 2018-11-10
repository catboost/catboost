#pragma once

#include <catboost/cuda/cuda_lib/fwd.h>

template <class TMapping>
void NonZeroFilter(
    const NCudaLib::TCudaBuffer<float, TMapping>& weights,
    NCudaLib::TCudaBuffer<ui32, TMapping>& status,
    ui32 stream = 0);
