#pragma once

#include <catboost/cuda/cuda_lib/fwd.h>

template <class TMapping, class TStatus>
void NonZeroFilter(
    const NCudaLib::TCudaBuffer<float, TMapping>& weights,
    NCudaLib::TCudaBuffer<TStatus, TMapping>& status,
    ui32 stream = 0);
