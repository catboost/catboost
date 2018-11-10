#pragma once

#include <catboost/cuda/cuda_lib/fwd.h>

template <typename T1, class T2, typename TMapping, class T3 = T1>
float DotProduct(
    const NCudaLib::TCudaBuffer<T1, TMapping>& x,
    const NCudaLib::TCudaBuffer<T2, TMapping>& y,
    const NCudaLib::TCudaBuffer<T3, TMapping>* weights = nullptr,
    ui64 stream = 0);
