#pragma once

#include <catboost/cuda/cuda_lib/fwd.h>

template <typename K, class TMapping>
void RadixSort(
    NCudaLib::TCudaBuffer<K, TMapping>& keys,
    bool compareGreater = false,
    ui32 stream = 0);

template <typename K, typename V, class TMapping>
void RadixSort(
    NCudaLib::TCudaBuffer<K, TMapping>& keys,
    NCudaLib::TCudaBuffer<V, TMapping>& values,
    bool compareGreater = false,
    ui32 stream = 0);

template <typename K, typename V, class TMapping>
void RadixSort(
    NCudaLib::TCudaBuffer<K, TMapping>& keys, NCudaLib::TCudaBuffer<V, TMapping>& values,
    NCudaLib::TCudaBuffer<K, TMapping>& tmpKeys, NCudaLib::TCudaBuffer<V, TMapping>& tmpValues,
    ui32 offset = 0,
    ui32 bits = sizeof(K) * 8,
    ui64 stream = 0);

template <typename K, typename V, class TMapping>
void RadixSort(
    NCudaLib::TCudaBuffer<K, TMapping>& keys,
    NCudaLib::TCudaBuffer<V, TMapping>& values,
    bool compareGreater,
    ui32 offset,
    ui32 bits,
    ui32 stream = 0);
