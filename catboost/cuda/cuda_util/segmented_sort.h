#pragma once

#include <catboost/cuda/cuda_lib/fwd.h>

template <typename K, typename V, class TMapping>
void SegmentedRadixSort(
    NCudaLib::TCudaBuffer<K, TMapping>& keys,
    NCudaLib::TCudaBuffer<V, TMapping>& values,
    NCudaLib::TCudaBuffer<K, TMapping>& tmpKeys,
    NCudaLib::TCudaBuffer<V, TMapping>& tmpValues,
    const NCudaLib::TCudaBuffer<ui32, TMapping>& offsets,
    ui32 partCount,
    ui32 fistBit = 0,
    ui32 lastBit = sizeof(K) * 8,
    bool compareGreater = false,
    ui64 stream = 0);

template <typename K, typename V, class TMapping>
void SegmentedRadixSort(
    NCudaLib::TCudaBuffer<K, TMapping>& keys,
    NCudaLib::TCudaBuffer<V, TMapping>& values,
    NCudaLib::TCudaBuffer<K, TMapping>& tmpKeys,
    NCudaLib::TCudaBuffer<V, TMapping>& tmpValues,
    const NCudaLib::TCudaBuffer<ui32, TMapping>& segmentStarts,
    const NCudaLib::TCudaBuffer<ui32, TMapping>& segmentEnds,
    ui32 partCount,
    ui32 fistBit = 0,
    ui32 lastBit = sizeof(K) * 8,
    bool compareGreater = false,
    ui64 stream = 0);
