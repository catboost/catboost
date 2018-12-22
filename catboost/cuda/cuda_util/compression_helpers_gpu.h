#pragma once

#include <catboost/cuda/cuda_lib/fwd.h>

#include <util/system/types.h>

template <typename TStorageType>
ui32 CompressedSize(ui32 count, ui32 uniqueValues);

template <typename TStorageType, typename TMapping>
TMapping CompressedSize(const NCudaLib::TCudaBuffer<ui32, TMapping>& src, ui32 uniqueValues);

template <typename T, typename TMapping, NCudaLib::EPtrType Type>
void Compress(
    const NCudaLib::TCudaBuffer<ui32, TMapping>& src,
    NCudaLib::TCudaBuffer<T, TMapping, Type>& dst,
    ui32 uniqueValues,
    ui32 stream = 0);

template <typename T, typename TMapping, NCudaLib::EPtrType Type>
void Decompress(
    const NCudaLib::TCudaBuffer<T, TMapping, Type>& src,
    NCudaLib::TCudaBuffer<ui32, TMapping>& dst,
    ui32 uniqueValues,
    ui32 stream = 0);

template <typename T, typename TMapping, NCudaLib::EPtrType Type, class TUi32 = ui32>
void GatherFromCompressed(
    const NCudaLib::TCudaBuffer<T, TMapping, Type>& src,
    const ui32 uniqueValues,
    const NCudaLib::TCudaBuffer<TUi32, TMapping>& map,
    const ui32 mask,
    NCudaLib::TCudaBuffer<ui32, TMapping>& dst,
    ui32 stream = 0);
