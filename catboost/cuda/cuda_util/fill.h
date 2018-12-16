#pragma once

#include <catboost/cuda/cuda_lib/fwd.h>

#include <type_traits>

template <typename T, class TMapping>
void FillBuffer(
    NCudaLib::TCudaBuffer<T, TMapping>& buffer,
    std::remove_const_t<T> value,
    ui32 streamId = 0);

template <typename T, class TMapping>
void MakeSequence(NCudaLib::TCudaBuffer<T, TMapping>& buffer, ui32 stream = 0);

template <typename T, class TMapping>
void MakeSequenceWithOffset(
    NCudaLib::TCudaBuffer<T, TMapping>& buffer,
    const NCudaLib::TDistributedObject<T>& offset,
    ui32 stream = 0);

template <typename T>
void MakeSequenceGlobal(
    NCudaLib::TCudaBuffer<T, NCudaLib::TStripeMapping>& buffer,
    ui32 stream = 0);

template <class TUi32, class TMapping>
void InversePermutation(
    const NCudaLib::TCudaBuffer<TUi32, TMapping>& order,
    NCudaLib::TCudaBuffer<ui32, TMapping>& inverseOrder,
    ui32 streamId = 0);
