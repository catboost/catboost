#pragma once

#include "operator.h"

#include <catboost/cuda/cuda_lib/fwd.h>

#include <type_traits>

template <typename T, class TMapping>
void ReduceVector(
    const NCudaLib::TCudaBuffer<T, TMapping>& input,
    NCudaLib::TCudaBuffer<std::remove_const_t<T>, TMapping>& output,
    EOperatorType type = EOperatorType::Sum,
    ui32 streamId = 0);

template <typename T, typename K, class TMapping>
void ReduceByKeyVector(
    const NCudaLib::TCudaBuffer<T, TMapping>& input,
    const NCudaLib::TCudaBuffer<K, TMapping>& keys,
    NCudaLib::TCudaBuffer<std::remove_const_t<K>, TMapping>& outputKeys,
    NCudaLib::TCudaBuffer<std::remove_const_t<T>, TMapping>& output,
    NCudaLib::TCudaBuffer<ui32, TMapping>& outputSizes,
    EOperatorType type = EOperatorType::Sum,
    ui32 streamId = 0);

template <typename T, class TMapping, NCudaLib::EPtrType OutputPtrType = NCudaLib::EPtrType::CudaDevice>
void SegmentedReduceVector(
    const NCudaLib::TCudaBuffer<T, TMapping>& input,
    const NCudaLib::TCudaBuffer<ui32, TMapping>& offsets,
    NCudaLib::TCudaBuffer<std::remove_const_t<T>, TMapping, OutputPtrType>& output,
    EOperatorType type = EOperatorType::Sum,
    ui32 streamId = 0);

template <typename T, class TMapping>
std::remove_const_t<T> ReduceToHost(
    const NCudaLib::TCudaBuffer<T, TMapping>& input,
    EOperatorType type = EOperatorType::Sum,
    ui32 streamId = 0);
