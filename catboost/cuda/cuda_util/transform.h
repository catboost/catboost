#pragma once

#include <catboost/cuda/cuda_lib/fwd.h>

#include <type_traits>

namespace NKernelHost {
    enum class EBinOpType {
        AddVec,
        AddConst,
        SubVec,
        MulVec,
        MulConst,
        DivVec
    };

    enum class EFuncType {
        Exp,
        Identity
    };

    enum class EMapCopyType {
        Gather,
        Scatter
    };
}

template <typename T, class TMapping>
void AddVector(
    NCudaLib::TCudaBuffer<std::remove_const_t<T>, TMapping>& x,
    const NCudaLib::TCudaBuffer<T, TMapping>& y,
    ui32 stream = 0);

template <typename T, class TMapping>
void AddVector(
    NCudaLib::TCudaBuffer<std::remove_const_t<T>, TMapping>& x,
    T value,
    ui32 stream = 0);

template <typename T, class TMapping>
void SubtractVector(
    NCudaLib::TCudaBuffer<std::remove_const_t<T>, TMapping>& x,
    const NCudaLib::TCudaBuffer<T, TMapping>& y,
    ui32 stream = 0);

template <typename T, class TMapping>
void MultiplyVector(
    NCudaLib::TCudaBuffer<std::remove_const_t<T>, TMapping>& x,
    const NCudaLib::TCudaBuffer<T, TMapping>& y,
    ui32 stream = 0);

template <typename T, class TMapping>
void MultiplyVector(
    NCudaLib::TCudaBuffer<std::remove_const_t<T>, TMapping>& x,
    T y,
    ui32 stream = 0);

template <typename T, class TMapping>
void DivideVector(
    NCudaLib::TCudaBuffer<std::remove_const_t<T>, TMapping>& x,
    const NCudaLib::TCudaBuffer<T, TMapping>& y,
    ui32 stream = 0);

template <typename T, class TMapping>
void ExpVector(
    NCudaLib::TCudaBuffer<T, TMapping>& x,
    ui32 stream = 0);

template <typename T, class TMapping, class U = const ui32>
void Gather(
    NCudaLib::TCudaBuffer<std::remove_const_t<T>, TMapping>& dst,
    const NCudaLib::TCudaBuffer<T, TMapping>& src,
    const NCudaLib::TCudaBuffer<U, TMapping>& map,
    ui32 stream = 0);

template <typename T, class U = const ui32>
void Gather(
    NCudaLib::TCudaBuffer<std::remove_const_t<T>, NCudaLib::TStripeMapping>& dst,
    const NCudaLib::TCudaBuffer<T, NCudaLib::TMirrorMapping>& src,
    const NCudaLib::TCudaBuffer<U, NCudaLib::TStripeMapping>& map,
    ui32 stream = 0);

template <typename T, class TMapping, class U = const ui32>
void GatherWithMask(
    NCudaLib::TCudaBuffer<std::remove_const_t<T>, TMapping>& dst,
    const NCudaLib::TCudaBuffer<T, TMapping>& src,
    const NCudaLib::TCudaBuffer<U, TMapping>& map,
    ui32 mask,
    ui32 stream = 0);

template <typename T, class TMapping, class U = const ui32>
void Scatter(
    NCudaLib::TCudaBuffer<std::remove_const_t<T>, TMapping>& dst,
    const NCudaLib::TCudaBuffer<T, TMapping>& src,
    const NCudaLib::TCudaBuffer<U, TMapping>& map,
    ui32 stream = 0);

template <typename T, class TMapping, class U = const ui32>
void ScatterWithMask(
    NCudaLib::TCudaBuffer<std::remove_const_t<T>, TMapping>& dst,
    const NCudaLib::TCudaBuffer<T, TMapping>& src,
    const NCudaLib::TCudaBuffer<U, TMapping>& map,
    ui32 mask,
    ui32 stream = 0);

template <typename T, class TMapping>
void Reverse(
    NCudaLib::TCudaBuffer<T, TMapping>& data,
    ui32 stream = 0);
