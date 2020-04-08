#pragma once

#include <catboost/cuda/cuda_lib/fwd.h>
#include <catboost/cuda/cuda_lib/cuda_base.h>

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
void MultiplyAddVector(
    NCudaLib::TCudaBuffer<std::remove_const_t<T>, TMapping>& x,
    NCudaLib::TCudaBuffer<std::remove_const_t<T>, TMapping>& y,
    T value,
    ui32 stream = 0
) {
    MultiplyVector(y, value, stream);
    AddVector(x, y, stream);
}

template <typename T, class TMapping>
void DivideVector(
    NCudaLib::TCudaBuffer<std::remove_const_t<T>, TMapping>& x,
    const NCudaLib::TCudaBuffer<T, TMapping>& y,
    ui32 stream = 0);

// [x_1, x_2, ..., x_n] -> [base^x_1, base^x_2, ..., base^x_n]
template <typename T, typename TMapping>
void PowVector(
    NCudaLib::TCudaBuffer<T, TMapping>& src,
    T base,
    ui32 stream = 0);

// [x_1, x_2, ..., x_n] -> [base^x_1, base^x_2, ..., base^x_n]
template <typename T, typename U, typename TMapping>
void PowVector(
    const NCudaLib::TCudaBuffer<T, TMapping>& src,
    std::remove_const_t<T> base,
    NCudaLib::TCudaBuffer<U, TMapping>& dst,
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
