//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023-24 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TUPLE_STRUCTURED_BINDINGS_H
#define _LIBCUDACXX___TUPLE_STRUCTURED_BINDINGS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wmismatched-tags")

#if !_CCCL_COMPILER(NVRTC)
// Fetch utility to get primary template for ::std::tuple_size necessary for the specialization of
// ::std::tuple_size<cuda::std::tuple> to enable structured bindings.
// See https://github.com/NVIDIA/libcudacxx/issues/316
#  include <utility>
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__fwd/array.h>
#include <cuda/std/__fwd/complex.h>
#include <cuda/std/__fwd/pair.h>
#include <cuda/std/__fwd/subrange.h>
#include <cuda/std/__fwd/tuple.h>
#include <cuda/std/__tuple_dir/tuple_element.h>
#include <cuda/std/__tuple_dir/tuple_size.h>
#include <cuda/std/__type_traits/integral_constant.h>

// This is a workaround for the fact that structured bindings require that the specializations of
// `tuple_size` and `tuple_element` reside in namespace std (https://eel.is/c++draft/dcl.struct.bind#4).
// See https://github.com/NVIDIA/libcudacxx/issues/316 for a short discussion

#include <cuda/std/__cccl/prologue.h>

namespace std
{
#if _CCCL_COMPILER(NVRTC)
template <class... _Tp>
struct tuple_size;

template <size_t _Ip, class... _Tp>
struct tuple_element;
#endif // _CCCL_COMPILER(NVRTC)

template <class _Tp, size_t _Size>
struct tuple_size<_CUDA_VSTD::array<_Tp, _Size>> : _CUDA_VSTD::tuple_size<_CUDA_VSTD::array<_Tp, _Size>>
{};

template <class _Tp, size_t _Size>
struct tuple_size<const _CUDA_VSTD::array<_Tp, _Size>> : _CUDA_VSTD::tuple_size<_CUDA_VSTD::array<_Tp, _Size>>
{};

template <class _Tp, size_t _Size>
struct tuple_size<volatile _CUDA_VSTD::array<_Tp, _Size>> : _CUDA_VSTD::tuple_size<_CUDA_VSTD::array<_Tp, _Size>>
{};

template <class _Tp, size_t _Size>
struct tuple_size<const volatile _CUDA_VSTD::array<_Tp, _Size>> : _CUDA_VSTD::tuple_size<_CUDA_VSTD::array<_Tp, _Size>>
{};

template <size_t _Ip, class _Tp, size_t _Size>
struct tuple_element<_Ip, _CUDA_VSTD::array<_Tp, _Size>> : _CUDA_VSTD::tuple_element<_Ip, _CUDA_VSTD::array<_Tp, _Size>>
{};

template <size_t _Ip, class _Tp, size_t _Size>
struct tuple_element<_Ip, const _CUDA_VSTD::array<_Tp, _Size>>
    : _CUDA_VSTD::tuple_element<_Ip, const _CUDA_VSTD::array<_Tp, _Size>>
{};

template <size_t _Ip, class _Tp, size_t _Size>
struct tuple_element<_Ip, volatile _CUDA_VSTD::array<_Tp, _Size>>
    : _CUDA_VSTD::tuple_element<_Ip, volatile _CUDA_VSTD::array<_Tp, _Size>>
{};

template <size_t _Ip, class _Tp, size_t _Size>
struct tuple_element<_Ip, const volatile _CUDA_VSTD::array<_Tp, _Size>>
    : _CUDA_VSTD::tuple_element<_Ip, const volatile _CUDA_VSTD::array<_Tp, _Size>>
{};

template <class _Tp>
struct tuple_size<_CUDA_VSTD::complex<_Tp>> : _CUDA_VSTD::tuple_size<_CUDA_VSTD::complex<_Tp>>
{};

template <size_t _Ip, class _Tp>
struct tuple_element<_Ip, _CUDA_VSTD::complex<_Tp>> : _CUDA_VSTD::tuple_element<_Ip, _CUDA_VSTD::complex<_Tp>>
{};

template <class _Tp, class _Up>
struct tuple_size<_CUDA_VSTD::pair<_Tp, _Up>> : _CUDA_VSTD::tuple_size<_CUDA_VSTD::pair<_Tp, _Up>>
{};

template <class _Tp, class _Up>
struct tuple_size<const _CUDA_VSTD::pair<_Tp, _Up>> : _CUDA_VSTD::tuple_size<_CUDA_VSTD::pair<_Tp, _Up>>
{};

template <class _Tp, class _Up>
struct tuple_size<volatile _CUDA_VSTD::pair<_Tp, _Up>> : _CUDA_VSTD::tuple_size<_CUDA_VSTD::pair<_Tp, _Up>>
{};

template <class _Tp, class _Up>
struct tuple_size<const volatile _CUDA_VSTD::pair<_Tp, _Up>> : _CUDA_VSTD::tuple_size<_CUDA_VSTD::pair<_Tp, _Up>>
{};

template <size_t _Ip, class _Tp, class _Up>
struct tuple_element<_Ip, _CUDA_VSTD::pair<_Tp, _Up>> : _CUDA_VSTD::tuple_element<_Ip, _CUDA_VSTD::pair<_Tp, _Up>>
{};

template <size_t _Ip, class _Tp, class _Up>
struct tuple_element<_Ip, const _CUDA_VSTD::pair<_Tp, _Up>>
    : _CUDA_VSTD::tuple_element<_Ip, const _CUDA_VSTD::pair<_Tp, _Up>>
{};

template <size_t _Ip, class _Tp, class _Up>
struct tuple_element<_Ip, volatile _CUDA_VSTD::pair<_Tp, _Up>>
    : _CUDA_VSTD::tuple_element<_Ip, volatile _CUDA_VSTD::pair<_Tp, _Up>>
{};

template <size_t _Ip, class _Tp, class _Up>
struct tuple_element<_Ip, const volatile _CUDA_VSTD::pair<_Tp, _Up>>
    : _CUDA_VSTD::tuple_element<_Ip, const volatile _CUDA_VSTD::pair<_Tp, _Up>>
{};

template <class... _Tp>
struct tuple_size<_CUDA_VSTD::tuple<_Tp...>> : _CUDA_VSTD::tuple_size<_CUDA_VSTD::tuple<_Tp...>>
{};

template <class... _Tp>
struct tuple_size<const _CUDA_VSTD::tuple<_Tp...>> : _CUDA_VSTD::tuple_size<_CUDA_VSTD::tuple<_Tp...>>
{};

template <class... _Tp>
struct tuple_size<volatile _CUDA_VSTD::tuple<_Tp...>> : _CUDA_VSTD::tuple_size<_CUDA_VSTD::tuple<_Tp...>>
{};

template <class... _Tp>
struct tuple_size<const volatile _CUDA_VSTD::tuple<_Tp...>> : _CUDA_VSTD::tuple_size<_CUDA_VSTD::tuple<_Tp...>>
{};

template <size_t _Ip, class... _Tp>
struct tuple_element<_Ip, _CUDA_VSTD::tuple<_Tp...>> : _CUDA_VSTD::tuple_element<_Ip, _CUDA_VSTD::tuple<_Tp...>>
{};

template <size_t _Ip, class... _Tp>
struct tuple_element<_Ip, const _CUDA_VSTD::tuple<_Tp...>>
    : _CUDA_VSTD::tuple_element<_Ip, const _CUDA_VSTD::tuple<_Tp...>>
{};

template <size_t _Ip, class... _Tp>
struct tuple_element<_Ip, volatile _CUDA_VSTD::tuple<_Tp...>>
    : _CUDA_VSTD::tuple_element<_Ip, volatile _CUDA_VSTD::tuple<_Tp...>>
{};

template <size_t _Ip, class... _Tp>
struct tuple_element<_Ip, const volatile _CUDA_VSTD::tuple<_Tp...>>
    : _CUDA_VSTD::tuple_element<_Ip, const volatile _CUDA_VSTD::tuple<_Tp...>>
{};

template <class _Ip, class _Sp, _CUDA_VRANGES::subrange_kind _Kp>
struct tuple_size<_CUDA_VRANGES::subrange<_Ip, _Sp, _Kp>>
    : _CUDA_VSTD::tuple_size<_CUDA_VRANGES::subrange<_Ip, _Sp, _Kp>>
{};

template <class _Ip, class _Sp, _CUDA_VRANGES::subrange_kind _Kp>
struct tuple_size<const _CUDA_VRANGES::subrange<_Ip, _Sp, _Kp>>
    : _CUDA_VSTD::tuple_size<_CUDA_VRANGES::subrange<_Ip, _Sp, _Kp>>
{};

template <class _Ip, class _Sp, _CUDA_VRANGES::subrange_kind _Kp>
struct tuple_size<volatile _CUDA_VRANGES::subrange<_Ip, _Sp, _Kp>>
    : _CUDA_VSTD::tuple_size<_CUDA_VRANGES::subrange<_Ip, _Sp, _Kp>>
{};

template <class _Ip, class _Sp, _CUDA_VRANGES::subrange_kind _Kp>
struct tuple_size<const volatile _CUDA_VRANGES::subrange<_Ip, _Sp, _Kp>>
    : _CUDA_VSTD::tuple_size<_CUDA_VRANGES::subrange<_Ip, _Sp, _Kp>>
{};

template <size_t _Idx, class _Ip, class _Sp, _CUDA_VRANGES::subrange_kind _Kp>
struct tuple_element<_Idx, _CUDA_VRANGES::subrange<_Ip, _Sp, _Kp>>
    : _CUDA_VSTD::tuple_element<_Idx, _CUDA_VRANGES::subrange<_Ip, _Sp, _Kp>>
{};

template <size_t _Idx, class _Ip, class _Sp, _CUDA_VRANGES::subrange_kind _Kp>
struct tuple_element<_Idx, const _CUDA_VRANGES::subrange<_Ip, _Sp, _Kp>>
    : _CUDA_VSTD::tuple_element<_Idx, const _CUDA_VRANGES::subrange<_Ip, _Sp, _Kp>>
{};

template <size_t _Idx, class _Ip, class _Sp, _CUDA_VRANGES::subrange_kind _Kp>
struct tuple_element<_Idx, volatile _CUDA_VRANGES::subrange<_Ip, _Sp, _Kp>>
    : _CUDA_VSTD::tuple_element<_Idx, volatile _CUDA_VRANGES::subrange<_Ip, _Sp, _Kp>>
{};

template <size_t _Idx, class _Ip, class _Sp, _CUDA_VRANGES::subrange_kind _Kp>
struct tuple_element<_Idx, const volatile _CUDA_VRANGES::subrange<_Ip, _Sp, _Kp>>
    : _CUDA_VSTD::tuple_element<_Idx, const volatile _CUDA_VRANGES::subrange<_Ip, _Sp, _Kp>>
{};
} // namespace std

#include <cuda/std/__cccl/epilogue.h>

_CCCL_DIAG_POP

#endif // _LIBCUDACXX___TUPLE_STRUCTURED_BINDINGS_H
