//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023-24 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FWD_GET_H
#define _LIBCUDACXX___FWD_GET_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/copyable.h>
#include <cuda/std/__fwd/array.h>
#include <cuda/std/__fwd/complex.h>
#include <cuda/std/__fwd/pair.h>
#include <cuda/std/__fwd/subrange.h>
#include <cuda/std/__fwd/tuple.h>
#include <cuda/std/__tuple_dir/tuple_element.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <size_t _Ip, class... _Tp>
_CCCL_API constexpr tuple_element_t<_Ip, tuple<_Tp...>>& get(tuple<_Tp...>&) noexcept;

template <size_t _Ip, class... _Tp>
_CCCL_API constexpr const tuple_element_t<_Ip, tuple<_Tp...>>& get(const tuple<_Tp...>&) noexcept;

template <size_t _Ip, class... _Tp>
_CCCL_API constexpr tuple_element_t<_Ip, tuple<_Tp...>>&& get(tuple<_Tp...>&&) noexcept;

template <size_t _Ip, class... _Tp>
_CCCL_API constexpr const tuple_element_t<_Ip, tuple<_Tp...>>&& get(const tuple<_Tp...>&&) noexcept;

template <size_t _Ip, class _T1, class _T2>
_CCCL_API constexpr tuple_element_t<_Ip, pair<_T1, _T2>>& get(pair<_T1, _T2>&) noexcept;

template <size_t _Ip, class _T1, class _T2>
_CCCL_API constexpr const tuple_element_t<_Ip, pair<_T1, _T2>>& get(const pair<_T1, _T2>&) noexcept;

template <size_t _Ip, class _T1, class _T2>
_CCCL_API constexpr tuple_element_t<_Ip, pair<_T1, _T2>>&& get(pair<_T1, _T2>&&) noexcept;

template <size_t _Ip, class _T1, class _T2>
_CCCL_API constexpr const tuple_element_t<_Ip, pair<_T1, _T2>>&& get(const pair<_T1, _T2>&&) noexcept;

template <size_t _Ip, class _Tp, size_t _Size>
_CCCL_API constexpr _Tp& get(array<_Tp, _Size>&) noexcept;

template <size_t _Ip, class _Tp, size_t _Size>
_CCCL_API constexpr const _Tp& get(const array<_Tp, _Size>&) noexcept;

template <size_t _Ip, class _Tp, size_t _Size>
_CCCL_API constexpr _Tp&& get(array<_Tp, _Size>&&) noexcept;

template <size_t _Ip, class _Tp, size_t _Size>
_CCCL_API constexpr const _Tp&& get(const array<_Tp, _Size>&&) noexcept;

template <size_t _Ip, class _Tp>
_CCCL_API constexpr _Tp& get(complex<_Tp>&) noexcept;

template <size_t _Ip, class _Tp>
_CCCL_API constexpr _Tp&& get(complex<_Tp>&&) noexcept;

template <size_t _Ip, class _Tp>
_CCCL_API constexpr const _Tp& get(const complex<_Tp>&) noexcept;

template <size_t _Ip, class _Tp>
_CCCL_API constexpr const _Tp&& get(const complex<_Tp>&&) noexcept;

_LIBCUDACXX_END_NAMESPACE_STD

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

#if _CCCL_HAS_CONCEPTS()
template <size_t _Index, class _Iter, class _Sent, subrange_kind _Kind>
  requires((_Index == 0) && copyable<_Iter>) || (_Index == 1)
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <size_t _Index,
          class _Iter,
          class _Sent,
          subrange_kind _Kind,
          enable_if_t<((_Index == 0) && copyable<_Iter>) || (_Index == 1), int> = 0>
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^
_CCCL_API constexpr auto get(const subrange<_Iter, _Sent, _Kind>& __subrange);

#if _CCCL_HAS_CONCEPTS()
template <size_t _Index, class _Iter, class _Sent, subrange_kind _Kind>
  requires(_Index < 2)
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <
  size_t _Index,
  class _Iter,
  class _Sent,
  subrange_kind _Kind,
  enable_if_t<_Index<2, int> = 0>
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^
_CCCL_API constexpr auto get(subrange<_Iter, _Sent, _Kind>&& __subrange);

_LIBCUDACXX_END_NAMESPACE_RANGES

_LIBCUDACXX_BEGIN_NAMESPACE_STD

using _CUDA_VRANGES::get;

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___FWD_GET_H
