//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===---------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FWD_SUBRANGE_H
#define _LIBCUDACXX___FWD_SUBRANGE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__iterator/concepts.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

enum class _CCCL_TYPE_VISIBILITY_DEFAULT subrange_kind : bool
{
  unsized,
  sized
};

#if _CCCL_HAS_CONCEPTS()
template <input_or_output_iterator _Iter,
          sentinel_for<_Iter> _Sent = _Iter,
          subrange_kind _Kind       = sized_sentinel_for<_Sent, _Iter> ? subrange_kind::sized : subrange_kind::unsized>
  requires(_Kind == subrange_kind::sized || !sized_sentinel_for<_Sent, _Iter>)
class _CCCL_TYPE_VISIBILITY_DEFAULT subrange;
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class _Iter,
          class _Sent         = _Iter,
          subrange_kind _Kind = sized_sentinel_for<_Sent, _Iter> ? subrange_kind::sized : subrange_kind::unsized,
          enable_if_t<input_or_output_iterator<_Iter>, int>                                      = 0,
          enable_if_t<sentinel_for<_Sent, _Iter>, int>                                           = 0,
          enable_if_t<(_Kind == subrange_kind::sized || !sized_sentinel_for<_Sent, _Iter>), int> = 0>
class _CCCL_TYPE_VISIBILITY_DEFAULT subrange;
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

_LIBCUDACXX_END_NAMESPACE_RANGES

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___FWD_SUBRANGE_H
