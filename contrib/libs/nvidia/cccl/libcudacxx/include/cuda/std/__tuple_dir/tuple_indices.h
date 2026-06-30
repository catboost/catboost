//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TUPLE_MAKE_TUPLE_INDICES_H
#define _LIBCUDACXX___TUPLE_MAKE_TUPLE_INDICES_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <size_t _Ep, size_t _Sp = 0>
struct __make_tuple_indices
{
  static_assert(_Sp <= _Ep, "__make_tuple_indices input error");
  using type = __make_indices_imp<_Ep, _Sp>;
};

template <size_t _Ep, size_t _Sp = 0>
using __make_tuple_indices_t = typename __make_tuple_indices<_Ep, _Sp>::type;

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TUPLE_MAKE_TUPLE_INDICES_H
