//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_ALIGNED_UNION_H
#define _LIBCUDACXX___TYPE_TRAITS_ALIGNED_UNION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/aligned_storage.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <size_t _I0, size_t... _In>
struct __static_max;

template <size_t _I0>
struct __static_max<_I0>
{
  static const size_t value = _I0;
};

template <size_t _I0, size_t _I1, size_t... _In>
struct __static_max<_I0, _I1, _In...>
{
  static const size_t value = _I0 >= _I1 ? __static_max<_I0, _In...>::value : __static_max<_I1, _In...>::value;
};

template <size_t _Len, class _Type0, class... _Types>
struct aligned_union
{
  static const size_t alignment_value =
    __static_max<_LIBCUDACXX_PREFERRED_ALIGNOF(_Type0), _LIBCUDACXX_PREFERRED_ALIGNOF(_Types)...>::value;
  static const size_t __len = __static_max<_Len, sizeof(_Type0), sizeof(_Types)...>::value;
  using type                = typename aligned_storage<__len, alignment_value>::type;
};

template <size_t _Len, class... _Types>
using aligned_union_t _CCCL_NODEBUG_ALIAS = typename aligned_union<_Len, _Types...>::type;

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_ALIGNED_UNION_H
