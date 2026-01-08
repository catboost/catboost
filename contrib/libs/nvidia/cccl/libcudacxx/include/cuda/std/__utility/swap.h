//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___UTILITY_SWAP_H
#define _LIBCUDACXX___UTILITY_SWAP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_move_assignable.h>
#include <cuda/std/__type_traits/is_move_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/is_swappable.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// we use type_identity_t<_Tp> as second parameter, to avoid ambiguity with std::swap, which will thus be preferred by
// overload resolution (which is ok since std::swap is only considered when explicitly called, or found by ADL for types
// from std::)
_CCCL_EXEC_CHECK_DISABLE
template <class _Tp>
_CCCL_API constexpr __swap_result_t<_Tp> swap(_Tp& __x, type_identity_t<_Tp>& __y) noexcept(
  _CCCL_TRAIT(is_nothrow_move_constructible, _Tp) && _CCCL_TRAIT(is_nothrow_move_assignable, _Tp))
{
  _Tp __t(_CUDA_VSTD::move(__x));
  __x = _CUDA_VSTD::move(__y);
  __y = _CUDA_VSTD::move(__t);
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, size_t _Np>
_CCCL_API constexpr enable_if_t<__detect_adl_swap::__has_no_adl_swap_array<_Tp, _Np>::value && __is_swappable<_Tp>::value>
swap(_Tp (&__a)[_Np], _Tp (&__b)[_Np]) noexcept(__is_nothrow_swappable<_Tp>::value)
{
  for (size_t __i = 0; __i != _Np; ++__i)
  {
    swap(__a[__i], __b[__i]);
  }
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___UTILITY_SWAP_H
