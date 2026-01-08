//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_BASE_OF_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_BASE_OF_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_class.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_IS_BASE_OF) && !defined(_LIBCUDACXX_USE_IS_BASE_OF_FALLBACK)

template <class _Bp, class _Dp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_base_of : public integral_constant<bool, _CCCL_BUILTIN_IS_BASE_OF(_Bp, _Dp)>
{};

template <class _Bp, class _Dp>
inline constexpr bool is_base_of_v = _CCCL_BUILTIN_IS_BASE_OF(_Bp, _Dp);

#else // defined(_CCCL_BUILTIN_IS_BASE_OF) && !defined(_LIBCUDACXX_USE_IS_BASE_OF_FALLBACK)

namespace __is_base_of_imp
{
template <class _Tp>
struct _Dst
{
  _CCCL_API inline _Dst(const volatile _Tp&);
};
template <class _Tp>
struct _Src
{
  _CCCL_API inline operator const volatile _Tp&();
  template <class _Up>
  _CCCL_API inline operator const _Dst<_Up>&();
};
template <size_t>
struct __one
{
  using type = char;
};
template <class _Bp, class _Dp>
_CCCL_HOST_DEVICE typename __one<sizeof(_Dst<_Bp>(_CUDA_VSTD::declval<_Src<_Dp>>()))>::type __test(int);
template <class _Bp, class _Dp>
_CCCL_HOST_DEVICE __two __test(...);
} // namespace __is_base_of_imp

template <class _Bp, class _Dp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
is_base_of : public integral_constant<bool, is_class<_Bp>::value && sizeof(__is_base_of_imp::__test<_Bp, _Dp>(0)) == 2>
{};

template <class _Bp, class _Dp>
inline constexpr bool is_base_of_v = is_base_of<_Bp, _Dp>::value;

#endif // defined(_CCCL_BUILTIN_IS_BASE_OF) && !defined(_LIBCUDACXX_USE_IS_BASE_OF_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_BASE_OF_H
