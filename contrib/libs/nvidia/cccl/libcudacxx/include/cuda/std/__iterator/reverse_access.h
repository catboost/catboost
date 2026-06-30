// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_REVERSE_ACCESS_H
#define _LIBCUDACXX___ITERATOR_REVERSE_ACCESS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__iterator/reverse_iterator.h>
#include <cuda/std/cstddef>
#include <cuda/std/initializer_list>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

namespace __rbegin
{
struct __fn
{
  template <class _Tp, size_t _Np>
  _CCCL_API constexpr reverse_iterator<_Tp*> operator()(_Tp (&__array)[_Np]) const noexcept
  {
    return reverse_iterator<_Tp*>(__array + _Np);
  }

  template <class _Ep>
  _CCCL_API constexpr reverse_iterator<const _Ep*> operator()(initializer_list<_Ep> __il) const noexcept
  {
    return reverse_iterator<const _Ep*>(__il.end());
  }

  template <class _Cp>
  _CCCL_API constexpr auto operator()(_Cp& __c) const noexcept(noexcept(__c.rbegin())) -> decltype(__c.rbegin())
  {
    return __c.rbegin();
  }

  template <class _Cp>
  _CCCL_API constexpr auto operator()(const _Cp& __c) const noexcept(noexcept(__c.rbegin())) -> decltype(__c.rbegin())
  {
    return __c.rbegin();
  }
};
} // namespace __rbegin

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto rbegin = __rbegin::__fn{};
} // namespace __cpo

namespace __rend
{
struct __fn
{
  template <class _Tp, size_t _Np>
  _CCCL_API constexpr reverse_iterator<_Tp*> operator()(_Tp (&__array)[_Np]) const noexcept
  {
    return reverse_iterator<_Tp*>(__array);
  }

  template <class _Ep>
  _CCCL_API constexpr reverse_iterator<const _Ep*> operator()(initializer_list<_Ep> __il) const noexcept
  {
    return reverse_iterator<const _Ep*>(__il.begin());
  }

  template <class _Cp>
  _CCCL_API constexpr auto operator()(_Cp& __c) const noexcept(noexcept(__c.rend())) -> decltype(__c.rend())
  {
    return __c.rend();
  }

  template <class _Cp>
  _CCCL_API constexpr auto operator()(const _Cp& __c) const noexcept(noexcept(__c.rend())) -> decltype(__c.rend())
  {
    return __c.rend();
  }
};
} // namespace __rend

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto rend = __rend::__fn{};
} // namespace __cpo

namespace __crbegin
{
struct __fn
{
  template <class _Cp>
  _CCCL_API constexpr auto operator()(const _Cp& __c) const noexcept(noexcept(_CUDA_VSTD::rbegin(__c)))
    -> decltype(_CUDA_VSTD::rbegin(__c))
  {
    return _CUDA_VSTD::rbegin(__c);
  }
};
} // namespace __crbegin

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto crbegin = __crbegin::__fn{};
} // namespace __cpo

namespace __crend
{
struct __fn
{
  template <class _Cp>
  _CCCL_API constexpr auto operator()(const _Cp& __c) const noexcept(noexcept(_CUDA_VSTD::rend(__c)))
    -> decltype(_CUDA_VSTD::rend(__c))
  {
    return _CUDA_VSTD::rend(__c);
  }
};
} // namespace __crend

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto crend = __crend::__fn{};
} // namespace __cpo

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ITERATOR_REVERSE_ACCESS_H
