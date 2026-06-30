// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023-25 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FUNCTIONAL_COMPOSE_H
#define _LIBCUDACXX___FUNCTIONAL_COMPOSE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__functional/perfect_forward.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__utility/forward.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

struct __compose_op
{
  template <class _Fn1, class _Fn2, class... _Args>
  _CCCL_API constexpr auto operator()(_Fn1&& __f1, _Fn2&& __f2, _Args&&... __args) const noexcept(noexcept(
    _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fn1>(__f1),
                       _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fn2>(__f2), _CUDA_VSTD::forward<_Args>(__args)...))))
    -> decltype(_CUDA_VSTD::invoke(
      _CUDA_VSTD::forward<_Fn1>(__f1),
      _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fn2>(__f2), _CUDA_VSTD::forward<_Args>(__args)...)))
  {
    return _CUDA_VSTD::invoke(
      _CUDA_VSTD::forward<_Fn1>(__f1),
      _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fn2>(__f2), _CUDA_VSTD::forward<_Args>(__args)...));
  }
};

template <class _Fn1, class _Fn2>
struct __compose_t : __perfect_forward<__compose_op, _Fn1, _Fn2>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__compose_t, __perfect_forward, __compose_op, _Fn1, _Fn2);
};

template <class _Fn1, class _Fn2>
_CCCL_API constexpr auto __compose(_Fn1&& __f1, _Fn2&& __f2) noexcept(
  noexcept(__compose_t<decay_t<_Fn1>, decay_t<_Fn2>>(_CUDA_VSTD::forward<_Fn1>(__f1), _CUDA_VSTD::forward<_Fn2>(__f2))))
  -> decltype(__compose_t<decay_t<_Fn1>, decay_t<_Fn2>>(
    _CUDA_VSTD::forward<_Fn1>(__f1), _CUDA_VSTD::forward<_Fn2>(__f2)))
{
  return __compose_t<decay_t<_Fn1>, decay_t<_Fn2>>(_CUDA_VSTD::forward<_Fn1>(__f1), _CUDA_VSTD::forward<_Fn2>(__f2));
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___FUNCTIONAL_COMPOSE_H
