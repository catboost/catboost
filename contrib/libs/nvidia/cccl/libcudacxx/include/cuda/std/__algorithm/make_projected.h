//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_MAKE_PROJECTED_H
#define _LIBCUDACXX___ALGORITHM_MAKE_PROJECTED_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__functional/identity.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_member_pointer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/forward.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Pred, class _Proj>
struct _ProjectedPred
{
  _Pred& __pred; // Can be a unary or a binary predicate.
  _Proj& __proj;

  _CCCL_API constexpr _ProjectedPred(_Pred& __pred_arg, _Proj& __proj_arg)
      : __pred(__pred_arg)
      , __proj(__proj_arg)
  {}

  template <class _Tp>
  typename __invoke_of<
    _Pred&,
    decltype(_CUDA_VSTD::__invoke(_CUDA_VSTD::declval<_Proj&>(), _CUDA_VSTD::declval<_Tp>()))>::type constexpr
    _CCCL_API inline
    operator()(_Tp&& __v) const
  {
    return _CUDA_VSTD::__invoke(__pred, _CUDA_VSTD::__invoke(__proj, _CUDA_VSTD::forward<_Tp>(__v)));
  }

  template <class _T1, class _T2>
  typename __invoke_of<
    _Pred&,
    decltype(_CUDA_VSTD::__invoke(_CUDA_VSTD::declval<_Proj&>(), _CUDA_VSTD::declval<_T1>())),
    decltype(_CUDA_VSTD::__invoke(_CUDA_VSTD::declval<_Proj&>(), _CUDA_VSTD::declval<_T2>()))>::type constexpr
    _CCCL_API inline
    operator()(_T1&& __lhs, _T2&& __rhs) const
  {
    return _CUDA_VSTD::__invoke(__pred,
                                _CUDA_VSTD::__invoke(__proj, _CUDA_VSTD::forward<_T1>(__lhs)),
                                _CUDA_VSTD::__invoke(__proj, _CUDA_VSTD::forward<_T2>(__rhs)));
  }
};

template <class _Pred,
          class _Proj,
          enable_if_t<!(!is_member_pointer<decay_t<_Pred>>::value && __is_identity<decay_t<_Proj>>::value), int> = 0>
_CCCL_API constexpr _ProjectedPred<_Pred, _Proj> __make_projected(_Pred& __pred, _Proj& __proj)
{
  return _ProjectedPred<_Pred, _Proj>(__pred, __proj);
}

// Avoid creating the functor and just use the pristine comparator -- for certain algorithms, this would enable
// optimizations that rely on the type of the comparator. Additionally, this results in less layers of indirection in
// the call stack when the comparator is invoked, even in an unoptimized build.
template <class _Pred,
          class _Proj,
          enable_if_t<!is_member_pointer<decay_t<_Pred>>::value && __is_identity<decay_t<_Proj>>::value, int> = 0>
_CCCL_API constexpr _Pred& __make_projected(_Pred& __pred, _Proj&)
{
  return __pred;
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_MAKE_PROJECTED_H
