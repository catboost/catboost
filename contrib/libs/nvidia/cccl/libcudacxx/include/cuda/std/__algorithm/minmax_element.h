//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_MINMAX_ELEMENT_H
#define _LIBCUDACXX___ALGORITHM_MINMAX_ELEMENT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/comp.h>
#include <cuda/std/__functional/identity.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__utility/pair.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Comp, class _Proj>
class _MinmaxElementLessFunc
{
  _Comp& __comp_;
  _Proj& __proj_;

public:
  _CCCL_API constexpr _MinmaxElementLessFunc(_Comp& __comp, _Proj& __proj)
      : __comp_(__comp)
      , __proj_(__proj)
  {}

  template <class _Iter>
  _CCCL_API constexpr bool operator()(_Iter& __it1, _Iter& __it2)
  {
    return _CUDA_VSTD::__invoke(__comp_, _CUDA_VSTD::__invoke(__proj_, *__it1), _CUDA_VSTD::__invoke(__proj_, *__it2));
  }
};

_CCCL_EXEC_CHECK_DISABLE
template <class _Iter, class _Sent, class _Proj, class _Comp>
_CCCL_API constexpr pair<_Iter, _Iter> __minmax_element_impl(_Iter __first, _Sent __last, _Comp& __comp, _Proj& __proj)
{
  auto __less = _MinmaxElementLessFunc<_Comp, _Proj>(__comp, __proj);

  pair<_Iter, _Iter> __result(__first, __first);
  if (__first == __last || ++__first == __last)
  {
    return __result;
  }

  if (__less(__first, __result.first))
  {
    __result.first = __first;
  }
  else
  {
    __result.second = __first;
  }

  while (++__first != __last)
  {
    _Iter __i = __first;
    if (++__first == __last)
    {
      if (__less(__i, __result.first))
      {
        __result.first = __i;
      }
      else if (!__less(__i, __result.second))
      {
        __result.second = __i;
      }
      return __result;
    }

    if (__less(__first, __i))
    {
      if (__less(__first, __result.first))
      {
        __result.first = __first;
      }
      if (!__less(__i, __result.second))
      {
        __result.second = __i;
      }
    }
    else
    {
      if (__less(__i, __result.first))
      {
        __result.first = __i;
      }
      if (!__less(__first, __result.second))
      {
        __result.second = __first;
      }
    }
  }

  return __result;
}

template <class _ForwardIterator, class _Compare>
[[nodiscard]] _CCCL_API constexpr pair<_ForwardIterator, _ForwardIterator>
minmax_element(_ForwardIterator __first, _ForwardIterator __last, _Compare __comp)
{
  static_assert(__has_input_traversal<_ForwardIterator>, "_CUDA_VSTD::minmax_element requires a ForwardIterator");
  static_assert(__is_callable<_Compare, decltype(*__first), decltype(*__first)>::value,
                "The comparator has to be callable");
  auto __proj = identity();
  return _CUDA_VSTD::__minmax_element_impl(__first, __last, __comp, __proj);
}

template <class _ForwardIterator>
[[nodiscard]] _CCCL_API constexpr pair<_ForwardIterator, _ForwardIterator>
minmax_element(_ForwardIterator __first, _ForwardIterator __last)
{
  return _CUDA_VSTD::minmax_element(__first, __last, __less{});
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_MINMAX_ELEMENT_H
