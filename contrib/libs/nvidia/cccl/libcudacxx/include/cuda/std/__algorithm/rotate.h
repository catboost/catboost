//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_ROTATE_H
#define _LIBCUDACXX___ALGORITHM_ROTATE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/iterator_operations.h>
#include <cuda/std/__algorithm/move.h>
#include <cuda/std/__algorithm/move_backward.h>
#include <cuda/std/__algorithm/swap_ranges.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__type_traits/is_trivially_move_assignable.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/pair.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _AlgPolicy, class _ForwardIterator>
_CCCL_API constexpr _ForwardIterator __rotate_left(_ForwardIterator __first, _ForwardIterator __last)
{
  using value_type = typename iterator_traits<_ForwardIterator>::value_type;
  using _Ops       = _IterOps<_AlgPolicy>;

  value_type __tmp       = _Ops::__iter_move(__first);
  _ForwardIterator __lm1 = _CUDA_VSTD::__move<_AlgPolicy>(_Ops::next(__first), __last, __first).second;
  *__lm1                 = _CUDA_VSTD::move(__tmp);
  return __lm1;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _AlgPolicy, class _BidirectionalIterator>
_CCCL_API constexpr _BidirectionalIterator __rotate_right(_BidirectionalIterator __first, _BidirectionalIterator __last)
{
  using value_type = typename iterator_traits<_BidirectionalIterator>::value_type;
  using _Ops       = _IterOps<_AlgPolicy>;

  _BidirectionalIterator __lm1 = _Ops::prev(__last);
  value_type __tmp             = _Ops::__iter_move(__lm1);
  _BidirectionalIterator __fp1 =
    _CUDA_VSTD::__move_backward<_AlgPolicy>(__first, __lm1, _CUDA_VSTD::move(__last)).second;
  *__first = _CUDA_VSTD::move(__tmp);
  return __fp1;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _AlgPolicy, class _ForwardIterator>
_CCCL_API constexpr _ForwardIterator
__rotate_forward(_ForwardIterator __first, _ForwardIterator __middle, _ForwardIterator __last)
{
  _ForwardIterator __i = __middle;
  while (true)
  {
    _IterOps<_AlgPolicy>::iter_swap(__first, __i);
    ++__first;
    if (++__i == __last)
    {
      break;
    }
    if (__first == __middle)
    {
      __middle = __i;
    }
  }
  _ForwardIterator __r = __first;
  if (__first != __middle)
  {
    __i = __middle;
    while (true)
    {
      _IterOps<_AlgPolicy>::iter_swap(__first, __i);
      ++__first;
      if (++__i == __last)
      {
        if (__first == __middle)
        {
          break;
        }
        __i = __middle;
      }
      else if (__first == __middle)
      {
        __middle = __i;
      }
    }
  }
  return __r;
}

_CCCL_EXEC_CHECK_DISABLE
template <typename _Integral>
_CCCL_API constexpr _Integral __algo_gcd(_Integral __x, _Integral __y)
{
  do
  {
    _Integral __t = __x % __y;
    __x           = __y;
    __y           = __t;
  } while (__y);
  return __x;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _AlgPolicy, typename _RandomAccessIterator>
_CCCL_API constexpr _RandomAccessIterator
__rotate_gcd(_RandomAccessIterator __first, _RandomAccessIterator __middle, _RandomAccessIterator __last)
{
  using difference_type = typename iterator_traits<_RandomAccessIterator>::difference_type;
  using value_type      = typename iterator_traits<_RandomAccessIterator>::value_type;
  using _Ops            = _IterOps<_AlgPolicy>;

  const difference_type __m1 = __middle - __first;
  const difference_type __m2 = _Ops::distance(__middle, __last);
  if (__m1 == __m2)
  {
    _CUDA_VSTD::__swap_ranges<_AlgPolicy>(__first, __middle, __middle, __last);
    return __middle;
  }
  const difference_type __g = _CUDA_VSTD::__algo_gcd(__m1, __m2);
  for (_RandomAccessIterator __p = __first + __g; __p != __first;)
  {
    value_type __t(_Ops::__iter_move(--__p));
    _RandomAccessIterator __p1 = __p;
    _RandomAccessIterator __p2 = __p1 + __m1;
    do
    {
      *__p1                     = _Ops::__iter_move(__p2);
      __p1                      = __p2;
      const difference_type __d = _Ops::distance(__p2, __last);
      if (__m1 < __d)
      {
        __p2 += __m1;
      }
      else
      {
        __p2 = __first + (__m1 - __d);
      }
    } while (__p2 != __p);
    *__p1 = _CUDA_VSTD::move(__t);
  }
  return __first + __m2;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _AlgPolicy, class _ForwardIterator>
_CCCL_API constexpr _ForwardIterator __rotate_impl(
  _ForwardIterator __first, _ForwardIterator __middle, _ForwardIterator __last, _CUDA_VSTD::forward_iterator_tag)
{
  using value_type = typename iterator_traits<_ForwardIterator>::value_type;
  if (_CCCL_TRAIT(is_trivially_move_assignable, value_type))
  {
    if (_IterOps<_AlgPolicy>::next(__first) == __middle)
    {
      return _CUDA_VSTD::__rotate_left<_AlgPolicy>(__first, __last);
    }
  }
  return _CUDA_VSTD::__rotate_forward<_AlgPolicy>(__first, __middle, __last);
}

_CCCL_EXEC_CHECK_DISABLE
template <class _AlgPolicy, class _BidirectionalIterator>
_CCCL_API constexpr _BidirectionalIterator __rotate_impl(
  _BidirectionalIterator __first,
  _BidirectionalIterator __middle,
  _BidirectionalIterator __last,
  bidirectional_iterator_tag)
{
  using value_type = typename iterator_traits<_BidirectionalIterator>::value_type;
  if (_CCCL_TRAIT(is_trivially_move_assignable, value_type))
  {
    if (_IterOps<_AlgPolicy>::next(__first) == __middle)
    {
      return _CUDA_VSTD::__rotate_left<_AlgPolicy>(__first, __last);
    }
    if (_IterOps<_AlgPolicy>::next(__middle) == __last)
    {
      return _CUDA_VSTD::__rotate_right<_AlgPolicy>(__first, __last);
    }
  }
  return _CUDA_VSTD::__rotate_forward<_AlgPolicy>(__first, __middle, __last);
}

_CCCL_EXEC_CHECK_DISABLE
template <class _AlgPolicy, class _RandomAccessIterator>
_CCCL_API constexpr _RandomAccessIterator __rotate_impl(
  _RandomAccessIterator __first,
  _RandomAccessIterator __middle,
  _RandomAccessIterator __last,
  random_access_iterator_tag)
{
  using value_type = typename iterator_traits<_RandomAccessIterator>::value_type;
  if (_CCCL_TRAIT(is_trivially_move_assignable, value_type))
  {
    if (_IterOps<_AlgPolicy>::next(__first) == __middle)
    {
      return _CUDA_VSTD::__rotate_left<_AlgPolicy>(__first, __last);
    }
    if (_IterOps<_AlgPolicy>::next(__middle) == __last)
    {
      return _CUDA_VSTD::__rotate_right<_AlgPolicy>(__first, __last);
    }
    return _CUDA_VSTD::__rotate_gcd<_AlgPolicy>(__first, __middle, __last);
  }
  return _CUDA_VSTD::__rotate_forward<_AlgPolicy>(__first, __middle, __last);
}

_CCCL_EXEC_CHECK_DISABLE
template <class _AlgPolicy, class _Iterator, class _Sentinel>
_CCCL_API constexpr pair<_Iterator, _Iterator> __rotate(_Iterator __first, _Iterator __middle, _Sentinel __last)
{
  using _Ret            = pair<_Iterator, _Iterator>;
  _Iterator __last_iter = _IterOps<_AlgPolicy>::next(__middle, __last);

  if (__first == __middle)
  {
    return _Ret(__last_iter, __last_iter);
  }
  if (__middle == __last)
  {
    return _Ret(_CUDA_VSTD::move(__first), _CUDA_VSTD::move(__last_iter));
  }

  using _IterCategory = typename _IterOps<_AlgPolicy>::template __iterator_category<_Iterator>;
  auto __result       = _CUDA_VSTD::__rotate_impl<_AlgPolicy>(
    _CUDA_VSTD::move(__first), _CUDA_VSTD::move(__middle), __last_iter, _IterCategory());

  return _Ret(_CUDA_VSTD::move(__result), _CUDA_VSTD::move(__last_iter));
}

_CCCL_EXEC_CHECK_DISABLE
template <class _ForwardIterator>
_CCCL_API constexpr _ForwardIterator rotate(_ForwardIterator __first, _ForwardIterator __middle, _ForwardIterator __last)
{
  return _CUDA_VSTD::__rotate<_ClassicAlgPolicy>(
           _CUDA_VSTD::move(__first), _CUDA_VSTD::move(__middle), _CUDA_VSTD::move(__last))
    .first;
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_ROTATE_H
