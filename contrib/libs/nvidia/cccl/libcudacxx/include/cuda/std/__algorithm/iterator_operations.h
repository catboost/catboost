//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_ITERATOR_OPERATIONS_H
#define _LIBCUDACXX___ALGORITHM_ITERATOR_OPERATIONS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/iter_swap.h>
#include <cuda/std/__algorithm/ranges_iterator_concept.h>
#include <cuda/std/__iterator/advance.h>
#include <cuda/std/__iterator/distance.h>
#include <cuda/std/__iterator/incrementable_traits.h>
#include <cuda/std/__iterator/iter_move.h>
#include <cuda/std/__iterator/iter_swap.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__iterator/next.h>
#include <cuda/std/__iterator/prev.h>
#include <cuda/std/__iterator/readable_traits.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _AlgPolicy>
struct _IterOps;

struct _RangeAlgPolicy
{};

template <>
struct _IterOps<_RangeAlgPolicy>
{
  template <class _Iter>
  using __value_type = iter_value_t<_Iter>;

  template <class _Iter>
  using __iterator_category = _CUDA_VRANGES::__iterator_concept<_Iter>;

  template <class _Iter>
  using __difference_type = iter_difference_t<_Iter>;

  static constexpr auto advance      = _CUDA_VRANGES::advance;
  static constexpr auto distance     = _CUDA_VRANGES::distance;
  static constexpr auto __iter_move  = _CUDA_VRANGES::iter_move;
  static constexpr auto iter_swap    = _CUDA_VRANGES::iter_swap;
  static constexpr auto next         = _CUDA_VRANGES::next;
  static constexpr auto prev         = _CUDA_VRANGES::prev;
  static constexpr auto __advance_to = _CUDA_VRANGES::advance;
};

struct _ClassicAlgPolicy
{};

template <>
struct _IterOps<_ClassicAlgPolicy>
{
  template <class _Iter>
  using __value_type = typename iterator_traits<_Iter>::value_type;

  template <class _Iter>
  using __iterator_category = typename iterator_traits<_Iter>::iterator_category;

  template <class _Iter>
  using __difference_type = typename iterator_traits<_Iter>::difference_type;

  // advance
  template <class _Iter, class _Distance>
  _CCCL_API constexpr static void advance(_Iter& __iter, _Distance __count)
  {
    _CUDA_VSTD::advance(__iter, __count);
  }

  // distance
  template <class _Iter>
  _CCCL_API constexpr static typename iterator_traits<_Iter>::difference_type distance(_Iter __first, _Iter __last)
  {
    return _CUDA_VSTD::distance(__first, __last);
  }

  template <class _Iter>
  using __deref_t = decltype(*_CUDA_VSTD::declval<_Iter&>());

  template <class _Iter>
  using __move_t = decltype(_CUDA_VSTD::move(*_CUDA_VSTD::declval<_Iter&>()));

  template <class _Iter>
  _CCCL_API constexpr static void __validate_iter_reference()
  {
    static_assert(
      is_same<__deref_t<_Iter>, typename iterator_traits<remove_cvref_t<_Iter>>::reference>::value,
      "It looks like your iterator's `iterator_traits<It>::reference` does not match the return type of "
      "dereferencing the iterator, i.e., calling `*it`. This is undefined behavior according to [input.iterators] "
      "and can lead to dangling reference issues at runtime, so we are flagging this.");
  }

  // iter_move
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter, enable_if_t<is_reference<__deref_t<_Iter>>::value, int> = 0>
  _CCCL_API constexpr static
    // If the result of dereferencing `_Iter` is a reference type, deduce the result of calling `_CUDA_VSTD::move` on
    // it. Note that the C++03 mode doesn't support `decltype(auto)` as the return type.
    __move_t<_Iter>
    __iter_move(_Iter&& __i)
  {
    __validate_iter_reference<_Iter>();

    return _CUDA_VSTD::move(*_CUDA_VSTD::forward<_Iter>(__i));
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter, enable_if_t<!is_reference<__deref_t<_Iter>>::value, int> = 0>
  _CCCL_API constexpr static
    // If the result of dereferencing `_Iter` is a value type, deduce the return value of this function to also be a
    // value -- otherwise, after `operator*` returns a temporary, this function would return a dangling reference to
    // that temporary. Note that the C++03 mode doesn't support `auto` as the return type.
    __deref_t<_Iter>
    __iter_move(_Iter&& __i)
  {
    __validate_iter_reference<_Iter>();

    return *_CUDA_VSTD::forward<_Iter>(__i);
  }

  // iter_swap
  template <class _Iter1, class _Iter2>
  _CCCL_API constexpr static void iter_swap(_Iter1&& __a, _Iter2&& __b)
  {
    _CUDA_VSTD::iter_swap(_CUDA_VSTD::forward<_Iter1>(__a), _CUDA_VSTD::forward<_Iter2>(__b));
  }

  // next
  template <class _Iterator>
  _CCCL_API static constexpr _Iterator next(_Iterator, _Iterator __last)
  {
    return __last;
  }

  template <class _Iter>
  _CCCL_API static constexpr remove_cvref_t<_Iter> next(_Iter&& __it, __difference_type<remove_cvref_t<_Iter>> __n = 1)
  {
    return _CUDA_VSTD::next(_CUDA_VSTD::forward<_Iter>(__it), __n);
  }

  // prev
  template <class _Iter>
  _CCCL_API static constexpr remove_cvref_t<_Iter> prev(_Iter&& __iter, __difference_type<remove_cvref_t<_Iter>> __n = 1)
  {
    return _CUDA_VSTD::prev(_CUDA_VSTD::forward<_Iter>(__iter), __n);
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter>
  _CCCL_API static constexpr void __advance_to(_Iter& __first, _Iter __last)
  {
    __first = __last;
  }
};

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ALGORITHM_ITERATOR_OPERATIONS_H
