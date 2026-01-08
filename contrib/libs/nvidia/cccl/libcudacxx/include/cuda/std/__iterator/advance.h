// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_ADVANCE_H
#define _LIBCUDACXX___ITERATOR_ADVANCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/assignable.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/incrementable_traits.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__utility/convert_to_integral.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_EXEC_CHECK_DISABLE
template <class _InputIter,
          class _Distance,
          class _IntegralDistance = decltype(_CUDA_VSTD::__convert_to_integral(_CUDA_VSTD::declval<_Distance>())),
          class                   = enable_if_t<is_integral<_IntegralDistance>::value>>
_CCCL_API constexpr void advance(_InputIter& __i, _Distance __orig_n)
{
  using _Difference = typename iterator_traits<_InputIter>::difference_type;
  _Difference __n   = static_cast<_Difference>(_CUDA_VSTD::__convert_to_integral(__orig_n));
  if constexpr (__has_random_access_traversal<_InputIter>) // To support pointers to incomplete types
  {
    __i += __n;
  }
  else if constexpr (__has_bidirectional_traversal<_InputIter>)
  {
    if (__n >= 0)
    {
      for (; __n > 0; --__n)
      {
        ++__i;
      }
    }
    else
    {
      for (; __n < 0; ++__n)
      {
        --__i;
      }
    }
  }
  else
  {
    _CCCL_ASSERT(__n >= 0, "Attempt to advance(it, n) with negative n on a non-bidirectional iterator");
    for (; __n > 0; --__n)
    {
      ++__i;
    }
  }
}

_LIBCUDACXX_END_NAMESPACE_STD

// [range.iter.op.advance]

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES
_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__advance)
struct __fn
{
private:
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Iter_difference>
  [[nodiscard]] _CCCL_API static constexpr auto __magnitude_geq(_Iter_difference __a, _Iter_difference __b) noexcept
  {
    return __a == 0 ? __b == 0 : //
             __a > 0 ? __a >= __b
                     : __a <= __b;
  }

public:
  // Preconditions: If `I` does not model `bidirectional_iterator`, `n` is not negative.

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Ip)
  _CCCL_REQUIRES(input_or_output_iterator<_Ip>)
  _CCCL_API constexpr void operator()(_Ip& __i, iter_difference_t<_Ip> __n) const
  {
    _CCCL_ASSERT(__n >= 0 || bidirectional_iterator<_Ip>, "If `n < 0`, then `bidirectional_iterator<I>` must be true.");

    // If `I` models `random_access_iterator`, equivalent to `i += n`.
    if constexpr (random_access_iterator<_Ip>)
    {
      __i += __n;
      return;
    }
    else if constexpr (bidirectional_iterator<_Ip>)
    {
      // Otherwise, if `n` is non-negative, increments `i` by `n`.
      while (__n > 0)
      {
        --__n;
        ++__i;
      }
      // Otherwise, decrements `i` by `-n`.
      while (__n < 0)
      {
        ++__n;
        --__i;
      }
      return;
    }
    else
    {
      // Otherwise, if `n` is non-negative, increments `i` by `n`.
      while (__n > 0)
      {
        --__n;
        ++__i;
      }
      return;
    }
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Ip, class _Sp)
  _CCCL_REQUIRES(input_or_output_iterator<_Ip> _CCCL_AND sentinel_for<_Sp, _Ip>)
  _CCCL_API constexpr void operator()(_Ip& __i, _Sp __bound_sentinel) const
  {
    // If `I` and `S` model `assignable_from<I&, S>`, equivalent to `i = std::move(bound_sentinel)`.
    if constexpr (assignable_from<_Ip&, _Sp>)
    {
      __i = _CUDA_VSTD::move(__bound_sentinel);
    }
    // Otherwise, if `S` and `I` model `sized_sentinel_for<S, I>`,
    // equivalent to `ranges::advance(i, bound_sentinel - i)`.
    else if constexpr (sized_sentinel_for<_Sp, _Ip>)
    {
      (*this)(__i, __bound_sentinel - __i);
    }
    // Otherwise, while `bool(i != bound_sentinel)` is true, increments `i`.
    else
    {
      while (__i != __bound_sentinel)
      {
        ++__i;
      }
    }
  }

  // Preconditions:
  //   * If `n > 0`, [i, bound_sentinel) denotes a range.
  //   * If `n == 0`, [i, bound_sentinel) or [bound_sentinel, i) denotes a range.
  //   * If `n < 0`, [bound_sentinel, i) denotes a range, `I` models `bidirectional_iterator`,
  //     and `I` and `S` model `same_as<I, S>`.
  // Returns: `n - M`, where `M` is the difference between the ending and starting position.
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Ip, class _Sp)
  _CCCL_REQUIRES(input_or_output_iterator<_Ip> _CCCL_AND sentinel_for<_Sp, _Ip>)
  _CCCL_API constexpr iter_difference_t<_Ip> operator()(_Ip& __i, iter_difference_t<_Ip> __n, _Sp __bound_sentinel) const
  {
    _CCCL_ASSERT((__n >= 0) || (bidirectional_iterator<_Ip> && same_as<_Ip, _Sp>),
                 "If `n < 0`, then `bidirectional_iterator<I> && same_as<I, S>` must be true.");
    // If `S` and `I` model `sized_sentinel_for<S, I>`:
    if constexpr (sized_sentinel_for<_Sp, _Ip>)
    {
      // If |n| >= |bound_sentinel - i|, equivalent to `ranges::advance(i, bound_sentinel)`.
      // __magnitude_geq(a, b) returns |a| >= |b|, assuming they have the same sign.
      const auto __M = __bound_sentinel - __i;
      if (__magnitude_geq(__n, __M))
      {
        (*this)(__i, __bound_sentinel);
        return __n - __M;
      }

      // Otherwise, equivalent to `ranges::advance(i, n)`.
      (*this)(__i, __n);
      return 0;
    }
    else
    {
      // Otherwise, if `n` is non-negative, while `bool(i != bound_sentinel)` is true, increments `i` but at
      // most `n` times.
      while (__i != __bound_sentinel && __n > 0)
      {
        ++__i;
        --__n;
      }

      // Otherwise, while `bool(i != bound_sentinel)` is true, decrements `i` but at most `-n` times.
      if constexpr (bidirectional_iterator<_Ip> && same_as<_Ip, _Sp>)
      {
        while (__i != __bound_sentinel && __n < 0)
        {
          --__i;
          ++__n;
        }
      }
      return __n;
    }
    _CCCL_UNREACHABLE();
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto advance = __advance::__fn{};
} // namespace __cpo

_LIBCUDACXX_END_NAMESPACE_RANGES

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ITERATOR_ADVANCE_H
