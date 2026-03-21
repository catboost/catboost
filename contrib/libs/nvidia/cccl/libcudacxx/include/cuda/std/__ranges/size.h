// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_SIZE_H
#define _LIBCUDACXX___RANGES_SIZE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/arithmetic.h>
#include <cuda/std/__concepts/class_or_enum.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__type_traits/is_unbounded_array.h>
#include <cuda/std/__type_traits/make_signed.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/auto_cast.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

template <class>
inline constexpr bool disable_sized_range = false;

// [range.prim.size]

_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__size)
template <class _Tp>
void size(_Tp&) = delete;
template <class _Tp>
void size(const _Tp&) = delete;

template <class _Tp>
_CCCL_CONCEPT __size_enabled = !disable_sized_range<remove_cvref_t<_Tp>>;

#if _CCCL_HAS_CONCEPTS()
template <class _Tp>
concept __member_size = __size_enabled<_Tp> && __workaround_52970<_Tp> && requires(_Tp&& __t) {
  { _LIBCUDACXX_AUTO_CAST(__t.size()) } -> __integer_like;
};

template <class _Tp>
concept __unqualified_size =
  __size_enabled<_Tp> && !__member_size<_Tp> && __class_or_enum<remove_cvref_t<_Tp>> && requires(_Tp&& __t) {
    { _LIBCUDACXX_AUTO_CAST(size(__t)) } -> __integer_like;
  };

template <class _Tp>
concept __difference =
  !__member_size<_Tp> && !__unqualified_size<_Tp> && __class_or_enum<remove_cvref_t<_Tp>> && requires(_Tp&& __t) {
    { _CUDA_VRANGES::begin(__t) } -> forward_iterator;
    { _CUDA_VRANGES::end(__t) } -> sized_sentinel_for<decltype(_CUDA_VRANGES::begin(_CUDA_VSTD::declval<_Tp>()))>;
  };
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(__member_size_,
                       requires(_Tp&& __t)(requires(__size_enabled<_Tp>),
                                           requires(__workaround_52970<_Tp>),
                                           requires(__integer_like<decltype(_LIBCUDACXX_AUTO_CAST(__t.size()))>)));

template <class _Tp>
_CCCL_CONCEPT __member_size = _CCCL_FRAGMENT(__member_size_, _Tp);

template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(
  __unqualified_size_,
  requires(_Tp&& __t)(requires(__size_enabled<_Tp>),
                      requires(!__member_size<_Tp>),
                      requires(__class_or_enum<remove_cvref_t<_Tp>>),
                      requires(__integer_like<decltype(_LIBCUDACXX_AUTO_CAST(size(__t)))>)));

template <class _Tp>
_CCCL_CONCEPT __unqualified_size = _CCCL_FRAGMENT(__unqualified_size_, _Tp);

template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(
  __difference_,
  requires(_Tp&& __t)(requires(!__member_size<_Tp>),
                      requires(!__unqualified_size<_Tp>),
                      requires(__class_or_enum<remove_cvref_t<_Tp>>),
                      requires(forward_iterator<decltype(_CUDA_VRANGES::begin(__t))>),
                      requires(sized_sentinel_for<decltype(_CUDA_VRANGES::end(__t)),
                                                  decltype(_CUDA_VRANGES::begin(_CUDA_VSTD::declval<_Tp>()))>)));

template <class _Tp>
_CCCL_CONCEPT __difference = _CCCL_FRAGMENT(__difference_, _Tp);
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

struct __fn
{
  // `[range.prim.size]`: the array case (for rvalues).
  template <class _Tp, size_t _Sz>
  [[nodiscard]] _CCCL_API constexpr size_t operator()(_Tp (&&)[_Sz]) const noexcept
  {
    return _Sz;
  }

  // `[range.prim.size]`: the array case (for lvalues).
  template <class _Tp, size_t _Sz>
  [[nodiscard]] _CCCL_API constexpr size_t operator()(_Tp (&)[_Sz]) const noexcept
  {
    return _Sz;
  }

  // `[range.prim.size]`: `auto(t.size())` is a valid expression.
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__member_size<_Tp>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Tp&& __t) const
    noexcept(noexcept(_LIBCUDACXX_AUTO_CAST(__t.size())))
  {
    return _LIBCUDACXX_AUTO_CAST(__t.size());
  }

  // `[range.prim.size]`: `auto(size(t))` is a valid expression.
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__unqualified_size<_Tp>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Tp&& __t) const
    noexcept(noexcept(_LIBCUDACXX_AUTO_CAST(size(__t))))
  {
    return _LIBCUDACXX_AUTO_CAST(size(__t));
  }

  // [range.prim.size]: the `to-unsigned-like` case.
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__difference<_Tp>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Tp&& __t) const
    noexcept(noexcept(_CUDA_VSTD::__to_unsigned_like(_CUDA_VRANGES::end(__t) - _CUDA_VRANGES::begin(__t))))
      -> decltype(_CUDA_VSTD::__to_unsigned_like(_CUDA_VRANGES::end(__t) - _CUDA_VRANGES::begin(__t)))
  {
    return _CUDA_VSTD::__to_unsigned_like(_CUDA_VRANGES::end(__t) - _CUDA_VRANGES::begin(__t));
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto size = __size::__fn{};
} // namespace __cpo

// [range.prim.ssize]

_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__ssize)
#if _CCCL_HAS_CONCEPTS()
template <class _Tp>
concept __can_ssize = requires(_Tp&& __t) { _CUDA_VRANGES::size(__t); };
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(__can_ssize_,
                       requires(_Tp&& __t)(requires(!is_unbounded_array_v<_Tp>), ((void) _CUDA_VRANGES::size(__t))));

template <class _Tp>
_CCCL_CONCEPT __can_ssize = _CCCL_FRAGMENT(__can_ssize_, _Tp);
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

struct __fn
{
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__can_ssize<_Tp>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Tp&& __t) const noexcept(noexcept(_CUDA_VRANGES::size(__t)))
  {
    using _Signed = make_signed_t<decltype(_CUDA_VRANGES::size(__t))>;
    using _Result = conditional_t<(sizeof(ptrdiff_t) > sizeof(_Signed)), ptrdiff_t, _Signed>;
    return static_cast<_Result>(_CUDA_VRANGES::size(__t));
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto ssize = __ssize::__fn{};
} // namespace __cpo

_LIBCUDACXX_END_NAMESPACE_RANGES

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___RANGES_SIZE_H
