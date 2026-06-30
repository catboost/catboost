// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_ACCESS_H
#define _LIBCUDACXX___RANGES_ACCESS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/class_or_enum.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/readable_traits.h>
#include <cuda/std/__ranges/enable_borrowed_range.h>
#include <cuda/std/__type_traits/is_array.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/auto_cast.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

template <class _Tp>
_CCCL_CONCEPT __can_borrow = is_lvalue_reference_v<_Tp> || enable_borrowed_range<remove_cvref_t<_Tp>>;

// [range.access.begin]

_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__begin)
template <class _Tp>
void begin(_Tp&) = delete;
template <class _Tp>
void begin(const _Tp&) = delete;

#if _CCCL_HAS_CONCEPTS()
template <class _Tp>
concept __member_begin = __can_borrow<_Tp> && __workaround_52970<_Tp> && requires(_Tp&& __t) {
  { _LIBCUDACXX_AUTO_CAST(__t.begin()) } -> input_or_output_iterator;
};

template <class _Tp>
concept __unqualified_begin =
  !__member_begin<_Tp> && __can_borrow<_Tp> && __class_or_enum<remove_cvref_t<_Tp>> && requires(_Tp&& __t) {
    { _LIBCUDACXX_AUTO_CAST(begin(__t)) } -> input_or_output_iterator;
  };
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(
  __member_begin_,
  requires(_Tp&& __t)(requires(__can_borrow<_Tp>),
                      requires(__workaround_52970<_Tp>),
                      requires(input_or_output_iterator<decltype(_LIBCUDACXX_AUTO_CAST(__t.begin()))>)));

template <class _Tp>
_CCCL_CONCEPT __member_begin = _CCCL_FRAGMENT(__member_begin_, _Tp);

template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(
  __unqualified_begin_,
  requires(_Tp&& __t)(requires(!__member_begin<_Tp>),
                      requires(__can_borrow<_Tp>),
                      requires(__class_or_enum<remove_cvref_t<_Tp>>),
                      requires(input_or_output_iterator<decltype(_LIBCUDACXX_AUTO_CAST(begin(__t)))>)));

template <class _Tp>
_CCCL_CONCEPT __unqualified_begin = _CCCL_FRAGMENT(__unqualified_begin_, _Tp);
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

struct __fn
{
  // This has been made valid as a defect report for C++17 onwards, however gcc below 11.0 does not implement it
#if (!_CCCL_COMPILER(GCC) || _CCCL_COMPILER(GCC, >=, 11))
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES((sizeof(_Tp) >= 0)) // Disallow incomplete element types.
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Tp (&__t)[]) const noexcept
  {
    return __t + 0;
  }
#endif // (!_CCCL_COMPILER(GCC) || _CCCL_COMPILER(GCC, >=, 11))

  _CCCL_TEMPLATE(class _Tp, size_t _Np)
  _CCCL_REQUIRES((sizeof(_Tp) >= 0)) // Disallow incomplete element types.
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Tp (&__t)[_Np]) const noexcept
  {
    return __t + 0;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__member_begin<_Tp>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Tp&& __t) const
    noexcept(noexcept(_LIBCUDACXX_AUTO_CAST(__t.begin())))
  {
    return _LIBCUDACXX_AUTO_CAST(__t.begin());
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__unqualified_begin<_Tp>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Tp&& __t) const
    noexcept(noexcept(_LIBCUDACXX_AUTO_CAST(begin(__t))))
  {
    return _LIBCUDACXX_AUTO_CAST(begin(__t));
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES((!__member_begin<_Tp>) _CCCL_AND(!__unqualified_begin<_Tp>))
  void operator()(_Tp&&) const = delete;

#if _CCCL_COMPILER(MSVC, <, 19, 23)
  template <class _Tp>
  void operator()(_Tp (&&)[]) const = delete;

  template <class _Tp, size_t _Np>
  void operator()(_Tp (&&)[_Np]) const = delete;
#endif // _CCCL_COMPILER(MSVC, <, 19, 23)
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto begin = __begin::__fn{};
} // namespace __cpo

// [range.range]

template <class _Tp>
using iterator_t = decltype(_CUDA_VRANGES::begin(_CUDA_VSTD::declval<_Tp&>()));

// [range.access.end]

_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__end)
template <class _Tp>
void end(_Tp&) = delete;
template <class _Tp>
void end(const _Tp&) = delete;

#if _CCCL_HAS_CONCEPTS()
template <class _Tp>
concept __member_end = __can_borrow<_Tp> && __workaround_52970<_Tp> && requires(_Tp&& __t) {
  typename iterator_t<_Tp>;
  { _LIBCUDACXX_AUTO_CAST(__t.end()) } -> sentinel_for<iterator_t<_Tp>>;
};

template <class _Tp>
concept __unqualified_end =
  !__member_end<_Tp> && __can_borrow<_Tp> && __class_or_enum<remove_cvref_t<_Tp>> && requires(_Tp&& __t) {
    typename iterator_t<_Tp>;
    { _LIBCUDACXX_AUTO_CAST(end(__t)) } -> sentinel_for<iterator_t<_Tp>>;
  };
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(
  __member_end_,
  requires(_Tp&& __t)(requires(__can_borrow<_Tp>),
                      requires(__workaround_52970<_Tp>),
                      typename(iterator_t<_Tp>),
                      requires(sentinel_for<decltype(_LIBCUDACXX_AUTO_CAST(__t.end())), iterator_t<_Tp>>)));

template <class _Tp>
_CCCL_CONCEPT __member_end = _CCCL_FRAGMENT(__member_end_, _Tp);

template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(
  __unqualified_end_,
  requires(_Tp&& __t)(requires(!__member_end<_Tp>),
                      requires(__can_borrow<_Tp>),
                      requires(__class_or_enum<remove_cvref_t<_Tp>>),
                      typename(iterator_t<_Tp>),
                      requires(sentinel_for<decltype(_LIBCUDACXX_AUTO_CAST(end(__t))), iterator_t<_Tp>>)));

template <class _Tp>
_CCCL_CONCEPT __unqualified_end = _CCCL_FRAGMENT(__unqualified_end_, _Tp);
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

struct __fn
{
  _CCCL_TEMPLATE(class _Tp, size_t _Np)
  _CCCL_REQUIRES((sizeof(_Tp) >= 0)) // Disallow incomplete element types.
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Tp (&__t)[_Np]) const noexcept
  {
    return __t + _Np;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__member_end<_Tp>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Tp&& __t) const
    noexcept(noexcept(_LIBCUDACXX_AUTO_CAST(__t.end())))
  {
    return _LIBCUDACXX_AUTO_CAST(__t.end());
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__unqualified_end<_Tp>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Tp&& __t) const noexcept(noexcept(_LIBCUDACXX_AUTO_CAST(end(__t))))
  {
    return _LIBCUDACXX_AUTO_CAST(end(__t));
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES((!__member_end<_Tp>) _CCCL_AND(!__unqualified_end<_Tp>))
  void operator()(_Tp&&) const = delete;

#if _CCCL_COMPILER(MSVC, <, 19, 23)
  template <class _Tp>
  void operator()(_Tp (&&)[]) const = delete;

  template <class _Tp, size_t _Np>
  void operator()(_Tp (&&)[_Np]) const = delete;
#endif // _CCCL_COMPILER(MSVC, <, 19, 23)
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto end = __end::__fn{};
} // namespace __cpo

// [range.access.cbegin]

_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__cbegin)
struct __fn
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(is_lvalue_reference_v<_Tp&&>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Tp&& __t) const
    noexcept(noexcept(_CUDA_VRANGES::begin(static_cast<const remove_reference_t<_Tp>&>(__t))))
      -> decltype(_CUDA_VRANGES::begin(static_cast<const remove_reference_t<_Tp>&>(__t)))
  {
    return _CUDA_VRANGES::begin(static_cast<const remove_reference_t<_Tp>&>(__t));
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(is_rvalue_reference_v<_Tp&&>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Tp&& __t) const
    noexcept(noexcept(_CUDA_VRANGES::begin(static_cast<const _Tp&&>(__t))))
      -> decltype(_CUDA_VRANGES::begin(static_cast<const _Tp&&>(__t)))
  {
    return _CUDA_VRANGES::begin(static_cast<const _Tp&&>(__t));
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto cbegin = __cbegin::__fn{};
} // namespace __cpo

// [range.access.cend]

_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__cend)
struct __fn
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(is_lvalue_reference_v<_Tp&&>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Tp&& __t) const
    noexcept(noexcept(_CUDA_VRANGES::end(static_cast<const remove_reference_t<_Tp>&>(__t))))
      -> decltype(_CUDA_VRANGES::end(static_cast<const remove_reference_t<_Tp>&>(__t)))
  {
    return _CUDA_VRANGES::end(static_cast<const remove_reference_t<_Tp>&>(__t));
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(is_rvalue_reference_v<_Tp&&>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Tp&& __t) const
    noexcept(noexcept(_CUDA_VRANGES::end(static_cast<const _Tp&&>(__t))))
      -> decltype(_CUDA_VRANGES::end(static_cast<const _Tp&&>(__t)))
  {
    return _CUDA_VRANGES::end(static_cast<const _Tp&&>(__t));
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto cend = __cend::__fn{};
} // namespace __cpo

_LIBCUDACXX_END_NAMESPACE_RANGES

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___RANGES_ACCESS_H
