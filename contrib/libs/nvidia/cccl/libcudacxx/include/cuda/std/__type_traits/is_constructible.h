//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_IS_CONSTRUCTIBLE_H
#define _LIBCUDACXX___TYPE_IS_CONSTRUCTIBLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/conjunction.h>
#include <cuda/std/__type_traits/disjunction.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_base_of.h>
#include <cuda/std/__type_traits/is_destructible.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/negation.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

namespace __is_construct
{
struct __nat
{};
} // namespace __is_construct

// FIXME: This logic isn't awesome.
#if (!defined(_CCCL_BUILTIN_IS_CONSTRUCTIBLE) || defined(_LIBCUDACXX_TESTING_FALLBACK_IS_CONSTRUCTIBLE) \
     || defined(_LIBCUDACXX_USE_IS_CONSTRUCTIBLE_FALLBACK))

template <class _Tp, class... _Args>
struct __cccl_is_constructible;

template <class _To, class _From>
struct __is_invalid_base_to_derived_cast
{
  static_assert(is_reference<_To>::value, "Wrong specialization");
  using _RawFrom = remove_cvref_t<_From>;
  using _RawTo   = remove_cvref_t<_To>;
  static const bool value =
    _And<_IsNotSame<_RawFrom, _RawTo>, is_base_of<_RawFrom, _RawTo>, _Not<__cccl_is_constructible<_RawTo, _From>>>::value;
};

template <class _To, class _From>
struct __is_invalid_lvalue_to_rvalue_cast : false_type
{
  static_assert(is_reference<_To>::value, "Wrong specialization");
};

template <class _ToRef, class _FromRef>
struct __is_invalid_lvalue_to_rvalue_cast<_ToRef&&, _FromRef&>
{
  using _RawFrom = remove_cvref_t<_FromRef>;
  using _RawTo   = remove_cvref_t<_ToRef>;
  static const bool value =
    _And<_Not<is_function<_RawTo>>, _Or<_IsSame<_RawFrom, _RawTo>, is_base_of<_RawTo, _RawFrom>>>::value;
};

struct __is_constructible_helper
{
  template <class _To>
  _CCCL_API inline static void __eat(_To);

  // This overload is needed to work around a Clang bug that disallows
  // static_cast<T&&>(e) for non-reference-compatible types.
  // Example: static_cast<int&&>(declval<double>());
  // NOTE: The static_cast implementation below is required to support
  //  classes with explicit conversion operators.
  template <class _To, class _From, class = decltype(__eat<_To>(_CUDA_VSTD::declval<_From>()))>
  _CCCL_API inline static true_type __test_cast(int);

  template <class _To, class _From, class = decltype(static_cast<_To>(_CUDA_VSTD::declval<_From>()))>
  _CCCL_API inline static integral_constant<
    bool,
    !__is_invalid_base_to_derived_cast<_To, _From>::value && !__is_invalid_lvalue_to_rvalue_cast<_To, _From>::value>
  __test_cast(long);

  template <class, class>
  _CCCL_API inline static false_type __test_cast(...);

  template <class _Tp, class... _Args, class = decltype(_Tp(_CUDA_VSTD::declval<_Args>()...))>
  _CCCL_API inline static true_type __test_nary(int);
  template <class _Tp, class...>
  _CCCL_API inline static false_type __test_nary(...);

  template <class _Tp, class _A0, class = decltype(::new _Tp(_CUDA_VSTD::declval<_A0>()))>
  _CCCL_API inline static is_destructible<_Tp> __test_unary(int);
  template <class, class>
  _CCCL_API inline static false_type __test_unary(...);
};

template <class _Tp, bool = is_void<_Tp>::value>
struct __is_default_constructible : decltype(__is_constructible_helper::__test_nary<_Tp>(0))
{};

template <class _Tp>
struct __is_default_constructible<_Tp, true> : false_type
{};

template <class _Tp>
struct __is_default_constructible<_Tp[], false> : false_type
{};

template <class _Tp, size_t _Nx>
struct __is_default_constructible<_Tp[_Nx], false> : __is_default_constructible<remove_all_extents_t<_Tp>>
{};

template <class _Tp, class... _Args>
struct __cccl_is_constructible
{
  static_assert(sizeof...(_Args) > 1, "Wrong specialization");
  using type = decltype(__is_constructible_helper::__test_nary<_Tp, _Args...>(0));
};

template <class _Tp>
struct __cccl_is_constructible<_Tp> : __is_default_constructible<_Tp>
{};

template <class _Tp, class _A0>
struct __cccl_is_constructible<_Tp, _A0> : public decltype(__is_constructible_helper::__test_unary<_Tp, _A0>(0))
{};

template <class _Tp, class _A0>
struct __cccl_is_constructible<_Tp&, _A0> : public decltype(__is_constructible_helper::__test_cast<_Tp&, _A0>(0))
{};

template <class _Tp, class _A0>
struct __cccl_is_constructible<_Tp&&, _A0> : public decltype(__is_constructible_helper::__test_cast<_Tp&&, _A0>(0))
{};

#endif

#if defined(_CCCL_BUILTIN_IS_CONSTRUCTIBLE) && !defined(_LIBCUDACXX_USE_IS_CONSTRUCTIBLE_FALLBACK)
template <class _Tp, class... _Args>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
is_constructible : public integral_constant<bool, _CCCL_BUILTIN_IS_CONSTRUCTIBLE(_Tp, _Args...)>
{};

template <class _Tp, class... _Args>
inline constexpr bool is_constructible_v = _CCCL_BUILTIN_IS_CONSTRUCTIBLE(_Tp, _Args...);

#else
template <class _Tp, class... _Args>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_constructible : public __cccl_is_constructible<_Tp, _Args...>::type
{};

template <class _Tp, class... _Args>
inline constexpr bool is_constructible_v = is_constructible<_Tp, _Args...>::value;

#endif // defined(_CCCL_BUILTIN_IS_CONSTRUCTIBLE) && !defined(_LIBCUDACXX_USE_IS_CONSTRUCTIBLE_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_IS_CONSTRUCTIBLE_H
