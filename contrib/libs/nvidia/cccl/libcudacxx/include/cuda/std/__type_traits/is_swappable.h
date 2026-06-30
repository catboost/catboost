//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_SWAPPABLE_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_SWAPPABLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/add_lvalue_reference.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/disjunction.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_move_assignable.h>
#include <cuda/std/__type_traits/is_move_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/is_referenceable.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/nat.h>
#include <cuda/std/__type_traits/type_identity.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// We need to detect whether there is already a free function swap that would end up being ambiguous.
// This can happen when a type pulls in both namespace std and namespace cuda::std via ADL.
// In that case we are always safe to just not do anything because that type must be host only.
// However, we must be careful to ensure that we still create the overload if there is just a hidden friend swap
namespace __detect_hidden_friend_swap
{
// This will intentionally create an ambiguity with std::swap if that is find-able by ADL. But it will not interfere
// with hidden friend swap
template <class _Tp>
_CCCL_HOST_DEVICE void swap(_Tp&, _Tp&);

struct __hidden_friend_swap_found
{};

template <class _Tp>
_CCCL_API inline auto __swap(_Tp& __lhs, _Tp& __rhs) -> decltype(swap(__lhs, __rhs));
_CCCL_API inline auto __swap(...) -> __hidden_friend_swap_found;
template <class _Tp>
struct __has_hidden_friend_swap
    : is_same<decltype(__detect_hidden_friend_swap::__swap(_CUDA_VSTD::declval<_Tp&>(), _CUDA_VSTD::declval<_Tp&>())),
              void>
{};
} // namespace __detect_hidden_friend_swap

namespace __detect_adl_swap
{

template <class _Tp>
void swap(_Tp&, _Tp&) = delete;

struct __no_adl_swap_found
{};
template <class _Tp>
_CCCL_API inline auto __swap(_Tp& __lhs, _Tp& __rhs) -> decltype(swap(__lhs, __rhs));
_CCCL_API inline auto __swap(...) -> __no_adl_swap_found;
template <class _Tp>
struct __has_no_adl_swap
    : is_same<decltype(__detect_adl_swap::__swap(_CUDA_VSTD::declval<_Tp&>(), _CUDA_VSTD::declval<_Tp&>())),
              __no_adl_swap_found>
{};
template <class _Tp, size_t _Np>
struct __has_no_adl_swap_array
    : is_same<
        decltype(__detect_adl_swap::__swap(_CUDA_VSTD::declval<_Tp (&)[_Np]>(), _CUDA_VSTD::declval<_Tp (&)[_Np]>())),
        __no_adl_swap_found>
{};

// We should only define swap if there is no ADL found function or it is a hidden friend
template <class _Tp>
struct __can_define_swap : _Or<__has_no_adl_swap<_Tp>, __detect_hidden_friend_swap::__has_hidden_friend_swap<_Tp>>
{};
} // namespace __detect_adl_swap

template <class _Tp>
struct __is_swappable;
template <class _Tp>
struct __is_nothrow_swappable;

template <class _Tp>
using __swap_result_t _CCCL_NODEBUG_ALIAS =
  enable_if_t<__detect_adl_swap::__can_define_swap<_Tp>::value && _CCCL_TRAIT(is_move_constructible, _Tp)
              && _CCCL_TRAIT(is_move_assignable, _Tp)>;

// we use type_identity_t<_Tp> as second parameter, to avoid ambiguity with std::swap, which will thus be preferred by
// overload resolution (which is ok since std::swap is only considered when explicitly called, or found by ADL for types
// from std::)
template <class _Tp>
_CCCL_API constexpr __swap_result_t<_Tp> swap(_Tp& __x, type_identity_t<_Tp>& __y) noexcept(
  _CCCL_TRAIT(is_nothrow_move_constructible, _Tp) && _CCCL_TRAIT(is_nothrow_move_assignable, _Tp));

template <class _Tp, size_t _Np>
_CCCL_API constexpr enable_if_t<__detect_adl_swap::__has_no_adl_swap_array<_Tp, _Np>::value && __is_swappable<_Tp>::value>
  swap(_Tp (&__a)[_Np], _Tp (&__b)[_Np]) noexcept(__is_nothrow_swappable<_Tp>::value);

namespace __detail
{
// ALL generic swap overloads MUST already have a declaration available at this point.

template <class _Tp, class _Up = _Tp, bool _NotVoid = !_CCCL_TRAIT(is_void, _Tp) && !_CCCL_TRAIT(is_void, _Up)>
struct __swappable_with
{
  template <class _LHS, class _RHS>
  _CCCL_API inline static decltype(swap(_CUDA_VSTD::declval<_LHS>(), _CUDA_VSTD::declval<_RHS>())) __test_swap(int);
  template <class, class>
  _CCCL_API inline static __nat __test_swap(long);

  // Extra parens are needed for the C++03 definition of decltype.
  using __swap1 = decltype((__test_swap<_Tp, _Up>(0)));
  using __swap2 = decltype((__test_swap<_Up, _Tp>(0)));

  static const bool value = _IsNotSame<__swap1, __nat>::value && _IsNotSame<__swap2, __nat>::value;
};

template <class _Tp, class _Up>
struct __swappable_with<_Tp, _Up, false> : false_type
{};

template <class _Tp, class _Up = _Tp, bool _Swappable = __swappable_with<_Tp, _Up>::value>
struct __nothrow_swappable_with
{
  static const bool value = noexcept(swap(_CUDA_VSTD::declval<_Tp>(), _CUDA_VSTD::declval<_Up>()))
                         && noexcept(swap(_CUDA_VSTD::declval<_Up>(), _CUDA_VSTD::declval<_Tp>()));
};

template <class _Tp, class _Up>
struct __nothrow_swappable_with<_Tp, _Up, false> : false_type
{};

} // namespace __detail

template <class _Tp>
struct __is_swappable : public integral_constant<bool, __detail::__swappable_with<_Tp&>::value>
{};

template <class _Tp>
struct __is_nothrow_swappable : public integral_constant<bool, __detail::__nothrow_swappable_with<_Tp&>::value>
{};

template <class _Tp, class _Up>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
is_swappable_with : public integral_constant<bool, __detail::__swappable_with<_Tp, _Up>::value>
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_swappable
    : public conditional_t<__cccl_is_referenceable<_Tp>::value,
                           is_swappable_with<add_lvalue_reference_t<_Tp>, add_lvalue_reference_t<_Tp>>,
                           false_type>
{};

template <class _Tp, class _Up>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
is_nothrow_swappable_with : public integral_constant<bool, __detail::__nothrow_swappable_with<_Tp, _Up>::value>
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_nothrow_swappable
    : public conditional_t<__cccl_is_referenceable<_Tp>::value,
                           is_nothrow_swappable_with<add_lvalue_reference_t<_Tp>, add_lvalue_reference_t<_Tp>>,
                           false_type>
{};

template <class _Tp, class _Up>
inline constexpr bool is_swappable_with_v = is_swappable_with<_Tp, _Up>::value;

template <class _Tp>
inline constexpr bool is_swappable_v = is_swappable<_Tp>::value;

template <class _Tp, class _Up>
inline constexpr bool is_nothrow_swappable_with_v = is_nothrow_swappable_with<_Tp, _Up>::value;

template <class _Tp>
inline constexpr bool is_nothrow_swappable_v = is_nothrow_swappable<_Tp>::value;

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_SWAPPABLE_H
