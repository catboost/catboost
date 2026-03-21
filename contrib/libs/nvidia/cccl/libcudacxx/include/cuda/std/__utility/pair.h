//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___UTILITY_PAIR_H
#define _LIBCUDACXX___UTILITY_PAIR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#  include <cuda/std/__compare/common_comparison_category.h>
#  include <cuda/std/__compare/synth_three_way.h>
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

#include <cuda/std/__functional/unwrap_ref.h>
#include <cuda/std/__fwd/get.h>
#include <cuda/std/__fwd/pair.h>
#include <cuda/std/__fwd/tuple.h>
#include <cuda/std/__tuple_dir/sfinae_helpers.h>
#include <cuda/std/__tuple_dir/structured_bindings.h>
#include <cuda/std/__tuple_dir/tuple_element.h>
#include <cuda/std/__tuple_dir/tuple_indices.h>
#include <cuda/std/__tuple_dir/tuple_size.h>
#include <cuda/std/__type_traits/common_reference.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_assignable.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_copy_assignable.h>
#include <cuda/std/__type_traits/is_default_constructible.h>
#include <cuda/std/__type_traits/is_implicitly_default_constructible.h>
#include <cuda/std/__type_traits/is_move_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_copy_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_swappable.h>
#include <cuda/std/__type_traits/make_const_lvalue_ref.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/piecewise_construct.h>
#include <cuda/std/cstddef>

// Provide compatibility between `std::pair` and `cuda::std::pair`
#if !_CCCL_COMPILER(NVRTC)
#  include <utility>
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

struct __invalid_pair_constraints
{
  static constexpr bool __implicit_constructible = false;
  static constexpr bool __explicit_constructible = false;
  static constexpr bool __enable_assign          = false;
};

template <class _T1, class _T2>
struct __pair_constraints
{
  static constexpr bool __implicit_default_constructible =
    __is_implicitly_default_constructible<_T1>::value && __is_implicitly_default_constructible<_T2>::value;

  static constexpr bool __explicit_default_constructible =
    !__implicit_default_constructible && _CCCL_TRAIT(is_default_constructible, _T1)
    && _CCCL_TRAIT(is_default_constructible, _T2);

  static constexpr bool __explicit_constructible_from_elements =
    _CCCL_TRAIT(is_copy_constructible, _T1) && _CCCL_TRAIT(is_copy_constructible, _T2)
    && (!_CCCL_TRAIT(is_convertible, __make_const_lvalue_ref<_T1>, _T1)
        || !_CCCL_TRAIT(is_convertible, __make_const_lvalue_ref<_T2>, _T2));

  static constexpr bool __implicit_constructible_from_elements =
    _CCCL_TRAIT(is_copy_constructible, _T1) && _CCCL_TRAIT(is_copy_constructible, _T2)
    && _CCCL_TRAIT(is_convertible, __make_const_lvalue_ref<_T1>, _T1)
    && _CCCL_TRAIT(is_convertible, __make_const_lvalue_ref<_T2>, _T2);

  template <class _U1, class _U2>
  struct __constructible
  {
    static constexpr bool __explicit_constructible =
      _CCCL_TRAIT(is_constructible, _T1, _U1) && _CCCL_TRAIT(is_constructible, _T2, _U2)
      && (!_CCCL_TRAIT(is_convertible, _U1, _T1) || !_CCCL_TRAIT(is_convertible, _U2, _T2));

    static constexpr bool __implicit_constructible =
      _CCCL_TRAIT(is_constructible, _T1, _U1) && _CCCL_TRAIT(is_constructible, _T2, _U2)
      && _CCCL_TRAIT(is_convertible, _U1, _T1) && _CCCL_TRAIT(is_convertible, _U2, _T2);
  };

  template <class _U1, class _U2>
  struct __assignable
  {
    static constexpr bool __enable_assign =
      _CCCL_TRAIT(is_assignable, _T1&, _U1) && _CCCL_TRAIT(is_assignable, _T2&, _U2);
  };
};

// base class to ensure `is_trivially_copyable` when possible
template <class _T1, class _T2, bool = __must_synthesize_assignment_v<_T1> || __must_synthesize_assignment_v<_T2>>
struct __pair_base
{
  _T1 first;
  _T2 second;

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Constraints                                               = __pair_constraints<_T1, _T2>,
            enable_if_t<_Constraints::__explicit_default_constructible, int> = 0>
  _CCCL_API explicit constexpr __pair_base() noexcept(
    _CCCL_TRAIT(is_nothrow_default_constructible, _T1) && _CCCL_TRAIT(is_nothrow_default_constructible, _T2))
      : first()
      , second()
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Constraints                                               = __pair_constraints<_T1, _T2>,
            enable_if_t<_Constraints::__implicit_default_constructible, int> = 0>
  _CCCL_API constexpr __pair_base() noexcept(_CCCL_TRAIT(is_nothrow_default_constructible, _T1)
                                             && _CCCL_TRAIT(is_nothrow_default_constructible, _T2))
      : first()
      , second()
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class _U1, class _U2>
  _CCCL_API constexpr __pair_base(_U1&& __t1, _U2&& __t2) noexcept(
    _CCCL_TRAIT(is_nothrow_constructible, _T1, _U1) && _CCCL_TRAIT(is_nothrow_constructible, _T2, _U2))
      : first(_CUDA_VSTD::forward<_U1>(__t1))
      , second(_CUDA_VSTD::forward<_U2>(__t2))
  {}

protected:
  template <class... _Args1, class... _Args2, size_t... _I1, size_t... _I2>
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 __pair_base(
    piecewise_construct_t,
    tuple<_Args1...>& __first_args,
    tuple<_Args2...>& __second_args,
    __tuple_indices<_I1...>,
    __tuple_indices<_I2...>);
};

template <class _T1, class _T2>
struct __pair_base<_T1, _T2, true>
{
  _T1 first;
  _T2 second;

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Constraints                                               = __pair_constraints<_T1, _T2>,
            enable_if_t<_Constraints::__explicit_default_constructible, int> = 0>
  _CCCL_API explicit constexpr __pair_base() noexcept(
    _CCCL_TRAIT(is_nothrow_default_constructible, _T1) && _CCCL_TRAIT(is_nothrow_default_constructible, _T2))
      : first()
      , second()
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Constraints                                               = __pair_constraints<_T1, _T2>,
            enable_if_t<_Constraints::__implicit_default_constructible, int> = 0>
  _CCCL_API constexpr __pair_base() noexcept(_CCCL_TRAIT(is_nothrow_default_constructible, _T1)
                                             && _CCCL_TRAIT(is_nothrow_default_constructible, _T2))
      : first()
      , second()
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HIDE_FROM_ABI constexpr __pair_base(const __pair_base&) = default;
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HIDE_FROM_ABI constexpr __pair_base(__pair_base&&) = default;

  // We need to ensure that a reference type, which would inhibit the implicit copy assignment still works
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr __pair_base& operator=(
    conditional_t<_CCCL_TRAIT(is_copy_assignable, _T1) && _CCCL_TRAIT(is_copy_assignable, _T2), __pair_base, __nat> const&
      __p) noexcept(_CCCL_TRAIT(is_nothrow_copy_assignable, _T1) && _CCCL_TRAIT(is_nothrow_copy_assignable, _T2))
  {
    first  = __p.first;
    second = __p.second;
    return *this;
  }

  // We need to ensure that a reference type, which would inhibit the implicit move assignment still works
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr __pair_base& operator=(
    conditional_t<_CCCL_TRAIT(is_move_assignable, _T1) && _CCCL_TRAIT(is_move_assignable, _T2), __pair_base, __nat>&&
      __p) noexcept(_CCCL_TRAIT(is_nothrow_move_assignable, _T1) && _CCCL_TRAIT(is_nothrow_move_assignable, _T2))
  {
    first  = _CUDA_VSTD::forward<_T1>(__p.first);
    second = _CUDA_VSTD::forward<_T2>(__p.second);
    return *this;
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class _U1, class _U2>
  _CCCL_API constexpr __pair_base(_U1&& __t1, _U2&& __t2) noexcept(
    _CCCL_TRAIT(is_nothrow_constructible, _T1, _U1) && _CCCL_TRAIT(is_nothrow_constructible, _T2, _U2))
      : first(_CUDA_VSTD::forward<_U1>(__t1))
      , second(_CUDA_VSTD::forward<_U2>(__t2))
  {}

protected:
  template <class... _Args1, class... _Args2, size_t... _I1, size_t... _I2>
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 __pair_base(
    piecewise_construct_t,
    tuple<_Args1...>& __first_args,
    tuple<_Args2...>& __second_args,
    __tuple_indices<_I1...>,
    __tuple_indices<_I2...>);
};

template <class _T1, class _T2>
struct _CCCL_TYPE_VISIBILITY_DEFAULT pair : public __pair_base<_T1, _T2>
{
  using __base = __pair_base<_T1, _T2>;

  using first_type  = _T1;
  using second_type = _T2;

  template <class _Constraints                                               = __pair_constraints<_T1, _T2>,
            enable_if_t<_Constraints::__explicit_default_constructible, int> = 0>
  _CCCL_API explicit constexpr pair() noexcept(_CCCL_TRAIT(is_nothrow_default_constructible, _T1)
                                               && _CCCL_TRAIT(is_nothrow_default_constructible, _T2))
      : __base()
  {}

  template <class _Constraints                                               = __pair_constraints<_T1, _T2>,
            enable_if_t<_Constraints::__implicit_default_constructible, int> = 0>
  _CCCL_API constexpr pair() noexcept(_CCCL_TRAIT(is_nothrow_default_constructible, _T1)
                                      && _CCCL_TRAIT(is_nothrow_default_constructible, _T2))
      : __base()
  {}

  // element wise constructors
  template <class _Constraints                                                     = __pair_constraints<_T1, _T2>,
            enable_if_t<_Constraints::__explicit_constructible_from_elements, int> = 0>
  _CCCL_API explicit constexpr pair(const _T1& __t1, const _T2& __t2) noexcept(
    _CCCL_TRAIT(is_nothrow_copy_constructible, _T1) && _CCCL_TRAIT(is_nothrow_copy_constructible, _T2))
      : __base(__t1, __t2)
  {}

  template <class _Constraints                                                     = __pair_constraints<_T1, _T2>,
            enable_if_t<_Constraints::__implicit_constructible_from_elements, int> = 0>
  _CCCL_API constexpr pair(const _T1& __t1, const _T2& __t2) noexcept(
    _CCCL_TRAIT(is_nothrow_copy_constructible, _T1) && _CCCL_TRAIT(is_nothrow_copy_constructible, _T2))
      : __base(__t1, __t2)
  {}

  template <class _U1, class _U2>
  using __pair_constructible = typename __pair_constraints<_T1, _T2>::template __constructible<_U1, _U2>;

  template <class _U1                                                = _T1,
            class _U2                                                = _T2,
            class _Constraints                                       = __pair_constructible<_U1, _U2>,
            enable_if_t<_Constraints::__explicit_constructible, int> = 0>
  _CCCL_API explicit constexpr pair(_U1&& __u1, _U2&& __u2) noexcept(
    _CCCL_TRAIT(is_nothrow_constructible, _T1, _U1) && _CCCL_TRAIT(is_nothrow_constructible, _T2, _U2))
      : __base(_CUDA_VSTD::forward<_U1>(__u1), _CUDA_VSTD::forward<_U2>(__u2))
  {}

  template <class _U1                                                = _T1,
            class _U2                                                = _T2,
            class _Constraints                                       = __pair_constructible<_U1, _U2>,
            enable_if_t<_Constraints::__implicit_constructible, int> = 0>
  _CCCL_API constexpr pair(_U1&& __u1, _U2&& __u2) noexcept(
    _CCCL_TRAIT(is_nothrow_constructible, _T1, _U1) && _CCCL_TRAIT(is_nothrow_constructible, _T2, _U2))
      : __base(_CUDA_VSTD::forward<_U1>(__u1), _CUDA_VSTD::forward<_U2>(__u2))
  {}

  template <class... _Args1, class... _Args2>
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20
  pair(piecewise_construct_t __pc, tuple<_Args1...> __first_args, tuple<_Args2...> __second_args) noexcept(
    (is_nothrow_constructible<_T1, _Args1...>::value && is_nothrow_constructible<_T2, _Args2...>::value))
      : __base(__pc,
               __first_args,
               __second_args,
               __make_tuple_indices_t<sizeof...(_Args1)>(),
               __make_tuple_indices_t<sizeof...(_Args2)>())
  {}

  // copy and move constructors
  _CCCL_HIDE_FROM_ABI pair(pair const&) = default;
  _CCCL_HIDE_FROM_ABI pair(pair&&)      = default;

  template <class _U1                                                = _T1,
            class _U2                                                = _T2,
            class _Constraints                                       = __pair_constructible<const _U1&, const _U2&>,
            enable_if_t<_Constraints::__explicit_constructible, int> = 0>
  _CCCL_API explicit constexpr pair(const pair<_U1, _U2>& __p) noexcept(
    _CCCL_TRAIT(is_nothrow_constructible, _T1, const _U1&) && _CCCL_TRAIT(is_nothrow_constructible, _T2, const _U2&))
      : __base(__p.first, __p.second)
  {}

  template <class _U1                                                = _T1,
            class _U2                                                = _T2,
            class _Constraints                                       = __pair_constructible<const _U1&, const _U2&>,
            enable_if_t<_Constraints::__implicit_constructible, int> = 0>
  _CCCL_API constexpr pair(const pair<_U1, _U2>& __p) noexcept(
    _CCCL_TRAIT(is_nothrow_constructible, _T1, const _U1&) && _CCCL_TRAIT(is_nothrow_constructible, _T2, const _U2&))
      : __base(__p.first, __p.second)
  {}

  // move constructors
  template <class _U1                                                = _T1,
            class _U2                                                = _T2,
            class _Constraints                                       = __pair_constructible<_U1, _U2>,
            enable_if_t<_Constraints::__explicit_constructible, int> = 0>
  _CCCL_API explicit constexpr pair(pair<_U1, _U2>&& __p) noexcept(
    _CCCL_TRAIT(is_nothrow_constructible, _T1, _U1) && _CCCL_TRAIT(is_nothrow_constructible, _T2, _U2))
      : __base(_CUDA_VSTD::forward<_U1>(__p.first), _CUDA_VSTD::forward<_U2>(__p.second))
  {}

  template <class _U1                                                = _T1,
            class _U2                                                = _T2,
            class _Constraints                                       = __pair_constructible<_U1, _U2>,
            enable_if_t<_Constraints::__implicit_constructible, int> = 0>
  _CCCL_API constexpr pair(pair<_U1, _U2>&& __p) noexcept(
    _CCCL_TRAIT(is_nothrow_constructible, _T1, _U1) && _CCCL_TRAIT(is_nothrow_constructible, _T2, _U2))
      : __base(_CUDA_VSTD::forward<_U1>(__p.first), _CUDA_VSTD::forward<_U2>(__p.second))
  {}

  // std compatibility
#if !_CCCL_COMPILER(NVRTC)
  template <class _U1,
            class _U2,
            class _Constraints                                       = __pair_constructible<const _U1&, const _U2&>,
            enable_if_t<_Constraints::__explicit_constructible, int> = 0>
  _CCCL_HOST _CCCL_API explicit constexpr pair(const ::std::pair<_U1, _U2>& __p) noexcept(
    _CCCL_TRAIT(is_nothrow_constructible, _T1, const _U1&) && _CCCL_TRAIT(is_nothrow_constructible, _T2, const _U2&))
      : __base(__p.first, __p.second)
  {}

  template <class _U1,
            class _U2,
            class _Constraints                                       = __pair_constructible<const _U1&, const _U2&>,
            enable_if_t<_Constraints::__implicit_constructible, int> = 0>
  _CCCL_HOST _CCCL_API constexpr pair(const ::std::pair<_U1, _U2>& __p) noexcept(
    _CCCL_TRAIT(is_nothrow_constructible, _T1, const _U1&) && _CCCL_TRAIT(is_nothrow_constructible, _T2, const _U2&))
      : __base(__p.first, __p.second)
  {}

  template <class _U1,
            class _U2,
            class _Constraints                                       = __pair_constructible<_U1, _U2>,
            enable_if_t<_Constraints::__explicit_constructible, int> = 0>
  _CCCL_HOST _CCCL_API explicit constexpr pair(::std::pair<_U1, _U2>&& __p) noexcept(
    _CCCL_TRAIT(is_nothrow_constructible, _T1, _U1) && _CCCL_TRAIT(is_nothrow_constructible, _T2, _U2))
      : __base(_CUDA_VSTD::forward<_U1>(__p.first), _CUDA_VSTD::forward<_U2>(__p.second))
  {}

  template <class _U1,
            class _U2,
            class _Constraints                                       = __pair_constructible<_U1, _U2>,
            enable_if_t<_Constraints::__implicit_constructible, int> = 0>
  _CCCL_HOST _CCCL_API constexpr pair(::std::pair<_U1, _U2>&& __p) noexcept(
    _CCCL_TRAIT(is_nothrow_constructible, _T1, _U1) && _CCCL_TRAIT(is_nothrow_constructible, _T2, _U2))
      : __base(_CUDA_VSTD::forward<_U1>(__p.first), _CUDA_VSTD::forward<_U2>(__p.second))
  {}
#endif // !_CCCL_COMPILER(NVRTC)

  // assignments
  _CCCL_HIDE_FROM_ABI pair& operator=(const pair&) = default;
  _CCCL_HIDE_FROM_ABI pair& operator=(pair&&)      = default;

  template <class _U1,
            class _U2,
            class _Constraints = typename __pair_constraints<_T1, _T2>::template __assignable<const _U1&, const _U2&>,
            enable_if_t<_Constraints::__enable_assign, int> = 0>
  _CCCL_API constexpr pair& operator=(const pair<_U1, _U2>& __p) noexcept(
    _CCCL_TRAIT(is_nothrow_assignable, _T1, const _U1&) && _CCCL_TRAIT(is_nothrow_assignable, _T2, const _U2&))
  {
    this->first  = __p.first;
    this->second = __p.second;
    return *this;
  }

  template <class _U1,
            class _U2,
            class _Constraints = typename __pair_constraints<_T1, _T2>::template __assignable<_U1, _U2>,
            enable_if_t<_Constraints::__enable_assign, int> = 0>
  _CCCL_API constexpr pair& operator=(pair<_U1, _U2>&& __p) noexcept(
    _CCCL_TRAIT(is_nothrow_assignable, _T1, _U1) && _CCCL_TRAIT(is_nothrow_assignable, _T2, _U2))
  {
    this->first  = _CUDA_VSTD::forward<_U1>(__p.first);
    this->second = _CUDA_VSTD::forward<_U2>(__p.second);
    return *this;
  }

  // std assignments
#if !_CCCL_COMPILER(NVRTC)
  template <class _UT1 = _T1, enable_if_t<is_copy_assignable<_UT1>::value && is_copy_assignable<_T2>::value, int> = 0>
  _CCCL_HOST constexpr pair& operator=(::std::pair<_T1, _T2> const& __p) noexcept(
    _CCCL_TRAIT(is_nothrow_copy_assignable, _T1) && _CCCL_TRAIT(is_nothrow_copy_assignable, _T2))
  {
    this->first  = __p.first;
    this->second = __p.second;
    return *this;
  }

  template <class _UT1 = _T1, enable_if_t<is_move_assignable<_UT1>::value && is_move_assignable<_T2>::value, int> = 0>
  _CCCL_HOST constexpr pair& operator=(::std::pair<_T1, _T2>&& __p) noexcept(
    _CCCL_TRAIT(is_nothrow_copy_assignable, _T1) && _CCCL_TRAIT(is_nothrow_copy_assignable, _T2))
  {
    this->first  = _CUDA_VSTD::forward<_T1>(__p.first);
    this->second = _CUDA_VSTD::forward<_T2>(__p.second);
    return *this;
  }
#endif // !_CCCL_COMPILER(NVRTC)

#if _CCCL_STD_VER >= 2023
  _CCCL_API constexpr const pair& operator=(pair const& __p) const
    noexcept(_CCCL_TRAIT(is_nothrow_copy_assignable, const _T1) && _CCCL_TRAIT(is_nothrow_copy_assignable, const _T2))
    requires(is_copy_assignable_v<const _T1> && is_copy_assignable_v<const _T2>)
  {
    this->first  = __p.first;
    this->second = __p.second;
    return *this;
  }

#  if !_CCCL_COMPILER(NVRTC)
  _CCCL_API inline _CCCL_HOST constexpr const pair& operator=(::std::pair<_T1, _T2> const& __p) const
    noexcept(_CCCL_TRAIT(is_nothrow_copy_assignable, const _T1) && _CCCL_TRAIT(is_nothrow_copy_assignable, const _T2))
    requires(is_copy_assignable_v<const _T1> && is_copy_assignable_v<const _T2>)
  {
    this->first  = __p.first;
    this->second = __p.second;
    return *this;
  }
#  endif // !_CCCL_COMPILER(NVRTC)

  _CCCL_API constexpr const pair& operator=(pair&& __p) const
    noexcept(_CCCL_TRAIT(is_nothrow_assignable, const _T1&, _T1) && _CCCL_TRAIT(is_nothrow_assignable, const _T2&, _T2))
    requires(is_assignable_v<const _T1&, _T1> && is_assignable_v<const _T2&, _T2>)
  {
    this->first  = _CUDA_VSTD::forward<_T1>(__p.first);
    this->second = _CUDA_VSTD::forward<_T2>(__p.second);
    return *this;
  }

#  if !_CCCL_COMPILER(NVRTC)
  _CCCL_API inline _CCCL_HOST constexpr const pair& operator=(::std::pair<_T1, _T2>&& __p) const
    noexcept(_CCCL_TRAIT(is_nothrow_assignable, const _T1&, _T1) && _CCCL_TRAIT(is_nothrow_assignable, const _T2&, _T2))
    requires(is_assignable_v<const _T1&, _T1> && is_assignable_v<const _T2&, _T2>)
  {
    this->first  = _CUDA_VSTD::forward<_T1>(__p.first);
    this->second = _CUDA_VSTD::forward<_T2>(__p.second);
    return *this;
  }
#  endif // !_CCCL_COMPILER(NVRTC)

  template <class _U1, class _U2>
  _CCCL_API constexpr const pair& operator=(const pair<_U1, _U2>& __p) const
    requires(is_assignable_v<const _T1&, const _U1&> && is_assignable_v<const _T2&, const _U2&>)
  {
    this->first  = __p.first;
    this->second = __p.second;
    return *this;
  }

#  if !_CCCL_COMPILER(NVRTC)
  template <class _U1, class _U2>
  _CCCL_API inline _CCCL_HOST constexpr const pair& operator=(const ::std::pair<_U1, _U2>& __p) const
    requires(is_assignable_v<const _T1&, const _U1&> && is_assignable_v<const _T2&, const _U2&>)
  {
    this->first  = __p.first;
    this->second = __p.second;
    return *this;
  }
#  endif // !_CCCL_COMPILER(NVRTC)

  template <class _U1, class _U2>
  _CCCL_API constexpr const pair& operator=(pair<_U1, _U2>&& __p) const
    requires(is_assignable_v<const _T1&, _U1> && is_assignable_v<const _T2&, _U2>)
  {
    this->first  = _CUDA_VSTD::forward<_U1>(__p.first);
    this->second = _CUDA_VSTD::forward<_U2>(__p.second);
    return *this;
  }

#  if !_CCCL_COMPILER(NVRTC)
  template <class _U1, class _U2>
  _CCCL_API inline _CCCL_HOST constexpr const pair& operator=(::std::pair<_U1, _U2>&& __p) const
    requires(is_assignable_v<const _T1&, _U1> && is_assignable_v<const _T2&, _U2>)
  {
    this->first  = _CUDA_VSTD::forward<_U1>(__p.first);
    this->second = _CUDA_VSTD::forward<_U2>(__p.second);
    return *this;
  }
#  endif // !_CCCL_COMPILER(NVRTC)
#endif // _CCCL_STD_VER >= 2023

  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 void
  swap(pair& __p) noexcept(__is_nothrow_swappable<_T1>::value && __is_nothrow_swappable<_T2>::value)
  {
    using _CUDA_VSTD::swap;
    swap(this->first, __p.first);
    swap(this->second, __p.second);
  }

#if _CCCL_STD_VER >= 2023
  _CCCL_API constexpr void swap(const pair& __p) const
    noexcept(__is_nothrow_swappable<const _T1>::value && __is_nothrow_swappable<const _T2>::value)
  {
    using _CUDA_VSTD::swap;
    swap(this->first, __p.first);
    swap(this->second, __p.second);
  }
#endif // _CCCL_STD_VER >= 2023

#if !_CCCL_COMPILER(NVRTC)
  _CCCL_HOST constexpr operator ::std::pair<_T1, _T2>() const
  {
    return {this->first, this->second};
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

template <class _T1, class _T2>
_CCCL_HOST_DEVICE pair(_T1, _T2) -> pair<_T1, _T2>;

// [pairs.spec], specialized algorithms

_CCCL_EXEC_CHECK_DISABLE
template <class _T1, class _T2>
_CCCL_API constexpr bool operator==(const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y)
{
  return __x.first == __y.first && __x.second == __y.second;
}

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

_CCCL_EXEC_CHECK_DISABLE
template <class _T1, class _T2>
_CCCL_API constexpr common_comparison_category_t<__synth_three_way_result<_T1>, __synth_three_way_result<_T2>>
operator<=>(const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y)
{
  if (auto __c = _CUDA_VSTD::__synth_three_way(__x.first, __y.first); __c != 0)
  {
    return __c;
  }
  return _CUDA_VSTD::__synth_three_way(__x.second, __y.second);
}

#else // ^^^ _LIBCUDACXX_HAS_SPACESHIP_OPERATOR() ^^^ / vvv !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR() vvv

_CCCL_EXEC_CHECK_DISABLE
template <class _T1, class _T2>
_CCCL_API constexpr bool operator!=(const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y)
{
  return !(__x == __y);
}

_CCCL_EXEC_CHECK_DISABLE
template <class _T1, class _T2>
_CCCL_API constexpr bool operator<(const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y)
{
  return __x.first < __y.first || (!(__y.first < __x.first) && __x.second < __y.second);
}

_CCCL_EXEC_CHECK_DISABLE
template <class _T1, class _T2>
_CCCL_API constexpr bool operator>(const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y)
{
  return __y < __x;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _T1, class _T2>
_CCCL_API constexpr bool operator>=(const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y)
{
  return !(__x < __y);
}

_CCCL_EXEC_CHECK_DISABLE
template <class _T1, class _T2>
_CCCL_API constexpr bool operator<=(const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y)
{
  return !(__y < __x);
}

#endif // !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

#if _CCCL_STD_VER >= 2023
template <class _T1, class _T2, class _U1, class _U2, template <class> class _TQual, template <class> class _UQual>
  requires requires {
    typename pair<common_reference_t<_TQual<_T1>, _UQual<_U1>>, common_reference_t<_TQual<_T2>, _UQual<_U2>>>;
  }
struct basic_common_reference<pair<_T1, _T2>, pair<_U1, _U2>, _TQual, _UQual>
{
  using type = pair<common_reference_t<_TQual<_T1>, _UQual<_U1>>, common_reference_t<_TQual<_T2>, _UQual<_U2>>>;
};

template <class _T1, class _T2, class _U1, class _U2>
  requires requires { typename pair<common_type_t<_T1, _U1>, common_type_t<_T2, _U2>>; }
struct common_type<pair<_T1, _T2>, pair<_U1, _U2>>
{
  using type = pair<common_type_t<_T1, _U1>, common_type_t<_T2, _U2>>;
};
#endif // _CCCL_STD_VER >= 2023

template <class _T1, class _T2>
_CCCL_API inline _CCCL_CONSTEXPR_CXX20 enable_if_t<__is_swappable<_T1>::value && __is_swappable<_T2>::value, void>
swap(pair<_T1, _T2>& __x,
     pair<_T1, _T2>& __y) noexcept((__is_nothrow_swappable<_T1>::value && __is_nothrow_swappable<_T2>::value))
{
  __x.swap(__y);
}

#if _CCCL_STD_VER >= 2023
template <class _T1, class _T2>
  requires(__is_swappable<const _T1>::value && __is_swappable<const _T2>::value)
_CCCL_API constexpr void swap(const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y) noexcept(noexcept(__x.swap(__y)))
{
  __x.swap(__y);
}
#endif // _CCCL_STD_VER >= 2023

template <class _T1, class _T2>
_CCCL_API constexpr pair<unwrap_ref_decay_t<_T1>, unwrap_ref_decay_t<_T2>> make_pair(_T1&& __t1, _T2&& __t2)
{
  return pair<unwrap_ref_decay_t<_T1>, unwrap_ref_decay_t<_T2>>(
    _CUDA_VSTD::forward<_T1>(__t1), _CUDA_VSTD::forward<_T2>(__t2));
}

template <class _T1, class _T2>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_size<pair<_T1, _T2>> : public integral_constant<size_t, 2>
{};

template <size_t _Ip, class _T1, class _T2>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element<_Ip, pair<_T1, _T2>>
{
  static_assert(_Ip < 2, "Index out of bounds in std::tuple_element<std::pair<T1, T2>>");
};

template <class _T1, class _T2>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element<0, pair<_T1, _T2>>
{
  using type _CCCL_NODEBUG_ALIAS = _T1;
};

template <class _T1, class _T2>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element<1, pair<_T1, _T2>>
{
  using type _CCCL_NODEBUG_ALIAS = _T2;
};

template <size_t _Ip>
struct __get_pair;

template <>
struct __get_pair<0>
{
  template <class _T1, class _T2>
  static _CCCL_API constexpr _T1& get(pair<_T1, _T2>& __p) noexcept
  {
    return __p.first;
  }

  template <class _T1, class _T2>
  static _CCCL_API constexpr const _T1& get(const pair<_T1, _T2>& __p) noexcept
  {
    return __p.first;
  }

  template <class _T1, class _T2>
  static _CCCL_API constexpr _T1&& get(pair<_T1, _T2>&& __p) noexcept
  {
    return _CUDA_VSTD::forward<_T1>(__p.first);
  }

  template <class _T1, class _T2>
  static _CCCL_API constexpr const _T1&& get(const pair<_T1, _T2>&& __p) noexcept
  {
    return _CUDA_VSTD::forward<const _T1>(__p.first);
  }
};

template <>
struct __get_pair<1>
{
  template <class _T1, class _T2>
  static _CCCL_API constexpr _T2& get(pair<_T1, _T2>& __p) noexcept
  {
    return __p.second;
  }

  template <class _T1, class _T2>
  static _CCCL_API constexpr const _T2& get(const pair<_T1, _T2>& __p) noexcept
  {
    return __p.second;
  }

  template <class _T1, class _T2>
  static _CCCL_API constexpr _T2&& get(pair<_T1, _T2>&& __p) noexcept
  {
    return _CUDA_VSTD::forward<_T2>(__p.second);
  }

  template <class _T1, class _T2>
  static _CCCL_API constexpr const _T2&& get(const pair<_T1, _T2>&& __p) noexcept
  {
    return _CUDA_VSTD::forward<const _T2>(__p.second);
  }
};

template <size_t _Ip, class _T1, class _T2>
_CCCL_API constexpr tuple_element_t<_Ip, pair<_T1, _T2>>& get(pair<_T1, _T2>& __p) noexcept
{
  return __get_pair<_Ip>::get(__p);
}

template <size_t _Ip, class _T1, class _T2>
_CCCL_API constexpr const tuple_element_t<_Ip, pair<_T1, _T2>>& get(const pair<_T1, _T2>& __p) noexcept
{
  return __get_pair<_Ip>::get(__p);
}

template <size_t _Ip, class _T1, class _T2>
_CCCL_API constexpr tuple_element_t<_Ip, pair<_T1, _T2>>&& get(pair<_T1, _T2>&& __p) noexcept
{
  return __get_pair<_Ip>::get(_CUDA_VSTD::move(__p));
}

template <size_t _Ip, class _T1, class _T2>
_CCCL_API constexpr const tuple_element_t<_Ip, pair<_T1, _T2>>&& get(const pair<_T1, _T2>&& __p) noexcept
{
  return __get_pair<_Ip>::get(_CUDA_VSTD::move(__p));
}

template <class _T1, class _T2>
_CCCL_API constexpr _T1& get(pair<_T1, _T2>& __p) noexcept
{
  return __get_pair<0>::get(__p);
}

template <class _T1, class _T2>
_CCCL_API constexpr _T1 const& get(pair<_T1, _T2> const& __p) noexcept
{
  return __get_pair<0>::get(__p);
}

template <class _T1, class _T2>
_CCCL_API constexpr _T1&& get(pair<_T1, _T2>&& __p) noexcept
{
  return __get_pair<0>::get(_CUDA_VSTD::move(__p));
}

template <class _T1, class _T2>
_CCCL_API constexpr _T1 const&& get(pair<_T1, _T2> const&& __p) noexcept
{
  return __get_pair<0>::get(_CUDA_VSTD::move(__p));
}

template <class _T1, class _T2>
_CCCL_API constexpr _T1& get(pair<_T2, _T1>& __p) noexcept
{
  return __get_pair<1>::get(__p);
}

template <class _T1, class _T2>
_CCCL_API constexpr _T1 const& get(pair<_T2, _T1> const& __p) noexcept
{
  return __get_pair<1>::get(__p);
}

template <class _T1, class _T2>
_CCCL_API constexpr _T1&& get(pair<_T2, _T1>&& __p) noexcept
{
  return __get_pair<1>::get(_CUDA_VSTD::move(__p));
}

template <class _T1, class _T2>
_CCCL_API constexpr _T1 const&& get(pair<_T2, _T1> const&& __p) noexcept
{
  return __get_pair<1>::get(_CUDA_VSTD::move(__p));
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___UTILITY_PAIR_H
