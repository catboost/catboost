//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023-24 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TUPLE_SFINAE_HELPERS_H
#define _LIBCUDACXX___TUPLE_SFINAE_HELPERS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__fwd/tuple.h>
#include <cuda/std/__tuple_dir/make_tuple_types.h>
#include <cuda/std/__tuple_dir/tuple_element.h>
#include <cuda/std/__tuple_dir/tuple_like_ext.h>
#include <cuda/std/__tuple_dir/tuple_size.h>
#include <cuda/std/__tuple_dir/tuple_types.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_assignable.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_copy_assignable.h>
#include <cuda/std/__type_traits/is_move_assignable.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <bool... _Preds>
struct __all_dummy;

template <bool... _Pred>
using __all = is_same<__all_dummy<_Pred...>, __all_dummy<((void) _Pred, true)...>>;

struct __tuple_sfinae_base
{
  template <class, class>
  struct __test_size : false_type
  {};

  template <class... _Tp, class... _Up>
  struct __test_size<__tuple_types<_Tp...>, __tuple_types<_Up...>> : bool_constant<sizeof...(_Tp) == sizeof...(_Up)>
  {};

  template <template <class, class...> class, class _Tp, class _Up, bool = __test_size<_Tp, _Up>::value>
  struct __test : false_type
  {};

  template <template <class, class...> class _Trait, class... _LArgs, class... _RArgs>
  struct __test<_Trait, __tuple_types<_LArgs...>, __tuple_types<_RArgs...>, true>
      : __all<_Trait<_LArgs, _RArgs>::value...>
  {};

  template <class _FromArgs, class _ToArgs>
  using __constructible = __test<is_constructible, _ToArgs, _FromArgs>;
  template <class _FromArgs, class _ToArgs>
  using __convertible = __test<is_convertible, _FromArgs, _ToArgs>;
  template <class _FromArgs, class _ToArgs>
  using __assignable = __test<is_assignable, _ToArgs, _FromArgs>;
};

// __tuple_convertible

template <class _Tp,
          class _Up,
          bool = __tuple_like_ext<remove_reference_t<_Tp>>::value,
          bool = __tuple_like_ext<_Up>::value>
struct __tuple_convertible : public false_type
{};

template <class _Tp, class _Up>
struct __tuple_convertible<_Tp, _Up, true, true>
    : public __tuple_sfinae_base::__convertible<__make_tuple_types_t<_Tp>, __make_tuple_types_t<_Up>>
{};

// __tuple_constructible

template <class _Tp,
          class _Up,
          bool = __tuple_like_ext<remove_reference_t<_Tp>>::value,
          bool = __tuple_like_ext<_Up>::value>
struct __tuple_constructible : public false_type
{};

template <class _Tp, class _Up>
struct __tuple_constructible<_Tp, _Up, true, true>
    : public __tuple_sfinae_base::__constructible<__make_tuple_types_t<_Tp>, __make_tuple_types_t<_Up>>
{};

// __tuple_assignable

template <class _Tp,
          class _Up,
          bool = __tuple_like_ext<remove_reference_t<_Tp>>::value,
          bool = __tuple_like_ext<_Up>::value>
struct __tuple_assignable : public false_type
{};

template <class _Tp, class _Up>
struct __tuple_assignable<_Tp, _Up, true, true>
    : public __tuple_sfinae_base::__assignable<__make_tuple_types_t<_Tp>, __make_tuple_types_t<_Up&>>
{};

template <size_t _Ip, class... _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element<_Ip, tuple<_Tp...>>
{
  using type _CCCL_NODEBUG_ALIAS = tuple_element_t<_Ip, __tuple_types<_Tp...>>;
};

template <bool _IsTuple, class _SizeTrait, size_t _Expected>
struct __tuple_like_with_size_imp : false_type
{};

template <class _SizeTrait, size_t _Expected>
struct __tuple_like_with_size_imp<true, _SizeTrait, _Expected> : integral_constant<bool, _SizeTrait::value == _Expected>
{};

template <class _Tuple, size_t _ExpectedSize, class _RawTuple = remove_cvref_t<_Tuple>>
using __tuple_like_with_size _CCCL_NODEBUG_ALIAS =
  __tuple_like_with_size_imp<__tuple_like_ext<_RawTuple>::value, tuple_size<_RawTuple>, _ExpectedSize>;

struct _CCCL_TYPE_VISIBILITY_DEFAULT __check_tuple_constructor_fail
{
  template <int&...>
  using __enable_explicit_default = false_type;
  template <int&...>
  using __enable_implicit_default = false_type;
  template <class...>
  using __enable_explicit = false_type;
  template <class...>
  using __enable_implicit = false_type;
  template <class...>
  using __enable_assign = false_type;
};

enum class __smf_availability
{
  __trivial,
  __available,
  __deleted,
};

template <bool _CanCopy>
struct __sfinae_copy_base
{};
template <>
struct __sfinae_copy_base<false>
{
  _CCCL_HIDE_FROM_ABI __sfinae_copy_base()                                     = default;
  __sfinae_copy_base(__sfinae_copy_base const&)                                = delete;
  _CCCL_HIDE_FROM_ABI __sfinae_copy_base(__sfinae_copy_base&&)                 = default;
  _CCCL_HIDE_FROM_ABI __sfinae_copy_base& operator=(__sfinae_copy_base const&) = default;
  _CCCL_HIDE_FROM_ABI __sfinae_copy_base& operator=(__sfinae_copy_base&&)      = default;
};

template <bool _CanCopy, bool _CanMove>
struct __sfinae_move_base : __sfinae_copy_base<_CanCopy>
{};
template <bool _CanCopy>
struct __sfinae_move_base<_CanCopy, false> : __sfinae_copy_base<_CanCopy>
{
  _CCCL_HIDE_FROM_ABI __sfinae_move_base()                                     = default;
  _CCCL_HIDE_FROM_ABI __sfinae_move_base(__sfinae_move_base const&)            = default;
  __sfinae_move_base(__sfinae_move_base&&)                                     = delete;
  _CCCL_HIDE_FROM_ABI __sfinae_move_base& operator=(__sfinae_move_base const&) = default;
  _CCCL_HIDE_FROM_ABI __sfinae_move_base& operator=(__sfinae_move_base&&)      = default;
};

template <bool _CanCopy, bool _CanMove, bool _CanCopyAssign>
struct __sfinae_copy_assign_base : __sfinae_move_base<_CanCopy, _CanMove>
{};
template <bool _CanCopy, bool _CanMove>
struct __sfinae_copy_assign_base<_CanCopy, _CanMove, false> : __sfinae_move_base<_CanCopy, _CanMove>
{
  _CCCL_HIDE_FROM_ABI __sfinae_copy_assign_base()                                       = default;
  _CCCL_HIDE_FROM_ABI __sfinae_copy_assign_base(__sfinae_copy_assign_base const&)       = default;
  _CCCL_HIDE_FROM_ABI __sfinae_copy_assign_base(__sfinae_copy_assign_base&&)            = default;
  __sfinae_copy_assign_base& operator=(__sfinae_copy_assign_base const&)                = delete;
  _CCCL_HIDE_FROM_ABI __sfinae_copy_assign_base& operator=(__sfinae_copy_assign_base&&) = default;
};

template <bool _CanCopy, bool _CanMove, bool _CanCopyAssign, bool _CanMoveAssign>
struct __sfinae_move_assign_base : __sfinae_copy_assign_base<_CanCopy, _CanMove, _CanCopyAssign>
{};
template <bool _CanCopy, bool _CanMove, bool _CanCopyAssign>
struct __sfinae_move_assign_base<_CanCopy, _CanMove, _CanCopyAssign, false>
    : __sfinae_copy_assign_base<_CanCopy, _CanMove, _CanCopyAssign>
{
  _CCCL_HIDE_FROM_ABI __sfinae_move_assign_base()                                            = default;
  _CCCL_HIDE_FROM_ABI __sfinae_move_assign_base(__sfinae_move_assign_base const&)            = default;
  _CCCL_HIDE_FROM_ABI __sfinae_move_assign_base(__sfinae_move_assign_base&&)                 = default;
  _CCCL_HIDE_FROM_ABI __sfinae_move_assign_base& operator=(__sfinae_move_assign_base const&) = default;
  __sfinae_move_assign_base& operator=(__sfinae_move_assign_base&&)                          = delete;
};

template <bool _CanCopy, bool _CanMove, bool _CanCopyAssign, bool _CanMoveAssign>
using __sfinae_base = __sfinae_move_assign_base<_CanCopy, _CanMove, _CanCopyAssign, _CanMoveAssign>;

// We need to synthesize the copy / move assignment if it would be implicitly deleted as a member of a class
// In that case _Tp would be copy assignable but _TestSynthesizeAssignment<_Tp> would not
// This happens e.g for reference types
template <class _Tp>
struct _TestSynthesizeAssignment
{
  _Tp __dummy;
};

template <class _Tp>
inline constexpr bool __must_synthesize_assignment_v =
  (is_copy_assignable_v<_Tp> && !is_copy_assignable_v<_TestSynthesizeAssignment<_Tp>>)
  || (is_move_assignable_v<_Tp> && !is_move_assignable_v<_TestSynthesizeAssignment<_Tp>>);

// We need to ensure that __tuple_impl_sfinae_helper is unique for every instantiation of __tuple_impl, so its templated
// on the impl
template <class _Impl, bool _AllCopyAssignable, bool _AllMoveAssignable>
struct _CCCL_DECLSPEC_EMPTY_BASES __tuple_impl_sfinae_helper
{};

template <class _Impl>
struct _CCCL_DECLSPEC_EMPTY_BASES __tuple_impl_sfinae_helper<_Impl, false, true>
{
  __tuple_impl_sfinae_helper()                                             = default;
  __tuple_impl_sfinae_helper(const __tuple_impl_sfinae_helper&)            = default;
  __tuple_impl_sfinae_helper(__tuple_impl_sfinae_helper&&)                 = default;
  __tuple_impl_sfinae_helper& operator=(const __tuple_impl_sfinae_helper&) = delete;
  __tuple_impl_sfinae_helper& operator=(__tuple_impl_sfinae_helper&&)      = default;
};

template <class _Impl>
struct _CCCL_DECLSPEC_EMPTY_BASES __tuple_impl_sfinae_helper<_Impl, true, false>
{
  __tuple_impl_sfinae_helper()                                             = default;
  __tuple_impl_sfinae_helper(const __tuple_impl_sfinae_helper&)            = default;
  __tuple_impl_sfinae_helper(__tuple_impl_sfinae_helper&&)                 = default;
  __tuple_impl_sfinae_helper& operator=(const __tuple_impl_sfinae_helper&) = default;
  __tuple_impl_sfinae_helper& operator=(__tuple_impl_sfinae_helper&&)      = delete;
};

template <class _Impl>
struct _CCCL_DECLSPEC_EMPTY_BASES __tuple_impl_sfinae_helper<_Impl, false, false>
{
  __tuple_impl_sfinae_helper()                                             = default;
  __tuple_impl_sfinae_helper(const __tuple_impl_sfinae_helper&)            = default;
  __tuple_impl_sfinae_helper(__tuple_impl_sfinae_helper&&)                 = default;
  __tuple_impl_sfinae_helper& operator=(const __tuple_impl_sfinae_helper&) = delete;
  __tuple_impl_sfinae_helper& operator=(__tuple_impl_sfinae_helper&&)      = delete;
};

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___TUPLE_SFINAE_HELPERS_H
