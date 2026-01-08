//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___EXPECTED_EXPECTED_BASE_H
#define _LIBCUDACXX___EXPECTED_EXPECTED_BASE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/invocable.h>
#include <cuda/std/__expected/unexpect.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__memory/construct_at.h>
#include <cuda/std/__tuple_dir/sfinae_helpers.h>
#include <cuda/std/__type_traits/is_assignable.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_copy_assignable.h>
#include <cuda/std/__type_traits/is_copy_constructible.h>
#include <cuda/std/__type_traits/is_default_constructible.h>
#include <cuda/std/__type_traits/is_move_assignable.h>
#include <cuda/std/__type_traits/is_move_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_copy_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/is_trivially_copy_assignable.h>
#include <cuda/std/__type_traits/is_trivially_copy_constructible.h>
#include <cuda/std/__type_traits/is_trivially_destructible.h>
#include <cuda/std/__type_traits/is_trivially_move_assignable.h>
#include <cuda/std/__type_traits/is_trivially_move_constructible.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__utility/exception_guard.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/in_place.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// MSVC complains about [[no_unique_address]] prior to C++20 as a vendor extension
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4848)

struct __expected_construct_from_invoke_tag
{
  _CCCL_HIDE_FROM_ABI explicit __expected_construct_from_invoke_tag() = default;
};

template <class _Tp,
          class _Err,
          bool = _CCCL_TRAIT(is_trivially_destructible, _Tp) && _CCCL_TRAIT(is_trivially_destructible, _Err)>
union __expected_union_t
{
  struct __empty_t
  {};

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Tp2 = _Tp)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_default_constructible, _Tp2))
  _CCCL_API constexpr __expected_union_t() noexcept(_CCCL_TRAIT(is_nothrow_default_constructible, _Tp2))
      : __val_()
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Tp2 = _Tp)
  _CCCL_REQUIRES((!_CCCL_TRAIT(is_default_constructible, _Tp2)))
  _CCCL_API constexpr __expected_union_t() noexcept
      : __empty_()
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class... _Args>
  _CCCL_API constexpr __expected_union_t(in_place_t, _Args&&... __args) noexcept(
    _CCCL_TRAIT(is_nothrow_constructible, _Tp, _Args...))
      : __val_(_CUDA_VSTD::forward<_Args>(__args)...)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class... _Args>
  _CCCL_API constexpr __expected_union_t(unexpect_t, _Args&&... __args) noexcept(
    _CCCL_TRAIT(is_nothrow_constructible, _Err, _Args...))
      : __unex_(_CUDA_VSTD::forward<_Args>(__args)...)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Fun, class... _Args>
  _CCCL_API constexpr __expected_union_t(
    __expected_construct_from_invoke_tag,
    in_place_t,
    _Fun&& __fun,
    _Args&&... __args) noexcept(_CCCL_TRAIT(is_nothrow_constructible, _Tp, invoke_result_t<_Fun, _Args...>))
      : __val_(_CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun), _CUDA_VSTD::forward<_Args>(__args)...))
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Fun, class... _Args>
  _CCCL_API constexpr __expected_union_t(
    __expected_construct_from_invoke_tag,
    unexpect_t,
    _Fun&& __fun,
    _Args&&... __args) noexcept(_CCCL_TRAIT(is_nothrow_constructible, _Err, invoke_result_t<_Fun, _Args...>))
      : __unex_(_CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun), _CUDA_VSTD::forward<_Args>(__args)...))
  {}

  // the __expected_destruct's destructor handles this
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 ~__expected_union_t() {}

  _CCCL_NO_UNIQUE_ADDRESS __empty_t __empty_;
  _CCCL_NO_UNIQUE_ADDRESS _Tp __val_;
  _CCCL_NO_UNIQUE_ADDRESS _Err __unex_;
};

template <class _Tp, class _Err>
union __expected_union_t<_Tp, _Err, true>
{
  struct __empty_t
  {};

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Tp2 = _Tp)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_default_constructible, _Tp2))
  _CCCL_API constexpr __expected_union_t() noexcept(_CCCL_TRAIT(is_nothrow_default_constructible, _Tp2))
      : __val_()
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Tp2 = _Tp)
  _CCCL_REQUIRES((!_CCCL_TRAIT(is_default_constructible, _Tp2)))
  _CCCL_API constexpr __expected_union_t() noexcept
      : __empty_()
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class... _Args>
  _CCCL_API constexpr __expected_union_t(in_place_t, _Args&&... __args) noexcept(
    _CCCL_TRAIT(is_nothrow_constructible, _Tp, _Args...))
      : __val_(_CUDA_VSTD::forward<_Args>(__args)...)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class... _Args>
  _CCCL_API constexpr __expected_union_t(unexpect_t, _Args&&... __args) noexcept(
    _CCCL_TRAIT(is_nothrow_constructible, _Err, _Args...))
      : __unex_(_CUDA_VSTD::forward<_Args>(__args)...)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Fun, class... _Args>
  _CCCL_API constexpr __expected_union_t(
    __expected_construct_from_invoke_tag,
    in_place_t,
    _Fun&& __fun,
    _Args&&... __args) noexcept(_CCCL_TRAIT(is_nothrow_constructible, _Tp, invoke_result_t<_Fun, _Args...>))
      : __val_(_CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun), _CUDA_VSTD::forward<_Args>(__args)...))
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Fun, class... _Args>
  _CCCL_API constexpr __expected_union_t(
    __expected_construct_from_invoke_tag,
    unexpect_t,
    _Fun&& __fun,
    _Args&&... __args) noexcept(_CCCL_TRAIT(is_nothrow_constructible, _Err, invoke_result_t<_Fun, _Args...>))
      : __unex_(_CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun), _CUDA_VSTD::forward<_Args>(__args)...))
  {}

  _CCCL_NO_UNIQUE_ADDRESS __empty_t __empty_;
  _CCCL_NO_UNIQUE_ADDRESS _Tp __val_;
  _CCCL_NO_UNIQUE_ADDRESS _Err __unex_;
};

template <class _Tp,
          class _Err,
          bool = _CCCL_TRAIT(is_trivially_destructible, _Tp),
          bool = _CCCL_TRAIT(is_trivially_destructible, _Err)>
struct __expected_destruct;

template <class _Tp, class _Err>
struct __expected_destruct<_Tp, _Err, false, false>
{
  _CCCL_NO_UNIQUE_ADDRESS __expected_union_t<_Tp, _Err> __union_{};
  bool __has_val_{true};

  _CCCL_HIDE_FROM_ABI constexpr __expected_destruct() = default;

  _CCCL_API constexpr __expected_destruct(const bool __has_val) noexcept(
    _CCCL_TRAIT(is_nothrow_default_constructible, _Tp))
      : __has_val_(__has_val)
  {}

  template <class... _Args>
  _CCCL_API constexpr __expected_destruct(in_place_t, _Args&&... __args) noexcept(
    _CCCL_TRAIT(is_nothrow_constructible, _Tp, _Args...))
      : __union_(in_place, _CUDA_VSTD::forward<_Args>(__args)...)
      , __has_val_(true)
  {}

  template <class... _Args>
  _CCCL_API constexpr __expected_destruct(unexpect_t, _Args&&... __args) noexcept(
    _CCCL_TRAIT(is_nothrow_constructible, _Err, _Args...))
      : __union_(unexpect, _CUDA_VSTD::forward<_Args>(__args)...)
      , __has_val_(false)
  {}

  template <class _Fun, class... _Args>
  _CCCL_API constexpr __expected_destruct(
    __expected_construct_from_invoke_tag,
    in_place_t,
    _Fun&& __fun,
    _Args&&... __args) noexcept(_CCCL_TRAIT(is_nothrow_constructible, _Tp, invoke_result_t<_Fun, _Args...>))
      : __union_(__expected_construct_from_invoke_tag{},
                 in_place,
                 _CUDA_VSTD::forward<_Fun>(__fun),
                 _CUDA_VSTD::forward<_Args>(__args)...)
      , __has_val_(true)
  {}

  template <class _Fun, class... _Args>
  _CCCL_API constexpr __expected_destruct(
    __expected_construct_from_invoke_tag,
    unexpect_t,
    _Fun&& __fun,
    _Args&&... __args) noexcept(_CCCL_TRAIT(is_nothrow_constructible, _Err, invoke_result_t<_Fun, _Args...>))
      : __union_(__expected_construct_from_invoke_tag{},
                 unexpect,
                 _CUDA_VSTD::forward<_Fun>(__fun),
                 _CUDA_VSTD::forward<_Args>(__args)...)
      , __has_val_(false)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 ~__expected_destruct()
  {
    if (__has_val_)
    {
      __union_.__val_.~_Tp();
    }
    else
    {
      __union_.__unex_.~_Err();
    }
  }
};

template <class _Tp, class _Err>
struct __expected_destruct<_Tp, _Err, true, false>
{
  _CCCL_NO_UNIQUE_ADDRESS __expected_union_t<_Tp, _Err> __union_{};
  bool __has_val_{true};

  _CCCL_HIDE_FROM_ABI constexpr __expected_destruct() = default;

  _CCCL_API constexpr __expected_destruct(const bool __has_val) noexcept(
    _CCCL_TRAIT(is_nothrow_default_constructible, _Tp))
      : __has_val_(__has_val)
  {}

  template <class... _Args>
  _CCCL_API constexpr __expected_destruct(in_place_t, _Args&&... __args) noexcept(
    _CCCL_TRAIT(is_nothrow_constructible, _Tp, _Args...))
      : __union_(in_place, _CUDA_VSTD::forward<_Args>(__args)...)
      , __has_val_(true)
  {}

  template <class... _Args>
  _CCCL_API constexpr __expected_destruct(unexpect_t, _Args&&... __args) noexcept(
    _CCCL_TRAIT(is_nothrow_constructible, _Err, _Args...))
      : __union_(unexpect, _CUDA_VSTD::forward<_Args>(__args)...)
      , __has_val_(false)
  {}

  template <class _Fun, class... _Args>
  _CCCL_API constexpr __expected_destruct(
    __expected_construct_from_invoke_tag,
    in_place_t,
    _Fun&& __fun,
    _Args&&... __args) noexcept(_CCCL_TRAIT(is_nothrow_constructible, _Tp, invoke_result_t<_Fun, _Args...>))
      : __union_(__expected_construct_from_invoke_tag{},
                 in_place,
                 _CUDA_VSTD::forward<_Fun>(__fun),
                 _CUDA_VSTD::forward<_Args>(__args)...)
      , __has_val_(true)
  {}

  template <class _Fun, class... _Args>
  _CCCL_API constexpr __expected_destruct(
    __expected_construct_from_invoke_tag,
    unexpect_t,
    _Fun&& __fun,
    _Args&&... __args) noexcept(_CCCL_TRAIT(is_nothrow_constructible, _Err, invoke_result_t<_Fun, _Args...>))
      : __union_(__expected_construct_from_invoke_tag{},
                 unexpect,
                 _CUDA_VSTD::forward<_Fun>(__fun),
                 _CUDA_VSTD::forward<_Args>(__args)...)
      , __has_val_(false)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 ~__expected_destruct()
  {
    if (!__has_val_)
    {
      __union_.__unex_.~_Err();
    }
  }
};

template <class _Tp, class _Err>
struct __expected_destruct<_Tp, _Err, false, true>
{
  _CCCL_NO_UNIQUE_ADDRESS __expected_union_t<_Tp, _Err> __union_{};
  bool __has_val_{true};

  _CCCL_HIDE_FROM_ABI constexpr __expected_destruct() = default;

  _CCCL_API constexpr __expected_destruct(const bool __has_val) noexcept(
    _CCCL_TRAIT(is_nothrow_default_constructible, _Tp))
      : __has_val_(__has_val)
  {}

  template <class... _Args>
  _CCCL_API constexpr __expected_destruct(in_place_t, _Args&&... __args) noexcept(
    _CCCL_TRAIT(is_nothrow_constructible, _Tp, _Args...))
      : __union_(in_place, _CUDA_VSTD::forward<_Args>(__args)...)
      , __has_val_(true)
  {}

  template <class... _Args>
  _CCCL_API constexpr __expected_destruct(unexpect_t, _Args&&... __args) noexcept(
    _CCCL_TRAIT(is_nothrow_constructible, _Err, _Args...))
      : __union_(unexpect, _CUDA_VSTD::forward<_Args>(__args)...)
      , __has_val_(false)
  {}

  template <class _Fun, class... _Args>
  _CCCL_API constexpr __expected_destruct(
    __expected_construct_from_invoke_tag,
    in_place_t,
    _Fun&& __fun,
    _Args&&... __args) noexcept(_CCCL_TRAIT(is_nothrow_constructible, _Tp, invoke_result_t<_Fun, _Args...>))
      : __union_(__expected_construct_from_invoke_tag{},
                 in_place,
                 _CUDA_VSTD::forward<_Fun>(__fun),
                 _CUDA_VSTD::forward<_Args>(__args)...)
      , __has_val_(true)
  {}

  template <class _Fun, class... _Args>
  _CCCL_API constexpr __expected_destruct(
    __expected_construct_from_invoke_tag,
    unexpect_t,
    _Fun&& __fun,
    _Args&&... __args) noexcept(_CCCL_TRAIT(is_nothrow_constructible, _Err, invoke_result_t<_Fun, _Args...>))
      : __union_(__expected_construct_from_invoke_tag{},
                 unexpect,
                 _CUDA_VSTD::forward<_Fun>(__fun),
                 _CUDA_VSTD::forward<_Args>(__args)...)
      , __has_val_(false)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 ~__expected_destruct()
  {
    if (__has_val_)
    {
      __union_.__val_.~_Tp();
    }
  }
};

template <class _Tp, class _Err>
struct __expected_destruct<_Tp, _Err, true, true>
{
  // This leads to an ICE with nvcc, see nvbug4103076
  /* _CCCL_NO_UNIQUE_ADDRESS */ __expected_union_t<_Tp, _Err> __union_{};
  bool __has_val_{true};

  _CCCL_HIDE_FROM_ABI constexpr __expected_destruct() = default;

  _CCCL_API constexpr __expected_destruct(const bool __has_val) noexcept(
    _CCCL_TRAIT(is_nothrow_default_constructible, _Tp))
      : __has_val_(__has_val)
  {}

  template <class... _Args>
  _CCCL_API constexpr __expected_destruct(in_place_t, _Args&&... __args) noexcept(
    _CCCL_TRAIT(is_nothrow_constructible, _Tp, _Args...))
      : __union_(in_place, _CUDA_VSTD::forward<_Args>(__args)...)
      , __has_val_(true)
  {}

  template <class... _Args>
  _CCCL_API constexpr __expected_destruct(unexpect_t, _Args&&... __args) noexcept(
    _CCCL_TRAIT(is_nothrow_constructible, _Err, _Args...))
      : __union_(unexpect, _CUDA_VSTD::forward<_Args>(__args)...)
      , __has_val_(false)
  {}

  template <class _Fun, class... _Args>
  _CCCL_API constexpr __expected_destruct(
    __expected_construct_from_invoke_tag,
    in_place_t,
    _Fun&& __fun,
    _Args&&... __args) noexcept(_CCCL_TRAIT(is_nothrow_constructible, _Tp, invoke_result_t<_Fun, _Args...>))
      : __union_(__expected_construct_from_invoke_tag{},
                 in_place,
                 _CUDA_VSTD::forward<_Fun>(__fun),
                 _CUDA_VSTD::forward<_Args>(__args)...)
      , __has_val_(true)
  {}

  template <class _Fun, class... _Args>
  _CCCL_API constexpr __expected_destruct(
    __expected_construct_from_invoke_tag,
    unexpect_t,
    _Fun&& __fun,
    _Args&&... __args) noexcept(_CCCL_TRAIT(is_nothrow_constructible, _Err, invoke_result_t<_Fun, _Args...>))
      : __union_(__expected_construct_from_invoke_tag{},
                 unexpect,
                 _CUDA_VSTD::forward<_Fun>(__fun),
                 _CUDA_VSTD::forward<_Args>(__args)...)
      , __has_val_(false)
  {}
};

_CCCL_DIAG_POP

template <class _Tp, class _Err>
struct __expected_storage : __expected_destruct<_Tp, _Err>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__expected_storage, __expected_destruct, _Tp, _Err);

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _T1, class _T2, class... _Args)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_nothrow_constructible, _T1, _Args...))
  static _CCCL_API inline _CCCL_CONSTEXPR_CXX20 void
  __reinit_expected(_T1& __newval, _T2& __oldval, _Args&&... __args) noexcept
  {
    _CUDA_VSTD::__destroy_at(_CUDA_VSTD::addressof(__oldval));
    _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(__newval), _CUDA_VSTD::forward<_Args>(__args)...);
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _T1, class _T2, class... _Args)
  _CCCL_REQUIRES(
    (!_CCCL_TRAIT(is_nothrow_constructible, _T1, _Args...)) _CCCL_AND _CCCL_TRAIT(is_nothrow_move_constructible, _T1))
  static _CCCL_API inline _CCCL_CONSTEXPR_CXX20 void __reinit_expected(_T1& __newval, _T2& __oldval, _Args&&... __args)
  {
    _T1 __tmp(_CUDA_VSTD::forward<_Args>(__args)...);
    _CUDA_VSTD::__destroy_at(_CUDA_VSTD::addressof(__oldval));
    _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(__newval), _CUDA_VSTD::move(__tmp));
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _T1, class _T2, class... _Args)
  _CCCL_REQUIRES(
    (!_CCCL_TRAIT(is_nothrow_constructible, _T1, _Args...)) _CCCL_AND(!_CCCL_TRAIT(is_nothrow_move_constructible, _T1)))
  static _CCCL_API inline _CCCL_CONSTEXPR_CXX20 void __reinit_expected(_T1& __newval, _T2& __oldval, _Args&&... __args)
  {
    static_assert(
      _CCCL_TRAIT(is_nothrow_move_constructible, _T2),
      "To provide strong exception guarantee, T2 has to satisfy `is_nothrow_move_constructible_v` so that it can "
      "be reverted to the previous state in case an exception is thrown during the assignment.");
    _T2 __tmp(_CUDA_VSTD::move(__oldval));
    _CUDA_VSTD::__destroy_at(_CUDA_VSTD::addressof(__oldval));
    auto __trans = _CUDA_VSTD::__make_exception_guard([&] {
      _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(__oldval), _CUDA_VSTD::move(__tmp));
    });
    _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(__newval), _CUDA_VSTD::forward<_Args>(__args)...);
    __trans.__complete();
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Err2 = _Err)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_nothrow_move_constructible, _Err2))
  static _CCCL_API inline _CCCL_CONSTEXPR_CXX20 void
  __swap_val_unex_impl(__expected_storage<_Tp, _Err2>& __with_val, __expected_storage& __with_err)
  {
    _Err __tmp(_CUDA_VSTD::move(__with_err.__union_.__unex_));
    _CUDA_VSTD::__destroy_at(_CUDA_VSTD::addressof(__with_err.__union_.__unex_));
    auto __trans = _CUDA_VSTD::__make_exception_guard([&] {
      _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(__with_err.__union_.__unex_), _CUDA_VSTD::move(__tmp));
    });
    _CUDA_VSTD::__construct_at(
      _CUDA_VSTD::addressof(__with_err.__union_.__val_), _CUDA_VSTD::move(__with_val.__union_.__val_));
    __trans.__complete();
    _CUDA_VSTD::__destroy_at(_CUDA_VSTD::addressof(__with_val.__union_.__val_));
    _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(__with_val.__union_.__unex_), _CUDA_VSTD::move(__tmp));
    __with_val.__has_val_ = false;
    __with_err.__has_val_ = true;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Err2 = _Err)
  _CCCL_REQUIRES((!_CCCL_TRAIT(is_nothrow_move_constructible, _Err2)))
  static _CCCL_API inline _CCCL_CONSTEXPR_CXX20 void
  __swap_val_unex_impl(__expected_storage<_Tp, _Err2>& __with_val, __expected_storage& __with_err)
  {
    static_assert(_CCCL_TRAIT(is_nothrow_move_constructible, _Tp),
                  "To provide strong exception guarantee, Tp has to satisfy `is_nothrow_move_constructible_v` so "
                  "that it can be reverted to the previous state in case an exception is thrown during swap.");
    _Tp __tmp(_CUDA_VSTD::move(__with_val.__union_.__val_));
    _CUDA_VSTD::__destroy_at(_CUDA_VSTD::addressof(__with_val.__union_.__val_));
    auto __trans = _CUDA_VSTD::__make_exception_guard([&] {
      _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(__with_val.__union_.__val_), _CUDA_VSTD::move(__tmp));
    });
    _CUDA_VSTD::__construct_at(
      _CUDA_VSTD::addressof(__with_val.__union_.__unex_), _CUDA_VSTD::move(__with_err.__union_.__unex_));
    __trans.__complete();
    _CUDA_VSTD::__destroy_at(_CUDA_VSTD::addressof(__with_err.__union_.__unex_));
    _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(__with_err.__union_.__val_), _CUDA_VSTD::move(__tmp));
    __with_val.__has_val_ = false;
    __with_err.__has_val_ = true;
  }
};

template <class _Tp, class _Err>
inline constexpr __smf_availability __expected_can_copy_construct =
  (_CCCL_TRAIT(is_trivially_copy_constructible, _Tp) || _CCCL_TRAIT(is_same, _Tp, void))
      && _CCCL_TRAIT(is_trivially_copy_constructible, _Err)
    ? __smf_availability::__trivial
  : (_CCCL_TRAIT(is_copy_constructible, _Tp) || _CCCL_TRAIT(is_same, _Tp, void))
      && _CCCL_TRAIT(is_copy_constructible, _Err)
    ? __smf_availability::__available
    : __smf_availability::__deleted;

template <class _Tp, class _Err, __smf_availability = __expected_can_copy_construct<_Tp, _Err>>
struct __expected_copy : __expected_storage<_Tp, _Err>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__expected_copy, __expected_storage, _Tp, _Err);
};

template <class _Tp, class _Err>
struct __expected_copy<_Tp, _Err, __smf_availability::__available> : __expected_storage<_Tp, _Err>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__expected_copy, __expected_storage, _Tp, _Err);

  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 __expected_copy(const __expected_copy& __other) noexcept(
    _CCCL_TRAIT(is_nothrow_copy_constructible, _Tp) && _CCCL_TRAIT(is_nothrow_copy_constructible, _Err))
      : __base(__other.__has_val_)
  {
    if (__other.__has_val_)
    {
      _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(this->__union_.__val_), __other.__union_.__val_);
    }
    else
    {
      _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(this->__union_.__unex_), __other.__union_.__unex_);
    }
  }

  _CCCL_HIDE_FROM_ABI __expected_copy(__expected_copy&&)                 = default;
  _CCCL_HIDE_FROM_ABI __expected_copy& operator=(const __expected_copy&) = default;
  _CCCL_HIDE_FROM_ABI __expected_copy& operator=(__expected_copy&&)      = default;
};

template <class _Tp, class _Err>
struct __expected_copy<_Tp, _Err, __smf_availability::__deleted> : __expected_storage<_Tp, _Err>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__expected_copy, __expected_storage, _Tp, _Err);

  __expected_copy(const __expected_copy&)                                = delete;
  _CCCL_HIDE_FROM_ABI __expected_copy(__expected_copy&&)                 = default;
  _CCCL_HIDE_FROM_ABI __expected_copy& operator=(const __expected_copy&) = default;
  _CCCL_HIDE_FROM_ABI __expected_copy& operator=(__expected_copy&&)      = default;
};

template <class _Tp, class _Err>
inline constexpr __smf_availability __expected_can_move_construct =
  (_CCCL_TRAIT(is_trivially_move_constructible, _Tp) || _CCCL_TRAIT(is_same, _Tp, void))
      && _CCCL_TRAIT(is_trivially_move_constructible, _Err)
    ? __smf_availability::__trivial
  : (_CCCL_TRAIT(is_move_constructible, _Tp) || _CCCL_TRAIT(is_same, _Tp, void))
      && _CCCL_TRAIT(is_move_constructible, _Err)
    ? __smf_availability::__available
    : __smf_availability::__deleted;

template <class _Tp, class _Err, __smf_availability = __expected_can_move_construct<_Tp, _Err>>
struct __expected_move : __expected_copy<_Tp, _Err>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__expected_move, __expected_copy, _Tp, _Err);
};

template <class _Tp, class _Err>
struct __expected_move<_Tp, _Err, __smf_availability::__available> : __expected_copy<_Tp, _Err>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__expected_move, __expected_copy, _Tp, _Err);

  _CCCL_HIDE_FROM_ABI __expected_move(const __expected_move&) = default;

  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 __expected_move(__expected_move&& __other) noexcept(
    _CCCL_TRAIT(is_nothrow_move_constructible, _Tp) && _CCCL_TRAIT(is_nothrow_move_constructible, _Err))
      : __base(__other.__has_val_)
  {
    if (__other.__has_val_)
    {
      _CUDA_VSTD::__construct_at(
        _CUDA_VSTD::addressof(this->__union_.__val_), _CUDA_VSTD::move(__other.__union_.__val_));
    }
    else
    {
      _CUDA_VSTD::__construct_at(
        _CUDA_VSTD::addressof(this->__union_.__unex_), _CUDA_VSTD::move(__other.__union_.__unex_));
    }
  }

  _CCCL_HIDE_FROM_ABI __expected_move& operator=(const __expected_move&) = default;
  _CCCL_HIDE_FROM_ABI __expected_move& operator=(__expected_move&&)      = default;
};

template <class _Tp, class _Err>
struct __expected_move<_Tp, _Err, __smf_availability::__deleted> : __expected_copy<_Tp, _Err>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__expected_move, __expected_copy, _Tp, _Err);

  _CCCL_HIDE_FROM_ABI __expected_move(const __expected_move&)            = default;
  __expected_move(__expected_move&&)                                     = delete;
  _CCCL_HIDE_FROM_ABI __expected_move& operator=(const __expected_move&) = default;
  _CCCL_HIDE_FROM_ABI __expected_move& operator=(__expected_move&&)      = default;
};

// Need to also check against is_nothrow_move_constructible in the trivial case as that is stupidly in the constraints
template <class _Tp, class _Err>
inline constexpr __smf_availability __expected_can_copy_assign =
  (_CCCL_TRAIT(is_trivially_destructible, _Tp) || _CCCL_TRAIT(is_same, _Tp, void))
      && _CCCL_TRAIT(is_trivially_destructible, _Err)
      && (_CCCL_TRAIT(is_trivially_copy_constructible, _Tp) || _CCCL_TRAIT(is_same, _Tp, void))
      && _CCCL_TRAIT(is_trivially_copy_constructible, _Err)
      && (_CCCL_TRAIT(is_trivially_copy_assignable, _Tp) || _CCCL_TRAIT(is_same, _Tp, void))
      && _CCCL_TRAIT(is_trivially_copy_assignable, _Err)
      && (_CCCL_TRAIT(is_nothrow_move_constructible, _Tp) || _CCCL_TRAIT(is_same, _Tp, void)
          || _CCCL_TRAIT(is_nothrow_move_constructible, _Err))
    ? __smf_availability::__trivial
  : (_CCCL_TRAIT(is_copy_constructible, _Tp) || _CCCL_TRAIT(is_same, _Tp, void))
      && _CCCL_TRAIT(is_copy_constructible, _Err)
      && (_CCCL_TRAIT(is_copy_assignable, _Tp) || _CCCL_TRAIT(is_same, _Tp, void))
      && _CCCL_TRAIT(is_copy_assignable, _Err)
      && (_CCCL_TRAIT(is_nothrow_move_constructible, _Tp) || _CCCL_TRAIT(is_same, _Tp, void)
          || _CCCL_TRAIT(is_nothrow_move_constructible, _Err))
    ? __smf_availability::__available
    : __smf_availability::__deleted;

template <class _Tp, class _Err, __smf_availability = __expected_can_copy_assign<_Tp, _Err>>
struct __expected_copy_assign : __expected_move<_Tp, _Err>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__expected_copy_assign, __expected_move, _Tp, _Err);
};

template <class _Tp, class _Err>
struct __expected_copy_assign<_Tp, _Err, __smf_availability::__available> : __expected_move<_Tp, _Err>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__expected_copy_assign, __expected_move, _Tp, _Err);

  _CCCL_HIDE_FROM_ABI __expected_copy_assign(const __expected_copy_assign&) = default;
  _CCCL_HIDE_FROM_ABI __expected_copy_assign(__expected_copy_assign&&)      = default;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 __expected_copy_assign&
  operator=(const __expected_copy_assign& __other) noexcept(
    _CCCL_TRAIT(is_nothrow_copy_assignable, _Tp) && _CCCL_TRAIT(is_nothrow_copy_constructible, _Tp)
    && _CCCL_TRAIT(is_nothrow_copy_assignable, _Err)
    && _CCCL_TRAIT(is_nothrow_copy_constructible, _Err)) // strengthened
  {
    if (this->__has_val_ && __other.__has_val_)
    {
      this->__union_.__val_ = __other.__union_.__val_;
    }
    else if (this->__has_val_ && !__other.__has_val_)
    {
      this->__reinit_expected(this->__union_.__unex_, this->__union_.__val_, __other.__union_.__unex_);
      this->__has_val_ = false;
    }
    else if (!this->__has_val_ && __other.__has_val_)
    {
      this->__reinit_expected(this->__union_.__val_, this->__union_.__unex_, __other.__union_.__val_);
      this->__has_val_ = true;
    }
    else
    { // !this->__has_val_ && !__other.__has_val_
      this->__union_.__unex_ = __other.__union_.__unex_;
    }
    return *this;
  }

  _CCCL_HIDE_FROM_ABI __expected_copy_assign& operator=(__expected_copy_assign&&) = default;
};

template <class _Tp, class _Err>
struct __expected_copy_assign<_Tp, _Err, __smf_availability::__deleted> : __expected_move<_Tp, _Err>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__expected_copy_assign, __expected_move, _Tp, _Err);

  _CCCL_HIDE_FROM_ABI __expected_copy_assign(const __expected_copy_assign&)       = default;
  _CCCL_HIDE_FROM_ABI __expected_copy_assign(__expected_copy_assign&&)            = default;
  __expected_copy_assign& operator=(const __expected_copy_assign&)                = delete;
  _CCCL_HIDE_FROM_ABI __expected_copy_assign& operator=(__expected_copy_assign&&) = default;
};

template <class _Tp, class _Err>
inline constexpr __smf_availability __expected_can_move_assign =
  (_CCCL_TRAIT(is_trivially_destructible, _Tp) || _CCCL_TRAIT(is_same, _Tp, void))
      && _CCCL_TRAIT(is_trivially_destructible, _Err)
      && (_CCCL_TRAIT(is_trivially_move_constructible, _Tp) || _CCCL_TRAIT(is_same, _Tp, void))
      && _CCCL_TRAIT(is_trivially_move_constructible, _Err)
      && (_CCCL_TRAIT(is_trivially_move_assignable, _Tp) || _CCCL_TRAIT(is_same, _Tp, void))
      && _CCCL_TRAIT(is_trivially_move_assignable, _Err)
    ? __smf_availability::__trivial
  : (_CCCL_TRAIT(is_move_constructible, _Tp) || _CCCL_TRAIT(is_same, _Tp, void))
      && _CCCL_TRAIT(is_move_constructible, _Err)
      && (_CCCL_TRAIT(is_move_assignable, _Tp) || _CCCL_TRAIT(is_same, _Tp, void))
      && _CCCL_TRAIT(is_move_assignable, _Err)
      && (_CCCL_TRAIT(is_nothrow_move_constructible, _Tp) || _CCCL_TRAIT(is_same, _Tp, void)
          || _CCCL_TRAIT(is_nothrow_move_constructible, _Err))
    ? __smf_availability::__available
    : __smf_availability::__deleted;

template <class _Tp, class _Err, __smf_availability = __expected_can_move_assign<_Tp, _Err>>
struct __expected_move_assign : __expected_copy_assign<_Tp, _Err>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__expected_move_assign, __expected_copy_assign, _Tp, _Err);
};

template <class _Tp, class _Err>
struct __expected_move_assign<_Tp, _Err, __smf_availability::__available> : __expected_copy_assign<_Tp, _Err>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__expected_move_assign, __expected_copy_assign, _Tp, _Err);

  _CCCL_HIDE_FROM_ABI __expected_move_assign(const __expected_move_assign&)            = default;
  _CCCL_HIDE_FROM_ABI __expected_move_assign(__expected_move_assign&&)                 = default;
  _CCCL_HIDE_FROM_ABI __expected_move_assign& operator=(const __expected_move_assign&) = default;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 __expected_move_assign& operator=(__expected_move_assign&& __other) noexcept(
    _CCCL_TRAIT(is_nothrow_move_assignable, _Tp) && _CCCL_TRAIT(is_nothrow_move_constructible, _Tp)
    && _CCCL_TRAIT(is_nothrow_move_assignable, _Err)
    && _CCCL_TRAIT(is_nothrow_move_constructible, _Err)) // strengthened
  {
    if (this->__has_val_ && __other.__has_val_)
    {
      this->__union_.__val_ = _CUDA_VSTD::move(__other.__union_.__val_);
    }
    else if (this->__has_val_ && !__other.__has_val_)
    {
      this->__reinit_expected(this->__union_.__unex_, this->__union_.__val_, _CUDA_VSTD::move(__other.__union_.__unex_));
      this->__has_val_ = false;
    }
    else if (!this->__has_val_ && __other.__has_val_)
    {
      this->__reinit_expected(this->__union_.__val_, this->__union_.__unex_, _CUDA_VSTD::move(__other.__union_.__val_));
      this->__has_val_ = true;
    }
    else
    { // !this->__has_val_ && !__other.__has_val_
      this->__union_.__unex_ = _CUDA_VSTD::move(__other.__union_.__unex_);
    }
    return *this;
  }
};

template <class _Tp, class _Err>
struct __expected_move_assign<_Tp, _Err, __smf_availability::__deleted> : __expected_copy_assign<_Tp, _Err>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__expected_move_assign, __expected_copy_assign, _Tp, _Err);

  _CCCL_HIDE_FROM_ABI __expected_move_assign(const __expected_move_assign&)            = default;
  _CCCL_HIDE_FROM_ABI __expected_move_assign(__expected_move_assign&&)                 = default;
  _CCCL_HIDE_FROM_ABI __expected_move_assign& operator=(const __expected_move_assign&) = default;
  __expected_move_assign& operator=(__expected_move_assign&&)                          = delete;
};

// expected<void, E> base classtemplate <class _Tp, class _Err>
// MSVC complains about [[no_unique_address]] prior to C++20 as a vendor extension
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4848)

template <class _Err>
struct __expected_destruct<void, _Err, false, false>
{
  _CCCL_NO_UNIQUE_ADDRESS union __expected_union_t
  {
    struct __empty_t
    {};

    _CCCL_API constexpr __expected_union_t() noexcept
        : __empty_()
    {}

    _CCCL_EXEC_CHECK_DISABLE
    template <class... _Args>
    _CCCL_API constexpr __expected_union_t(unexpect_t, _Args&&... __args) noexcept(
      _CCCL_TRAIT(is_nothrow_constructible, _Err, _Args...))
        : __unex_(_CUDA_VSTD::forward<_Args>(__args)...)
    {}

    _CCCL_EXEC_CHECK_DISABLE
    template <class _Fun, class... _Args>
    _CCCL_API constexpr __expected_union_t(
      __expected_construct_from_invoke_tag,
      unexpect_t,
      _Fun&& __fun,
      _Args&&... __args) noexcept(_CCCL_TRAIT(is_nothrow_constructible, _Err, invoke_result_t<_Fun, _Args...>))
        : __unex_(_CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun), _CUDA_VSTD::forward<_Args>(__args)...))
    {}

    // the __expected_destruct's destructor handles this
    _CCCL_EXEC_CHECK_DISABLE
    _CCCL_API inline _CCCL_CONSTEXPR_CXX20 ~__expected_union_t() {}

    _CCCL_NO_UNIQUE_ADDRESS __empty_t __empty_;
    _CCCL_NO_UNIQUE_ADDRESS _Err __unex_;
  } __union_{};
  bool __has_val_{true};

  _CCCL_HIDE_FROM_ABI constexpr __expected_destruct() = default;

  _CCCL_API constexpr __expected_destruct(const bool __has_val) noexcept
      : __has_val_(__has_val)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class... _Args>
  _CCCL_API constexpr __expected_destruct(unexpect_t, _Args&&... __args) noexcept(
    _CCCL_TRAIT(is_nothrow_constructible, _Err, _Args...))
      : __union_(unexpect, _CUDA_VSTD::forward<_Args>(__args)...)
      , __has_val_(false)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Fun, class... _Args>
  _CCCL_API constexpr __expected_destruct(
    __expected_construct_from_invoke_tag,
    unexpect_t,
    _Fun&& __fun,
    _Args&&... __args) noexcept(_CCCL_TRAIT(is_nothrow_constructible, _Err, invoke_result_t<_Fun, _Args...>))
      : __union_(__expected_construct_from_invoke_tag{},
                 unexpect,
                 _CUDA_VSTD::forward<_Fun>(__fun),
                 _CUDA_VSTD::forward<_Args>(__args)...)
      , __has_val_(false)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 ~__expected_destruct()
  {
    if (!__has_val_)
    {
      __union_.__unex_.~_Err();
    }
  }
};

template <class _Err>
struct __expected_destruct<void, _Err, false, true>
{
  // Using `_CCCL_NO_UNIQUE_ADDRESS` here crashes nvcc
  /* _CCCL_NO_UNIQUE_ADDRESS */ union __expected_union_t
  {
    struct __empty_t
    {};

    _CCCL_API constexpr __expected_union_t() noexcept
        : __empty_()
    {}

    _CCCL_EXEC_CHECK_DISABLE
    template <class... _Args>
    _CCCL_API constexpr __expected_union_t(unexpect_t, _Args&&... __args) noexcept(
      _CCCL_TRAIT(is_nothrow_constructible, _Err, _Args...))
        : __unex_(_CUDA_VSTD::forward<_Args>(__args)...)
    {}

    _CCCL_EXEC_CHECK_DISABLE
    template <class _Fun, class... _Args>
    _CCCL_API constexpr __expected_union_t(
      __expected_construct_from_invoke_tag,
      unexpect_t,
      _Fun&& __fun,
      _Args&&... __args) noexcept(_CCCL_TRAIT(is_nothrow_constructible, _Err, invoke_result_t<_Fun, _Args...>))
        : __unex_(_CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fun>(__fun), _CUDA_VSTD::forward<_Args>(__args)...))
    {}

    _CCCL_NO_UNIQUE_ADDRESS __empty_t __empty_;
    _CCCL_NO_UNIQUE_ADDRESS _Err __unex_;
  } __union_{};
  bool __has_val_{true};

  _CCCL_HIDE_FROM_ABI constexpr __expected_destruct() = default;

  _CCCL_EXEC_CHECK_DISABLE
  template <class... _Args>
  _CCCL_API constexpr __expected_destruct(in_place_t) noexcept(_CCCL_TRAIT(is_nothrow_constructible, _Err, _Args...))
      : __union_()
      , __has_val_(true)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class... _Args>
  _CCCL_API constexpr __expected_destruct(unexpect_t, _Args&&... __args) noexcept(
    _CCCL_TRAIT(is_nothrow_constructible, _Err, _Args...))
      : __union_(unexpect, _CUDA_VSTD::forward<_Args>(__args)...)
      , __has_val_(false)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Fun, class... _Args>
  _CCCL_API constexpr __expected_destruct(
    __expected_construct_from_invoke_tag,
    unexpect_t,
    _Fun&& __fun,
    _Args&&... __args) noexcept(_CCCL_TRAIT(is_nothrow_constructible, _Err, invoke_result_t<_Fun, _Args...>))
      : __union_(__expected_construct_from_invoke_tag{},
                 unexpect,
                 _CUDA_VSTD::forward<_Fun>(__fun),
                 _CUDA_VSTD::forward<_Args>(__args)...)
      , __has_val_(false)
  {}

  _CCCL_API constexpr __expected_destruct(const bool __has_val) noexcept
      : __has_val_(__has_val)
  {}
};

_CCCL_DIAG_POP

template <class _Err>
struct __expected_storage<void, _Err> : __expected_destruct<void, _Err>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__expected_storage, __expected_destruct, void, _Err);

  _CCCL_EXEC_CHECK_DISABLE
  static _CCCL_API inline _CCCL_CONSTEXPR_CXX20 void __swap_val_unex_impl(
    __expected_storage& __with_val,
    __expected_storage& __with_err) noexcept(_CCCL_TRAIT(is_nothrow_move_constructible, _Err))
  {
    _CUDA_VSTD::__construct_at(
      _CUDA_VSTD::addressof(__with_val.__union_.__unex_), _CUDA_VSTD::move(__with_err.__union_.__unex_));
    _CUDA_VSTD::__destroy_at(_CUDA_VSTD::addressof(__with_err.__union_.__unex_));
    __with_val.__has_val_ = false;
    __with_err.__has_val_ = true;
  }
};

template <class _Err>
struct __expected_copy<void, _Err, __smf_availability::__available> : __expected_storage<void, _Err>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__expected_copy, __expected_storage, void, _Err);

  _CCCL_API inline _CCCL_CONSTEXPR_CXX20
  __expected_copy(const __expected_copy& __other) noexcept(_CCCL_TRAIT(is_nothrow_copy_constructible, _Err))
      : __base(__other.__has_val_)
  {
    if (!__other.__has_val_)
    {
      _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(this->__union_.__unex_), __other.__union_.__unex_);
    }
  }

  _CCCL_HIDE_FROM_ABI __expected_copy(__expected_copy&&)                 = default;
  _CCCL_HIDE_FROM_ABI __expected_copy& operator=(const __expected_copy&) = default;
  _CCCL_HIDE_FROM_ABI __expected_copy& operator=(__expected_copy&&)      = default;
};

template <class _Err>
struct __expected_move<void, _Err, __smf_availability::__available> : __expected_copy<void, _Err>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__expected_move, __expected_copy, void, _Err);

  _CCCL_HIDE_FROM_ABI __expected_move(const __expected_move&) = default;

  _CCCL_API inline _CCCL_CONSTEXPR_CXX20
  __expected_move(__expected_move&& __other) noexcept(_CCCL_TRAIT(is_nothrow_move_constructible, _Err))
      : __base(__other.__has_val_)
  {
    if (!__other.__has_val_)
    {
      _CUDA_VSTD::__construct_at(
        _CUDA_VSTD::addressof(this->__union_.__unex_), _CUDA_VSTD::move(__other.__union_.__unex_));
    }
  }

  _CCCL_HIDE_FROM_ABI __expected_move& operator=(const __expected_move&) = default;
  _CCCL_HIDE_FROM_ABI __expected_move& operator=(__expected_move&&)      = default;
};

template <class _Err>
struct __expected_copy_assign<void, _Err, __smf_availability::__available> : __expected_move<void, _Err>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__expected_copy_assign, __expected_move, void, _Err);

  _CCCL_HIDE_FROM_ABI __expected_copy_assign(const __expected_copy_assign&) = default;
  _CCCL_HIDE_FROM_ABI __expected_copy_assign(__expected_copy_assign&&)      = default;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 __expected_copy_assign&
  operator=(const __expected_copy_assign& __other) noexcept(
    _CCCL_TRAIT(is_nothrow_copy_assignable, _Err) && _CCCL_TRAIT(is_nothrow_copy_constructible, _Err)) // strengthened
  {
    if (this->__has_val_ && __other.__has_val_)
    {
      // nothing to do
    }
    else if (this->__has_val_ && !__other.__has_val_)
    {
      _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(this->__union_.__unex_), __other.__union_.__unex_);
      this->__has_val_ = false;
    }
    else if (!this->__has_val_ && __other.__has_val_)
    {
      _CUDA_VSTD::__destroy_at(_CUDA_VSTD::addressof(this->__union_.__unex_));
      this->__has_val_ = true;
    }
    else
    { // !this->__has_val_ && !__other.__has_val_
      this->__union_.__unex_ = __other.__union_.__unex_;
    }
    return *this;
  }

  _CCCL_HIDE_FROM_ABI __expected_copy_assign& operator=(__expected_copy_assign&&) = default;
};

template <class _Err>
struct __expected_move_assign<void, _Err, __smf_availability::__available> : __expected_copy_assign<void, _Err>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__expected_move_assign, __expected_copy_assign, void, _Err);

  _CCCL_HIDE_FROM_ABI __expected_move_assign(const __expected_move_assign&)            = default;
  _CCCL_HIDE_FROM_ABI __expected_move_assign(__expected_move_assign&&)                 = default;
  _CCCL_HIDE_FROM_ABI __expected_move_assign& operator=(const __expected_move_assign&) = default;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 __expected_move_assign& operator=(__expected_move_assign&& __other) noexcept(
    _CCCL_TRAIT(is_nothrow_move_assignable, _Err) && _CCCL_TRAIT(is_nothrow_move_constructible, _Err)) // strengthened
  {
    if (this->__has_val_ && __other.__has_val_)
    {
      // nothing to do
    }
    else if (this->__has_val_ && !__other.__has_val_)
    {
      _CUDA_VSTD::__construct_at(
        _CUDA_VSTD::addressof(this->__union_.__unex_), _CUDA_VSTD::move(__other.__union_.__unex_));
      this->__has_val_ = false;
    }
    else if (!this->__has_val_ && __other.__has_val_)
    {
      _CUDA_VSTD::__destroy_at(_CUDA_VSTD::addressof(this->__union_.__unex_));
      this->__has_val_ = true;
    }
    else
    { // !this->__has_val_ && !__other.__has_val_
      this->__union_.__unex_ = _CUDA_VSTD::move(__other.__union_.__unex_);
    }
    return *this;
  }
};

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___EXPECTED_EXPECTED_BASE_H
