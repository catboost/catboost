// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___RANGES_MOVABLE_BOX_H
#define _LIBCUDACXX___RANGES_MOVABLE_BOX_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__concepts/copyable.h>
#include <cuda/std/__concepts/movable.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__memory/construct_at.h>
#include <cuda/std/__tuple_dir/sfinae_helpers.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/optional>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

// __movable_box allows turning a type that is move-constructible (but maybe not move-assignable) into
// a type that is both move-constructible and move-assignable. It does that by introducing an empty state
// and basically doing destroy-then-copy-construct in the assignment operator. The empty state is necessary
// to handle the case where the copy construction fails after destroying the object.
//
// In some cases, we can completely avoid the use of an empty state; we provide a specialization of
// __movable_box that does this, see below for the details.

// until C++23, `__movable_box` was named `__copyable_box` and required the stored type to be copy-constructible, not
// just move-constructible; we always use the C++23 behavior
template <class _Tp>
_CCCL_CONCEPT __movable_box_object = move_constructible<_Tp> && is_object_v<_Tp>;

// The partial specialization implements an optimization for when we know we don't need to store
// an empty state to represent failure to perform an assignment. For copy-assignment, this happens:
//
// 1. If the type is copyable (which includes copy-assignment), we can use the type's own assignment operator
//    directly and avoid using _CUDA_VSTD::optional.
// 2. If the type is not copyable, but it is nothrow-copy-constructible, then we can implement assignment as
//    destroy-and-then-construct and we know it will never fail, so we don't need an empty state.
//
// The exact same reasoning can be applied for move-assignment, with copyable replaced by movable and
// nothrow-copy-constructible replaced by nothrow-move-constructible. This specialization is enabled
// whenever we can apply any of these optimizations for both the copy assignment and the move assignment
// operator.
template <class _Tp>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL bool __doesnt_need_empty_state() noexcept
{
  if constexpr (copy_constructible<_Tp>)
  {
    // 1. If copy_constructible<T> is true, movable-box<T> should store only a T if either T models
    //    copyable, or is_nothrow_move_constructible_v<T> && is_nothrow_copy_constructible_v<T> is true.
    return copyable<_Tp> || (is_nothrow_move_constructible_v<_Tp> && is_nothrow_copy_constructible_v<_Tp>);
  }
  else
  {
    // 2. Otherwise, movable-box<T> should store only a T if either T models movable or
    //    is_nothrow_move_constructible_v<T> is true.
    return movable<_Tp> || is_nothrow_move_constructible_v<_Tp>;
  }
}

// When _Tp doesn't have an assignment operator, we must implement __movable_box's assignment operator
// by doing destroy_at followed by construct_at. However, that implementation strategy leads to UB if the nested
// _Tp is potentially overlapping, as it is doing a non-transparent replacement of the sub-object, which means that
// we're not considered "nested" inside the movable-box anymore, and since we're not nested within it, [basic.life]/1.5
// says that we essentially just reused the storage of the movable-box for a completely unrelated object and ended the
// movable-box's lifetime.
// https://github.com/llvm/llvm-project/issues/70494#issuecomment-1845646490
//
// Hence, when the _Tp doesn't have an assignment operator, we can't risk making it a potentially-overlapping
// subobject because of the above, and we don't use [[no_unique_address]] in that case.
template <class _Tp>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL bool __can_use_no_unique_address() noexcept
{
  if constexpr (copy_constructible<_Tp>)
  {
    return copyable<_Tp>;
  }
  else
  {
    return movable<_Tp>;
  }
}

// base class

template <class _Tp, bool = default_initializable<_Tp>>
struct __mb_optional_destruct_base
{
  _CCCL_NO_UNIQUE_ADDRESS optional<_Tp> __val_;

  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES(is_constructible_v<_Tp, _Args...>)
  _CCCL_API constexpr explicit __mb_optional_destruct_base(in_place_t, _Args&&... __args) noexcept(
    is_nothrow_constructible_v<_Tp, _Args...>)
      : __val_(in_place, _CUDA_VSTD::forward<_Args>(__args)...)
  {}
};

template <class _Tp>
struct __mb_optional_destruct_base<_Tp, true>
{
  _CCCL_NO_UNIQUE_ADDRESS optional<_Tp> __val_;

  _CCCL_API constexpr __mb_optional_destruct_base() noexcept(is_nothrow_default_constructible_v<_Tp>)
      : __val_(in_place)
  {}

  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES(is_constructible_v<_Tp, _Args...>)
  _CCCL_API constexpr explicit __mb_optional_destruct_base(in_place_t, _Args&&... __args) noexcept(
    is_nothrow_constructible_v<_Tp, _Args...>)
      : __val_(in_place, _CUDA_VSTD::forward<_Args>(__args)...)
  {}
};

template <class _Tp, bool = copy_constructible<_Tp>>
struct __mb_optional_copy_assign : __mb_optional_destruct_base<_Tp>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__mb_optional_copy_assign, __mb_optional_destruct_base, _Tp);

  _CCCL_HIDE_FROM_ABI constexpr __mb_optional_copy_assign(const __mb_optional_copy_assign&) = default;
  _CCCL_HIDE_FROM_ABI constexpr __mb_optional_copy_assign(__mb_optional_copy_assign&&)      = default;

  _CCCL_HIDE_FROM_ABI constexpr __mb_optional_copy_assign& operator=(const __mb_optional_copy_assign&) = delete;
  _CCCL_HIDE_FROM_ABI constexpr __mb_optional_copy_assign& operator=(__mb_optional_copy_assign&&)      = default;
};

template <class _Tp>
struct __mb_optional_copy_assign<_Tp, true> : __mb_optional_destruct_base<_Tp>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__mb_optional_copy_assign, __mb_optional_destruct_base, _Tp);

  _CCCL_HIDE_FROM_ABI constexpr __mb_optional_copy_assign(const __mb_optional_copy_assign&) = default;
  _CCCL_HIDE_FROM_ABI constexpr __mb_optional_copy_assign(__mb_optional_copy_assign&&)      = default;

  _CCCL_API constexpr __mb_optional_copy_assign&
  operator=(const __mb_optional_copy_assign& __other) noexcept(is_nothrow_copy_constructible_v<_Tp>)
  {
    if (this != _CUDA_VSTD::addressof(__other))
    {
      if (__other.__has_value())
      {
        this->__val_.emplace(*__other);
      }
      else
      {
        this->__val_.reset();
      }
    }
    return *this;
  };
  _CCCL_HIDE_FROM_ABI constexpr __mb_optional_copy_assign& operator=(__mb_optional_copy_assign&&) = default;
};

template <class _Tp, bool = movable<_Tp>>
struct __mb_optional_move_assign : __mb_optional_copy_assign<_Tp>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__mb_optional_move_assign, __mb_optional_copy_assign, _Tp);
};

template <class _Tp>
struct __mb_optional_move_assign<_Tp, false> : __mb_optional_copy_assign<_Tp>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__mb_optional_move_assign, __mb_optional_copy_assign, _Tp);

  _CCCL_HIDE_FROM_ABI constexpr __mb_optional_move_assign(const __mb_optional_move_assign&)            = default;
  _CCCL_HIDE_FROM_ABI constexpr __mb_optional_move_assign(__mb_optional_move_assign&&)                 = default;
  _CCCL_HIDE_FROM_ABI constexpr __mb_optional_move_assign& operator=(const __mb_optional_move_assign&) = default;

  _CCCL_API constexpr __mb_optional_move_assign&
  operator=(__mb_optional_move_assign&& __other) noexcept(is_nothrow_move_constructible_v<_Tp>)
  {
    if (this != _CUDA_VSTD::addressof(__other))
    {
      if (__other.__has_value())
      {
        this->__val_.emplace(_CUDA_VSTD::move(*__other));
      }
      else
      {
        this->__val_.reset();
      }
    }
    return *this;
  }
};

template <class _Tp>
struct __mb_optional_base
    : __mb_optional_move_assign<_Tp>
    , __sfinae_move_base<copy_constructible<_Tp>, true>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__mb_optional_base, __mb_optional_move_assign, _Tp);

  [[nodiscard]] _CCCL_API constexpr _Tp const& operator*() const noexcept
  {
    return *this->__val_;
  }
  [[nodiscard]] _CCCL_API constexpr _Tp& operator*() noexcept
  {
    return *this->__val_;
  }

  [[nodiscard]] _CCCL_API constexpr const _Tp* operator->() const noexcept
  {
    return this->__val_.operator->();
  }
  [[nodiscard]] _CCCL_API constexpr _Tp* operator->() noexcept
  {
    return this->__val_.operator->();
  }

  [[nodiscard]] _CCCL_API constexpr bool __has_value() const noexcept
  {
    return this->__val_.has_value();
  }
};

// Specialization without a boolean
template <class _Tp, bool = __can_use_no_unique_address<_Tp>()>
struct __mb_holder
{
  _Tp __val_;

  template <class... _Args>
  _CCCL_API constexpr explicit __mb_holder(in_place_t,
                                           _Args&&... __args) noexcept(is_nothrow_constructible_v<_Tp, _Args...>)
      : __val_(_CUDA_VSTD::forward<_Args>(__args)...)
  {}
};

template <class _Tp>
struct __mb_holder<_Tp, true>
{
  _CCCL_NO_UNIQUE_ADDRESS _Tp __val_;

  template <class... _Args>
  _CCCL_API constexpr explicit __mb_holder(in_place_t,
                                           _Args&&... __args) noexcept(is_nothrow_constructible_v<_Tp, _Args...>)
      : __val_(_CUDA_VSTD::forward<_Args>(__args)...)
  {}
};

template <class _Tp, bool = default_initializable<_Tp>>
struct __mb_holder_base
{
  _CCCL_NO_UNIQUE_ADDRESS __mb_holder<_Tp> __holder_;

  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES(is_constructible_v<_Tp, _Args...>)
  _CCCL_API constexpr explicit __mb_holder_base(in_place_t,
                                                _Args&&... __args) noexcept(is_nothrow_constructible_v<_Tp, _Args...>)
      : __holder_(in_place, _CUDA_VSTD::forward<_Args>(__args)...)
  {}
};

template <class _Tp>
struct __mb_holder_base<_Tp, true>
{
  _CCCL_NO_UNIQUE_ADDRESS __mb_holder<_Tp> __holder_;

  _CCCL_API constexpr __mb_holder_base() noexcept(is_nothrow_default_constructible_v<_Tp>)
      : __holder_(in_place)
  {}

  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES(is_constructible_v<_Tp, _Args...>)
  _CCCL_API constexpr explicit __mb_holder_base(in_place_t,
                                                _Args&&... __args) noexcept(is_nothrow_constructible_v<_Tp, _Args...>)
      : __holder_(in_place, _CUDA_VSTD::forward<_Args>(__args)...)
  {}
};

template <class _Tp, bool = copyable<_Tp>>
struct __mb_copy_assign : __mb_holder_base<_Tp>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__mb_copy_assign, __mb_holder_base, _Tp);
};

template <class _Tp>
struct __mb_copy_assign<_Tp, false> : __mb_holder_base<_Tp>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__mb_copy_assign, __mb_holder_base, _Tp);

  _CCCL_HIDE_FROM_ABI constexpr __mb_copy_assign(const __mb_copy_assign&) = default;
  _CCCL_HIDE_FROM_ABI constexpr __mb_copy_assign(__mb_copy_assign&&)      = default;

  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 __mb_copy_assign& operator=(const __mb_copy_assign& __other) noexcept
  {
    static_assert(is_nothrow_copy_constructible_v<_Tp>);
    static_assert(!__can_use_no_unique_address<_Tp>());
    if (this != _CUDA_VSTD::addressof(__other))
    {
      _CUDA_VSTD::__destroy_at(_CUDA_VSTD::addressof(this->__holder_.__val_));
      _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(this->__holder_.__val_), __other.__holder_.__val_);
    }
    return *this;
  };
  _CCCL_HIDE_FROM_ABI constexpr __mb_copy_assign& operator=(__mb_copy_assign&&) = default;
};

template <class _Tp, bool = movable<_Tp>>
struct __mb_move_assign : __mb_copy_assign<_Tp>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__mb_move_assign, __mb_copy_assign, _Tp);
};

template <class _Tp>
struct __mb_move_assign<_Tp, false> : __mb_copy_assign<_Tp>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__mb_move_assign, __mb_copy_assign, _Tp);

  _CCCL_HIDE_FROM_ABI constexpr __mb_move_assign(const __mb_move_assign&)            = default;
  _CCCL_HIDE_FROM_ABI constexpr __mb_move_assign(__mb_move_assign&&)                 = default;
  _CCCL_HIDE_FROM_ABI constexpr __mb_move_assign& operator=(const __mb_move_assign&) = default;

  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 __mb_move_assign&
  operator=(__mb_move_assign&& __other) noexcept(is_nothrow_move_constructible_v<_Tp>)
  {
    static_assert(is_nothrow_move_constructible_v<_Tp>);
    static_assert(!__can_use_no_unique_address<_Tp>);
    if (this != _CUDA_VSTD::addressof(__other))
    {
      _CUDA_VSTD::__destroy_at(_CUDA_VSTD::addressof(this->__holder_.__val_));
      _CUDA_VSTD::__construct_at(
        _CUDA_VSTD::addressof(this->__holder_.__val_), _CUDA_VSTD::move(__other.__holder_.__val_));
    }
    return *this;
  }
};

template <class _Tp>
struct __mb_base : __mb_move_assign<_Tp>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__mb_base, __mb_move_assign, _Tp);

  [[nodiscard]] _CCCL_API constexpr _Tp const& operator*() const noexcept
  {
    return this->__holder_.__val_;
  }
  [[nodiscard]] _CCCL_API constexpr _Tp& operator*() noexcept
  {
    return this->__holder_.__val_;
  }

  [[nodiscard]] _CCCL_API constexpr const _Tp* operator->() const noexcept
  {
    return _CUDA_VSTD::addressof(this->__holder_.__val_);
  }
  [[nodiscard]] _CCCL_API constexpr _Tp* operator->() noexcept
  {
    return _CUDA_VSTD::addressof(this->__holder_.__val_);
  }

  [[nodiscard]] _CCCL_API constexpr bool __has_value() const noexcept
  {
    return true;
  }
};

template <class _Tp>
using __movable_box_base = _If<__doesnt_need_empty_state<_Tp>(), __mb_base<_Tp>, __mb_optional_base<_Tp>>;

// Primary template - uses _CUDA_VSTD::optional and introduces an empty state in case assignment fails.
template <class _Tp, bool = __movable_box_object<_Tp>>
struct __movable_box;

template <class _Tp>
struct __movable_box<_Tp, true> : __movable_box_base<_Tp>
{
  using __base = __movable_box_base<_Tp>;

  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES(is_constructible_v<_Tp, _Args...>)
  _CCCL_API constexpr explicit __movable_box(in_place_t,
                                             _Args&&... __args) noexcept(is_nothrow_constructible_v<_Tp, _Args...>)
      : __base(in_place, _CUDA_VSTD::forward<_Args>(__args)...)
  {}

  _CCCL_HIDE_FROM_ABI constexpr __movable_box() = default;
};

_LIBCUDACXX_END_NAMESPACE_RANGES

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___RANGES_MOVABLE_BOX_H
