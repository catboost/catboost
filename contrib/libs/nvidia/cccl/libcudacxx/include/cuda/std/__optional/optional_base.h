//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___OPTIONAL_OPTIONAL_BASE_H
#define _LIBCUDACXX___OPTIONAL_OPTIONAL_BASE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__memory/construct_at.h>
#include <cuda/std/__tuple_dir/sfinae_helpers.h>
#include <cuda/std/__type_traits/is_copy_assignable.h>
#include <cuda/std/__type_traits/is_copy_constructible.h>
#include <cuda/std/__type_traits/is_destructible.h>
#include <cuda/std/__type_traits/is_move_assignable.h>
#include <cuda/std/__type_traits/is_move_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_copy_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/is_object.h>
#include <cuda/std/__type_traits/is_trivially_copy_assignable.h>
#include <cuda/std/__type_traits/is_trivially_copy_constructible.h>
#include <cuda/std/__type_traits/is_trivially_destructible.h>
#include <cuda/std/__type_traits/is_trivially_move_assignable.h>
#include <cuda/std/__type_traits/is_trivially_move_constructible.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/in_place.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

struct __optional_construct_from_invoke_tag
{};

template <class _Tp, bool = is_trivially_destructible_v<_Tp>>
struct __optional_destruct_base;

template <class _Tp>
struct __optional_destruct_base<_Tp, false>
{
  using value_type = _Tp;
  static_assert(_CCCL_TRAIT(is_object, value_type),
                "instantiation of optional with a non-object type is undefined behavior");
  union __storage
  {
    char __null_state_;
    remove_cv_t<value_type> __val_;

    _CCCL_API constexpr __storage() noexcept
        : __null_state_()
    {}
    _CCCL_EXEC_CHECK_DISABLE
    template <class... _Args>
    _CCCL_API constexpr __storage(in_place_t,
                                  _Args&&... __args) noexcept(is_nothrow_constructible_v<value_type, _Args...>)
        : __val_(_CUDA_VSTD::forward<_Args>(__args)...)
    {}
    _CCCL_EXEC_CHECK_DISABLE
    template <class _Fp, class... _Args>
    _CCCL_API constexpr __storage(__optional_construct_from_invoke_tag, _Fp&& __f, _Args&&... __args)
        : __val_(_CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fp>(__f), _CUDA_VSTD::forward<_Args>(__args)...))
    {}
    _CCCL_API _CCCL_CONSTEXPR_CXX20 ~__storage() noexcept {}
  };
  __storage __storage_;
  bool __engaged_;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 ~__optional_destruct_base()
  {
    if (__engaged_)
    {
      __storage_.__val_.~value_type();
    }
  }

  _CCCL_API constexpr __optional_destruct_base() noexcept
      : __storage_()
      , __engaged_(false)
  {}

  template <class... _Args>
  _CCCL_API constexpr explicit __optional_destruct_base(in_place_t, _Args&&... __args) noexcept(
    is_nothrow_constructible_v<value_type, _Args...>)
      : __storage_(in_place, _CUDA_VSTD::forward<_Args>(__args)...)
      , __engaged_(true)
  {}

  template <class _Fp, class... _Args>
  _CCCL_API constexpr __optional_destruct_base(__optional_construct_from_invoke_tag, _Fp&& __f, _Args&&... __args)
      : __storage_(
          __optional_construct_from_invoke_tag{}, _CUDA_VSTD::forward<_Fp>(__f), _CUDA_VSTD::forward<_Args>(__args)...)
      , __engaged_(true)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 void reset() noexcept
  {
    if (__engaged_)
    {
      __storage_.__val_.~value_type();
      __engaged_ = false;
    }
  }
};

template <class _Tp>
struct __optional_destruct_base<_Tp, true>
{
  using value_type = _Tp;
  static_assert(_CCCL_TRAIT(is_object, value_type),
                "instantiation of optional with a non-object type is undefined behavior");
  union __storage
  {
    char __null_state_;
    remove_cv_t<value_type> __val_;

    _CCCL_API constexpr __storage() noexcept
        : __null_state_()
    {}
    _CCCL_EXEC_CHECK_DISABLE
    template <class... _Args>
    _CCCL_API constexpr __storage(in_place_t,
                                  _Args&&... __args) noexcept(is_nothrow_constructible_v<value_type, _Args...>)
        : __val_(_CUDA_VSTD::forward<_Args>(__args)...)
    {}
    _CCCL_EXEC_CHECK_DISABLE
    template <class _Fp, class... _Args>
    _CCCL_API constexpr __storage(__optional_construct_from_invoke_tag, _Fp&& __f, _Args&&... __args)
        : __val_(_CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Fp>(__f), _CUDA_VSTD::forward<_Args>(__args)...))
    {}
  };
  __storage __storage_;
  bool __engaged_;

  _CCCL_API constexpr __optional_destruct_base() noexcept
      : __storage_()
      , __engaged_(false)
  {}

  template <class... _Args>
  _CCCL_API constexpr explicit __optional_destruct_base(in_place_t, _Args&&... __args) noexcept(
    is_nothrow_constructible_v<value_type, _Args...>)
      : __storage_(in_place, _CUDA_VSTD::forward<_Args>(__args)...)
      , __engaged_(true)
  {}

  template <class _Fp, class... _Args>
  _CCCL_API constexpr __optional_destruct_base(__optional_construct_from_invoke_tag, _Fp&& __f, _Args&&... __args)
      : __storage_(
          __optional_construct_from_invoke_tag{}, _CUDA_VSTD::forward<_Fp>(__f), _CUDA_VSTD::forward<_Args>(__args)...)
      , __engaged_(true)
  {}

  _CCCL_API constexpr void reset() noexcept
  {
    if (__engaged_)
    {
      __engaged_ = false;
    }
  }
};

template <class _Tp>
struct __optional_storage_base : __optional_destruct_base<_Tp>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__optional_storage_base, __optional_destruct_base, _Tp);

  using value_type = _Tp;

  [[nodiscard]] _CCCL_API constexpr bool has_value() const noexcept
  {
    return this->__engaged_;
  }

  [[nodiscard]] _CCCL_API constexpr value_type& __get() & noexcept
  {
    return this->__storage_.__val_;
  }
  [[nodiscard]] _CCCL_API constexpr const value_type& __get() const& noexcept
  {
    return this->__storage_.__val_;
  }
  [[nodiscard]] _CCCL_API constexpr value_type&& __get() && noexcept
  {
    return _CUDA_VSTD::move(this->__storage_.__val_);
  }
  [[nodiscard]] _CCCL_API constexpr const value_type&& __get() const&& noexcept
  {
    return _CUDA_VSTD::move(this->__storage_.__val_);
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class... _Args>
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 void __construct(_Args&&... __args)
  {
    _CCCL_ASSERT(!has_value(), "__construct called for engaged __optional_storage");
    _CUDA_VSTD::__construct_at(_CUDA_VSTD::addressof(this->__storage_.__val_), _CUDA_VSTD::forward<_Args>(__args)...);
    this->__engaged_ = true;
  }

  template <class _That>
  _CCCL_API constexpr void __construct_from(_That&& __opt)
  {
    if (__opt.has_value())
    {
      __construct(_CUDA_VSTD::forward<_That>(__opt).__get());
    }
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class _That>
  _CCCL_API constexpr void __assign_from(_That&& __opt)
  {
    if (this->__engaged_ == __opt.has_value())
    {
      if (this->__engaged_)
      {
        this->__storage_.__val_ = _CUDA_VSTD::forward<_That>(__opt).__get();
      }
    }
    else
    {
      if (this->__engaged_)
      {
        this->reset();
      }
      else
      {
        __construct(_CUDA_VSTD::forward<_That>(__opt).__get());
      }
    }
  }
};

template <class _Tp>
inline constexpr __smf_availability __optional_can_copy_construct =
  _CCCL_TRAIT(is_trivially_copy_constructible, _Tp) ? __smf_availability::__trivial
  : _CCCL_TRAIT(is_copy_constructible, _Tp)
    ? __smf_availability::__available
    : __smf_availability::__deleted;

template <class _Tp, __smf_availability = __optional_can_copy_construct<_Tp>>
struct __optional_copy_base : __optional_storage_base<_Tp>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__optional_copy_base, __optional_storage_base, _Tp);
};

template <class _Tp>
struct __optional_copy_base<_Tp, __smf_availability::__available> : __optional_storage_base<_Tp>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__optional_copy_base, __optional_storage_base, _Tp);

  // This ctor shouldn't need to initialize the base explicitly, but g++ 9 considers it to be uninitialized
  // during constexpr evaluation if it isn't initialized explicitly. This can be replaced with the pattern
  // below, in __optional_move_base, once g++ 9 falls off our support matrix.
  _CCCL_API constexpr __optional_copy_base(const __optional_copy_base& __opt) noexcept(
    is_nothrow_copy_constructible_v<_Tp>)
      : __base()
  {
    this->__construct_from(__opt);
  }

  _CCCL_HIDE_FROM_ABI __optional_copy_base(__optional_copy_base&&)                 = default;
  _CCCL_HIDE_FROM_ABI __optional_copy_base& operator=(const __optional_copy_base&) = default;
  _CCCL_HIDE_FROM_ABI __optional_copy_base& operator=(__optional_copy_base&&)      = default;
};

template <class _Tp>
struct __optional_copy_base<_Tp, __smf_availability::__deleted> : __optional_storage_base<_Tp>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__optional_copy_base, __optional_storage_base, _Tp);
  _CCCL_HIDE_FROM_ABI __optional_copy_base(const __optional_copy_base&)            = delete;
  _CCCL_HIDE_FROM_ABI __optional_copy_base(__optional_copy_base&&)                 = default;
  _CCCL_HIDE_FROM_ABI __optional_copy_base& operator=(const __optional_copy_base&) = default;
  _CCCL_HIDE_FROM_ABI __optional_copy_base& operator=(__optional_copy_base&&)      = default;
};

template <class _Tp>
inline constexpr __smf_availability __optional_can_move_construct =
  _CCCL_TRAIT(is_trivially_move_constructible, _Tp) ? __smf_availability::__trivial
  : _CCCL_TRAIT(is_move_constructible, _Tp)
    ? __smf_availability::__available
    : __smf_availability::__deleted;

template <class _Tp, __smf_availability = __optional_can_move_construct<_Tp>>
struct __optional_move_base : __optional_copy_base<_Tp>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__optional_move_base, __optional_copy_base, _Tp);
};

template <class _Tp>
struct __optional_move_base<_Tp, __smf_availability::__available> : __optional_copy_base<_Tp>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__optional_move_base, __optional_copy_base, _Tp);

  _CCCL_HIDE_FROM_ABI __optional_move_base(const __optional_move_base&) = default;

  _CCCL_API constexpr __optional_move_base(__optional_move_base&& __opt) noexcept(
    _CCCL_TRAIT(is_nothrow_move_constructible, _Tp))
  {
    this->__construct_from(_CUDA_VSTD::move(__opt));
  }

  _CCCL_HIDE_FROM_ABI __optional_move_base& operator=(const __optional_move_base&) = default;
  _CCCL_HIDE_FROM_ABI __optional_move_base& operator=(__optional_move_base&&)      = default;
};

template <class _Tp>
struct __optional_move_base<_Tp, __smf_availability::__deleted> : __optional_copy_base<_Tp>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__optional_move_base, __optional_copy_base, _Tp);

  _CCCL_HIDE_FROM_ABI __optional_move_base(const __optional_move_base&)            = default;
  _CCCL_HIDE_FROM_ABI __optional_move_base(__optional_move_base&&)                 = delete;
  _CCCL_HIDE_FROM_ABI __optional_move_base& operator=(const __optional_move_base&) = default;
  _CCCL_HIDE_FROM_ABI __optional_move_base& operator=(__optional_move_base&&)      = default;
};

template <class _Tp>
inline constexpr __smf_availability __optional_can_copy_assign =
  _CCCL_TRAIT(is_trivially_destructible, _Tp) && _CCCL_TRAIT(is_trivially_copy_constructible, _Tp)
      && _CCCL_TRAIT(is_trivially_copy_assignable, _Tp)
    ? __smf_availability::__trivial
  : _CCCL_TRAIT(is_destructible, _Tp) && _CCCL_TRAIT(is_copy_constructible, _Tp) && _CCCL_TRAIT(is_copy_assignable, _Tp)
    ? __smf_availability::__available
    : __smf_availability::__deleted;

template <class _Tp, __smf_availability = __optional_can_copy_assign<_Tp>>
struct __optional_copy_assign_base : __optional_move_base<_Tp>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__optional_copy_assign_base, __optional_move_base, _Tp);
};

template <class _Tp>
struct __optional_copy_assign_base<_Tp, __smf_availability::__available> : __optional_move_base<_Tp>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__optional_copy_assign_base, __optional_move_base, _Tp);

  _CCCL_HIDE_FROM_ABI __optional_copy_assign_base(const __optional_copy_assign_base&) = default;
  _CCCL_HIDE_FROM_ABI __optional_copy_assign_base(__optional_copy_assign_base&&)      = default;

  _CCCL_API constexpr __optional_copy_assign_base& operator=(const __optional_copy_assign_base& __opt)
  {
    this->__assign_from(__opt);
    return *this;
  }

  __optional_copy_assign_base& operator=(__optional_copy_assign_base&&) = default;
};

template <class _Tp>
struct __optional_copy_assign_base<_Tp, __smf_availability::__deleted> : __optional_move_base<_Tp>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__optional_copy_assign_base, __optional_move_base, _Tp);

  _CCCL_HIDE_FROM_ABI __optional_copy_assign_base(const __optional_copy_assign_base&)            = default;
  _CCCL_HIDE_FROM_ABI __optional_copy_assign_base(__optional_copy_assign_base&&)                 = default;
  _CCCL_HIDE_FROM_ABI __optional_copy_assign_base& operator=(const __optional_copy_assign_base&) = delete;
  _CCCL_HIDE_FROM_ABI __optional_copy_assign_base& operator=(__optional_copy_assign_base&&)      = default;
};

template <class _Tp>
inline constexpr __smf_availability __optional_can_move_assign =
  _CCCL_TRAIT(is_trivially_destructible, _Tp) && _CCCL_TRAIT(is_trivially_move_constructible, _Tp)
      && _CCCL_TRAIT(is_trivially_move_assignable, _Tp)
    ? __smf_availability::__trivial
  : _CCCL_TRAIT(is_destructible, _Tp) && _CCCL_TRAIT(is_move_constructible, _Tp) && _CCCL_TRAIT(is_move_assignable, _Tp)
    ? __smf_availability::__available
    : __smf_availability::__deleted;

template <class _Tp, __smf_availability = __optional_can_move_assign<_Tp>>
struct __optional_move_assign_base : __optional_copy_assign_base<_Tp>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__optional_move_assign_base, __optional_copy_assign_base, _Tp);
};

template <class _Tp>
struct __optional_move_assign_base<_Tp, __smf_availability::__available> : __optional_copy_assign_base<_Tp>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__optional_move_assign_base, __optional_copy_assign_base, _Tp);

  _CCCL_HIDE_FROM_ABI __optional_move_assign_base(const __optional_move_assign_base& __opt)      = default;
  _CCCL_HIDE_FROM_ABI __optional_move_assign_base(__optional_move_assign_base&&)                 = default;
  _CCCL_HIDE_FROM_ABI __optional_move_assign_base& operator=(const __optional_move_assign_base&) = default;

  _CCCL_API constexpr __optional_move_assign_base& operator=(__optional_move_assign_base&& __opt) noexcept(
    _CCCL_TRAIT(is_nothrow_move_assignable, _Tp) && _CCCL_TRAIT(is_nothrow_move_constructible, _Tp))
  {
    this->__assign_from(_CUDA_VSTD::move(__opt));
    return *this;
  }
};

template <class _Tp>
struct __optional_move_assign_base<_Tp, __smf_availability::__deleted> : __optional_copy_assign_base<_Tp>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__optional_move_assign_base, __optional_copy_assign_base, _Tp);

  _CCCL_HIDE_FROM_ABI __optional_move_assign_base(const __optional_move_assign_base& __opt)      = default;
  _CCCL_HIDE_FROM_ABI __optional_move_assign_base(__optional_move_assign_base&&)                 = default;
  _CCCL_HIDE_FROM_ABI __optional_move_assign_base& operator=(const __optional_move_assign_base&) = default;
  _CCCL_HIDE_FROM_ABI __optional_move_assign_base& operator=(__optional_move_assign_base&&)      = delete;
};

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___OPTIONAL_OPTIONAL_BASE_H
