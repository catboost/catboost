//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___RANGES_COMPRESSED_MOVABLE_BOX_H
#define _CUDA_STD___RANGES_COMPRESSED_MOVABLE_BOX_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__concepts/copyable.h>
#include <cuda/std/__concepts/movable.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__memory/construct_at.h>
#include <cuda/std/__tuple_dir/sfinae_helpers.h>
#include <cuda/std/__type_traits/is_copy_assignable.h>
#include <cuda/std/__type_traits/is_copy_constructible.h>
#include <cuda/std/__type_traits/is_default_constructible.h>
#include <cuda/std/__type_traits/is_empty.h>
#include <cuda/std/__type_traits/is_final.h>
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
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/in_place.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

//! @brief Different alternatives for @c __compressed_box_base
// This is effectively a __movable-box__ that does not rely on `[[no_unique_address]]`
enum class __compressed_box_specialization
{
  __with_engaged, //!< We must store an empty state to represent failure of assignment
  __store_inline, //!< We can directly store the type inline because it behaves nice
  __empty_non_final, //!< We can directly store the type inline and employ EBCO because it is empty and non-final
};

//! @brief Determines which specialization of @c __compressed_box_base we can select
template <class _Tp>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL __compressed_box_specialization __compressed_box_choose()
{
  constexpr bool __can_ebco = is_empty_v<_Tp> && !is_final_v<_Tp>;
  // See __movable_box for a more in depth explanation
  if constexpr (copy_constructible<_Tp>)
  {
    // 1. If copy_constructible<T> is true, we should only store a T if either T models
    //    copyable, or is_nothrow_move_constructible_v<T> && is_nothrow_copy_constructible_v<T> is true.
    constexpr bool __can_store_inline =
      copyable<_Tp> || (is_nothrow_move_constructible_v<_Tp> && is_nothrow_copy_constructible_v<_Tp>);
    return __can_store_inline
           ? __can_ebco ? __compressed_box_specialization::__empty_non_final
                        : __compressed_box_specialization::__store_inline
           : __compressed_box_specialization::__with_engaged;
  }
  else
  {
    // 2. Otherwise, movable-box<T> should store only a T if either T models movable or
    //    is_nothrow_move_constructible_v<T> is true.
    constexpr bool __can_store_inline = movable<_Tp> || is_nothrow_move_constructible_v<_Tp>;
    return __can_store_inline
           ? __can_ebco ? __compressed_box_specialization::__empty_non_final
                        : __compressed_box_specialization::__store_inline
           : __compressed_box_specialization::__with_engaged;
  }
}

//! @brief Base class for @c __compressed_movable_box to store a T and ensure it is always copyable
template <size_t _Index, class _Tp, __compressed_box_specialization = __compressed_box_choose<_Tp>()>
struct _CCCL_DECLSPEC_EMPTY_BASES __compressed_box_base;

//! @brief Simple wrapper around a non-empty T to access it within a @c __compressed_movable_box
template <size_t _Index, class _Tp>
struct _CCCL_DECLSPEC_EMPTY_BASES __compressed_box_base<_Index, _Tp, __compressed_box_specialization::__store_inline>
{
  _Tp __elem_;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Tp2 = _Tp)
  _CCCL_REQUIRES(is_default_constructible_v<_Tp2>)
  _CCCL_API constexpr __compressed_box_base() noexcept(is_nothrow_default_constructible_v<_Tp2>)
      : __elem_()
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES(is_constructible_v<_Tp, _Args...>)
  _CCCL_API constexpr __compressed_box_base(in_place_t,
                                            _Args&&... __args) noexcept(is_nothrow_constructible_v<_Tp, _Args...>)
      : __elem_(::cuda::std::forward<_Args>(__args)...)
  {}

  [[nodiscard]] _CCCL_API constexpr _Tp& __get() noexcept
  {
    return __elem_;
  }
  [[nodiscard]] _CCCL_API constexpr const _Tp& __get() const noexcept
  {
    return __elem_;
  }

  [[nodiscard]] _CCCL_API static constexpr bool __engaged() noexcept
  {
    return true;
  }

  _CCCL_API static constexpr void __set_engaged(const bool) noexcept {}

  template <class... _Args>
  _CCCL_API constexpr void __construct(_Args&&... __args) noexcept(is_nothrow_constructible_v<_Tp, _Args...>)
  {
    ::cuda::std::__construct_at(cuda::std::addressof(__elem_), ::cuda::std::forward<_Args>(__args)...);
  }
};

//! @brief Simple wrapper around an empty T to access it within a @c __compressed_movable_box using EBCO
template <size_t _Index, class _Tp>
struct _CCCL_DECLSPEC_EMPTY_BASES
__compressed_box_base<_Index, _Tp, __compressed_box_specialization::__empty_non_final> : _Tp
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Tp2 = _Tp)
  _CCCL_REQUIRES(is_default_constructible_v<_Tp2>)
  _CCCL_API constexpr __compressed_box_base() noexcept(is_nothrow_default_constructible_v<_Tp2>)
      : _Tp()
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES(is_constructible_v<_Tp, _Args...>)
  _CCCL_API constexpr __compressed_box_base(in_place_t,
                                            _Args&&... __args) noexcept(is_nothrow_constructible_v<_Tp, _Args...>)
      : _Tp(::cuda::std::forward<_Args>(__args)...)
  {}

  [[nodiscard]] _CCCL_API constexpr _Tp& __get() noexcept
  {
    return *static_cast<_Tp*>(this);
  }
  [[nodiscard]] _CCCL_API constexpr const _Tp& __get() const noexcept
  {
    return *static_cast<const _Tp*>(this);
  }

  [[nodiscard]] _CCCL_API static constexpr bool __engaged() noexcept
  {
    return true;
  }

  _CCCL_API static constexpr void __set_engaged(const bool) noexcept {}

  template <class... _Args>
  _CCCL_API constexpr void __construct(_Args&&... __args) noexcept(is_nothrow_constructible_v<_Tp, _Args...>)
  {
    ::cuda::std::__construct_at(static_cast<_Tp*>(this), ::cuda::std::forward<_Args>(__args)...);
  }
};

template <size_t _Index, class _Tp, bool = is_trivially_destructible_v<_Tp>>
struct _CCCL_DECLSPEC_EMPTY_BASES __compressed_box_storage
{
  union __storage
  {
    char __null_state_;
    remove_cv_t<_Tp> __val_;

    _CCCL_API constexpr __storage() noexcept
        : __null_state_()
    {}
    _CCCL_EXEC_CHECK_DISABLE
    _CCCL_TEMPLATE(class... _Args)
    _CCCL_REQUIRES(is_constructible_v<_Tp, _Args...>)
    _CCCL_API constexpr __storage(in_place_t, _Args&&... __args) noexcept(is_nothrow_constructible_v<_Tp, _Args...>)
        : __val_(::cuda::std::forward<_Args>(__args)...)
    {}
    _CCCL_API _CCCL_CONSTEXPR_CXX20 ~__storage() noexcept {}
  };
  __storage __storage_{};
  bool __engaged_{};

  _CCCL_API constexpr __compressed_box_storage(bool __engaged) noexcept
      : __storage_()
      , __engaged_(__engaged)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Tp2 = _Tp)
  _CCCL_REQUIRES(is_default_constructible_v<_Tp2>)
  _CCCL_API constexpr __compressed_box_storage() noexcept(is_nothrow_default_constructible_v<_Tp2>)
      : __storage_(in_place)
      , __engaged_(true)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES(is_constructible_v<_Tp, _Args...>)
  _CCCL_API constexpr explicit __compressed_box_storage(in_place_t, _Args&&... __args) noexcept(
    is_nothrow_constructible_v<_Tp, _Args...>)
      : __storage_(in_place, ::cuda::std::forward<_Args>(__args)...)
      , __engaged_(true)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 ~__compressed_box_storage()
  {
    if (__engaged_)
    {
      __storage_.__val_.~_Tp();
    }
  }
};

template <size_t _Index, class _Tp>
struct _CCCL_DECLSPEC_EMPTY_BASES __compressed_box_storage<_Index, _Tp, true>
{
  union __storage
  {
    char __null_state_;
    remove_cv_t<_Tp> __val_;

    _CCCL_API constexpr __storage() noexcept
        : __null_state_()
    {}
    _CCCL_EXEC_CHECK_DISABLE
    _CCCL_TEMPLATE(class... _Args)
    _CCCL_REQUIRES(is_constructible_v<_Tp, _Args...>)
    _CCCL_API constexpr __storage(in_place_t, _Args&&... __args) noexcept(is_nothrow_constructible_v<_Tp, _Args...>)
        : __val_(::cuda::std::forward<_Args>(__args)...)
    {}
  };
  __storage __storage_{};
  bool __engaged_{};

  _CCCL_API constexpr __compressed_box_storage(bool __engaged) noexcept
      : __storage_()
      , __engaged_(__engaged)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Tp2 = _Tp)
  _CCCL_REQUIRES(is_default_constructible_v<_Tp2>)
  _CCCL_API constexpr __compressed_box_storage() noexcept(is_nothrow_default_constructible_v<_Tp2>)
      : __storage_(in_place)
      , __engaged_(true)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES(is_constructible_v<_Tp, _Args...>)
  _CCCL_API constexpr explicit __compressed_box_storage(in_place_t, _Args&&... __args) noexcept(
    is_nothrow_constructible_v<_Tp, _Args...>)
      : __storage_(in_place, ::cuda::std::forward<_Args>(__args)...)
      , __engaged_(true)
  {}
};

//! @brief Wrapper around T that might not be assignable to access within a @c __compressed_movable_box
template <size_t _Index, class _Tp>
struct _CCCL_DECLSPEC_EMPTY_BASES __compressed_box_base<_Index, _Tp, __compressed_box_specialization::__with_engaged>
    : __compressed_box_storage<_Index, _Tp>
{
  using __base = __compressed_box_storage<_Index, _Tp>;

  _CCCL_API constexpr __compressed_box_base(bool __engaged) noexcept
      : __base(__engaged)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Tp2 = _Tp)
  _CCCL_REQUIRES(is_default_constructible_v<_Tp2>)
  _CCCL_API constexpr __compressed_box_base() noexcept(is_nothrow_default_constructible_v<_Tp2>)
      : __base(in_place)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES(is_constructible_v<_Tp, _Args...>)
  _CCCL_API constexpr __compressed_box_base(in_place_t,
                                            _Args&&... __args) noexcept(is_nothrow_constructible_v<_Tp, _Args...>)
      : __base(in_place, ::cuda::std::forward<_Args>(__args)...)
  {}

  [[nodiscard]] _CCCL_API constexpr _Tp& __get() noexcept
  {
    return this->__storage_.__val_;
  }
  [[nodiscard]] _CCCL_API constexpr const _Tp& __get() const noexcept
  {
    return this->__storage_.__val_;
  }

  [[nodiscard]] _CCCL_API constexpr bool __engaged() const noexcept
  {
    return this->__engaged_;
  }

  _CCCL_API constexpr void __set_engaged(const bool __engaged) noexcept
  {
    this->__engaged_ = __engaged;
  }

  template <class... _Args>
  _CCCL_API constexpr void __construct(_Args&&... __args) noexcept(is_nothrow_constructible_v<_Tp, _Args...>)
  {
    ::cuda::std::__construct_at(::cuda::std::addressof(this->__storage_.__val_), ::cuda::std::forward<_Args>(__args)...);
  }
};

//! @brief We only need to do something for types that require the engaged state
template <class _Tp>
inline constexpr __smf_availability __compressed_box_copy_construct_available =
  (is_copy_constructible_v<_Tp> && __compressed_box_choose<_Tp>() != __compressed_box_specialization::__with_engaged)
    ? __smf_availability::__trivial
  : is_copy_constructible_v<_Tp>
    ? __smf_availability::__available
    : __smf_availability::__deleted;

//! @brief Nothing to do for copy constructible types
template <size_t _Index, class _Tp, __smf_availability = __compressed_box_copy_construct_available<_Tp>>
struct _CCCL_DECLSPEC_EMPTY_BASES __compressed_box_copy_base : __compressed_box_base<_Index, _Tp>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__compressed_box_copy_base, __compressed_box_base, _Index, _Tp);
};

//! @brief We must ensure we only copy when engaged
template <size_t _Index, class _Tp>
struct _CCCL_DECLSPEC_EMPTY_BASES
__compressed_box_copy_base<_Index, _Tp, __smf_availability::__available> : __compressed_box_base<_Index, _Tp>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__compressed_box_copy_base, __compressed_box_base, _Index, _Tp);

  _CCCL_API _CCCL_CONSTEXPR_CXX20
  __compressed_box_copy_base(const __compressed_box_copy_base& __other) noexcept(is_nothrow_copy_constructible_v<_Tp>)
      : __base(__other.__engaged())
  {
    if (__other.__engaged())
    {
      this->__construct(__other.__get());
    }
  }

  _CCCL_HIDE_FROM_ABI __compressed_box_copy_base(__compressed_box_copy_base&&)                 = default;
  _CCCL_HIDE_FROM_ABI __compressed_box_copy_base& operator=(const __compressed_box_copy_base&) = default;
  _CCCL_HIDE_FROM_ABI __compressed_box_copy_base& operator=(__compressed_box_copy_base&&)      = default;
};

//! @brief Copy construction is deleted
template <size_t _Index, class _Tp>
struct _CCCL_DECLSPEC_EMPTY_BASES
__compressed_box_copy_base<_Index, _Tp, __smf_availability::__deleted> : __compressed_box_base<_Index, _Tp>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__compressed_box_copy_base, __compressed_box_base, _Index, _Tp);
  _CCCL_HIDE_FROM_ABI __compressed_box_copy_base(const __compressed_box_copy_base&)            = delete;
  _CCCL_HIDE_FROM_ABI __compressed_box_copy_base(__compressed_box_copy_base&&)                 = default;
  _CCCL_HIDE_FROM_ABI __compressed_box_copy_base& operator=(const __compressed_box_copy_base&) = default;
  _CCCL_HIDE_FROM_ABI __compressed_box_copy_base& operator=(__compressed_box_copy_base&&)      = default;
};

//! @brief We only need to do something for types that require the engaged state
template <class _Tp>
inline constexpr __smf_availability __compressed_box_move_construct_available =
  (is_move_constructible_v<_Tp> && __compressed_box_choose<_Tp>() != __compressed_box_specialization::__with_engaged)
    ? __smf_availability::__trivial
  : is_move_constructible_v<_Tp>
    ? __smf_availability::__available
    : __smf_availability::__deleted;

//! @brief Nothing to do for move constructible types
template <size_t _Index, class _Tp, __smf_availability = __compressed_box_move_construct_available<_Tp>>
struct _CCCL_DECLSPEC_EMPTY_BASES __compressed_box_move_base : __compressed_box_copy_base<_Index, _Tp>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__compressed_box_move_base, __compressed_box_copy_base, _Index, _Tp);
};

//! @brief We must ensure we only move when engaged
template <size_t _Index, class _Tp>
struct _CCCL_DECLSPEC_EMPTY_BASES
__compressed_box_move_base<_Index, _Tp, __smf_availability::__available> : __compressed_box_copy_base<_Index, _Tp>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__compressed_box_move_base, __compressed_box_copy_base, _Index, _Tp);

  _CCCL_HIDE_FROM_ABI __compressed_box_move_base(const __compressed_box_move_base&) = default;
  _CCCL_API _CCCL_CONSTEXPR_CXX20
  __compressed_box_move_base(__compressed_box_move_base&& __other) noexcept(is_nothrow_move_constructible_v<_Tp>)
      : __base(__other.__engaged())
  {
    if (__other.__engaged())
    {
      this->__construct(::cuda::std::move(__other.__get()));
    }
  }

  _CCCL_HIDE_FROM_ABI __compressed_box_move_base& operator=(const __compressed_box_move_base&) = default;
  _CCCL_HIDE_FROM_ABI __compressed_box_move_base& operator=(__compressed_box_move_base&&)      = default;
};

//! @brief Move construction is deleted
template <size_t _Index, class _Tp>
struct _CCCL_DECLSPEC_EMPTY_BASES
__compressed_box_move_base<_Index, _Tp, __smf_availability::__deleted> : __compressed_box_copy_base<_Index, _Tp>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__compressed_box_move_base, __compressed_box_copy_base, _Index, _Tp);
  _CCCL_HIDE_FROM_ABI __compressed_box_move_base(const __compressed_box_move_base&)            = default;
  _CCCL_HIDE_FROM_ABI __compressed_box_move_base(__compressed_box_move_base&&)                 = delete;
  _CCCL_HIDE_FROM_ABI __compressed_box_move_base& operator=(const __compressed_box_move_base&) = default;
  _CCCL_HIDE_FROM_ABI __compressed_box_move_base& operator=(__compressed_box_move_base&&)      = default;
};

template <class _Tp>
inline constexpr __smf_availability __compressed_box_copy_assign_available =
  copyable<_Tp> && is_trivially_copy_assignable_v<_Tp> ? __smf_availability::__trivial
  : copyable<_Tp> || copy_constructible<_Tp>
    ? __smf_availability::__available
    : __smf_availability::__deleted;

//! @brief Nothing to do for trivially copy assignable types
template <size_t _Index, class _Tp, __smf_availability = __compressed_box_copy_assign_available<_Tp>>
struct _CCCL_DECLSPEC_EMPTY_BASES __compressed_box_copy_assign_base : __compressed_box_move_base<_Index, _Tp>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__compressed_box_copy_assign_base, __compressed_box_move_base, _Index, _Tp);
};

//! @brief If we can either assign or copy construct we do so
template <size_t _Index, class _Tp>
struct _CCCL_DECLSPEC_EMPTY_BASES __compressed_box_copy_assign_base<_Index, _Tp, __smf_availability::__available>
    : __compressed_box_move_base<_Index, _Tp>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__compressed_box_copy_assign_base, __compressed_box_move_base, _Index, _Tp);

  _CCCL_HIDE_FROM_ABI __compressed_box_copy_assign_base(const __compressed_box_copy_assign_base&) = default;
  _CCCL_HIDE_FROM_ABI __compressed_box_copy_assign_base(__compressed_box_copy_assign_base&&)      = default;

  static constexpr bool __nothrow_copy_assignable =
    copyable<_Tp> ? is_nothrow_copy_constructible_v<_Tp> && is_nothrow_copy_assignable_v<_Tp>
                  : is_nothrow_copy_constructible_v<_Tp>;

  _CCCL_API _CCCL_CONSTEXPR_CXX20 __compressed_box_copy_assign_base&
  operator=(const __compressed_box_copy_assign_base& __other) noexcept(__nothrow_copy_assignable)
  {
    if (::cuda::std::addressof(__other) == this)
    {
      return *this;
    }

    // We can assign if there is something, or destroy
    if constexpr (copyable<_Tp>)
    {
      if (this->__engaged())
      {
        if (__other.__engaged())
        {
          this->__get() = __other.__get();
        }
        else
        {
          this->__get().~_Tp();
          this->__set_engaged(false);
        }
      }
      else
      {
        if (__other.__engaged())
        {
          this->__construct(__other.__get());
          this->__set_engaged(true);
        }
        else
        {
          // nothing to do
        }
      }
    }
    else
    { // we must always destroy first
      if (this->__engaged())
      {
        this->__get().~_Tp();
        this->__set_engaged(false);
      }

      if (__other.__engaged())
      {
        this->__construct(__other.__get());
        this->__set_engaged(true);
      }
    }
    return *this;
  }
  _CCCL_HIDE_FROM_ABI __compressed_box_copy_assign_base& operator=(__compressed_box_copy_assign_base&&) = default;
};

//! @brief No copy assignment
template <size_t _Index, class _Tp>
struct _CCCL_DECLSPEC_EMPTY_BASES
__compressed_box_copy_assign_base<_Index, _Tp, __smf_availability::__deleted> : __compressed_box_move_base<_Index, _Tp>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__compressed_box_copy_assign_base, __compressed_box_move_base, _Index, _Tp);
  _CCCL_HIDE_FROM_ABI __compressed_box_copy_assign_base(const __compressed_box_copy_assign_base&)            = default;
  _CCCL_HIDE_FROM_ABI __compressed_box_copy_assign_base(__compressed_box_copy_assign_base&&)                 = default;
  _CCCL_HIDE_FROM_ABI __compressed_box_copy_assign_base& operator=(const __compressed_box_copy_assign_base&) = delete;
  _CCCL_HIDE_FROM_ABI __compressed_box_copy_assign_base& operator=(__compressed_box_copy_assign_base&&)      = default;
};

template <class _Tp>
inline constexpr __smf_availability __compressed_box_move_assign_available =
  movable<_Tp> && is_trivially_move_assignable_v<_Tp> ? __smf_availability::__trivial
  : movable<_Tp> || is_move_constructible_v<_Tp>
    ? __smf_availability::__available
    : __smf_availability::__deleted;

//! @brief Nothing to do for trivially copy assignable types
template <size_t _Index, class _Tp, __smf_availability = __compressed_box_move_assign_available<_Tp>>
struct _CCCL_DECLSPEC_EMPTY_BASES __compressed_box_move_assign_base : __compressed_box_copy_assign_base<_Index, _Tp>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__compressed_box_move_assign_base, __compressed_box_copy_assign_base, _Index, _Tp);
};

//! @brief If we can either assign or copy construct we do so
template <size_t _Index, class _Tp>
struct _CCCL_DECLSPEC_EMPTY_BASES __compressed_box_move_assign_base<_Index, _Tp, __smf_availability::__available>
    : __compressed_box_copy_assign_base<_Index, _Tp>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__compressed_box_move_assign_base, __compressed_box_copy_assign_base, _Index, _Tp);

  _CCCL_HIDE_FROM_ABI __compressed_box_move_assign_base(const __compressed_box_move_assign_base&)            = default;
  _CCCL_HIDE_FROM_ABI __compressed_box_move_assign_base(__compressed_box_move_assign_base&&)                 = default;
  _CCCL_HIDE_FROM_ABI __compressed_box_move_assign_base& operator=(const __compressed_box_move_assign_base&) = default;

  static constexpr bool __nothrow_move_assignable =
    movable<_Tp> ? is_nothrow_move_constructible_v<_Tp> && is_nothrow_move_assignable_v<_Tp>
                 : is_nothrow_move_constructible_v<_Tp>;

  _CCCL_API _CCCL_CONSTEXPR_CXX20 __compressed_box_move_assign_base&
  operator=(__compressed_box_move_assign_base&& __other) noexcept(__nothrow_move_assignable)
  {
    if (::cuda::std::addressof(__other) == this)
    {
      return *this;
    }

    // We can assign if there is something, or destroyx
    if constexpr (movable<_Tp>)
    {
      if (this->__engaged())
      {
        if (__other.__engaged())
        {
          this->__get() = ::cuda::std::move(__other.__get());
        }
        else
        {
          this->__get().~_Tp();
          this->__set_engaged(false);
        }
      }
      else
      {
        if (__other.__engaged())
        {
          this->__construct(::cuda::std::move(__other.__get()));
          this->__set_engaged(true);
        }
        else
        {
          // nothing to do
        }
      }
    }
    else
    { // we must always destroy first
      if (this->__engaged())
      {
        this->__get().~_Tp();
        this->__set_engaged(false);
      }

      if (__other.__engaged())
      {
        this->__construct(::cuda::std::move(__other.__get()));
        this->__set_engaged(true);
      }
    }
    return *this;
  }
};

//! @brief No copy assignment
template <size_t _Index, class _Tp>
struct _CCCL_DECLSPEC_EMPTY_BASES __compressed_box_move_assign_base<_Index, _Tp, __smf_availability::__deleted>
    : __compressed_box_copy_assign_base<_Index, _Tp>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__compressed_box_move_assign_base, __compressed_box_copy_assign_base, _Index, _Tp);
  _CCCL_HIDE_FROM_ABI __compressed_box_move_assign_base(const __compressed_box_move_assign_base&)            = default;
  _CCCL_HIDE_FROM_ABI __compressed_box_move_assign_base(__compressed_box_move_assign_base&&)                 = default;
  _CCCL_HIDE_FROM_ABI __compressed_box_move_assign_base& operator=(const __compressed_box_move_assign_base&) = default;
  _CCCL_HIDE_FROM_ABI __compressed_box_move_assign_base& operator=(__compressed_box_move_assign_base&&)      = delete;
};

template <size_t _Index, class _Tp>
struct _CCCL_DECLSPEC_EMPTY_BASES __compressed_box : __compressed_box_move_assign_base<_Index, _Tp>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__compressed_box, __compressed_box_move_assign_base, _Index, _Tp);
};

template <class...>
struct _CCCL_DECLSPEC_EMPTY_BASES __compressed_movable_box;

template <class _Elem1>
struct _CCCL_DECLSPEC_EMPTY_BASES __compressed_movable_box<_Elem1> : __compressed_box<0, _Elem1>
{
  using __base1 = __compressed_box<0, _Elem1>;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Elem1_ = _Elem1)
  _CCCL_REQUIRES(is_default_constructible_v<_Elem1_>)
  _CCCL_API constexpr __compressed_movable_box() noexcept(is_nothrow_default_constructible_v<_Elem1_>)
      : __base1()
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES((sizeof...(_Args) != 0) _CCCL_AND is_constructible_v<_Elem1, _Args...>)
  _CCCL_API constexpr __compressed_movable_box(_Args&&... __args) noexcept(is_nothrow_constructible_v<_Elem1, _Args...>)
      : __base1(in_place, ::cuda::std::forward<_Args>(__args)...)
  {}

  _CCCL_TEMPLATE(size_t _Index)
  _CCCL_REQUIRES((_Index < 1))
  [[nodiscard]] _CCCL_API constexpr _Elem1& __get() noexcept
  {
    return static_cast<__base1*>(this)->__get();
  }

  _CCCL_TEMPLATE(size_t _Index)
  _CCCL_REQUIRES((_Index < 1))
  [[nodiscard]] _CCCL_API constexpr const _Elem1& __get() const noexcept
  {
    return static_cast<const __base1*>(this)->__get();
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API friend constexpr void swap(__compressed_movable_box& __x, __compressed_movable_box& __y)
  {
    swap(__x.__get<0>(), __y.__get<0>());
  }
};

template <class _Elem1, class _Elem2>
struct _CCCL_DECLSPEC_EMPTY_BASES __compressed_movable_box<_Elem1, _Elem2>
    : __compressed_box<0, _Elem1>
    , __compressed_box<1, _Elem2>
{
  using __base1 = __compressed_box<0, _Elem1>;
  using __base2 = __compressed_box<1, _Elem2>;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Elem1_ = _Elem1, class _Elem2_ = _Elem2)
  _CCCL_REQUIRES(is_default_constructible_v<_Elem1_> _CCCL_AND is_default_constructible_v<_Elem2_>)
  _CCCL_API constexpr __compressed_movable_box() noexcept(
    is_nothrow_default_constructible_v<_Elem1_> && is_nothrow_default_constructible_v<_Elem2_>)
      : __base1()
      , __base2()
  {}

  template <class _Arg1>
  static constexpr bool __is_constructible_from_one_arg =
    is_constructible_v<_Elem1, _Arg1> && is_default_constructible_v<_Elem2>;

  template <class _Arg1>
  static constexpr bool __is_nothrow_constructible_from_one_arg =
    is_nothrow_constructible_v<_Elem1, _Arg1> && is_nothrow_default_constructible_v<_Elem2>;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Arg1)
  _CCCL_REQUIRES(__is_constructible_from_one_arg<_Arg1>)
  _CCCL_API constexpr __compressed_movable_box(_Arg1&& __arg1) noexcept(__is_nothrow_constructible_from_one_arg<_Arg1>)
      : __base1(in_place, ::cuda::std::forward<_Arg1>(__arg1))
      , __base2()
  {}

  template <class _Arg1, class _Arg2>
  static constexpr bool __is_constructible_from_two_args =
    is_constructible_v<_Elem1, _Arg1> && is_constructible_v<_Elem2, _Arg2>;

  template <class _Arg1, class _Arg2>
  static constexpr bool __is_nothrow_constructible_from_two_args =
    is_nothrow_constructible_v<_Elem1, _Arg1> && is_nothrow_constructible_v<_Elem2, _Arg2>;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Arg1, class _Arg2)
  _CCCL_REQUIRES(__is_constructible_from_two_args<_Arg1, _Arg2>)
  _CCCL_API constexpr __compressed_movable_box(_Arg1&& __arg1, _Arg2&& __arg2) noexcept(
    __is_nothrow_constructible_from_two_args<_Arg1, _Arg2>)
      : __base1(in_place, ::cuda::std::forward<_Arg1>(__arg1))
      , __base2(in_place, ::cuda::std::forward<_Arg2>(__arg2))
  {}

  _CCCL_TEMPLATE(size_t _Index)
  _CCCL_REQUIRES((_Index < 2))
  [[nodiscard]] _CCCL_API constexpr decltype(auto) __get() noexcept
  {
    if constexpr (_Index == 0)
    {
      return static_cast<__base1*>(this)->__get();
    }
    else // if constexpr (_Index == 1)
    {
      return static_cast<__base2*>(this)->__get();
    }
    _CCCL_UNREACHABLE();
  }

  _CCCL_TEMPLATE(size_t _Index)
  _CCCL_REQUIRES((_Index < 2))
  [[nodiscard]] _CCCL_API constexpr decltype(auto) __get() const noexcept
  {
    if constexpr (_Index == 0)
    {
      return static_cast<const __base1*>(this)->__get();
    }
    else // if constexpr (_Index == 1)
    {
      return static_cast<const __base2*>(this)->__get();
    }
    _CCCL_UNREACHABLE();
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API friend constexpr void swap(__compressed_movable_box& __x, __compressed_movable_box& __y)
  {
    swap(__x.__get<0>(), __y.__get<0>());
    swap(__x.__get<1>(), __y.__get<1>());
  }
};

template <class _Elem1, class _Elem2, class _Elem3>
struct _CCCL_DECLSPEC_EMPTY_BASES __compressed_movable_box<_Elem1, _Elem2, _Elem3>
    : __compressed_box<0, _Elem1>
    , __compressed_box<1, _Elem2>
    , __compressed_box<2, _Elem3>
{
  using __base1 = __compressed_box<0, _Elem1>;
  using __base2 = __compressed_box<1, _Elem2>;
  using __base3 = __compressed_box<2, _Elem3>;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Elem1_ = _Elem1, class _Elem2_ = _Elem2, class _Elem3_ = _Elem3)
  _CCCL_REQUIRES(is_default_constructible_v<_Elem1_> _CCCL_AND is_default_constructible_v<_Elem2_> _CCCL_AND
                   is_default_constructible_v<_Elem3_>)
  _CCCL_API constexpr __compressed_movable_box() noexcept(
    is_nothrow_default_constructible_v<_Elem1_> && is_nothrow_default_constructible_v<_Elem2_>
    && is_nothrow_default_constructible_v<_Elem3_>)
      : __base1()
      , __base2()
      , __base3()
  {}

  template <class _Arg1>
  static constexpr bool __is_constructible_from_one_arg =
    is_constructible_v<_Elem1, _Arg1> && is_default_constructible_v<_Elem2> && is_default_constructible_v<_Elem3>;

  template <class _Arg1>
  static constexpr bool __is_nothrow_constructible_from_one_arg =
    is_nothrow_constructible_v<_Elem1, _Arg1> && is_nothrow_default_constructible_v<_Elem2>
    && is_nothrow_default_constructible_v<_Elem3>;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Arg1)
  _CCCL_REQUIRES(__is_constructible_from_one_arg<_Arg1>)
  _CCCL_API constexpr __compressed_movable_box(_Arg1&& __arg1) noexcept(__is_nothrow_constructible_from_one_arg<_Arg1>)
      : __base1(in_place, ::cuda::std::forward<_Arg1>(__arg1))
      , __base2()
      , __base3()
  {}

  template <class _Arg1, class _Arg2>
  static constexpr bool __is_constructible_from_two_args =
    is_constructible_v<_Elem1, _Arg1> && is_constructible_v<_Elem2, _Arg2> && is_default_constructible_v<_Elem3>;

  template <class _Arg1, class _Arg2>
  static constexpr bool __is_nothrow_constructible_from_two_args =
    is_nothrow_constructible_v<_Elem1, _Arg1> && is_nothrow_constructible_v<_Elem2, _Arg2>
    && is_nothrow_default_constructible_v<_Elem3>;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Arg1, class _Arg2)
  _CCCL_REQUIRES(__is_constructible_from_two_args<_Arg1, _Arg2>)
  _CCCL_API constexpr __compressed_movable_box(_Arg1&& __arg1, _Arg2&& __arg2) noexcept(
    __is_nothrow_constructible_from_two_args<_Arg1, _Arg2>)
      : __base1(in_place, ::cuda::std::forward<_Arg1>(__arg1))
      , __base2(in_place, ::cuda::std::forward<_Arg2>(__arg2))
      , __base3()
  {}

  template <class _Arg1, class _Arg2, class _Arg3>
  static constexpr bool __is_constructible_from_three_args =
    is_constructible_v<_Elem1, _Arg1> && is_constructible_v<_Elem2, _Arg2> && is_constructible_v<_Elem3, _Arg3>;

  template <class _Arg1, class _Arg2, class _Arg3>
  static constexpr bool __is_nothrow_constructible_from_three_args =
    is_nothrow_constructible_v<_Elem1, _Arg1> && is_nothrow_constructible_v<_Elem2, _Arg2>
    && is_nothrow_constructible_v<_Elem3, _Arg3>;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Arg1, class _Arg2, class _Arg3)
  _CCCL_REQUIRES(__is_constructible_from_three_args<_Arg1, _Arg2, _Arg3>)
  _CCCL_API constexpr __compressed_movable_box(_Arg1&& __arg1, _Arg2&& __arg2, _Arg3&& __arg3) noexcept(
    __is_nothrow_constructible_from_three_args<_Arg1, _Arg2, _Arg3>)
      : __base1(in_place, ::cuda::std::forward<_Arg1>(__arg1))
      , __base2(in_place, ::cuda::std::forward<_Arg2>(__arg2))
      , __base3(in_place, ::cuda::std::forward<_Arg3>(__arg3))
  {}

  _CCCL_TEMPLATE(size_t _Index)
  _CCCL_REQUIRES((_Index < 3))
  [[nodiscard]] _CCCL_API constexpr decltype(auto) __get() noexcept
  {
    if constexpr (_Index == 0)
    {
      return static_cast<__base1*>(this)->__get();
    }
    else if constexpr (_Index == 1)
    {
      return static_cast<__base2*>(this)->__get();
    }
    else // if constexpr (_Index == 2)
    {
      return static_cast<__base3*>(this)->__get();
    }
    _CCCL_UNREACHABLE();
  }

  _CCCL_TEMPLATE(size_t _Index)
  _CCCL_REQUIRES((_Index < 3))
  [[nodiscard]] _CCCL_API constexpr decltype(auto) __get() const noexcept
  {
    if constexpr (_Index == 0)
    {
      return static_cast<const __base1*>(this)->__get();
    }
    else if constexpr (_Index == 1)
    {
      return static_cast<const __base2*>(this)->__get();
    }
    else // if constexpr (_Index == 2)
    {
      return static_cast<const __base3*>(this)->__get();
    }
    _CCCL_UNREACHABLE();
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API friend constexpr void swap(__compressed_movable_box& __x, __compressed_movable_box& __y)
  {
    swap(__x.__get<0>(), __y.__get<0>());
    swap(__x.__get<1>(), __y.__get<1>());
    swap(__x.__get<2>(), __y.__get<2>());
  }
};

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___RANGES_COMPRESSED_MOVABLE_BOX_H
