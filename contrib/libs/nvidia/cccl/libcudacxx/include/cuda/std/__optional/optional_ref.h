//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___OPTIONAL_OPTIONAL_REF_H
#define _LIBCUDACXX___OPTIONAL_OPTIONAL_REF_H

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
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__fwd/optional.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__optional/bad_optional_access.h>
#include <cuda/std/__optional/nullopt.h>
#include <cuda/std/__optional/optional_base.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/is_array.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_copy_constructible.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/reference_constructs_from_temporary.h>
#include <cuda/std/__type_traits/reference_converts_from_temporary.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/in_place.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/swap.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#ifdef CCCL_ENABLE_OPTIONAL_REF
template <class _Tp>
class optional<_Tp&>
{
private:
  using __raw_type     = remove_reference_t<_Tp>;
  __raw_type* __value_ = nullptr;

  _CCCL_TEMPLATE(class _Ref, class _Arg)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_constructible, _Ref, _Arg))
  [[nodiscard]] _CCCL_API static constexpr _Ref __make_reference(_Arg&& __arg) noexcept
  {
    static_assert(_CCCL_TRAIT(is_reference, _Ref), "optional<T&>: make-reference requires a reference as argument");
    return _Ref(_CUDA_VSTD::forward<_Arg>(__arg));
  }

  // Needed to interface with optional<T>
  template <class>
  friend struct __optional_storage_base;

  [[nodiscard]] _CCCL_API constexpr _Tp& __get() noexcept
  {
    return *__value_;
  }
  [[nodiscard]] _CCCL_API constexpr const _Tp& __get() const noexcept
  {
    return *__value_;
  }

#  if defined(_CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY)
  template <class _Up>
  static constexpr bool __from_temporary = reference_constructs_from_temporary_v<_Tp&, _Up>;
#  else
  template <class _Up>
  static constexpr bool __from_temporary = false;
#  endif // !_CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY

public:
  using value_type = __raw_type&;

  _CCCL_API constexpr optional() noexcept {}
  _CCCL_HIDE_FROM_ABI constexpr optional(const optional&) noexcept = default;
  _CCCL_HIDE_FROM_ABI constexpr optional(optional&&) noexcept      = default;
  _CCCL_API constexpr optional(nullopt_t) noexcept {}

  _CCCL_TEMPLATE(class _Arg)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_constructible, _Tp&, _Arg) _CCCL_AND(!__from_temporary<_Arg>))
  _CCCL_API explicit constexpr optional(in_place_t, _Arg&& __arg) noexcept
      : __value_(_CUDA_VSTD::addressof(__make_reference<_Tp&>(_CUDA_VSTD::forward<_Arg>(__arg))))
  {}

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES((!__is_std_optional_v<decay_t<_Up>>) _CCCL_AND _CCCL_TRAIT(is_convertible, _Up, _Tp&)
                   _CCCL_AND(!__from_temporary<_Up>))
  _CCCL_API constexpr optional(_Up&& __u) noexcept(noexcept(static_cast<_Tp&>(_CUDA_VSTD::declval<_Up>())))
      : __value_(_CUDA_VSTD::addressof(static_cast<_Tp&>(_CUDA_VSTD::forward<_Up>(__u))))
  {}

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES((!__is_std_optional_v<decay_t<_Up>>) _CCCL_AND(!_CCCL_TRAIT(is_convertible, _Up, _Tp&))
                   _CCCL_AND _CCCL_TRAIT(is_constructible, _Tp&, _Up) _CCCL_AND(!__from_temporary<_Up>))
  _CCCL_API explicit constexpr optional(_Up&& __u) noexcept(noexcept(static_cast<_Tp&>(_CUDA_VSTD::declval<_Up>())))
      : __value_(_CUDA_VSTD::addressof(static_cast<_Tp&>(_CUDA_VSTD::forward<_Up>(__u))))
  {}

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES((!__is_std_optional_v<decay_t<_Up>>) _CCCL_AND __from_temporary<_Up>)
  _CCCL_API constexpr optional(_Up&&) = delete;

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_convertible, _Up&, _Tp&) _CCCL_AND(!__from_temporary<_Up&>))
  _CCCL_API constexpr optional(optional<_Up>& __u) noexcept(noexcept(static_cast<_Tp&>(_CUDA_VSTD::declval<_Up&>())))
      : __value_(__u.has_value() ? _CUDA_VSTD::addressof(static_cast<_Tp&>(__u.value())) : nullptr)
  {}

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES((!_CCCL_TRAIT(is_convertible, _Up&, _Tp&)) _CCCL_AND _CCCL_TRAIT(is_constructible, _Tp&, _Up&)
                   _CCCL_AND(!__from_temporary<_Up&>))
  _CCCL_API explicit constexpr optional(optional<_Up>& __u) noexcept(
    noexcept(static_cast<_Tp&>(_CUDA_VSTD::declval<_Up&>())))
      : __value_(__u.has_value() ? _CUDA_VSTD::addressof(static_cast<_Tp&>(__u.value())) : nullptr)
  {}

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES(__from_temporary<_Up&>)
  _CCCL_API constexpr optional(optional<_Up>&) = delete;

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_convertible, const _Up&, _Tp&) _CCCL_AND(!__from_temporary<const _Up&>))
  _CCCL_API constexpr optional(const optional<_Up>& __u) noexcept(
    noexcept(static_cast<_Tp&>(_CUDA_VSTD::declval<const _Up&>())))
      : __value_(__u.has_value() ? _CUDA_VSTD::addressof(static_cast<_Tp&>(__u.value())) : nullptr)
  {}

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES((!_CCCL_TRAIT(is_convertible, const _Up&, _Tp&)) _CCCL_AND _CCCL_TRAIT(
    is_constructible, _Tp&, const _Up&) _CCCL_AND(!__from_temporary<const _Up&>))
  _CCCL_API explicit constexpr optional(const optional<_Up>& __u) noexcept(
    noexcept(static_cast<_Tp&>(_CUDA_VSTD::declval<const _Up&>())))
      : __value_(__u.has_value() ? _CUDA_VSTD::addressof(static_cast<_Tp&>(__u.value())) : nullptr)
  {}

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES(__from_temporary<const _Up&>)
  _CCCL_API constexpr optional(const optional<_Up>&) = delete;

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_convertible, _Up, _Tp&) _CCCL_AND(!__from_temporary<_Up>))
  _CCCL_API constexpr optional(optional<_Up>&& __u) noexcept(noexcept(static_cast<_Tp&>(_CUDA_VSTD::declval<_Up>())))
      : __value_(
          __u.has_value() ? _CUDA_VSTD::addressof(static_cast<_Tp&>(_CUDA_VSTD::forward<_Up>(__u.value()))) : nullptr)
  {}

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES((!_CCCL_TRAIT(is_convertible, _Up, _Tp&)) _CCCL_AND _CCCL_TRAIT(is_constructible, _Tp&, _Up)
                   _CCCL_AND(!__from_temporary<_Up>))
  _CCCL_API explicit constexpr optional(optional<_Up>&& __u) noexcept(
    noexcept(static_cast<_Tp&>(_CUDA_VSTD::declval<_Up>())))
      : __value_(
          __u.has_value() ? _CUDA_VSTD::addressof(static_cast<_Tp&>(_CUDA_VSTD::forward<_Up>(__u.value()))) : nullptr)
  {}

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES(__from_temporary<_Up>)
  _CCCL_API constexpr optional(optional<_Up>&&) = delete;

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_convertible, const _Up, _Tp&) _CCCL_AND(!__from_temporary<const _Up>))
  _CCCL_API constexpr optional(const optional<_Up>&& __u) noexcept(
    noexcept(static_cast<_Tp&>(_CUDA_VSTD::declval<const _Up>())))
      : __value_(__u.has_value() ? _CUDA_VSTD::addressof(static_cast<_Tp&>(__u.value())) : nullptr)
  {}

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES((!_CCCL_TRAIT(is_convertible, const _Up, _Tp&)) _CCCL_AND _CCCL_TRAIT(
    is_constructible, _Tp&, const _Up) _CCCL_AND(!__from_temporary<const _Up>))
  _CCCL_API explicit constexpr optional(const optional<_Up>&& __u) noexcept(
    noexcept(static_cast<_Tp&>(_CUDA_VSTD::declval<const _Up>())))
      : __value_(__u.has_value() ? _CUDA_VSTD::addressof(static_cast<_Tp&>(__u.value())) : nullptr)
  {}

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES(__from_temporary<const _Up>)
  _CCCL_API constexpr optional(const optional<_Up>&&) = delete;

  _CCCL_HIDE_FROM_ABI constexpr optional& operator=(const optional&) noexcept = default;
  _CCCL_HIDE_FROM_ABI constexpr optional& operator=(optional&&) noexcept      = default;

  _CCCL_API constexpr optional& operator=(nullopt_t) noexcept
  {
    __value_ = nullptr;
    return *this;
  }

  _CCCL_TEMPLATE(class _Up = _Tp)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_constructible, _Tp&, _Up) _CCCL_AND(!__from_temporary<_Up>))
  _CCCL_API constexpr _Tp& emplace(_Up&& __u) noexcept(noexcept(static_cast<_Tp&>(_CUDA_VSTD::forward<_Up>(__u))))
  {
    __value_ = _CUDA_VSTD::addressof(static_cast<_Tp&>(_CUDA_VSTD::forward<_Up>(__u)));
    return *__value_;
  }

  _CCCL_API constexpr void swap(optional& __rhs) noexcept
  {
    return _CUDA_VSTD::swap(__value_, __rhs.__value_);
  }

  _CCCL_API constexpr _Tp* operator->() const noexcept
  {
    _CCCL_ASSERT(__value_ != nullptr, "optional operator-> called on a disengaged value");
    return __value_;
  }

  _CCCL_API constexpr _Tp& operator*() const noexcept
  {
    _CCCL_ASSERT(__value_ != nullptr, "optional operator* called on a disengaged value");
    return *__value_;
  }

  _CCCL_API constexpr explicit operator bool() const noexcept
  {
    return __value_ != nullptr;
  }

  _CCCL_API constexpr bool has_value() const noexcept
  {
    return __value_ != nullptr;
  }

  _CCCL_API constexpr _Tp& value() const noexcept
  {
    if (__value_ != nullptr)
    {
      return *__value_;
    }
    else
    {
      __throw_bad_optional_access();
    }
  }

  template <class _Up>
  _CCCL_API constexpr remove_cvref_t<_Tp> value_or(_Up&& __v) const
  {
    static_assert(_CCCL_TRAIT(is_copy_constructible, _Tp), "optional<T&>::value_or: T must be copy constructible");
    static_assert(_CCCL_TRAIT(is_convertible, _Up, _Tp), "optional<T&>::value_or: U must be convertible to T");
    return __value_ != nullptr ? *__value_ : static_cast<_Tp>(_CUDA_VSTD::forward<_Up>(__v));
  }

  template <class _Func>
  _CCCL_API constexpr auto and_then(_Func&& __f) const
  {
    using _Up = invoke_result_t<_Func, _Tp&>;
    static_assert(__is_std_optional_v<remove_cvref_t<_Up>>,
                  "optional<T&>::and_then: Result of f(value()) must be a specialization of std::optional");
    if (__value_ != nullptr)
    {
      return _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Func>(__f), *__value_);
    }
    return remove_cvref_t<_Up>();
  }

  template <class _Func>
  _CCCL_API constexpr auto transform(_Func&& __f) const
  {
    using _Up = invoke_result_t<_Func, _Tp&>;
    static_assert(!_CCCL_TRAIT(is_array, _Up), "optional<T&>::transform: Result of f(value()) should not be an Array");
    static_assert(!_CCCL_TRAIT(is_same, _Up, in_place_t),
                  "optional<T&>::transform: Result of f(value()) should not be std::in_place_t");
    static_assert(!_CCCL_TRAIT(is_same, _Up, nullopt_t),
                  "optional<T&>::transform: Result of f(value()) should not be std::nullopt_t");
    if (__value_ != nullptr)
    {
      if constexpr (_CCCL_TRAIT(is_lvalue_reference, _Up))
      {
        return optional<_Up>(_CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Func>(__f), *__value_));
      }
      else
      {
        return optional<_Up>(__optional_construct_from_invoke_tag{}, _CUDA_VSTD::forward<_Func>(__f), *__value_);
      }
    }
    return optional<_Up>();
  }

  _CCCL_TEMPLATE(class _Func)
  _CCCL_REQUIRES(invocable<_Func>)
  _CCCL_API constexpr optional or_else(_Func&& __f) const
  {
    using _Up = invoke_result_t<_Func>;
    static_assert(_CCCL_TRAIT(is_same, remove_cvref_t<_Up>, optional),
                  "optional<T&>::or_else: Result of f() should be the same type as this optional");
    if (__value_ != nullptr)
    {
      return *this;
    }
    return _CUDA_VSTD::forward<_Func>(__f)();
  }

  _CCCL_API constexpr void reset() noexcept
  {
    __value_ = nullptr;
  }
};

#endif // CCCL_ENABLE_OPTIONAL_REF

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___OPTIONAL_OPTIONAL_REF_H
