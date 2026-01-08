//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___OPTIONAL_OPTIONAL_H
#define _LIBCUDACXX___OPTIONAL_OPTIONAL_H

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
#include <cuda/std/__type_traits/disjunction.h>
#include <cuda/std/__type_traits/is_copy_constructible.h>
#include <cuda/std/__type_traits/is_move_constructible.h>
#include <cuda/std/__type_traits/is_object.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/reference_constructs_from_temporary.h>
#include <cuda/std/__type_traits/reference_converts_from_temporary.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/in_place.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/swap.h>
#include <cuda/std/initializer_list>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// Constraints
template <class _Tp, class _Up, class _Opt = optional<_Up>>
using __opt_check_constructible_from_opt =
  _Or<is_constructible<_Tp, _Opt&>,
      is_constructible<_Tp, _Opt const&>,
      is_constructible<_Tp, _Opt&&>,
      is_constructible<_Tp, _Opt const&&>,
      is_convertible<_Opt&, _Tp>,
      is_convertible<_Opt const&, _Tp>,
      is_convertible<_Opt&&, _Tp>,
      is_convertible<_Opt const&&, _Tp>>;

template <class _Tp, class _Up, class _Opt = optional<_Up>>
using __opt_check_assignable_from_opt =
  _Or<is_assignable<_Tp&, _Opt&>,
      is_assignable<_Tp&, _Opt const&>,
      is_assignable<_Tp&, _Opt&&>,
      is_assignable<_Tp&, _Opt const&&>>;

template <class _Tp, class _Up>
inline constexpr bool __opt_is_implictly_constructible =
  _CCCL_TRAIT(is_constructible, _Tp, _Up) && _CCCL_TRAIT(is_convertible, _Up, _Tp);

template <class _Tp, class _Up>
inline constexpr bool __opt_is_explictly_constructible =
  _CCCL_TRAIT(is_constructible, _Tp, _Up) && !_CCCL_TRAIT(is_convertible, _Up, _Tp);

template <class _Tp, class _Up>
inline constexpr bool __opt_is_constructible_from_U =
  !_CCCL_TRAIT(is_same, remove_cvref_t<_Up>, in_place_t) && !_CCCL_TRAIT(is_same, remove_cvref_t<_Up>, optional<_Tp>);

template <class _Tp, class _Up>
inline constexpr bool __opt_is_constructible_from_opt =
  !_CCCL_TRAIT(is_same, _Up, _Tp) && !__opt_check_constructible_from_opt<_Tp, _Up>::value;

template <class _Tp, class _Up>
inline constexpr bool __opt_is_assignable =
  _CCCL_TRAIT(is_constructible, _Tp, _Up) && _CCCL_TRAIT(is_assignable, _Tp&, _Up);

template <class _Tp, class _Up>
inline constexpr bool __opt_is_assignable_from_U =
  !_CCCL_TRAIT(is_same, remove_cvref_t<_Up>, optional<_Tp>)
  && (!_CCCL_TRAIT(is_same, remove_cvref_t<_Up>, _Tp) || !_CCCL_TRAIT(is_scalar, _Tp));

template <class _Tp, class _Up>
inline constexpr bool __opt_is_assignable_from_opt =
  !_CCCL_TRAIT(is_same, _Up, _Tp) && !__opt_check_constructible_from_opt<_Tp, _Up>::value
  && !__opt_check_assignable_from_opt<_Tp, _Up>::value;

template <class _Tp>
class optional : private __optional_move_assign_base<_Tp>
{
  using __base = __optional_move_assign_base<_Tp>;

  template <class>
  friend class optional;

public:
  using value_type = _Tp;

private:
  // Disable the reference extension using this static assert.
  static_assert(!_CCCL_TRAIT(is_same, remove_cvref_t<value_type>, in_place_t),
                "instantiation of optional with in_place_t is ill-formed");
  static_assert(!_CCCL_TRAIT(is_same, remove_cvref_t<value_type>, nullopt_t),
                "instantiation of optional with nullopt_t is ill-formed");
  static_assert(!_CCCL_TRAIT(is_reference, value_type),
                "instantiation of optional with a reference type is ill-formed. Define "
                "CCCL_ENABLE_OPTIONAL_REF to enable it as a non-standard extension");
  static_assert(_CCCL_TRAIT(is_destructible, value_type),
                "instantiation of optional with a non-destructible type is ill-formed");
  static_assert(!_CCCL_TRAIT(is_array, value_type), "instantiation of optional with an array type is ill-formed");

public:
  _CCCL_API constexpr optional() noexcept {}
  _CCCL_HIDE_FROM_ABI constexpr optional(const optional&) = default;
  _CCCL_HIDE_FROM_ABI constexpr optional(optional&&)      = default;
  _CCCL_API constexpr optional(nullopt_t) noexcept {}

  _CCCL_TEMPLATE(class _In_place_t, class... _Args)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_same, _In_place_t, in_place_t)
                   _CCCL_AND _CCCL_TRAIT(is_constructible, value_type, _Args...))
  _CCCL_API constexpr explicit optional(_In_place_t, _Args&&... __args)
      : __base(in_place, _CUDA_VSTD::forward<_Args>(__args)...)
  {}

  _CCCL_TEMPLATE(class _Up, class... _Args)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_constructible, value_type, initializer_list<_Up>&, _Args...))
  _CCCL_API constexpr explicit optional(in_place_t, initializer_list<_Up> __il, _Args&&... __args)
      : __base(in_place, __il, _CUDA_VSTD::forward<_Args>(__args)...)
  {}

  _CCCL_TEMPLATE(class _Up = value_type)
  _CCCL_REQUIRES(__opt_is_constructible_from_U<_Tp, _Up> _CCCL_AND __opt_is_implictly_constructible<_Tp, _Up>)
  _CCCL_API constexpr optional(_Up&& __v)
      : __base(in_place, _CUDA_VSTD::forward<_Up>(__v))
  {}

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES(__opt_is_constructible_from_U<_Tp, _Up> _CCCL_AND __opt_is_explictly_constructible<_Tp, _Up>)
  _CCCL_API constexpr explicit optional(_Up&& __v)
      : __base(in_place, _CUDA_VSTD::forward<_Up>(__v))
  {}

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES(__opt_is_constructible_from_opt<_Tp, _Up> _CCCL_AND __opt_is_implictly_constructible<_Tp, const _Up&>)
  _CCCL_API constexpr optional(const optional<_Up>& __v)
  {
    this->__construct_from(__v);
  }

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES(__opt_is_constructible_from_opt<_Tp, _Up> _CCCL_AND __opt_is_explictly_constructible<_Tp, const _Up&>)
  _CCCL_API constexpr explicit optional(const optional<_Up>& __v)
  {
    this->__construct_from(__v);
  }

#ifdef CCCL_ENABLE_OPTIONAL_REF
  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES((_CCCL_TRAIT(is_same, remove_cv_t<_Tp>, bool) || __opt_is_constructible_from_opt<_Tp, _Up>)
                   _CCCL_AND __opt_is_implictly_constructible<_Tp, const _Up&>)
  _CCCL_API constexpr optional(const optional<_Up&>& __v)
  {
    this->__construct_from(__v);
  }

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES((_CCCL_TRAIT(is_same, remove_cv_t<_Tp>, bool) || __opt_is_constructible_from_opt<_Tp, _Up>)
                   _CCCL_AND __opt_is_explictly_constructible<_Tp, const _Up&>)
  _CCCL_API constexpr explicit optional(const optional<_Up&>& __v)
  {
    this->__construct_from(__v);
  }

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES(__opt_is_constructible_from_opt<_Tp, _Up> _CCCL_AND
                   __opt_is_implictly_constructible<_Tp, _Up> _CCCL_AND(!_CCCL_TRAIT(is_reference, _Up)))
  _CCCL_API constexpr optional(optional<_Up>&& __v)
  {
    this->__construct_from(_CUDA_VSTD::move(__v));
  }

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES(__opt_is_constructible_from_opt<_Tp, _Up> _CCCL_AND
                   __opt_is_explictly_constructible<_Tp, _Up> _CCCL_AND(!_CCCL_TRAIT(is_reference, _Up)))
  _CCCL_API constexpr explicit optional(optional<_Up>&& __v)
  {
    this->__construct_from(_CUDA_VSTD::move(__v));
  }
#else // ^^^ CCCL_ENABLE_OPTIONAL_REF ^^^ / vvv !CCCL_ENABLE_OPTIONAL_REF vvv
  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES(__opt_is_constructible_from_opt<_Tp, _Up> _CCCL_AND __opt_is_implictly_constructible<_Tp, _Up>)
  _CCCL_API constexpr optional(optional<_Up>&& __v)
  {
    this->__construct_from(_CUDA_VSTD::move(__v));
  }

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES(__opt_is_constructible_from_opt<_Tp, _Up> _CCCL_AND __opt_is_explictly_constructible<_Tp, _Up>)
  _CCCL_API constexpr explicit optional(optional<_Up>&& __v)
  {
    this->__construct_from(_CUDA_VSTD::move(__v));
  }
#endif // !CCCL_ENABLE_OPTIONAL_REF

private:
  template <class _Fp, class... _Args>
  _CCCL_API constexpr explicit optional(__optional_construct_from_invoke_tag, _Fp&& __f, _Args&&... __args)
      : __base(
          __optional_construct_from_invoke_tag{}, _CUDA_VSTD::forward<_Fp>(__f), _CUDA_VSTD::forward<_Args>(__args)...)
  {}

public:
  _CCCL_API constexpr optional& operator=(nullopt_t) noexcept
  {
    reset();
    return *this;
  }

  constexpr optional& operator=(const optional&) = default;
  constexpr optional& operator=(optional&&)      = default;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Up = value_type)
  _CCCL_REQUIRES(__opt_is_assignable_from_U<_Tp, _Up> _CCCL_AND __opt_is_assignable<_Tp, _Up>)
  _CCCL_API constexpr optional& operator=(_Up&& __v)
  {
    if (this->has_value())
    {
      this->__get() = _CUDA_VSTD::forward<_Up>(__v);
    }
    else
    {
      this->__construct(_CUDA_VSTD::forward<_Up>(__v));
    }
    return *this;
  }

#ifdef CCCL_ENABLE_OPTIONAL_REF
  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES((!_CCCL_TRAIT(is_reference, _Up))
                   _CCCL_AND __opt_is_assignable_from_opt<_Tp, _Up> _CCCL_AND __opt_is_assignable<_Tp, const _Up&>)
  _CCCL_API constexpr optional& operator=(const optional<_Up>& __v)
  {
    this->__assign_from(__v);
    return *this;
  }

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_reference, _Up)
                   _CCCL_AND __opt_is_assignable_from_opt<_Tp, _Up&> _CCCL_AND __opt_is_assignable<_Tp, _Up&>)
  _CCCL_API constexpr optional& operator=(const optional<_Up>& __v)
  {
    this->__assign_from(__v);
    return *this;
  }
#else // ^^^ CCCL_ENABLE_OPTIONAL_REF ^^^ / vvv !CCCL_ENABLE_OPTIONAL_REF vvv
  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES(__opt_is_assignable_from_opt<_Tp, _Up> _CCCL_AND __opt_is_assignable<_Tp, const _Up&>)
  _CCCL_API constexpr optional& operator=(const optional<_Up>& __v)
  {
    this->__assign_from(__v);
    return *this;
  }
#endif // !CCCL_ENABLE_OPTIONAL_REF

  _CCCL_TEMPLATE(class _Up)
  _CCCL_REQUIRES(__opt_is_assignable_from_opt<_Tp, _Up> _CCCL_AND __opt_is_assignable<_Tp, _Up>)
  _CCCL_API constexpr optional& operator=(optional<_Up>&& __v)
  {
    this->__assign_from(_CUDA_VSTD::move(__v));
    return *this;
  }

  template <class... _Args, enable_if_t<_CCCL_TRAIT(is_constructible, value_type, _Args...), int> = 0>
  _CCCL_API constexpr _Tp& emplace(_Args&&... __args)
  {
    reset();
    this->__construct(_CUDA_VSTD::forward<_Args>(__args)...);
    return this->__get();
  }

  template <class _Up,
            class... _Args,
            enable_if_t<_CCCL_TRAIT(is_constructible, value_type, initializer_list<_Up>&, _Args...), int> = 0>
  _CCCL_API constexpr _Tp& emplace(initializer_list<_Up> __il, _Args&&... __args)
  {
    reset();
    this->__construct(__il, _CUDA_VSTD::forward<_Args>(__args)...);
    return this->__get();
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr void swap(optional& __opt) noexcept(
    _CCCL_TRAIT(is_nothrow_move_constructible, value_type) && _CCCL_TRAIT(is_nothrow_swappable, value_type))
  {
    if (this->has_value() == __opt.has_value())
    {
      using _CUDA_VSTD::swap;
      if (this->has_value())
      {
        swap(this->__get(), __opt.__get());
      }
    }
    else
    {
      if (this->has_value())
      {
        __opt.__construct(_CUDA_VSTD::move(this->__get()));
        reset();
      }
      else
      {
        this->__construct(_CUDA_VSTD::move(__opt.__get()));
        __opt.reset();
      }
    }
  }

  _CCCL_API constexpr add_pointer_t<value_type const> operator->() const
  {
    _CCCL_ASSERT(this->has_value(), "optional operator-> called on a disengaged value");
    return _CUDA_VSTD::addressof(this->__get());
  }

  _CCCL_API constexpr add_pointer_t<value_type> operator->()
  {
    _CCCL_ASSERT(this->has_value(), "optional operator-> called on a disengaged value");
    return _CUDA_VSTD::addressof(this->__get());
  }

  _CCCL_API constexpr const value_type& operator*() const& noexcept
  {
    _CCCL_ASSERT(this->has_value(), "optional operator* called on a disengaged value");
    return this->__get();
  }

  _CCCL_API constexpr value_type& operator*() & noexcept
  {
    _CCCL_ASSERT(this->has_value(), "optional operator* called on a disengaged value");
    return this->__get();
  }

  _CCCL_API constexpr value_type&& operator*() && noexcept
  {
    _CCCL_ASSERT(this->has_value(), "optional operator* called on a disengaged value");
    return _CUDA_VSTD::move(this->__get());
  }

  _CCCL_API constexpr const value_type&& operator*() const&& noexcept
  {
    _CCCL_ASSERT(this->has_value(), "optional operator* called on a disengaged value");
    return _CUDA_VSTD::move(this->__get());
  }

  _CCCL_API constexpr explicit operator bool() const noexcept
  {
    return has_value();
  }

  using __base::__get;
  using __base::has_value;

  _CCCL_API constexpr value_type const& value() const&
  {
    if (!this->has_value())
    {
      __throw_bad_optional_access();
    }
    return this->__get();
  }

  _CCCL_API constexpr value_type& value() &
  {
    if (!this->has_value())
    {
      __throw_bad_optional_access();
    }
    return this->__get();
  }

  _CCCL_API constexpr value_type&& value() &&
  {
    if (!this->has_value())
    {
      __throw_bad_optional_access();
    }
    return _CUDA_VSTD::move(this->__get());
  }

  _CCCL_API constexpr value_type const&& value() const&&
  {
    if (!this->has_value())
    {
      __throw_bad_optional_access();
    }
    return _CUDA_VSTD::move(this->__get());
  }

  template <class _Up>
  _CCCL_API constexpr value_type value_or(_Up&& __v) const&
  {
    static_assert(_CCCL_TRAIT(is_copy_constructible, value_type),
                  "optional<T>::value_or: T must be copy constructible");
    static_assert(_CCCL_TRAIT(is_convertible, _Up, value_type), "optional<T>::value_or: U must be convertible to T");
    return this->has_value() ? this->__get() : static_cast<value_type>(_CUDA_VSTD::forward<_Up>(__v));
  }

  template <class _Up>
  _CCCL_API constexpr value_type value_or(_Up&& __v) &&
  {
    static_assert(_CCCL_TRAIT(is_move_constructible, value_type),
                  "optional<T>::value_or: T must be move constructible");
    static_assert(_CCCL_TRAIT(is_convertible, _Up, value_type), "optional<T>::value_or: U must be convertible to T");
    return this->has_value() ? _CUDA_VSTD::move(this->__get()) : static_cast<value_type>(_CUDA_VSTD::forward<_Up>(__v));
  }

  template <class _Func>
  _CCCL_API constexpr auto and_then(_Func&& __f) &
  {
    using _Up = invoke_result_t<_Func, value_type&>;
    static_assert(__is_std_optional_v<remove_cvref_t<_Up>>,
                  "Result of f(value()) must be a specialization of std::optional");
    if (this->__engaged_)
    {
      return _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Func>(__f), this->__get());
    }
    return remove_cvref_t<_Up>();
  }

  template <class _Func>
  _CCCL_API constexpr auto and_then(_Func&& __f) const&
  {
    using _Up = invoke_result_t<_Func, const value_type&>;
    static_assert(__is_std_optional_v<remove_cvref_t<_Up>>,
                  "Result of f(value()) must be a specialization of std::optional");
    if (this->__engaged_)
    {
      return _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Func>(__f), this->__get());
    }
    return remove_cvref_t<_Up>();
  }

  template <class _Func>
  _CCCL_API constexpr auto and_then(_Func&& __f) &&
  {
    using _Up = invoke_result_t<_Func, value_type&&>;
    static_assert(__is_std_optional_v<remove_cvref_t<_Up>>,
                  "Result of f(std::move(value())) must be a specialization of std::optional");
    if (this->__engaged_)
    {
      return _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Func>(__f), _CUDA_VSTD::move(this->__get()));
    }
    return remove_cvref_t<_Up>();
  }

  template <class _Func>
  _CCCL_API constexpr auto and_then(_Func&& __f) const&&
  {
    using _Up = invoke_result_t<_Func, const value_type&&>;
    static_assert(__is_std_optional_v<remove_cvref_t<_Up>>,
                  "Result of f(std::move(value())) must be a specialization of std::optional");
    if (this->__engaged_)
    {
      return _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Func>(__f), _CUDA_VSTD::move(this->__get()));
    }
    return remove_cvref_t<_Up>();
  }

  template <class _Func>
  _CCCL_API constexpr auto transform(_Func&& __f) &
  {
    using _Up = remove_cv_t<invoke_result_t<_Func, value_type&>>;
    static_assert(!_CCCL_TRAIT(is_array, _Up), "Result of f(value()) should not be an Array");
    static_assert(!_CCCL_TRAIT(is_same, _Up, in_place_t), "Result of f(value()) should not be std::in_place_t");
    static_assert(!_CCCL_TRAIT(is_same, _Up, nullopt_t), "Result of f(value()) should not be std::nullopt_t");
    static_assert(_CCCL_TRAIT(is_object, _Up), "Result of f(value()) should be an object type");
    if (this->__engaged_)
    {
      return optional<_Up>(__optional_construct_from_invoke_tag{}, _CUDA_VSTD::forward<_Func>(__f), this->__get());
    }
    return optional<_Up>();
  }

  template <class _Func>
  _CCCL_API constexpr auto transform(_Func&& __f) const&
  {
    using _Up = remove_cv_t<invoke_result_t<_Func, const value_type&>>;
    static_assert(!_CCCL_TRAIT(is_array, _Up), "Result of f(value()) should not be an Array");
    static_assert(!_CCCL_TRAIT(is_same, _Up, in_place_t), "Result of f(value()) should not be std::in_place_t");
    static_assert(!_CCCL_TRAIT(is_same, _Up, nullopt_t), "Result of f(value()) should not be std::nullopt_t");
    static_assert(_CCCL_TRAIT(is_object, _Up), "Result of f(value()) should be an object type");
    if (this->__engaged_)
    {
      return optional<_Up>(__optional_construct_from_invoke_tag{}, _CUDA_VSTD::forward<_Func>(__f), this->__get());
    }
    return optional<_Up>();
  }

  template <class _Func>
  _CCCL_API constexpr auto transform(_Func&& __f) &&
  {
    using _Up = remove_cv_t<invoke_result_t<_Func, value_type&&>>;
    static_assert(!_CCCL_TRAIT(is_array, _Up), "Result of f(std::move(value())) should not be an Array");
    static_assert(!_CCCL_TRAIT(is_same, _Up, in_place_t),
                  "Result of f(std::move(value())) should not be std::in_place_t");
    static_assert(!_CCCL_TRAIT(is_same, _Up, nullopt_t),
                  "Result of f(std::move(value())) should not be std::nullopt_t");
    static_assert(_CCCL_TRAIT(is_object, _Up), "Result of f(std::move(value())) should be an object type");
    if (this->__engaged_)
    {
      return optional<_Up>(
        __optional_construct_from_invoke_tag{}, _CUDA_VSTD::forward<_Func>(__f), _CUDA_VSTD::move(this->__get()));
    }
    return optional<_Up>();
  }

  template <class _Func>
  _CCCL_API constexpr auto transform(_Func&& __f) const&&
  {
    using _Up = remove_cvref_t<invoke_result_t<_Func, const value_type&&>>;
    static_assert(!_CCCL_TRAIT(is_array, _Up), "Result of f(std::move(value())) should not be an Array");
    static_assert(!_CCCL_TRAIT(is_same, _Up, in_place_t),
                  "Result of f(std::move(value())) should not be std::in_place_t");
    static_assert(!_CCCL_TRAIT(is_same, _Up, nullopt_t),
                  "Result of f(std::move(value())) should not be std::nullopt_t");
    static_assert(_CCCL_TRAIT(is_object, _Up), "Result of f(std::move(value())) should be an object type");
    if (this->__engaged_)
    {
      return optional<_Up>(
        __optional_construct_from_invoke_tag{}, _CUDA_VSTD::forward<_Func>(__f), _CUDA_VSTD::move(this->__get()));
    }
    return optional<_Up>();
  }

  _CCCL_TEMPLATE(class _Func, class _Tp2 = _Tp)
  _CCCL_REQUIRES(invocable<_Func> _CCCL_AND _CCCL_TRAIT(is_copy_constructible, _Tp2))
  _CCCL_API constexpr optional or_else(_Func&& __f) const&
  {
    static_assert(_CCCL_TRAIT(is_same, remove_cvref_t<invoke_result_t<_Func>>, optional),
                  "Result of f() should be the same type as this optional");
    if (this->__engaged_)
    {
      return *this;
    }
    return _CUDA_VSTD::forward<_Func>(__f)();
  }

  _CCCL_TEMPLATE(class _Func, class _Tp2 = _Tp)
  _CCCL_REQUIRES(invocable<_Func> _CCCL_AND _CCCL_TRAIT(is_move_constructible, _Tp2))
  _CCCL_API constexpr optional or_else(_Func&& __f) &&
  {
    static_assert(_CCCL_TRAIT(is_same, remove_cvref_t<invoke_result_t<_Func>>, optional),
                  "Result of f() should be the same type as this optional");
    if (this->__engaged_)
    {
      return _CUDA_VSTD::move(*this);
    }
    return _CUDA_VSTD::forward<_Func>(__f)();
  }

  using __base::reset;
};

template <class _Tp>
_CCCL_HOST_DEVICE optional(_Tp) -> optional<_Tp>;

// Comparisons between optionals
_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr enable_if_t<
  _CCCL_TRAIT(is_convertible, decltype(declval<const _Tp&>() == declval<const _Up&>()), bool),
  bool>
operator==(const optional<_Tp>& __x, const optional<_Up>& __y)
{
  if (static_cast<bool>(__x) != static_cast<bool>(__y))
  {
    return false;
  }
  if (!static_cast<bool>(__x))
  {
    return true;
  }
  return *__x == *__y;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr enable_if_t<
  _CCCL_TRAIT(is_convertible, decltype(declval<const _Tp&>() != declval<const _Up&>()), bool),
  bool>
operator!=(const optional<_Tp>& __x, const optional<_Up>& __y)
{
  if (static_cast<bool>(__x) != static_cast<bool>(__y))
  {
    return true;
  }
  if (!static_cast<bool>(__x))
  {
    return false;
  }
  return *__x != *__y;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr enable_if_t<
  _CCCL_TRAIT(is_convertible, decltype(declval<const _Tp&>() < declval<const _Up&>()), bool),
  bool>
operator<(const optional<_Tp>& __x, const optional<_Up>& __y)
{
  if (!static_cast<bool>(__y))
  {
    return false;
  }
  if (!static_cast<bool>(__x))
  {
    return true;
  }
  return *__x < *__y;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr enable_if_t<
  _CCCL_TRAIT(is_convertible, decltype(declval<const _Tp&>() > declval<const _Up&>()), bool),
  bool>
operator>(const optional<_Tp>& __x, const optional<_Up>& __y)
{
  if (!static_cast<bool>(__x))
  {
    return false;
  }
  if (!static_cast<bool>(__y))
  {
    return true;
  }
  return *__x > *__y;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr enable_if_t<
  _CCCL_TRAIT(is_convertible, decltype(declval<const _Tp&>() <= declval<const _Up&>()), bool),
  bool>
operator<=(const optional<_Tp>& __x, const optional<_Up>& __y)
{
  if (!static_cast<bool>(__x))
  {
    return true;
  }
  if (!static_cast<bool>(__y))
  {
    return false;
  }
  return *__x <= *__y;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr enable_if_t<
  _CCCL_TRAIT(is_convertible, decltype(declval<const _Tp&>() >= declval<const _Up&>()), bool),
  bool>
operator>=(const optional<_Tp>& __x, const optional<_Up>& __y)
{
  if (!static_cast<bool>(__y))
  {
    return true;
  }
  if (!static_cast<bool>(__x))
  {
    return false;
  }
  return *__x >= *__y;
}

// Comparisons with nullopt
template <class _Tp>
_CCCL_API constexpr bool operator==(const optional<_Tp>& __x, nullopt_t) noexcept
{
  return !static_cast<bool>(__x);
}

template <class _Tp>
_CCCL_API constexpr bool operator==(nullopt_t, const optional<_Tp>& __x) noexcept
{
  return !static_cast<bool>(__x);
}

template <class _Tp>
_CCCL_API constexpr bool operator!=(const optional<_Tp>& __x, nullopt_t) noexcept
{
  return static_cast<bool>(__x);
}

template <class _Tp>
_CCCL_API constexpr bool operator!=(nullopt_t, const optional<_Tp>& __x) noexcept
{
  return static_cast<bool>(__x);
}

template <class _Tp>
_CCCL_API constexpr bool operator<(const optional<_Tp>&, nullopt_t) noexcept
{
  return false;
}

template <class _Tp>
_CCCL_API constexpr bool operator<(nullopt_t, const optional<_Tp>& __x) noexcept
{
  return static_cast<bool>(__x);
}

template <class _Tp>
_CCCL_API constexpr bool operator<=(const optional<_Tp>& __x, nullopt_t) noexcept
{
  return !static_cast<bool>(__x);
}

template <class _Tp>
_CCCL_API constexpr bool operator<=(nullopt_t, const optional<_Tp>&) noexcept
{
  return true;
}

template <class _Tp>
_CCCL_API constexpr bool operator>(const optional<_Tp>& __x, nullopt_t) noexcept
{
  return static_cast<bool>(__x);
}

template <class _Tp>
_CCCL_API constexpr bool operator>(nullopt_t, const optional<_Tp>&) noexcept
{
  return false;
}

template <class _Tp>
_CCCL_API constexpr bool operator>=(const optional<_Tp>&, nullopt_t) noexcept
{
  return true;
}

template <class _Tp>
_CCCL_API constexpr bool operator>=(nullopt_t, const optional<_Tp>& __x) noexcept
{
  return !static_cast<bool>(__x);
}

// Comparisons with T
_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr enable_if_t<
  _CCCL_TRAIT(is_convertible, decltype(declval<const _Tp&>() == declval<const _Up&>()), bool),
  bool>
operator==(const optional<_Tp>& __x, const _Up& __v)
{
  return static_cast<bool>(__x) ? *__x == __v : false;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr enable_if_t<
  _CCCL_TRAIT(is_convertible, decltype(declval<const _Tp&>() == declval<const _Up&>()), bool),
  bool>
operator==(const _Tp& __v, const optional<_Up>& __x)
{
  return static_cast<bool>(__x) ? __v == *__x : false;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr enable_if_t<
  _CCCL_TRAIT(is_convertible, decltype(declval<const _Tp&>() != declval<const _Up&>()), bool),
  bool>
operator!=(const optional<_Tp>& __x, const _Up& __v)
{
  return static_cast<bool>(__x) ? *__x != __v : true;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr enable_if_t<
  _CCCL_TRAIT(is_convertible, decltype(declval<const _Tp&>() != declval<const _Up&>()), bool),
  bool>
operator!=(const _Tp& __v, const optional<_Up>& __x)
{
  return static_cast<bool>(__x) ? __v != *__x : true;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr enable_if_t<
  _CCCL_TRAIT(is_convertible, decltype(declval<const _Tp&>() < declval<const _Up&>()), bool),
  bool>
operator<(const optional<_Tp>& __x, const _Up& __v)
{
  return static_cast<bool>(__x) ? *__x < __v : true;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr enable_if_t<
  _CCCL_TRAIT(is_convertible, decltype(declval<const _Tp&>() < declval<const _Up&>()), bool),
  bool>
operator<(const _Tp& __v, const optional<_Up>& __x)
{
  return static_cast<bool>(__x) ? __v < *__x : false;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr enable_if_t<
  _CCCL_TRAIT(is_convertible, decltype(declval<const _Tp&>() <= declval<const _Up&>()), bool),
  bool>
operator<=(const optional<_Tp>& __x, const _Up& __v)
{
  return static_cast<bool>(__x) ? *__x <= __v : true;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr enable_if_t<
  _CCCL_TRAIT(is_convertible, decltype(declval<const _Tp&>() <= declval<const _Up&>()), bool),
  bool>
operator<=(const _Tp& __v, const optional<_Up>& __x)
{
  return static_cast<bool>(__x) ? __v <= *__x : false;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr enable_if_t<
  _CCCL_TRAIT(is_convertible, decltype(declval<const _Tp&>() > declval<const _Up&>()), bool),
  bool>
operator>(const optional<_Tp>& __x, const _Up& __v)
{
  return static_cast<bool>(__x) ? *__x > __v : false;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr enable_if_t<
  _CCCL_TRAIT(is_convertible, decltype(declval<const _Tp&>() > declval<const _Up&>()), bool),
  bool>
operator>(const _Tp& __v, const optional<_Up>& __x)
{
  return static_cast<bool>(__x) ? __v > *__x : true;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr enable_if_t<
  _CCCL_TRAIT(is_convertible, decltype(declval<const _Tp&>() >= declval<const _Up&>()), bool),
  bool>
operator>=(const optional<_Tp>& __x, const _Up& __v)
{
  return static_cast<bool>(__x) ? *__x >= __v : false;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class _Up>
_CCCL_API constexpr enable_if_t<
  _CCCL_TRAIT(is_convertible, decltype(declval<const _Tp&>() >= declval<const _Up&>()), bool),
  bool>
operator>=(const _Tp& __v, const optional<_Up>& __x)
{
  return static_cast<bool>(__x) ? __v >= *__x : true;
}

template <class _Tp>
_CCCL_API constexpr enable_if_t<
#ifdef CCCL_ENABLE_OPTIONAL_REF
  _CCCL_TRAIT(is_reference, _Tp) ||
#endif // CCCL_ENABLE_OPTIONAL_REF
    (_CCCL_TRAIT(is_move_constructible, _Tp) && _CCCL_TRAIT(is_swappable, _Tp)),
  void>
swap(optional<_Tp>& __x, optional<_Tp>& __y) noexcept(noexcept(__x.swap(__y)))
{
  __x.swap(__y);
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___OPTIONAL_OPTIONAL_H
