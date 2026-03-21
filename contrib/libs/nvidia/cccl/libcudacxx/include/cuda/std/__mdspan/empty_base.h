//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___MDSPAN_EMPTY_BASE_H
#define _LIBCUDACXX___MDSPAN_EMPTY_BASE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_default_constructible.h>
#include <cuda/std/__type_traits/is_empty.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__utility/forward.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <size_t _Index, class _Elem, bool = _CCCL_TRAIT(is_empty, _Elem)>
struct _CCCL_DECLSPEC_EMPTY_BASES __mdspan_ebco_impl
{
  _Elem __elem_;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Elem_ = _Elem)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_default_constructible, _Elem_))
  _CCCL_API constexpr __mdspan_ebco_impl() noexcept(_CCCL_TRAIT(is_nothrow_default_constructible, _Elem_))
      : __elem_()
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES((sizeof...(_Args) != 0) _CCCL_AND _CCCL_TRAIT(is_constructible, _Elem, _Args...))
  _CCCL_API constexpr __mdspan_ebco_impl(_Args&&... __args) noexcept(
    _CCCL_TRAIT(is_nothrow_constructible, _Elem, _Args...))
      : __elem_(_CUDA_VSTD::forward<_Args>(__args)...)
  {}

  [[nodiscard]] _CCCL_API constexpr _Elem& __get() noexcept
  {
    return __elem_;
  }
  [[nodiscard]] _CCCL_API constexpr const _Elem& __get() const noexcept
  {
    return __elem_;
  }
};

template <size_t _Index, class _Elem>
struct _CCCL_DECLSPEC_EMPTY_BASES __mdspan_ebco_impl<_Index, _Elem, true> : _Elem
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Elem_ = _Elem)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_default_constructible, _Elem_))
  _CCCL_API constexpr __mdspan_ebco_impl() noexcept(_CCCL_TRAIT(is_nothrow_default_constructible, _Elem_))
      : _Elem()
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES((sizeof...(_Args) != 0) _CCCL_AND _CCCL_TRAIT(is_constructible, _Elem, _Args...))
  _CCCL_API constexpr __mdspan_ebco_impl(_Args&&... __args) noexcept(
    _CCCL_TRAIT(is_nothrow_constructible, _Elem, _Args...))
      : _Elem(_CUDA_VSTD::forward<_Args>(__args)...)
  {}

  [[nodiscard]] _CCCL_API constexpr _Elem& __get() noexcept
  {
    return *static_cast<_Elem*>(this);
  }
  [[nodiscard]] _CCCL_API constexpr const _Elem& __get() const noexcept
  {
    return *static_cast<const _Elem*>(this);
  }
};

template <class...>
struct _CCCL_DECLSPEC_EMPTY_BASES __mdspan_ebco;

template <class _Elem1>
struct _CCCL_DECLSPEC_EMPTY_BASES __mdspan_ebco<_Elem1> : __mdspan_ebco_impl<0, _Elem1>
{
  using __base1 = __mdspan_ebco_impl<0, _Elem1>;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Elem1_ = _Elem1)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_default_constructible, _Elem1_))
  _CCCL_API constexpr __mdspan_ebco() noexcept(_CCCL_TRAIT(is_nothrow_default_constructible, _Elem1_))
      : __base1()
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES((sizeof...(_Args) != 0) _CCCL_AND _CCCL_TRAIT(is_constructible, _Elem1, _Args...))
  _CCCL_API constexpr __mdspan_ebco(_Args&&... __args) noexcept(_CCCL_TRAIT(is_nothrow_constructible, _Elem1, _Args...))
      : __base1(_CUDA_VSTD::forward<_Args>(__args)...)
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
  _CCCL_API friend constexpr void swap(__mdspan_ebco& __x, __mdspan_ebco& __y)
  {
    swap(__x.__get<0>(), __y.__get<0>());
  }
};

template <class _Elem1, class _Elem2>
struct _CCCL_DECLSPEC_EMPTY_BASES __mdspan_ebco<_Elem1, _Elem2>
    : __mdspan_ebco_impl<0, _Elem1>
    , __mdspan_ebco_impl<1, _Elem2>
{
  using __base1 = __mdspan_ebco_impl<0, _Elem1>;
  using __base2 = __mdspan_ebco_impl<1, _Elem2>;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Elem1_ = _Elem1, class _Elem2_ = _Elem2)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_default_constructible, _Elem1_)
                   _CCCL_AND _CCCL_TRAIT(is_default_constructible, _Elem2_))
  _CCCL_API constexpr __mdspan_ebco() noexcept(_CCCL_TRAIT(is_nothrow_default_constructible, _Elem1_)
                                               && _CCCL_TRAIT(is_nothrow_default_constructible, _Elem2_))
      : __base1()
      , __base2()
  {}

  template <class _Arg1>
  static constexpr bool __is_constructible_from_one_arg =
    _CCCL_TRAIT(is_constructible, _Elem1, _Arg1) && _CCCL_TRAIT(is_default_constructible, _Elem2);

  template <class _Arg1>
  static constexpr bool __is_nothrow_constructible_from_one_arg =
    _CCCL_TRAIT(is_nothrow_constructible, _Elem1, _Arg1) && _CCCL_TRAIT(is_nothrow_default_constructible, _Elem2);

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Arg1)
  _CCCL_REQUIRES(__is_constructible_from_one_arg<_Arg1>)
  _CCCL_API constexpr __mdspan_ebco(_Arg1&& __arg1) noexcept(__is_nothrow_constructible_from_one_arg<_Arg1>)
      : __base1(_CUDA_VSTD::forward<_Arg1>(__arg1))
      , __base2()
  {}

  template <class _Arg1, class _Arg2>
  static constexpr bool __is_constructible_from_two_args =
    _CCCL_TRAIT(is_constructible, _Elem1, _Arg1) && _CCCL_TRAIT(is_constructible, _Elem2, _Arg2);

  template <class _Arg1, class _Arg2>
  static constexpr bool __is_nothrow_constructible_from_two_args =
    _CCCL_TRAIT(is_nothrow_constructible, _Elem1, _Arg1) && _CCCL_TRAIT(is_nothrow_constructible, _Elem2, _Arg2);

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Arg1, class _Arg2)
  _CCCL_REQUIRES(__is_constructible_from_two_args<_Arg1, _Arg2>)
  _CCCL_API constexpr __mdspan_ebco(_Arg1&& __arg1,
                                    _Arg2&& __arg2) noexcept(__is_nothrow_constructible_from_two_args<_Arg1, _Arg2>)
      : __base1(_CUDA_VSTD::forward<_Arg1>(__arg1))
      , __base2(_CUDA_VSTD::forward<_Arg2>(__arg2))
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
  _CCCL_API friend constexpr void swap(__mdspan_ebco& __x, __mdspan_ebco& __y)
  {
    swap(__x.__get<0>(), __y.__get<0>());
    swap(__x.__get<1>(), __y.__get<1>());
  }
};

template <class _Elem1, class _Elem2, class _Elem3>
struct _CCCL_DECLSPEC_EMPTY_BASES __mdspan_ebco<_Elem1, _Elem2, _Elem3>
    : __mdspan_ebco_impl<0, _Elem1>
    , __mdspan_ebco_impl<1, _Elem2>
    , __mdspan_ebco_impl<2, _Elem3>
{
  using __base1 = __mdspan_ebco_impl<0, _Elem1>;
  using __base2 = __mdspan_ebco_impl<1, _Elem2>;
  using __base3 = __mdspan_ebco_impl<2, _Elem3>;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Elem1_ = _Elem1, class _Elem2_ = _Elem2, class _Elem3_ = _Elem3)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_default_constructible, _Elem1_) _CCCL_AND _CCCL_TRAIT(is_default_constructible, _Elem2_)
                   _CCCL_AND _CCCL_TRAIT(is_default_constructible, _Elem3_))
  _CCCL_API constexpr __mdspan_ebco() noexcept(
    _CCCL_TRAIT(is_nothrow_default_constructible, _Elem1_) && _CCCL_TRAIT(is_nothrow_default_constructible, _Elem2_)
    && _CCCL_TRAIT(is_nothrow_default_constructible, _Elem3_))
      : __base1()
      , __base2()
      , __base3()
  {}

  template <class _Arg1>
  static constexpr bool __is_constructible_from_one_arg =
    _CCCL_TRAIT(is_constructible, _Elem1, _Arg1) && _CCCL_TRAIT(is_default_constructible, _Elem2)
    && _CCCL_TRAIT(is_default_constructible, _Elem3);

  template <class _Arg1>
  static constexpr bool __is_nothrow_constructible_from_one_arg =
    _CCCL_TRAIT(is_nothrow_constructible, _Elem1, _Arg1) && _CCCL_TRAIT(is_nothrow_default_constructible, _Elem2)
    && _CCCL_TRAIT(is_nothrow_default_constructible, _Elem3);

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Arg1)
  _CCCL_REQUIRES(__is_constructible_from_one_arg<_Arg1>)
  _CCCL_API constexpr __mdspan_ebco(_Arg1&& __arg1) noexcept(__is_nothrow_constructible_from_one_arg<_Arg1>)
      : __base1(_CUDA_VSTD::forward<_Arg1>(__arg1))
      , __base2()
      , __base3()
  {}

  template <class _Arg1, class _Arg2>
  static constexpr bool __is_constructible_from_two_args =
    _CCCL_TRAIT(is_constructible, _Elem1, _Arg1) && _CCCL_TRAIT(is_constructible, _Elem2, _Arg2)
    && _CCCL_TRAIT(is_default_constructible, _Elem3);

  template <class _Arg1, class _Arg2>
  static constexpr bool __is_nothrow_constructible_from_two_args =
    _CCCL_TRAIT(is_nothrow_constructible, _Elem1, _Arg1) && _CCCL_TRAIT(is_nothrow_constructible, _Elem2, _Arg2)
    && _CCCL_TRAIT(is_nothrow_default_constructible, _Elem3);

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Arg1, class _Arg2)
  _CCCL_REQUIRES(__is_constructible_from_two_args<_Arg1, _Arg2>)
  _CCCL_API constexpr __mdspan_ebco(_Arg1&& __arg1,
                                    _Arg2&& __arg2) noexcept(__is_nothrow_constructible_from_two_args<_Arg1, _Arg2>)
      : __base1(_CUDA_VSTD::forward<_Arg1>(__arg1))
      , __base2(_CUDA_VSTD::forward<_Arg2>(__arg2))
      , __base3()
  {}

  template <class _Arg1, class _Arg2, class _Arg3>
  static constexpr bool __is_constructible_from_three_args =
    _CCCL_TRAIT(is_constructible, _Elem1, _Arg1) && _CCCL_TRAIT(is_constructible, _Elem2, _Arg2)
    && _CCCL_TRAIT(is_constructible, _Elem3, _Arg3);

  template <class _Arg1, class _Arg2, class _Arg3>
  static constexpr bool __is_nothrow_constructible_from_three_args =
    _CCCL_TRAIT(is_nothrow_constructible, _Elem1, _Arg1) && _CCCL_TRAIT(is_nothrow_constructible, _Elem2, _Arg2)
    && _CCCL_TRAIT(is_nothrow_constructible, _Elem3, _Arg3);

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Arg1, class _Arg2, class _Arg3)
  _CCCL_REQUIRES(__is_constructible_from_three_args<_Arg1, _Arg2, _Arg3>)
  _CCCL_API constexpr __mdspan_ebco(_Arg1&& __arg1, _Arg2&& __arg2, _Arg3&& __arg3) noexcept(
    __is_nothrow_constructible_from_three_args<_Arg1, _Arg2, _Arg3>)
      : __base1(_CUDA_VSTD::forward<_Arg1>(__arg1))
      , __base2(_CUDA_VSTD::forward<_Arg2>(__arg2))
      , __base3(_CUDA_VSTD::forward<_Arg3>(__arg3))
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
  _CCCL_API friend constexpr void swap(__mdspan_ebco& __x, __mdspan_ebco& __y)
  {
    swap(__x.__get<0>(), __y.__get<0>());
    swap(__x.__get<1>(), __y.__get<1>());
    swap(__x.__get<2>(), __y.__get<2>());
  }
};

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___MDSPAN_EMPTY_BASE_H
