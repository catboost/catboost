// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___MEMORY_COMPRESSED_PAIR_H
#define _LIBCUDACXX___MEMORY_COMPRESSED_PAIR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__fwd/get.h>
#include <cuda/std/__fwd/tuple.h>
#include <cuda/std/__tuple_dir/tuple_indices.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/dependent_type.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_default_constructible.h>
#include <cuda/std/__type_traits/is_empty.h>
#include <cuda/std/__type_traits/is_final.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_swappable.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/piecewise_construct.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// Tag used to default initialize one or both of the pair's elements.
struct __default_init_tag
{};
struct __value_init_tag
{};

template <class _Tp, int _Idx, bool _CanBeEmptyBase = _CCCL_TRAIT(is_empty, _Tp) && !_CCCL_TRAIT(is_final, _Tp)>
struct __compressed_pair_elem
{
  using _ParamT         = _Tp;
  using reference       = _Tp&;
  using const_reference = const _Tp&;

  _CCCL_API constexpr explicit __compressed_pair_elem(__default_init_tag) noexcept(
    _CCCL_TRAIT(is_nothrow_default_constructible, _Tp))
  {}
  _CCCL_API constexpr explicit __compressed_pair_elem(__value_init_tag) noexcept(
    _CCCL_TRAIT(is_nothrow_default_constructible, _Tp))
      : __value_()
  {}

  template <class _Up, enable_if_t<!_CCCL_TRAIT(is_same, __compressed_pair_elem, decay_t<_Up>), int> = 0>
  _CCCL_API constexpr explicit __compressed_pair_elem(_Up&& __u) noexcept(
    _CCCL_TRAIT(is_nothrow_constructible, _Tp, _Up))
      : __value_(_CUDA_VSTD::forward<_Up>(__u))
  {}

  template <class... _Args, size_t... _Indices>
  _CCCL_API constexpr explicit __compressed_pair_elem(
    piecewise_construct_t,
    tuple<_Args...> __args,
    __tuple_indices<_Indices...>) noexcept(_CCCL_TRAIT(is_nothrow_constructible, _Tp, _Args...))
      : __value_(_CUDA_VSTD::forward<_Args>(_CUDA_VSTD::get<_Indices>(__args))...)
  {}

  _CCCL_API constexpr reference __get() noexcept
  {
    return __value_;
  }
  _CCCL_API constexpr const_reference __get() const noexcept
  {
    return __value_;
  }

private:
  _Tp __value_;
};

template <class _Tp, int _Idx>
struct __compressed_pair_elem<_Tp, _Idx, true> : private _Tp
{
  using _ParamT         = _Tp;
  using reference       = _Tp&;
  using const_reference = const _Tp&;
  using __value_type    = _Tp;

  _CCCL_HIDE_FROM_ABI explicit constexpr __compressed_pair_elem() = default;

  _CCCL_API constexpr explicit __compressed_pair_elem(__default_init_tag) noexcept(
    _CCCL_TRAIT(is_nothrow_default_constructible, _Tp))
  {}
  _CCCL_API constexpr explicit __compressed_pair_elem(__value_init_tag) noexcept(
    _CCCL_TRAIT(is_nothrow_default_constructible, _Tp))
      : __value_type()
  {}

  template <class _Up, enable_if_t<!_CCCL_TRAIT(is_same, __compressed_pair_elem, decay_t<_Up>), int> = 0>
  _CCCL_API constexpr explicit __compressed_pair_elem(_Up&& __u) noexcept(
    _CCCL_TRAIT(is_nothrow_constructible, _Tp, _Up))
      : __value_type(_CUDA_VSTD::forward<_Up>(__u))
  {}

  template <class... _Args, size_t... _Indices>
  _CCCL_API constexpr __compressed_pair_elem(
    piecewise_construct_t,
    tuple<_Args...> __args,
    __tuple_indices<_Indices...>) noexcept(_CCCL_TRAIT(is_nothrow_constructible, _Tp, _Args...))
      : __value_type(_CUDA_VSTD::forward<_Args>(_CUDA_VSTD::get<_Indices>(__args))...)
  {}

  _CCCL_API constexpr reference __get() noexcept
  {
    return *this;
  }
  _CCCL_API constexpr const_reference __get() const noexcept
  {
    return *this;
  }
};

template <class _T1, class _T2>
class __compressed_pair
    : private __compressed_pair_elem<_T1, 0>
    , private __compressed_pair_elem<_T2, 1>
{
public:
  // NOTE: This static assert should never fire because __compressed_pair
  // is *almost never* used in a scenario where it's possible for T1 == T2.
  // (The exception is std::function where it is possible that the function
  //  object and the allocator have the same type).
  static_assert((!_CCCL_TRAIT(is_same, _T1, _T2)),
                "__compressed_pair cannot be instantiated when T1 and T2 are the same type; "
                "The current implementation is NOT ABI-compatible with the previous implementation for this "
                "configuration");

  using _Base1 _CCCL_NODEBUG_ALIAS = __compressed_pair_elem<_T1, 0>;
  using _Base2 _CCCL_NODEBUG_ALIAS = __compressed_pair_elem<_T2, 1>;

  template <bool _Dummy = true,
            class       = enable_if_t<__dependent_type<is_default_constructible<_T1>, _Dummy>::value
                                      && __dependent_type<is_default_constructible<_T2>, _Dummy>::value>>
  _CCCL_API constexpr explicit __compressed_pair() noexcept(
    _CCCL_TRAIT(is_nothrow_default_constructible, _T1) && _CCCL_TRAIT(is_nothrow_default_constructible, _T2))
      : _Base1(__value_init_tag())
      , _Base2(__value_init_tag())
  {}

  template <class _U1, class _U2>
  _CCCL_API constexpr explicit __compressed_pair(_U1&& __t1, _U2&& __t2) noexcept(
    _CCCL_TRAIT(is_constructible, _T1, _U1) && _CCCL_TRAIT(is_constructible, _T2, _U2))
      : _Base1(_CUDA_VSTD::forward<_U1>(__t1))
      , _Base2(_CUDA_VSTD::forward<_U2>(__t2))
  {}

  template <class... _Args1, class... _Args2>
  _CCCL_API constexpr explicit __compressed_pair(
    piecewise_construct_t __pc,
    tuple<_Args1...> __first_args,
    tuple<_Args2...> __second_args) noexcept(_CCCL_TRAIT(is_constructible, _T1, _Args1...)
                                             && _CCCL_TRAIT(is_constructible, _T2, _Args2...))
      : _Base1(__pc, _CUDA_VSTD::move(__first_args), typename __make_tuple_indices<sizeof...(_Args1)>::type())
      , _Base2(__pc, _CUDA_VSTD::move(__second_args), typename __make_tuple_indices<sizeof...(_Args2)>::type())
  {}

  _CCCL_API constexpr typename _Base1::reference first() noexcept
  {
    return static_cast<_Base1&>(*this).__get();
  }

  _CCCL_API constexpr typename _Base1::const_reference first() const noexcept
  {
    return static_cast<_Base1 const&>(*this).__get();
  }

  _CCCL_API constexpr typename _Base2::reference second() noexcept
  {
    return static_cast<_Base2&>(*this).__get();
  }

  _CCCL_API constexpr typename _Base2::const_reference second() const noexcept
  {
    return static_cast<_Base2 const&>(*this).__get();
  }

  _CCCL_API constexpr static _Base1* __get_first_base(__compressed_pair* __pair) noexcept
  {
    return static_cast<_Base1*>(__pair);
  }
  _CCCL_API constexpr static _Base2* __get_second_base(__compressed_pair* __pair) noexcept
  {
    return static_cast<_Base2*>(__pair);
  }

  _CCCL_API constexpr void
  swap(__compressed_pair& __x) noexcept(__is_nothrow_swappable<_T1>::value && __is_nothrow_swappable<_T2>::value)
  {
    using _CUDA_VSTD::swap;
    swap(first(), __x.first());
    swap(second(), __x.second());
  }
};

template <class _T1, class _T2>
_CCCL_API constexpr void swap(__compressed_pair<_T1, _T2>& __x, __compressed_pair<_T1, _T2>& __y) noexcept(
  __is_nothrow_swappable<_T1>::value && __is_nothrow_swappable<_T2>::value)
{
  __x.swap(__y);
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___MEMORY_COMPRESSED_PAIR_H
