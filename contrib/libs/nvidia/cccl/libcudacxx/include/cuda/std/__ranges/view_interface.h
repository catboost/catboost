// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef _LIBCUDACXX___RANGES_VIEW_INTERFACE_H
#define _LIBCUDACXX___RANGES_VIEW_INTERFACE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/derived_from.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__iterator/prev.h>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/empty.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_class.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

#if _CCCL_HAS_CONCEPTS()
template <class _Tp>
concept __can_empty = requires(_Tp& __t) { _CUDA_VRANGES::empty(__t); };
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(__can_empty_, requires(_Tp& __t)(typename(decltype(_CUDA_VRANGES::empty(__t)))));

template <class _Tp>
_CCCL_CONCEPT __can_empty = _CCCL_FRAGMENT(__can_empty_, _Tp);
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

#if _CCCL_HAS_CONCEPTS()
template <class _Derived>
  requires is_class_v<_Derived> && same_as<_Derived, remove_cv_t<_Derived>>
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
template <class _Derived, enable_if_t<is_class_v<_Derived> && same_as<_Derived, remove_cv_t<_Derived>>, int>>
#endif //  ^^^ !_CCCL_HAS_CONCEPTS() ^^^
class view_interface
{
  _CCCL_API constexpr _Derived& __derived() noexcept
  {
    static_assert(sizeof(_Derived) && derived_from<_Derived, view_interface> && view<_Derived>, "");
    return static_cast<_Derived&>(*this);
  }

  _CCCL_API constexpr _Derived const& __derived() const noexcept
  {
    static_assert(sizeof(_Derived) && derived_from<_Derived, view_interface> && view<_Derived>, "");
    return static_cast<_Derived const&>(*this);
  }

public:
  _CCCL_TEMPLATE(class _D2 = _Derived)
  _CCCL_REQUIRES(forward_range<_D2>)
  [[nodiscard]] _CCCL_API constexpr bool empty()
  {
    return _CUDA_VRANGES::begin(__derived()) == _CUDA_VRANGES::end(__derived());
  }

  _CCCL_TEMPLATE(class _D2 = _Derived)
  _CCCL_REQUIRES(forward_range<const _D2>)
  [[nodiscard]] _CCCL_API constexpr bool empty() const
  {
    return _CUDA_VRANGES::begin(__derived()) == _CUDA_VRANGES::end(__derived());
  }

  _CCCL_TEMPLATE(class _D2 = _Derived)
  _CCCL_REQUIRES(__can_empty<_D2>)
  _CCCL_API constexpr explicit operator bool()
  {
    return !_CUDA_VRANGES::empty(__derived());
  }

  _CCCL_TEMPLATE(class _D2 = _Derived)
  _CCCL_REQUIRES(__can_empty<const _D2>)
  _CCCL_API constexpr explicit operator bool() const
  {
    return !_CUDA_VRANGES::empty(__derived());
  }

  _CCCL_TEMPLATE(class _D2 = _Derived)
  _CCCL_REQUIRES(contiguous_iterator<iterator_t<_D2>>)
  _CCCL_API constexpr auto data()
  {
    return _CUDA_VSTD::to_address(_CUDA_VRANGES::begin(__derived()));
  }

  _CCCL_TEMPLATE(class _D2 = _Derived)
  _CCCL_REQUIRES(range<const _D2> _CCCL_AND contiguous_iterator<iterator_t<const _D2>>)
  _CCCL_API constexpr auto data() const
  {
    return _CUDA_VSTD::to_address(_CUDA_VRANGES::begin(__derived()));
  }

  _CCCL_TEMPLATE(class _D2 = _Derived)
  _CCCL_REQUIRES(forward_range<_D2> _CCCL_AND sized_sentinel_for<sentinel_t<_D2>, iterator_t<_D2>>)
  _CCCL_API constexpr auto size()
  {
    return _CUDA_VSTD::__to_unsigned_like(_CUDA_VRANGES::end(__derived()) - _CUDA_VRANGES::begin(__derived()));
  }

  _CCCL_TEMPLATE(class _D2 = _Derived)
  _CCCL_REQUIRES(forward_range<const _D2> _CCCL_AND sized_sentinel_for<sentinel_t<const _D2>, iterator_t<const _D2>>)
  _CCCL_API constexpr auto size() const
  {
    return _CUDA_VSTD::__to_unsigned_like(_CUDA_VRANGES::end(__derived()) - _CUDA_VRANGES::begin(__derived()));
  }

  _CCCL_TEMPLATE(class _D2 = _Derived)
  _CCCL_REQUIRES(forward_range<_D2>)
  _CCCL_API constexpr decltype(auto) front()
  {
    _CCCL_ASSERT(!empty(), "Precondition `!empty()` not satisfied. `.front()` called on an empty view.");
    return *_CUDA_VRANGES::begin(__derived());
  }

  _CCCL_TEMPLATE(class _D2 = _Derived)
  _CCCL_REQUIRES(forward_range<const _D2>)
  _CCCL_API constexpr decltype(auto) front() const
  {
    _CCCL_ASSERT(!empty(), "Precondition `!empty()` not satisfied. `.front()` called on an empty view.");
    return *_CUDA_VRANGES::begin(__derived());
  }

  _CCCL_TEMPLATE(class _D2 = _Derived)
  _CCCL_REQUIRES(bidirectional_range<_D2> _CCCL_AND common_range<_D2>)
  _CCCL_API constexpr decltype(auto) back()
  {
    _CCCL_ASSERT(!empty(), "Precondition `!empty()` not satisfied. `.back()` called on an empty view.");
    return *_CUDA_VRANGES::prev(_CUDA_VRANGES::end(__derived()));
  }

  _CCCL_TEMPLATE(class _D2 = _Derived)
  _CCCL_REQUIRES(bidirectional_range<const _D2> _CCCL_AND common_range<const _D2>)
  _CCCL_API constexpr decltype(auto) back() const
  {
    _CCCL_ASSERT(!empty(), "Precondition `!empty()` not satisfied. `.back()` called on an empty view.");
    return *_CUDA_VRANGES::prev(_CUDA_VRANGES::end(__derived()));
  }

  _CCCL_TEMPLATE(class _RARange = _Derived)
  _CCCL_REQUIRES(random_access_range<_RARange>)
  _CCCL_API constexpr decltype(auto) operator[](range_difference_t<_RARange> __index)
  {
    return _CUDA_VRANGES::begin(__derived())[__index];
  }

  _CCCL_TEMPLATE(class _RARange = const _Derived)
  _CCCL_REQUIRES(random_access_range<_RARange>)
  _CCCL_API constexpr decltype(auto) operator[](range_difference_t<_RARange> __index) const
  {
    return _CUDA_VRANGES::begin(__derived())[__index];
  }
};

_LIBCUDACXX_END_NAMESPACE_RANGES

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___RANGES_VIEW_INTERFACE_H
