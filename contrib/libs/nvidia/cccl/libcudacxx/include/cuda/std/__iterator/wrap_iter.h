// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_WRAP_ITER_H
#define _LIBCUDACXX___ITERATOR_WRAP_ITER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_trivially_copy_assignable.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Iter>
class __wrap_iter
{
public:
  using iterator_type     = _Iter;
  using value_type        = typename iterator_traits<iterator_type>::value_type;
  using difference_type   = typename iterator_traits<iterator_type>::difference_type;
  using pointer           = typename iterator_traits<iterator_type>::pointer;
  using reference         = typename iterator_traits<iterator_type>::reference;
  using iterator_category = typename iterator_traits<iterator_type>::iterator_category;

  using iterator_concept = contiguous_iterator_tag;

private:
  iterator_type __i_;

public:
  _CCCL_API constexpr __wrap_iter() noexcept
      : __i_()
  {}
  template <class _Up>
  _CCCL_API constexpr __wrap_iter(
    const __wrap_iter<_Up>& __u,
    typename enable_if<is_convertible<_Up, iterator_type>::value>::type* = nullptr) noexcept
      : __i_(__u.base())
  {}
  _CCCL_API constexpr reference operator*() const noexcept
  {
    return *__i_;
  }
  _CCCL_API constexpr pointer operator->() const noexcept
  {
    return _CUDA_VSTD::__to_address(__i_);
  }
  _CCCL_API constexpr __wrap_iter& operator++() noexcept
  {
    ++__i_;
    return *this;
  }
  _CCCL_API constexpr __wrap_iter operator++(int) noexcept
  {
    __wrap_iter __tmp(*this);
    ++(*this);
    return __tmp;
  }

  _CCCL_API constexpr __wrap_iter& operator--() noexcept
  {
    --__i_;
    return *this;
  }
  _CCCL_API constexpr __wrap_iter operator--(int) noexcept
  {
    __wrap_iter __tmp(*this);
    --(*this);
    return __tmp;
  }
  _CCCL_API constexpr __wrap_iter operator+(difference_type __n) const noexcept
  {
    __wrap_iter __w(*this);
    __w += __n;
    return __w;
  }
  _CCCL_API constexpr __wrap_iter& operator+=(difference_type __n) noexcept
  {
    __i_ += __n;
    return *this;
  }
  _CCCL_API constexpr __wrap_iter operator-(difference_type __n) const noexcept
  {
    return *this + (-__n);
  }
  _CCCL_API constexpr __wrap_iter& operator-=(difference_type __n) noexcept
  {
    *this += -__n;
    return *this;
  }
  _CCCL_API constexpr reference operator[](difference_type __n) const noexcept
  {
    return __i_[__n];
  }

  _CCCL_API constexpr iterator_type base() const noexcept
  {
    return __i_;
  }

private:
  _CCCL_API constexpr __wrap_iter(iterator_type __x) noexcept
      : __i_(__x)
  {}

  template <class _Up>
  friend class __wrap_iter;
  template <class _CharT, class _Traits, class _Alloc>
  friend class basic_string;
  template <class _Tp, class _Alloc>
  friend class vector;
  template <class _Tp, size_t>
  friend class span;
};

template <class _Iter1>
_CCCL_API constexpr bool operator==(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter1>& __y) noexcept
{
  return __x.base() == __y.base();
}

template <class _Iter1, class _Iter2>
_CCCL_API constexpr bool operator==(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) noexcept
{
  return __x.base() == __y.base();
}

template <class _Iter1>
_CCCL_API constexpr bool operator<(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter1>& __y) noexcept
{
  return __x.base() < __y.base();
}

template <class _Iter1, class _Iter2>
_CCCL_API constexpr bool operator<(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) noexcept
{
  return __x.base() < __y.base();
}

template <class _Iter1>
_CCCL_API constexpr bool operator!=(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter1>& __y) noexcept
{
  return !(__x == __y);
}

template <class _Iter1, class _Iter2>
_CCCL_API constexpr bool operator!=(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) noexcept
{
  return !(__x == __y);
}

template <class _Iter1>
_CCCL_API constexpr bool operator>(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter1>& __y) noexcept
{
  return __y < __x;
}

template <class _Iter1, class _Iter2>
_CCCL_API constexpr bool operator>(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) noexcept
{
  return __y < __x;
}

template <class _Iter1>
_CCCL_API constexpr bool operator>=(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter1>& __y) noexcept
{
  return !(__x < __y);
}

template <class _Iter1, class _Iter2>
_CCCL_API constexpr bool operator>=(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) noexcept
{
  return !(__x < __y);
}

template <class _Iter1>
_CCCL_API constexpr bool operator<=(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter1>& __y) noexcept
{
  return !(__y < __x);
}

template <class _Iter1, class _Iter2>
_CCCL_API constexpr bool operator<=(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) noexcept
{
  return !(__y < __x);
}

template <class _Iter1, class _Iter2>
_CCCL_API constexpr auto operator-(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) noexcept
  -> decltype(__x.base() - __y.base())
{
  return __x.base() - __y.base();
}

template <class _Iter1>
_CCCL_API constexpr __wrap_iter<_Iter1>
operator+(typename __wrap_iter<_Iter1>::difference_type __n, __wrap_iter<_Iter1> __x) noexcept
{
  __x += __n;
  return __x;
}

#if _CCCL_STD_VER <= 2017
template <class _It>
inline constexpr bool __has_contiguous_traversal<__wrap_iter<_It>> = true;
#endif

template <class _It>
struct _CCCL_TYPE_VISIBILITY_DEFAULT pointer_traits<__wrap_iter<_It>>
{
  using pointer         = __wrap_iter<_It>;
  using element_type    = typename pointer_traits<_It>::element_type;
  using difference_type = typename pointer_traits<_It>::difference_type;

  _CCCL_API constexpr static element_type* to_address(pointer __w) noexcept
  {
    return _CUDA_VSTD::__to_address(__w.base());
  }
};

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ITERATOR_WRAP_ITER_H
