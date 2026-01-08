// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_ISTREAM_ITERATOR_H
#define _LIBCUDACXX___ITERATOR_ISTREAM_ITERATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__fwd/char_traits.h>
#include <cuda/std/__iterator/default_sentinel.h>
#include <cuda/std/__iterator/iterator.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/cstddef>
#include <cuda/std/detail/libcxx/include/iosfwd>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_SUPPRESS_DEPRECATED_PUSH
template <class _Tp, class _CharT = char, class _Traits = char_traits<_CharT>, class _Distance = ptrdiff_t>
class _CCCL_TYPE_VISIBILITY_DEFAULT istream_iterator
#if !defined(_LIBCUDACXX_ABI_NO_ITERATOR_BASES)
    : public iterator<input_iterator_tag, _Tp, _Distance, const _Tp*, const _Tp&>
#endif // !_LIBCUDACXX_ABI_NO_ITERATOR_BASES
{
public:
  using iterator_category = input_iterator_tag;
  using value_type        = _Tp;
  using difference_type   = _Distance;
  using pointer           = const _Tp*;
  using reference         = const _Tp&;
  using char_type         = _CharT;
  using traits_type       = _Traits;
  using istream_type      = basic_istream<_CharT, _Traits>;

private:
  istream_type* __in_stream_;
  _Tp __value_;

public:
  _CCCL_API constexpr istream_iterator()
      : __in_stream_(nullptr)
      , __value_()
  {}
  _CCCL_API constexpr istream_iterator(default_sentinel_t)
      : istream_iterator()
  {}
  _CCCL_API inline istream_iterator(istream_type& __s)
      : __in_stream_(_CUDA_VSTD::addressof(__s))
  {
    if (!(*__in_stream_ >> __value_))
    {
      __in_stream_ = nullptr;
    }
  }

  _CCCL_API inline const _Tp& operator*() const
  {
    return __value_;
  }
  _CCCL_API inline const _Tp* operator->() const
  {
    return _CUDA_VSTD::addressof((operator*()));
  }
  _CCCL_API inline istream_iterator& operator++()
  {
    if (!(*__in_stream_ >> __value_))
    {
      __in_stream_ = nullptr;
    }
    return *this;
  }
  _CCCL_API inline istream_iterator operator++(int)
  {
    istream_iterator __t(*this);
    ++(*this);
    return __t;
  }

  template <class _Up, class _CharU, class _TraitsU, class _DistanceU>
  friend _CCCL_API inline bool operator==(const istream_iterator<_Up, _CharU, _TraitsU, _DistanceU>& __x,
                                          const istream_iterator<_Up, _CharU, _TraitsU, _DistanceU>& __y);

  friend _CCCL_API inline bool operator==(const istream_iterator& __i, default_sentinel_t)
  {
    return __i.__in_stream_ == nullptr;
  }
#if _CCCL_STD_VER < 2020
  friend _CCCL_API inline bool operator==(default_sentinel_t, const istream_iterator& __i)
  {
    return __i.__in_stream_ == nullptr;
  }
  friend _CCCL_API inline bool operator!=(const istream_iterator& __i, default_sentinel_t)
  {
    return __i.__in_stream_ != nullptr;
  }
  friend _CCCL_API inline bool operator!=(default_sentinel_t, const istream_iterator& __i)
  {
    return __i.__in_stream_ != nullptr;
  }
#endif // _CCCL_STD_VER < 2020
};
_CCCL_SUPPRESS_DEPRECATED_POP

template <class _Tp, class _CharT, class _Traits, class _Distance>
_CCCL_API inline bool operator==(const istream_iterator<_Tp, _CharT, _Traits, _Distance>& __x,
                                 const istream_iterator<_Tp, _CharT, _Traits, _Distance>& __y)
{
  return __x.__in_stream_ == __y.__in_stream_;
}

template <class _Tp, class _CharT, class _Traits, class _Distance>
_CCCL_API inline bool operator!=(const istream_iterator<_Tp, _CharT, _Traits, _Distance>& __x,
                                 const istream_iterator<_Tp, _CharT, _Traits, _Distance>& __y)
{
  return !(__x == __y);
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ITERATOR_ISTREAM_ITERATOR_H
