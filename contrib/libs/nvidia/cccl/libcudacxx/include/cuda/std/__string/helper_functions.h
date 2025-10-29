//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___STRING_HELPER_FUNCTIONS_H
#define _LIBCUDACXX___STRING_HELPER_FUNCTIONS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/find.h>
#include <cuda/std/__algorithm/find_end.h>
#include <cuda/std/__algorithm/find_first_of.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__string/char_traits.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _CharT, class _SizeT, class _Traits, _SizeT __npos>
_CCCL_API constexpr _SizeT __cccl_str_find(const _CharT* __p, _SizeT __sz, _CharT __c, _SizeT __pos) noexcept
{
  if (__pos >= __sz)
  {
    return __npos;
  }
  const _CharT* __r = _Traits::find(__p + __pos, __sz - __pos, __c);
  if (__r == nullptr)
  {
    return __npos;
  }
  return static_cast<_SizeT>(__r - __p);
}

template <class _CharT, class _Traits>
_CCCL_API constexpr const _CharT*
__cccl_search_substring(const _CharT* __first1, const _CharT* __last1, const _CharT* __first2, const _CharT* __last2)
{
  // Take advantage of knowing source and pattern lengths.
  // Stop short when source is smaller than pattern.
  const ptrdiff_t __len2 = __last2 - __first2;
  if (__len2 == 0)
  {
    return __first1;
  }

  ptrdiff_t __len1 = __last1 - __first1;
  if (__len1 < __len2)
  {
    return __last1;
  }

  // First element of __first2 is loop invariant.
  _CharT __f2 = *__first2;
  while (true)
  {
    __len1 = __last1 - __first1;
    // Check whether __first1 still has at least __len2 bytes.
    if (__len1 < __len2)
    {
      return __last1;
    }

    // Find __f2 the first byte matching in __first1.
    __first1 = _Traits::find(__first1, __len1 - __len2 + 1, __f2);
    if (__first1 == 0)
    {
      return __last1;
    }

    // It is faster to compare from the first byte of __first1 even if we
    // already know that it matches the first byte of __first2: this is because
    // __first2 is most likely aligned, as it is user's "pattern" string, and
    // __first1 + 1 is most likely not aligned, as the match is in the middle of
    // the string.
    if (_Traits::compare(__first1, __first2, __len2) == 0)
    {
      return __first1;
    }

    ++__first1;
  }
}

template <class _CharT, class _SizeT, class _Traits, _SizeT __npos>
_CCCL_API constexpr _SizeT
__cccl_str_find(const _CharT* __p, _SizeT __sz, const _CharT* __s, _SizeT __pos, _SizeT __n) noexcept
{
  if (__pos > __sz)
  {
    return __npos;
  }

  if (__n == 0) // There is nothing to search, just return __pos.
  {
    return __pos;
  }

  const _CharT* __r = _CUDA_VSTD::__cccl_search_substring<_CharT, _Traits>(__p + __pos, __p + __sz, __s, __s + __n);

  if (__r == __p + __sz)
  {
    return __npos;
  }
  return static_cast<_SizeT>(__r - __p);
}

template <class _CharT, class _SizeT, class _Traits, _SizeT __npos>
_CCCL_API constexpr _SizeT __cccl_str_rfind(const _CharT* __p, _SizeT __sz, _CharT __c, _SizeT __pos) noexcept
{
  if (__sz < 1)
  {
    return __npos;
  }
  if (__pos < __sz)
  {
    ++__pos;
  }
  else
  {
    __pos = __sz;
  }
  for (const _CharT* __ps = __p + __pos; __ps != __p;)
  {
    if (_Traits::eq(*--__ps, __c))
    {
      return static_cast<_SizeT>(__ps - __p);
    }
  }
  return __npos;
}

template <class _CharT, class _SizeT, class _Traits, _SizeT __npos>
_CCCL_API constexpr _SizeT
__cccl_str_rfind(const _CharT* __p, _SizeT __sz, const _CharT* __s, _SizeT __pos, _SizeT __n) noexcept
{
  __pos = _CUDA_VSTD::min(__pos, __sz);
  if (__n < __sz - __pos)
  {
    __pos += __n;
  }
  else
  {
    __pos = __sz;
  }
  const _CharT* __r = _CUDA_VSTD::__find_end(
    __p, __p + __pos, __s, __s + __n, _Traits::eq, random_access_iterator_tag(), random_access_iterator_tag());
  if (__n > 0 && __r == __p + __pos)
  {
    return __npos;
  }
  return static_cast<_SizeT>(__r - __p);
}

template <class _CharT, class _SizeT, class _Traits, _SizeT __npos>
_CCCL_API constexpr _SizeT
__cccl_str_find_first_of(const _CharT* __p, _SizeT __sz, const _CharT* __s, _SizeT __pos, _SizeT __n) noexcept
{
  if (__pos >= __sz || __n == 0)
  {
    return __npos;
  }
  const _CharT* __r = _CUDA_VSTD::__find_first_of_ce(__p + __pos, __p + __sz, __s, __s + __n, _Traits::eq);
  if (__r == __p + __sz)
  {
    return __npos;
  }
  return static_cast<_SizeT>(__r - __p);
}

template <class _CharT, class _SizeT, class _Traits, _SizeT __npos>
_CCCL_API constexpr _SizeT
__cccl_str_find_last_of(const _CharT* __p, _SizeT __sz, const _CharT* __s, _SizeT __pos, _SizeT __n) noexcept
{
  if (__n == 0)
  {
    return __npos;
  }
  if (__pos < __sz)
  {
    ++__pos;
  }
  else
  {
    __pos = __sz;
  }
  for (const _CharT* __ps = __p + __pos; __ps != __p;)
  {
    if (_Traits::find(__s, __n, *--__ps) != nullptr)
    {
      return static_cast<_SizeT>(__ps - __p);
    }
  }
  return __npos;
}

template <class _CharT, class _SizeT, class _Traits, _SizeT __npos>
_CCCL_API constexpr _SizeT
__cccl_str_find_first_not_of(const _CharT* __p, _SizeT __sz, const _CharT* __s, _SizeT __pos, _SizeT __n) noexcept
{
  if (__pos >= __sz)
  {
    return __npos;
  }
  const _CharT* __pe = __p + __sz;
  for (const _CharT* __ps = __p + __pos; __ps != __pe; ++__ps)
  {
    if (_Traits::find(__s, __n, *__ps) == nullptr)
    {
      return static_cast<_SizeT>(__ps - __p);
    }
  }
  return __npos;
}

template <class _CharT, class _SizeT, class _Traits, _SizeT __npos>
_CCCL_API constexpr _SizeT
__cccl_str_find_first_not_of(const _CharT* __p, _SizeT __sz, _CharT __c, _SizeT __pos) noexcept
{
  if (__pos < __sz)
  {
    const _CharT* __pe = __p + __sz;
    for (const _CharT* __ps = __p + __pos; __ps != __pe; ++__ps)
    {
      if (!_Traits::eq(*__ps, __c))
      {
        return static_cast<_SizeT>(__ps - __p);
      }
    }
  }
  return __npos;
}

template <class _CharT, class _SizeT, class _Traits, _SizeT __npos>
_CCCL_API constexpr _SizeT
__cccl_str_find_last_not_of(const _CharT* __p, _SizeT __sz, const _CharT* __s, _SizeT __pos, _SizeT __n) noexcept
{
  if (__pos < __sz)
  {
    ++__pos;
  }
  else
  {
    __pos = __sz;
  }
  for (const _CharT* __ps = __p + __pos; __ps != __p;)
  {
    if (_Traits::find(__s, __n, *--__ps) == nullptr)
    {
      return static_cast<_SizeT>(__ps - __p);
    }
  }
  return __npos;
}

template <class _CharT, class _SizeT, class _Traits, _SizeT __npos>
_CCCL_API constexpr _SizeT __cccl_str_find_last_not_of(const _CharT* __p, _SizeT __sz, _CharT __c, _SizeT __pos) noexcept
{
  if (__pos < __sz)
  {
    ++__pos;
  }
  else
  {
    __pos = __sz;
  }
  for (const _CharT* __ps = __p + __pos; __ps != __p;)
  {
    if (!_Traits::eq(*--__ps, __c))
    {
      return static_cast<_SizeT>(__ps - __p);
    }
  }
  return __npos;
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___STRING_HELPER_FUNCTIONS_H
