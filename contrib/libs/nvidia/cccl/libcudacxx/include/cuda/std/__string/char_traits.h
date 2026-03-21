//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___STRING_CHAR_TRAITS_H
#define _LIBCUDACXX___STRING_CHAR_TRAITS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__fwd/char_traits.h>
#include <cuda/std/__string/constexpr_c_functions.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _CharT, class _IntT, _IntT _EOFVal = _IntT(-1) /*todo: remove default argument*/>
struct __cccl_char_traits_impl
{
  using char_type = _CharT;
  using int_type  = _IntT;
#if 0 // todo: add stream support
  using off_type            = streamoff;
  using pos_type            = fpos<mbstate_t>;
  using state_type          = mbstate_t;
#endif
#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  using comparison_category = strong_ordering;
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

  _CCCL_API static constexpr void assign(char_type& __lhs, const char_type& __rhs) noexcept
  {
    __lhs = __rhs;
  }

  [[nodiscard]] _CCCL_API static constexpr bool eq(char_type __lhs, char_type __rhs) noexcept
  {
    return __lhs == __rhs;
  }

  [[nodiscard]] _CCCL_API static constexpr bool lt(char_type __lhs, char_type __rhs) noexcept
  {
    if constexpr (_CCCL_TRAIT(is_same, char_type, char))
    {
      return static_cast<unsigned char>(__lhs) < static_cast<unsigned char>(__rhs);
    }
    else
    {
      return __lhs < __rhs;
    }
  }

  [[nodiscard]] _CCCL_API static constexpr int
  compare(const char_type* __lhs, const char_type* __rhs, size_t __count) noexcept
  {
    if (__count > 0)
    {
      _CCCL_ASSERT(__lhs != nullptr, "char_traits::compare: lhs pointer is null");
      _CCCL_ASSERT(__rhs != nullptr, "char_traits::compare: rhs pointer is null");
    }
    return _CUDA_VSTD::__cccl_memcmp(__lhs, __rhs, __count);
  }

  [[nodiscard]] _CCCL_API inline static size_t constexpr length(const char_type* __s) noexcept
  {
    _CCCL_ASSERT(__s != nullptr, "char_traits::length: nullptr passed as an argument");
    return _CUDA_VSTD::__cccl_strlen(__s);
  }

  [[nodiscard]] _CCCL_API static constexpr const char_type*
  find(const char_type* __s, size_t __n, const char_type& __a) noexcept
  {
    if (__n > 0)
    {
      _CCCL_ASSERT(__s != nullptr, "char_traits::find: nullptr passed as an argument");
    }
    return _CUDA_VSTD::__cccl_memchr<const char_type>(__s, __a, __n);
  }

  _CCCL_API static constexpr char_type* move(char_type* __s1, const char_type* __s2, size_t __n) noexcept
  {
    if (__n > 0)
    {
      _CCCL_ASSERT(__s1 != nullptr, "char_traits::move: destination pointer is null");
      _CCCL_ASSERT(__s2 != nullptr, "char_traits::move: source pointer is null");
    }
    return _CUDA_VSTD::__cccl_memmove(__s1, __s2, __n);
  }

  _CCCL_API static constexpr char_type* copy(char_type* __s1, const char_type* __s2, size_t __n) noexcept
  {
    if (__n > 0)
    {
      _CCCL_ASSERT(__s1 != nullptr, "char_traits::copy: destination pointer is null");
      _CCCL_ASSERT(__s2 != nullptr, "char_traits::copy: source pointer is null");
    }
    return _CUDA_VSTD::__cccl_memcpy(__s1, __s2, __n);
  }

  _CCCL_API static constexpr char_type* assign(char_type* __s, size_t __n, char_type __a) noexcept
  {
    if (__n > 0)
    {
      _CCCL_ASSERT(__s != nullptr, "char_traits::assign: destination pointer is null");
    }
    return _CUDA_VSTD::__cccl_memset(__s, __a, __n);
  }

  [[nodiscard]] _CCCL_API static constexpr char_type to_char_type(int_type __c) noexcept
  {
    return char_type(__c);
  }

  [[nodiscard]] _CCCL_API static constexpr int_type to_int_type(char_type __c) noexcept
  {
    if constexpr (_CCCL_TRAIT(is_same, char_type, char))
    {
      return int_type(static_cast<unsigned char>(__c));
    }
    else
    {
      return int_type(__c);
    }
  }

  [[nodiscard]] _CCCL_API static constexpr bool eq_int_type(int_type __lhs, int_type __rhs) noexcept
  {
    return __lhs == __rhs;
  }

#if 0 // todo: add EOF support
  [[nodiscard]] _CCCL_API static constexpr int_type eof() noexcept
  {
    return _EOFVal;
  }

  [[nodiscard]] _CCCL_API static constexpr int_type not_eof(int_type __c) noexcept
  {
    return eq_int_type(__c, eof()) ? static_cast<int_type>(~eof()) : __c;
  }
#endif
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT char_traits<char> : __cccl_char_traits_impl<char, int /*, EOF*/>
{};

#if _CCCL_HAS_CHAR8_T()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT char_traits<char8_t> : __cccl_char_traits_impl<char8_t, unsigned /*, ??? */>
{};
#endif // _CCCL_HAS_CHAR8_T()

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
char_traits<char16_t> : __cccl_char_traits_impl<char16_t, uint_least16_t /*, ??? */>
{};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
char_traits<char32_t> : __cccl_char_traits_impl<char32_t, uint_least32_t /*, ??? */>
{};

#if _CCCL_HAS_WCHAR_T()
template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT char_traits<wchar_t> : __cccl_char_traits_impl<wchar_t, wint_t /*, WEOF*/>
{};
#endif // _CCCL_HAS_WCHAR_T()

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___STRING_CHAR_TRAITS_H
