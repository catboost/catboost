//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___STRING_CONSTEXPR_C_FUNCTIONS_H
#define _LIBCUDACXX___STRING_CONSTEXPR_C_FUNCTIONS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/__type_traits/remove_const.h>
#include <cuda/std/climits>

#if !_CCCL_COMPILER(NVRTC)
#  include <cstring>
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// __cccl_strcpy

template <class _CharT>
_CCCL_API constexpr _CharT*
__cccl_strcpy_impl_constexpr(_CharT* _CCCL_RESTRICT __dst, const _CharT* _CCCL_RESTRICT __src) noexcept
{
  _CharT* __dst_it = __dst;
  while ((*__dst_it++ = *__src++) != _CharT('\0'))
  {
  }
  return __dst;
}

#if !_CCCL_COMPILER(NVRTC)
template <class _CharT>
_CCCL_HIDE_FROM_ABI _CCCL_HOST _CharT*
__cccl_strcpy_impl_host(_CharT* _CCCL_RESTRICT __dst, const _CharT* _CCCL_RESTRICT __src) noexcept
{
  if constexpr (sizeof(_CharT) == 1)
  {
    return reinterpret_cast<_CharT*>(::strcpy(reinterpret_cast<char*>(__dst), reinterpret_cast<const char*>(__src)));
  }
  else
  {
    return _CUDA_VSTD::__cccl_strcpy_impl_constexpr(__dst, __src);
  }
}
#endif // !_CCCL_COMPILER(NVRTC)

template <class _CharT>
_CCCL_API constexpr _CharT* __cccl_strcpy(_CharT* _CCCL_RESTRICT __dst, const _CharT* _CCCL_RESTRICT __src) noexcept
{
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    NV_IF_TARGET(NV_IS_HOST, (return _CUDA_VSTD::__cccl_strcpy_impl_host(__dst, __src);))
  }
  return _CUDA_VSTD::__cccl_strcpy_impl_constexpr(__dst, __src);
}

// __cccl_strncpy

template <class _CharT>
_CCCL_API constexpr _CharT*
__cccl_strncpy_impl_constexpr(_CharT* _CCCL_RESTRICT __dst, const _CharT* _CCCL_RESTRICT __src, size_t __n) noexcept
{
  _CharT* __dst_it = __dst;
  while (__n--)
  {
    if ((*__dst_it++ = *__src++) == _CharT('\0'))
    {
      while (__n--)
      {
        *__dst_it++ = _CharT('\0');
      }
      break;
    }
  }
  return __dst;
}

#if !_CCCL_COMPILER(NVRTC)
template <class _CharT>
_CCCL_HIDE_FROM_ABI _CCCL_HOST _CharT*
__cccl_strncpy_impl_host(_CharT* _CCCL_RESTRICT __dst, const _CharT* _CCCL_RESTRICT __src, size_t __n) noexcept
{
  if constexpr (sizeof(_CharT) == 1)
  {
    return reinterpret_cast<_CharT*>(
      ::strncpy(reinterpret_cast<char*>(__dst), reinterpret_cast<const char*>(__src), __n));
  }
  else
  {
    return _CUDA_VSTD::__cccl_strncpy_impl_constexpr(__dst, __src, __n);
  }
}
#endif // !_CCCL_COMPILER(NVRTC)

template <class _CharT>
_CCCL_API constexpr _CharT*
__cccl_strncpy(_CharT* _CCCL_RESTRICT __dst, const _CharT* _CCCL_RESTRICT __src, size_t __n) noexcept
{
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    NV_IF_TARGET(NV_IS_HOST, (return _CUDA_VSTD::__cccl_strncpy_impl_host(__dst, __src, __n);))
  }
  return _CUDA_VSTD::__cccl_strncpy_impl_constexpr(__dst, __src, __n);
}

// __cccl_strlen

template <class _CharT>
[[nodiscard]] _CCCL_API constexpr size_t __cccl_strlen_impl_constexpr(const _CharT* __ptr) noexcept
{
  size_t __len = 0;
  while (*__ptr++ != _CharT('\0'))
  {
    ++__len;
  }
  return __len;
}

#if !_CCCL_COMPILER(NVRTC)
template <class _CharT>
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_HOST size_t __cccl_strlen_impl_host(const _CharT* __ptr) noexcept
{
  if constexpr (sizeof(_CharT) == 1)
  {
    return ::strlen(reinterpret_cast<const char*>(__ptr));
  }
  else
  {
    return _CUDA_VSTD::__cccl_strlen_impl_constexpr(__ptr);
  }
}
#endif // !_CCCL_COMPILER(NVRTC)

template <class _CharT>
[[nodiscard]] _CCCL_API constexpr size_t __cccl_strlen(const _CharT* __ptr) noexcept
{
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    NV_IF_TARGET(NV_IS_HOST, (return _CUDA_VSTD::__cccl_strlen_impl_host(__ptr);))
  }
  return _CUDA_VSTD::__cccl_strlen_impl_constexpr(__ptr);
}

// __cccl_strcmp

template <class _CharT>
[[nodiscard]] _CCCL_API constexpr int __cccl_strcmp_impl_constexpr(const _CharT* __lhs, const _CharT* __rhs) noexcept
{
  using _UCharT = __make_nbit_uint_t<sizeof(_CharT) * CHAR_BIT>;

  while (*__lhs == *__rhs)
  {
    if (*__lhs == _CharT('\0'))
    {
      return 0;
    }

    ++__lhs;
    ++__rhs;
  }
  return (static_cast<_UCharT>(*__lhs) < static_cast<_UCharT>(*__rhs)) ? -1 : 1;
}

#if !_CCCL_COMPILER(NVRTC)
template <class _CharT>
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_HOST int
__cccl_strcmp_impl_host(const _CharT* __lhs, const _CharT* __rhs) noexcept
{
  if constexpr (sizeof(_CharT) == 1)
  {
    return ::strcmp(reinterpret_cast<const char*>(__lhs), reinterpret_cast<const char*>(__rhs));
  }
  else
  {
    return _CUDA_VSTD::__cccl_strcmp_impl_constexpr(__lhs, __rhs);
  }
}
#endif // !_CCCL_COMPILER(NVRTC)

template <class _CharT>
[[nodiscard]] _CCCL_API constexpr int __cccl_strcmp(const _CharT* __lhs, const _CharT* __rhs) noexcept
{
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    NV_IF_TARGET(NV_IS_HOST, (return _CUDA_VSTD::__cccl_strcmp_impl_host(__lhs, __rhs);))
  }
  return _CUDA_VSTD::__cccl_strcmp_impl_constexpr(__lhs, __rhs);
}

// __cccl_strncmp

template <class _CharT>
[[nodiscard]] _CCCL_API constexpr int
__cccl_strncmp_impl_constexpr(const _CharT* __lhs, const _CharT* __rhs, size_t __n) noexcept
{
  using _UCharT = __make_nbit_uint_t<sizeof(_CharT) * CHAR_BIT>;

  while (__n--)
  {
    if (*__lhs != *__rhs)
    {
      return (static_cast<_UCharT>(*__lhs) < static_cast<_UCharT>(*__rhs)) ? -1 : 1;
    }

    if (*__lhs == _CharT('\0'))
    {
      return 0;
    }

    ++__lhs;
    ++__rhs;
  }
  return 0;
}

#if !_CCCL_COMPILER(NVRTC)
template <class _CharT>
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_HOST int
__cccl_strncmp_impl_host(const _CharT* __lhs, const _CharT* __rhs, size_t __n) noexcept
{
  if constexpr (sizeof(_CharT) == 1)
  {
    return ::strncmp(reinterpret_cast<const char*>(__lhs), reinterpret_cast<const char*>(__rhs), __n);
  }
  else
  {
    return _CUDA_VSTD::__cccl_strncmp_impl_constexpr(__lhs, __rhs, __n);
  }
}
#endif // !_CCCL_COMPILER(NVRTC)

template <class _CharT>
[[nodiscard]] _CCCL_API constexpr int __cccl_strncmp(const _CharT* __lhs, const _CharT* __rhs, size_t __n) noexcept
{
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    NV_IF_TARGET(NV_IS_HOST, (return _CUDA_VSTD::__cccl_strncmp_impl_host(__lhs, __rhs, __n);))
  }
  return _CUDA_VSTD::__cccl_strncmp_impl_constexpr(__lhs, __rhs, __n);
}

// __cccl_strchr

template <class _CharT>
[[nodiscard]] _CCCL_API constexpr _CharT* __cccl_strchr_impl_constexpr(_CharT* __ptr, _CharT __c) noexcept
{
  while (*__ptr != __c)
  {
    if (*__ptr == _CharT('\0'))
    {
      return nullptr;
    }
    ++__ptr;
  }
  return __ptr;
}

#if !_CCCL_COMPILER(NVRTC)
template <class _CharT>
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_HOST _CharT* __cccl_strchr_impl_host(_CharT* __ptr, _CharT __c) noexcept
{
  if constexpr (sizeof(_CharT) == 1)
  {
    using _Up = remove_const_t<_CharT>;
    return const_cast<_CharT*>(
      reinterpret_cast<_Up*>(::strchr(reinterpret_cast<char*>(const_cast<_Up*>(__ptr)), static_cast<int>(__c))));
  }
  else
  {
    return _CUDA_VSTD::__cccl_strchr_impl_constexpr<_CharT>(__ptr, __c);
  }
}
#endif // !_CCCL_COMPILER(NVRTC)

template <class _CharT>
[[nodiscard]] _CCCL_API constexpr _CharT* __cccl_strchr(_CharT* __ptr, _CharT __c) noexcept
{
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    NV_IF_TARGET(NV_IS_HOST, (return _CUDA_VSTD::__cccl_strchr_impl_host<_CharT>(__ptr, __c);))
  }
  return _CUDA_VSTD::__cccl_strchr_impl_constexpr<_CharT>(__ptr, __c);
}

// __cccl_strrchr

template <class _CharT>
[[nodiscard]] _CCCL_API constexpr _CharT* __cccl_strrchr_impl_constexpr(_CharT* __ptr, _CharT __c) noexcept
{
  if (__c == _CharT('\0'))
  {
    return __ptr + _CUDA_VSTD::__cccl_strlen(__ptr);
  }

  _CharT* __last{};
  while (*__ptr != _CharT('\0'))
  {
    if (*__ptr == __c)
    {
      __last = __ptr;
    }
    ++__ptr;
  }
  return __last;
}

#if !_CCCL_COMPILER(NVRTC)
template <class _CharT>
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_HOST _CharT* __cccl_strrchr_impl_host(_CharT* __ptr, _CharT __c) noexcept
{
  if constexpr (sizeof(_CharT) == 1)
  {
    using _Up = remove_const_t<_CharT>;
    return const_cast<_CharT*>(
      reinterpret_cast<_Up*>(::strrchr(reinterpret_cast<char*>(const_cast<_Up*>(__ptr)), static_cast<int>(__c))));
  }
  else
  {
    return _CUDA_VSTD::__cccl_strrchr_impl_constexpr<_CharT>(__ptr, __c);
  }
}
#endif // !_CCCL_COMPILER(NVRTC)

template <class _CharT>
[[nodiscard]] _CCCL_API constexpr _CharT* __cccl_strrchr(_CharT* __ptr, _CharT __c) noexcept
{
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    NV_IF_TARGET(NV_IS_HOST, (return _CUDA_VSTD::__cccl_strrchr_impl_host<_CharT>(__ptr, __c);))
  }
  return _CUDA_VSTD::__cccl_strrchr_impl_constexpr<_CharT>(__ptr, __c);
}

// __cccl_memchr

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp* __cccl_memchr_impl_constexpr(_Tp* __ptr, _Tp __c, size_t __n) noexcept
{
  while (__n--)
  {
    if (*__ptr == __c)
    {
      return __ptr;
    }
    ++__ptr;
  }
  return nullptr;
}

#if !_CCCL_COMPILER(NVRTC)
template <class _Tp>
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_HOST _Tp* __cccl_memchr_impl_host(_Tp* __ptr, _Tp __c, size_t __n) noexcept
{
  if constexpr (sizeof(_Tp) == 1)
  {
    using _Up = remove_const_t<_Tp>;
    return const_cast<_Tp*>(reinterpret_cast<_Up*>(::memchr(const_cast<_Up*>(__ptr), static_cast<int>(__c), __n)));
  }
  else
  {
    return _CUDA_VSTD::__cccl_memchr_impl_constexpr<_Tp>(__ptr, __c, __n);
  }
}
#endif // !_CCCL_COMPILER(NVRTC)

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp* __cccl_memchr(_Tp* __ptr, _Tp __c, size_t __n) noexcept
{
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    NV_IF_TARGET(NV_IS_HOST, (return _CUDA_VSTD::__cccl_memchr_impl_host<_Tp>(__ptr, __c, __n);))
  }
  return _CUDA_VSTD::__cccl_memchr_impl_constexpr<_Tp>(__ptr, __c, __n);
}

// __cccl_memmove

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp* __cccl_memmove_impl_constexpr(_Tp* __dst, const _Tp* __src, size_t __n) noexcept
{
  const auto __dst_copy = __dst;

  if (__src < __dst && __dst < __src + __n)
  {
    __dst += __n;
    __src += __n;

    while (__n-- > 0)
    {
      *--__dst = *--__src;
    }
  }
  else
  {
    while (__n-- > 0)
    {
      *__dst++ = *__src++;
    }
  }
  return __dst_copy;
}

#if !_CCCL_COMPILER(NVRTC)
template <class _Tp>
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_HOST _Tp*
__cccl_memmove_impl_host(_Tp* __dst, const _Tp* __src, size_t __n) noexcept
{
  return reinterpret_cast<_Tp*>(::memmove(__dst, __src, __n * sizeof(_Tp)));
}
#endif // !_CCCL_COMPILER(NVRTC)

template <class _Tp>
_CCCL_API constexpr _Tp* __cccl_memmove(_Tp* __dst, const _Tp* __src, size_t __n) noexcept
{
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
#if defined(_CCCL_BUILTIN_MEMMOVE)
    return reinterpret_cast<_Tp*>(_CCCL_BUILTIN_MEMMOVE(__dst, __src, __n * sizeof(_Tp)));
#else // ^^^ _CCCL_BUILTIN_MEMMOVE ^^^ / vvv !_CCCL_BUILTIN_MEMMOVE vvv
    NV_IF_TARGET(NV_IS_HOST, (return _CUDA_VSTD::__cccl_memmove_impl_host(__dst, __src, __n);))
#endif // ^^^ !_CCCL_BUILTIN_MEMMOVE ^^^
  }
  return _CUDA_VSTD::__cccl_memmove_impl_constexpr(__dst, __src, __n);
}

// __cccl_memcmp

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr int
__cccl_memcmp_impl_constexpr(const _Tp* __lhs, const _Tp* __rhs, size_t __n) noexcept
{
  using _Up = __make_nbit_uint_t<sizeof(_Tp) * CHAR_BIT>;

  while (__n--)
  {
    if (*__lhs != *__rhs)
    {
      return static_cast<_Up>(*__lhs) < static_cast<_Up>(*__rhs) ? -1 : 1;
    }
    ++__lhs;
    ++__rhs;
  }
  return 0;
}

#if !_CCCL_COMPILER(NVRTC)
template <class _Tp>
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_HOST int
__cccl_memcmp_impl_host(const _Tp* __lhs, const _Tp* __rhs, size_t __n) noexcept
{
  if constexpr (sizeof(_Tp) == 1)
  {
    return ::memcmp(__lhs, __rhs, __n);
  }
  else
  {
    return _CUDA_VSTD::__cccl_memcmp_impl_constexpr(__lhs, __rhs, __n);
  }
}
#endif // !_CCCL_COMPILER(NVRTC)

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr int __cccl_memcmp(const _Tp* __lhs, const _Tp* __rhs, size_t __n) noexcept
{
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
#if defined(_CCCL_BUILTIN_MEMCMP)
    return _CCCL_BUILTIN_MEMCMP(__lhs, __rhs, __n * sizeof(_Tp));
#else // ^^^ _CCCL_BUILTIN_MEMCMP ^^^ / vvv !_CCCL_BUILTIN_MEMCMP vvv
    NV_IF_TARGET(NV_IS_HOST, (return _CUDA_VSTD::__cccl_memcmp_impl_host(__lhs, __rhs, __n);))
#endif // ^^^ !_CCCL_BUILTIN_MEMCMP ^^^
  }
  return _CUDA_VSTD::__cccl_memcmp_impl_constexpr(__lhs, __rhs, __n);
}

// __cccl_memcpy

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp*
__cccl_memcpy_impl_constexpr(_Tp* _CCCL_RESTRICT __dst, const _Tp* _CCCL_RESTRICT __src, size_t __n) noexcept
{
  const auto __dst_copy = __dst;

  while (__n--)
  {
    *__dst++ = *__src++;
  }
  return __dst_copy;
}

#if !_CCCL_COMPILER(NVRTC)
template <class _Tp>
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_HOST _Tp*
__cccl_memcpy_impl_host(_Tp* _CCCL_RESTRICT __dst, const _Tp* _CCCL_RESTRICT __src, size_t __n) noexcept
{
  return reinterpret_cast<_Tp*>(::memcpy(__dst, __src, __n * sizeof(_Tp)));
}
#endif // !_CCCL_COMPILER(NVRTC)

template <class _Tp>
_CCCL_API constexpr _Tp* __cccl_memcpy(_Tp* _CCCL_RESTRICT __dst, const _Tp* _CCCL_RESTRICT __src, size_t __n) noexcept
{
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
#if defined(_CCCL_BUILTIN_MEMCPY)
    return reinterpret_cast<_Tp*>(_CCCL_BUILTIN_MEMCPY(__dst, __src, __n * sizeof(_Tp)));
#else // ^^^ _CCCL_BUILTIN_MEMCPY ^^^ / vvv !_CCCL_BUILTIN_MEMCPY vvv
    NV_IF_TARGET(NV_IS_HOST, (return _CUDA_VSTD::__cccl_memcpy_impl_host(__dst, __src, __n);))
#endif // ^^^ !_CCCL_BUILTIN_MEMCPY ^^^
  }
  return _CUDA_VSTD::__cccl_memcpy_impl_constexpr(__dst, __src, __n);
}

// __cccl_memset

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp* __cccl_memset_impl_constexpr(_Tp* __ptr, _Tp __c, size_t __n) noexcept
{
  const auto __ptr_copy = __ptr;

  while (__n--)
  {
    *__ptr++ = __c;
  }
  return __ptr_copy;
}

#if !_CCCL_COMPILER(NVRTC)
template <class _Tp>
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_HOST _Tp* __cccl_memset_impl_host(_Tp* __ptr, _Tp __c, size_t __n) noexcept
{
  if constexpr (sizeof(_Tp) == 1)
  {
    return reinterpret_cast<_Tp*>(::memset(__ptr, static_cast<int>(__c), __n));
  }
  else
  {
    return _CUDA_VSTD::__cccl_memset_impl_constexpr(__ptr, __c, __n);
  }
}
#endif // !_CCCL_COMPILER(NVRTC)

template <class _Tp>
_CCCL_API constexpr _Tp* __cccl_memset(_Tp* __ptr, _Tp __c, size_t __n) noexcept
{
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
#if defined(_CCCL_BUILTIN_MEMSET)
    return reinterpret_cast<_Tp*>(_CCCL_BUILTIN_MEMSET(__ptr, __c, __n * sizeof(_Tp)));
#else // ^^^ _CCCL_BUILTIN_MEMSET ^^^ / vvv !_CCCL_BUILTIN_MEMSET vvv
    NV_IF_TARGET(NV_IS_HOST, (return _CUDA_VSTD::__cccl_memset_impl_host(__ptr, __c, __n);))
#endif // ^^^ !_CCCL_BUILTIN_MEMSET ^^^
  }
  return _CUDA_VSTD::__cccl_memset_impl_constexpr(__ptr, __c, __n);
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___STRING_CONSTEXPR_C_FUNCTIONS_H
