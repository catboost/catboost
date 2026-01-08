//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FWD_STRING_H
#define _LIBCUDACXX___FWD_STRING_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__fwd/allocator.h>
#include <cuda/std/__fwd/char_traits.h>
#include <cuda/std/__fwd/memory_resource.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if 0 // we don't support these features
template <class _CharT, class _Traits = char_traits<_CharT>, class _Allocator = allocator<_CharT>>
class _CCCL_TYPE_VISIBILITY_DEFAULT basic_string;

using string  = basic_string<char>;
using wstring = basic_string<wchar_t>;
#  if _CCCL_HAS_CHAR8_T()
using u8string = basic_string<char8_t>;
#  endif // _CCCL_HAS_CHAR8_T()
using u16string = basic_string<char16_t>;
using u32string = basic_string<char32_t>;

namespace pmr
{

template <class _CharT, class _Traits = char_traits<_CharT>>
using basic_string = _CUDA_VSTD::basic_string<_CharT, _Traits, polymorphic_allocator<_CharT>>;

using string  = basic_string<char>;
using wstring = basic_string<wchar_t>;
#  if _CCCL_HAS_CHAR8_T()
using u8string = basic_string<char8_t>;
#  endif // _CCCL_HAS_CHAR8_T()
using u16string = basic_string<char16_t>;
using u32string = basic_string<char32_t>;

} // namespace pmr

// clang-format off
template <class _CharT, class _Traits, class _Allocator>
class _CCCL_PREFERRED_NAME(string)
      _CCCL_PREFERRED_NAME(wstring)
#if _CCCL_HAS_CHAR8_T()
      _CCCL_PREFERRED_NAME(u8string)
#endif // _CCCL_HAS_CHAR8_T()
      _CCCL_PREFERRED_NAME(u16string)
      _CCCL_PREFERRED_NAME(u32string)
      _CCCL_PREFERRED_NAME(pmr::string)
      _CCCL_PREFERRED_NAME(pmr::wstring)
#  if _CCCL_HAS_CHAR8_T()
      _CCCL_PREFERRED_NAME(pmr::u8string)
#  endif // _CCCL_HAS_CHAR8_T()
      _CCCL_PREFERRED_NAME(pmr::u16string)
      _CCCL_PREFERRED_NAME(pmr::u32string)
      basic_string;
// clang-format on
#endif // 0

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___FWD_STRING_H
