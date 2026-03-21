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

#ifndef _LIBCUDACXX___CSTDDEF_BYTE_H
#define _LIBCUDACXX___CSTDDEF_BYTE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_integral.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD_NOVERSION

enum class byte : unsigned char
{
};

_CCCL_API constexpr byte operator|(byte __lhs, byte __rhs) noexcept
{
  return static_cast<byte>(
    static_cast<unsigned char>(static_cast<unsigned int>(__lhs) | static_cast<unsigned int>(__rhs)));
}

_CCCL_API constexpr byte& operator|=(byte& __lhs, byte __rhs) noexcept
{
  return __lhs = __lhs | __rhs;
}

_CCCL_API constexpr byte operator&(byte __lhs, byte __rhs) noexcept
{
  return static_cast<byte>(
    static_cast<unsigned char>(static_cast<unsigned int>(__lhs) & static_cast<unsigned int>(__rhs)));
}

_CCCL_API constexpr byte& operator&=(byte& __lhs, byte __rhs) noexcept
{
  return __lhs = __lhs & __rhs;
}

_CCCL_API constexpr byte operator^(byte __lhs, byte __rhs) noexcept
{
  return static_cast<byte>(
    static_cast<unsigned char>(static_cast<unsigned int>(__lhs) ^ static_cast<unsigned int>(__rhs)));
}

_CCCL_API constexpr byte& operator^=(byte& __lhs, byte __rhs) noexcept
{
  return __lhs = __lhs ^ __rhs;
}

_CCCL_API constexpr byte operator~(byte __b) noexcept
{
  return static_cast<byte>(static_cast<unsigned char>(~static_cast<unsigned int>(__b)));
}

_CCCL_TEMPLATE(class _Integer)
_CCCL_REQUIRES(_CCCL_TRAIT(is_integral, _Integer))
_CCCL_API constexpr byte& operator<<=(byte& __lhs, _Integer __shift) noexcept
{
  return __lhs = __lhs << __shift;
}

_CCCL_TEMPLATE(class _Integer)
_CCCL_REQUIRES(_CCCL_TRAIT(is_integral, _Integer))
_CCCL_API constexpr byte operator<<(byte __lhs, _Integer __shift) noexcept
{
  return static_cast<byte>(static_cast<unsigned char>(static_cast<unsigned int>(__lhs) << __shift));
}

_CCCL_TEMPLATE(class _Integer)
_CCCL_REQUIRES(_CCCL_TRAIT(is_integral, _Integer))
_CCCL_API constexpr byte& operator>>=(byte& __lhs, _Integer __shift) noexcept
{
  return __lhs = __lhs >> __shift;
}

_CCCL_TEMPLATE(class _Integer)
_CCCL_REQUIRES(_CCCL_TRAIT(is_integral, _Integer))
_CCCL_API constexpr byte operator>>(byte __lhs, _Integer __shift) noexcept
{
  return static_cast<byte>(static_cast<unsigned char>(static_cast<unsigned int>(__lhs) >> __shift));
}

_CCCL_TEMPLATE(class _Integer)
_CCCL_REQUIRES(_CCCL_TRAIT(is_integral, _Integer))
_CCCL_API constexpr _Integer to_integer(byte __b) noexcept
{
  return static_cast<_Integer>(__b);
}

_LIBCUDACXX_END_NAMESPACE_STD_NOVERSION

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___CSTDDEF_BYTE_H
