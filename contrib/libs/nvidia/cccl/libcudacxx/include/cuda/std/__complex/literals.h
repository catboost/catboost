//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___COMPLEX_LITERALS_H
#define _LIBCUDACXX___COMPLEX_LITERALS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__complex/complex.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#ifdef _LIBCUDACXX_HAS_STL_LITERALS
// Literal suffix for complex number literals [complex.literals]

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wliteral-suffix")
_CCCL_DIAG_SUPPRESS_CLANG("-Wuser-defined-literals")
_CCCL_DIAG_SUPPRESS_MSVC(4455)

inline namespace literals
{
inline namespace complex_literals
{
#  if !_CCCL_CUDA_COMPILER(NVCC) && !_CCCL_COMPILER(NVRTC)
// NOTE: if you get a warning from GCC <7 here that "literal operator suffixes not preceded by ‘_’ are reserved for
// future standardization" then we are sorry. The warning was implemented before GCC 7, but can only be disabled since
// GCC 7. See also: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=69523
_CCCL_API constexpr complex<long double> operator""il(long double __im)
{
  return {0.0l, __im};
}
_CCCL_API constexpr complex<long double> operator""il(unsigned long long __im)
{
  return {0.0l, static_cast<long double>(__im)};
}

_CCCL_API constexpr complex<double> operator""i(long double __im)
{
  return {0.0, static_cast<double>(__im)};
}

_CCCL_API constexpr complex<double> operator""i(unsigned long long __im)
{
  return {0.0, static_cast<double>(__im)};
}

_CCCL_API constexpr complex<float> operator""if(long double __im)
{
  return {0.0f, static_cast<float>(__im)};
}

_CCCL_API constexpr complex<float> operator""if(unsigned long long __im)
{
  return {0.0f, static_cast<float>(__im)};
}
#  else // ^^^ !_CCCL_CUDA_COMPILER(NVCC) && !_CCCL_COMPILER(NVRTC) ^^^ / vvv other compilers vvv
_CCCL_API constexpr complex<double> operator""i(double __im)
{
  return {0.0, static_cast<double>(__im)};
}

_CCCL_API constexpr complex<double> operator""i(unsigned long long __im)
{
  return {0.0, static_cast<double>(__im)};
}

_CCCL_API constexpr complex<float> operator""if(double __im)
{
  return {0.0f, static_cast<float>(__im)};
}

_CCCL_API constexpr complex<float> operator""if(unsigned long long __im)
{
  return {0.0f, static_cast<float>(__im)};
}
#  endif // other compilers
} // namespace complex_literals
} // namespace literals

_CCCL_DIAG_POP

#endif // _LIBCUDACXX_HAS_STL_LITERALS

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___COMPLEX_LITERALS_H
