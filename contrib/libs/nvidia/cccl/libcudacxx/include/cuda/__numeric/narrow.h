//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___NUMERIC_NARROW_H
#define _CUDA___NUMERIC_NARROW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)
#  include <stdexcept>
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__exception/terminate.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__utility/forward.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

//! Uses static_cast to cast a value \p __from to type \p _To. \p _To needs to be constructible from \p _From, and \p
//! implement operator!=. This function is intended to show that narrowing and a potential change of the value is
//! intended. Modelled after `gsl::narrow_cast`. See also the C++ Core Guidelines <a
//! href="https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Res-narrowing">ES.46</a> and <a
//! href="https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Res-casts-named">ES.49</a>.
template <class _To, class _From>
[[nodiscard]] _CCCL_API constexpr _To
narrow_cast(_From&& __from) noexcept(noexcept(static_cast<_To>(_CUDA_VSTD::forward<_From>(__from))))
{
  return static_cast<_To>(_CUDA_VSTD::forward<_From>(__from));
}

#if _CCCL_HAS_EXCEPTIONS()
struct narrowing_error : ::std::runtime_error
{
  narrowing_error()
      : ::std::runtime_error("Narrowing error")
  {}
};
#endif // _CCCL_HAS_EXCEPTIONS()

[[noreturn]] _CCCL_API inline void __throw_narrowing_error()
{
#if _CCCL_HAS_EXCEPTIONS()
  NV_IF_ELSE_TARGET(NV_IS_HOST, (throw narrowing_error{};), (_CUDA_VSTD_NOVERSION::terminate();))
#else // ^^^ _CCCL_HAS_EXCEPTIONS() ^^^ / vvv !_CCCL_HAS_EXCEPTIONS() vvv
  _CUDA_VSTD_NOVERSION::terminate();
#endif // !_CCCL_HAS_EXCEPTIONS()
}

//! Uses static_cast to cast a value \p __from to type \p _To and checks whether the value has changed. \p _To needs
//! to be constructible from \p _From and vice versa, and \p implement operator!=. Throws \ref narrowing_error in host
//! code and traps in device code if the value has changed. Modelled after `gsl::narrow`. See also the C++ Core
//! Guidelines <a href="https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Res-narrowing">ES.46</a> and <a
//! href="https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Res-casts-named">ES.49</a>.
template <class _To, class _From>
[[nodiscard]] _CCCL_API constexpr _To narrow(_From __from)
{
  static_assert(_CUDA_VSTD::is_constructible_v<_From, _To>);
  static_assert(_CUDA_VSTD::is_constructible_v<_To, _From>);

  const auto __converted = static_cast<_To>(__from);
  if (static_cast<_From>(__converted) != __from)
  {
    ::cuda::__throw_narrowing_error();
  }

  if constexpr (_CUDA_VSTD::is_arithmetic_v<_From>)
  {
    if constexpr (_CUDA_VSTD::is_signed_v<_From> && !_CUDA_VSTD::is_signed_v<_To>)
    {
      if (__from < _From{})
      {
        ::cuda::__throw_narrowing_error();
      }
    }
    if constexpr (!_CUDA_VSTD::is_signed_v<_From> && _CUDA_VSTD::is_signed_v<_To>)
    {
      if (__converted < _To{})
      {
        ::cuda::__throw_narrowing_error();
      }
    }
  }
  return __converted;
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___NUMERIC_NARROW_H
