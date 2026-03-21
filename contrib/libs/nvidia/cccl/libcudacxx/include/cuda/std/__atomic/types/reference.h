//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ATOMIC_TYPES_REFERENCE_H
#define _LIBCUDACXX___ATOMIC_TYPES_REFERENCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__atomic/types/base.h>
#include <cuda/std/__type_traits/is_trivially_copyable.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// Reference is compatible with __atomic_base_tag and uses the default dispatch
template <typename _Tp>
struct __atomic_ref_storage
{
  using __underlying_t                = _Tp;
  static constexpr __atomic_tag __tag = __atomic_tag::__atomic_base_tag;

#if !_CCCL_COMPILER(GCC) || _CCCL_COMPILER(GCC, >=, 5)
  static_assert(_CCCL_TRAIT(is_trivially_copyable, _Tp),
                "std::atomic_ref<Tp> requires that 'Tp' be a trivially copyable type");
#endif

  _Tp* __a_value;

  __atomic_ref_storage() = delete;

  _CCCL_HOST_DEVICE constexpr explicit inline __atomic_ref_storage(_Tp* value) noexcept
      : __a_value(value)
  {}

  _CCCL_HOST_DEVICE inline auto get() noexcept -> __underlying_t*
  {
    return __a_value;
  }
  _CCCL_HOST_DEVICE inline auto get() const noexcept -> __underlying_t*
  {
    return __a_value;
  }
  _CCCL_HOST_DEVICE inline auto get() volatile noexcept -> volatile __underlying_t*
  {
    return __a_value;
  }
  _CCCL_HOST_DEVICE inline auto get() const volatile noexcept -> volatile __underlying_t*
  {
    return __a_value;
  }
};

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ATOMIC_TYPES_REFERENCE_H
