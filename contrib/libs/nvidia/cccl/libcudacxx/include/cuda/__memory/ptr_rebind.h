//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_PTR_REBIND_H
#define _CUDA___MEMORY_PTR_REBIND_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__memory/runtime_assume_aligned.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <typename _Up, typename _Tp>
[[nodiscard]] _CCCL_API inline _Up* ptr_rebind(_Tp* __ptr) noexcept
{
  if constexpr (_CUDA_VSTD::is_same_v<_Up, _Tp>) // also handle _Tp == _Up == void
  {
    return __ptr;
  }
  else if constexpr (_CUDA_VSTD::is_void_v<_Up>) // _Tp: non-void, _Up: void
  {
    _CCCL_ASSERT(reinterpret_cast<_CUDA_VSTD::uintptr_t>(__ptr) % alignof(_Tp) == 0, "ptr is not aligned");
    return _CUDA_VSTD::__runtime_assume_aligned(reinterpret_cast<_Up*>(__ptr), alignof(_Tp));
  }
  else
  {
    constexpr auto __max_alignment = alignof(_Up) > alignof(_Tp) ? alignof(_Up) : alignof(_Tp);
    _CCCL_ASSERT(reinterpret_cast<_CUDA_VSTD::uintptr_t>(__ptr) % __max_alignment == 0, "ptr is not aligned");
    return _CUDA_VSTD::__runtime_assume_aligned(reinterpret_cast<_Up*>(__ptr), __max_alignment);
  }
}

template <typename _Up, typename _Tp>
[[nodiscard]] _CCCL_API inline const _Up* ptr_rebind(const _Tp* __ptr) noexcept
{
  return ::cuda::ptr_rebind<const _Up>(const_cast<_Tp*>(__ptr));
}

template <typename _Up, typename _Tp>
[[nodiscard]] _CCCL_API inline volatile _Up* ptr_rebind(volatile _Tp* __ptr) noexcept
{
  return ::cuda::ptr_rebind<volatile _Up>(const_cast<_Tp*>(__ptr));
}

template <typename _Up, typename _Tp>
[[nodiscard]] _CCCL_API inline const volatile _Up* ptr_rebind(const volatile _Tp* __ptr) noexcept
{
  return ::cuda::ptr_rebind<const volatile _Up>(const_cast<_Tp*>(__ptr));
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MEMORY_PTR_REBIND_H
