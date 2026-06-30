// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_PTX_HELPER_FUNCTIONS_H_
#define _CUDA_PTX_HELPER_FUNCTIONS_H_

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#if _CCCL_CUDA_COMPILATION()

#  include <cuda/std/__cccl/prologue.h>

#  if defined(__CUDACC__) || defined(_NVHPC_CUDA) || defined(__CUDACC_RTC__)
#    define _CUDA_PTX_CUDACC_MAJOR() __CUDACC_VER_MAJOR__
#  elif defined(__CUDA__) && defined(__clang__)
#    define _CUDA_PTX_CUDACC_MAJOR() (CUDA_VERSION / 1000)
#  endif // ^^^ has cuda compiler ^^^

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_PTX

#  if _CUDA_PTX_CUDACC_MAJOR() < 13
struct alignas(32) longlong4_32a
{
  long long x, y, z, w;
};
struct alignas(32) ulonglong4_32a
{
  unsigned long long x, y, z, w;
};
struct alignas(32) double4_32a
{
  double x, y, z, w;
};
#  else
using ::double4_32a;
using ::longlong4_32a;
using ::ulonglong4_32a;
#  endif // _CUDA_PTX_CUDACC_MAJOR() < 13

/*************************************************************
 *
 * Conversion from generic pointer -> state space "pointer"
 *
 **************************************************************/
inline _CCCL_DEVICE ::cuda::std::uint32_t __as_ptr_smem(const void* __ptr)
{
  // Consider adding debug asserts here.
  return static_cast<::cuda::std::uint32_t>(::__cvta_generic_to_shared(__ptr));
}

inline _CCCL_DEVICE ::cuda::std::uint32_t __as_ptr_dsmem(const void* __ptr)
{
  // No difference in implementation to __as_ptr_smem.
  // Consider adding debug asserts here.
  return __as_ptr_smem(__ptr);
}

inline _CCCL_DEVICE ::cuda::std::uint32_t __as_ptr_remote_dsmem(const void* __ptr)
{
  // No difference in implementation to __as_ptr_smem.
  // Consider adding debug asserts here.
  return __as_ptr_smem(__ptr);
}

inline _CCCL_DEVICE ::cuda::std::uint64_t __as_ptr_gmem(const void* __ptr)
{
  // Consider adding debug asserts here.
  return static_cast<::cuda::std::uint64_t>(::__cvta_generic_to_global(__ptr));
}

/*************************************************************
 *
 * Conversion from state space "pointer" -> generic pointer
 *
 **************************************************************/
template <typename _Tp>
inline _CCCL_DEVICE _Tp* __from_ptr_smem(::cuda::std::size_t __ptr)
{
  // Consider adding debug asserts here.
  return reinterpret_cast<_Tp*>(::__cvta_shared_to_generic(__ptr));
}

template <typename _Tp>
inline _CCCL_DEVICE _Tp* __from_ptr_dsmem(::cuda::std::size_t __ptr)
{
  // Consider adding debug asserts here.
  return __from_ptr_smem<_Tp>(__ptr);
}

template <typename _Tp>
inline _CCCL_DEVICE _Tp* __from_ptr_remote_dsmem(::cuda::std::size_t __ptr)
{
  // Consider adding debug asserts here.
  return __from_ptr_smem<_Tp>(__ptr);
}

template <typename _Tp>
inline _CCCL_DEVICE _Tp* __from_ptr_gmem(::cuda::std::size_t __ptr)
{
  // Consider adding debug asserts here.
  return reinterpret_cast<_Tp*>(::__cvta_global_to_generic(__ptr));
}

/*************************************************************
 *
 * Conversion from template type -> concrete binary type
 *
 **************************************************************/
template <typename _Tp>
inline _CCCL_DEVICE ::cuda::std::uint32_t __as_b32(_Tp __val)
{
  static_assert(sizeof(_Tp) == 4, "");
  // Consider using std::bitcast
  return *reinterpret_cast<::cuda::std::uint32_t*>(&__val);
}

template <typename _Tp>
inline _CCCL_DEVICE ::cuda::std::uint64_t __as_b64(_Tp __val)
{
  static_assert(sizeof(_Tp) == 8, "");
  // Consider using std::bitcast
  return *reinterpret_cast<::cuda::std::uint64_t*>(&__val);
}

/*************************************************************
 *
 * Conversion to and from b8 type
 *
 **************************************************************/

template <typename _B8>
inline _CCCL_DEVICE uint32_t __b8_as_u32(_B8 __val)
{
  static_assert(sizeof(_B8) == 1);
  ::cuda::std::uint32_t __u32 = 0;
  ::memcpy(&__u32, &__val, 1);
  return __u32;
}

template <typename _B8>
inline _CCCL_DEVICE _B8 __u32_as_b8(uint32_t __u32)
{
  static_assert(sizeof(_B8) == 1);
  _B8 b8;
  ::memcpy(&b8, &__u32, 1);
  return b8;
}

_LIBCUDACXX_END_NAMESPACE_CUDA_PTX

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CUDA_COMPILATION()

#endif // _CUDA_PTX_HELPER_FUNCTIONS_H_
