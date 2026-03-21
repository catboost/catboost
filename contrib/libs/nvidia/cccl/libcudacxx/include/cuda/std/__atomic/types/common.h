//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ATOMIC_TYPES_COMMON_H
#define _LIBCUDACXX___ATOMIC_TYPES_COMMON_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_assignable.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/cstring>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

enum class __atomic_tag
{
  __atomic_base_tag,
  __atomic_locked_tag,
  __atomic_small_tag,
};

// Helpers to SFINAE on the tag inside the storage object
template <typename _Sto>
using __atomic_storage_is_base = enable_if_t<__atomic_tag::__atomic_base_tag == remove_cvref_t<_Sto>::__tag, int>;
template <typename _Sto>
using __atomic_storage_is_locked = enable_if_t<__atomic_tag::__atomic_locked_tag == remove_cvref_t<_Sto>::__tag, int>;
template <typename _Sto>
using __atomic_storage_is_small = enable_if_t<__atomic_tag::__atomic_small_tag == remove_cvref_t<_Sto>::__tag, int>;

template <typename _Tp>
using __atomic_underlying_t = typename _Tp::__underlying_t;
template <typename _Tp>
using __atomic_underlying_remove_cv_t = remove_cv_t<typename _Tp::__underlying_t>;

// [atomics.types.generic]p1 guarantees _Tp is trivially copyable. Because
// the default operator= in an object is not volatile, a byte-by-byte copy
// is required.
template <typename _Tp, typename _Tv>
_CCCL_HOST_DEVICE enable_if_t<_CCCL_TRAIT(is_assignable, _Tp&, _Tv)>
__atomic_assign_volatile(_Tp* __a_value, _Tv const& __val)
{
  *__a_value = __val;
}

template <typename _Tp, typename _Tv>
_CCCL_HOST_DEVICE enable_if_t<_CCCL_TRAIT(is_assignable, _Tp&, _Tv)>
__atomic_assign_volatile(_Tp volatile* __a_value, _Tv volatile const& __val)
{
  volatile char* __to         = reinterpret_cast<volatile char*>(__a_value);
  volatile char* __end        = __to + sizeof(_Tp);
  volatile const char* __from = reinterpret_cast<volatile const char*>(&__val);
  while (__to != __end)
  {
    *__to++ = *__from++;
  }
}

_CCCL_HOST_DEVICE inline int __atomic_memcmp(void const* __lhs, void const* __rhs, size_t __count)
{
  NV_DISPATCH_TARGET(
    NV_IS_DEVICE,
    (unsigned char const* __lhs_c; unsigned char const* __rhs_c;
     // NVCC recommended laundering through inline asm to compare padding bytes.
     asm("mov.b64 %0, %2;\n mov.b64 %1, %3;" : "=l"(__lhs_c), "=l"(__rhs_c) : "l"(__lhs), "l"(__rhs));
     while (__count--) {
       auto const __lhs_v = *__lhs_c++;
       auto const __rhs_v = *__rhs_c++;
       if (__lhs_v < __rhs_v)
       {
         return -1;
       }
       if (__lhs_v > __rhs_v)
       {
         return 1;
       }
     } return 0;),
    NV_IS_HOST,
    (return _CUDA_VSTD::memcmp(__lhs, __rhs, __count);))
  _CCCL_UNREACHABLE();
}

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___ATOMIC_TYPES_COMMON_H
