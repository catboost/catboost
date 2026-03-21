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

#ifndef _LIBCUDACXX___MEMORY_BUILTIN_NEW_ALLOCATOR_H
#define _LIBCUDACXX___MEMORY_BUILTIN_NEW_ALLOCATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__memory/unique_ptr.h>
#include <cuda/std/__new_>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// __builtin_new_allocator -- A non-templated helper for allocating and
// deallocating memory using __builtin_operator_new and
// __builtin_operator_delete. It should be used in preference to
// `std::allocator<T>` to avoid additional instantiations.
struct __builtin_new_allocator
{
  struct __builtin_new_deleter
  {
    using pointer_type = void*;

    _CCCL_API constexpr explicit __builtin_new_deleter(size_t __size, size_t __align) noexcept
        : __size_(__size)
        , __align_(__align)
    {}

    _CCCL_API inline void operator()(void* __p) const noexcept
    {
      _CUDA_VSTD::__cccl_deallocate(__p, __size_, __align_);
    }

  private:
    size_t __size_;
    size_t __align_;
  };

  using __holder_t = unique_ptr<void, __builtin_new_deleter>;

  _CCCL_API inline static __holder_t __allocate_bytes(size_t __s, size_t __align)
  {
    return __holder_t(_CUDA_VSTD::__cccl_allocate(__s, __align), __builtin_new_deleter(__s, __align));
  }

  _CCCL_API inline static void __deallocate_bytes(void* __p, size_t __s, size_t __align) noexcept
  {
    _CUDA_VSTD::__cccl_deallocate(__p, __s, __align);
  }

  template <class _Tp>
  _CCCL_NODEBUG_ALIAS _CCCL_API inline static __holder_t __allocate_type(size_t __n)
  {
    return __allocate_bytes(__n * sizeof(_Tp), alignof(_Tp));
  }

  template <class _Tp>
  _CCCL_NODEBUG_ALIAS _CCCL_API inline static void __deallocate_type(void* __p, size_t __n) noexcept
  {
    __deallocate_bytes(__p, __n * sizeof(_Tp), alignof(_Tp));
  }
};

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___MEMORY_BUILTIN_NEW_ALLOCATOR_H
