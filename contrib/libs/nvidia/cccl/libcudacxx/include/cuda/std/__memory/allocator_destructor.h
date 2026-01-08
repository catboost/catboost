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

#ifndef _LIBCUDACXX___MEMORY_ALLOCATOR_DESTRUCTOR_H
#define _LIBCUDACXX___MEMORY_ALLOCATOR_DESTRUCTOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__memory/allocator_traits.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Alloc>
class __allocator_destructor
{
  using __alloc_traits _CCCL_NODEBUG_ALIAS = allocator_traits<_Alloc>;

public:
  using pointer _CCCL_NODEBUG_ALIAS   = typename __alloc_traits::pointer;
  using size_type _CCCL_NODEBUG_ALIAS = typename __alloc_traits::size_type;

private:
  _Alloc& __alloc_;
  size_type __s_;

public:
  _CCCL_API inline __allocator_destructor(_Alloc& __a, size_type __s) noexcept
      : __alloc_(__a)
      , __s_(__s)
  {}
  _CCCL_API inline void operator()(pointer __p) noexcept
  {
    __alloc_traits::deallocate(__alloc_, __p, __s_);
  }
};

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___MEMORY_ALLOCATOR_DESTRUCTOR_H
