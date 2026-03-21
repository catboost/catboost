//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___OPTIONAL_HASH_H
#define _LIBCUDACXX___OPTIONAL_HASH_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__fwd/hash.h>
#include <cuda/std/__optional/optional.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#ifndef __cuda_std__

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT hash<__enable_hash_helper<optional<_Tp>, remove_const_t<_Tp>>>
{
#  if _CCCL_STD_VER <= 2017 || defined(_LIBCUDACXX_ENABLE_CXX20_REMOVED_BINDER_TYPEDEFS)
  using argument_type _LIBCUDACXX_DEPRECATED = optional<_Tp>;
  using result_type _LIBCUDACXX_DEPRECATED   = size_t;
#  endif

  _CCCL_API inline size_t operator()(const optional<_Tp>& __opt) const
  {
    return static_cast<bool>(__opt) ? hash<remove_const_t<_Tp>>()(*__opt) : 0;
  }
};

#endif // __cuda_std__

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___OPTIONAL_HASH_H
