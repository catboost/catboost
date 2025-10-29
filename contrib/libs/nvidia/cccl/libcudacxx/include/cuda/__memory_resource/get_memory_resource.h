//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_RESOURCE_GET_MEMORY_RESOURCE_CUH
#define _CUDA___MEMORY_RESOURCE_GET_MEMORY_RESOURCE_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__memory_resource/properties.h>
#include <cuda/__memory_resource/resource.h>
#include <cuda/std/__concepts/equality_comparable.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/stream_ref>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_MR

struct __get_memory_resource_t;

template <class _Tp>
_CCCL_CONCEPT __has_member_get_resource = _CCCL_REQUIRES_EXPR((_Tp), const _Tp& __t)(
  requires(resource<_CUDA_VSTD::remove_cvref_t<decltype(__t.get_memory_resource())>>));

template <class _Env>
_CCCL_CONCEPT __has_query_get_memory_resource = _CCCL_REQUIRES_EXPR((_Env))(
  requires(!__has_member_get_resource<_Env>),
  requires(
    resource<_CUDA_VSTD::remove_cvref_t<_CUDA_STD_EXEC::__query_result_t<const _Env&, __get_memory_resource_t>>>));

//! @brief `__get_memory_resource_t` is a customization point object that queries a type `T` for an associated memory
//! resource
struct __get_memory_resource_t
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__has_member_get_resource<_Tp>)
  [[nodiscard]] _CCCL_API constexpr decltype(auto) operator()(const _Tp& __t) const noexcept
  {
    static_assert(noexcept(__t.get_memory_resource()), "get_memory_resource must be noexcept");
    return __t.get_memory_resource();
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(__has_query_get_memory_resource<_Env>)
  [[nodiscard]] _CCCL_API constexpr decltype(auto) operator()(const _Env& __env) const noexcept
  {
    static_assert(noexcept(__env.query(*this)), "get_memory_resource_t query must be noexcept");
    return __env.query(*this);
  }
};

_CCCL_GLOBAL_CONSTANT auto __get_memory_resource = __get_memory_resource_t{};

using get_memory_resource_t = __get_memory_resource_t;

_CCCL_GLOBAL_CONSTANT auto get_memory_resource = get_memory_resource_t{};

_LIBCUDACXX_END_NAMESPACE_CUDA_MR

#include <cuda/std/__cccl/epilogue.h>

#endif //_CUDAX__MEMORY_RESOURCE_GET_MEMORY_RESOURCE_CUH
