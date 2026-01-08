//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___STREAM_GET_STREAM_H
#define _CUDA___STREAM_GET_STREAM_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()

#  include <cuda/__fwd/get_stream.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__concepts/convertible_to.h>
#  include <cuda/std/__cuda/api_wrapper.h>
#  include <cuda/std/__execution/env.h>
#  include <cuda/std/__type_traits/is_convertible.h>

#  include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

class stream_ref;

template <class _Tp>
_CCCL_CONCEPT __convertible_to_stream_ref = _CUDA_VSTD::convertible_to<_Tp, ::cuda::stream_ref>;

template <class _Tp>
_CCCL_CONCEPT __has_member_stream = _CCCL_REQUIRES_EXPR((_Tp), const _Tp& __t)(
  requires(!__convertible_to_stream_ref<_Tp>), //
  requires(__convertible_to_stream_ref<decltype(__t.stream())>));

template <class _Tp>
_CCCL_CONCEPT __has_member_get_stream = _CCCL_REQUIRES_EXPR((_Tp), const _Tp& __t)(
  requires(!__convertible_to_stream_ref<_Tp>), //
  requires(__convertible_to_stream_ref<decltype(__t.get_stream())>));

template <class _Env>
_CCCL_CONCEPT __has_query_get_stream = _CCCL_REQUIRES_EXPR((_Env), const _Env& __env, const get_stream_t& __cpo)(
  requires(!__convertible_to_stream_ref<_Env>),
  requires(!__has_member_stream<_Env>),
  requires(__convertible_to_stream_ref<decltype(__env.query(__cpo))>));

//! @brief `get_stream` is a customization point object that queries a type `T` for an associated stream
struct get_stream_t
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__convertible_to_stream_ref<_Tp>)
  [[nodiscard]] _CCCL_API constexpr ::cuda::stream_ref operator()(const _Tp& __t) const
    noexcept(noexcept(static_cast<::cuda::stream_ref>(__t)))
  {
    return static_cast<::cuda::stream_ref>(__t);
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__has_member_stream<_Tp>)
  [[nodiscard]] _CCCL_API constexpr ::cuda::stream_ref operator()(const _Tp& __t) const noexcept(noexcept(__t.stream()))
  {
    return __t.stream();
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__has_member_get_stream<_Tp>)
  [[nodiscard]] _CCCL_API constexpr ::cuda::stream_ref operator()(const _Tp& __t) const
    noexcept(noexcept(__t.get_stream()))
  {
    return __t.get_stream();
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(__has_query_get_stream<_Env>)
  [[nodiscard]] _CCCL_API constexpr ::cuda::stream_ref operator()(const _Env& __env) const noexcept
  {
    static_assert(noexcept(__env.query(*this)), "");
    return __env.query(*this);
  }

  [[nodiscard]] _CCCL_API static constexpr auto query(_CUDA_STD_EXEC::forwarding_query_t) noexcept -> bool
  {
    return true;
  }
};

_CCCL_GLOBAL_CONSTANT auto get_stream = get_stream_t{};

_LIBCUDACXX_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif // _CUDA___STREAM_GET_STREAM_H
