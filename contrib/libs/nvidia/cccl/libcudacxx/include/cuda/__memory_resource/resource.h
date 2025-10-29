//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA__MEMORY_RESOURCE_RESOURCE_H
#define _CUDA__MEMORY_RESOURCE_RESOURCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__memory_resource/get_property.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__concepts/equality_comparable.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__tuple_dir/sfinae_helpers.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/fold.h>
#include <cuda/stream_ref>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_MR

//! @brief The \c synchronous_resource concept verifies that a type Resource satisfies the basic requirements of a
//! memory resource
//! @rst
//! We require that a resource supports the following interface
//!
//!   - ``allocate(size_t bytes, size_t alignment)``
//!   - ``deallocate(void* ptr, size_t bytes, size_t alignment)``
//!   - ``T() == T()``
//!   - ``T() != T()``
//!
//! @endrst
//! @tparam _Resource The type that should implement the synchronous resource concept
template <class _Resource>
_CCCL_CONCEPT synchronous_resource =
  _CCCL_REQUIRES_EXPR((_Resource), _Resource& __res, void* __ptr, size_t __bytes, size_t __alignment)(
    _Same_as(void*) __res.allocate_sync(__bytes, __alignment), //
    _Same_as(void) __res.deallocate_sync(__ptr, __bytes, __alignment),
    requires(_CUDA_VSTD::equality_comparable<_Resource>));

//! @brief The \c resource concept verifies that a type Resource satisfies the basic requirements of a
//! memory resource and additionally supports stream ordered allocations
//! @rst
//! We require that an resource supports the following interface
//!
//!   - ``allocate(size_t bytes, size_t alignment)``
//!   - ``deallocate(void* ptr, size_t bytes, size_t alignment)``
//!   - ``T() == T()``
//!   - ``T() != T()``
//!
//!   - ``allocate(cuda::stream_ref stream, size_t bytes, size_t alignment)``
//!   - ``deallocate( cuda::stream_ref stream, void* ptr, size_t bytes,  size_t alignment)``
//!
//! @endrst
//! @tparam _Resource The type that should implement the resource concept
template <class _Resource>
_CCCL_CONCEPT resource = _CCCL_REQUIRES_EXPR(
  (_Resource), _Resource& __res, void* __ptr, size_t __bytes, size_t __alignment, ::cuda::stream_ref __stream)(
  _Same_as(void*) __res.allocate(__stream, __bytes, __alignment),
  _Same_as(void) __res.deallocate(__stream, __ptr, __bytes, __alignment),
  requires(synchronous_resource<_Resource>));

//! @brief The \c resource_with concept verifies that a type Resource satisfies the `synchronous_resource` concept and
//! also satisfies all the provided Properties
//! @tparam _Resource
//! @tparam _Properties
// We cannot use fold expressions here due to a nvcc bug
template <class _Resource, class... _Properties>
_CCCL_CONCEPT synchronous_resource_with = _CCCL_REQUIRES_EXPR((_Resource, variadic _Properties))(
  requires(synchronous_resource<_Resource>),
  requires(_CUDA_VSTD::__all<has_property<_Resource, _Properties>...>::value));

//! @brief The \c resource_with concept verifies that a type Resource satisfies the `resource`
//! concept and also satisfies all the provided Properties
//! @tparam _Resource
//! @tparam _Properties
// We cannot use fold expressions here due to a nvcc bug
template <class _Resource, class... _Properties>
_CCCL_CONCEPT resource_with = _CCCL_REQUIRES_EXPR((_Resource, variadic _Properties))(
  requires(resource<_Resource>), requires(_CUDA_VSTD::__all<has_property<_Resource, _Properties>...>::value));

template <bool _Convertible>
struct __different_resource__
{
  template <class _OtherResource>
  static constexpr bool __value(_OtherResource*) noexcept
  {
    return synchronous_resource<_OtherResource>;
  }
};

template <>
struct __different_resource__<true>
{
  static constexpr bool __value(void*) noexcept
  {
    return false;
  }
};

template <class _Resource, class _OtherResource>
_CCCL_CONCEPT __different_resource =
  __different_resource__<_CUDA_VSTD::convertible_to<_OtherResource const&, _Resource const&>>::__value(
    static_cast<_OtherResource*>(nullptr));

_LIBCUDACXX_END_NAMESPACE_CUDA_MR
#include <cuda/std/__cccl/epilogue.h>

#endif //_CUDA__MEMORY_RESOURCE_RESOURCE_H
