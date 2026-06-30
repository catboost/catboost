//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA__MEMORY_RESOURCE_GET_PROPERTY_H
#define _CUDA__MEMORY_RESOURCE_GET_PROPERTY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__type_traits/remove_const_ref.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

//! @brief The \c has_property concept verifies that a Resource satisfies a given Property
//! @rst
//! For \c has_property we require the following free function to be callable
//!
//! .. code cpp::
//!
//!    get_property(const Resource& res, Property prop);
//!
//! @endrst
template <class _Resource, class _Property>
_CCCL_CONCEPT has_property = _CCCL_REQUIRES_EXPR((_Resource, _Property), const _Resource& __res, _Property __prop)(
  ((void) get_property(__res, __prop)));

template <class _Property>
using __property_value_t = typename _Property::value_type;

//! @brief The \c property_with_value concept verifies that a Property is stateful and signals this through the
//! `value_type` alias
//! @rst
//! .. code cpp::
//!
//!    struct stateless_property {};
//!    static_assert(!cuda::property_with_value<stateless_property>);
//!
//!    struct stateful_property { using value_type = int; };
//!    static_assert(!cuda::property_with_value<stateful_property>);
//!
//! @endrst
template <class _Property>
_CCCL_CONCEPT property_with_value = _CCCL_REQUIRES_EXPR((_Property))(typename(__property_value_t<_Property>));

//! @brief The \c has_property_with concept verifies that a Resource satisfies a given stateful Property
//! @rst
//! For \c has_property_with we require the following free function to be callable and its return type to exactly match
//! the ``value_type`` of the Property
//!
//! .. code cpp::
//!
//!    struct stateless_property {};
//!    constexpr void get_property(const Resource& res, stateless_property) {}
//!
//!    // The resource must be stateful
//!    static_assert(!cuda::has_property_with<Resource, stateless_property, int>);
//!
//!    struct stateful_property { using value_type = int; };
//!    constexpr int get_property(const Resource& res, stateful_property) {}
//!
//!    // The resource is stateful and has the correct return type
//!    static_assert(cuda::has_property_with<Resource, stateful_property, int>);
//!
//!    // The resource is stateful but the return type is incorrect
//!    static_assert(!cuda::has_property_with<Resource, stateful_property, double>);
//!
//!    constexpr double get_property(const OtherResource& res, stateful_property) {}
//!
//!    // The resource is stateful but the value_type does not match the `get_property` return type
//!    static_assert(!cuda::has_property_with<OtherResource, stateful_property, double>);
//!
//! @endrst
template <class _Resource, class _Property, class _Return>
_CCCL_CONCEPT_FRAGMENT(
  __has_property_with_,
  requires(const _Resource& __res)(requires(property_with_value<_Property>),
                                   requires(_CUDA_VSTD::same_as<_Return, decltype(get_property(__res, _Property{}))>)));
template <class _Resource, class _Property, class _Return>
_CCCL_CONCEPT has_property_with = _CCCL_FRAGMENT(__has_property_with_, _Resource, _Property, _Return);

template <class _Resource, class _Upstream>
_CCCL_CONCEPT_FRAGMENT(
  __has_upstream_resource_,
  requires(const _Resource& __res)(
    requires(_CUDA_VSTD::same_as<_CUDA_VSTD::__remove_const_ref_t<decltype(__res.upstream_resource())>, _Upstream>)));
template <class _Resource, class _Upstream>
_CCCL_CONCEPT __has_upstream_resource = _CCCL_FRAGMENT(__has_upstream_resource_, _Resource, _Upstream);

_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__forward_property)
template <class _Derived, class _Upstream>
struct __fn
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Property)
  _CCCL_REQUIRES((!property_with_value<_Property>) _CCCL_AND has_property<_Upstream, _Property>)
  _CCCL_API friend constexpr void get_property(const _Derived&, _Property) noexcept {}

  // The indirection is needed, otherwise the compiler might believe that _Derived is an incomplete type
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Property, class _Derived2 = _Derived)
  _CCCL_REQUIRES(property_with_value<_Property> _CCCL_AND has_property<_Upstream, _Property> _CCCL_AND
                   __has_upstream_resource<_Derived2, _Upstream>)
  _CCCL_API friend constexpr __property_value_t<_Property> get_property(const _Derived& __res, _Property __prop)
  {
    return get_property(__res.upstream_resource(), __prop);
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

//! @brief The \c forward_property CRTP template allows Derived to forward all properties of Upstream
//! @rst
//! .. code cpp::
//!
//!    class UpstreamWithProperties;
//!
//!    class DerivedClass : cuda::forward_properties<DerivedClass, UpstreamWithProperties> {
//!      // This method is needed to forward stateful properties
//!      UpstreamWithProperties& upstream_resource() const { ... }
//!    };
//!
//! .. note::
//!
//!    In order to forward stateful properties, a type needs do implement an `upstream_resource()` method that returns
//!    an instance of the upstream.
//!
//! @endrst
template <class _Derived, class _Upstream>
using forward_property = __forward_property::__fn<_Derived, _Upstream>;

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif //_CUDA__MEMORY_RESOURCE_GET_PROPERTY_H
