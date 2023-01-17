/*
 *  Copyright 2008-2018 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file is_operator_plus_function_object.h
 *  \brief Type traits for determining if a \c BinaryFunction is equivalent to
///        \c operator+.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/functional.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/type_traits/pointer_traits.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{

template <typename FunctionObject>
struct is_operator_plus_function_object_impl;

} // namespace detail

/// Unary metafunction returns \c true_type if \c FunctionObject is equivalent
/// to \c operator<, and \c false_type otherwise.
template <typename FunctionObject>
#if THRUST_CPP_DIALECT >= 2011
using is_operator_plus_function_object =
#else
struct is_operator_plus_function_object :
#endif
  detail::is_operator_plus_function_object_impl<FunctionObject>
#if THRUST_CPP_DIALECT < 2011
{}
#endif
;

#if THRUST_CPP_DIALECT >= 2014
/// <code>constexpr bool</code> that is \c true if \c FunctionObject is
/// equivalent to \c operator<, and \c false otherwise.
template <typename FunctionObject>
constexpr bool is_operator_plus_function_object_v
  = is_operator_plus_function_object<FunctionObject>::value;
#endif

///////////////////////////////////////////////////////////////////////////////

namespace detail
{

template <typename FunctionObject>
struct is_operator_plus_function_object_impl                   : false_type {};
template <typename T>
struct is_operator_plus_function_object_impl<thrust::plus<T> > : true_type {};
template <typename T>
struct is_operator_plus_function_object_impl<std::plus<T>    > : true_type {};

} // namespace detail

THRUST_NAMESPACE_END

