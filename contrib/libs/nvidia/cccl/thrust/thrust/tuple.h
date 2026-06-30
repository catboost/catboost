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

/*! \file tuple.h
 *  \brief A type encapsulating a heterogeneous collection of elements.
 */

/*
 * Copyright (C) 1999, 2000 Jaakko Järvi (jaakko.jarvi@cs.utu.fi)
 *
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying NOTICE file for the complete license)
 *
 * For more information, see http://www.boost.org
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/tuple>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup utility
 *  \{
 */

/*! \addtogroup tuple
 *  \{
 */

/*! This metafunction returns the type of a
 *  \p tuple's <tt>N</tt>th element.
 *
 *  \tparam N This parameter selects the element of interest.
 *  \tparam T A \c tuple type of interest.
 *
 *  \see pair
 *  \see tuple
 */
#ifdef _CCCL_DOXYGEN_INVOKED // Provide a fake alias for doxygen
template <size_t N, class T>
using tuple_element = _CUDA_VSTD::tuple_element<N, T>;
#else // ^^^ _CCCL_DOXYGEN_INVOKED ^^^ / vvv !_CCCL_DOXYGEN_INVOKED vvv
using _CUDA_VSTD::tuple_element;
#endif // _CCCL_DOXYGEN_INVOKED

/*! This metafunction returns the number of elements
 *  of a \p tuple type of interest.
 *
 *  \tparam T A \c tuple type of interest.
 *
 *  \see pair
 *  \see tuple
 */
#ifdef _CCCL_DOXYGEN_INVOKED // Provide a fake alias for doxygen
template <class T>
using tuple_size = _CUDA_VSTD::tuple_size<T>;
#else // ^^^ _CCCL_DOXYGEN_INVOKED ^^^ / vvv !_CCCL_DOXYGEN_INVOKED vvv
using _CUDA_VSTD::tuple_size;
#endif // _CCCL_DOXYGEN_INVOKED

/*! \brief \p tuple is a heterogeneous, fixed-size collection of values.
 *  An instantiation of \p tuple with two arguments is similar to an
 *  instantiation of \p pair with the same two arguments. Individual elements
 *  of a \p tuple may be accessed with the \p get function.
 *
 *  \tparam Ts The type of the <tt>N</tt> \c tuple element. Thrust's \p tuple
 *          type currently supports up to ten elements.
 *
 *  The following code snippet demonstrates how to create a new \p tuple object
 *  and inspect and modify the value of its elements.
 *
 *  \code
 *  #include <thrust/tuple.h>
 *  #include <iostream>
 *
 *  int main() {
 *    // Create a tuple containing an `int`, a `float`, and a string.
 *    thrust::tuple<int, float, const char*> t(13, 0.1f, "thrust");
 *
 *    // Individual members are accessed with the free function `get`.
 *    std::cout << "The first element's value is " << thrust::get<0>(t) << std::endl;
 *
 *    // ... or the member function `get`.
 *    std::cout << "The second element's value is " << t.get<1>() << std::endl;
 *
 *    // We can also modify elements with the same function.
 *    thrust::get<0>(t) += 10;
 *  }
 *  \endcode
 *
 *  \see pair
 *  \see get
 *  \see make_tuple
 *  \see tuple_element
 *  \see tuple_size
 *  \see tie
 */
#ifdef _CCCL_DOXYGEN_INVOKED // Provide a fake alias for doxygen
template <class... Ts>
using tuple = _CUDA_VSTD::tuple<T...>;
#else // ^^^ _CCCL_DOXYGEN_INVOKED ^^^ / vvv !_CCCL_DOXYGEN_INVOKED vvv
using _CUDA_VSTD::tuple;
#endif // _CCCL_DOXYGEN_INVOKED

using _CUDA_VSTD::get;
using _CUDA_VSTD::make_tuple;
using _CUDA_VSTD::tie;

/*! \endcond
 */

/*! \} // tuple
 */

/*! \} // utility
 */

THRUST_NAMESPACE_END
